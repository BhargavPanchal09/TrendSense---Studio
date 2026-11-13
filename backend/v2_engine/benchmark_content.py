"""Run from backend: python -m v2_engine.benchmark_content.

Select on validation only; the reused historical test set is a retrospective
benchmark, NOT a fresh prospective test of current worldwide trends.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from v2_engine.content_model import ContentRegressor, ContentEnsemble

ROOT = Path(__file__).resolve().parents[2]


def load_data(path):
    frame = pd.read_parquet(path)
    if frame.video_id.isna().any() or frame.video_id.duplicated().any():
        raise ValueError('Video IDs must be present and unique before splitting')
    embeddings = np.stack([json.loads(v) if isinstance(v, str) else v
                           for v in frame.fused_text_embedding]).astype('float32')
    if embeddings.shape != (len(frame), 384):
        raise ValueError('Expected 384-dimensional SBERT embeddings')
    features = np.column_stack([embeddings, frame.clip_cohesion_score]).astype('float32')
    labels = frame.target_virality_score.to_numpy(dtype='float32')
    if not np.isfinite(features).all() or not np.isfinite(labels).all():
        raise ValueError('Non-finite data must be repaired before training')
    if ((labels < 0) | (labels > 100)).any() or len(frame) < 100:
        raise ValueError('Need at least 100 rows with targets in [0, 100]')
    return frame, features, labels


def metrics(actual, predicted):
    error = np.abs(actual - predicted)
    return {'mae': float(error.mean()), 'rmse': float(np.sqrt(mean_squared_error(actual, predicted))),
            'r2': float(r2_score(actual, predicted)),
            'within_10_percent': float(100 * (error <= 10).mean()),
            'within_20_percent': float(100 * (error <= 20).mean())}


def split_indices(features):
    # Match the previous training split for a paired comparison. Remove exact
    # feature duplicates from later splits to prevent train/test contamination.
    order = torch.randperm(len(features), generator=torch.Generator().manual_seed(42)).numpy()
    a = int(.7 * len(order))
    b = a + int(.15 * len(order))
    seen, splits = set(), []
    for indices in (order[:a], order[a:b], order[b:]):
        kept = []
        for i in indices:
            key = hashlib.sha256(features[i].tobytes()).digest()
            if key not in seen:
                seen.add(key)
                kept.append(i)
        splits.append(np.array(kept, dtype=int))
    return splits


def fit_member(x, y, train, validation, config, seed, epochs, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ContentRegressor(**config).to(device)
    model.feature_mean.copy_(torch.from_numpy(x[train].mean(0)).to(device))
    model.feature_scale.copy_(torch.from_numpy(x[train].std(0).clip(1e-6)).to(device))
    loader = DataLoader(TensorDataset(torch.from_numpy(x[train]), torch.from_numpy(y[train, None])),
                        batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(seed))
    vx, vy = torch.from_numpy(x[validation]).to(device), torch.from_numpy(y[validation, None]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=.5)
    best, stale, state, history = float('inf'), 0, None, []
    for epoch in range(epochs):
        model.train()
        total = 0.
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            # Huber loss smooths the gradient near zero and limits outlier influence.
            loss = torch.nn.functional.huber_loss(model(bx), by, delta=10.)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += loss.item() * len(bx)
        model.eval()
        with torch.no_grad():
            score = torch.nn.functional.l1_loss(model(vx), vy).item()
        history.append({'seed': seed, 'width': config['width'], 'epoch': epoch + 1,
                        'train_huber': total / len(train), 'validation_mae': score})
        if score < best:
            best, stale = score, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        scheduler.step(score)
        if stale >= 15:
            break
    print(f"width={config['width']} seed={seed}: best validation MAE {best:.3f}, epochs {len(history)}", flush=True)
    return state, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT / 'data/trendsense_v8_master.parquet')
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/content_benchmark')
    parser.add_argument('--epochs', type=int, default=120)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    frame, x, y = load_data(args.data)
    train, val, test = split_indices(x)
    print(f'{device}: train={len(train)}, validation={len(val)}, test={len(test)}', flush=True)
    predictions, validation_scores = {}, {}
    median = float(np.median(y[train]))
    predictions['train_median'] = np.full(len(test), median)
    validation_scores['train_median'] = float(np.abs(y[val] - median).mean())
    # A simple baseline tells us whether deep learning adds useful signal.
    best_ridge, ridge_mae, best_alpha = None, float('inf'), None
    for alpha in (10., 100., 1000., 10000.):
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(x[train], y[train])
        score = mean_absolute_error(y[val], ridge.predict(x[val]).clip(0, 100))
        if score < ridge_mae:
            best_ridge, ridge_mae, best_alpha = ridge, float(score), alpha
    validation_scores['ridge'] = ridge_mae
    predictions['ridge'] = best_ridge.predict(x[test]).clip(0, 100)
    # Load the old architecture without importing the API and its heavyweight models.
    from v2_engine.train_trendsense import ViralityPredictor
    old = ViralityPredictor().to(device)
    old.load_state_dict(torch.load(ROOT / 'backend/v2_engine/trendsense_core_v5.pt', map_location=device, weights_only=True))
    old.eval()
    with torch.no_grad():
        validation_scores['existing_mlp'] = float(np.abs(old(torch.from_numpy(x[val]).to(device)).cpu().numpy().ravel().clip(0, 100) - y[val]).mean())
        predictions['existing_mlp'] = old(torch.from_numpy(x[test]).to(device)).cpu().numpy().ravel().clip(0, 100)
    history, candidates = [], []
    for config in ({'width': 128, 'dropout': .35}, {'width': 256, 'dropout': .5}):
        states = []
        for seed in (17, 42, 83):
            state, rows = fit_member(x, y, train, val, config, seed, args.epochs, device)
            states.append(state)
            history.extend(rows)
        ensemble = ContentEnsemble(config, states).to(device).eval()
        with torch.no_grad():
            score = float(np.abs(ensemble(torch.from_numpy(x[val]).to(device)).cpu().numpy().ravel() - y[val]).mean())
        candidates.append((score, config, states))
    score, config, states = min(candidates, key=lambda candidate: candidate[0])
    validation_scores['deep_ensemble'] = score
    artifact = {'format_version': 1, 'input_features': 385, 'config': config, 'states': states}
    torch.save(artifact, args.output / 'content_candidate.pt')
    model = ContentEnsemble(config, states).to(device).eval()
    with torch.no_grad():
        predictions['deep_ensemble'] = model(torch.from_numpy(x[test]).to(device)).cpu().numpy().ravel()
    selected = min(validation_scores, key=validation_scores.get)
    results = {name: metrics(y[test], pred) for name, pred in predictions.items()}
    differences = np.abs(predictions['existing_mlp'] - y[test]) - np.abs(predictions['deep_ensemble'] - y[test])
    rng = np.random.default_rng(42)
    bootstrap = [float(rng.choice(differences, len(test), replace=True).mean()) for _ in range(2000)]
    report = {'dataset_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
              'rows': len(frame), 'split': {'train': len(train), 'validation': len(val), 'test': len(test), 'seed': 42},
              'excluded_duplicate_features': len(frame) - sum(map(len, (train, val, test))),
              'validation_mae': validation_scores, 'selected_on_validation': selected,
              'ensemble_config': config, 'ensemble_seeds': [17, 42, 83], 'ridge_alpha': best_alpha,
              'test': results, 'paired_mae_improvement_95_percent_interval': np.quantile(bootstrap, [.025, .975]).tolist(),
              'limitations': ['Retrospective reused test split; not a fresh prospective evaluation.',
                             'No timestamps/channel IDs: temporal and creator leakage cannot be assessed.',
                             'Targets are whole-dataset engagement percentiles, not future popularity.',
                             'Training text includes descriptions/transcripts; user titles/hooks may differ.'],
              'versions': {'torch': str(torch.__version__), 'numpy': np.__version__}}
    (args.output / 'evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    pd.DataFrame(history).to_csv(args.output / 'training_history.csv', index=False)
    pd.DataFrame({'video_id': frame.video_id.iloc[test].to_numpy(), 'actual': y[test], **predictions}).to_csv(args.output / 'test_predictions.csv', index=False)
    pd.DataFrame([{'row': int(i), 'video_id': str(frame.video_id.iloc[i]), 'split': name}
                  for name, indices in [('train', train), ('validation', val), ('test', test)] for i in indices]).to_csv(args.output / 'split_manifest.csv', index=False)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
