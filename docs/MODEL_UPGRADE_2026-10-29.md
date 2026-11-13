# TrendSense model experiment and live-growth upgrade

## What was completed

The existing app, 6,000-row dataset, and prior saved evaluation were inspected.
Six new networks were trained on the RTX 4050: two network sizes, each with three
random seeds. A three-member ensemble was selected using validation error. It was
compared against the existing MLP, a training-median baseline, and ridge regression.
The original checkpoint remains active because the experiment did not establish
an improvement. The new candidate is saved separately for research.

The app now stores regional YouTube chart observations in SQLite and displays
measured view growth beside matched videos. Live validation returned 50 YouTube
items and 10 Google search items per region across IN, US, GB, JP, and BR: 300
source entries total (not necessarily 300 unique topics/videos). The 250 YouTube
observations were saved. No second real observation interval has been validated
yet; growth calculations were verified with deterministic test fixtures.

## Measured comparison

Same seeded split: 4,200 training rows, 900 validation rows, 900 test rows.
Lower MAE/RMSE is better. The old test set has already been examined in earlier
work, so this is a **retrospective benchmark**, not an untouched final evaluation.

| Model | Validation MAE | Test MAE | Test RMSE | Within 10 points |
|---|---:|---:|---:|---:|
| Training median | 24.64 | 24.81 | 28.79 | 20.56% |
| Ridge regression | 22.55 | 22.34 | 26.73 | 24.67% |
| Existing deep MLP (active) | 21.50 | **20.87** | **26.47** | **29.67%** |
| New deep ensemble (experimental) | **21.44** | 21.45 | 26.93 | 29.56% |

The candidate's validation advantage was only 0.06 points. Its test MAE was
0.59 points worse. A paired bootstrap interval for improvement over the existing
MLP was [-1.30, 0.14] points; it does not establish a reliable gain. This interval
assumes independently sampled rows and cannot account for unknown creator groups.
We did not keep tuning against this test set to manufacture a winning number.

Full numbers, dataset hash, saved predictions, training history, and row split
manifest are in `docs/content_benchmark/`. Open `comparison.html` for the interactive
comparison and validation curves. A percentage within a tolerance is not general
classification accuracy, and MAE 20.87 does not mean 79.13% accurate.

## Deep learning you can explain

1. **Transfer learning:** SBERT and CLIP provide pretrained text/image features.
2. **Trainable neural regression:** the existing MLP learns the engagement target
   through backpropagation. It remains the production predictor.
3. **New experiment:** 385 inputs → 128 → 64 → 1, with LayerNorm, GELU, dropout,
   and a sigmoid output scaled to 0–100. A 256 → 128 variant was also tested.
4. **Train-only standardization:** feature means/scales are fitted on training
   rows and stored with weights, so inference uses identical preprocessing.
5. **Huber loss and AdamW:** a smooth robust loss and weight-decayed optimization.
6. **Regularization:** dropout, gradient clipping, learning-rate reduction, and
   early stopping reduce instability/overfitting. They cannot create missing data.
7. **Ensembling:** average three independently initialized neural networks to
   reduce variation. Here that did not beat the active model on the test set.

Official references: [PyTorch LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)
and [scikit-learn leakage/preprocessing guidance](https://scikit-learn.org/stable/common_pitfalls.html).

## How measured live growth works

- An uncached live fetch records video URL, regional chart, title, view count,
  chart position, and observation time in `data/trend_observations.sqlite3`.
- A comparison needs the same video in the same region, at least 30 minutes apart.
  It prefers an approximately 24-hour baseline, with a maximum age of 48 hours.
- `views per hour = (current views − previous views) / elapsed hours`.
- Missing observations do not mean zero demand. Missing/revised counters and
  insufficient history receive explicit states instead of invented growth.
- Cached refreshes do not add observations. History is retained for 90 days.
- Search topics are still separate evidence; their counts are not mixed with
  YouTube views. Five sampled countries do not constitute worldwide coverage.
- Ordinary view growth is not acceleration or proof that a topic is trending.
  A declining *growth rate* needs multiple comparable time windows.

The dashboard shows the matched source links and growth/history state. Storage
failure does not disable live evidence. No background scheduler was installed.

## Run and reproduce (PowerShell from the project root)

Restart the backend to load the new collection code:

```powershell
cd backend
.\.venv\Scripts\python.exe -X utf8 -m uvicorn v2_engine.api:app --host 127.0.0.1 --port 8000
```

Collect a sample from another backend terminal; repeat after at least 30 minutes:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m v2_engine.collect_trends --region MULTI
```

The existing backend `.env` supplies the YouTube key. The command prints source
statuses only. The app also collects on uncached requests. No need to retrain for
each new chart observation.

Reproduce the research experiment and graphs from `backend`:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m v2_engine.benchmark_content
.\.venv\Scripts\python.exe -X utf8 -m v2_engine.plot_content_benchmark
```

This writes `docs/content_benchmark/content_candidate.pt`, including scaler
buffers and all three members. The active API checkpoint is not overwritten.
Research code can load the candidate with
`v2_engine.content_model.load_content_model(path, device)` and pass an N×385
float32 tensor with the same SBERT+CLIP feature definition. Do not rename this
artifact to the existing checkpoint: the architectures and formats differ.

## What prevents an accurate worldwide forecast today

The master data has no timestamps, creator IDs, or repeated outcomes; its labels
are engagement percentiles computed across the whole original dataset. The input
also contains full descriptions/transcripts while users enter titles/short hooks.
The current chart samples mainly contain popular items and lack representative
non-trending controls. These limitations matter more than adding another layer.
The builder now preserves raw engagement counts and available timestamp/creator
columns on future rebuilds; it cannot recover metadata absent from source data.

For a defensible forecasting study, collect repeated observations and lawful,
representative non-chart controls over several weeks. Define a fixed future
outcome (for example next-24-hour view growth relative to an age/category cohort),
keep creators/videos from crossing splits, and reserve the newest time period.
Fit any ranking transform on training outcomes only. Compare a persistence
baseline to a small GRU or temporal convolution model; report errors by region
and topic. Evaluate a separate human-labelled relevance set for the current
semantic threshold, especially Japanese/Portuguese matching with the existing
encoder. No calibrated trending/not-trending accuracy is claimed yet.

## Short explanation for your friend or sir

> “TrendSense combines a deep-learning historical content estimate with current
> source evidence. We trained and evaluated a more regularized neural ensemble,
> but kept the earlier MLP because the new model did not improve the test result.
> We also added timestamped chart observations, so the app can measure actual
> video view growth. We distinguish observed growth from future prediction and
> are building the time-series data needed to evaluate a forecasting model.”

## Verification

31 backend tests passed, including checkpoint round-trip, duplicate isolation,
growth intervals, region/video separation, missing counts, stale observations,
feed failures, evidence reports, and caption checks. Frontend TypeScript checking
passed. Live feed collection succeeded across all five configured regions.
The new dashboard rendering was not manually browser-tested in this run.
