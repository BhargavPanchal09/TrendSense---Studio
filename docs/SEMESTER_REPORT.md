# TrendSense semester report notes

## Verified environment

| Item | Verified result |
| --- | --- |
| Python | 3.11.9 in `backend/.venv` |
| PyTorch | 2.11.0+cu130; CUDA runtime 13.0 |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU; CUDA available |
| Python packages | `pip check` reported no broken requirements |
| API path | Sample multipart `/analyze` request returned HTTP 200 with a score, CLIP cohesion, LLaVA description, and Qwen advice |
| Ollama models | API models `llava:7b-v1.6-mistral-q2_K` and `qwen2.5:3b` pulled and called successfully; `llama3.2:3b-instruct-q4_K_M` checked with helper |
| Spark | PySpark 4.1.1 created a local session and counted 3 rows with Temurin Java 17 |

The sample API result proves the software path runs. It does not measure prediction accuracy because the sample has no verified ground-truth label.

## Accuracy report: same-split comparison

The `OLD_FILES` archive supplied with this working copy contained the v8 master dataset, 19 transcript Parquet chunks, 6,000 thumbnails, and prior reports. The master dataset has 6,000 rows, no missing values in the training columns, 6,000 unique video IDs, and 6,000 matching thumbnail files. The features are already computed, so training used the master file directly instead of re-running the expensive image pipeline. The local archive and dataset are excluded from Git; do not add them to a public commit.

Both checkpoints were evaluated against the same deterministic 900-row test split (seed 42) from the 6,000-row dataset. The new model was selected using validation MAE; the test set was not used for early stopping.

| Model | Test MAE (points; lower is better) | Within ±10 points | Within ±20 points | Prediction mean |
| --- | ---: | ---: | ---: | ---: |
| Naive training-median baseline | 24.81 | 20.56% | 40.78% | 50.14 |
| Previous checkpoint | 21.11 | 32.56% | 57.78% | 48.96 |
| Retrained checkpoint | **20.87** | 29.67% | **58.11%** | 51.50 |

The retrained checkpoint reduced MAE by **0.24 points (1.13%)** versus the previous checkpoint. Its ±20 rate rose by 0.33 percentage points, while its ±10 rate fell by 2.89 points. A paired bootstrap 95% interval for the MAE reduction was -0.49 to +0.95 points, which includes zero. Report this as a small observed change, not proof of a reliable accuracy improvement. Both MLP checkpoints beat the median-only baseline on MAE.

The earlier performance PDF reported a test MAE of 20.97 and the same rounded tolerance rates. Re-evaluating the saved previous checkpoint with per-row MAE gives 21.11. The older trainer averaged batch MAEs equally, including a final batch of only four rows; the updated trainer computes the exact per-row MAE. Use the same per-row calculation for this comparison.

The new run used 4,200 training rows, 900 validation rows, and 900 test rows. Best validation MAE was 21.50 at epoch 99; early stopping fired at epoch 114. The saved reports and graphs are in `docs/training_results/`:

- [`before_after_results.json`](training_results/before_after_results.json) - exact same-split metrics and bootstrap interval.
- [Interactive before/after comparison chart](training_results/before_after_comparison.html).
- [`training_history.csv`](training_results/training_history.csv) and [training curves](training_results/training_curves.html) - train/validation loss by epoch.
- [`test_predictions.csv`](training_results/test_predictions.csv) and [held-out predictions chart](training_results/test_predictions.html) - actual vs. predicted scores.
- [`evaluation_results.json`](training_results/evaluation_results.json) - metrics for the retrained checkpoint.

To repeat training from the already prepared master file, activate `.venv` and run from `backend`:

```powershell
$env:PYTHONUTF8 = '1'
& .\.venv\Scripts\python.exe -X utf8 v2_engine/train_trendsense.py
```

To rebuild the master file from raw chunks, the builder expects the chunk Parquets in `data/checkpoints/` and images in `data/thumbnails/`. Those raw files are preserved under the ignored `OLD_FILES` archive in this workstation, not included in the public Git branch.

## Improvement options considered

| Option | Why it fits | Expected gain | Difficulty / time | Hardware |
| --- | --- | --- | --- | --- |
| Clean labels, deduplicate videos, and split by channel or upload time | Reduces label noise and leakage; directly targets the weakest evidence in the current pipeline | Potentially high, but must be measured | Medium; depends on source data | CPU/GPU; Spark helps for large input |
| Keep the 385-feature MLP and add reproducible validation, early stopping, weight decay, and error analysis | Appropriate for the current tabular vector input; exposes overfitting and improves model selection | Moderate and measurable; no gain is guaranteed | Low to medium; hours once data exists | Existing RTX 4050 is sufficient |
| Fine-tune SBERT or CLIP end to end | Could adapt representations to creator/video language and thumbnail semantics | Uncertain; often limited by labeled data size | High; days and careful tuning | More VRAM preferred; 6 GB is restrictive |
| Add an LSTM/Transformer over transcripts or LoRA for Ollama models | Models sequence/context or improves generated advice | Does not automatically improve the scalar predictor; advice quality needs a separate evaluation | High; days | More VRAM or cloud GPU recommended |

**Recommended:** keep the current MLP for this demo, because it fits the 385-dimensional feature set and runs on the RTX 4050. The measured MAE change is small and the tolerance metrics are mixed, so do not claim a large accuracy gain. The next research step is to test a time- or channel-aware split and add representative low-engagement examples; only keep a change if validation improves and a fresh held-out test confirms it. LLM coaching remains a separate qualitative feature, not a source of numeric accuracy.

## What the model learns (viva explanation)

The trainer concatenates a 384-number SBERT embedding of the title and hook with one CLIP title-thumbnail cohesion value. Its MLP maps the 385 inputs through 256, 128, and 64 hidden units to one regression score. ReLU adds nonlinearity; batch normalization stabilizes hidden activations; dropout randomly disables some units during training to reduce memorization. MAE is the average absolute distance from the target score. Adam updates weights using backpropagation; weight decay penalizes overly large weights; gradient clipping limits unusually large updates. The scheduler lowers the learning rate after validation stalls, and early stopping restores the checkpoint with the best validation MAE.

The v8 builder creates its target by ranking engagement rate as a percentile. That target is relative to the collected dataset. It does not mean a video has that probability of going viral, nor does it predict an absolute view count. The API's fixed entity multipliers are hand-coded rules and are not learned by the MLP.

## Viva questions and short answers

1. **What problem does TrendSense solve?** It gives creators an early estimate and feedback for a title, hook, and thumbnail before publishing.
2. **What are the model inputs?** A 384-dimensional SBERT embedding for title plus hook, and one CLIP title-thumbnail cohesion value, giving 385 numeric features.
3. **Why use SBERT?** It turns sentence meaning into a compact vector that the MLP can use as numeric input.
4. **Why use CLIP?** It compares text and image representations to estimate whether the thumbnail matches the title.
5. **Where is deep learning used?** In the PyTorch MLP, and inside pretrained SBERT, CLIP, LLaVA, and Qwen models.
6. **Why use MAE?** The output is a continuous 0–100 score, and MAE is easy to explain as average error in score points.
7. **What do dropout and weight decay do?** They discourage the network from memorizing training examples, which can improve performance on unseen examples.
8. **What is early stopping?** Training stops when validation error no longer improves, and the best validation checkpoint is kept.
9. **What is the Big Data component?** PySpark transforms transcript and thumbnail data into Parquet features. Spark is used in offline preparation, not for each prediction request.
10. **How do you know accuracy changed?** Both checkpoints were compared on the same 900 held-out examples. MAE changed from 21.11 to 20.87, while the ±10 rate decreased; this is a small, mixed result rather than a strong improvement.

## Limitations to state clearly

- The v8 target is a percentile of engagement within the collected successful-video dataset; it is not a probability of going viral or an absolute view prediction.
- The MLP still has a large train/validation gap, indicating overfitting and limited predictive signal in title, hook, and cohesion features.
- The saved v8 feature file and raw thumbnails are local-only and are not included in the public Git branch.
- The optional trend multiplier is a manually chosen heuristic and can inflate scores independently of the learned model.
- LLaVA descriptions and Qwen coaching are generated text; they can be wrong and are not part of the measured regression target.
