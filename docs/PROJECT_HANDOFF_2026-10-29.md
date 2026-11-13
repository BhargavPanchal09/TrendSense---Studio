# TrendSense Studio — project handoff (29 October 2026)

This note records the work completed so far and gives a short explanation for a project teammate or class presentation.

## Is the work saved?

Yes. Changes made in this workspace are saved in the project folder on the laptop:

`D:\College Projects\TrendSense---Studio`

The YouTube key belongs in `backend/.env`; that file is ignored by Git and must stay private. Code and documentation changes are in the project folder, but they have not been committed or pushed to GitHub yet. Restart the backend tomorrow so it loads the latest source logic.

## What changed

- **Separated the two kinds of output.** The 0–100 model output is labelled a *Historical Content Estimate*. It estimates the historical engagement target learned from the training data; it is not a current-trend score or the probability that a video will go viral.
- **Added live evidence beside the historical estimate.** The dashboard can sample the official YouTube `mostPopular` chart and Google search-interest trends. A separate Google News lookup checks recent topic-specific coverage. These sources provide evidence, not guaranteed audience demand.
- **Connected YouTube for India.** The latest dashboard screenshot showed `YouTube - IN: available - 50 items`. That confirms the API key is being read and the YouTube feed is responding.
- **Added anonymous topic-search aggregation.** The backend groups similarly worded short topic queries with SBERT and counts distinct browser IDs by region over a rolling seven-day window. It retains the normalized topic, region, a one-way hash of a browser-generated ID, timestamps, and summary counts from public news/YouTube sources for up to 30 days; the title, hook, and thumbnail are not stored by this feature. Counts are scoped to users of the same backend instance. This is a current-interest data signal and does not alter the trained model score yet.
- **Made missing coverage explicit.** The momentum check now relies on the Google search and YouTube feeds. Reddit is disabled while the access approval required by Reddit's current policy is pending. If no related item is found, the dashboard says so without claiming the topic is definitely unpopular.
- **Reduced unsupported AI claims.** The Coach report builds factual statements from the measured score and returned source evidence. It does not let a generated explanation assert that a topic is trending without a matching source item. When Qwen is unavailable, advice now uses input-aware rules rather than the same fixed pair. Generated image captions are labelled for review and shortened if the vision model returns too much unsupported detail; the dashboard also distinguishes an unavailable Ollama caption.
- **Added a transparent current-evidence lift.** Matched YouTube chart, Google Trends, and multi-publisher Google News evidence can add a capped experimental adjustment (maximum 15 points) to the displayed estimate. The historical MLP score remains visible separately, and this heuristic is excluded from its test MAE. A famous entity name alone never earns points.

## Deep learning concepts in the project

The project already had a deep-learning pipeline. The recent work made its parts and limitations clearer; it did **not** train a new model or prove a new accuracy improvement.

1. **SBERT transformer for text:** encodes the video title and hook as a 384-dimensional semantic vector. The same encoder compares a topic with live-feed headlines using cosine similarity. The 0.55 matching threshold is a heuristic, not calibrated confidence.
2. **CLIP vision-language model:** compares the thumbnail image with the video title and produces one cohesion feature. This is a similarity measure, not a measured click-through rate.
3. **PyTorch deep neural network:** combines the 384 SBERT values and one CLIP value (385 total inputs) and maps them through hidden layers `256 → 128 → 64 → 1` to estimate the historical target.
4. **Training concepts:** the training code uses backpropagation, MAE loss, Adam, ReLU, batch normalization, dropout, weight decay, gradient clipping, a validation-based learning-rate scheduler, and early stopping.
5. **Local generative models:** LLaVA describes the thumbnail and Qwen selects bounded coaching suggestions. Their prose is not used as proof of current popularity or as the source of the numeric estimate.
6. **Semantic search grouping:** SBERT also groups similar short topic searches for the current-interest count. This is an aggregation heuristic, not a trained popularity predictor.

## What the accuracy number means

The saved evaluation reports a **test MAE of 20.87 points** on 900 held-out examples. In plain language, the model's estimate differed from the dataset target by about 20.87 points on average in that test. It does not mean the model is “79.13% accurate,” and it does not promise that one video's estimate is within 20.87 points.

The source-integration and search-count changes do not retrain the checkpoint or change its score. Better claims about prediction accuracy would require representative labelled video outcomes, topic/news snapshots recorded at prediction time, a time-based evaluation split, retraining, and a fresh held-out evaluation.

## How to explain it to a friend or instructor

> “TrendSense combines a historical engagement estimate with a separate current-evidence check. SBERT turns the title and hook into semantic features, CLIP compares the thumbnail with the title, and a trained PyTorch multilayer perceptron predicts the historical engagement target. YouTube trending-chart items and Google search trends are checked separately and matched semantically. If the sampled sources do not contain the topic, the app reports that limited evidence instead of inventing a trend score. The current model has a held-out MAE of 20.87 points, so the result is an estimate, not exact accuracy.”

## Next time you open the project

1. Open this project folder in VS Code.
2. In the backend terminal, press **Ctrl+C** if the server is still running, then restart it:

   ```powershell
   cd "D:\College Projects\TrendSense---Studio\backend"
   .\.venv\Scripts\Activate.ps1
   python -X utf8 -m uvicorn v2_engine.api:app --host 127.0.0.1 --port 8000
   ```

3. Refresh the frontend and run the engine. The source panel should show Google search and YouTube. A no-match result means no related item was found in those sampled feeds; it does not prove there is no worldwide interest.

4. Keep the API key only in `backend/.env`. Do not add it to frontend code, screenshots, or GitHub.

