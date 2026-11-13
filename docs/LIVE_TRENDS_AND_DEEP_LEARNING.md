# Live evidence and deep learning demonstration

The project retains its YouTube-trained deep regression model and a legacy Reddit fetcher that is not called by the running dashboard. The dashboard samples official YouTube `mostPopular` charts and supplementary Google search trends. Reddit is disabled pending the approval required by Reddit's current policy. Default coverage samples India, US, UK, Japan and Brazil; users can select one region. This is not exhaustive worldwide coverage.

## Source handling

Every item carries its source, region, publication date and source link. YouTube chart items also carry the chart observation time: a currently charting video may have an older publication date. Search items older than 36 hours are excluded. Individual source failures remain visible, including partial coverage. Fetches run concurrently with bounded timeouts and payload sizes, and are cached for ten minutes. No topics are invented when a source fails.

YouTube live charts require `YOUTUBE_API_KEY` in `backend/.env`. Keep the key private; do not put it in frontend code. Official API reference: https://developers.google.com/youtube/data/reference/rest/v3/videos/list . Supplementary Google search interest is labelled separately and does not claim YouTube demand. Reddit data is not requested unless access is approved under Reddit's [Responsible Builder Policy](https://support.reddithelp.com/hc/en-us/articles/42728983564564-Responsible-Builder-Policy).

## Deep learning

1. SBERT transformer produces 384-dimensional text embeddings and retrieves semantically related current source items using normalized embeddings and cosine similarity. A heuristic 0.55 threshold excludes weak matches; it is not calibrated confidence.
2. CLIP vision and text transformers produce a title-thumbnail similarity index. This supplies the 385th regression feature, not a measured CTR or clickbait label.
3. The trained PyTorch deep network maps 385 → 256 → 128 → 64 → 1. Training uses backpropagation, ReLU, batch normalization, dropout, Adam, weight decay, gradient clipping, validation-based learning rate scheduling and early stopping.
4. LLaVA generates a thumbnail description, labelled as generated and requiring review.
5. Qwen chooses from bounded improvement advice codes. Code renders the associated advice and all factual score/evidence statements. Free-form model claims about current popularity are never used in the Coach report. Invalid model choices use safe default advice.

## Live Pulse topic clustering

The dashboard groups the currently sampled Google Trends and YouTube chart titles with SBERT embeddings, UMAP dimensionality reduction, and HDBSCAN through BERTopic. BERTopic's c-TF-IDF representation supplies topic keywords. Repeated copies of the same title/video across regions are merged and shown as geographic support rather than additional distinct content. When repeat YouTube observations exist, clusters can show measured views per hour; until then they are ordered by source, region, and item support. This is a current source snapshot, not Reddit's `top?t=day/week` feed and not a virality probability. Reddit data remains disconnected unless the required Reddit approval is obtained.

The trained MLP score estimates the historical training target (engagement percentile) and does not measure current topic popularity. The displayed estimate may receive a separate, experimental live-evidence lift capped at 15 points when matching YouTube chart, Google Trends, or multi-publisher Google News evidence is found. A famous name by itself earns no lift. The lift is a transparent heuristic, not a trained or calibrated part of the MLP, and is excluded from its test MAE. A current topic and a well-performing future video are different outcomes.

## Measured performance

The existing checkpoint was trained using 4,200 rows, with 900 validation and 900 test rows. Test MAE is 20.87 score points, within ±10 points is 29.67%, and within ±20 is 58.11%. The training-median baseline MAE is 24.81. These results come from the saved held-out comparison; this integration change does not retrain weights or establish improved prediction accuracy. The average error is not a per-video uncertainty interval. The present random split does not establish future performance on new topics or languages; English input is recommended because multilingual accuracy has not been evaluated.

## Demonstration

Restart the backend and refresh the frontend. Select evidence coverage, supply a title, hook and image, and run analysis. Show the historical model estimate separately from dated current evidence. Try an unrelated topic too: it should report insufficient evidence rather than inventing a trend claim. A missing YouTube key or unavailable Google search feed must be shown explicitly.

Eight automated checks cover age filtering, invalid regions, unavailable and partial sources, semantic matching without corrupting cached data, YouTube publication/observation timestamps, and the regression that previously allowed an empty match list to produce a positive Coach trend claim. Python compilation and frontend TypeScript checks pass. Live source statuses and end-to-end observations are recorded separately after integration checking.

For the presentation: “We combine pretrained text and vision transformers with a trained deep regression network. Current evidence is retrieved separately with semantic matching and dated source links. We report held-out prediction error and show when source coverage is insufficient.”

## Integration observations

A real end-to-end FastAPI request using a local thumbnail completed with HTTP 200. SBERT, CLIP and the trained CUDA checkpoint produced a score of 45.3, LLaVA returned a description, and the Coach returned an evidence-controlled report with no positive popularity claim for the unmatched input. This verifies integration, not real-world score accuracy.

During the original integration check all five regional search feeds were available (ten topics each), and YouTube had not yet been configured. A later user run showed the India YouTube chart connected with 50 items. The Jio IPO query shown in that run had no related match in the sampled chart/search items; that is limited evidence and does not prove the topic is not trending. A separate SBERT retrieval check matched current feed titles from India, US and Japan and returned zero matches for an invented unrelated query. Exact-title checks do not evaluate arbitrary-topic retrieval accuracy or multilingual generalization.

## Topic-specific lookup correction

The general top-topic feed is a small sample and missed Jio IPO. `/analyze` now also searches recent Google News coverage for a focused query derived from the first entity and title acronyms, or the optional explicit Topic to verify field. SBERT ranks and filters article headlines at cosine similarity >=0.55. Publication dates must be within the past 48 hours; duplicate titles, future timestamps and unsafe links are rejected. The dashboard shows dated article links and publisher-domain counts, separately from the historical content estimate. Google News is supplementary news discovery, not a YouTube popularity measurement.

A live check of the exact supplied example derived the query Jio Platforms IPO and retrieved 38 related articles from 25 publisher domains in the 48-hour window. This demonstrates recent coverage only. It does not establish rising search volume, video demand, independence of publishers, factual correctness of articles, or a justified change to the MLP score. A separate experimental evidence lift now uses matched live items and remains capped and visibly distinct from the MLP result. This feature does not retrain or improve the held-out regression metrics.

## Anonymous topic-search interest

The analysis endpoint now records the short topic query and selected region with a hash of a browser-generated anonymous ID, plus the current aggregate Google News and matched YouTube evidence counts. SBERT groups similar query wording at a heuristic cosine threshold of 0.76. A browser is counted at most once per topic per UTC day; the dashboard reports distinct browser IDs in the rolling seven-day window. Search rows and topic clusters expire after 30 days. The title, hook, and thumbnail are not stored by this feature. Counts are limited to one backend instance; a local laptop does not represent all TrendSense users, and browser IDs are not verified people.

This is a local live-interest signal beside Google News coverage and YouTube chart evidence. It is not a feature in the 385-input regressor and is not counted in the current evidence lift. It is not calibrated against future video performance. To train a model that learns whether search interest and news coverage predict later video performance, collect these signals at a known prediction time and join them to actual public view growth at a fixed horizon. Evaluate on newer videos before changing the production score.

## Local AI availability and Coach fallback

LLaVA image captions and Qwen advice both depend on the local Ollama service and the configured model tags. The API returns an explicit caption status and prints the underlying exception in the backend terminal. If LLaVA is unavailable, the UI tells the user to check Ollama and the installed model. If Qwen is unavailable or returns invalid advice codes, the Coach selects contextual advice from the title, hook, CLIP index, and current source evidence rather than always using the same two defaults. This fallback is rule-based and is labelled separately from Qwen-assisted advice.

The dashboard polls `/health` for core model readiness, Ollama availability, and whether the configured LLaVA/Qwen tags are installed. Uploaded images are downscaled to a maximum 768-pixel edge and JPEG-encoded for the caption request to reduce local vision-model memory pressure. The original image remains used for the CLIP thumbnail comparison. These runtime improvements do not change the virality model's weights or claim better prediction accuracy.
