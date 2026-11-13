# Global news clusters in Live Pulse

TrendSense uses the GDELT DOC 2.0 Article List API as its global news feed. It requests up to 250 results from the previous 24 hours with a broad, configurable topic query. GDELT says its search spans reporting in 65 languages using machine-translated coverage, so English query terms can find coverage published in other monitored languages.

The returned article titles are combined with the existing Google Trends and YouTube snapshot and grouped using the project's SBERT embeddings, UMAP, HDBSCAN, and BERTopic pipeline. Each news cluster reports sampled article count, distinct publisher domains, and source countries. The cluster ordering uses publisher/country coverage first, then any measured YouTube view growth and other source support.

## Limits

- This is a sampled news-coverage map, not a complete list of all global news and not a measure of social audience or video virality.
- The initial query covers broad world topics. Override it with `TRENDSENSE_GDELT_QUERY` in `backend/.env` if a different keyword query is appropriate. Use GDELT's documented Boolean syntax, such as `(climate OR election OR technology)`.
- The feed uses a rolling 24-hour window and is cached for 15 minutes. GDELT's `seendate` is stored as an observation timestamp, not represented as a publisher-verified publication time.
- GDELT is a public endpoint and does not require an API key. The endpoint may still rate-limit requests; Live Pulse labels HTTP 429 as `rate limited` and caches that status briefly instead of inventing data.
- GDELT article coverage is kept separate from the historical virality score and from current YouTube/Google match evidence. News clusters do not automatically increase a video's score.

## API check

On 2026-11-02, the public API endpoint was reachable from the development environment but returned HTTP 429 Too Many Requests for a read-only sample query. This confirms the server responded, but it did not return articles during this check. The project should show the source as rate limited until a later request succeeds.

## References

- [GDELT DOC 2.0 API announcement and documentation](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/)
- API endpoint: `https://api.gdeltproject.org/api/v2/doc/doc`
