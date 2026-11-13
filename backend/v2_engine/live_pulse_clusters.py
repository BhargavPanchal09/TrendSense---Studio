"""Cluster the current, approved live-feed snapshot into readable pulse topics."""
from threading import Lock

from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

_lock = Lock()
_cache = {}


def _unique_items(rows):
    """Merge the same title/video across regions so regional repeats are support, not volume."""
    merged = {}
    for source_row in rows:
        title = " ".join(str(source_row.get("topic", "")).split())
        source = str(source_row.get("source", ""))
        if not title:
            continue
        link = source_row.get("source_url", "")
        key = (source, link if (source.startswith(("YouTube", "GDELT")) and link) else title.casefold())
        if key not in merged:
            merged[key] = {**source_row, "topic": title, "regions": [], "publishers": [], "countries": []}
        item = merged[key]
        publisher = str(source_row.get("publisher", "")).strip()
        country = str(source_row.get("country", "")).strip()
        if publisher and publisher not in item["publishers"]:
            item["publishers"].append(publisher)
        if country and country not in item["countries"]:
            item["countries"].append(country)
        region = source_row.get("region")
        if region and region not in item["regions"]:
            item["regions"].append(region)
        # Keep the strongest currently observed view/rank values for duplicate
        # cross-region copies of the same YouTube video.
        views = source_row.get("views")
        if isinstance(views, int):
            item["views"] = max(int(item.get("views") or 0), views)
        rank = source_row.get("chart_rank")
        if isinstance(rank, int) and (item.get("chart_rank") is None or rank < item["chart_rank"]):
            item["chart_rank"] = rank
        growth = source_row.get("growth", {})
        if growth.get("status") == "measured":
            previous = item.get("growth", {})
            if previous.get("status") != "measured" or growth.get("views_per_hour", 0) > previous.get("views_per_hour", 0):
                item["growth"] = growth
    return list(merged.values())


def get_live_pulse(feed, embedding_model):
    """Build cached BERTopic clusters over the latest source snapshot."""
    region = feed.get("region", "MULTI")
    snapshot = feed.get("fetched_at") or "unavailable"
    cache_key = (region, snapshot)
    with _lock:
        if cache_key in _cache:
            return _cache[cache_key]

        rows = _unique_items([*feed.get("topics", []), *feed.get("global_news", [])])
        if len(rows) < 10:
            result = {
                "status": "insufficient_items",
                "method": "SBERT embeddings + UMAP + HDBSCAN (BERTopic)",
                "items_analyzed": len(rows),
                "clusters": [],
                "note": "At least 10 distinct current-feed items are needed for topic discovery.",
            }
        else:
            texts = [row["topic"] for row in rows]
            try:
                model = BERTopic(
                    embedding_model=embedding_model,
                    umap_model=UMAP(
                        n_neighbors=min(15, len(texts) - 1),
                        n_components=min(5, len(texts) - 2),
                        min_dist=0.0,
                        metric="cosine",
                        random_state=42,
                    ),
                    hdbscan_model=HDBSCAN(
                        min_cluster_size=3,
                        min_samples=1,
                        metric="euclidean",
                        cluster_selection_method="eom",
                        prediction_data=True,
                    ),
                    vectorizer_model=CountVectorizer(
                        stop_words="english", ngram_range=(1, 2), min_df=1,
                    ),
                    min_topic_size=3,
                    nr_topics="auto",
                    calculate_probabilities=False,
                    verbose=False,
                )
                labels, _ = model.fit_transform(texts)
                clusters = []
                for info in model.get_topic_info().itertuples(index=False):
                    topic_id = int(info.Topic)
                    if topic_id == -1:
                        continue
                    members = [row for row, label in zip(rows, labels) if int(label) == topic_id]
                    if not members:
                        continue
                    keywords = [word for word, _weight in (model.get_topic(topic_id) or [])[:4]]
                    sample_rows = sorted(
                        members,
                        key=lambda row: (
                            0 if row.get("source", "").startswith("YouTube") else 1,
                            row.get("chart_rank") if row.get("chart_rank") is not None else 10_000,
                            -(row.get("views") or 0),
                        ),
                    )
                    measured_rates = [
                        row.get("growth", {}).get("views_per_hour")
                        for row in members
                        if row.get("growth", {}).get("status") == "measured"
                    ]
                    sources = sorted({row.get("source", "") for row in members if row.get("source")})
                    regions = sorted({region_name for row in members for region_name in row.get("regions", [])})
                    publishers = sorted({publisher for row in members for publisher in row.get("publishers", [])})
                    countries = sorted({country for row in members for country in row.get("countries", [])})
                    clusters.append({
                        "topic_id": topic_id,
                        "name": " · ".join(keywords) or f"Topic {topic_id + 1}",
                        "item_count": len(members),
                        "source_count": len(sources),
                        "sources": sources,
                        "regions": regions,
                        "publisher_count": len(publishers),
                        "country_count": len(countries),
                        "observed_views_per_hour": round(sum(measured_rates), 1) if measured_rates else None,
                        "sample_items": [
                            {
                                "title": row["topic"],
                                "source": row.get("source", ""),
                                "region": row.get("region", ""),
                                "source_url": row.get("source_url", ""),
                                "publisher": row.get("publisher", ""),
                                "country": row.get("country", ""),
                                "views": row.get("views"),
                                "chart_rank": row.get("chart_rank"),
                            }
                            for row in sample_rows[:3]
                        ],
                    })
                # If growth has been measured, surface it first. Before the
                # first repeat measurement, rank by sampled source support.
                clusters.sort(key=lambda row: (
                    row["publisher_count"], row["country_count"], row["source_count"],
                    row["observed_views_per_hour"] or 0, row["item_count"], len(row["regions"]),
                ), reverse=True)
                ranking = "distinct news publishers and countries, then observed YouTube views/hour and source support"
                result = {
                    "status": "available" if clusters else "no_clusters_found",
                    "method": "SBERT embeddings + UMAP + HDBSCAN (BERTopic); c-TF-IDF topic words",
                    "items_analyzed": len(rows),
                    "ranking": ranking,
                    "clusters": clusters[:8],
                    "note": "Clusters combine GDELT's sampled global news coverage from the past 24 hours with current Google Trends and YouTube feeds. Publisher/country counts indicate coverage, not audience size or virality. Reddit is not connected.",
                }
            except Exception as error:
                result = {
                    "status": "unavailable",
                    "method": "SBERT embeddings + UMAP + HDBSCAN (BERTopic)",
                    "items_analyzed": len(rows),
                    "clusters": [],
                    "note": f"Live topic clustering failed ({type(error).__name__}); source feeds remain available.",
                }

        if len(_cache) > 10:
            _cache.clear()
        _cache[cache_key] = result
        return result
