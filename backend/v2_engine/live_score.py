"""Small, explainable current-evidence adjustment kept separate from the trained MLP."""


MAX_EVIDENCE_LIFT = 15


def calculate_evidence_lift(trends, topic):
    """Return a capped heuristic lift only when live, semantically matched evidence exists."""
    matches = (trends or {}).get("matches", [])
    youtube_count = sum(
        1 for item in matches
        if item.get("source") == "YouTube mostPopular chart"
    )
    google_trends_count = sum(
        1 for item in matches
        if item.get("source") == "Google search interest"
    )
    publisher_count = (topic or {}).get("publisher_count", 0)
    if (topic or {}).get("status") != "recent_coverage_found":
        publisher_count = 0

    components = {
        "youtube_chart": min(8, youtube_count * 4),
        "google_trends": min(4, google_trends_count * 2),
        # Publisher diversity is a weak coverage signal, so it contributes less.
        "news_coverage": min(3, int(publisher_count) // 3),
    }
    total = min(MAX_EVIDENCE_LIFT, sum(components.values()))
    return {
        "points": total,
        "maximum_points": MAX_EVIDENCE_LIFT,
        "components": components,
        "matched_items": {
            "youtube_chart": youtube_count,
            "google_trends": google_trends_count,
            "news_publishers": int(publisher_count),
        },
        "method": "Experimental heuristic; fresh source matches only. Not included in model validation.",
    }
