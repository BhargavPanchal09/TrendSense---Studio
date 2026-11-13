"""Evidence statements are rendered from data, never invented by a language model."""
ADVICE = {
    "hook_specificity": "Name the specific moment viewers will see in the opening, using only details verified in your video.",
    "thumbnail_focus": "Keep one clear thumbnail subject and make its visible action consistent with the title.",
    "title_clarity": "Make the title promise specific and ensure the video delivers that promise.",
    "opening_delivery": "Show the promised moment early; remove introductory filler before the main action.",
    "news_story": "For this news-led topic, open with one dated, verifiable development; article volume shows coverage, not expected video views.",
    "topic_validation": "Check the topic against current sources in the selected region before describing it as trending.",
}


def _fallback_codes(title="", hook="", cohesion=None, trends=None, topic=None):
    """Choose deterministic, input-aware advice when local Qwen is unavailable."""
    title_words = str(title or "").split()
    hook_text = str(hook or "").strip()
    hook_words = hook_text.split()
    hook_start = hook_text.casefold()
    codes = []

    if cohesion is not None and cohesion < 0.28:
        codes.append("thumbnail_focus")
    if hook_start.startswith(("welcome", "hello", "hi guys", "today we", "in this video")):
        codes.append("opening_delivery")
    elif len(hook_words) < 16:
        codes.append("hook_specificity")
    if len(title_words) < 4 or len(title_words) > 14:
        codes.append("title_clarity")

    if topic and topic.get("status") == "recent_coverage_found":
        codes.append("news_story")
    elif trends and trends.get("momentum", {}).get("status") == "no_direct_signal":
        codes.append("topic_validation")

    # Add a second relevant suggestion if the input did not trigger two rules.
    for code in ("hook_specificity", "title_clarity", "opening_delivery", "thumbnail_focus"):
        if len(codes) >= 2:
            break
        if code not in codes:
            codes.append(code)
    return codes[:2]


def build_report(score, trends, recommendations=None, topic=None, title="", hook="", cohesion=None,
                evidence_adjustment=None):
    matches = trends.get("matches", [])
    if matches:
        evidence = f"Found {len(matches)} related source items. Semantic matches need review and do not prove the exact topic is trending worldwide."
    elif trends.get("status") in ("available", "partial"):
        evidence = "No strong match was found in the sampled sources. Current popularity of this topic is unverified; absence from this sample does not prove it is unpopular."
    else:
        evidence = "Live sources are unavailable. Current popularity could not be verified."
    if topic and topic.get("status") == "recent_coverage_found":
        evidence += f" Topic-specific search found {topic['article_count']} related articles from {topic['publisher_count']} publisher domains in the last 48 hours. This confirms recent coverage only, not trend growth or future video performance."
    elif topic:
        evidence += " The topic-specific news check was unavailable." if topic.get("status") == "unavailable" else " No related articles passed the topic-specific relevance filter in the last 48 hours."
    momentum = trends.get("momentum", {})
    if momentum.get("status") == "unavailable":
        evidence += " Trend strength cannot be assessed because one or more of the checked YouTube or Google search feeds is unavailable."
    elif momentum.get("status") == "no_direct_signal":
        evidence += " No related item appeared in the sampled Google and YouTube feeds; this does not prove that the topic is not trending."
    elif momentum.get("status") == "observed":
        evidence += " Related items appear in the current charts; this is a snapshot, not a growth or future-view prediction."
    if evidence_adjustment and evidence_adjustment.get("points", 0):
        evidence += (
            f" The displayed estimate adds {evidence_adjustment['points']} live-evidence points "
            f"(maximum {evidence_adjustment['maximum_points']}); this experimental heuristic "
            "is separate from the historical model and was not included in its test MAE."
        )
    codes = [code for code in (recommendations or []) if isinstance(code, str) and code in ADVICE]
    codes = list(dict.fromkeys(codes))[:2]
    contextual = _fallback_codes(title, hook, cohesion, trends, topic)
    if not codes:
        codes = contextual
    elif len(codes) < 2:
        codes.extend(code for code in contextual if code not in codes)
    if topic and topic.get("status") == "recent_coverage_found" and "news_story" not in codes:
        codes = ["news_story", *[code for code in codes if code != "news_story"][:1]]
    elif trends.get("momentum", {}).get("status") == "no_direct_signal" and "topic_validation" not in codes:
        codes = ["topic_validation", *[code for code in codes if code != "topic_validation"][:1]]
    codes = codes[:2]
    advice = " ".join(ADVICE[code] for code in codes)
    return (
        f"1. **Model estimate: {score:.1f}/100.** This estimates historical engagement percentile. "
        "Held-out test MAE was 20.87 points across 900 examples; this is not a calibrated probability or a per-video error bound.\n\n"
        f"2. **Current evidence:** {evidence}\n\n"
        f"3. **Suggested improvements:** {advice}"
    )
