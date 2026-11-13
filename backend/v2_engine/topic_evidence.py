"""Topic-specific recent coverage, separate from popularity and video prediction."""
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime
from urllib.parse import urlencode, urlparse
import copy
import re
import threading
import time
import xml.etree.ElementTree as ET
from v2_engine.live_trends import read_url

_cache = {}
_lock = threading.Lock()
LOCALES = {"IN": "en-IN", "US": "en-US", "GB": "en-GB", "JP": "ja", "BR": "pt-BR"}


def make_topic_query(title, entities="", explicit=""):
    source = (explicit or title).strip()
    # A long verification sentence can make Google News require too many terms.
    # Keep the leading topic phrase (e.g. "Tesla Roadster 2026" before its details).
    topic = re.split(r"[:|\u2014\u2013,;]", source, maxsplit=1)[0].strip()
    if len(topic.split()) > 6:
        topic = " ".join(topic.split()[:6])
    anchor = next((part.strip() for part in entities.split(",") if part.strip()), "")
    topic_tokens = {token.casefold() for token in re.findall(r"[\w]+", topic)}
    anchor_tokens = re.findall(r"[\w]+", anchor)
    if anchor and not any(token.casefold() in topic_tokens for token in anchor_tokens):
        topic = f"{anchor} {topic}".strip()
    return topic[:160]


def parse_news(payload, now=None):
    now = now or datetime.now(timezone.utc)
    items, seen = [], set()
    for item in ET.fromstring(payload).findall("./channel/item"):
        try:
            date = parsedate_to_datetime(item.findtext("pubDate", ""))
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            title = item.findtext("title", "").strip()[:500]
            url = item.findtext("link", "").strip()
            source = item.find("source")
            publisher = source.text if source is not None else "Unknown publisher"
            publisher_url = source.get("url", "") if source is not None else ""
            if not title or title.casefold() in seen or urlparse(url).scheme != "https":
                continue
            if not timedelta(0) <= now - date <= timedelta(hours=48):
                continue
            seen.add(title.casefold())
            items.append({"title": title, "url": url, "publisher": publisher,
                          "publisher_url": publisher_url, "published_at": date.isoformat()})
        except (ValueError, TypeError, OverflowError):
            continue
    return items[:50]


def check_topic(query, encoder, region="IN"):
    import numpy as np
    if region not in (*LOCALES, "MULTI"):
        raise ValueError("Unsupported news region")
    countries = tuple(LOCALES) if region == "MULTI" else (region,)
    cache_key = (query.casefold(), region)
    with _lock:
        cached = _cache.get(cache_key)
        if cached and time.monotonic() < cached[0]:
            return copy.deepcopy(cached[1])
    def fetch_country(country):
        locale = LOCALES[country]
        url = "https://news.google.com/rss/search?" + urlencode(
            {"q": query + " when:2d", "hl": locale, "gl": country, "ceid": country + ":" + locale.split("-")[0]})
        try:
            rows = parse_news(read_url(url))
            return rows, {"region": country, "status": "available", "recent_items": len(rows), "search_url": url}
        except Exception:
            return [], {"region": country, "status": "unavailable", "recent_items": 0, "search_url": url}

    rows, sources, seen = [], [], set()
    with ThreadPoolExecutor(max_workers=5) as executor:
        for items, source in executor.map(fetch_country, countries):
            sources.append(source)
            for item in items:
                key = item['title'].casefold()
                if key not in seen:
                    seen.add(key)
                    rows.append(item)
    status, selected = "unavailable", []
    fetched_at = datetime.now(timezone.utc).isoformat() if any(s['status'] == 'available' for s in sources) else None
    try:
        if rows:
            vectors = np.asarray(encoder.encode([query] + [row["title"] for row in rows], normalize_embeddings=True))
            similarities = vectors[1:] @ vectors[0]
            for index in np.argsort(similarities)[::-1]:
                if similarities[index] >= 0.55:
                    selected.append({**rows[int(index)], "semantic_similarity": round(float(similarities[index]), 4)})
        status = "recent_coverage_found" if selected else (
            "no_related_coverage_found" if all(s['status'] == 'available' for s in sources) else "unavailable")
    except Exception:
        status = "unavailable"
    result = {"query": query, "status": status, "window_hours": 48, "fetched_at": fetched_at,
              "search_url": sources[0]['search_url'], "discovery_region": region,
              "sources": sources, "coverage": list(countries), "candidate_count": len(rows),
              "relevance_threshold": 0.55, "article_count": len(selected),
              "publisher_count": len({row["publisher_url"] or row["publisher"] for row in selected}),
              "articles": selected[:8], "current_popularity": "unverified",
              "note": "Recent news coverage is evidence of discussion, not measured audience demand, trend growth or future video virality. Semantic relevance requires review."}
    with _lock:
        # Bound memory for arbitrary user-entered queries.
        if len(_cache) >= 100:
            _cache.pop(next(iter(_cache)))
        _cache[cache_key] = (time.monotonic() + (300 if status != "unavailable" else 30), copy.deepcopy(result))
    return result
