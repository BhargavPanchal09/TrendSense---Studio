"""Fresh multi-source evidence; never modifies the historical regression score."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import urlencode
import copy
import json
import sqlite3
import os
import base64
import threading
import time
import urllib.request
import xml.etree.ElementTree as ET
from v2_engine.trend_history import record_observations

SOURCE = "https://trends.google.com/trending/rss?geo=IN"
REGIONS = ("IN", "US", "GB", "JP", "BR")
_cache = {}
_lock = threading.Lock()
_gdelt_cache = {"expires": 0.0, "rows": None, "status": "unavailable"}
_reddit_token = None
_reddit_token_expires = 0.0
_reddit_token_lock = threading.Lock()


def read_url(url, extra_headers=None, data=None):
    request = urllib.request.Request(url, headers={"User-Agent": "TrendSenseAcademic/1.0", **(extra_headers or {})})
    if data is not None:
        request.data = data
    with urllib.request.urlopen(request, timeout=8) as response:
        payload = response.read(2_000_001)
    if len(payload) > 2_000_000:
        raise ValueError("Source exceeds size limit")
    return payload


def parse_feed(payload, now=None, region="IN"):
    now = now or datetime.now(timezone.utc)
    rows = []
    for item in ET.fromstring(payload).findall("./channel/item"):
        try:
            published = parsedate_to_datetime(item.findtext("pubDate", ""))
            if published.tzinfo is None:
                published = published.replace(tzinfo=timezone.utc)
            title = item.findtext("title", "").strip()
            if not title or not timedelta(0) <= now - published <= timedelta(hours=36):
                continue
            rows.append({"topic": title[:300], "published_at": published.isoformat(),
                         "source": "Google search interest", "region": region,
                         "source_url": f"https://trends.google.com/trending?geo={region}"})
        except (ValueError, TypeError, OverflowError):
            continue
    return rows[:100]


def google_source(region):
    return parse_feed(read_url(f"https://trends.google.com/trending/rss?geo={region}"), region=region)


def _reddit_access_token():
    global _reddit_token, _reddit_token_expires
    supplied = os.getenv("REDDIT_ACCESS_TOKEN", "").strip()
    if supplied:
        return supplied
    client_id = os.getenv("REDDIT_CLIENT_ID", "").strip()
    client_secret = os.getenv("REDDIT_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        return None
    with _reddit_token_lock:
        if _reddit_token and time.monotonic() < _reddit_token_expires:
            return _reddit_token
        basic = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode("ascii")
        payload = read_url(
            "https://www.reddit.com/api/v1/access_token",
            {"Authorization": "Basic " + basic,
             "Content-Type": "application/x-www-form-urlencoded",
             "User-Agent": os.getenv("REDDIT_USER_AGENT", "TrendSenseAcademic/1.0")},
            data=b"grant_type=client_credentials",
        )
        token_data = json.loads(payload)
        _reddit_token = token_data["access_token"]
        _reddit_token_expires = time.monotonic() + max(60, int(token_data.get("expires_in", 3600)) - 60)
        return _reddit_token


def reddit_source():
    token = _reddit_access_token()
    if not token:
        return None
    payload = read_url(
        "https://oauth.reddit.com/r/all/top?t=day&limit=50",
        {"Authorization": "Bearer " + token,
         "User-Agent": os.getenv("REDDIT_USER_AGENT", "TrendSenseAcademic/1.0")},
    )
    data = json.loads(payload)
    now = datetime.now(timezone.utc)
    rows = []
    for child in data.get("data", {}).get("children", []):
        post = child.get("data", {})
        published = datetime.fromtimestamp(post.get("created_utc", 0), timezone.utc)
        permalink = post.get("permalink", "")
        if post.get("over_18") or not permalink.startswith("/r/"):
            continue
        if timedelta(0) <= now - published <= timedelta(hours=36) and post.get("title"):
            rows.append({"topic": post["title"][:300], "published_at": published.isoformat(),
                         "source": "Reddit r/all top daily", "region": "Reddit community",
                         "source_url": "https://www.reddit.com" + permalink,
                         "upvotes": post.get("score", 0), "comments": post.get("num_comments", 0)})
    return rows


def youtube_source(region):
    key = os.getenv("YOUTUBE_API_KEY", "")
    if not key:
        return None
    params = urlencode({"part": "snippet,statistics", "chart": "mostPopular", "regionCode": region,
                        "maxResults": 50, "key": key})
    data = json.loads(read_url("https://www.googleapis.com/youtube/v3/videos?" + params))
    observed = datetime.now(timezone.utc).isoformat()
    return [{"topic": item["snippet"]["title"][:300], "published_at": item["snippet"]["publishedAt"],
             "observed_at": observed, "source": "YouTube mostPopular chart", "region": region,
             "source_url": "https://www.youtube.com/watch?" + urlencode({"v": item["id"]}),
             "chart_rank": rank,
             "views": int(item["statistics"]["viewCount"]) if item.get("statistics", {}).get("viewCount") is not None else None}
            for rank, item in enumerate(data.get("items", []), 1)]


# A broad, transparent sampling query for global coverage clusters. GDELT
# translates monitored reporting into English for matching across languages.
GDELT_QUERY = os.getenv(
    "TRENDSENSE_GDELT_QUERY",
    "(AI OR technology OR election OR conflict OR war OR economy OR inflation OR climate OR earthquake OR flood OR storm OR protest OR summit OR sports OR entertainment)",
)


def gdelt_source():
    """Return a cached 24-hour sample of global GDELT headlines and source coverage."""
    now_mono = time.monotonic()
    with _lock:
        if _gdelt_cache["expires"] > now_mono and _gdelt_cache["rows"] is not None:
            return copy.deepcopy(_gdelt_cache["rows"]), _gdelt_cache["status"]

    params = urlencode({
        "query": GDELT_QUERY,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": 250,
        "timespan": "24h",
        "sort": "DateDesc",
    })
    payload = read_url("https://api.gdeltproject.org/api/v2/doc/doc?" + params)
    body = json.loads(payload)
    rows = []
    seen = set()
    for article in body.get("articles", []):
        title = " ".join(str(article.get("title", "")).split())
        url = str(article.get("url", "")).strip()
        if not title or not url.startswith(("https://", "http://")) or url in seen:
            continue
        seen.add(url)
        observed = None
        raw_date = str(article.get("seendate", ""))
        for date_format in ("%Y%m%dT%H%M%SZ", "%Y%m%dT%H%M%S"):
            try:
                observed = datetime.strptime(raw_date, date_format).replace(tzinfo=timezone.utc)
                break
            except ValueError:
                continue
        # Do not label stale/invalid records as current coverage.
        current = datetime.now(timezone.utc)
        if not observed or not timedelta(0) <= current - observed <= timedelta(hours=25):
            continue
        rows.append({
            "topic": title[:300], "observed_at": observed.isoformat(),
            "source": "GDELT global news", "region": "Global",
            "source_url": url, "publisher": str(article.get("domain", "")).strip(),
            "country": str(article.get("sourcecountry", "")).strip(),
        })
    with _lock:
        _gdelt_cache.update(expires=time.monotonic() + 900, rows=copy.deepcopy(rows), status="available" if rows else "no_recent_topics")
    return rows, "available" if rows else "no_recent_topics"


def fetch_trends(region="MULTI"):
    if region not in (*REGIONS, "MULTI"):
        raise ValueError("Unsupported region")
    with _lock:
        cached = _cache.get(region)
        if cached and time.monotonic() < cached[0]:
            return {**copy.deepcopy(cached[1]), "cached": True}
    regions = REGIONS if region == "MULTI" else (region,)
    jobs = [(f"Google search - {code}", google_source, (code,)) for code in regions]
    jobs += [(f"YouTube - {code}", youtube_source, (code,)) for code in regions]
    topics, sources = [], []
    def run(job):
        name, function, args = job
        try:
            rows = function(*args)
            status = "not_configured" if rows is None else "available" if rows else "no_recent_topics"
            return rows or [], {"name": name, "status": status, "items": len(rows or [])}
        except Exception:
            return [], {"name": name, "status": "unavailable", "items": 0}
    with ThreadPoolExecutor(max_workers=6) as executor:
        for rows, source in executor.map(run, jobs):
            topics.extend(rows)
            sources.append(source)
    global_news, gdelt_status = [], "unavailable"
    try:
        global_news, gdelt_status = gdelt_source()
    except Exception as error:
        from urllib.error import HTTPError
        gdelt_status = "rate_limited" if isinstance(error, HTTPError) and error.code == 429 else "unavailable"
        # Avoid hammering a temporarily throttled endpoint on each region view.
        with _lock:
            _gdelt_cache.update(expires=time.monotonic() + (120 if gdelt_status == "rate_limited" else 30),
                                rows=[], status=gdelt_status)
    sources.append({"name": "GDELT global news · 24h", "status": gdelt_status, "items": len(global_news)})
    available = sum(source["status"] == "available" for source in sources)
    history_status = "saved"
    try:
        topics = record_observations(topics)
    except (OSError, ValueError, sqlite3.Error):
        # A storage failure must not turn working live feeds into an outage.
        history_status = "unavailable"
    result = {"status": ("available" if available == len(sources) else "partial") if topics or global_news else "unavailable",
              "history_status": history_status,
              "region": region, "coverage": list(regions), "source": "GDELT global news, YouTube, and supplementary Google search interest",
              "source_url": "https://www.youtube.com/feed/trending",
              "fetched_at": datetime.now(timezone.utc).isoformat() if topics or global_news else None,
              "topics": topics, "global_news": global_news, "sources": sources, "cached": False}
    with _lock:
        _cache[region] = (time.monotonic() + (600 if topics else 30), copy.deepcopy(result))
    return result


def match_trends(query, encoder, region="MULTI"):
    import numpy as np
    feed = fetch_trends(region)
    topics = feed.pop("topics")
    matches = []
    if topics:
        embeddings = encoder.encode([query] + [row["topic"] for row in topics], normalize_embeddings=True)
        similarities = np.asarray(embeddings[1:]) @ np.asarray(embeddings[0])
        seen = set()
        for index in np.argsort(similarities)[::-1]:
            row = topics[int(index)]
            identity = (row.get('source'), row.get('region'), row.get('source_url'), row['topic'].casefold())
            if similarities[index] < 0.55 or identity in seen:
                continue
            seen.add(identity)
            matches.append({**row, "semantic_similarity": round(float(similarities[index]), 4)})
            if len(matches) == 5:
                break
    momentum = assess_momentum(feed, matches)
    return {**feed, "matches": matches, "momentum": momentum, "similarity_threshold": 0.55,
            "note": "Related evidence from sampled regions, not comprehensive worldwide coverage. Similarity cutoff is heuristic. The API may use these matches in a separate, capped experimental evidence lift; it is not a popularity probability."}


def assess_momentum(feed, matches):
    """Categorical current-chart observation; deliberately not a numerical score."""
    sources = feed.get("sources", [])
    def configured(*prefixes):
        return [row for row in sources if row.get("name", "").startswith(prefixes)]
    search = configured("Google search")
    youtube = configured("YouTube")
    full_coverage = all(
        group and all(row.get("status") in ("available", "no_recent_topics") for row in group)
        for group in (search, youtube)
    )
    if matches:
        return {"status": "observed", "label": "Related item found in a current chart/search feed",
                "evidence_count": len(matches), "interpretation": "Current snapshot only; it does not measure trend growth or predict video views."}
    if not full_coverage:
        missing = [row["name"] for row in (*search, *youtube)
                   if row.get("status") in ("unavailable", "not_configured")]
        return {"status": "unavailable", "label": "Cannot assess current trend strength",
                "missing_sources": missing, "interpretation": "Coverage is incomplete; no low-trend conclusion is justified."}
    return {"status": "no_direct_signal", "label": "No matching trend item in the checked feeds",
            "evidence_count": 0, "interpretation": "No related item appeared in the sampled Google and YouTube feeds. This does not prove the topic is not trending; the samples are limited."}
