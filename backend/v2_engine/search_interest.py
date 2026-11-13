"""Privacy-minimal, local topic-search counts for live interest evidence.

Only the short topic query, selected region, a hash of a browser-generated ID,
timestamps, and aggregate public-source counts are retained. Video titles,
hooks, and thumbnail data are not stored by this module. Counts are distinct
browsers in a rolling seven-day window, not verified people or a prediction of
future views.
"""
from contextlib import closing
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import os
import re
import sqlite3
import unicodedata

import numpy as np

DEFAULT_DB = Path(__file__).resolve().parents[2] / "data/topic_search_interest.sqlite3"

SIMILARITY_THRESHOLD = 0.76
RETENTION_DAYS = 30


def _clean_topic(topic):
    topic = unicodedata.normalize("NFKC", str(topic or ""))
    topic = re.sub(r"[^\w\s]", " ", topic, flags=re.UNICODE)
    return " ".join(topic.casefold().split())[:160]


def _embedding(encoder, topic):
    vector = np.asarray(encoder.encode(topic, normalize_embeddings=True), dtype=np.float32).reshape(-1)
    if vector.shape != (384,) or not np.isfinite(vector).all():
        raise ValueError("Topic encoder must return one finite 384-value SBERT vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        raise ValueError("Topic embedding has zero length")
    return vector / norm


def record_topic_search(topic, region, visitor_id, encoder, path=None, now=None,
                        news_article_count=None, news_publisher_count=None,
                        youtube_match_count=None, measured_views_per_hour=None):
    """Record one anonymous browser/topic/day and return its rolling interest count."""
    clean_topic = _clean_topic(topic)
    if not clean_topic or not visitor_id:
        return {"status": "not_recorded", "topic": clean_topic, "region": region,
                "unique_browsers_7d": 0, "window_days": 7,
                "note": "A topic and anonymous browser ID are required to count searches."}

    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("Search timestamp must include a timezone")
    timestamp = now.timestamp()
    visitor_hash = sha256(str(visitor_id).encode("utf-8")[:256]).hexdigest()
    vector = _embedding(encoder, clean_topic)
    db_path = Path(path or os.getenv("TRENDSENSE_SEARCH_DB", str(DEFAULT_DB)))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    day = now.astimezone(timezone.utc).date().isoformat()
    cutoff = timestamp - 7 * 86400
    retention_cutoff = timestamp - RETENTION_DAYS * 86400
    selected = None
    similarity = 1.0

    with closing(sqlite3.connect(db_path, timeout=10)) as db, db:
        db.execute("PRAGMA busy_timeout=10000")
        db.execute("""CREATE TABLE IF NOT EXISTS topic_clusters (
            cluster_id INTEGER PRIMARY KEY,
            region TEXT NOT NULL,
            canonical_topic TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at REAL NOT NULL,
            last_seen REAL NOT NULL)""")
        db.execute("""CREATE TABLE IF NOT EXISTS topic_searches (
            cluster_id INTEGER NOT NULL,
            visitor_hash TEXT NOT NULL,
            search_day TEXT NOT NULL,
            searched_at REAL NOT NULL,
            news_article_count INTEGER,
            news_publisher_count INTEGER,
            youtube_match_count INTEGER,
            measured_views_per_hour REAL,
            PRIMARY KEY (cluster_id, visitor_hash, search_day),
            FOREIGN KEY (cluster_id) REFERENCES topic_clusters(cluster_id))""")
        db.execute("CREATE INDEX IF NOT EXISTS idx_topic_searches_time ON topic_searches(searched_at)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_topic_clusters_region ON topic_clusters(region, last_seen)")
        db.commit()
        db.execute("BEGIN IMMEDIATE")

        # Semantic grouping lets small wording changes such as “Jio IPO” and
        # “Jio Platforms IPO” share a topic count. Region remains part of the key.
        rows = db.execute("""SELECT cluster_id, canonical_topic, embedding
            FROM topic_clusters WHERE region=? AND last_seen>=?""", (region, retention_cutoff)).fetchall()
        best = None
        for cluster_id, canonical_topic, stored in rows:
            old = np.frombuffer(stored, dtype=np.float32)
            if old.shape != vector.shape or not np.isfinite(old).all():
                continue
            score = float(np.dot(vector, old))
            if best is None or score > best[0]:
                best = (score, cluster_id, canonical_topic)

        if best and best[0] >= SIMILARITY_THRESHOLD:
            similarity, selected, canonical = best
            db.execute("UPDATE topic_clusters SET last_seen=? WHERE cluster_id=?", (timestamp, selected))
        else:
            cursor = db.execute("""INSERT INTO topic_clusters
                (region, canonical_topic, embedding, created_at, last_seen)
                VALUES (?, ?, ?, ?, ?)""",
                (region, clean_topic, vector.tobytes(), timestamp, timestamp))
            selected, canonical = cursor.lastrowid, clean_topic

        db.execute("""INSERT INTO topic_searches
            (cluster_id, visitor_hash, search_day, searched_at, news_article_count,
             news_publisher_count, youtube_match_count, measured_views_per_hour)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(cluster_id, visitor_hash, search_day) DO UPDATE SET
             searched_at=excluded.searched_at,
             news_article_count=excluded.news_article_count,
             news_publisher_count=excluded.news_publisher_count,
             youtube_match_count=excluded.youtube_match_count,
             measured_views_per_hour=excluded.measured_views_per_hour""",
            (selected, visitor_hash, day, timestamp, news_article_count,
             news_publisher_count, youtube_match_count, measured_views_per_hour))
        db.execute("DELETE FROM topic_searches WHERE searched_at<?", (retention_cutoff,))
        db.execute("""DELETE FROM topic_clusters
            WHERE last_seen<? AND cluster_id NOT IN
                (SELECT DISTINCT cluster_id FROM topic_searches)""", (retention_cutoff,))
        count = db.execute("""SELECT COUNT(DISTINCT visitor_hash) FROM topic_searches
            WHERE cluster_id=? AND searched_at>=?""", (selected, cutoff)).fetchone()[0]

    return {
        "status": "counted",
        "topic": canonical,
        "region": region,
        "unique_browsers_7d": int(count),
        "window_days": 7,
        "semantic_match": round(float(similarity), 3),
        "scope": "This backend instance",
        "note": "Distinct browser IDs on this backend; not verified people or a measure of future views.",
    }
