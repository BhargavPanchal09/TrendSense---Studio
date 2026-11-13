"""Persist chart observations to measure view growth instead of guessing it.

Only a repeated observation of the SAME video in the SAME regional chart can
produce a rate. Missing chart entries are not treated as zero views.
"""
from datetime import datetime, timezone
from contextlib import closing
from pathlib import Path
import os
import sqlite3

DEFAULT_DB = Path(__file__).resolve().parents[2] / 'data/trend_observations.sqlite3'


def record_observations(rows, path=None, now=None):
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError('Observation time must include a timezone')
    timestamp = now.timestamp()
    path = Path(path or os.getenv('TRENDSENSE_HISTORY_DB', str(DEFAULT_DB)))
    path.parent.mkdir(parents=True, exist_ok=True)
    result = []
    # sqlite's context manager commits transactions but does not close handles.
    with closing(sqlite3.connect(path, timeout=10)) as db, db:
        db.execute('''CREATE TABLE IF NOT EXISTS observations (
            source TEXT, region TEXT, item TEXT, observed REAL, title TEXT,
            views INTEGER, rank INTEGER,
            PRIMARY KEY (source, region, item, observed))''')
        for row in rows:
            row = dict(row)
            if row.get('source') != 'YouTube mostPopular chart' or not row.get('source_url'):
                result.append(row)
                continue
            views = row.get('views')
            if not isinstance(views, int) or isinstance(views, bool) or views < 0:
                row['growth'] = {'status': 'missing_view_count'}
                result.append(row)
                continue
            key = (row['source'], row.get('region', ''), row['source_url'])
            # Prefer a ~24-hour comparison; use a shorter interval when history
            # is young. Never compare an instant refresh or a very stale sample.
            previous = db.execute('''SELECT observed, views, rank FROM observations
                WHERE source=? AND region=? AND item=? AND observed BETWEEN ? AND ?
                ORDER BY ABS(observed - ?) LIMIT 1''',
                (*key, timestamp - 48 * 3600, timestamp - 1800, timestamp - 24 * 3600)).fetchone()
            growth = {'status': 'collecting_history', 'minimum_interval_minutes': 30}
            if previous:
                then, before, old_rank = previous
                hours = (timestamp - then) / 3600
                change = views - before
                if change < 0:
                    growth = {'status': 'counter_revised', 'note': 'View count decreased; no growth rate inferred.'}
                else:
                    growth = {'status': 'measured', 'window_hours': round(hours, 2),
                              'views_added': change, 'views_per_hour': round(change / hours, 2),
                              'previous_observed_at': datetime.fromtimestamp(then, timezone.utc).isoformat(),
                              'note': 'Observed video view growth, not worldwide topic growth or a forecast.'}
                    if old_rank is not None and row.get('chart_rank') is not None:
                        growth['rank_positions_gained'] = old_rank - row['chart_rank']
            row['growth'] = growth
            db.execute('INSERT OR IGNORE INTO observations VALUES (?, ?, ?, ?, ?, ?, ?)',
                       (*key, timestamp, row.get('topic', ''), views, row.get('chart_rank')))
            result.append(row)
        # Bound disk usage. Export a research dataset before this 90-day window.
        db.execute('DELETE FROM observations WHERE observed < ?', (timestamp - 90 * 86400,))
    return result
