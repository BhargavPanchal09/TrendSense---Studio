import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from v2_engine.trend_history import record_observations


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / 'history.sqlite3'
        self.now = datetime(2026, 10, 29, tzinfo=timezone.utc)
        self.row = {'source': 'YouTube mostPopular chart', 'source_url': 'https://www.youtube.com/watch?v=abc',
                    'topic': 'Example', 'region': 'IN', 'views': 100, 'chart_rank': 4}

    def record(self, hours=0, **updates):
        return record_observations([{**self.row, **updates}], self.path, self.now + timedelta(hours=hours))[0]['growth']

    def test_first_and_quick_refresh_do_not_invent_growth(self):
        self.assertEqual(self.record()['status'], 'collecting_history')
        self.assertEqual(self.record(.1, views=200)['status'], 'collecting_history')

    def test_rate_and_rank_are_from_repeated_same_video(self):
        self.record()
        result = self.record(2, views=500, chart_rank=2)
        self.assertEqual(result['views_per_hour'], 200)
        self.assertEqual(result['views_added'], 400)
        self.assertEqual(result['rank_positions_gained'], 2)

    def test_other_region_or_video_does_not_share_history(self):
        self.record()
        self.assertEqual(self.record(2, region='US')['status'], 'collecting_history')
        self.assertEqual(self.record(2, source_url='https://www.youtube.com/watch?v=def')['status'], 'collecting_history')

    def test_revised_and_missing_counters_are_not_negative_trends(self):
        self.record()
        self.assertEqual(self.record(1, views=90)['status'], 'counter_revised')
        self.assertEqual(self.record(2, views=None)['status'], 'missing_view_count')

    def test_stale_or_future_observations_not_used(self):
        self.record()
        self.assertEqual(self.record(49)['status'], 'collecting_history')
        self.assertEqual(self.record(-1)['status'], 'collecting_history')

    def test_empty_chart_is_not_a_zero_count(self):
        self.record()
        self.assertEqual(record_observations([], self.path, self.now + timedelta(hours=1)), [])
        self.assertEqual(self.record(2, views=300)['views_per_hour'], 100)


if __name__ == '__main__':
    unittest.main()
