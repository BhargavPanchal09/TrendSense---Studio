import unittest
from datetime import datetime, timezone
from unittest.mock import patch
from v2_engine import live_trends


class LiveTrendsTests(unittest.TestCase):
    def setUp(self):
        live_trends._cache.clear()
        history = patch.object(live_trends, 'record_observations', side_effect=lambda rows: rows)
        self.history = history.start()
        self.addCleanup(history.stop)
        live_trends._reddit_token = None
        live_trends._reddit_token_expires = 0.0

    def test_feed_excludes_old_future_and_invalid_dates(self):
        payload = '<rss><channel>' + ''.join(
            f'<item><title>{title}</title><pubDate>{date}</pubDate></item>'
            for title, date in [('recent', 'Thu, 29 Oct 2026 00:00:00 GMT'),
                                ('old', 'Mon, 26 Oct 2026 00:00:00 GMT'),
                                ('future', 'Fri, 30 Oct 2026 00:00:00 GMT'),
                                ('bad', 'unknown')]) + '</channel></rss>'
        rows = live_trends.parse_feed(payload, datetime(2026, 10, 29, 1, tzinfo=timezone.utc))
        self.assertEqual([row['topic'] for row in rows], ['recent'])

    def test_network_failure_does_not_invent_trends(self):
        with patch.object(live_trends, 'read_url', side_effect=OSError), patch.dict('os.environ', {'YOUTUBE_API_KEY': ''}):
            feed = live_trends.fetch_trends('IN')
        self.assertEqual(feed['status'], 'unavailable')
        self.assertEqual(feed['topics'], [])
        self.assertIsNone(feed['fetched_at'])
        self.assertEqual(feed['sources'][-1]['status'], 'not_configured')

    def test_reddit_without_credentials_is_not_configured(self):
        with patch.dict('os.environ', {'REDDIT_ACCESS_TOKEN': '', 'REDDIT_CLIENT_ID': '', 'REDDIT_CLIENT_SECRET': ''}):
            self.assertIsNone(live_trends.reddit_source())

    def test_reddit_client_credentials_fetch_and_cache_token(self):
        import json
        from unittest.mock import Mock
        response = {'access_token': 'test-token', 'expires_in': 3600}
        with patch.dict('os.environ', {'REDDIT_ACCESS_TOKEN': '', 'REDDIT_CLIENT_ID': 'client', 'REDDIT_CLIENT_SECRET': 'secret'}), \
             patch.object(live_trends, 'read_url', side_effect=[json.dumps(response), json.dumps({'data': {'children': []}})]) as request:
            self.assertEqual(live_trends.reddit_source(), [])
            self.assertIn('Basic ', request.call_args_list[0].args[1]['Authorization'])
            self.assertEqual(request.call_args_list[0].kwargs['data'], b'grant_type=client_credentials')
            self.assertEqual(request.call_args_list[1].args[0], 'https://oauth.reddit.com/r/all/top?t=day&limit=50')
            self.assertEqual(request.call_args_list[1].args[1]['Authorization'], 'Bearer test-token')

    def test_semantic_matching_preserves_cached_topics(self):
        import numpy as np
        class Encoder:
            def encode(self, texts, normalize_embeddings):
                return np.array([[1., 0.], [0., 1.], [1., 0.]])
        live_trends._cache['MULTI'] = (float('inf'), {'topics': [{'topic': 'other'}, {'topic': 'related'}], 'status': 'available'})
        result = live_trends.match_trends('query', Encoder())
        self.assertEqual([row['topic'] for row in result['matches']], ['related'])
        self.assertEqual(len(live_trends.fetch_trends()['topics']), 2)

    def test_partial_source_failure_preserves_provenance(self):
        row = {'topic': 'sample', 'source': 'Google search interest', 'region': 'IN'}
        with patch.object(live_trends, 'google_source', return_value=[row]), patch.object(live_trends, 'reddit_source', side_effect=AssertionError('Reddit must remain disabled')), patch.object(live_trends, 'youtube_source', return_value=None):
            feed = live_trends.fetch_trends('IN')
        self.assertEqual(feed['status'], 'partial')
        self.assertEqual(feed['topics'][0]['source'], 'Google search interest')
        self.assertEqual([source['status'] for source in feed['sources']], ['available', 'not_configured'])

    def test_momentum_requires_google_search_and_youtube_coverage(self):
        feed = {'sources': [
            {'name': 'Google search - IN', 'status': 'available'},
            {'name': 'YouTube - IN', 'status': 'not_configured'},
        ]}
        momentum = live_trends.assess_momentum(feed, [])
        self.assertEqual(momentum['status'], 'unavailable')
        self.assertEqual(momentum['missing_sources'], ['YouTube - IN'])
        self.assertIn('no low-trend conclusion is justified', momentum['interpretation'])

    def test_complete_source_coverage_with_no_match_is_low_evidence(self):
        feed = {'sources': [
            {'name': 'Google search - IN', 'status': 'available'},
            {'name': 'YouTube - IN', 'status': 'no_recent_topics'},
            {'name': 'Reddit - r/all', 'status': 'not_configured'},
        ]}
        momentum = live_trends.assess_momentum(feed, [])
        self.assertEqual(momentum['status'], 'no_direct_signal')

    def test_chart_match_is_observed_without_claiming_growth(self):
        feed = {'sources': []}
        momentum = live_trends.assess_momentum(feed, [{'topic': 'matched'}])
        self.assertEqual(momentum['status'], 'observed')
        self.assertIn('does not measure trend growth', momentum['interpretation'])

    def test_invalid_region_is_rejected(self):
        with self.assertRaises(ValueError):
            live_trends.fetch_trends('invalid')

    def test_cached_fetch_does_not_record_duplicate_observations(self):
        row = {'topic': 'sample', 'source': 'YouTube mostPopular chart', 'region': 'IN'}
        with patch.object(live_trends, 'google_source', return_value=[]), patch.object(live_trends, 'youtube_source', return_value=[row]):
            live_trends.fetch_trends('IN')
            cached = live_trends.fetch_trends('IN')
        self.assertTrue(cached['cached'])
        self.history.assert_called_once()

    def test_storage_failure_preserves_live_evidence(self):
        self.history.side_effect = OSError('disk unavailable')
        row = {'topic': 'sample', 'source': 'YouTube mostPopular chart', 'region': 'IN'}
        with patch.object(live_trends, 'google_source', return_value=[]), patch.object(live_trends, 'youtube_source', return_value=[row]):
            feed = live_trends.fetch_trends('IN')
        self.assertEqual(feed['history_status'], 'unavailable')
        self.assertEqual(feed['topics'], [row])
        self.assertEqual(feed['status'], 'partial')

    def test_youtube_chart_observation_preserves_publication_date(self):
        import json
        payload = {'items': [{'id': 'abc', 'snippet': {'title': 'sample', 'publishedAt': '2025-01-01T00:00:00Z'}, 'statistics': {'viewCount': '123'}}]}
        with patch.dict('os.environ', {'YOUTUBE_API_KEY': 'test-key'}), patch.object(live_trends, 'read_url', return_value=json.dumps(payload)):
            rows = live_trends.youtube_source('US')
        self.assertEqual(rows[0]['views'], 123)
        self.assertEqual(rows[0]['published_at'], '2025-01-01T00:00:00Z')
        self.assertIn('observed_at', rows[0])

    def test_empty_evidence_cannot_become_positive_coach_claim(self):
        from v2_engine.evidence_report import build_report
        report = build_report(36.5, {'status': 'partial', 'matches': []}, ['This person is trending', {'bad': True}])
        self.assertIn('Current popularity of this topic is unverified', report)
        self.assertNotIn('This person is trending', report)
        self.assertIn('36.5/100', report)

    def test_unavailable_evidence_is_explicit(self):
        from v2_engine.evidence_report import build_report
        self.assertIn('Live sources are unavailable', build_report(40, {'status': 'unavailable'}))


if __name__ == '__main__':
    unittest.main()
