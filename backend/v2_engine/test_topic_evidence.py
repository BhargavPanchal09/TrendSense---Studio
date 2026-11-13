from datetime import datetime, timezone
from unittest.mock import patch
import unittest
import numpy as np
from v2_engine import topic_evidence as topic
from v2_engine.evidence_report import build_report


class TopicEvidenceTests(unittest.TestCase):
    def setUp(self):
        topic._cache.clear()

    def test_focused_query_is_not_entire_marketing_title(self):
        self.assertEqual(topic.make_topic_query('Jio Platforms IPO Explained: What Investors Should Check', 'Jio Platforms, Reliance'), 'Jio Platforms IPO')
        self.assertEqual(topic.make_topic_query('title', 'entities', 'specific topic'), 'specific topic')

    def test_news_dates_links_and_duplicates(self):
        rows = [('recent', 'Thu, 29 Oct 2026 00:00:00 GMT', 'https://example.com/a'),
                ('recent', 'Thu, 29 Oct 2026 00:00:00 GMT', 'https://example.com/b'),
                ('old', 'Mon, 26 Oct 2026 00:00:00 GMT', 'https://example.com/c'),
                ('future', 'Fri, 30 Oct 2026 00:00:00 GMT', 'https://example.com/d'),
                ('unsafe', 'Thu, 29 Oct 2026 00:00:00 GMT', 'javascript:bad')]
        xml = '<rss><channel>' + ''.join(f'<item><title>{name}</title><pubDate>{date}</pubDate><link>{link}</link><source url="https://example.com">Example</source></item>' for name, date, link in rows) + '</channel></rss>'
        actual = topic.parse_news(xml, datetime(2026, 10, 29, 1, tzinfo=timezone.utc))
        self.assertEqual([row['title'] for row in actual], ['recent'])

    def test_fetch_failure_is_not_zero_popularity(self):
        with patch.object(topic, 'read_url', side_effect=OSError):
            result = topic.check_topic('anything', None)
        self.assertEqual(result['status'], 'unavailable')
        self.assertEqual(result['current_popularity'], 'unverified')

    def test_multi_region_news_checks_all_selected_countries(self):
        from urllib.parse import urlparse, parse_qs
        with patch.object(topic, 'read_url', return_value=b'<rss><channel/></rss>') as request:
            result = topic.check_topic('Jio IPO', None, 'MULTI')
        countries = {parse_qs(urlparse(call.args[0]).query)['gl'][0] for call in request.call_args_list}
        self.assertEqual(countries, set(topic.LOCALES))
        self.assertEqual(result['candidate_count'], 0)
        self.assertEqual(result['status'], 'no_related_coverage_found')
        self.assertEqual(len(result['sources']), 5)

    def test_partial_news_failure_is_not_complete_negative_result(self):
        def fetch(url):
            if 'gl=IN' in url:
                raise OSError('offline')
            return b'<rss><channel/></rss>'
        with patch.object(topic, 'read_url', side_effect=fetch):
            result = topic.check_topic('Jio IPO', None, 'MULTI')
        self.assertEqual(result['status'], 'unavailable')
        self.assertEqual(result['sources'][0]['status'], 'unavailable')

    def test_focused_search_finds_coverage_without_general_feed_match(self):
        class Encoder:
            def encode(self, texts, normalize_embeddings):
                return np.array([[1., 0.], [1., 0.]])
        row = {'title': 'IPO news', 'publisher': 'Example', 'publisher_url': 'https://example.com', 'url': 'https://example.com/a', 'published_at': '2026-10-29T00:00:00Z'}
        with patch.object(topic, 'read_url', return_value=b'ignored'), patch.object(topic, 'parse_news', return_value=[row]):
            result = topic.check_topic('IPO', Encoder())
        report = build_report(34.7, {'status': 'partial', 'matches': []}, topic=result)
        self.assertEqual(result['status'], 'recent_coverage_found')
        self.assertEqual(result['current_popularity'], 'unverified')
        self.assertIn('Topic-specific search found 1 related articles', report)
        self.assertIn('34.7/100', report)


if __name__ == '__main__':
    unittest.main()
