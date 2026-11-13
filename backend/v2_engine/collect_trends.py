"""Collect one live sample: python -m v2_engine.collect_trends --region MULTI.

Run samples at least 30 minutes apart. The app also records uncached fetches.
This command does not install a scheduler or run a hidden background process.
"""
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv
from v2_engine.live_trends import fetch_trends, REGIONS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--region', choices=(*REGIONS, 'MULTI'), default='MULTI')
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / '.env')
    feed = fetch_trends(args.region)
    print(json.dumps({key: feed[key] for key in ('status', 'coverage', 'fetched_at', 'sources', 'history_status')}, indent=2))


if __name__ == '__main__':
    main()
