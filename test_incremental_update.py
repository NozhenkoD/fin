#!/usr/bin/env python3
"""
Test script for incremental data updates.

This script demonstrates and tests the incremental update feature.
You can run this weekly via cron to keep your cache up-to-date.
"""

import sys
import os
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.cache import CacheManager
from src.data.fetcher import DataFetcher


def main():
    """Test incremental update on a single ticker."""

    print("=" * 70)
    print("INCREMENTAL UPDATE TEST")
    print("=" * 70)

    # Initialize cache manager
    cache_manager = CacheManager()

    # Test ticker
    ticker = "AAPL"

    print(f"\nTesting incremental update for {ticker}...")

    # Check current cache status
    last_date = cache_manager.get_last_cached_date(ticker)
    gap_info = cache_manager.detect_gaps(ticker)

    if last_date:
        print(f"\nCurrent cache status:")
        print(f"  Last cached date: {last_date.date()}")
        print(f"  Missing days:     {gap_info['missing_days']}")
        print(f"  Needs update:     {gap_info['needs_update']}")
    else:
        print(f"\n{ticker} is not cached yet. Will download full history.")

    if not gap_info['needs_update']:
        print(f"\n{ticker} is already up-to-date! No update needed.")
        print("\nTo test incremental update:")
        print("1. Wait a day or two (for new market data)")
        print("2. Run this script again")
        print("3. Or manually delete the cache file to test full download")
        return

    # Initialize fetcher
    fetcher = DataFetcher(period='max', interval='1d')

    print(f"\nUpdating {ticker}...")
    print("This will only fetch data since the last cached date.")

    # Perform update
    results = cache_manager.batch_update(
        tickers=[ticker],
        fetcher=fetcher,
        rate_limit_delay=0.5,
        force_update=False,
        verbose=True
    )

    # Show results
    print("\n" + "=" * 70)
    print("UPDATE RESULTS")
    print("=" * 70)

    if ticker in results['updated']:
        print(f"✓ {ticker} updated successfully")

        # Check new cache status
        new_last_date = cache_manager.get_last_cached_date(ticker)
        df = cache_manager.get_ticker_data(ticker)

        print(f"\nNew cache status:")
        print(f"  Last cached date: {new_last_date.date()}")
        print(f"  Total rows:       {len(df)}")
        print(f"  Date range:       {df.index.min().date()} to {df.index.max().date()}")

    elif ticker in results['skipped']:
        print(f"⊘ {ticker} skipped (already up-to-date)")

    else:
        error_msg = next((e for t, e in results['errors'] if t == ticker), 'Unknown error')
        print(f"✗ {ticker} failed: {error_msg}")

    print("=" * 70)
    print("\nIncremental update feature is working!")
    print("Run this weekly via cron to keep your data up-to-date.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
