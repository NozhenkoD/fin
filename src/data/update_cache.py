#!/usr/bin/env python3
"""
Cache Update Utility

CLI tool for incremental updates of local Parquet cache with gap detection.
Ideal for weekly cron jobs to keep cache up-to-date.
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.cache import CacheManager
from src.data.sp500_loader import load_sp500_tickers
from src.data.fetcher import DataFetcher

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def main():
    """Main entry point for the update utility."""
    parser = argparse.ArgumentParser(
        description='Update local Parquet cache with latest data (incremental updates only)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update all cached tickers
  python -m src.data.update_cache

  # Update specific tickers
  python -m src.data.update_cache --tickers AAPL MSFT GOOGL

  # Update S&P 500 (reads from sp500.csv)
  python -m src.data.update_cache --sp500

  # Increase rate limit delay
  python -m src.data.update_cache --delay 1.0

  # Force update all (even if up-to-date)
  python -m src.data.update_cache --force

Note: This tool only fetches missing dates. It's designed for incremental updates.
For initial downloads, use: python -m src.data.download_history
        """
    )

    # Ticker source
    parser.add_argument(
        '--sp500',
        action='store_true',
        help='Update all S&P 500 tickers from data/sp500.csv'
    )

    parser.add_argument(
        '--tickers',
        nargs='+',
        help='Specific tickers to update (space-separated). If not specified, updates all cached tickers.'
    )

    parser.add_argument(
        '--sp500-file',
        default='data/sp500.csv',
        help='Path to S&P 500 CSV file (default: data/sp500.csv)'
    )

    # Options
    parser.add_argument(
        '--delay',
        type=float,
        default=0.5,
        help='Delay between API calls in seconds (default: 0.5)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force update even if already up-to-date'
    )

    parser.add_argument(
        '--cache-dir',
        default='data/cache/ohlcv',
        help='Cache directory (default: data/cache/ohlcv)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Initialize cache manager
    cache_manager = CacheManager(cache_dir=args.cache_dir)

    print("=" * 60)
    print("CACHE UPDATE UTILITY")
    print("=" * 60)

    # Get ticker list
    if args.sp500:
        try:
            tickers = load_sp500_tickers(args.sp500_file)
            print(f"Loaded {len(tickers)} S&P 500 tickers from {args.sp500_file}")
        except FileNotFoundError as e:
            print(f"\nError: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"\nError loading S&P 500 tickers: {e}")
            sys.exit(1)
    elif args.tickers:
        tickers = args.tickers
        print(f"Using {len(tickers)} tickers from command line")
    else:
        # Update all cached tickers
        cache_stats = cache_manager.get_cache_stats()
        tickers = list(cache_manager.metadata.keys())

        if not tickers:
            print("\nNo cached tickers found. Use download_history.py for initial download.")
            print("Example: python -m src.data.download_history --sp500")
            sys.exit(1)

        print(f"Updating {len(tickers)} cached tickers")

    print(f"Cache directory: {args.cache_dir}")
    print(f"Rate limit delay: {args.delay}s")
    print()

    # Check gaps before updating
    print("Checking for gaps...")
    gaps_found = []
    up_to_date = []

    for ticker in tickers:
        gap_info = cache_manager.detect_gaps(ticker)
        if gap_info['needs_update']:
            gaps_found.append((ticker, gap_info))
        else:
            up_to_date.append(ticker)

    print(f"  Needs update: {len(gaps_found)}")
    print(f"  Up-to-date:   {len(up_to_date)}")

    if not gaps_found and not args.force:
        print("\nAll tickers are up-to-date! No updates needed.")
        stats = cache_manager.get_cache_stats()
        print(f"\nCache statistics:")
        print(f"  Total tickers: {stats['num_tickers']}")
        print(f"  Total size:    {stats['total_size_mb']:.1f} MB")
        print("=" * 60)
        return

    # Show sample gaps
    if gaps_found:
        print("\nSample gaps (first 5):")
        for ticker, gap_info in gaps_found[:5]:
            if gap_info['last_cached_date']:
                print(f"  {ticker}: Last cached {gap_info['last_cached_date'].date()}, {gap_info['missing_days']} days behind")
            else:
                print(f"  {ticker}: Not cached yet")

    # Determine tickers to update
    if args.force:
        tickers_to_update = tickers
        print(f"\nForce mode: Updating all {len(tickers)} tickers")
    else:
        tickers_to_update = [t for t, _ in gaps_found]
        print(f"\nUpdating {len(tickers_to_update)} tickers with gaps")

    if not tickers_to_update:
        print("\nNo tickers to update.")
        return

    print("\nStarting update...")
    print("Press Ctrl+C to stop (progress will be saved).\n")

    # Initialize data fetcher (use max period to get full history)
    fetcher = DataFetcher(period='max', interval='1d')

    try:
        # Use batch_update with progress tracking
        if HAS_TQDM:
            results = {
                'updated': [],
                'skipped': [],
                'errors': []
            }

            with tqdm(total=len(tickers_to_update), desc="Updating", unit="ticker") as pbar:
                for ticker in tickers_to_update:
                    batch_result = cache_manager.batch_update(
                        tickers=[ticker],
                        fetcher=fetcher,
                        rate_limit_delay=args.delay,
                        force_update=args.force,
                        verbose=args.verbose
                    )

                    results['updated'].extend(batch_result['updated'])
                    results['skipped'].extend(batch_result['skipped'])
                    results['errors'].extend(batch_result['errors'])

                    pbar.update(1)
                    pbar.set_postfix({
                        'updated': len(results['updated']),
                        'errors': len(results['errors'])
                    })
        else:
            results = cache_manager.batch_update(
                tickers=tickers_to_update,
                fetcher=fetcher,
                rate_limit_delay=args.delay,
                force_update=args.force,
                verbose=True
            )

    except KeyboardInterrupt:
        print("\n\nUpdate interrupted by user. Progress has been saved.")
        results = {
            'updated': [t for t in tickers_to_update if cache_manager.is_cached(t)],
            'skipped': [],
            'errors': []
        }

    # Print summary
    print("\n" + "=" * 60)
    print("UPDATE SUMMARY")
    print("=" * 60)
    print(f"Tickers checked:    {len(tickers)}")
    print(f"Already up-to-date: {len(up_to_date)}")
    print(f"Updated:            {len(results['updated'])}")
    print(f"Skipped:            {len(results['skipped'])}")
    print(f"Errors:             {len(results['errors'])}")

    if results['errors']:
        print("\nErrors:")
        for ticker, error in results['errors'][:10]:
            print(f"  {ticker}: {error}")
        if len(results['errors']) > 10:
            print(f"  ... and {len(results['errors']) - 10} more")

    # Cache stats
    stats = cache_manager.get_cache_stats()
    print(f"\nCache statistics:")
    print(f"  Total tickers: {stats['num_tickers']}")
    print(f"  Total size:    {stats['total_size_mb']:.1f} MB")
    print(f"  Cache dir:     {stats['cache_dir']}")
    print("=" * 60)

    if len(results['updated']) > 0:
        print("\nCache updated successfully!")
    else:
        print("\nNo updates performed.")


if __name__ == '__main__':
    main()
