#!/usr/bin/env python3
"""
Historical Data Downloader

CLI tool for initial download of historical OHLCV data to local Parquet cache.
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.cache import CacheManager
from src.data.sp500_loader import load_sp500_tickers, load_custom_tickers, load_all_tickers
from src.data.fetcher import DataFetcher

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Warning: tqdm not installed. Progress bars will not be shown.")
    print("Install with: pip install tqdm")


def parse_date(date_string: str) -> datetime:
    """Parse date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_string, '%Y-%m-%d')
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_string}. Use YYYY-MM-DD")


def main():
    """Main entry point for the download utility."""
    parser = argparse.ArgumentParser(
        description='Download historical stock data to local Parquet cache',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download S&P 500 (default: max available data)
  python -m src.data.download_history --sp500

  # Download custom tickers (from data/custom_tickers.csv)
  python -m src.data.download_history --custom

  # Download all tickers (S&P 500 + custom)
  python -m src.data.download_history --all

  # Custom period (5 years instead of max)
  python -m src.data.download_history --sp500 --period 5y

  # Custom date range
  python -m src.data.download_history --sp500 --start 2018-01-01 --end 2023-12-31

  # Rate limit adjustment
  python -m src.data.download_history --sp500 --delay 1.0

  # Test with limited tickers
  python -m src.data.download_history --sp500 --limit 10

  # Custom ticker list
  python -m src.data.download_history --tickers AAPL MSFT GOOGL

  # Force re-download (ignore existing cache)
  python -m src.data.download_history --sp500 --force
        """
    )

    # Ticker source
    parser.add_argument(
        '--sp500',
        action='store_true',
        help='Download all S&P 500 tickers from data/sp500.csv'
    )

    parser.add_argument(
        '--custom',
        action='store_true',
        help='Download custom tickers from data/custom_tickers.csv'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all tickers (S&P 500 + custom combined)'
    )

    parser.add_argument(
        '--tickers',
        nargs='+',
        help='Specific tickers to download (space-separated)'
    )

    parser.add_argument(
        '--sp500-file',
        default='data/sp500.csv',
        help='Path to S&P 500 CSV file (default: data/sp500.csv)'
    )

    parser.add_argument(
        '--custom-file',
        default='data/custom_tickers.csv',
        help='Path to custom tickers CSV file (default: data/custom_tickers.csv)'
    )

    # Date range
    parser.add_argument(
        '--start',
        type=parse_date,
        help='Start date (YYYY-MM-DD). Default: max available data'
    )

    parser.add_argument(
        '--end',
        type=parse_date,
        help='End date (YYYY-MM-DD). Default: today'
    )

    parser.add_argument(
        '--period',
        default='max',
        help='Period for yfinance (e.g., 1y, 2y, 5y, 10y, max). Ignored if --start is provided.'
    )

    # Rate limiting
    parser.add_argument(
        '--delay',
        type=float,
        default=0.5,
        help='Delay between API calls in seconds (default: 0.5)'
    )

    # Options
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of tickers (for testing)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if already cached'
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

    # Validate arguments
    ticker_sources = sum([args.sp500, args.custom, args.all, bool(args.tickers)])
    if ticker_sources == 0:
        parser.error("Must specify one of: --sp500, --custom, --all, or --tickers")
    if ticker_sources > 1:
        parser.error("Can only specify one ticker source at a time")

    # Get ticker list
    print("=" * 60)
    print("HISTORICAL DATA DOWNLOADER")
    print("=" * 60)

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
    elif args.custom:
        try:
            tickers = load_custom_tickers(args.custom_file)
            print(f"Loaded {len(tickers)} custom tickers from {args.custom_file}")
        except FileNotFoundError as e:
            print(f"\nError: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"\nError loading custom tickers: {e}")
            sys.exit(1)
    elif args.all:
        try:
            tickers = load_all_tickers(args.sp500_file, args.custom_file)
            print(f"Loaded {len(tickers)} total tickers (S&P 500 + custom)")
        except Exception as e:
            print(f"\nError loading tickers: {e}")
            sys.exit(1)
    else:
        tickers = args.tickers
        print(f"Using {len(tickers)} tickers from command line")

    # Apply limit
    if args.limit:
        tickers = tickers[:args.limit]
        print(f"Limited to first {args.limit} tickers for testing")

    # Determine date range
    if args.start or args.end:
        # Use explicit dates
        start_date = args.start if args.start else datetime(1900, 1, 1)  # Very old date to get max data
        end_date = args.end if args.end else datetime.now()
        print(f"Date range: {start_date.date()} to {end_date.date()}")
        period = None
    else:
        # Use period
        period = args.period
        print(f"Period: {period}")
        start_date = None
        end_date = None

    print(f"Rate limit delay: {args.delay}s")
    print(f"Cache directory: {args.cache_dir}")
    print(f"Force re-download: {args.force}")
    print()

    # Initialize components
    cache_manager = CacheManager(cache_dir=args.cache_dir)

    # Create DataFetcher with appropriate period
    if period:
        fetcher = DataFetcher(period=period, interval='1d')
    else:
        # yfinance doesn't support explicit start/end in constructor
        # We'll use period='max' and filter after
        fetcher = DataFetcher(period='max', interval='1d')

    # Download data
    print(f"Downloading historical data for {len(tickers)} tickers...")
    print("This may take a while. Press Ctrl+C to stop (progress will be saved).\n")

    try:
        # Use batch_update with progress tracking
        if HAS_TQDM:
            # Wrap the batch update with tqdm
            results = {
                'updated': [],
                'skipped': [],
                'errors': []
            }

            with tqdm(total=len(tickers), desc="Downloading", unit="ticker") as pbar:
                for ticker in tickers:
                    # Create mini-batch of 1 for progress tracking
                    batch_result = cache_manager.batch_update(
                        tickers=[ticker],
                        fetcher=fetcher,
                        rate_limit_delay=args.delay,
                        force_update=args.force,
                        verbose=args.verbose
                    )

                    # Accumulate results
                    results['updated'].extend(batch_result['updated'])
                    results['skipped'].extend(batch_result['skipped'])
                    results['errors'].extend(batch_result['errors'])

                    # Update progress
                    pbar.update(1)
                    pbar.set_postfix({
                        'updated': len(results['updated']),
                        'skipped': len(results['skipped']),
                        'errors': len(results['errors'])
                    })
        else:
            # No progress bar, use batch update directly
            results = cache_manager.batch_update(
                tickers=tickers,
                fetcher=fetcher,
                rate_limit_delay=args.delay,
                force_update=args.force,
                verbose=True
            )

    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user. Progress has been saved.")

        # Get current results from cache
        results = {
            'updated': [t for t in tickers if cache_manager.is_cached(t)],
            'skipped': [],
            'errors': []
        }

    # Print summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Total tickers processed: {len(tickers)}")
    print(f"Successfully downloaded:  {len(results['updated'])}")
    print(f"Skipped (up-to-date):     {len(results['skipped'])}")
    print(f"Errors:                   {len(results['errors'])}")

    if results['errors']:
        print("\nErrors:")
        for ticker, error in results['errors'][:10]:  # Show first 10
            print(f"  {ticker}: {error}")
        if len(results['errors']) > 10:
            print(f"  ... and {len(results['errors']) - 10} more")

    # Cache stats
    stats = cache_manager.get_cache_stats()
    print(f"\nCache statistics:")
    print(f"  Total tickers cached: {stats['num_tickers']}")
    print(f"  Total cache size:     {stats['total_size_mb']:.1f} MB")
    print(f"  Cache directory:      {stats['cache_dir']}")
    print("=" * 60)

    if len(results['updated']) > 0:
        print("\nSuccess! You can now run backtests with:")
        print("  python -m src.backtest.run --strategy SMA200Crossover --sp500")
    else:
        print("\nNo data downloaded. Check errors above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
