#!/usr/bin/env python3
"""
Ticker Fetcher Utility

Fetches all tickers from NASDAQ (or other exchanges) and saves them
to tickers.json with metadata (company name, sector).
"""

import sys
import os
import json
import time
from typing import Dict, Any

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import yfinance as yf
import pandas as pd


def get_nasdaq_tickers() -> list:
    """
    Fetch NASDAQ ticker list from official NASDAQ FTP site.

    Returns:
        List of ticker symbols
    """
    try:
        # NASDAQ provides ticker lists via FTP
        url = "ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqtraded.txt"
        df = pd.read_csv(url, sep='|')

        # Remove the last row which is often a footer/summary row
        df = df[:-1]

        # Filter for NASDAQ listed stocks (not OTC, not test symbols)
        nasdaq_df = df[
            (df['Listing Exchange'] == 'Q') |  # NASDAQ Global Select
            (df['Listing Exchange'] == 'G')    # NASDAQ Global Market
        ]

        # Get symbols and convert to string, handle NaN values
        symbols = nasdaq_df['Symbol'].dropna().astype(str)

        # Remove $ signs (used for special shares)
        symbols = symbols.str.replace('$', '', regex=False)

        # Get unique values and convert to list
        tickers = symbols.unique().tolist()

        # Remove empty strings, whitespace, and clean up
        tickers = [t.strip() for t in tickers if isinstance(t, str) and t.strip()]

        print(f"Successfully fetched {len(tickers)} NASDAQ tickers")

        return tickers

    except Exception as e:
        print(f"Error fetching from NASDAQ FTP: {e}")
        import traceback
        traceback.print_exc()
        return []

def get_nyse_tickers() -> list:
    """
    Fetch NYSE ticker list.

    Returns:
        List of ticker symbols
    """
    try:
        url = "ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqtraded.txt"
        df = pd.read_csv(url, sep='|')

        # Remove the last row which is often a footer/summary row
        df = df[:-1]

        # Filter for NYSE listed stocks
        nyse_df = df[df['Listing Exchange'] == 'N']

        # Get symbols and convert to string, handle NaN values
        symbols = nyse_df['Symbol'].dropna().astype(str)

        # Remove $ signs (used for special shares)
        symbols = symbols.str.replace('$', '', regex=False)

        # Get unique values and convert to list
        tickers = symbols.unique().tolist()

        # Remove empty strings, whitespace, and clean up
        tickers = [t.strip() for t in tickers if isinstance(t, str) and t.strip()]

        print(f"Successfully fetched {len(tickers)} NYSE tickers")

        return tickers

    except Exception as e:
        print(f"Error fetching NYSE tickers: {e}")
        import traceback
        traceback.print_exc()
        return []


def fetch_ticker_metadata(ticker: str, retry_count: int = 3, skip_metadata: bool = False) -> dict:
    """
    Fetch metadata for a single ticker using yfinance.

    Args:
        ticker: Stock ticker symbol
        retry_count: Number of retries on failure
        skip_metadata: If True, skip yfinance fetch and return minimal info

    Returns:
        Dictionary with company_name and sector, or minimal info if fetch fails
    """
    if skip_metadata:
        # Skip API call to avoid rate limiting
        return {
            'ticker': ticker,
            'company_name': ticker,
            'sector': 'Unknown'
        }

    for attempt in range(retry_count):
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info

            # Extract relevant information
            company_name = info.get('longName') or info.get('shortName') or ticker
            sector = info.get('sector') or info.get('industry') or 'Unknown'

            return {
                'ticker': ticker,
                'company_name': company_name,
                'sector': sector
            }

        except Exception as e:
            error_msg = str(e).lower()

            # Check for rate limiting
            if 'rate limit' in error_msg or 'too many requests' in error_msg:
                if attempt < retry_count - 1:
                    wait_time = (attempt + 1) * 5  # 5, 10, 15 seconds
                    print(f"Rate limited on {ticker}, waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Warning: Rate limited on {ticker}, skipping metadata")
                    return {
                        'ticker': ticker,
                        'company_name': ticker,
                        'sector': 'Unknown'
                    }
            else:
                if attempt < retry_count - 1:
                    time.sleep(0.5)  # Brief delay before retry
                    continue
                else:
                    # If all retries fail, return minimal info
                    print(f"Warning: Could not fetch metadata for {ticker}: {e}")
                    return {
                        'ticker': ticker,
                        'company_name': ticker,
                        'sector': 'Unknown'
                    }


def fetch_all_tickers(
    nasdaq: bool = True,
    nyse: bool = False,
    amex: bool = False,
    output_file: str = 'data/tickers.json',
    delay: float = 0.5,
    limit: int = None,
    skip_metadata: bool = False
) -> Dict[str, dict]:
    """
    Fetch all tickers from specified exchanges and save to JSON.

    Args:
        nasdaq: Include NASDAQ tickers
        nyse: Include NYSE tickers
        amex: Include AMEX tickers
        output_file: Path to output JSON file
        delay: Delay between API calls (seconds) to avoid rate limiting
        limit: Optional limit on number of tickers to fetch (for testing)
        skip_metadata: If True, skip fetching company names/sectors (much faster, avoids rate limits)

    Returns:
        Dictionary mapping ticker symbols to metadata
    """
    print(f"Fetching ticker list...")
    print(f"  NASDAQ: {nasdaq}")
    print(f"  NYSE: {nyse}")
    print(f"  AMEX: {amex}")

    # Get ticker list
    tickers = []

    if nasdaq:
        print("Downloading NASDAQ tickers...")
        nasdaq_list = get_nasdaq_tickers()
        tickers.extend(nasdaq_list)
        print(f"  Found {len(nasdaq_list)} NASDAQ tickers")

    if nyse:
        print("Downloading NYSE tickers...")
        nyse_list = get_nyse_tickers()
        tickers.extend(nyse_list)
        print(f"  Found {len(nyse_list)} NYSE tickers")

    if amex:
        print("Note: AMEX ticker fetching not yet implemented")

    # Remove duplicates
    tickers = list(set(tickers))

    print(f"\nTotal unique tickers found: {len(tickers)}")

    if not tickers:
        print("\nError: No tickers found. Please check your internet connection or try again later.")
        return {}

    if limit:
        tickers = tickers[:limit]
        print(f"Limiting to first {limit} tickers for testing")

    # Fetch metadata for each ticker
    ticker_data = {}
    total = len(tickers)

    if skip_metadata:
        print(f"\nSkipping metadata fetch (--skip-metadata enabled)")
        print(f"Creating ticker list with {total} tickers (company names will be set to ticker symbols)")
        for ticker in tickers:
            ticker_data[ticker] = {
                'ticker': ticker,
                'company_name': ticker,
                'sector': 'Unknown'
            }
    else:
        print(f"\nFetching metadata for {total} tickers...")
        print("This may take a while. Progress will be shown every 10 tickers.")
        print(f"Delay between requests: {delay}s (increase with --delay if you hit rate limits)\n")

        for i, ticker in enumerate(tickers, 1):
            try:
                metadata = fetch_ticker_metadata(ticker, skip_metadata=skip_metadata)
                ticker_data[ticker] = metadata

                # Show progress every 10 tickers
                if i % 10 == 0:
                    print(f"Progress: {i}/{total} ({i/total*100:.1f}%) - Last: {ticker}")

                # Delay to avoid rate limiting
                if delay > 0:
                    time.sleep(delay)

            except KeyboardInterrupt:
                print(f"\n\nInterrupted by user at {i}/{total} tickers")
                print(f"Saving {len(ticker_data)} tickers fetched so far...")
                break

            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue

    # Save to JSON file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(ticker_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SUCCESS: Saved {len(ticker_data)} tickers to {output_file}")
    print(f"{'='*60}\n")

    return ticker_data


def main():
    """Main entry point for the ticker fetcher utility."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Fetch stock tickers and save to tickers.json',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch all NASDAQ tickers (default)
  python -m src.utils.ticker_fetcher

  # Fetch NASDAQ + NYSE
  python -m src.utils.ticker_fetcher --nasdaq --nyse

  # Fetch with custom output file
  python -m src.utils.ticker_fetcher --output data/my_tickers.json

  # Test with first 50 tickers only
  python -m src.utils.ticker_fetcher --limit 50

  # Faster fetching (less delay, higher risk of rate limiting)
  python -m src.utils.ticker_fetcher --delay 0.05
        """
    )

    parser.add_argument(
        '--nasdaq',
        action='store_true',
        default=True,
        help='Include NASDAQ tickers (default: True)'
    )

    parser.add_argument(
        '--nyse',
        action='store_true',
        help='Include NYSE tickers (default: False)'
    )

    parser.add_argument(
        '--amex',
        action='store_true',
        help='Include AMEX tickers (default: False)'
    )

    parser.add_argument(
        '--no-nasdaq',
        action='store_true',
        help='Exclude NASDAQ tickers'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/tickers.json',
        help='Output file path (default: data/tickers.json)'
    )

    parser.add_argument(
        '--delay',
        type=float,
        default=0.5,
        help='Delay between API calls in seconds (default: 0.5, increase to 1.0+ if rate limited)'
    )

    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of tickers to fetch (for testing)'
    )

    parser.add_argument(
        '--skip-metadata',
        action='store_true',
        help='Skip fetching company names/sectors (FAST, avoids rate limits, recommended for large lists)'
    )

    args = parser.parse_args()

    # Handle --no-nasdaq flag
    nasdaq = args.nasdaq and not args.no_nasdaq

    # Fetch and save tickers
    try:
        fetch_all_tickers(
            nasdaq=nasdaq,
            nyse=args.nyse,
            amex=args.amex,
            output_file=args.output,
            delay=args.delay,
            limit=args.limit,
            skip_metadata=args.skip_metadata
        )

        print("You can now run the screener with:")
        print(f"  python src/main.py")

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
