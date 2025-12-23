#!/usr/bin/env python3
"""
Cache Manager for Historical OHLCV Data

Manages local Parquet cache for historical stock data with gap detection
and incremental updates.
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading

import pandas as pd

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class CacheManager:
    """
    Manages Parquet-based cache for historical OHLCV data.

    Features:
    - One Parquet file per ticker (AAPL.parquet, MSFT.parquet, etc.)
    - Fast columnar reads with pyarrow engine
    - Snappy compression (balance speed/size)
    - Gap detection for incremental updates
    - Thread-safe operations
    """

    def __init__(
        self,
        cache_dir: str = "data/cache/ohlcv",
        metadata_dir: str = "data/cache/metadata"
    ):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory for Parquet files
            metadata_dir: Directory for metadata (cache_info.json)
        """
        self.cache_dir = Path(cache_dir)
        self.metadata_dir = Path(metadata_dir)

        # Create directories if they don't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        # Metadata file path
        self.metadata_file = self.metadata_dir / "cache_info.json"

        # Thread lock for metadata updates
        self._lock = threading.Lock()

        # Load or initialize metadata
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from JSON file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """Save cache metadata to JSON file."""
        with self._lock:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)

    def _get_ticker_path(self, ticker: str) -> Path:
        """Get Parquet file path for a ticker."""
        return self.cache_dir / f"{ticker}.parquet"

    def get_ticker_data(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load ticker data from cache.

        Args:
            ticker: Stock ticker symbol
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            DataFrame with OHLCV data, or None if not cached
        """
        file_path = self._get_ticker_path(ticker)

        if not file_path.exists():
            return None

        try:
            # Read with pyarrow engine (faster)
            df = pd.read_parquet(file_path, engine='pyarrow')

            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            # Ensure timezone-naive index for consistent comparisons
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            # Apply date filters if provided
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]

            return df

        except Exception as e:
            print(f"Error reading cache for {ticker}: {e}")
            return None

    def save_ticker_data(self, ticker: str, df: pd.DataFrame) -> bool:
        """
        Save ticker data to cache.

        Args:
            ticker: Stock ticker symbol
            df: DataFrame with OHLCV data (DatetimeIndex required)

        Returns:
            True if successful, False otherwise
        """
        if df is None or df.empty:
            print(f"Warning: Cannot save empty data for {ticker}")
            return False

        try:
            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            # Convert to timezone-naive for consistent storage
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            # Sort by date
            df = df.sort_index()

            file_path = self._get_ticker_path(ticker)

            # Save with snappy compression
            df.to_parquet(
                file_path,
                engine='pyarrow',
                compression='snappy',
                index=True
            )

            # Update metadata (ensure timezone-naive for consistent comparisons)
            start_date = df.index.min().to_pydatetime()
            end_date = df.index.max().to_pydatetime()

            # Remove timezone info if present
            if start_date.tzinfo is not None:
                start_date = start_date.replace(tzinfo=None)
            if end_date.tzinfo is not None:
                end_date = end_date.replace(tzinfo=None)

            with self._lock:
                self.metadata[ticker] = {
                    'last_updated': datetime.now().isoformat(),
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'num_rows': len(df),
                    'file_size_kb': file_path.stat().st_size / 1024
                }

            self._save_metadata()

            return True

        except Exception as e:
            print(f"Error saving cache for {ticker}: {e}")
            return False

    def get_last_cached_date(self, ticker: str) -> Optional[datetime]:
        """
        Get the last cached date for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Last cached date (timezone-naive), or None if not cached
        """
        # Check metadata first (faster)
        if ticker in self.metadata:
            try:
                end_date = datetime.fromisoformat(self.metadata[ticker]['end_date'])
                # Strip timezone if present (handles old metadata)
                if end_date.tzinfo is not None:
                    end_date = end_date.replace(tzinfo=None)
                return end_date
            except (KeyError, ValueError):
                pass

        # Fallback: read from file
        df = self.get_ticker_data(ticker)
        if df is not None and not df.empty:
            last_date = df.index.max().to_pydatetime()
            # Remove timezone info to make it naive
            if last_date.tzinfo is not None:
                last_date = last_date.replace(tzinfo=None)
            return last_date

        return None

    def is_cached(self, ticker: str) -> bool:
        """Check if ticker is cached."""
        return self._get_ticker_path(ticker).exists()

    def detect_gaps(
        self,
        ticker: str,
        target_end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Detect date gaps between cached data and target date.

        Args:
            ticker: Stock ticker symbol
            target_end_date: Target end date (default: today)

        Returns:
            Dictionary with gap information:
            - has_gap: bool
            - last_cached_date: datetime or None
            - missing_days: int
            - needs_update: bool
        """
        if target_end_date is None:
            target_end_date = datetime.now()

        last_cached = self.get_last_cached_date(ticker)

        if last_cached is None:
            # Not cached at all
            return {
                'has_gap': True,
                'last_cached_date': None,
                'missing_days': None,
                'needs_update': True,
                'reason': 'not_cached'
            }

        # Calculate gap in days
        gap_days = (target_end_date - last_cached).days

        # Consider up-to-date if within 1 day (accounts for weekends)
        needs_update = gap_days > 1

        return {
            'has_gap': needs_update,
            'last_cached_date': last_cached,
            'missing_days': gap_days,
            'needs_update': needs_update,
            'reason': 'outdated' if needs_update else 'up_to_date'
        }

    def batch_update(
        self,
        tickers: List[str],
        fetcher,  # DataFetcher instance
        rate_limit_delay: float = 0.5,
        force_update: bool = False,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Batch update cache for multiple tickers.

        Args:
            tickers: List of ticker symbols
            fetcher: DataFetcher instance for downloading data
            rate_limit_delay: Delay between API calls (seconds)
            force_update: Force update even if up-to-date
            verbose: Print progress messages

        Returns:
            Dictionary with update results:
            - updated: List of updated tickers
            - skipped: List of skipped tickers (up-to-date)
            - errors: List of tickers with errors
        """
        results = {
            'updated': [],
            'skipped': [],
            'errors': []
        }

        total = len(tickers)

        for i, ticker in enumerate(tickers, 1):
            try:
                # Check if update needed
                gap_info = self.detect_gaps(ticker)

                if not force_update and not gap_info['needs_update']:
                    results['skipped'].append(ticker)
                    if verbose:
                        print(f"[{i}/{total}] {ticker}: Up-to-date, skipping")
                    continue

                # Fetch data
                if verbose:
                    if gap_info['last_cached_date']:
                        print(f"[{i}/{total}] {ticker}: Updating from {gap_info['last_cached_date'].date()}...")
                    else:
                        print(f"[{i}/{total}] {ticker}: Downloading full history...")

                # Use fetcher to download data
                ticker_data = fetcher.fetch_batch([ticker])

                if ticker not in ticker_data or ticker_data[ticker]['error']:
                    error_msg = ticker_data[ticker].get('error', 'Unknown error') if ticker in ticker_data else 'Fetch failed'
                    results['errors'].append((ticker, error_msg))
                    if verbose:
                        print(f"  ERROR: {error_msg}")
                    continue

                # Get the DataFrame
                df = ticker_data[ticker]['data']

                if df.empty:
                    results['errors'].append((ticker, 'No data returned'))
                    if verbose:
                        print(f"  ERROR: No data returned")
                    continue

                # Convert to timezone-naive before merging (yfinance returns timezone-aware data)
                if hasattr(df.index, 'tz') and df.index.tz is not None:
                    df.index = df.index.tz_localize(None)

                # Merge with existing data if available
                existing_df = self.get_ticker_data(ticker)
                if existing_df is not None and not existing_df.empty:
                    # Combine and remove duplicates
                    df = pd.concat([existing_df, df])
                    df = df[~df.index.duplicated(keep='last')]
                    df = df.sort_index()

                # Save to cache
                if self.save_ticker_data(ticker, df):
                    results['updated'].append(ticker)
                    if verbose:
                        print(f"  SUCCESS: Saved {len(df)} rows")
                else:
                    results['errors'].append((ticker, 'Failed to save'))
                    if verbose:
                        print(f"  ERROR: Failed to save")

                # Rate limiting
                if rate_limit_delay > 0 and i < total:
                    time.sleep(rate_limit_delay)

            except KeyboardInterrupt:
                if verbose:
                    print(f"\n\nInterrupted by user at {i}/{total}")
                break

            except Exception as e:
                results['errors'].append((ticker, str(e)))
                if verbose:
                    print(f"  ERROR: {e}")

        return results

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_size = sum(
            f.stat().st_size
            for f in self.cache_dir.glob("*.parquet")
        )

        return {
            'num_tickers': len(self.metadata),
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir),
            'metadata_file': str(self.metadata_file)
        }

    def clear_cache(self, ticker: Optional[str] = None):
        """
        Clear cache for a specific ticker or all tickers.

        Args:
            ticker: Ticker to clear, or None to clear all
        """
        if ticker:
            file_path = self._get_ticker_path(ticker)
            if file_path.exists():
                file_path.unlink()

            with self._lock:
                if ticker in self.metadata:
                    del self.metadata[ticker]

            self._save_metadata()
        else:
            # Clear all
            for file_path in self.cache_dir.glob("*.parquet"):
                file_path.unlink()

            with self._lock:
                self.metadata = {}

            self._save_metadata()
