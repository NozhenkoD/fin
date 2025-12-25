import yfinance as yf
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime


class DataFetcher:
    """
    Data acquisition layer - wraps yfinance for fetching stock data.

    This is a source adapter. In Iteration 2, you'll create a parallel
    websocket_stream.py that outputs the same structure.
    """

    def __init__(self, period: str = '1y', interval: str = '1d'):
        """
        Initialize the data fetcher.

        Args:
            period: Data period for yfinance (e.g., '1mo', '3mo', '1y')
            interval: Data interval (e.g., '1d', '1h', '5m')
        """
        self.period = period
        self.interval = interval

    def fetch_ticker_data(
        self,
        ticker: str,
        period: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a single ticker.

        Args:
            ticker: Stock ticker symbol
            period: Optional override for the default period (ignored if start/end provided)
            start: Optional start date for fetching (enables incremental updates)
            end: Optional end date for fetching (defaults to today)

        Returns:
            DataFrame with columns: ['Open', 'High', 'Low', 'Close', 'Volume']
            Index: DatetimeIndex

        Raises:
            Exception: If data fetch fails

        Note:
            If start/end dates are provided, they take precedence over period.
            For incremental updates, pass the last cached date as 'start'.
        """
        ticker_obj = yf.Ticker(ticker)

        # Use date range if provided (for incremental updates)
        if start is not None or end is not None:
            df = ticker_obj.history(start=start, end=end, interval=self.interval)
        else:
            # Fall back to period-based fetching
            period = period or self.period
            df = ticker_obj.history(period=period, interval=self.interval)

        if df.empty:
            raise ValueError(f"No data returned for {ticker}")

        return df

    def fetch_ticker_info(self, ticker: str) -> dict:
        """
        Fetch metadata for a ticker (market cap, company name, etc.).

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with ticker metadata:
            {'marketCap': int, 'shortName': str, ...}

        Raises:
            Exception: If info fetch fails
        """
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info

        if not info:
            raise ValueError(f"No info returned for {ticker}")

        return info

    def fetch_batch(
        self,
        tickers: List[str],
        period: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> Dict[str, dict]:
        """
        Fetch data for multiple tickers with per-ticker error handling.

        This method is resilient - failures for individual tickers don't stop
        the entire batch. Failed tickers are returned with error information.

        Args:
            tickers: List of ticker symbols
            period: Optional override for the default period (ignored if start/end provided)
            start: Optional start date for fetching (enables incremental updates)
            end: Optional end date for fetching (defaults to today)

        Returns:
            Dictionary mapping ticker to results:
            {
                'AAPL': {
                    'data': DataFrame,
                    'info': dict,
                    'error': None
                },
                'INVALID': {
                    'data': None,
                    'info': None,
                    'error': 'Error message'
                }
            }
        """
        results = {}

        for ticker in tickers:
            try:
                # Fetch data and info with date range support
                data = self.fetch_ticker_data(ticker, period=period, start=start, end=end)
                info = self.fetch_ticker_info(ticker)

                results[ticker] = {
                    'data': data,
                    'info': info,
                    'error': None
                }

            except Exception as e:
                # Log warning but continue processing other tickers
                print(f"Warning: Failed to fetch {ticker}: {str(e)}")

                results[ticker] = {
                    'data': None,
                    'info': None,
                    'error': str(e)
                }

        return results

    def fetch_single(
        self,
        ticker: str,
        period: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> dict:
        """
        Fetch data for a single ticker with error handling.

        Args:
            ticker: Stock ticker symbol
            period: Optional override for the default period (ignored if start/end provided)
            start: Optional start date for fetching (enables incremental updates)
            end: Optional end date for fetching (defaults to today)

        Returns:
            Dictionary with 'data', 'info', and 'error' keys
        """
        try:
            data = self.fetch_ticker_data(ticker, period=period, start=start, end=end)
            info = self.fetch_ticker_info(ticker)

            return {
                'data': data,
                'info': info,
                'error': None
            }

        except Exception as e:
            return {
                'data': None,
                'info': None,
                'error': str(e)
            }
