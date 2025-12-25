#!/usr/bin/env python3
"""
Ticker List Loader

Loads ticker lists from CSV files (S&P 500, custom watchlist, or all combined).
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def load_sp500_tickers(csv_path: str = "data/sp500.csv") -> List[str]:
    """
    Load S&P 500 ticker symbols from CSV file.

    Args:
        csv_path: Path to CSV file with S&P 500 tickers

    Returns:
        List of ticker symbols

    Expected CSV format:
        Symbol,Name,Sector
        AAPL,Apple Inc.,Technology
        MSFT,Microsoft Corporation,Technology
    """
    csv_file = Path(csv_path)

    if not csv_file.exists():
        raise FileNotFoundError(
            f"S&P 500 CSV file not found: {csv_path}\n\n"
            f"Please create a CSV file with columns: Symbol,Name,Sector\n"
            f"Example:\n"
            f"  Symbol,Name,Sector\n"
            f"  AAPL,Apple Inc.,Technology\n"
            f"  MSFT,Microsoft Corporation,Technology\n"
        )

    try:
        df = pd.read_csv(csv_file)

        # Strip whitespace from column names
        df.columns = df.columns.str.strip()

        # Check for required column (Symbol is primary)
        if 'Symbol' not in df.columns and 'Ticker' not in df.columns:
            raise ValueError(
                f"CSV must have 'Symbol' or 'Ticker' column. Found columns: {list(df.columns)}"
            )

        # Use Symbol column, or Ticker as fallback
        symbol_col = 'Symbol' if 'Symbol' in df.columns else 'Ticker'

        # Get unique symbols and clean (strip whitespace, remove empty)
        tickers = df[symbol_col].dropna().astype(str).str.strip().unique().tolist()

        # Remove empty strings
        tickers = [t for t in tickers if t]

        return tickers

    except Exception as e:
        raise RuntimeError(f"Error loading S&P 500 tickers from {csv_path}: {e}")


def load_sp500_metadata(csv_path: str = "data/sp500.csv") -> Dict[str, Dict[str, Any]]:
    """
    Load S&P 500 ticker metadata from CSV file.

    Args:
        csv_path: Path to CSV file with S&P 500 tickers

    Returns:
        Dictionary mapping ticker symbols to metadata:
        {
            'AAPL': {
                'ticker': 'AAPL',
                'company_name': 'Apple Inc.',
                'sector': 'Technology'
            },
            ...
        }

    Expected CSV format:
        Symbol,Name,Sector
        AAPL,Apple Inc.,Technology
        MSFT,Microsoft Corporation,Technology
    """
    csv_file = Path(csv_path)

    if not csv_file.exists():
        raise FileNotFoundError(
            f"S&P 500 CSV file not found: {csv_path}\n\n"
            f"Please create a CSV file with columns: Symbol,Name,Sector\n"
        )

    try:
        df = pd.read_csv(csv_file)

        # Strip whitespace from column names
        df.columns = df.columns.str.strip()

        # Check for required column
        if 'Symbol' not in df.columns and 'Ticker' not in df.columns:
            raise ValueError(
                f"CSV must have 'Symbol' or 'Ticker' column. Found columns: {list(df.columns)}"
            )

        # Use Symbol column, or Ticker as fallback
        symbol_col = 'Symbol' if 'Symbol' in df.columns else 'Ticker'

        # Map column names
        name_col = None
        for col_name in ['Name', 'Company', 'CompanyName', 'Company Name']:
            if col_name in df.columns:
                name_col = col_name
                break

        sector_col = None
        for col_name in ['Sector', 'Industry', 'GICS Sector']:
            if col_name in df.columns:
                sector_col = col_name
                break

        # Build metadata dictionary
        metadata = {}

        for _, row in df.iterrows():
            ticker = str(row[symbol_col]).strip()

            if not ticker or pd.isna(ticker):
                continue

            metadata[ticker] = {
                'ticker': ticker,
                'company_name': str(row[name_col]).strip() if name_col and not pd.isna(row[name_col]) else ticker,
                'sector': str(row[sector_col]).strip() if sector_col and not pd.isna(row[sector_col]) else 'Unknown'
            }

        return metadata

    except Exception as e:
        raise RuntimeError(f"Error loading S&P 500 metadata from {csv_path}: {e}")


def load_custom_tickers(csv_path: str = "data/custom_tickers.csv") -> List[str]:
    """
    Load custom ticker symbols from CSV file.

    Args:
        csv_path: Path to CSV file with custom tickers (watchlist)

    Returns:
        List of ticker symbols

    Expected CSV format:
        Symbol,Name,Sector
        TSLA,Tesla Inc.,Consumer Discretionary
        NVDA,NVIDIA Corporation,Information Technology
    """
    return load_sp500_tickers(csv_path)  # Reuse same loading logic


def load_all_tickers(
    sp500_path: str = "data/sp500.csv",
    custom_path: str = "data/custom_tickers.csv"
) -> List[str]:
    """
    Load all tickers from both S&P 500 and custom watchlist.

    Args:
        sp500_path: Path to S&P 500 CSV file
        custom_path: Path to custom tickers CSV file

    Returns:
        Combined list of unique ticker symbols (S&P 500 + custom)

    Note:
        Duplicates are automatically removed. If a ticker exists in both files,
        it will only appear once in the output.
    """
    tickers = []

    # Load S&P 500 if file exists
    try:
        sp500_tickers = load_sp500_tickers(sp500_path)
        tickers.extend(sp500_tickers)
    except FileNotFoundError:
        print(f"Warning: S&P 500 file not found: {sp500_path}")
    except Exception as e:
        print(f"Warning: Error loading S&P 500 tickers: {e}")

    # Load custom tickers if file exists
    try:
        custom_tickers = load_custom_tickers(custom_path)
        tickers.extend(custom_tickers)
    except FileNotFoundError:
        print(f"Warning: Custom tickers file not found: {custom_path}")
    except Exception as e:
        print(f"Warning: Error loading custom tickers: {e}")

    # Remove duplicates while preserving order
    seen = set()
    unique_tickers = []
    for ticker in tickers:
        if ticker not in seen:
            seen.add(ticker)
            unique_tickers.append(ticker)

    return unique_tickers


if __name__ == '__main__':
    """Test the loader with a sample CSV."""
    # Test loading S&P 500 tickers
    try:
        print("=" * 60)
        print("S&P 500 TICKERS")
        print("=" * 60)
        tickers = load_sp500_tickers()
        print(f"Loaded {len(tickers)} S&P 500 tickers")
        print(f"First 10 tickers: {tickers[:10]}")

        # Test loading metadata
        metadata = load_sp500_metadata()
        print(f"\nLoaded metadata for {len(metadata)} tickers")

        # Show sample
        sample_ticker = tickers[0]
        print(f"\nSample metadata for {sample_ticker}:")
        print(f"  {metadata[sample_ticker]}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error: {e}")

    # Test loading custom tickers
    try:
        print("\n" + "=" * 60)
        print("CUSTOM TICKERS")
        print("=" * 60)
        custom = load_custom_tickers()
        print(f"Loaded {len(custom)} custom tickers")
        print(f"Custom tickers: {custom}")

    except FileNotFoundError as e:
        print(f"Custom tickers file not found (this is optional): {e}")
    except Exception as e:
        print(f"Error: {e}")

    # Test loading all tickers
    try:
        print("\n" + "=" * 60)
        print("ALL TICKERS (S&P 500 + CUSTOM)")
        print("=" * 60)
        all_tickers = load_all_tickers()
        print(f"Loaded {len(all_tickers)} total tickers")
        print(f"First 10: {all_tickers[:10]}")
        print(f"Last 10: {all_tickers[-10:]}")

    except Exception as e:
        print(f"Error: {e}")
