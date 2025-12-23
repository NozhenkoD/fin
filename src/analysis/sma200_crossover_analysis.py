#!/usr/bin/env python3
"""
SMA200 Crossover Analysis

Analyzes what happens when price crosses above SMA200 from below.
For each crossover, tracks min/max/final prices over the next 7 days.
"""
import numpy as np  # Add this at the top of your file
import os
import sys
import argparse
from datetime import datetime
from typing import List, Dict, Optional

import pandas as pd

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.cache import CacheManager
from src.data.sp500_loader import load_sp500_tickers
from src.indicators.technical import calculate_sma

#
# def detect_sma200_crossovers(df: pd.DataFrame) -> List[int]:
#     """
#     Detect where price crosses above SMA200 from below.
#
#     Args:
#         df: DataFrame with 'Close' and 'SMA200' columns
#
#     Returns:
#         List of indices where crossover occurs
#     """
#     crossovers = []
#
#     # Need at least 201 rows (200 for SMA + 1 for comparison)
#     if len(df) < 201:
#         return crossovers
#
#     for i in range(200, len(df)):
#         prev_open = df['Open'].iloc[i - 1]
#         prev_close = df['Close'].iloc[i - 1]
#         prev_sma200 = df['SMA200'].iloc[i - 1]
#         curr_close = df['Close'].iloc[i]
#         curr_sma200 = df['SMA200'].iloc[i]
#
#         # Check for crossover: was below, now above or at
#         if pd.notna(prev_close) and pd.notna(prev_sma200) and pd.notna(curr_close) and pd.notna(curr_sma200):
#             if prev_close < prev_sma200 and curr_close >= curr_sma200:
#                 crossovers.append(i)
#
#     return crossovers


def analyze_forward_days(df: pd.DataFrame, crossover_idx: int, period: int = 20,
                         stop_loss_pct: float = -5.0, take_profit_pct: float = 10.0) -> Optional[Dict]:
    """
    Analyze what happens in the forward period following a crossover.

    Calculates both stop-loss/take-profit outcome AND min/max/last_day metrics
    for full visibility and verification.

    Args:
        df: DataFrame with OHLCV data
        crossover_idx: Index of the crossover event
        period: Days to look forward (default: 20)
        stop_loss_pct: Stop-loss threshold (default: -5.0)
        take_profit_pct: Take-profit threshold (default: 10.0)

    Returns:
        Dictionary with analysis results, or None if insufficient data
    """
    # Need enough data after crossover
    if crossover_idx + period >= len(df):
        return None

    # Entry data
    entry_date = df.index[crossover_idx]
    entry_price = df['Close'].iloc[crossover_idx]

    # Forward window
    forward_window = df.iloc[crossover_idx + 1: crossover_idx + period + 1]

    # Check if we have full period
    if len(forward_window) < period:
        return None

    # Calculate stop-loss/take-profit outcome
    sl_tp_outcome = calculate_stop_loss_take_profit_outcome(
        df, crossover_idx, period, stop_loss_pct, take_profit_pct
    )

    if sl_tp_outcome is None:
        return None

    # Calculate min/max/last_day metrics (for verification)
    min_price = forward_window['Low'].min()
    max_price = forward_window['High'].max()
    last_day_close = forward_window['Close'].iloc[-1]

    # Percentage changes
    min_pct = ((min_price - entry_price) / entry_price) * 100
    max_pct = ((max_price - entry_price) / entry_price) * 100
    last_day_pct = ((last_day_close - entry_price) / entry_price) * 100

    return {
        'date': entry_date,
        'entry_price': entry_price,
        # Stop-loss/take-profit columns (used for win/loss)
        'exit_type': sl_tp_outcome['exit_type'],
        'exit_day': sl_tp_outcome['exit_day'],
        'exit_price': sl_tp_outcome['exit_price'],
        'exit_pct': sl_tp_outcome['exit_pct'],
        'is_winner': sl_tp_outcome['is_winner'],
        # Min/max/last_day columns (for verification)
        'min_price': min_price,
        'max_price': max_price,
        'last_day_close': last_day_close,
        'min_pct': min_pct,
        'max_pct': max_pct,
        'last_day_pct': last_day_pct
    }


def detect_sma200_crossovers(df, proximity_pct=0.01):
    """
    Detects when price moves from below SMA200 to opening/closing above it.
    Returns integer positions (iloc indices).
    """
    # 1. Price was below SMA200 yesterday (Using capitalized 'Close')
    prev_close_below = df['Close'].shift(1) < df['SMA200'].shift(1)

    # 2. Price opened above SMA200 today
    open_above = df['Open'] > df['SMA200']

    # 3. Price closed above SMA200 today
    close_above = df['Close'] > df['SMA200']

    # 4. "Really Close" - Open is within X% of the SMA200
    # We use (Open / SMA) - 1 to get the percentage distance
    is_close_to_sma = (df['Open'] / df['SMA200'] - 1) <= proximity_pct

    # Combine all conditions
    signals = prev_close_below & open_above & close_above & is_close_to_sma

    # IMPORTANT: Return integer positions so .iloc[crossover_idx] works
    return np.flatnonzero(signals).tolist()


def detect_sustained_sma200_crossovers(df, min_days_below=15, proximity_pct=0.01):
    """
    Detect crossovers where price was below SMA200 for a sustained period.

    Strategy: SMA200 as RESISTANCE that gets broken (resistance becomes support)

    This filters out whipsaw crossovers by requiring price to be below SMA200
    for a minimum number of consecutive days before the crossover.

    Args:
        df: DataFrame with OHLCV and SMA200 columns
        min_days_below: Minimum consecutive days price must be below SMA200 (default: 15)
        proximity_pct: Maximum distance from SMA200 for open (default: 0.01 = 1%)

    Returns:
        List of integer indices where sustained crossover occurs
    """
    # Step 1: Detect basic crossovers
    basic_crossovers = detect_sma200_crossovers(df, proximity_pct)

    # Step 2: Filter for sustained below
    sustained_crossovers = []

    for idx in basic_crossovers:
        # Need enough history to check lookback period
        if idx < min_days_below:
            continue

        # Look back min_days_below days
        lookback_window = df.iloc[idx - min_days_below : idx]

        # Check if Close was below SMA200 for ALL days in lookback
        all_below = (lookback_window['Close'] < lookback_window['SMA200']).all()

        if all_below:
            sustained_crossovers.append(idx)

    return sustained_crossovers


def detect_sma200_support_bounce(df, min_days_above=15, proximity_pct=0.02):
    """
    Detect when price touches SMA200 from above and bounces back up.

    Strategy: SMA200 as SUPPORT (price dips down, touches it, bounces back)

    This detects cases where price was above SMA200, dips down to touch or
    get close to it, then bounces back up. SMA200 acts as a support level.

    Args:
        df: DataFrame with OHLCV and SMA200 columns
        min_days_above: Minimum consecutive days price must be above SMA200 (default: 15)
        proximity_pct: Maximum distance for Low to get to SMA200 (default: 0.02 = 2%)

    Returns:
        List of integer indices where support bounce occurs
    """
    support_bounces = []

    # Need enough history
    if len(df) < min_days_above + 1:
        return support_bounces

    for i in range(min_days_above, len(df)):
        # Look back min_days_above days
        lookback_window = df.iloc[i - min_days_above : i]
        current_row = df.iloc[i]

        # Check if Close was above SMA200 for ALL days in lookback
        all_above = (lookback_window['Close'] > lookback_window['SMA200']).all()

        if not all_above:
            continue

        # Check if Low touched or got close to SMA200 today
        current_low = current_row['Low']
        current_sma = current_row['SMA200']
        current_close = current_row['Close']

        if pd.isna(current_low) or pd.isna(current_sma) or pd.isna(current_close):
            continue

        # Calculate how close Low got to SMA200
        distance_to_sma = ((current_low - current_sma) / current_sma)

        # Low touched or got within proximity_pct of SMA200 (could dip slightly below or stay above)
        # We allow slight penetration below (down to -proximity_pct) or close approach from above
        if -proximity_pct <= distance_to_sma <= proximity_pct:
            # Must close back above SMA200 (bounce back)
            if current_close > current_sma:
                support_bounces.append(i)

    return support_bounces


def calculate_stop_loss_take_profit_outcome(
    df: pd.DataFrame,
    crossover_idx: int,
    period: int = 20,
    stop_loss_pct: float = -5.0,
    take_profit_pct: float = 10.0
) -> Optional[Dict]:
    """
    Determine trade outcome using stop-loss and take-profit logic.

    Iterates day-by-day through the forward period, checking if price hits
    stop-loss (using Low) or take-profit (using High) first. This simulates
    realistic exits rather than just looking at end-of-period price.

    Args:
        df: DataFrame with OHLCV data
        crossover_idx: Index of entry (crossover event)
        period: Days to look forward (default: 20)
        stop_loss_pct: Stop-loss threshold, e.g., -5.0 for -5% (default: -5.0)
        take_profit_pct: Take-profit threshold, e.g., 10.0 for +10% (default: 10.0)

    Returns:
        Dictionary with:
        - exit_type: 'stop_loss', 'take_profit', or 'period_end'
        - exit_day: Day number when exit occurred (1-based)
        - exit_price: Price at exit
        - exit_pct: Percentage gain/loss at exit
        - is_winner: True if take_profit or positive at period end

        Returns None if insufficient forward data
    """
    if crossover_idx + period >= len(df):
        return None

    entry_price = df['Close'].iloc[crossover_idx]
    forward_window = df.iloc[crossover_idx + 1 : crossover_idx + period + 1]

    if len(forward_window) < period:
        return None

    # Check each day for stop-loss or take-profit hit
    for day_num, (idx, row) in enumerate(forward_window.iterrows(), start=1):
        low_pct = ((row['Low'] - entry_price) / entry_price) * 100
        high_pct = ((row['High'] - entry_price) / entry_price) * 100

        # Check stop-loss first (using Low of the day)
        if low_pct <= stop_loss_pct:
            return {
                'exit_type': 'stop_loss',
                'exit_day': day_num,
                'exit_price': entry_price * (1 + stop_loss_pct / 100),
                'exit_pct': stop_loss_pct,
                'is_winner': False
            }

        # Check take-profit (using High of the day)
        if high_pct >= take_profit_pct:
            return {
                'exit_type': 'take_profit',
                'exit_day': day_num,
                'exit_price': entry_price * (1 + take_profit_pct / 100),
                'exit_pct': take_profit_pct,
                'is_winner': True
            }

    # Neither hit - use end-of-period price
    exit_price = forward_window['Close'].iloc[-1]
    exit_pct = ((exit_price - entry_price) / entry_price) * 100

    return {
        'exit_type': 'period_end',
        'exit_day': period,
        'exit_price': exit_price,
        'exit_pct': exit_pct,
        'is_winner': exit_pct > 0
    }

def run_analysis(ticker: str, cache_manager: CacheManager,
                 strategy: str = 'resistance_break',
                 min_days: int = 15, proximity_pct: float = 0.01,
                 period: int = 30, stop_loss_pct: float = -5.0,
                 take_profit_pct: float = 10.0) -> pd.DataFrame:
    """
    Run SMA200 analysis on a ticker with specified strategy.

    Args:
        ticker: Stock ticker symbol
        cache_manager: CacheManager instance
        strategy: 'resistance_break' (price breaks above SMA200) or
                 'support_bounce' (price touches SMA200 from above and bounces)
        min_days: Minimum days price must be above/below SMA200 (default: 15)
        proximity_pct: Maximum distance from SMA200 (default: 0.01 for resistance, 0.02 for support)
        period: Forward analysis period in days (default: 20)
        stop_loss_pct: Stop-loss threshold (default: -5.0)
        take_profit_pct: Take-profit threshold (default: 10.0)

    Returns:
        DataFrame with analysis results
    """
    print(f"\nAnalyzing {ticker}...")
    df = cache_manager.get_ticker_data(ticker)

    if df is None or df.empty:
        return pd.DataFrame()

    # Calculate SMA200
    df['SMA200'] = calculate_sma(df, window=200)

    # Drop rows where SMA200 is NaN
    df = df.dropna(subset=['SMA200'])

    # Detect signals based on strategy
    if strategy == 'resistance_break':
        signal_indices = detect_sustained_sma200_crossovers(
            df, min_days_below=min_days, proximity_pct=proximity_pct
        )
        print(f"  Found {len(signal_indices)} resistance breaks (min {min_days} days below)")
    elif strategy == 'support_bounce':
        signal_indices = detect_sma200_support_bounce(
            df, min_days_above=min_days, proximity_pct=proximity_pct
        )
        print(f"  Found {len(signal_indices)} support bounces (min {min_days} days above)")
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'resistance_break' or 'support_bounce'")

    if not signal_indices:
        return pd.DataFrame()

    results = []
    for idx in signal_indices:
        analysis = analyze_forward_days(
            df, idx, period=period,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct
        )
        if analysis:
            analysis['ticker'] = ticker
            analysis['strategy'] = strategy
            results.append(analysis)

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    # Column order: SL/TP columns first, then verification columns
    column_order = [
        'date', 'ticker', 'strategy', 'entry_price',
        'exit_type', 'exit_day', 'exit_price', 'exit_pct', 'is_winner',
        'min_price', 'max_price', 'last_day_close',
        'min_pct', 'max_pct', 'last_day_pct'
    ]
    return results_df[column_order]


def print_summary(results_df: pd.DataFrame, ticker: str,
                  strategy: str = 'resistance_break',
                  min_days: int = 15, stop_loss_pct: float = -5.0,
                  take_profit_pct: float = 10.0, period: int = 20):
    """
    Print summary statistics for SMA200 analysis.

    Args:
        results_df: DataFrame with analysis results
        ticker: Stock ticker symbol
        strategy: Strategy used ('resistance_break' or 'support_bounce')
        min_days: Minimum days above/below SMA200 filter
        stop_loss_pct: Stop-loss threshold
        take_profit_pct: Take-profit threshold
        period: Forward analysis period
    """
    if results_df.empty:
        print("\nNo results to summarize.")
        return

    print("\n" + "=" * 60)
    print(f"SMA200 ANALYSIS - {ticker}")
    print("=" * 60)

    # Period
    start_date = results_df['date'].min().date()
    end_date = results_df['date'].max().date()
    print(f"Period: {start_date} to {end_date}")

    # Strategy info
    strategy_name = "Resistance Break" if strategy == 'resistance_break' else "Support Bounce"
    strategy_desc = "Price breaks above SMA200" if strategy == 'resistance_break' else "Price bounces off SMA200"
    print(f"\nSTRATEGY: {strategy_name}")
    print(f"  {strategy_desc}")

    # Filter settings
    print(f"\nFILTER SETTINGS")
    if strategy == 'resistance_break':
        print(f"  Min days below SMA200: {min_days}")
    else:
        print(f"  Min days above SMA200: {min_days}")
    print(f"  Stop-loss:             {stop_loss_pct:+.1f}%")
    print(f"  Take-profit:           {take_profit_pct:+.1f}%")
    print(f"  Analysis period:        {period} days")

    # Crossover events
    print(f"\nCROSSOVER EVENTS")
    print(f"  Total crossovers found: {len(results_df)}")

    # Stop-loss/take-profit breakdown
    sl_hits = results_df[results_df['exit_type'] == 'stop_loss']
    tp_hits = results_df[results_df['exit_type'] == 'take_profit']
    period_end = results_df[results_df['exit_type'] == 'period_end']

    print(f"\nSTOP-LOSS / TAKE-PROFIT ANALYSIS")
    print(f"  Stop-loss hits:    {len(sl_hits):3d}  ({len(sl_hits)/len(results_df)*100:.1f}%)")
    print(f"  Take-profit hits:  {len(tp_hits):3d}  ({len(tp_hits)/len(results_df)*100:.1f}%)")
    print(f"  Period end:        {len(period_end):3d}  ({len(period_end)/len(results_df)*100:.1f}%)")

    # Win/loss metrics using is_winner column
    winners = results_df[results_df['is_winner'] == True]
    losers = results_df[results_df['is_winner'] == False]

    print(f"\nWIN/LOSS METRICS (SL/TP Based)")
    print(f"  Winners:  {len(winners):3d}  ({len(winners)/len(results_df)*100:.1f}%)")
    print(f"  Losers:   {len(losers):3d}  ({len(losers)/len(results_df)*100:.1f}%)")
    print(f"  Average exit day:  {results_df['exit_day'].mean():.1f} days")
    print(f"  Average exit %:    {results_df['exit_pct'].mean():+.1f}%")

    # Forward analysis (for reference)
    print(f"\n{period}-DAY FORWARD ANALYSIS (for reference)")
    print(f"  Average Min %:    {results_df['min_pct'].mean():+.1f}%")
    print(f"  Average Max %:    {results_df['max_pct'].mean():+.1f}%")
    print(f"  Average Last Day %: {results_df['last_day_pct'].mean():+.1f}%")

    # Distribution by exit percentage
    print(f"\nDISTRIBUTION (Exit %)")
    bins = [
        ('< -10%', results_df[results_df['exit_pct'] < -10]),
        ('-10% to 0%', results_df[(results_df['exit_pct'] >= -10) & (results_df['exit_pct'] <= 0)]),
        ('0% to +10%', results_df[(results_df['exit_pct'] > 0) & (results_df['exit_pct'] <= 10)]),
        ('+10% to +20%', results_df[(results_df['exit_pct'] > 10) & (results_df['exit_pct'] <= 20)]),
        ('> +20%', results_df[results_df['exit_pct'] > 20])
    ]

    for label, subset in bins:
        count = len(subset)
        pct = (count / len(results_df)) * 100
        print(f"  {label:14s} {count:3d}  ({pct:.1f}%)")

    # Dates of crossovers
    print(f"\nDATES OF CROSSOVERS")
    dates_str = ", ".join([d.strftime('%Y-%m-%d') for d in results_df['date'].head(10)])
    if len(results_df) > 10:
        dates_str += f", ... (+{len(results_df) - 10} more)"
    print(f"  {dates_str}")

    print("=" * 60 + "\n")


def print_aggregate_summary(combined_df: pd.DataFrame, strategy: str = 'resistance_break'):
    """
    Print aggregate summary across all tickers.

    Args:
        combined_df: Combined DataFrame with results from all tickers
        strategy: Strategy used
    """
    if combined_df.empty:
        print("\nNo aggregate results to summarize.")
        return

    print("\n" + "=" * 80)
    print("AGGREGATE SUMMARY - ALL TICKERS")
    print("=" * 80)

    strategy_name = "Resistance Break" if strategy == 'resistance_break' else "Support Bounce"
    print(f"Strategy: {strategy_name}")
    print(f"Total tickers analyzed: {combined_df['ticker'].nunique()}")
    print(f"Total signals: {len(combined_df)}")

    # Overall win/loss
    winners = combined_df[combined_df['is_winner'] == True]
    losers = combined_df[combined_df['is_winner'] == False]

    print(f"\nOVERALL WIN/LOSS")
    print(f"  Winners:  {len(winners):4d}  ({len(winners)/len(combined_df)*100:.1f}%)")
    print(f"  Losers:   {len(losers):4d}  ({len(losers)/len(combined_df)*100:.1f}%)")
    print(f"  Average exit day:  {combined_df['exit_day'].mean():.1f} days")
    print(f"  Average exit %:    {combined_df['exit_pct'].mean():+.1f}%")

    # Exit type breakdown
    sl_hits = combined_df[combined_df['exit_type'] == 'stop_loss']
    tp_hits = combined_df[combined_df['exit_type'] == 'take_profit']
    period_end = combined_df[combined_df['exit_type'] == 'period_end']

    print(f"\nEXIT TYPE BREAKDOWN")
    print(f"  Stop-loss hits:    {len(sl_hits):4d}  ({len(sl_hits)/len(combined_df)*100:.1f}%)")
    print(f"  Take-profit hits:  {len(tp_hits):4d}  ({len(tp_hits)/len(combined_df)*100:.1f}%)")
    print(f"  Period end:        {len(period_end):4d}  ({len(period_end)/len(combined_df)*100:.1f}%)")

    # Per-ticker summary
    print(f"\nPER-TICKER SUMMARY (sorted by win rate)")
    print("=" * 80)

    ticker_stats = []
    for ticker in combined_df['ticker'].unique():
        ticker_df = combined_df[combined_df['ticker'] == ticker]
        ticker_winners = ticker_df[ticker_df['is_winner'] == True]

        stats = {
            'ticker': ticker,
            'signals': len(ticker_df),
            'wins': len(ticker_winners),
            'losses': len(ticker_df) - len(ticker_winners),
            'win_rate': (len(ticker_winners) / len(ticker_df) * 100) if len(ticker_df) > 0 else 0,
            'avg_exit_pct': ticker_df['exit_pct'].mean(),
            'avg_exit_day': ticker_df['exit_day'].mean()
        }
        ticker_stats.append(stats)

    # Sort by win rate descending
    ticker_stats_df = pd.DataFrame(ticker_stats).sort_values('win_rate', ascending=False)

    # Format for display
    display_ticker_stats = ticker_stats_df.copy()
    display_ticker_stats['win_rate'] = display_ticker_stats['win_rate'].apply(lambda x: f"{x:.1f}%")
    display_ticker_stats['avg_exit_pct'] = display_ticker_stats['avg_exit_pct'].apply(lambda x: f"{x:+.1f}%")
    display_ticker_stats['avg_exit_day'] = display_ticker_stats['avg_exit_day'].apply(lambda x: f"{x:.1f}")

    # Rename columns for display
    display_ticker_stats.columns = ['Ticker', 'Signals', 'Wins', 'Losses', 'Win Rate', 'Avg Exit %', 'Avg Exit Day']

    print(display_ticker_stats.to_string(index=False))
    print("=" * 80 + "\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze SMA200 strategies with stop-loss/take-profit logic',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Resistance break strategy (price breaks above SMA200 from below)
  python -m src.analysis.sma200_crossover_analysis --ticker F --strategy resistance_break

  # Support bounce strategy (price touches SMA200 from above and bounces)
  python -m src.analysis.sma200_crossover_analysis --ticker F --strategy support_bounce

  # Run on all S&P 500 stocks
  python -m src.analysis.sma200_crossover_analysis --sp500 --strategy support_bounce --min-days 0 --stop-loss -10 --take-profit 10

  # Custom stop-loss and take-profit
  python -m src.analysis.sma200_crossover_analysis --ticker F --stop-loss -3.0 --take-profit 15.0

  # Multiple tickers
  python -m src.analysis.sma200_crossover_analysis --tickers F AAPL MSFT --strategy support_bounce

  # Export results to CSV
  python -m src.analysis.sma200_crossover_analysis --sp500 --strategy support_bounce --export results/sp500_support_bounce.csv
        """
    )

    parser.add_argument(
        '--ticker',
        default='F',
        help='Stock ticker to analyze (default: F - Ford)'
    )

    parser.add_argument(
        '--tickers',
        nargs='+',
        help='Multiple tickers to analyze (space-separated)'
    )

    parser.add_argument(
        '--sp500',
        action='store_true',
        help='Analyze all S&P 500 stocks'
    )

    parser.add_argument(
        '--strategy',
        choices=['resistance_break', 'support_bounce'],
        default='resistance_break',
        help='Strategy: resistance_break (break above SMA200) or support_bounce (bounce off SMA200) (default: resistance_break)'
    )

    parser.add_argument(
        '--min-days',
        type=int,
        default=15,
        help='Minimum consecutive days price must be above/below SMA200 (default: 15)'
    )

    parser.add_argument(
        '--stop-loss',
        type=float,
        default=-5.0,
        help='Stop-loss threshold in percent, e.g., -5.0 for -5%% (default: -5.0)'
    )

    parser.add_argument(
        '--take-profit',
        type=float,
        default=10.0,
        help='Take-profit threshold in percent, e.g., 10.0 for +10%% (default: 10.0)'
    )

    parser.add_argument(
        '--period',
        type=int,
        default=20,
        help='Forward analysis period in days (default: 20)'
    )

    parser.add_argument(
        '--export',
        type=str,
        help='Export results to CSV file'
    )

    parser.add_argument(
        '--cache-dir',
        default='data/cache/ohlcv',
        help='Cache directory (default: data/cache/ohlcv)'
    )

    args = parser.parse_args()

    # Initialize cache manager
    cache_manager = CacheManager(cache_dir=args.cache_dir)

    # Determine tickers to analyze
    if args.sp500:
        print("Loading S&P 500 tickers...")
        tickers = load_sp500_tickers()
        print(f"Loaded {len(tickers)} S&P 500 tickers")
        show_individual_summaries = False  # Don't show individual summaries for S&P 500
    elif args.tickers:
        tickers = args.tickers
        show_individual_summaries = True
    else:
        tickers = [args.ticker]
        show_individual_summaries = True

    # Run analysis for each ticker
    all_results = []
    failed_tickers = []

    for i, ticker in enumerate(tickers, 1):
        if args.sp500:
            print(f"\n[{i}/{len(tickers)}] {ticker}...", end=' ')

        try:
            results_df = run_analysis(
                ticker, cache_manager,
                strategy=args.strategy,
                min_days=args.min_days,
                period=args.period,
                stop_loss_pct=args.stop_loss,
                take_profit_pct=args.take_profit
            )

            if not results_df.empty:
                all_results.append(results_df)
                if args.sp500:
                    print(f"✓ {len(results_df)} signals")
                elif show_individual_summaries:
                    print_summary(
                        results_df, ticker,
                        strategy=args.strategy,
                        min_days=args.min_days,
                        stop_loss_pct=args.stop_loss,
                        take_profit_pct=args.take_profit,
                        period=args.period
                    )
            else:
                if args.sp500:
                    print("✗ No signals")
        except Exception as e:
            failed_tickers.append(ticker)
            if args.sp500:
                print(f"✗ Error: {str(e)[:50]}")
            else:
                print(f"Error analyzing {ticker}: {e}")

    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)

        # Show aggregate summary for multiple tickers
        if len(tickers) > 1:
            print_aggregate_summary(combined_df, strategy=args.strategy)

        # Display detailed results table (limit for S&P 500)
        if not args.sp500 or len(combined_df) <= 100:
            print("\nDETAILED RESULTS")
            print("=" * 60)

            # Format for display
            display_df = combined_df.copy()
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
            display_df['entry_price'] = display_df['entry_price'].apply(lambda x: f"${x:.2f}")
            display_df['exit_price'] = display_df['exit_price'].apply(lambda x: f"${x:.2f}")
            display_df['exit_pct'] = display_df['exit_pct'].apply(lambda x: f"{x:+.1f}%")
            display_df['min_price'] = display_df['min_price'].apply(lambda x: f"${x:.2f}")
            display_df['max_price'] = display_df['max_price'].apply(lambda x: f"${x:.2f}")
            display_df['last_day_close'] = display_df['last_day_close'].apply(lambda x: f"${x:.2f}")
            display_df['min_pct'] = display_df['min_pct'].apply(lambda x: f"{x:+.1f}%")
            display_df['max_pct'] = display_df['max_pct'].apply(lambda x: f"{x:+.1f}%")
            display_df['last_day_pct'] = display_df['last_day_pct'].apply(lambda x: f"{x:+.1f}%")

            print(display_df.to_string(index=False))
            print()
        elif args.sp500:
            print(f"\nDetailed results table skipped ({len(combined_df)} total signals too large to display)")
            print("Export to CSV to view all results.")

        # Export if requested
        if args.export:
            # Create directory if it doesn't exist
            export_dir = os.path.dirname(args.export)
            if export_dir and not os.path.exists(export_dir):
                os.makedirs(export_dir)

            combined_df.to_csv(args.export, index=False)
            print(f"\nResults exported to: {args.export}")

        # Show failed tickers summary
        if failed_tickers:
            print(f"\nFailed tickers ({len(failed_tickers)}): {', '.join(failed_tickers[:20])}")
            if len(failed_tickers) > 20:
                print(f"  ... and {len(failed_tickers) - 20} more")

    else:
        print("\nNo results to display.")


if __name__ == '__main__':
    main()
