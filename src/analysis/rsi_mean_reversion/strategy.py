#!/usr/bin/env python3
"""
RSI Mean Reversion Strategy Analysis

Strategy: Buy oversold stocks (RSI < 30) that are in long-term uptrends (above SMA200).
Exit when RSI recovers to overbought (RSI > 70) or hit SL/TP targets.

This is a mean reversion strategy that:
1. Uses SMA200 as a trend filter (only trade stocks in uptrends)
2. Buys when RSI indicates oversold conditions (temporary weakness)
3. Exits when RSI recovers or price targets are hit

Expected characteristics:
- Higher win rate than trend-following (~60-70%)
- Shorter hold times (5-15 days average)
- Lower profit per trade but more frequent signals
"""
import os
import sys
import argparse
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.cache import CacheManager
from src.data.sp500_loader import load_sp500_tickers
from src.indicators.technical import calculate_sma, calculate_rsi
from src.analysis.summary import print_summary, print_aggregate_summary


def detect_rsi_oversold_signals(df: pd.DataFrame, rsi_threshold: int = 30,
                                 sma_filter: int = 200, min_days_above: int = 15) -> List[int]:
    """
    Detect RSI oversold signals with SMA200 trend filter.

    Strategy Logic:
    1. Price must be above SMA200 (uptrend filter)
    2. Price was above SMA200 for min_days_above consecutive days (sustained uptrend)
    3. RSI crosses below threshold (new oversold condition)

    Args:
        df: DataFrame with OHLCV, SMA200, and RSI columns
        rsi_threshold: RSI threshold for oversold (default: 30)
        sma_filter: SMA period for trend filter (default: 200)
        min_days_above: Minimum days above SMA200 before signal (default: 15)

    Returns:
        List of integer indices where signals occur
    """
    if len(df) < sma_filter + min_days_above:
        return []

    signals = []

    for i in range(min_days_above, len(df)):
        # Check if we have enough history
        if i < 1:  # Need at least 1 previous day for RSI crossover check
            continue

        # Get current and previous values
        curr_rsi = df['RSI'].iloc[i]
        prev_rsi = df['RSI'].iloc[i - 1]
        curr_close = df['Close'].iloc[i]
        curr_sma = df['SMA200'].iloc[i]

        # Skip if any values are NaN
        if pd.isna(curr_rsi) or pd.isna(prev_rsi) or pd.isna(curr_close) or pd.isna(curr_sma):
            continue

        # Condition 1: Price above SMA200 (uptrend)
        if curr_close <= curr_sma:
            continue

        # Condition 2: Check sustained uptrend (min_days_above days above SMA200)
        lookback_window = df.iloc[i - min_days_above : i]
        all_above = (lookback_window['Close'] > lookback_window['SMA200']).all()

        if not all_above:
            continue

        # Condition 3: RSI crosses below threshold (new oversold)
        # Previous RSI >= threshold AND current RSI < threshold
        if prev_rsi >= rsi_threshold and curr_rsi < rsi_threshold:
            signals.append(i)

    return signals


def calculate_stop_loss_take_profit_outcome(
    df: pd.DataFrame,
    entry_idx: int,
    period: int = 10,
    stop_loss_pct: float = -3.0,
    take_profit_pct: float = 5.0,
    rsi_exit_threshold: int = 70
) -> Optional[Dict]:
    """
    Determine trade outcome using stop-loss, take-profit, and RSI exit logic.

    Exit conditions (in order of priority):
    1. Stop-loss hit (Low <= entry_price * (1 + stop_loss_pct/100))
    2. Take-profit hit (High >= entry_price * (1 + take_profit_pct/100))
    3. RSI exit (RSI >= rsi_exit_threshold, exit at next day's Open)
    4. Period end (hold for max period, exit at Close)

    When both SL and TP hit same day, uses bar structure heuristic.

    Args:
        df: DataFrame with OHLCV and RSI data
        entry_idx: Index of entry signal
        period: Days to look forward (default: 10)
        stop_loss_pct: Stop-loss threshold (default: -3.0)
        take_profit_pct: Take-profit threshold (default: 5.0)
        rsi_exit_threshold: RSI level to trigger exit (default: 70)

    Returns:
        Dictionary with exit details or None if insufficient data
    """
    if entry_idx + period >= len(df):
        return None

    entry_price = df['Close'].iloc[entry_idx]
    forward_window = df.iloc[entry_idx + 1 : entry_idx + period + 1]

    if len(forward_window) < period:
        return None

    # Check each day for exit conditions
    for day_num, (idx, row) in enumerate(forward_window.iterrows(), start=1):
        low_pct = ((row['Low'] - entry_price) / entry_price) * 100
        high_pct = ((row['High'] - entry_price) / entry_price) * 100

        sl_hit = low_pct <= stop_loss_pct
        tp_hit = high_pct >= take_profit_pct

        # Priority 1: Stop-loss only
        if sl_hit and not tp_hit:
            return {
                'exit_type': 'stop_loss',
                'exit_day': day_num,
                'exit_price': entry_price * (1 + stop_loss_pct / 100),
                'exit_pct': stop_loss_pct,
                'is_winner': False
            }

        # Priority 2: Take-profit only
        if tp_hit and not sl_hit:
            return {
                'exit_type': 'take_profit',
                'exit_day': day_num,
                'exit_price': entry_price * (1 + take_profit_pct / 100),
                'exit_pct': take_profit_pct,
                'is_winner': True
            }

        # Priority 3: Both hit - use bar structure heuristic
        if sl_hit and tp_hit:
            open_price = row['Open']
            distance_to_low = abs(open_price - row['Low'])
            distance_to_high = abs(row['High'] - open_price)

            if distance_to_low < distance_to_high:
                return {
                    'exit_type': 'stop_loss',
                    'exit_day': day_num,
                    'exit_price': entry_price * (1 + stop_loss_pct / 100),
                    'exit_pct': stop_loss_pct,
                    'is_winner': False
                }
            else:
                return {
                    'exit_type': 'take_profit',
                    'exit_day': day_num,
                    'exit_price': entry_price * (1 + take_profit_pct / 100),
                    'exit_pct': take_profit_pct,
                    'is_winner': True
                }

        # Priority 4: RSI exit (check if RSI >= threshold)
        curr_rsi = row['RSI']
        if not pd.isna(curr_rsi) and curr_rsi >= rsi_exit_threshold:
            # Exit at Close of the day RSI hits threshold
            exit_price = row['Close']
            exit_pct = ((exit_price - entry_price) / entry_price) * 100

            return {
                'exit_type': 'rsi_exit',
                'exit_day': day_num,
                'exit_price': exit_price,
                'exit_pct': exit_pct,
                'is_winner': exit_pct > 0
            }

    # Priority 5: Period end
    exit_price = forward_window['Close'].iloc[-1]
    exit_pct = ((exit_price - entry_price) / entry_price) * 100

    return {
        'exit_type': 'period_end',
        'exit_day': period,
        'exit_price': exit_price,
        'exit_pct': exit_pct,
        'is_winner': exit_pct > 0
    }


def analyze_forward_days(df: pd.DataFrame, entry_idx: int, period: int = 10,
                         stop_loss_pct: float = -3.0, take_profit_pct: float = 5.0,
                         rsi_exit_threshold: int = 70) -> Optional[Dict]:
    """
    Analyze what happens in the forward period following an RSI oversold signal.

    Args:
        df: DataFrame with OHLCV and RSI data
        entry_idx: Index of the entry signal
        period: Days to look forward (default: 10)
        stop_loss_pct: Stop-loss threshold (default: -3.0)
        take_profit_pct: Take-profit threshold (default: 5.0)
        rsi_exit_threshold: RSI level to trigger exit (default: 70)

    Returns:
        Dictionary with analysis results, or None if insufficient data
    """
    if entry_idx + period >= len(df):
        return None

    # Entry data
    entry_date = df.index[entry_idx]
    entry_price = df['Close'].iloc[entry_idx]
    entry_rsi = df['RSI'].iloc[entry_idx]

    # Calculate SL/TP outcome
    sl_tp_outcome = calculate_stop_loss_take_profit_outcome(
        df, entry_idx, period, stop_loss_pct, take_profit_pct, rsi_exit_threshold
    )

    if sl_tp_outcome is None:
        return None

    # Forward window for verification metrics
    forward_window = df.iloc[entry_idx + 1: entry_idx + period + 1]

    if len(forward_window) < period:
        return None

    # Calculate min/max/last_day metrics (for verification)
    min_price = forward_window['Low'].min()
    max_price = forward_window['High'].max()
    last_day_close = forward_window['Close'].iloc[-1]

    min_pct = ((min_price - entry_price) / entry_price) * 100
    max_pct = ((max_price - entry_price) / entry_price) * 100
    last_day_pct = ((last_day_close - entry_price) / entry_price) * 100

    return {
        'date': entry_date,
        'entry_price': entry_price,
        'entry_rsi': entry_rsi,
        # Exit columns
        'exit_type': sl_tp_outcome['exit_type'],
        'exit_day': sl_tp_outcome['exit_day'],
        'exit_price': sl_tp_outcome['exit_price'],
        'exit_pct': sl_tp_outcome['exit_pct'],
        'is_winner': sl_tp_outcome['is_winner'],
        # Verification columns
        'min_price': min_price,
        'max_price': max_price,
        'last_day_close': last_day_close,
        'min_pct': min_pct,
        'max_pct': max_pct,
        'last_day_pct': last_day_pct
    }


def run_analysis(ticker: str, cache_manager: CacheManager,
                 rsi_threshold: int = 30, rsi_exit: int = 70,
                 min_days_above: int = 15, period: int = 10,
                 stop_loss_pct: float = -3.0, take_profit_pct: float = 5.0) -> pd.DataFrame:
    """
    Run RSI mean reversion analysis on a ticker.

    Args:
        ticker: Stock ticker symbol
        cache_manager: CacheManager instance
        rsi_threshold: RSI oversold threshold (default: 30)
        rsi_exit: RSI exit threshold (default: 70)
        min_days_above: Min days above SMA200 before signal (default: 15)
        period: Forward analysis period in days (default: 10)
        stop_loss_pct: Stop-loss threshold (default: -3.0)
        take_profit_pct: Take-profit threshold (default: 5.0)

    Returns:
        DataFrame with analysis results
    """
    print(f"\nAnalyzing {ticker}...")
    df = cache_manager.get_ticker_data(ticker)

    if df is None or df.empty:
        return pd.DataFrame()

    # Calculate indicators
    df['SMA200'] = calculate_sma(df, window=200)
    df['RSI'] = calculate_rsi(df, window=14)

    # Drop rows where indicators are NaN
    df = df.dropna(subset=['SMA200', 'RSI'])

    # Detect signals
    signal_indices = detect_rsi_oversold_signals(
        df, rsi_threshold=rsi_threshold,
        sma_filter=200,
        min_days_above=min_days_above
    )

    print(f"  Found {len(signal_indices)} RSI oversold signals (RSI < {rsi_threshold}, above SMA200)")

    if not signal_indices:
        return pd.DataFrame()

    # Analyze each signal
    results = []
    for idx in signal_indices:
        analysis = analyze_forward_days(
            df, idx, period=period,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            rsi_exit_threshold=rsi_exit
        )
        if analysis:
            analysis['ticker'] = ticker
            analysis['strategy'] = 'rsi_mean_reversion'
            results.append(analysis)

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    # Column order
    column_order = [
        'date', 'ticker', 'strategy', 'entry_price', 'entry_rsi',
        'exit_type', 'exit_day', 'exit_price', 'exit_pct', 'is_winner',
        'min_price', 'max_price', 'last_day_close',
        'min_pct', 'max_pct', 'last_day_pct'
    ]
    return results_df[column_order]


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='RSI Mean Reversion Strategy Backtest',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test on single ticker
  python -m src.analysis.rsi_mean_reversion --ticker AAPL

  # Test on S&P 500
  python -m src.analysis.rsi_mean_reversion --sp500 --export results/rsi_mean_reversion.csv

  # Adjust RSI thresholds
  python -m src.analysis.rsi_mean_reversion --ticker AAPL --rsi-threshold 25 --rsi-exit 75

  # Adjust SL/TP
  python -m src.analysis.rsi_mean_reversion --ticker AAPL --stop-loss -4.0 --take-profit 6.0
        """
    )

    parser.add_argument('--ticker', default='AAPL', help='Stock ticker (default: AAPL)')
    parser.add_argument('--tickers', nargs='+', help='Multiple tickers')
    parser.add_argument('--sp500', action='store_true', help='Analyze all S&P 500 stocks')
    parser.add_argument('--rsi-threshold', type=int, default=30, help='RSI oversold threshold (default: 30)')
    parser.add_argument('--rsi-exit', type=int, default=70, help='RSI exit threshold (default: 70)')
    parser.add_argument('--min-days-above', type=int, default=15, help='Min days above SMA200 (default: 15)')
    parser.add_argument('--stop-loss', type=float, default=-3.0, help='Stop-loss %% (default: -3.0)')
    parser.add_argument('--take-profit', type=float, default=5.0, help='Take-profit %% (default: 5.0)')
    parser.add_argument('--period', type=int, default=10, help='Max hold period in days (default: 10)')
    parser.add_argument('--export', type=str, help='Export results to CSV file')
    parser.add_argument('--cache-dir', default='data/cache/ohlcv', help='Cache directory')
    parser.add_argument('--show-signal-details', action='store_true', help='Show detailed signal table with entry/exit dates')
    parser.add_argument('--max-signals', type=int, default=20, help='Max signals to show in details table (default: 20)')

    args = parser.parse_args()

    # Initialize cache
    cache_manager = CacheManager(cache_dir=args.cache_dir)

    # Determine tickers
    if args.sp500:
        print("Loading S&P 500 tickers...")
        tickers = load_sp500_tickers()
        print(f"Loaded {len(tickers)} S&P 500 tickers")
        show_individual = False
    elif args.tickers:
        tickers = args.tickers
        show_individual = True
    else:
        tickers = [args.ticker]
        show_individual = True

    # Run analysis
    all_results = []
    failed_tickers = []

    for i, ticker in enumerate(tickers, 1):
        if args.sp500:
            print(f"\n[{i}/{len(tickers)}] {ticker}...", end=' ')

        try:
            results_df = run_analysis(
                ticker, cache_manager,
                rsi_threshold=args.rsi_threshold,
                rsi_exit=args.rsi_exit,
                min_days_above=args.min_days_above,
                period=args.period,
                stop_loss_pct=args.stop_loss,
                take_profit_pct=args.take_profit
            )

            if not results_df.empty:
                all_results.append(results_df)
                if args.sp500:
                    print(f"✓ {len(results_df)} signals")
                elif show_individual:
                    print_summary(
                        results_df,
                        strategy_name="RSI Mean Reversion",
                        ticker=ticker,
                        strategy_description=f"Buy when RSI < {args.rsi_threshold} + above SMA200, exit RSI >= {args.rsi_exit} or SL/TP",
                        show_signal_dates=not args.show_signal_details,
                        show_signal_details=args.show_signal_details,
                        max_signals_display=args.max_signals,
                        settings={
                            'RSI threshold': args.rsi_threshold,
                            'RSI exit': args.rsi_exit,
                            'Stop-loss': f"{args.stop_loss:+.1f}%",
                            'Take-profit': f"{args.take_profit:+.1f}%",
                            'Max hold': f"{args.period} days"
                        }
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

    # Combine and display results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)

        if len(tickers) > 1:
            print_aggregate_summary(combined_df, strategy_name="RSI Mean Reversion")

        # Export if requested
        if args.export:
            export_dir = os.path.dirname(args.export)
            if export_dir and not os.path.exists(export_dir):
                os.makedirs(export_dir)

            combined_df.to_csv(args.export, index=False)
            print(f"\nResults exported to: {args.export}")

        if failed_tickers:
            print(f"\nFailed tickers ({len(failed_tickers)}): {', '.join(failed_tickers[:20])}")
    else:
        print("\nNo results to display.")


if __name__ == '__main__':
    main()
