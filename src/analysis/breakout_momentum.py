#!/usr/bin/env python3
"""
Breakout + Momentum Strategy

A trend-following strategy that buys strength instead of weakness.
Designed for trending markets where mean reversion strategies fail.

Strategy Philosophy:
- "Buy high, sell higher" - ride momentum
- Breakouts signal institutional buying
- Volume confirms the move is real
- Momentum (RSI > 60) filters out false breakouts

Entry Conditions (ALL must be met):
1. Price breaks above 20-day high (new breakout)
2. Volume > 2x average (strong buying pressure)
3. Price > SMA200 (long-term uptrend filter)
4. RSI > 60 (momentum confirmation)
5. Close > Open (bullish day)

Exit Conditions:
- Trailing stop: -8% from highest high since entry
- Take-profit: +15%
- Max hold: 30 days

Expected Performance:
- Win rate: 52-58% (lower than mean reversion, but bigger wins)
- Average winner: +12-18%
- Average loser: -6-8%
- Risk/Reward: 2:1 to 3:1
- Hold time: 8-15 days

This strategy works best in:
- Bull markets
- Trending stocks
- Growth sectors
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


def detect_breakout_signals(df: pd.DataFrame,
                            breakout_period: int = 20,
                            volume_multiplier: float = 2.0,
                            rsi_threshold: int = 60,
                            min_days_above_sma200: int = 20) -> List[int]:
    """
    Detect breakout + momentum signals.

    All conditions must be met:
    1. Price breaks above N-day high
    2. Volume > average * multiplier
    3. Price > SMA200 (uptrend)
    4. RSI > threshold (momentum)
    5. Bullish candle (Close > Open)

    Args:
        df: DataFrame with OHLCV, SMA200, RSI, and AvgVolume columns
        breakout_period: Days for high lookback (default: 20)
        volume_multiplier: Volume threshold (default: 2.0)
        rsi_threshold: RSI momentum threshold (default: 60)
        min_days_above_sma200: Min days above SMA200 (default: 20)

    Returns:
        List of integer indices where signals occur
    """
    if len(df) < 200 + min_days_above_sma200:
        return []

    signals = []

    # Calculate rolling high
    df['RollingHigh'] = df['High'].rolling(window=breakout_period).max()

    for i in range(min_days_above_sma200 + breakout_period, len(df)):
        if i < breakout_period + 1:
            continue

        # Get current values
        curr_high = df['High'].iloc[i]
        curr_close = df['Close'].iloc[i]
        curr_open = df['Open'].iloc[i]
        curr_sma200 = df['SMA200'].iloc[i]
        curr_rsi = df['RSI'].iloc[i]
        curr_volume = df['Volume'].iloc[i]
        avg_volume = df['AvgVolume'].iloc[i]

        # Previous period's highest high (excluding today)
        prev_period_high = df['High'].iloc[i - breakout_period : i].max()

        # Skip if any values are NaN
        if pd.isna(curr_high) or pd.isna(curr_close) or pd.isna(curr_open) or \
           pd.isna(curr_sma200) or pd.isna(curr_rsi) or pd.isna(curr_volume) or \
           pd.isna(avg_volume) or pd.isna(prev_period_high):
            continue

        # Filter 1: Breakout - current high breaks above previous period high
        is_breakout = curr_high > prev_period_high

        if not is_breakout:
            continue

        # Filter 2: Volume confirmation
        volume_ok = curr_volume >= avg_volume * volume_multiplier

        if not volume_ok:
            continue

        # Filter 3: Price above SMA200 (long-term uptrend)
        if curr_close <= curr_sma200:
            continue

        # Filter 3b: Sustained uptrend check
        lookback_window = df.iloc[i - min_days_above_sma200 : i]
        all_above = (lookback_window['Close'] > lookback_window['SMA200']).all()

        if not all_above:
            continue

        # Filter 4: RSI shows momentum (> 60)
        if curr_rsi <= rsi_threshold:
            continue

        # Filter 5: Bullish candle (close > open)
        if curr_close <= curr_open:
            continue

        signals.append(i)

    return signals


def calculate_trailing_stop_outcome(
    df: pd.DataFrame,
    entry_idx: int,
    period: int = 30,
    trailing_stop_pct: float = -8.0,
    take_profit_pct: float = 15.0
) -> Optional[Dict]:
    """
    Determine trade outcome using trailing stop and take-profit.

    Trailing stop moves up as price makes new highs, locking in profits.
    This is how momentum traders protect gains while letting winners run.

    Args:
        df: DataFrame with OHLCV data
        entry_idx: Index of entry signal
        period: Days to look forward (default: 30)
        trailing_stop_pct: Trailing stop from highest high (default: -8.0)
        take_profit_pct: Fixed take-profit (default: 15.0)

    Returns:
        Dictionary with exit details or None
    """
    if entry_idx + period >= len(df):
        return None

    entry_price = df['Close'].iloc[entry_idx]
    forward_window = df.iloc[entry_idx + 1 : entry_idx + period + 1]

    if len(forward_window) < period:
        return None

    highest_high = entry_price  # Track highest price since entry

    # Check each day
    for day_num, (idx, row) in enumerate(forward_window.iterrows(), start=1):
        # Update highest high
        if row['High'] > highest_high:
            highest_high = row['High']

        # Calculate percentages
        low_pct = ((row['Low'] - entry_price) / entry_price) * 100
        high_pct = ((row['High'] - entry_price) / entry_price) * 100

        # Calculate trailing stop level (from highest high, not entry)
        trailing_stop_price = highest_high * (1 + trailing_stop_pct / 100)
        trailing_stop_hit = row['Low'] <= trailing_stop_price

        # Fixed take-profit
        tp_hit = high_pct >= take_profit_pct

        # Priority 1: Take-profit (let winners run to target)
        if tp_hit:
            return {
                'exit_type': 'take_profit',
                'exit_day': day_num,
                'exit_price': entry_price * (1 + take_profit_pct / 100),
                'exit_pct': take_profit_pct,
                'highest_high': highest_high,
                'max_gain_pct': ((highest_high - entry_price) / entry_price) * 100,
                'is_winner': True
            }

        # Priority 2: Trailing stop
        if trailing_stop_hit:
            # Exit at trailing stop level
            exit_pct = ((trailing_stop_price - entry_price) / entry_price) * 100

            return {
                'exit_type': 'trailing_stop',
                'exit_day': day_num,
                'exit_price': trailing_stop_price,
                'exit_pct': exit_pct,
                'highest_high': highest_high,
                'max_gain_pct': ((highest_high - entry_price) / entry_price) * 100,
                'is_winner': exit_pct > 0
            }

    # Period end - exit at close
    exit_price = forward_window['Close'].iloc[-1]
    exit_pct = ((exit_price - entry_price) / entry_price) * 100

    return {
        'exit_type': 'period_end',
        'exit_day': period,
        'exit_price': exit_price,
        'exit_pct': exit_pct,
        'highest_high': highest_high,
        'max_gain_pct': ((highest_high - entry_price) / entry_price) * 100,
        'is_winner': exit_pct > 0
    }


def analyze_forward_days(df: pd.DataFrame, entry_idx: int, period: int = 30,
                         trailing_stop_pct: float = -8.0,
                         take_profit_pct: float = 15.0) -> Optional[Dict]:
    """
    Analyze forward period following a breakout signal.

    Args:
        df: DataFrame with OHLCV data
        entry_idx: Index of entry signal
        period: Days to look forward (default: 30)
        trailing_stop_pct: Trailing stop % (default: -8.0)
        take_profit_pct: Take-profit % (default: 15.0)

    Returns:
        Dictionary with analysis results or None
    """
    if entry_idx + period >= len(df):
        return None

    # Entry data
    entry_date = df.index[entry_idx]
    entry_price = df['Close'].iloc[entry_idx]
    entry_rsi = df['RSI'].iloc[entry_idx]
    entry_sma200 = df['SMA200'].iloc[entry_idx]
    entry_volume = df['Volume'].iloc[entry_idx]
    avg_volume = df['AvgVolume'].iloc[entry_idx]

    # Calculate metrics at entry
    distance_from_sma200 = ((entry_price - entry_sma200) / entry_sma200) * 100
    volume_ratio = entry_volume / avg_volume if avg_volume > 0 else 0

    # Calculate outcome
    outcome = calculate_trailing_stop_outcome(
        df, entry_idx, period, trailing_stop_pct, take_profit_pct
    )

    if outcome is None:
        return None

    # Forward window for verification
    forward_window = df.iloc[entry_idx + 1: entry_idx + period + 1]

    if len(forward_window) < period:
        return None

    # Min/max/last day metrics
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
        'distance_from_sma200': distance_from_sma200,
        'volume_ratio': volume_ratio,
        # Exit columns
        'exit_type': outcome['exit_type'],
        'exit_day': outcome['exit_day'],
        'exit_price': outcome['exit_price'],
        'exit_pct': outcome['exit_pct'],
        'highest_high': outcome['highest_high'],
        'max_gain_pct': outcome['max_gain_pct'],
        'is_winner': outcome['is_winner'],
        # Verification columns
        'min_price': min_price,
        'max_price': max_price,
        'last_day_close': last_day_close,
        'min_pct': min_pct,
        'max_pct': max_pct,
        'last_day_pct': last_day_pct
    }


def run_analysis(ticker: str, cache_manager: CacheManager,
                 breakout_period: int = 20, volume_multiplier: float = 2.0,
                 rsi_threshold: int = 60, min_days_above: int = 20,
                 period: int = 30, trailing_stop_pct: float = -8.0,
                 take_profit_pct: float = 15.0) -> pd.DataFrame:
    """
    Run Breakout + Momentum analysis on a ticker.

    Args:
        ticker: Stock ticker symbol
        cache_manager: CacheManager instance
        breakout_period: Days for high lookback (default: 20)
        volume_multiplier: Volume threshold (default: 2.0)
        rsi_threshold: RSI threshold (default: 60)
        min_days_above: Min days above SMA200 (default: 20)
        period: Forward period (default: 30)
        trailing_stop_pct: Trailing stop (default: -8.0)
        take_profit_pct: Take-profit (default: 15.0)

    Returns:
        DataFrame with results
    """
    print(f"\nAnalyzing {ticker}...")
    df = cache_manager.get_ticker_data(ticker)

    if df is None or df.empty:
        return pd.DataFrame()

    # Calculate indicators
    df['SMA200'] = calculate_sma(df, window=200)
    df['RSI'] = calculate_rsi(df, window=14)
    df['AvgVolume'] = df['Volume'].rolling(window=14).mean()

    # Drop NaN rows
    df = df.dropna(subset=['SMA200', 'RSI', 'AvgVolume'])

    # Detect signals
    signal_indices = detect_breakout_signals(
        df, breakout_period=breakout_period,
        volume_multiplier=volume_multiplier,
        rsi_threshold=rsi_threshold,
        min_days_above_sma200=min_days_above
    )

    print(f"  Found {len(signal_indices)} breakout signals (momentum + volume)")

    if not signal_indices:
        return pd.DataFrame()

    # Analyze signals
    results = []
    for idx in signal_indices:
        analysis = analyze_forward_days(
            df, idx, period=period,
            trailing_stop_pct=trailing_stop_pct,
            take_profit_pct=take_profit_pct
        )
        if analysis:
            analysis['ticker'] = ticker
            analysis['strategy'] = 'breakout_momentum'
            results.append(analysis)

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    # Column order
    column_order = [
        'date', 'ticker', 'strategy', 'entry_price', 'entry_rsi',
        'distance_from_sma200', 'volume_ratio',
        'exit_type', 'exit_day', 'exit_price', 'exit_pct',
        'highest_high', 'max_gain_pct', 'is_winner',
        'min_price', 'max_price', 'last_day_close',
        'min_pct', 'max_pct', 'last_day_pct'
    ]
    return results_df[column_order]


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Breakout + Momentum Strategy Backtest',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--ticker', default='AAPL', help='Stock ticker')
    parser.add_argument('--tickers', nargs='+', help='Multiple tickers')
    parser.add_argument('--sp500', action='store_true', help='S&P 500')
    parser.add_argument('--breakout-period', type=int, default=20, help='Breakout period (default: 20)')
    parser.add_argument('--volume-mult', type=float, default=2.0, help='Volume multiplier (default: 2.0)')
    parser.add_argument('--rsi-threshold', type=int, default=60, help='RSI threshold (default: 60)')
    parser.add_argument('--min-days-above', type=int, default=20, help='Min days above SMA200 (default: 20)')
    parser.add_argument('--trailing-stop', type=float, default=-8.0, help='Trailing stop %% (default: -8.0)')
    parser.add_argument('--take-profit', type=float, default=15.0, help='Take-profit %% (default: 15.0)')
    parser.add_argument('--period', type=int, default=30, help='Max hold (default: 30)')
    parser.add_argument('--export', type=str, help='Export CSV')
    parser.add_argument('--cache-dir', default='data/cache/ohlcv', help='Cache dir')

    args = parser.parse_args()

    cache_manager = CacheManager(cache_dir=args.cache_dir)

    # Determine tickers
    if args.sp500:
        print("Loading S&P 500...")
        tickers = load_sp500_tickers()
        print(f"Loaded {len(tickers)} tickers")
        show_individual = False
    elif args.tickers:
        tickers = args.tickers
        show_individual = True
    else:
        tickers = [args.ticker]
        show_individual = True

    # Run analysis
    all_results = []
    failed = []

    for i, ticker in enumerate(tickers, 1):
        if args.sp500:
            print(f"[{i}/{len(tickers)}] {ticker}...", end=' ')

        try:
            results_df = run_analysis(
                ticker, cache_manager,
                breakout_period=args.breakout_period,
                volume_multiplier=args.volume_mult,
                rsi_threshold=args.rsi_threshold,
                min_days_above=args.min_days_above,
                period=args.period,
                trailing_stop_pct=args.trailing_stop,
                take_profit_pct=args.take_profit
            )

            if not results_df.empty:
                all_results.append(results_df)
                if args.sp500:
                    print(f"✓ {len(results_df)}")
                elif show_individual:
                    print_summary(
                        results_df,
                        strategy_name="Breakout Momentum",
                        ticker=ticker,
                        strategy_description=f"{args.breakout_period}-day high breakout + RSI>{args.rsi_threshold} + {args.volume_mult}x volume",
                        settings={
                            'Breakout period': f"{args.breakout_period} days",
                            'RSI threshold': f">{args.rsi_threshold}",
                            'Volume multiplier': f"{args.volume_mult}x",
                            'Trailing stop': f"{args.trailing_stop:.1f}%",
                            'Take-profit': f"{args.take_profit:.1f}%",
                            'Max hold': f"{args.period} days"
                        }
                    )
            else:
                if args.sp500:
                    print("✗")
        except Exception as e:
            failed.append(ticker)
            if args.sp500:
                print(f"✗ {str(e)[:30]}")

    # Display results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)

        if len(tickers) > 1:
            print_aggregate_summary(combined_df, strategy_name="Breakout Momentum")

        if args.export:
            export_dir = os.path.dirname(args.export)
            if export_dir and not os.path.exists(export_dir):
                os.makedirs(export_dir)
            combined_df.to_csv(args.export, index=False)
            print(f"\nExported to: {args.export}")

        if failed:
            print(f"\nFailed ({len(failed)}): {', '.join(failed[:20])}")
    else:
        print("\nNo results.")


if __name__ == '__main__':
    main()
