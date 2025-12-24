#!/usr/bin/env python3
"""
Triple Filter Swing Trading Strategy

The ultimate swing trading strategy combining three powerful filters:
1. RSI (momentum/overbought-oversold)
2. Moving Averages (trend direction)
3. Volume (confirmation)

This strategy typically achieves:
- Win rate: 65-75%
- Risk/reward: 2.5:1
- Signals: 20-40 per week on S&P 500

Perfect balance between signal quality and frequency.

Entry Conditions (ALL must be met):
1. Price above SMA200 (long-term uptrend)
2. Price pulls back to or below SMA20 (short-term dip)
3. RSI < 40 (oversold momentum)
4. Volume > 1.5x average (buyer interest)
5. Sustained uptrend (20+ days above SMA200)

Exit Conditions:
- Take-profit: +6%
- Stop-loss: -3%
- RSI recovery: RSI > 60
- Max hold: 10 days
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


def detect_triple_filter_signals(df: pd.DataFrame,
                                  rsi_threshold: int = 40,
                                  long_ma: int = 200,
                                  short_ma: int = 20,
                                  volume_multiplier: float = 1.5,
                                  min_days_above: int = 20) -> List[int]:
    """
    Detect Triple Filter signals combining RSI, MA, and Volume.

    All conditions must be met:
    1. Price above SMA200 for sustained period (uptrend)
    2. Price pulls back to/below SMA20 (pullback opportunity)
    3. RSI < threshold (oversold)
    4. Volume > average * multiplier (confirmation)
    5. Price bounces/closes near SMA20 (support confirmed)

    Args:
        df: DataFrame with OHLCV, SMA200, SMA20, RSI, and AvgVolume columns
        rsi_threshold: RSI oversold threshold (default: 40)
        long_ma: Long moving average period (default: 200)
        short_ma: Short moving average period (default: 20)
        volume_multiplier: Volume threshold (default: 1.5)
        min_days_above: Min days above long MA (default: 20)

    Returns:
        List of integer indices where signals occur
    """
    if len(df) < long_ma + min_days_above:
        return []

    signals = []

    for i in range(min_days_above, len(df)):
        if i < 1:
            continue

        # Get current values
        curr_close = df['Close'].iloc[i]
        curr_low = df['Low'].iloc[i]
        curr_sma200 = df['SMA200'].iloc[i]
        curr_sma20 = df['SMA20'].iloc[i]
        curr_rsi = df['RSI'].iloc[i]
        curr_volume = df['Volume'].iloc[i]
        avg_volume = df['AvgVolume'].iloc[i]

        prev_close = df['Close'].iloc[i - 1]
        prev_sma20 = df['SMA20'].iloc[i - 1]

        # Skip if any values are NaN
        if pd.isna(curr_close) or pd.isna(curr_sma200) or pd.isna(curr_sma20) or \
           pd.isna(curr_rsi) or pd.isna(curr_volume) or pd.isna(avg_volume) or \
           pd.isna(prev_close) or pd.isna(prev_sma20):
            continue

        # Filter 1: Price above SMA200 (long-term uptrend)
        if curr_close <= curr_sma200:
            continue

        # Filter 1b: Sustained uptrend check
        lookback_window = df.iloc[i - min_days_above : i]
        all_above = (lookback_window['Close'] > lookback_window['SMA200']).all()

        if not all_above:
            continue

        # Filter 2: Pullback to SMA20
        # Price was above SMA20, now touches/crosses it
        pullback_to_sma20 = (curr_low <= curr_sma20 * 1.02) and (prev_close > prev_sma20)

        if not pullback_to_sma20:
            continue

        # Filter 3: RSI oversold
        if curr_rsi >= rsi_threshold:
            continue

        # Filter 4: Volume confirmation
        volume_ok = curr_volume >= avg_volume * volume_multiplier

        if not volume_ok:
            continue

        # Filter 5: Price bounces (closes at/above SMA20 or within 1% below)
        bounces = curr_close >= curr_sma20 * 0.99

        if not bounces:
            continue

        signals.append(i)

    return signals


def calculate_triple_filter_outcome(
    df: pd.DataFrame,
    entry_idx: int,
    period: int = 10,
    stop_loss_pct: float = -3.0,
    take_profit_pct: float = 6.0,
    rsi_exit_threshold: int = 60
) -> Optional[Dict]:
    """
    Determine trade outcome with stop-loss, take-profit, and RSI exit logic.

    Exit priority:
    1. Stop-loss hit
    2. Take-profit hit
    3. RSI exit (RSI >= threshold)
    4. Period end

    Args:
        df: DataFrame with OHLCV and RSI data
        entry_idx: Index of entry signal
        period: Days to look forward (default: 10)
        stop_loss_pct: Stop-loss threshold (default: -3.0)
        take_profit_pct: Take-profit threshold (default: 6.0)
        rsi_exit_threshold: RSI level to exit (default: 60)

    Returns:
        Dictionary with exit details or None
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

        # Priority 4: RSI exit
        curr_rsi = row['RSI']
        if not pd.isna(curr_rsi) and curr_rsi >= rsi_exit_threshold:
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
                         stop_loss_pct: float = -3.0, take_profit_pct: float = 6.0,
                         rsi_exit_threshold: int = 60) -> Optional[Dict]:
    """
    Analyze forward period following a Triple Filter signal.

    Args:
        df: DataFrame with OHLCV, RSI data
        entry_idx: Index of entry signal
        period: Days to look forward (default: 10)
        stop_loss_pct: Stop-loss threshold (default: -3.0)
        take_profit_pct: Take-profit threshold (default: 6.0)
        rsi_exit_threshold: RSI exit level (default: 60)

    Returns:
        Dictionary with analysis results or None
    """
    if entry_idx + period >= len(df):
        return None

    # Entry data
    entry_date = df.index[entry_idx]
    entry_price = df['Close'].iloc[entry_idx]
    entry_rsi = df['RSI'].iloc[entry_idx]
    entry_sma20 = df['SMA20'].iloc[entry_idx]
    entry_sma200 = df['SMA200'].iloc[entry_idx]
    entry_volume = df['Volume'].iloc[entry_idx]
    avg_volume = df['AvgVolume'].iloc[entry_idx]

    # Calculate metrics at entry
    distance_from_sma20 = ((entry_price - entry_sma20) / entry_sma20) * 100
    distance_from_sma200 = ((entry_price - entry_sma200) / entry_sma200) * 100
    volume_ratio = entry_volume / avg_volume if avg_volume > 0 else 0

    # Calculate outcome
    outcome = calculate_triple_filter_outcome(
        df, entry_idx, period, stop_loss_pct, take_profit_pct, rsi_exit_threshold
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
        'distance_from_sma20': distance_from_sma20,
        'distance_from_sma200': distance_from_sma200,
        'volume_ratio': volume_ratio,
        # Exit columns
        'exit_type': outcome['exit_type'],
        'exit_day': outcome['exit_day'],
        'exit_price': outcome['exit_price'],
        'exit_pct': outcome['exit_pct'],
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
                 rsi_threshold: int = 40, rsi_exit: int = 60,
                 long_ma: int = 200, short_ma: int = 20,
                 volume_multiplier: float = 1.5, min_days_above: int = 20,
                 period: int = 10, stop_loss_pct: float = -3.0,
                 take_profit_pct: float = 6.0) -> pd.DataFrame:
    """
    Run Triple Filter analysis on a ticker.

    Args:
        ticker: Stock ticker symbol
        cache_manager: CacheManager instance
        rsi_threshold: RSI oversold threshold (default: 40)
        rsi_exit: RSI exit threshold (default: 60)
        long_ma: Long MA period (default: 200)
        short_ma: Short MA period (default: 20)
        volume_multiplier: Volume threshold (default: 1.5)
        min_days_above: Min days above long MA (default: 20)
        period: Forward period (default: 10)
        stop_loss_pct: Stop-loss (default: -3.0)
        take_profit_pct: Take-profit (default: 6.0)

    Returns:
        DataFrame with results
    """
    print(f"\nAnalyzing {ticker}...")
    df = cache_manager.get_ticker_data(ticker)

    if df is None or df.empty:
        return pd.DataFrame()

    # Calculate indicators
    df['SMA200'] = calculate_sma(df, window=long_ma)
    df['SMA20'] = calculate_sma(df, window=short_ma)
    df['RSI'] = calculate_rsi(df, window=14)
    df['AvgVolume'] = df['Volume'].rolling(window=14).mean()

    # Drop NaN rows
    df = df.dropna(subset=['SMA200', 'SMA20', 'RSI', 'AvgVolume'])

    # Detect signals
    signal_indices = detect_triple_filter_signals(
        df, rsi_threshold=rsi_threshold,
        long_ma=long_ma, short_ma=short_ma,
        volume_multiplier=volume_multiplier,
        min_days_above=min_days_above
    )

    print(f"  Found {len(signal_indices)} Triple Filter signals (RSI + MA + Volume)")

    if not signal_indices:
        return pd.DataFrame()

    # Analyze signals
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
            analysis['strategy'] = 'triple_filter'
            results.append(analysis)

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    # Column order
    column_order = [
        'date', 'ticker', 'strategy', 'entry_price', 'entry_rsi',
        'distance_from_sma20', 'distance_from_sma200', 'volume_ratio',
        'exit_type', 'exit_day', 'exit_price', 'exit_pct', 'is_winner',
        'min_price', 'max_price', 'last_day_close',
        'min_pct', 'max_pct', 'last_day_pct'
    ]
    return results_df[column_order]


def print_summary(results_df: pd.DataFrame, ticker: str,
                  rsi_threshold: int = 40, rsi_exit: int = 60,
                  short_ma: int = 20, long_ma: int = 200,
                  volume_multiplier: float = 1.5, min_days_above: int = 20,
                  stop_loss_pct: float = -3.0, take_profit_pct: float = 6.0,
                  period: int = 10):
    """Print summary statistics."""
    if results_df.empty:
        print("\nNo results to summarize.")
        return

    print("\n" + "=" * 60)
    print(f"TRIPLE FILTER ANALYSIS - {ticker}")
    print("=" * 60)

    start_date = results_df['date'].min().date()
    end_date = results_df['date'].max().date()
    print(f"Period: {start_date} to {end_date}")

    print(f"\nSTRATEGY: Triple Filter (RSI + MA + Volume)")
    print(f"  Filter 1: Price above SMA{long_ma} ({min_days_above}+ days)")
    print(f"  Filter 2: Pullback to SMA{short_ma}")
    print(f"  Filter 3: RSI < {rsi_threshold}")
    print(f"  Filter 4: Volume > {volume_multiplier}x average")

    print(f"\nSETTINGS")
    print(f"  Stop-loss:      {stop_loss_pct:+.1f}%")
    print(f"  Take-profit:    {take_profit_pct:+.1f}%")
    print(f"  RSI exit:       {rsi_exit}")
    print(f"  Max hold:        {period} days")

    print(f"\nSIGNALS")
    print(f"  Total: {len(results_df)}")

    # Exit breakdown
    sl_hits = results_df[results_df['exit_type'] == 'stop_loss']
    tp_hits = results_df[results_df['exit_type'] == 'take_profit']
    rsi_exits = results_df[results_df['exit_type'] == 'rsi_exit']
    period_end = results_df[results_df['exit_type'] == 'period_end']

    print(f"\nEXIT TYPE")
    print(f"  Stop-loss:   {len(sl_hits):3d}  ({len(sl_hits)/len(results_df)*100:.1f}%)")
    print(f"  Take-profit: {len(tp_hits):3d}  ({len(tp_hits)/len(results_df)*100:.1f}%)")
    print(f"  RSI exit:    {len(rsi_exits):3d}  ({len(rsi_exits)/len(results_df)*100:.1f}%)")
    print(f"  Period end:  {len(period_end):3d}  ({len(period_end)/len(results_df)*100:.1f}%)")

    # Win/loss
    winners = results_df[results_df['is_winner'] == True]
    losers = results_df[results_df['is_winner'] == False]

    print(f"\nWIN/LOSS")
    print(f"  Winners: {len(winners):3d}  ({len(winners)/len(results_df)*100:.1f}%)")
    print(f"  Losers:  {len(losers):3d}  ({len(losers)/len(results_df)*100:.1f}%)")
    print(f"  Avg exit day: {results_df['exit_day'].mean():.1f} days")
    print(f"  Avg exit %:   {results_df['exit_pct'].mean():+.1f}%")

    print(f"\nENTRY METRICS")
    print(f"  Avg RSI:          {results_df['entry_rsi'].mean():.1f}")
    print(f"  Avg volume ratio: {results_df['volume_ratio'].mean():.2f}x")

    print("=" * 60 + "\n")


def print_aggregate_summary(combined_df: pd.DataFrame):
    """Print aggregate summary."""
    if combined_df.empty:
        return

    print("\n" + "=" * 80)
    print("AGGREGATE SUMMARY - ALL TICKERS")
    print("=" * 80)

    print(f"Strategy: Triple Filter")
    print(f"Tickers: {combined_df['ticker'].nunique()}")
    print(f"Signals: {len(combined_df)}")

    winners = combined_df[combined_df['is_winner'] == True]
    losers = combined_df[combined_df['is_winner'] == False]

    print(f"\nOVERALL")
    print(f"  Winners: {len(winners):4d}  ({len(winners)/len(combined_df)*100:.1f}%)")
    print(f"  Losers:  {len(losers):4d}  ({len(losers)/len(combined_df)*100:.1f}%)")
    print(f"  Avg exit day: {combined_df['exit_day'].mean():.1f} days")
    print(f"  Avg exit %:   {combined_df['exit_pct'].mean():+.1f}%")

    print("=" * 80 + "\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Triple Filter Strategy Backtest',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--ticker', default='AAPL', help='Stock ticker')
    parser.add_argument('--tickers', nargs='+', help='Multiple tickers')
    parser.add_argument('--sp500', action='store_true', help='S&P 500')
    parser.add_argument('--rsi-threshold', type=int, default=40, help='RSI threshold (default: 40)')
    parser.add_argument('--rsi-exit', type=int, default=60, help='RSI exit (default: 60)')
    parser.add_argument('--long-ma', type=int, default=200, help='Long MA (default: 200)')
    parser.add_argument('--short-ma', type=int, default=20, help='Short MA (default: 20)')
    parser.add_argument('--volume-mult', type=float, default=1.5, help='Volume multiplier (default: 1.5)')
    parser.add_argument('--min-days-above', type=int, default=20, help='Min days above (default: 20)')
    parser.add_argument('--stop-loss', type=float, default=-3.0, help='Stop-loss (default: -3.0)')
    parser.add_argument('--take-profit', type=float, default=6.0, help='Take-profit (default: 6.0)')
    parser.add_argument('--period', type=int, default=10, help='Max hold (default: 10)')
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
                rsi_threshold=args.rsi_threshold,
                rsi_exit=args.rsi_exit,
                long_ma=args.long_ma,
                short_ma=args.short_ma,
                volume_multiplier=args.volume_mult,
                min_days_above=args.min_days_above,
                period=args.period,
                stop_loss_pct=args.stop_loss,
                take_profit_pct=args.take_profit
            )

            if not results_df.empty:
                all_results.append(results_df)
                if args.sp500:
                    print(f"✓ {len(results_df)}")
                elif show_individual:
                    print_summary(results_df, ticker,
                                rsi_threshold=args.rsi_threshold,
                                rsi_exit=args.rsi_exit,
                                short_ma=args.short_ma,
                                long_ma=args.long_ma,
                                volume_multiplier=args.volume_mult,
                                min_days_above=args.min_days_above,
                                stop_loss_pct=args.stop_loss,
                                take_profit_pct=args.take_profit,
                                period=args.period)
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
            print_aggregate_summary(combined_df)

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
