#!/usr/bin/env python3
"""
ATR-Based Mean Reversion Strategy

A volatility-adaptive mean reversion strategy that uses ATR (Average True Range)
for dynamic stop-loss and take-profit levels instead of fixed percentages.

Key Innovation:
- Stop-loss: 1.5-2x ATR below entry (adapts to stock volatility)
- Take-profit: 2-3x ATR above entry (maintains favorable R:R)
- This prevents getting stopped out on volatile stocks while keeping tight
  stops on stable stocks.

Entry Conditions:
1. Price above SMA200 (uptrend filter)
2. Price sustained above SMA200 for N days (established trend)
3. RSI crosses below threshold (oversold condition)
4. ADX > 20 (confirms trending market, avoids chop)

Exit Conditions:
1. Stop-loss: Entry - (ATR_multiplier_sl * ATR)
2. Take-profit: Entry + (ATR_multiplier_tp * ATR)
3. RSI recovery exit
4. Max hold period

Expected Improvements:
- Better adaptation to individual stock volatility
- Improved risk/reward ratio (targeting 2:1+)
- Fewer whipsaw stops on volatile names
- Similar win rate with better average returns

This is for educational purposes only - not financial advice.
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
from src.indicators.technical import calculate_sma, calculate_rsi, calculate_atr, calculate_adx
from src.analysis.summary import print_summary, print_aggregate_summary


def detect_atr_mean_reversion_signals(
    df: pd.DataFrame,
    rsi_threshold: int = 30,
    min_days_above: int = 15,
    adx_threshold: int = 20,
    require_adx: bool = True
) -> List[int]:
    """
    Detect mean reversion signals with ATR-based risk management.

    Conditions:
    1. Price > SMA200 (uptrend)
    2. Price sustained above SMA200 for min_days_above
    3. RSI crosses below threshold (new oversold)
    4. ADX > adx_threshold (trending, not choppy) - optional

    Args:
        df: DataFrame with SMA200, RSI, ADX columns
        rsi_threshold: RSI oversold level (default: 30)
        min_days_above: Min days above SMA200 (default: 15)
        adx_threshold: Minimum ADX for trend strength (default: 20)
        require_adx: Whether to require ADX filter (default: True)

    Returns:
        List of signal indices
    """
    if len(df) < 200 + min_days_above:
        return []

    signals = []

    for i in range(min_days_above + 1, len(df)):
        # Get values
        curr_rsi = df['RSI'].iloc[i]
        prev_rsi = df['RSI'].iloc[i - 1]
        curr_close = df['Close'].iloc[i]
        curr_sma200 = df['SMA200'].iloc[i]
        curr_adx = df['ADX'].iloc[i] if 'ADX' in df.columns else None
        curr_atr = df['ATR'].iloc[i]

        # Skip NaN
        if pd.isna(curr_rsi) or pd.isna(prev_rsi) or pd.isna(curr_close) or \
           pd.isna(curr_sma200) or pd.isna(curr_atr):
            continue

        # Filter 1: Price above SMA200
        if curr_close <= curr_sma200:
            continue

        # Filter 2: Sustained uptrend
        lookback = df.iloc[i - min_days_above:i]
        all_above = (lookback['Close'] > lookback['SMA200']).all()
        if not all_above:
            continue

        # Filter 3: RSI crossover below threshold
        if not (prev_rsi >= rsi_threshold and curr_rsi < rsi_threshold):
            continue

        # Filter 4: ADX trend strength (optional)
        if require_adx and curr_adx is not None:
            if pd.isna(curr_adx) or curr_adx < adx_threshold:
                continue

        signals.append(i)

    return signals


def calculate_atr_based_outcome(
    df: pd.DataFrame,
    entry_idx: int,
    period: int = 15,
    atr_multiplier_sl: float = 2.0,
    atr_multiplier_tp: float = 3.0,
    rsi_exit_threshold: int = 65
) -> Optional[Dict]:
    """
    Calculate trade outcome using ATR-based dynamic stops.

    Key difference from fixed stops:
    - Stop-loss level = Entry - (ATR * atr_multiplier_sl)
    - Take-profit level = Entry + (ATR * atr_multiplier_tp)

    This adapts to each stock's volatility:
    - Volatile stock (high ATR): Wider stops
    - Stable stock (low ATR): Tighter stops

    Args:
        df: DataFrame with OHLCV and ATR
        entry_idx: Entry signal index
        period: Max hold days (default: 15)
        atr_multiplier_sl: ATR multiplier for stop (default: 2.0)
        atr_multiplier_tp: ATR multiplier for profit (default: 3.0)
        rsi_exit_threshold: RSI level to exit (default: 65)

    Returns:
        Dictionary with exit details
    """
    if entry_idx + period >= len(df):
        return None

    entry_price = df['Close'].iloc[entry_idx]
    entry_atr = df['ATR'].iloc[entry_idx]

    if pd.isna(entry_atr) or entry_atr <= 0:
        return None

    # Calculate ATR-based levels
    stop_loss_price = entry_price - (entry_atr * atr_multiplier_sl)
    take_profit_price = entry_price + (entry_atr * atr_multiplier_tp)

    # Calculate as percentages for reporting
    stop_loss_pct = ((stop_loss_price - entry_price) / entry_price) * 100
    take_profit_pct = ((take_profit_price - entry_price) / entry_price) * 100

    forward_window = df.iloc[entry_idx + 1:entry_idx + period + 1]

    if len(forward_window) < period:
        return None

    # Check each day
    for day_num, (idx, row) in enumerate(forward_window.iterrows(), start=1):
        sl_hit = row['Low'] <= stop_loss_price
        tp_hit = row['High'] >= take_profit_price

        # Priority 1: Stop-loss only
        if sl_hit and not tp_hit:
            return {
                'exit_type': 'stop_loss',
                'exit_day': day_num,
                'exit_price': stop_loss_price,
                'exit_pct': stop_loss_pct,
                'atr_at_entry': entry_atr,
                'sl_distance_atr': atr_multiplier_sl,
                'tp_distance_atr': atr_multiplier_tp,
                'is_winner': False
            }

        # Priority 2: Take-profit only
        if tp_hit and not sl_hit:
            return {
                'exit_type': 'take_profit',
                'exit_day': day_num,
                'exit_price': take_profit_price,
                'exit_pct': take_profit_pct,
                'atr_at_entry': entry_atr,
                'sl_distance_atr': atr_multiplier_sl,
                'tp_distance_atr': atr_multiplier_tp,
                'is_winner': True
            }

        # Priority 3: Both hit same day - bar heuristic
        if sl_hit and tp_hit:
            distance_to_low = abs(row['Open'] - row['Low'])
            distance_to_high = abs(row['High'] - row['Open'])

            if distance_to_low < distance_to_high:
                return {
                    'exit_type': 'stop_loss',
                    'exit_day': day_num,
                    'exit_price': stop_loss_price,
                    'exit_pct': stop_loss_pct,
                    'atr_at_entry': entry_atr,
                    'sl_distance_atr': atr_multiplier_sl,
                    'tp_distance_atr': atr_multiplier_tp,
                    'is_winner': False
                }
            else:
                return {
                    'exit_type': 'take_profit',
                    'exit_day': day_num,
                    'exit_price': take_profit_price,
                    'exit_pct': take_profit_pct,
                    'atr_at_entry': entry_atr,
                    'sl_distance_atr': atr_multiplier_sl,
                    'tp_distance_atr': atr_multiplier_tp,
                    'is_winner': True
                }

        # Priority 4: RSI exit
        curr_rsi = row['RSI'] if 'RSI' in row else None
        if curr_rsi is not None and not pd.isna(curr_rsi) and curr_rsi >= rsi_exit_threshold:
            exit_price = row['Close']
            exit_pct = ((exit_price - entry_price) / entry_price) * 100
            return {
                'exit_type': 'rsi_exit',
                'exit_day': day_num,
                'exit_price': exit_price,
                'exit_pct': exit_pct,
                'atr_at_entry': entry_atr,
                'sl_distance_atr': atr_multiplier_sl,
                'tp_distance_atr': atr_multiplier_tp,
                'is_winner': exit_pct > 0
            }

    # Period end
    exit_price = forward_window['Close'].iloc[-1]
    exit_pct = ((exit_price - entry_price) / entry_price) * 100

    return {
        'exit_type': 'period_end',
        'exit_day': period,
        'exit_price': exit_price,
        'exit_pct': exit_pct,
        'atr_at_entry': entry_atr,
        'sl_distance_atr': atr_multiplier_sl,
        'tp_distance_atr': atr_multiplier_tp,
        'is_winner': exit_pct > 0
    }


def analyze_forward_days(
    df: pd.DataFrame,
    entry_idx: int,
    period: int = 15,
    atr_multiplier_sl: float = 2.0,
    atr_multiplier_tp: float = 3.0,
    rsi_exit_threshold: int = 65
) -> Optional[Dict]:
    """Analyze forward period with ATR-based stops."""
    if entry_idx + period >= len(df):
        return None

    entry_date = df.index[entry_idx]
    entry_price = df['Close'].iloc[entry_idx]
    entry_rsi = df['RSI'].iloc[entry_idx]
    entry_atr = df['ATR'].iloc[entry_idx]
    entry_adx = df['ADX'].iloc[entry_idx] if 'ADX' in df.columns else None

    outcome = calculate_atr_based_outcome(
        df, entry_idx, period,
        atr_multiplier_sl=atr_multiplier_sl,
        atr_multiplier_tp=atr_multiplier_tp,
        rsi_exit_threshold=rsi_exit_threshold
    )

    if outcome is None:
        return None

    # Forward window for verification
    forward_window = df.iloc[entry_idx + 1:entry_idx + period + 1]

    if len(forward_window) < period:
        return None

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
        'entry_atr': entry_atr,
        'entry_adx': entry_adx,
        'atr_pct_of_price': (entry_atr / entry_price) * 100,
        # Exit columns
        'exit_type': outcome['exit_type'],
        'exit_day': outcome['exit_day'],
        'exit_price': outcome['exit_price'],
        'exit_pct': outcome['exit_pct'],
        'sl_distance_atr': outcome['sl_distance_atr'],
        'tp_distance_atr': outcome['tp_distance_atr'],
        'is_winner': outcome['is_winner'],
        # Verification
        'min_price': min_price,
        'max_price': max_price,
        'last_day_close': last_day_close,
        'min_pct': min_pct,
        'max_pct': max_pct,
        'last_day_pct': last_day_pct
    }


def run_analysis(
    ticker: str,
    cache_manager: CacheManager,
    rsi_threshold: int = 30,
    rsi_exit: int = 65,
    min_days_above: int = 15,
    adx_threshold: int = 20,
    require_adx: bool = True,
    period: int = 15,
    atr_multiplier_sl: float = 2.0,
    atr_multiplier_tp: float = 3.0
) -> pd.DataFrame:
    """Run ATR Mean Reversion analysis on a ticker."""
    print(f"\nAnalyzing {ticker}...")
    df = cache_manager.get_ticker_data(ticker)

    if df is None or df.empty:
        return pd.DataFrame()

    # Calculate indicators
    df['SMA200'] = calculate_sma(df, window=200)
    df['RSI'] = calculate_rsi(df, window=14)
    df['ATR'] = calculate_atr(df, window=14)

    # ADX calculation
    adx_df = calculate_adx(df, window=14)
    if not adx_df.empty:
        df['ADX'] = adx_df['ADX_14']

    # Drop NaN
    df = df.dropna(subset=['SMA200', 'RSI', 'ATR'])

    # Detect signals
    signal_indices = detect_atr_mean_reversion_signals(
        df,
        rsi_threshold=rsi_threshold,
        min_days_above=min_days_above,
        adx_threshold=adx_threshold,
        require_adx=require_adx
    )

    print(f"  Found {len(signal_indices)} ATR mean reversion signals")

    if not signal_indices:
        return pd.DataFrame()

    # Analyze signals
    results = []
    for idx in signal_indices:
        analysis = analyze_forward_days(
            df, idx, period=period,
            atr_multiplier_sl=atr_multiplier_sl,
            atr_multiplier_tp=atr_multiplier_tp,
            rsi_exit_threshold=rsi_exit
        )
        if analysis:
            analysis['ticker'] = ticker
            analysis['strategy'] = 'atr_mean_reversion'
            results.append(analysis)

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    column_order = [
        'date', 'ticker', 'strategy', 'entry_price', 'entry_rsi', 'entry_atr',
        'entry_adx', 'atr_pct_of_price',
        'exit_type', 'exit_day', 'exit_price', 'exit_pct',
        'sl_distance_atr', 'tp_distance_atr', 'is_winner',
        'min_price', 'max_price', 'last_day_close',
        'min_pct', 'max_pct', 'last_day_pct'
    ]
    available_cols = [c for c in column_order if c in results_df.columns]
    return results_df[available_cols]


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='ATR-Based Mean Reversion Strategy Backtest',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--ticker', default='AAPL', help='Stock ticker')
    parser.add_argument('--tickers', nargs='+', help='Multiple tickers')
    parser.add_argument('--sp500', action='store_true', help='S&P 500')
    parser.add_argument('--rsi-threshold', type=int, default=30, help='RSI threshold (default: 30)')
    parser.add_argument('--rsi-exit', type=int, default=65, help='RSI exit (default: 65)')
    parser.add_argument('--min-days-above', type=int, default=15, help='Min days above SMA200')
    parser.add_argument('--adx-threshold', type=int, default=20, help='ADX threshold (default: 20)')
    parser.add_argument('--no-adx', action='store_true', help='Disable ADX filter')
    parser.add_argument('--atr-sl', type=float, default=2.0, help='ATR multiplier for SL (default: 2.0)')
    parser.add_argument('--atr-tp', type=float, default=3.0, help='ATR multiplier for TP (default: 3.0)')
    parser.add_argument('--period', type=int, default=15, help='Max hold (default: 15)')
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
                min_days_above=args.min_days_above,
                adx_threshold=args.adx_threshold,
                require_adx=not args.no_adx,
                period=args.period,
                atr_multiplier_sl=args.atr_sl,
                atr_multiplier_tp=args.atr_tp
            )

            if not results_df.empty:
                all_results.append(results_df)
                if args.sp500:
                    print(f"OK {len(results_df)}")
                elif show_individual:
                    print_summary(
                        results_df,
                        strategy_name="ATR Mean Reversion",
                        ticker=ticker,
                        strategy_description=f"Volatility-adaptive stops: SL={args.atr_sl}xATR, TP={args.atr_tp}xATR",
                        settings={
                            'RSI threshold': args.rsi_threshold,
                            'RSI exit': args.rsi_exit,
                            'ATR SL multiplier': f"{args.atr_sl}x",
                            'ATR TP multiplier': f"{args.atr_tp}x",
                            'ADX threshold': args.adx_threshold if not args.no_adx else 'disabled',
                            'Max hold': f"{args.period} days"
                        }
                    )
            else:
                if args.sp500:
                    print("--")
        except Exception as e:
            failed.append(ticker)
            if args.sp500:
                print(f"ERR {str(e)[:30]}")

    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)

        if len(tickers) > 1:
            print_aggregate_summary(combined_df, strategy_name="ATR Mean Reversion")

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
