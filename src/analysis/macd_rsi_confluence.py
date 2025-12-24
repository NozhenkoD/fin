#!/usr/bin/env python3
"""
MACD + RSI Confluence Strategy

A momentum confirmation strategy that combines MACD crossovers with RSI
to filter for high-probability setups.

Strategy Philosophy:
MACD crossovers are powerful but generate many false signals. By requiring
RSI confirmation, we filter for setups where both momentum indicators agree.

Key Concept - Confluence:
When multiple independent indicators point in the same direction, the
probability of a successful trade increases. This strategy requires:
1. MACD bullish crossover (histogram turns positive)
2. RSI recovery from oversold OR RSI above threshold
3. Price in uptrend (above SMA200)

Entry Conditions:
1. Price above SMA200 (long-term uptrend)
2. MACD histogram crosses from negative to positive (bullish crossover)
3. RSI between 40-60 (not overbought, room to run)
4. Previous RSI was below 40 (recovering from weakness)
5. ADX > 20 (trending market)

Exit Conditions:
1. MACD histogram turns negative (momentum loss)
2. RSI > 70 (overbought)
3. Stop-loss based on ATR
4. Max hold period

Expected Characteristics:
- Win rate: 55-60%
- Moderate signal frequency
- Clean entry timing
- Better than pure MACD

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
from src.indicators.technical import calculate_sma, calculate_rsi, calculate_atr, calculate_macd, calculate_adx
from src.analysis.summary import print_summary, print_aggregate_summary


def detect_macd_rsi_signals(
    df: pd.DataFrame,
    rsi_entry_min: int = 40,
    rsi_entry_max: int = 60,
    rsi_prev_max: int = 45,
    adx_threshold: int = 20,
    min_days_above: int = 15
) -> List[int]:
    """
    Detect MACD + RSI confluence signals.

    Conditions:
    1. Price > SMA200 (uptrend)
    2. MACD histogram crosses from negative to positive
    3. RSI in recovery zone (rsi_entry_min to rsi_entry_max)
    4. Previous RSI was below rsi_prev_max (confirms recovery)
    5. ADX > threshold (trending)

    Args:
        df: DataFrame with SMA200, RSI, MACD histogram, ADX
        rsi_entry_min: Minimum RSI for entry (default: 40)
        rsi_entry_max: Maximum RSI for entry (default: 60)
        rsi_prev_max: Previous RSI must be below this (default: 45)
        adx_threshold: Minimum ADX (default: 20)
        min_days_above: Min days above SMA200 (default: 15)

    Returns:
        List of signal indices
    """
    if len(df) < 200 + min_days_above:
        return []

    signals = []

    for i in range(min_days_above + 26 + 1, len(df)):  # 26 for MACD slow period
        # Get current values
        curr_close = df['Close'].iloc[i]
        curr_sma200 = df['SMA200'].iloc[i]
        curr_rsi = df['RSI'].iloc[i]
        prev_rsi = df['RSI'].iloc[i - 1]
        curr_macd_hist = df['MACDh'].iloc[i]
        prev_macd_hist = df['MACDh'].iloc[i - 1]
        curr_adx = df['ADX'].iloc[i] if 'ADX' in df.columns else None

        # Skip NaN
        if pd.isna(curr_close) or pd.isna(curr_sma200) or pd.isna(curr_rsi) or \
           pd.isna(prev_rsi) or pd.isna(curr_macd_hist) or pd.isna(prev_macd_hist):
            continue

        # Filter 1: Price above SMA200
        if curr_close <= curr_sma200:
            continue

        # Filter 2: Sustained uptrend
        lookback = df.iloc[i - min_days_above:i]
        if not (lookback['Close'] > lookback['SMA200']).all():
            continue

        # Filter 3: MACD histogram crossover (negative to positive)
        if not (prev_macd_hist < 0 and curr_macd_hist > 0):
            continue

        # Filter 4: RSI in recovery zone
        if not (rsi_entry_min <= curr_rsi <= rsi_entry_max):
            continue

        # Filter 5: Previous RSI was lower (recovering)
        if prev_rsi > rsi_prev_max:
            continue

        # Filter 6: ADX trend strength
        if curr_adx is not None and not pd.isna(curr_adx):
            if curr_adx < adx_threshold:
                continue

        signals.append(i)

    return signals


def calculate_macd_outcome(
    df: pd.DataFrame,
    entry_idx: int,
    period: int = 15,
    atr_multiplier_sl: float = 2.0,
    rsi_overbought: int = 70
) -> Optional[Dict]:
    """
    Calculate trade outcome for MACD + RSI strategy.

    Exit Logic:
    1. MACD histogram turns negative (momentum lost)
    2. RSI > rsi_overbought (overbought)
    3. Stop-loss: entry - (ATR * atr_multiplier_sl)
    4. Period end

    Args:
        df: DataFrame with OHLCV, ATR, MACDh, RSI
        entry_idx: Entry index
        period: Max hold days (default: 15)
        atr_multiplier_sl: ATR multiplier for stop (default: 2.0)
        rsi_overbought: RSI overbought level (default: 70)

    Returns:
        Dictionary with exit details
    """
    if entry_idx + period >= len(df):
        return None

    entry_price = df['Close'].iloc[entry_idx]
    entry_atr = df['ATR'].iloc[entry_idx]

    if pd.isna(entry_atr) or entry_atr <= 0:
        return None

    stop_loss_price = entry_price - (entry_atr * atr_multiplier_sl)
    stop_loss_pct = ((stop_loss_price - entry_price) / entry_price) * 100

    forward_window = df.iloc[entry_idx + 1:entry_idx + period + 1]

    if len(forward_window) < period:
        return None

    highest_high = entry_price

    for day_num, (idx, row) in enumerate(forward_window.iterrows(), start=1):
        if row['High'] > highest_high:
            highest_high = row['High']

        # Priority 1: Stop-loss
        if row['Low'] <= stop_loss_price:
            return {
                'exit_type': 'stop_loss',
                'exit_day': day_num,
                'exit_price': stop_loss_price,
                'exit_pct': stop_loss_pct,
                'highest_high': highest_high,
                'is_winner': False
            }

        # Priority 2: RSI overbought exit
        curr_rsi = row['RSI'] if 'RSI' in row else None
        if curr_rsi is not None and not pd.isna(curr_rsi) and curr_rsi >= rsi_overbought:
            exit_price = row['Close']
            exit_pct = ((exit_price - entry_price) / entry_price) * 100
            return {
                'exit_type': 'rsi_overbought',
                'exit_day': day_num,
                'exit_price': exit_price,
                'exit_pct': exit_pct,
                'highest_high': highest_high,
                'is_winner': exit_pct > 0
            }

        # Priority 3: MACD histogram turns negative
        curr_macd_hist = row['MACDh'] if 'MACDh' in row else None
        if curr_macd_hist is not None and not pd.isna(curr_macd_hist) and curr_macd_hist < 0:
            exit_price = row['Close']
            exit_pct = ((exit_price - entry_price) / entry_price) * 100
            return {
                'exit_type': 'macd_exit',
                'exit_day': day_num,
                'exit_price': exit_price,
                'exit_pct': exit_pct,
                'highest_high': highest_high,
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
        'highest_high': highest_high,
        'is_winner': exit_pct > 0
    }


def analyze_forward_days(
    df: pd.DataFrame,
    entry_idx: int,
    period: int = 15,
    atr_multiplier_sl: float = 2.0,
    rsi_overbought: int = 70
) -> Optional[Dict]:
    """Analyze forward period for MACD + RSI strategy."""
    if entry_idx + period >= len(df):
        return None

    entry_date = df.index[entry_idx]
    entry_price = df['Close'].iloc[entry_idx]
    entry_rsi = df['RSI'].iloc[entry_idx]
    entry_atr = df['ATR'].iloc[entry_idx]
    entry_macd = df['MACD'].iloc[entry_idx]
    entry_macd_hist = df['MACDh'].iloc[entry_idx]
    entry_adx = df['ADX'].iloc[entry_idx] if 'ADX' in df.columns else None

    outcome = calculate_macd_outcome(
        df, entry_idx, period,
        atr_multiplier_sl=atr_multiplier_sl,
        rsi_overbought=rsi_overbought
    )

    if outcome is None:
        return None

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
        'entry_macd': entry_macd,
        'entry_macd_hist': entry_macd_hist,
        'entry_adx': entry_adx,
        # Exit columns
        'exit_type': outcome['exit_type'],
        'exit_day': outcome['exit_day'],
        'exit_price': outcome['exit_price'],
        'exit_pct': outcome['exit_pct'],
        'highest_high': outcome['highest_high'],
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
    rsi_entry_min: int = 40,
    rsi_entry_max: int = 60,
    rsi_prev_max: int = 45,
    rsi_overbought: int = 70,
    adx_threshold: int = 20,
    min_days_above: int = 15,
    period: int = 15,
    atr_multiplier_sl: float = 2.0
) -> pd.DataFrame:
    """Run MACD + RSI Confluence analysis on a ticker."""
    print(f"\nAnalyzing {ticker}...")
    df = cache_manager.get_ticker_data(ticker)

    if df is None or df.empty:
        return pd.DataFrame()

    # Calculate indicators
    df['SMA200'] = calculate_sma(df, window=200)
    df['RSI'] = calculate_rsi(df, window=14)
    df['ATR'] = calculate_atr(df, window=14)

    # MACD
    macd_df = calculate_macd(df)
    if not macd_df.empty:
        df['MACD'] = macd_df['MACD_12_26_9']
        df['MACDh'] = macd_df['MACDh_12_26_9']
        df['MACDs'] = macd_df['MACDs_12_26_9']
    else:
        return pd.DataFrame()

    # ADX
    adx_df = calculate_adx(df, window=14)
    if not adx_df.empty:
        df['ADX'] = adx_df['ADX_14']

    # Drop NaN
    df = df.dropna(subset=['SMA200', 'RSI', 'ATR', 'MACD', 'MACDh'])

    # Detect signals
    signal_indices = detect_macd_rsi_signals(
        df,
        rsi_entry_min=rsi_entry_min,
        rsi_entry_max=rsi_entry_max,
        rsi_prev_max=rsi_prev_max,
        adx_threshold=adx_threshold,
        min_days_above=min_days_above
    )

    print(f"  Found {len(signal_indices)} MACD + RSI confluence signals")

    if not signal_indices:
        return pd.DataFrame()

    # Analyze signals
    results = []
    for idx in signal_indices:
        analysis = analyze_forward_days(
            df, idx, period=period,
            atr_multiplier_sl=atr_multiplier_sl,
            rsi_overbought=rsi_overbought
        )
        if analysis:
            analysis['ticker'] = ticker
            analysis['strategy'] = 'macd_rsi_confluence'
            results.append(analysis)

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    column_order = [
        'date', 'ticker', 'strategy', 'entry_price', 'entry_rsi', 'entry_atr',
        'entry_macd', 'entry_macd_hist', 'entry_adx',
        'exit_type', 'exit_day', 'exit_price', 'exit_pct',
        'highest_high', 'is_winner',
        'min_price', 'max_price', 'last_day_close',
        'min_pct', 'max_pct', 'last_day_pct'
    ]
    available_cols = [c for c in column_order if c in results_df.columns]
    return results_df[available_cols]


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='MACD + RSI Confluence Strategy Backtest',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--ticker', default='AAPL', help='Stock ticker')
    parser.add_argument('--tickers', nargs='+', help='Multiple tickers')
    parser.add_argument('--sp500', action='store_true', help='S&P 500')
    parser.add_argument('--rsi-entry-min', type=int, default=40, help='Min RSI for entry')
    parser.add_argument('--rsi-entry-max', type=int, default=60, help='Max RSI for entry')
    parser.add_argument('--rsi-prev-max', type=int, default=45, help='Prev RSI must be below')
    parser.add_argument('--rsi-overbought', type=int, default=70, help='RSI overbought exit')
    parser.add_argument('--adx-threshold', type=int, default=20, help='ADX threshold')
    parser.add_argument('--min-days-above', type=int, default=15, help='Min days above SMA200')
    parser.add_argument('--atr-sl', type=float, default=2.0, help='ATR multiplier for SL')
    parser.add_argument('--period', type=int, default=15, help='Max hold')
    parser.add_argument('--export', type=str, help='Export CSV')
    parser.add_argument('--cache-dir', default='data/cache/ohlcv', help='Cache dir')

    args = parser.parse_args()

    cache_manager = CacheManager(cache_dir=args.cache_dir)

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

    all_results = []
    failed = []

    for i, ticker in enumerate(tickers, 1):
        if args.sp500:
            print(f"[{i}/{len(tickers)}] {ticker}...", end=' ')

        try:
            results_df = run_analysis(
                ticker, cache_manager,
                rsi_entry_min=args.rsi_entry_min,
                rsi_entry_max=args.rsi_entry_max,
                rsi_prev_max=args.rsi_prev_max,
                rsi_overbought=args.rsi_overbought,
                adx_threshold=args.adx_threshold,
                min_days_above=args.min_days_above,
                period=args.period,
                atr_multiplier_sl=args.atr_sl
            )

            if not results_df.empty:
                all_results.append(results_df)
                if args.sp500:
                    print(f"OK {len(results_df)}")
                elif show_individual:
                    print_summary(
                        results_df,
                        strategy_name="MACD RSI Confluence",
                        ticker=ticker,
                        strategy_description=f"MACD histogram cross + RSI {args.rsi_entry_min}-{args.rsi_entry_max} recovery",
                        settings={
                            'RSI entry range': f"{args.rsi_entry_min}-{args.rsi_entry_max}",
                            'RSI prev max': args.rsi_prev_max,
                            'RSI overbought': args.rsi_overbought,
                            'ADX threshold': args.adx_threshold,
                            'ATR stop-loss': f"{args.atr_sl}x",
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
            print_aggregate_summary(combined_df, strategy_name="MACD RSI Confluence")

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
