#!/usr/bin/env python3
"""
Bollinger Band Squeeze Breakout Strategy

A volatility contraction/expansion strategy that identifies periods of
unusually low volatility (the "squeeze") and enters on the breakout.

Strategy Philosophy:
Low volatility periods often precede significant price moves. When Bollinger
Bands contract to historically narrow levels, energy builds up for a move.
The direction is confirmed by the breakout direction and volume.

Key Concept - The Squeeze:
- Bollinger Band Width (BBW) = (Upper - Lower) / Middle
- When BBW is at multi-week lows, a squeeze is forming
- Breakout above upper band = bullish
- High volume confirms the move

Entry Conditions:
1. Price above SMA200 (long-term uptrend)
2. Bollinger Band Width at 20-day low (squeeze detected)
3. Price breaks above upper Bollinger Band
4. Volume > 1.5x average (confirms conviction)
5. RSI > 50 (momentum supports breakout)

Exit Conditions:
1. Trailing stop: 2x ATR from highest high
2. Take-profit: When price touches middle band from above after profit
3. Max hold period

Expected Characteristics:
- Win rate: 50-55%
- Average winner: Large (riding the volatility expansion)
- Risk/Reward: 2:1 to 3:1
- Signal frequency: Lower than mean reversion (quality over quantity)

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
from src.indicators.technical import calculate_sma, calculate_rsi, calculate_atr, calculate_bbands
from src.analysis.summary import print_summary, print_aggregate_summary


def detect_squeeze_breakout_signals(
    df: pd.DataFrame,
    squeeze_lookback: int = 20,
    volume_multiplier: float = 1.5,
    rsi_threshold: int = 50,
    min_days_above: int = 15
) -> List[int]:
    """
    Detect Bollinger Band Squeeze Breakout signals.

    A squeeze is detected when:
    - BB Width is at its lowest in the squeeze_lookback period

    Breakout is confirmed when:
    - Price closes above upper Bollinger Band
    - Volume is above average * volume_multiplier
    - RSI confirms momentum

    Args:
        df: DataFrame with OHLCV, SMA200, BBands, RSI, AvgVolume
        squeeze_lookback: Days to look back for squeeze detection (default: 20)
        volume_multiplier: Volume threshold multiplier (default: 1.5)
        rsi_threshold: Minimum RSI for momentum (default: 50)
        min_days_above: Min days above SMA200 (default: 15)

    Returns:
        List of signal indices
    """
    if len(df) < 200 + min_days_above + squeeze_lookback:
        return []

    signals = []

    for i in range(squeeze_lookback + min_days_above, len(df)):
        # Get current values
        curr_close = df['Close'].iloc[i]
        curr_sma200 = df['SMA200'].iloc[i]
        curr_rsi = df['RSI'].iloc[i]
        curr_volume = df['Volume'].iloc[i]
        avg_volume = df['AvgVolume'].iloc[i]
        curr_bbu = df['BBU'].iloc[i]  # Upper Bollinger Band
        curr_bbw = df['BBW'].iloc[i]  # Bollinger Band Width

        # Skip NaN
        if pd.isna(curr_close) or pd.isna(curr_sma200) or pd.isna(curr_rsi) or \
           pd.isna(curr_volume) or pd.isna(avg_volume) or pd.isna(curr_bbu) or \
           pd.isna(curr_bbw):
            continue

        # Filter 1: Price above SMA200
        if curr_close <= curr_sma200:
            continue

        # Filter 2: Sustained uptrend
        lookback = df.iloc[i - min_days_above:i]
        if not (lookback['Close'] > lookback['SMA200']).all():
            continue

        # Filter 3: Squeeze - BB Width at N-day low
        bbw_lookback = df['BBW'].iloc[i - squeeze_lookback:i]
        if curr_bbw > bbw_lookback.min():
            # Current BBW should be at or near the minimum
            # Allow some tolerance (within 10% of min)
            if curr_bbw > bbw_lookback.min() * 1.1:
                continue

        # Filter 4: Breakout above upper band
        if curr_close <= curr_bbu:
            continue

        # Filter 5: Volume confirmation
        if curr_volume < avg_volume * volume_multiplier:
            continue

        # Filter 6: RSI momentum
        if curr_rsi < rsi_threshold:
            continue

        signals.append(i)

    return signals


def calculate_squeeze_outcome(
    df: pd.DataFrame,
    entry_idx: int,
    period: int = 20,
    trailing_stop_atr: float = 2.0,
    use_middle_band_exit: bool = True
) -> Optional[Dict]:
    """
    Calculate trade outcome for squeeze breakout.

    Exit Logic:
    1. Trailing stop: highest high - (ATR * trailing_stop_atr)
    2. Middle band touch after profit (if enabled)
    3. Period end

    Args:
        df: DataFrame with OHLCV, ATR, BBM
        entry_idx: Entry index
        period: Max hold days (default: 20)
        trailing_stop_atr: ATR multiplier for trailing stop (default: 2.0)
        use_middle_band_exit: Exit if price touches middle band after profit

    Returns:
        Dictionary with exit details
    """
    if entry_idx + period >= len(df):
        return None

    entry_price = df['Close'].iloc[entry_idx]
    entry_atr = df['ATR'].iloc[entry_idx]

    if pd.isna(entry_atr) or entry_atr <= 0:
        return None

    forward_window = df.iloc[entry_idx + 1:entry_idx + period + 1]

    if len(forward_window) < period:
        return None

    highest_high = entry_price
    in_profit = False

    for day_num, (idx, row) in enumerate(forward_window.iterrows(), start=1):
        # Update highest high
        if row['High'] > highest_high:
            highest_high = row['High']

        # Calculate current profit
        curr_pct = ((row['Close'] - entry_price) / entry_price) * 100
        if curr_pct > 0:
            in_profit = True

        # Trailing stop calculation
        trailing_stop_price = highest_high - (entry_atr * trailing_stop_atr)
        trailing_stop_hit = row['Low'] <= trailing_stop_price

        if trailing_stop_hit:
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

        # Middle band exit (only after profit)
        if use_middle_band_exit and in_profit:
            curr_bbm = row['BBM'] if 'BBM' in row else None
            if curr_bbm is not None and not pd.isna(curr_bbm):
                if row['Low'] <= curr_bbm:
                    exit_price = curr_bbm
                    exit_pct = ((exit_price - entry_price) / entry_price) * 100
                    return {
                        'exit_type': 'middle_band',
                        'exit_day': day_num,
                        'exit_price': exit_price,
                        'exit_pct': exit_pct,
                        'highest_high': highest_high,
                        'max_gain_pct': ((highest_high - entry_price) / entry_price) * 100,
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
        'max_gain_pct': ((highest_high - entry_price) / entry_price) * 100,
        'is_winner': exit_pct > 0
    }


def analyze_forward_days(
    df: pd.DataFrame,
    entry_idx: int,
    period: int = 20,
    trailing_stop_atr: float = 2.0,
    use_middle_band_exit: bool = True
) -> Optional[Dict]:
    """Analyze forward period for squeeze breakout."""
    if entry_idx + period >= len(df):
        return None

    entry_date = df.index[entry_idx]
    entry_price = df['Close'].iloc[entry_idx]
    entry_rsi = df['RSI'].iloc[entry_idx]
    entry_atr = df['ATR'].iloc[entry_idx]
    entry_bbw = df['BBW'].iloc[entry_idx]
    entry_volume = df['Volume'].iloc[entry_idx]
    avg_volume = df['AvgVolume'].iloc[entry_idx]

    volume_ratio = entry_volume / avg_volume if avg_volume > 0 else 0

    outcome = calculate_squeeze_outcome(
        df, entry_idx, period,
        trailing_stop_atr=trailing_stop_atr,
        use_middle_band_exit=use_middle_band_exit
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
        'entry_bbw': entry_bbw,
        'volume_ratio': volume_ratio,
        # Exit columns
        'exit_type': outcome['exit_type'],
        'exit_day': outcome['exit_day'],
        'exit_price': outcome['exit_price'],
        'exit_pct': outcome['exit_pct'],
        'highest_high': outcome['highest_high'],
        'max_gain_pct': outcome['max_gain_pct'],
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
    squeeze_lookback: int = 20,
    volume_multiplier: float = 1.5,
    rsi_threshold: int = 50,
    min_days_above: int = 15,
    period: int = 20,
    trailing_stop_atr: float = 2.0,
    use_middle_band_exit: bool = True
) -> pd.DataFrame:
    """Run BB Squeeze Breakout analysis on a ticker."""
    print(f"\nAnalyzing {ticker}...")
    df = cache_manager.get_ticker_data(ticker)

    if df is None or df.empty:
        return pd.DataFrame()

    # Calculate indicators
    df['SMA200'] = calculate_sma(df, window=200)
    df['RSI'] = calculate_rsi(df, window=14)
    df['ATR'] = calculate_atr(df, window=14)
    df['AvgVolume'] = df['Volume'].rolling(window=14).mean()

    # Bollinger Bands
    bbands_df = calculate_bbands(df, window=20, std_dev=2.0)
    if not bbands_df.empty:
        df['BBL'] = bbands_df['BBL_20_2.0']
        df['BBM'] = bbands_df['BBM_20_2.0']
        df['BBU'] = bbands_df['BBU_20_2.0']
        df['BBW'] = bbands_df['BBB_20_2.0']  # Width
    else:
        return pd.DataFrame()

    # Drop NaN
    df = df.dropna(subset=['SMA200', 'RSI', 'ATR', 'BBU', 'BBW'])

    # Detect signals
    signal_indices = detect_squeeze_breakout_signals(
        df,
        squeeze_lookback=squeeze_lookback,
        volume_multiplier=volume_multiplier,
        rsi_threshold=rsi_threshold,
        min_days_above=min_days_above
    )

    print(f"  Found {len(signal_indices)} squeeze breakout signals")

    if not signal_indices:
        return pd.DataFrame()

    # Analyze signals
    results = []
    for idx in signal_indices:
        analysis = analyze_forward_days(
            df, idx, period=period,
            trailing_stop_atr=trailing_stop_atr,
            use_middle_band_exit=use_middle_band_exit
        )
        if analysis:
            analysis['ticker'] = ticker
            analysis['strategy'] = 'bb_squeeze_breakout'
            results.append(analysis)

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    column_order = [
        'date', 'ticker', 'strategy', 'entry_price', 'entry_rsi', 'entry_atr',
        'entry_bbw', 'volume_ratio',
        'exit_type', 'exit_day', 'exit_price', 'exit_pct',
        'highest_high', 'max_gain_pct', 'is_winner',
        'min_price', 'max_price', 'last_day_close',
        'min_pct', 'max_pct', 'last_day_pct'
    ]
    available_cols = [c for c in column_order if c in results_df.columns]
    return results_df[available_cols]


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='BB Squeeze Breakout Strategy Backtest',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--ticker', default='AAPL', help='Stock ticker')
    parser.add_argument('--tickers', nargs='+', help='Multiple tickers')
    parser.add_argument('--sp500', action='store_true', help='S&P 500')
    parser.add_argument('--squeeze-lookback', type=int, default=20, help='Squeeze detection period')
    parser.add_argument('--volume-mult', type=float, default=1.5, help='Volume multiplier')
    parser.add_argument('--rsi-threshold', type=int, default=50, help='Min RSI')
    parser.add_argument('--min-days-above', type=int, default=15, help='Min days above SMA200')
    parser.add_argument('--trailing-stop-atr', type=float, default=2.0, help='Trailing stop ATR mult')
    parser.add_argument('--no-middle-band-exit', action='store_true', help='Disable middle band exit')
    parser.add_argument('--period', type=int, default=20, help='Max hold')
    parser.add_argument('--export', type=str, help='Export CSV')
    parser.add_argument('--cache-dir', default='data/cache/ohlcv', help='Cache dir')
    parser.add_argument('--show-signal-details', action='store_true', help='Show detailed signal table with entry/exit dates')
    parser.add_argument('--max-signals', type=int, default=20, help='Max signals to show in details table (default: 20)')

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
                squeeze_lookback=args.squeeze_lookback,
                volume_multiplier=args.volume_mult,
                rsi_threshold=args.rsi_threshold,
                min_days_above=args.min_days_above,
                period=args.period,
                trailing_stop_atr=args.trailing_stop_atr,
                use_middle_band_exit=not args.no_middle_band_exit
            )

            if not results_df.empty:
                all_results.append(results_df)
                if args.sp500:
                    print(f"OK {len(results_df)}")
                elif show_individual:
                    print_summary(
                        results_df,
                        strategy_name="BB Squeeze Breakout",
                        ticker=ticker,
                        strategy_description=f"BB Width at {args.squeeze_lookback}-day low + upper band breakout",
                        show_signal_dates=not args.show_signal_details,
                        show_signal_details=args.show_signal_details,
                        max_signals_display=args.max_signals,
                        settings={
                            'Squeeze lookback': f"{args.squeeze_lookback} days",
                            'RSI threshold': f">{args.rsi_threshold}",
                            'Volume multiplier': f"{args.volume_mult}x",
                            'Trailing stop': f"{args.trailing_stop_atr}x ATR",
                            'Middle band exit': 'enabled' if not args.no_middle_band_exit else 'disabled',
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
            print_aggregate_summary(combined_df, strategy_name="BB Squeeze Breakout")

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
