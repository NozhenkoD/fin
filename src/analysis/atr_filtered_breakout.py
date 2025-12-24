#!/usr/bin/env python3
"""
ATR-Filtered False Breakout Strategy

Detects false breakouts below swing lows using ATR-based candle size validation
and enters long positions expecting mean reversion.

Strategy Logic:
1. Identify swing lows (low that is lower than N previous AND N subsequent lows)
2. Track "naked" (untouched) swing lows within lookback period
3. Trigger on close BELOW swing low (false breakout)
4. Validate breakout candle range is within ATR bounds (min/max multipliers)
5. Enter long at close, expecting bounce back
6. Exit via ATR-based stop-loss or take-profit (risk/reward ratio)

Ticket ID: ALGO-001
"""
import os
import sys
import argparse
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.cache import CacheManager
from src.data.sp500_loader import load_sp500_tickers
from src.indicators.technical import calculate_atr


def detect_swing_lows(
    df: pd.DataFrame,
    swing_strength: int = 2
) -> pd.Series:
    """
    Detect swing lows using rolling window comparison.

    A swing low at index i is valid if:
    - df['Low'][i] < df['Low'][i-1], df['Low'][i-2], ... df['Low'][i-swing_strength]
    - df['Low'][i] < df['Low'][i+1], df['Low'][i+2], ... df['Low'][i+swing_strength]

    Args:
        df: DataFrame with OHLC data
        swing_strength: Number of candles on each side (default 2 = 5-candle pattern)

    Returns:
        Series with swing low prices (NaN where no swing low exists)
    """
    swing_lows = pd.Series(index=df.index, data=np.nan, dtype=float)

    lows = df['Low'].values

    # Need at least swing_strength candles on each side
    for i in range(swing_strength, len(df) - swing_strength):
        current_low = lows[i]

        # Check if lower than all previous swing_strength candles
        is_lower_than_previous = all(
            current_low < lows[i - j] for j in range(1, swing_strength + 1)
        )

        # Check if lower than all subsequent swing_strength candles
        is_lower_than_subsequent = all(
            current_low < lows[i + j] for j in range(1, swing_strength + 1)
        )

        if is_lower_than_previous and is_lower_than_subsequent:
            swing_lows.iloc[i] = current_low

    return swing_lows


def get_active_swing_lows(
    df: pd.DataFrame,
    current_idx: int,
    swing_lows: pd.Series,
    lookback_candles: int = 200
) -> List[Tuple[int, float]]:
    """
    Get active (naked/untouched) swing lows within lookback period.

    A swing low is considered "naked" if price hasn't closed below it since it formed.

    Args:
        df: DataFrame with OHLC data
        current_idx: Current candle index (integer position)
        swing_lows: Series of swing low prices
        lookback_candles: How far back to look for swing lows

    Returns:
        List of tuples (integer_position, price) for active swing lows
    """
    # Get lookback window
    start_idx = max(0, current_idx - lookback_candles)
    lookback_window = swing_lows.iloc[start_idx:current_idx]

    # Find all swing lows in window
    swing_low_data = lookback_window.dropna()

    active_lows = []

    for i, swing_price in swing_low_data.items():
        # Get integer position of this swing low
        swing_pos = df.index.get_loc(i)

        # Check if price closed below this swing low between formation and current_idx
        # (this would invalidate/touch the level)
        if swing_pos + 1 < current_idx:
            subsequent_closes = df['Close'].iloc[swing_pos + 1:current_idx]

            # If no close below this level, it's still "naked"
            if len(subsequent_closes) == 0 or subsequent_closes.min() >= swing_price:
                active_lows.append((swing_pos, swing_price))
        else:
            # No subsequent closes to check
            active_lows.append((swing_pos, swing_price))

    return active_lows


def detect_false_breakout_signals(
    df: pd.DataFrame,
    atr_period: int = 14,
    swing_strength: int = 2,
    min_atr_mult: float = 1.0,
    max_atr_mult: float = 2.0,
    lookback_candles: int = 200
) -> List[int]:
    """
    Detect false breakout signals (long only).

    Signal occurs when:
    1. Price closes BELOW an active swing low
    2. Breakout candle range (High - Low) is within ATR bounds
    3. If multiple levels broken, take the deepest one

    Args:
        df: DataFrame with OHLC data and ATR
        atr_period: ATR calculation period
        swing_strength: Candles on each side for swing detection
        min_atr_mult: Minimum ATR multiplier for candle filter
        max_atr_mult: Maximum ATR multiplier for candle filter
        lookback_candles: How far back to track swing lows

    Returns:
        List of indices where entry signals occur
    """
    # Detect all swing lows
    swing_lows = detect_swing_lows(df, swing_strength=swing_strength)

    signals = []

    # Start after swing detection window + ATR period
    start_idx = max(swing_strength * 2, atr_period) + 1

    for i in range(start_idx, len(df)):
        current_close = df['Close'].iloc[i]
        current_high = df['High'].iloc[i]
        current_low = df['Low'].iloc[i]
        current_atr = df['ATR'].iloc[i]

        # Calculate breakout candle range
        candle_range = current_high - current_low

        # Apply ATR filter
        min_range = current_atr * min_atr_mult
        max_range = current_atr * max_atr_mult

        if not (min_range <= candle_range <= max_range):
            continue

        # Get active swing lows
        active_lows = get_active_swing_lows(df, i, swing_lows, lookback_candles)

        if not active_lows:
            continue

        # Check if current close breaks below any swing low
        broken_levels = [
            (idx, price) for idx, price in active_lows
            if current_close < price
        ]

        if broken_levels:
            # Take the deepest level broken (lowest price)
            deepest_level = min(broken_levels, key=lambda x: x[1])
            signals.append(i)

    return signals


def calculate_false_breakout_outcome(
    df: pd.DataFrame,
    entry_idx: int,
    sl_atr_mult: float = 1.0,
    risk_reward: float = 1.0,
    period: int = 15
) -> Optional[Dict]:
    """
    Calculate trade outcome with ATR-based dynamic stops.

    Entry: Close of breakout candle
    Stop Loss: Breakout candle low - (sl_atr_mult * ATR)
    Take Profit: Entry + (Risk * risk_reward)

    Args:
        df: DataFrame with OHLC data and ATR
        entry_idx: Entry signal index
        sl_atr_mult: ATR multiplier for stop-loss
        risk_reward: Risk-to-reward ratio
        period: Maximum hold period in days

    Returns:
        Dict with exit details or None if insufficient data
    """
    if entry_idx >= len(df) - 1:
        return None

    entry_row = df.iloc[entry_idx]
    entry_price = entry_row['Close']
    entry_atr = entry_row['ATR']
    breakout_low = entry_row['Low']

    # Calculate stop loss
    stop_loss_price = breakout_low - (sl_atr_mult * entry_atr)

    # Calculate risk and take profit
    risk = entry_price - stop_loss_price
    take_profit_price = entry_price + (risk * risk_reward)

    # Forward window
    forward_window = df.iloc[entry_idx + 1 : entry_idx + period + 1]

    if forward_window.empty:
        return None

    for day_num, (idx, row) in enumerate(forward_window.iterrows(), start=1):
        sl_hit = row['Low'] <= stop_loss_price
        tp_hit = row['High'] >= take_profit_price

        # Both hit on same day - use open-based heuristic
        if sl_hit and tp_hit:
            distance_to_sl = abs(row['Open'] - stop_loss_price)
            distance_to_tp = abs(row['Open'] - take_profit_price)

            if distance_to_sl < distance_to_tp:
                # Stop loss hit first
                exit_price = stop_loss_price
                exit_pct = ((exit_price - entry_price) / entry_price) * 100
                return {
                    'exit_type': 'stop_loss',
                    'exit_day': day_num,
                    'exit_price': exit_price,
                    'exit_pct': exit_pct,
                    'is_winner': False
                }
            else:
                # Take profit hit first
                exit_price = take_profit_price
                exit_pct = ((exit_price - entry_price) / entry_price) * 100
                return {
                    'exit_type': 'take_profit',
                    'exit_day': day_num,
                    'exit_price': exit_price,
                    'exit_pct': exit_pct,
                    'is_winner': True
                }

        # Only stop loss hit
        if sl_hit:
            exit_price = stop_loss_price
            exit_pct = ((exit_price - entry_price) / entry_price) * 100
            return {
                'exit_type': 'stop_loss',
                'exit_day': day_num,
                'exit_price': exit_price,
                'exit_pct': exit_pct,
                'is_winner': False
            }

        # Only take profit hit
        if tp_hit:
            exit_price = take_profit_price
            exit_pct = ((exit_price - entry_price) / entry_price) * 100
            return {
                'exit_type': 'take_profit',
                'exit_day': day_num,
                'exit_price': exit_price,
                'exit_pct': exit_pct,
                'is_winner': True
            }

    # Period end - exit at close
    final_row = forward_window.iloc[-1]
    exit_price = final_row['Close']
    exit_pct = ((exit_price - entry_price) / entry_price) * 100

    return {
        'exit_type': 'period_end',
        'exit_day': len(forward_window),
        'exit_price': exit_price,
        'exit_pct': exit_pct,
        'is_winner': exit_pct > 0
    }


def analyze_forward_days(
    df: pd.DataFrame,
    entry_idx: int,
    sl_atr_mult: float = 1.0,
    risk_reward: float = 1.0,
    period: int = 15
) -> Optional[Dict]:
    """
    Wrapper for calculate_false_breakout_outcome with additional metadata.

    Returns dict with entry/exit details and verification data.
    """
    outcome = calculate_false_breakout_outcome(
        df, entry_idx, sl_atr_mult, risk_reward, period
    )

    if outcome is None:
        return None

    entry_row = df.iloc[entry_idx]
    entry_price = entry_row['Close']
    entry_atr = entry_row['ATR']
    breakout_low = entry_row['Low']

    # Calculate stop loss and take profit for verification
    stop_loss_price = breakout_low - (sl_atr_mult * entry_atr)
    risk = entry_price - stop_loss_price
    take_profit_price = entry_price + (risk * risk_reward)

    # Get forward window for min/max prices
    forward_end = min(entry_idx + period + 1, len(df))
    forward_window = df.iloc[entry_idx + 1 : forward_end]

    min_price = forward_window['Low'].min() if not forward_window.empty else entry_price
    max_price = forward_window['High'].max() if not forward_window.empty else entry_price

    return {
        'date': df.index[entry_idx],
        'entry_price': entry_price,
        'entry_atr': entry_atr,
        'entry_atr_pct': (entry_atr / entry_price) * 100,
        'breakout_low': breakout_low,
        'candle_range': entry_row['High'] - entry_row['Low'],
        'candle_range_pct': ((entry_row['High'] - entry_row['Low']) / entry_price) * 100,
        'stop_loss_price': stop_loss_price,
        'take_profit_price': take_profit_price,
        'risk_pct': (risk / entry_price) * 100,
        'exit_type': outcome['exit_type'],
        'exit_day': outcome['exit_day'],
        'exit_price': outcome['exit_price'],
        'exit_pct': outcome['exit_pct'],
        'is_winner': outcome['is_winner'],
        'min_price_forward': min_price,
        'max_price_forward': max_price,
        'min_pct_forward': ((min_price - entry_price) / entry_price) * 100,
        'max_pct_forward': ((max_price - entry_price) / entry_price) * 100
    }


def run_analysis(
    ticker: str,
    cache_manager: CacheManager,
    atr_period: int = 14,
    swing_strength: int = 2,
    min_atr_mult: float = 1.0,
    max_atr_mult: float = 2.0,
    sl_atr_mult: float = 1.0,
    risk_reward: float = 1.0,
    lookback_candles: int = 200,
    period: int = 15,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Main backtest function for ATR-Filtered False Breakout Strategy.

    Args:
        ticker: Stock ticker symbol
        cache_manager: CacheManager instance
        atr_period: ATR calculation period
        swing_strength: Candles on each side for swing detection
        min_atr_mult: Minimum ATR multiplier for candle filter
        max_atr_mult: Maximum ATR multiplier for candle filter
        sl_atr_mult: ATR multiplier for stop-loss
        risk_reward: Risk-to-reward ratio
        lookback_candles: How far back to track swing lows
        period: Maximum hold period
        verbose: Enable verbose output

    Returns:
        DataFrame with backtest results
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Analyzing {ticker}")
        print(f"{'='*60}")

    # Load data
    df = cache_manager.get_ticker_data(ticker)

    if df is None or df.empty:
        if verbose:
            print(f"No data available for {ticker}")
        return pd.DataFrame()

    if verbose:
        print(f"Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")

    # Calculate ATR
    df['ATR'] = calculate_atr(df, window=atr_period)

    # Drop NaN values
    df = df.dropna(subset=['ATR'])

    if df.empty:
        if verbose:
            print(f"Insufficient data after indicator calculation for {ticker}")
        return pd.DataFrame()

    # Detect signals
    if verbose:
        print(f"\nDetecting false breakout signals...")
        print(f"  Swing Strength: {swing_strength} (={swing_strength*2+1}-candle pattern)")
        print(f"  ATR Filter: {min_atr_mult}x - {max_atr_mult}x ATR")
        print(f"  Lookback: {lookback_candles} candles")

    signals = detect_false_breakout_signals(
        df,
        atr_period=atr_period,
        swing_strength=swing_strength,
        min_atr_mult=min_atr_mult,
        max_atr_mult=max_atr_mult,
        lookback_candles=lookback_candles
    )

    if verbose:
        print(f"Found {len(signals)} signals")

    if not signals:
        return pd.DataFrame()

    # Analyze each signal
    results = []
    for idx in signals:
        analysis = analyze_forward_days(
            df, idx, sl_atr_mult, risk_reward, period
        )

        if analysis:
            analysis['ticker'] = ticker
            analysis['strategy'] = 'atr_filtered_breakout'
            results.append(analysis)

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    if verbose:
        print_summary(results_df, ticker)

    return results_df


def print_summary(results_df: pd.DataFrame, ticker: str) -> None:
    """Print formatted backtest results summary."""
    if results_df.empty:
        print(f"\nNo signals found for {ticker}")
        return

    total_trades = len(results_df)
    winners = results_df['is_winner'].sum()
    losers = total_trades - winners
    win_rate = (winners / total_trades * 100) if total_trades > 0 else 0

    avg_return = results_df['exit_pct'].mean()
    avg_winner = results_df[results_df['is_winner']]['exit_pct'].mean() if winners > 0 else 0
    avg_loser = results_df[~results_df['is_winner']]['exit_pct'].mean() if losers > 0 else 0

    avg_hold_days = results_df['exit_day'].mean()

    # Exit type distribution
    exit_types = results_df['exit_type'].value_counts()

    # Risk metrics
    avg_risk = results_df['risk_pct'].mean()

    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS: {ticker}")
    print(f"{'='*60}")
    print(f"Total Trades:     {total_trades}")
    print(f"Winners:          {winners} ({win_rate:.1f}%)")
    print(f"Losers:           {losers} ({100-win_rate:.1f}%)")
    print(f"\nReturns:")
    print(f"  Avg Return:     {avg_return:+.2f}%")
    print(f"  Avg Winner:     {avg_winner:+.2f}%")
    print(f"  Avg Loser:      {avg_loser:+.2f}%")
    print(f"\nRisk/Reward:")
    print(f"  Avg Risk:       {avg_risk:.2f}%")
    if avg_loser != 0:
        print(f"  Risk/Reward:    {avg_winner/abs(avg_loser):.2f}:1")
    print(f"\nHold Time:")
    print(f"  Avg Hold Days:  {avg_hold_days:.1f}")
    print(f"\nExit Types:")
    for exit_type, count in exit_types.items():
        pct = (count / total_trades * 100)
        print(f"  {exit_type:15s} {count:3d} ({pct:.1f}%)")
    print(f"{'='*60}\n")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='ATR-Filtered False Breakout Strategy (ALGO-001)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single ticker
  python -m src.analysis.atr_filtered_breakout --ticker AAPL --verbose

  # Multiple tickers
  python -m src.analysis.atr_filtered_breakout --tickers AAPL MSFT GOOGL

  # S&P 500 with custom parameters
  python -m src.analysis.atr_filtered_breakout --sp500 \\
    --atr-period 14 --swing-strength 2 \\
    --min-atr-mult 1.0 --max-atr-mult 2.0 \\
    --sl-atr-mult 1.0 --risk-reward 1.5 \\
    --lookback-candles 200 --period 15 \\
    --export results/atr_filtered_breakout.csv
        """
    )

    # Ticker selection
    ticker_group = parser.add_mutually_exclusive_group()
    ticker_group.add_argument('--ticker', type=str, default='AAPL',
                             help='Single ticker to analyze (default: AAPL)')
    ticker_group.add_argument('--tickers', nargs='+',
                             help='Multiple tickers to analyze')
    ticker_group.add_argument('--sp500', action='store_true',
                             help='Analyze all S&P 500 stocks')

    # Strategy parameters
    parser.add_argument('--atr-period', type=int, default=14,
                       help='ATR calculation period (default: 14)')
    parser.add_argument('--swing-strength', type=int, default=2,
                       help='Candles on each side for swing detection (default: 2)')
    parser.add_argument('--min-atr-mult', type=float, default=1.0,
                       help='Minimum ATR multiplier for candle filter (default: 1.0)')
    parser.add_argument('--max-atr-mult', type=float, default=2.0,
                       help='Maximum ATR multiplier for candle filter (default: 2.0)')
    parser.add_argument('--sl-atr-mult', type=float, default=1.0,
                       help='ATR multiplier for stop-loss (default: 1.0)')
    parser.add_argument('--risk-reward', type=float, default=1.0,
                       help='Risk-to-reward ratio for take-profit (default: 1.0)')
    parser.add_argument('--lookback-candles', type=int, default=200,
                       help='How far back to track swing lows (default: 200)')
    parser.add_argument('--period', type=int, default=15,
                       help='Maximum hold period in days (default: 15)')

    # Data/output
    parser.add_argument('--cache-dir', type=str, default='data/cache/ohlcv',
                       help='Cache directory for OHLCV data')
    parser.add_argument('--export', type=str,
                       help='Export results to CSV file')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')

    args = parser.parse_args()

    # Initialize cache manager
    cache_manager = CacheManager(cache_dir=args.cache_dir)

    # Determine tickers
    if args.sp500:
        tickers = load_sp500_tickers()
        print(f"Loaded {len(tickers)} S&P 500 tickers")
    elif args.tickers:
        tickers = args.tickers
    else:
        tickers = [args.ticker]

    # Run analysis
    all_results = []
    successful = 0
    failed = 0

    for i, ticker in enumerate(tickers, 1):
        if not args.verbose and len(tickers) > 1:
            print(f"Processing {i}/{len(tickers)}: {ticker}...", end='\r')

        try:
            results_df = run_analysis(
                ticker=ticker,
                cache_manager=cache_manager,
                atr_period=args.atr_period,
                swing_strength=args.swing_strength,
                min_atr_mult=args.min_atr_mult,
                max_atr_mult=args.max_atr_mult,
                sl_atr_mult=args.sl_atr_mult,
                risk_reward=args.risk_reward,
                lookback_candles=args.lookback_candles,
                period=args.period,
                verbose=args.verbose
            )

            if not results_df.empty:
                all_results.append(results_df)
                successful += 1
        except Exception as e:
            if args.verbose:
                print(f"Error analyzing {ticker}: {str(e)}")
            failed += 1
            continue

    # Combine and export results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)

        print(f"\n{'='*60}")
        print(f"ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Tickers processed:  {len(tickers)}")
        print(f"  Successful:       {successful}")
        print(f"  Failed:           {failed}")
        print(f"Total signals:      {len(combined_df)}")
        print(f"Overall win rate:   {(combined_df['is_winner'].sum() / len(combined_df) * 100):.1f}%")
        print(f"Avg return:         {combined_df['exit_pct'].mean():+.2f}%")

        if args.export:
            combined_df.to_csv(args.export, index=False)
            print(f"\nResults exported to: {args.export}")

        print(f"{'='*60}\n")
    else:
        print("\nNo signals found across all tickers.")


if __name__ == '__main__':
    main()
