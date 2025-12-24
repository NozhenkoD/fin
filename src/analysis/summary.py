#!/usr/bin/env python3
"""
Unified Summary Module for Trading Strategies

This module provides a centralized, reusable summary functionality that can be
used across all trading strategies. It eliminates code duplication and provides
consistent output formatting.

Features:
- calculate_metrics(): Comprehensive metrics calculation
- print_summary(): Print summary for single ticker/strategy
- print_aggregate_summary(): Print aggregate summary across multiple tickers
- Optional signal dates display (controlled via show_signal_dates argument)
- Consistent formatting across all strategies

Usage:
    from src.analysis.summary import print_summary, print_aggregate_summary, calculate_metrics

    # Single ticker summary
    print_summary(results_df, strategy_name="RSI Mean Reversion", show_signal_dates=True)

    # Aggregate summary
    print_aggregate_summary(combined_df, strategy_name="RSI Mean Reversion")

    # Get metrics as dictionary
    metrics = calculate_metrics(results_df)
"""
from typing import Dict, List, Optional, Any
from datetime import datetime

import pandas as pd
import numpy as np


def calculate_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate comprehensive strategy metrics from backtest results.

    This function computes all standard trading metrics including win rate,
    profit factor, risk/reward ratio, expectancy, and more.

    Args:
        df: DataFrame with backtest results. Required columns:
            - is_winner: bool indicating if trade was profitable
            - exit_pct: percentage gain/loss of trade
            - exit_day: number of days held
            - exit_type: type of exit (stop_loss, take_profit, period_end, rsi_exit)
            Optional columns:
            - date: trade entry date
            - ticker: stock ticker symbol

    Returns:
        Dictionary containing all calculated metrics:
        - total_trades: Total number of trades
        - num_winners: Number of winning trades
        - num_losers: Number of losing trades
        - win_rate: Win rate as percentage
        - avg_return: Average return per trade
        - avg_winner: Average return of winning trades
        - avg_loser: Average return of losing trades
        - profit_factor: Gross gains / Gross losses
        - risk_reward: Average winner / |Average loser|
        - avg_hold_days: Average holding period
        - max_consec_losses: Maximum consecutive losing trades
        - expectancy: Expected return per $100 traded
        - sharpe_like: Return / Volatility ratio
        - total_return: Sum of all returns
        - exit_type_dist: Distribution of exit types
        - date_range: Tuple of (start_date, end_date) if dates available
        - unique_tickers: Number of unique tickers if ticker column exists
    """
    if df.empty:
        return {}

    winners = df[df['is_winner'] == True]
    losers = df[df['is_winner'] == False]

    total_trades = len(df)
    num_winners = len(winners)
    num_losers = len(losers)

    # Win rate
    win_rate = (num_winners / total_trades * 100) if total_trades > 0 else 0

    # Average returns
    avg_return = df['exit_pct'].mean() if total_trades > 0 else 0
    avg_winner = winners['exit_pct'].mean() if num_winners > 0 else 0
    avg_loser = losers['exit_pct'].mean() if num_losers > 0 else 0

    # Profit factor (total gains / total losses)
    total_gains = winners['exit_pct'].sum() if num_winners > 0 else 0
    total_losses = abs(losers['exit_pct'].sum()) if num_losers > 0 else 0
    profit_factor = (total_gains / total_losses) if total_losses > 0 else float('inf')

    # Risk/reward ratio
    risk_reward = (avg_winner / abs(avg_loser)) if avg_loser != 0 else float('inf')

    # Average hold time
    avg_hold_days = df['exit_day'].mean() if 'exit_day' in df.columns else 0

    # Max consecutive losses
    max_consec_losses = 0
    if 'date' in df.columns:
        df_sorted = df.sort_values('date')
        current_consec = 0
        for is_winner in df_sorted['is_winner']:
            if not is_winner:
                current_consec += 1
                max_consec_losses = max(max_consec_losses, current_consec)
            else:
                current_consec = 0

    # Expectancy (average $ per trade if trading $100)
    expectancy = (win_rate / 100 * avg_winner) + ((100 - win_rate) / 100 * avg_loser)

    # Sharpe-like ratio (return / volatility)
    returns_std = df['exit_pct'].std() if total_trades > 1 else 0
    sharpe_like = (avg_return / returns_std) if returns_std > 0 else 0

    # Total return
    total_return = df['exit_pct'].sum()

    # Exit type distribution
    exit_type_dist = {}
    if 'exit_type' in df.columns:
        exit_type_dist = df['exit_type'].value_counts().to_dict()

    # Date range
    date_range = None
    if 'date' in df.columns and not df['date'].isna().all():
        dates = pd.to_datetime(df['date'])
        date_range = (dates.min(), dates.max())

    # Unique tickers
    unique_tickers = 0
    if 'ticker' in df.columns:
        unique_tickers = df['ticker'].nunique()

    return {
        'total_trades': total_trades,
        'num_winners': num_winners,
        'num_losers': num_losers,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'avg_winner': avg_winner,
        'avg_loser': avg_loser,
        'profit_factor': profit_factor,
        'risk_reward': risk_reward,
        'avg_hold_days': avg_hold_days,
        'max_consec_losses': max_consec_losses,
        'expectancy': expectancy,
        'sharpe_like': sharpe_like,
        'total_return': total_return,
        'exit_type_dist': exit_type_dist,
        'date_range': date_range,
        'unique_tickers': unique_tickers
    }


def format_signal_dates(
    df: pd.DataFrame,
    max_dates: int = 10,
    date_format: str = '%Y-%m-%d'
) -> str:
    """
    Format signal dates for display.

    Args:
        df: DataFrame with 'date' column
        max_dates: Maximum number of dates to show before truncating
        date_format: strftime format for dates

    Returns:
        Formatted string of dates
    """
    if 'date' not in df.columns or df.empty:
        return "No dates available"

    dates = pd.to_datetime(df['date']).sort_values()

    if len(dates) <= max_dates:
        return ", ".join([d.strftime(date_format) for d in dates])
    else:
        visible_dates = ", ".join([d.strftime(date_format) for d in dates.head(max_dates)])
        return f"{visible_dates}, ... (+{len(dates) - max_dates} more)"


def print_summary(
    results_df: pd.DataFrame,
    strategy_name: str = "Strategy",
    ticker: Optional[str] = None,
    show_signal_dates: bool = False,
    settings: Optional[Dict[str, Any]] = None,
    strategy_description: Optional[str] = None,
    width: int = 60
) -> None:
    """
    Print a standardized summary of strategy results.

    This function provides a unified way to display strategy performance
    across all trading strategies. It handles common metrics and can
    optionally display signal dates.

    Args:
        results_df: DataFrame with backtest results. Required columns:
            - is_winner: bool
            - exit_pct: float
            - exit_day: int
            - exit_type: str
            Optional columns:
            - date: datetime
            - entry_rsi, entry_atr, entry_adx, etc. (strategy-specific)
        strategy_name: Name of the strategy (e.g., "RSI Mean Reversion")
        ticker: Optional ticker symbol. If None, uses 'ticker' column if present
        show_signal_dates: If True, displays dates when signals occurred
        settings: Optional dict of strategy settings to display
        strategy_description: Optional description of strategy entry/exit rules
        width: Width of the output formatting

    Returns:
        None (prints to stdout)

    Example:
        print_summary(
            results_df,
            strategy_name="RSI Mean Reversion",
            ticker="AAPL",
            show_signal_dates=True,
            settings={
                'RSI Threshold': 30,
                'Stop-loss': '-3.0%',
                'Take-profit': '+5.0%'
            },
            strategy_description="Buy when RSI < 30, exit on RSI > 70 or SL/TP"
        )
    """
    if results_df.empty:
        print("\nNo results to summarize.")
        return

    # Calculate metrics
    metrics = calculate_metrics(results_df)

    # Determine ticker
    display_ticker = ticker
    if display_ticker is None and 'ticker' in results_df.columns:
        unique_tickers = results_df['ticker'].unique()
        if len(unique_tickers) == 1:
            display_ticker = unique_tickers[0]
        else:
            display_ticker = f"{len(unique_tickers)} tickers"

    # Header
    print("\n" + "=" * width)
    header = f"{strategy_name.upper()}"
    if display_ticker:
        header += f" - {display_ticker}"
    print(header)
    print("=" * width)

    # Date range
    if metrics['date_range']:
        start_date = metrics['date_range'][0]
        end_date = metrics['date_range'][1]
        if hasattr(start_date, 'date'):
            start_date = start_date.date()
        if hasattr(end_date, 'date'):
            end_date = end_date.date()
        print(f"Period: {start_date} to {end_date}")

    # Strategy description
    if strategy_description:
        print(f"\nSTRATEGY")
        print(f"  {strategy_description}")

    # Settings
    if settings:
        print(f"\nSETTINGS")
        for key, value in settings.items():
            print(f"  {key}: {value}")

    # Signals count
    print(f"\nSIGNALS")
    print(f"  Total signals: {metrics['total_trades']}")

    # Exit type breakdown
    if metrics['exit_type_dist']:
        print(f"\nEXIT TYPE BREAKDOWN")
        for exit_type, count in sorted(metrics['exit_type_dist'].items()):
            pct = (count / metrics['total_trades'] * 100)
            # Format exit type name nicely
            display_name = exit_type.replace('_', ' ').title()
            print(f"  {display_name + ':':<18} {count:4d}  ({pct:.1f}%)")

    # Win/loss metrics
    print(f"\nPERFORMANCE METRICS")
    print(f"  Winners:           {metrics['num_winners']:4d}  ({metrics['win_rate']:.1f}%)")
    print(f"  Losers:            {metrics['num_losers']:4d}  ({100 - metrics['win_rate']:.1f}%)")
    print(f"  Avg Return:        {metrics['avg_return']:+.2f}%")
    if metrics['num_winners'] > 0:
        print(f"  Avg Winner:        {metrics['avg_winner']:+.2f}%")
    if metrics['num_losers'] > 0:
        print(f"  Avg Loser:         {metrics['avg_loser']:+.2f}%")
    print(f"  Avg Hold Days:     {metrics['avg_hold_days']:.1f}")

    # Advanced metrics
    print(f"\nADVANCED METRICS")
    if metrics['profit_factor'] != float('inf'):
        print(f"  Profit Factor:     {metrics['profit_factor']:.2f}")
    else:
        print(f"  Profit Factor:     ∞ (no losses)")
    if metrics['risk_reward'] != float('inf'):
        print(f"  Risk/Reward:       {metrics['risk_reward']:.2f}:1")
    else:
        print(f"  Risk/Reward:       ∞ (no losses)")
    print(f"  Expectancy ($100): ${metrics['expectancy']:+.2f}")
    print(f"  Sharpe-like:       {metrics['sharpe_like']:.2f}")
    if metrics['max_consec_losses'] > 0:
        print(f"  Max Consec Losses: {metrics['max_consec_losses']}")

    # Entry characteristics (if present)
    entry_cols = [col for col in results_df.columns if col.startswith('entry_')]
    if entry_cols:
        print(f"\nENTRY CHARACTERISTICS")
        for col in entry_cols:
            if results_df[col].notna().any():
                col_name = col.replace('entry_', '').upper()
                mean_val = results_df[col].mean()
                min_val = results_df[col].min()
                max_val = results_df[col].max()
                print(f"  {col_name}: avg={mean_val:.2f}, min={min_val:.2f}, max={max_val:.2f}")

    # Distribution bins
    print(f"\nRETURN DISTRIBUTION")
    bins = [
        ('< -10%', results_df[results_df['exit_pct'] < -10]),
        ('-10% to -5%', results_df[(results_df['exit_pct'] >= -10) & (results_df['exit_pct'] < -5)]),
        ('-5% to 0%', results_df[(results_df['exit_pct'] >= -5) & (results_df['exit_pct'] < 0)]),
        ('0% to +5%', results_df[(results_df['exit_pct'] >= 0) & (results_df['exit_pct'] < 5)]),
        ('+5% to +10%', results_df[(results_df['exit_pct'] >= 5) & (results_df['exit_pct'] < 10)]),
        ('> +10%', results_df[results_df['exit_pct'] >= 10])
    ]

    for label, subset in bins:
        count = len(subset)
        if count > 0:
            pct = (count / metrics['total_trades']) * 100
            print(f"  {label:14s} {count:4d}  ({pct:.1f}%)")

    # Signal dates (optional)
    if show_signal_dates and 'date' in results_df.columns:
        print(f"\nSIGNAL DATES")
        dates_str = format_signal_dates(results_df, max_dates=10)
        print(f"  {dates_str}")

    print("=" * width + "\n")


def print_aggregate_summary(
    combined_df: pd.DataFrame,
    strategy_name: str = "Strategy",
    show_signal_dates: bool = False,
    show_per_ticker: bool = True,
    top_n_tickers: int = 10,
    width: int = 80
) -> None:
    """
    Print aggregate summary across multiple tickers.

    This function provides a comprehensive summary when a strategy
    has been tested across multiple stocks.

    Args:
        combined_df: DataFrame with results from multiple tickers
        strategy_name: Name of the strategy
        show_signal_dates: If True, shows dates for each ticker's signals
        show_per_ticker: If True, shows per-ticker breakdown
        top_n_tickers: Number of top tickers to show in breakdown
        width: Width of the output formatting

    Returns:
        None (prints to stdout)

    Example:
        print_aggregate_summary(
            combined_df,
            strategy_name="ATR Mean Reversion",
            show_signal_dates=True,
            show_per_ticker=True,
            top_n_tickers=15
        )
    """
    if combined_df.empty:
        print("\nNo aggregate results to summarize.")
        return

    # Calculate overall metrics
    metrics = calculate_metrics(combined_df)

    # Header
    print("\n" + "=" * width)
    print(f"AGGREGATE SUMMARY - {strategy_name.upper()}")
    print("=" * width)

    # Overview
    print(f"Strategy: {strategy_name}")
    if metrics['unique_tickers'] > 0:
        print(f"Total tickers: {metrics['unique_tickers']}")
    print(f"Total signals: {metrics['total_trades']}")

    # Date range
    if metrics['date_range']:
        start_date = metrics['date_range'][0]
        end_date = metrics['date_range'][1]
        if hasattr(start_date, 'date'):
            start_date = start_date.date()
        if hasattr(end_date, 'date'):
            end_date = end_date.date()
        print(f"Period: {start_date} to {end_date}")

    # Overall performance
    print(f"\nOVERALL PERFORMANCE")
    print(f"  Winners:       {metrics['num_winners']:5d}  ({metrics['win_rate']:.1f}%)")
    print(f"  Losers:        {metrics['num_losers']:5d}  ({100 - metrics['win_rate']:.1f}%)")
    print(f"  Avg Return:    {metrics['avg_return']:+.2f}%")
    print(f"  Avg Winner:    {metrics['avg_winner']:+.2f}%")
    print(f"  Avg Loser:     {metrics['avg_loser']:+.2f}%")
    print(f"  Avg Hold Days: {metrics['avg_hold_days']:.1f}")

    # Advanced metrics
    print(f"\nADVANCED METRICS")
    if metrics['profit_factor'] != float('inf'):
        print(f"  Profit Factor:     {metrics['profit_factor']:.2f}")
    else:
        print(f"  Profit Factor:     ∞")
    if metrics['risk_reward'] != float('inf'):
        print(f"  Risk/Reward:       {metrics['risk_reward']:.2f}:1")
    else:
        print(f"  Risk/Reward:       ∞")
    print(f"  Expectancy ($100): ${metrics['expectancy']:+.2f}")
    print(f"  Sharpe-like:       {metrics['sharpe_like']:.2f}")
    print(f"  Total Return:      {metrics['total_return']:+.2f}%")
    if metrics['max_consec_losses'] > 0:
        print(f"  Max Consec Losses: {metrics['max_consec_losses']}")

    # Exit type breakdown
    if metrics['exit_type_dist']:
        print(f"\nEXIT TYPE BREAKDOWN")
        for exit_type, count in sorted(metrics['exit_type_dist'].items()):
            pct = (count / metrics['total_trades'] * 100)
            display_name = exit_type.replace('_', ' ').title()
            print(f"  {display_name + ':':<18} {count:5d}  ({pct:.1f}%)")

    # Per-ticker summary
    if show_per_ticker and 'ticker' in combined_df.columns:
        print(f"\nPER-TICKER SUMMARY (sorted by win rate, top {top_n_tickers})")
        print("-" * width)

        ticker_stats = []
        for ticker in combined_df['ticker'].unique():
            ticker_df = combined_df[combined_df['ticker'] == ticker]
            ticker_metrics = calculate_metrics(ticker_df)

            stats = {
                'ticker': ticker,
                'signals': ticker_metrics['total_trades'],
                'win_rate': ticker_metrics['win_rate'],
                'avg_return': ticker_metrics['avg_return'],
                'profit_factor': ticker_metrics['profit_factor'],
                'avg_hold_days': ticker_metrics['avg_hold_days']
            }
            ticker_stats.append(stats)

        # Sort by win rate descending
        ticker_stats_df = pd.DataFrame(ticker_stats).sort_values('win_rate', ascending=False)

        # Print header
        print(f"{'Ticker':<8} {'Signals':>8} {'Win Rate':>10} {'Avg Ret':>10} {'PF':>8} {'Hold':>6}")
        print("-" * width)

        # Print top N tickers
        for _, row in ticker_stats_df.head(top_n_tickers).iterrows():
            pf_str = f"{row['profit_factor']:.2f}" if row['profit_factor'] != float('inf') else "∞"
            print(f"{row['ticker']:<8} {row['signals']:>8} {row['win_rate']:>9.1f}% {row['avg_return']:>+9.2f}% {pf_str:>8} {row['avg_hold_days']:>5.1f}d")

        if len(ticker_stats_df) > top_n_tickers:
            print(f"\n  ... and {len(ticker_stats_df) - top_n_tickers} more tickers")

    # Signal dates by ticker (optional)
    if show_signal_dates and 'date' in combined_df.columns and 'ticker' in combined_df.columns:
        print(f"\nSIGNAL DATES BY TICKER")
        print("-" * width)

        for ticker in combined_df['ticker'].unique()[:top_n_tickers]:
            ticker_df = combined_df[combined_df['ticker'] == ticker]
            dates_str = format_signal_dates(ticker_df, max_dates=5)
            print(f"  {ticker}: {dates_str}")

        if combined_df['ticker'].nunique() > top_n_tickers:
            print(f"\n  ... and {combined_df['ticker'].nunique() - top_n_tickers} more tickers")

    print("=" * width + "\n")


def get_metrics_dataframe(
    strategies: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Calculate metrics for multiple strategies and return as DataFrame.

    Useful for programmatic comparison of strategies.

    Args:
        strategies: Dict mapping strategy name to results DataFrame

    Returns:
        DataFrame with one row per strategy and metrics as columns

    Example:
        strategies = {
            'RSI': rsi_results_df,
            'SMA200': sma_results_df,
            'ATR': atr_results_df
        }
        comparison_df = get_metrics_dataframe(strategies)
    """
    rows = []
    for name, df in strategies.items():
        if not df.empty:
            metrics = calculate_metrics(df)
            metrics['strategy'] = name
            rows.append(metrics)

    if not rows:
        return pd.DataFrame()

    result_df = pd.DataFrame(rows)

    # Reorder columns
    column_order = [
        'strategy', 'total_trades', 'win_rate', 'avg_return',
        'avg_winner', 'avg_loser', 'profit_factor', 'risk_reward',
        'avg_hold_days', 'expectancy', 'sharpe_like', 'max_consec_losses'
    ]
    available_cols = [c for c in column_order if c in result_df.columns]
    return result_df[available_cols]
