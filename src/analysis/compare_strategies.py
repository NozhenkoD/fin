#!/usr/bin/env python3
"""
Strategy Comparison Tool

Compares multiple trading strategies side-by-side to identify the best approach.
Analyzes risk/reward, win rate, profit factor, and other key metrics.
"""
import os
import sys
import pandas as pd
import argparse
from typing import List, Dict

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def load_strategy_results(file_path: str, strategy_name: str = None) -> pd.DataFrame:
    """
    Load strategy results from CSV file.

    Args:
        file_path: Path to CSV file
        strategy_name: Optional name override for the strategy

    Returns:
        DataFrame with results
    """
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return pd.DataFrame()

    df = pd.read_csv(file_path)

    if strategy_name:
        df['strategy'] = strategy_name

    return df


def calculate_metrics(df: pd.DataFrame) -> Dict:
    """
    Calculate comprehensive strategy metrics.

    Args:
        df: DataFrame with backtest results

    Returns:
        Dictionary with all metrics
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
    avg_return = df['exit_pct'].mean()
    avg_winner = winners['exit_pct'].mean() if num_winners > 0 else 0
    avg_loser = losers['exit_pct'].mean() if num_losers > 0 else 0

    # Profit factor (total gains / total losses)
    total_gains = winners['exit_pct'].sum() if num_winners > 0 else 0
    total_losses = abs(losers['exit_pct'].sum()) if num_losers > 0 else 0
    profit_factor = (total_gains / total_losses) if total_losses > 0 else float('inf')

    # Risk/reward ratio
    risk_reward = (avg_winner / abs(avg_loser)) if avg_loser != 0 else 0

    # Average hold time
    avg_hold_days = df['exit_day'].mean()

    # Max consecutive losses
    df_sorted = df.sort_values('date')
    max_consec_losses = 0
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
    returns_std = df['exit_pct'].std()
    sharpe_like = (avg_return / returns_std) if returns_std > 0 else 0

    # Exit type distribution
    exit_types = df['exit_type'].value_counts()

    return {
        'total_trades': total_trades,
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
        'exit_type_dist': exit_types.to_dict(),
        'num_winners': num_winners,
        'num_losers': num_losers
    }


def print_comparison_table(strategies: Dict[str, pd.DataFrame]):
    """
    Print side-by-side comparison of strategies.

    Args:
        strategies: Dict mapping strategy name to DataFrame
    """
    print("\n" + "=" * 120)
    print("STRATEGY COMPARISON")
    print("=" * 120)

    # Calculate metrics for each strategy
    all_metrics = {}
    for name, df in strategies.items():
        if not df.empty:
            all_metrics[name] = calculate_metrics(df)

    if not all_metrics:
        print("No valid strategy results to compare.")
        return

    # Print header
    print(f"\n{'Metric':<30}", end='')
    for name in all_metrics.keys():
        print(f"{name:>25}", end='')
    print()
    print("-" * 120)

    # Print metrics
    metrics_to_print = [
        ('Total Trades', 'total_trades', '.0f'),
        ('Win Rate (%)', 'win_rate', '.2f'),
        ('Avg Return (%)', 'avg_return', '+.2f'),
        ('Avg Winner (%)', 'avg_winner', '+.2f'),
        ('Avg Loser (%)', 'avg_loser', '+.2f'),
        ('Profit Factor', 'profit_factor', '.2f'),
        ('Risk/Reward Ratio', 'risk_reward', '.2f'),
        ('Avg Hold Days', 'avg_hold_days', '.1f'),
        ('Max Consec Losses', 'max_consec_losses', '.0f'),
        ('Expectancy ($100)', 'expectancy', '+.2f'),
        ('Sharpe-Like', 'sharpe_like', '.2f'),
    ]

    for label, key, fmt in metrics_to_print:
        print(f"{label:<30}", end='')
        for name, metrics in all_metrics.items():
            value = metrics.get(key, 0)
            if value == float('inf'):
                print(f"{'∞':>25}", end='')
            else:
                print(f"{value:{fmt}}".rjust(25), end='')
        print()

    # Print exit type distribution
    print("\n" + "-" * 120)
    print("EXIT TYPE DISTRIBUTION")
    print("-" * 120)

    # Collect all unique exit types
    all_exit_types = set()
    for metrics in all_metrics.values():
        all_exit_types.update(metrics['exit_type_dist'].keys())

    for exit_type in sorted(all_exit_types):
        print(f"{exit_type:<30}", end='')
        for name, metrics in all_metrics.items():
            count = metrics['exit_type_dist'].get(exit_type, 0)
            pct = (count / metrics['total_trades'] * 100) if metrics['total_trades'] > 0 else 0
            print(f"{count} ({pct:.1f}%)".rjust(25), end='')
        print()

    # Recommendations
    print("\n" + "=" * 120)
    print("RECOMMENDATIONS")
    print("=" * 120)

    # Find best strategies by different criteria
    best_win_rate = max(all_metrics.items(), key=lambda x: x[1]['win_rate'])
    best_profit_factor = max(all_metrics.items(), key=lambda x: x[1]['profit_factor'] if x[1]['profit_factor'] != float('inf') else 0)
    best_expectancy = max(all_metrics.items(), key=lambda x: x[1]['expectancy'])
    best_sharpe = max(all_metrics.items(), key=lambda x: x[1]['sharpe_like'])

    print(f"\n✓ Best Win Rate:        {best_win_rate[0]:<25} ({best_win_rate[1]['win_rate']:.2f}%)")
    print(f"✓ Best Profit Factor:   {best_profit_factor[0]:<25} ({best_profit_factor[1]['profit_factor']:.2f})")
    print(f"✓ Best Expectancy:      {best_expectancy[0]:<25} (${best_expectancy[1]['expectancy']:.2f} per $100)")
    print(f"✓ Best Risk-Adjusted:   {best_sharpe[0]:<25} (Sharpe: {best_sharpe[1]['sharpe_like']:.2f})")

    # Overall recommendation
    print(f"\n{'OVERALL BEST STRATEGY:':<30}", end='')

    # Score each strategy (weighted average of normalized metrics)
    scores = {}
    for name, metrics in all_metrics.items():
        # Normalize metrics to 0-1 scale
        win_rate_score = metrics['win_rate'] / 100
        pf_score = min(metrics['profit_factor'] / 5, 1)  # Cap at 5
        exp_score = (metrics['expectancy'] + 10) / 20  # Normalize -10 to +10 → 0 to 1
        sharpe_score = (metrics['sharpe_like'] + 1) / 3  # Normalize -1 to +2 → 0 to 1

        # Weighted average (customize weights as needed)
        total_score = (
            win_rate_score * 0.3 +
            pf_score * 0.3 +
            exp_score * 0.25 +
            sharpe_score * 0.15
        )
        scores[name] = total_score

    best_overall = max(scores.items(), key=lambda x: x[1])
    print(f"{best_overall[0]:<25} (Score: {best_overall[1]:.2f})")

    print("\n" + "=" * 120 + "\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Compare trading strategies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare pre-existing results
  python -m src.analysis.compare_strategies \\
      --file results/resistance_break.csv "Resistance Break" \\
      --file results/support_bounce.csv "Support Bounce" \\
      --file results/rsi_mean_reversion.csv "RSI Mean Reversion"

  # Quick comparison (assumes standard file names)
  python -m src.analysis.compare_strategies --quick
        """
    )

    parser.add_argument(
        '--file',
        action='append',
        nargs=2,
        metavar=('PATH', 'NAME'),
        help='Add strategy result file with name (can use multiple times)'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick comparison of standard results in results/ directory'
    )

    args = parser.parse_args()

    strategies = {}

    if args.quick:
        # Load standard files
        standard_files = [
            ('results/resistance_break.csv', 'Resistance Break'),
            ('results/support_bounce.csv', 'Support Bounce'),
            ('results/rsi_mean_reversion.csv', 'RSI Mean Reversion'),
        ]

        for file_path, name in standard_files:
            df = load_strategy_results(file_path, name)
            if not df.empty:
                strategies[name] = df
    elif args.file:
        for file_path, name in args.file:
            df = load_strategy_results(file_path, name)
            if not df.empty:
                strategies[name] = df
    else:
        print("Error: Must specify --file or --quick")
        parser.print_help()
        return

    if not strategies:
        print("Error: No valid strategy results loaded")
        return

    # Print comparison
    print_comparison_table(strategies)


if __name__ == '__main__':
    main()
