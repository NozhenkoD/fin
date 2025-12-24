#!/usr/bin/env python3
"""
Batch Strategy Testing Tool

Runs multiple strategy configurations from a YAML config file,
saves all results, and automatically compares them to find the best.

Usage:
    python -m src.analysis.batch_test --config config/strategy_test_config.yaml

This will:
1. Load all strategy parameter sets from config
2. Run each one on S&P 500 (or test tickers)
3. Save results as CSV files
4. Generate comprehensive comparison report
5. Show top N best strategies

Perfect for parameter optimization and finding winning strategies!
"""
import os
import sys
import yaml
import argparse
import importlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import pandas as pd

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.cache import CacheManager
from src.data.sp500_loader import load_sp500_tickers


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_strategy(strategy_name: str, script_path: str, params: Dict,
                 tickers: List[str], cache_manager: CacheManager,
                 output_file: str) -> Dict:
    """
    Run a single strategy with given parameters.

    Args:
        strategy_name: Name of the strategy variant
        script_path: Python module path (e.g., 'src.analysis.rsi_mean_reversion')
        params: Parameter dictionary
        tickers: List of tickers to test
        cache_manager: CacheManager instance
        output_file: Path to save results CSV

    Returns:
        Dictionary with summary stats
    """
    print(f"\n{'='*80}")
    print(f"Running: {strategy_name}")
    print(f"Parameters: {params}")
    print(f"{'='*80}")

    # Import the strategy module
    try:
        module = importlib.import_module(script_path)
    except ImportError as e:
        print(f"ERROR: Could not import {script_path}: {e}")
        return None

    # Run analysis on all tickers
    all_results = []
    failed_tickers = []

    for i, ticker in enumerate(tickers, 1):
        if len(tickers) > 100:  # Only show progress for large runs
            if i % 50 == 0:
                print(f"Progress: {i}/{len(tickers)} tickers...")

        try:
            # Call the run_analysis function from the strategy module
            results_df = module.run_analysis(ticker, cache_manager, **params)

            if not results_df.empty:
                all_results.append(results_df)

        except Exception as e:
            failed_tickers.append(ticker)
            if len(tickers) <= 10:  # Show errors for small runs
                print(f"  Error with {ticker}: {str(e)[:50]}")

    # Combine results
    if not all_results:
        print(f"ERROR: No results for {strategy_name}")
        return None

    combined_df = pd.concat(all_results, ignore_index=True)

    # Save to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    combined_df.to_csv(output_file, index=False)

    # Calculate summary stats
    winners = combined_df[combined_df['is_winner'] == True]
    losers = combined_df[combined_df['is_winner'] == False]

    summary = {
        'name': strategy_name,
        'total_signals': len(combined_df),
        'win_rate': (len(winners) / len(combined_df) * 100) if len(combined_df) > 0 else 0,
        'avg_return': combined_df['exit_pct'].mean(),
        'avg_winner': winners['exit_pct'].mean() if len(winners) > 0 else 0,
        'avg_loser': losers['exit_pct'].mean() if len(losers) > 0 else 0,
        'total_gains': winners['exit_pct'].sum() if len(winners) > 0 else 0,
        'total_losses': abs(losers['exit_pct'].sum()) if len(losers) > 0 else 0,
        'profit_factor': 0,
        'csv_file': output_file,
        'failed_tickers': len(failed_tickers)
    }

    # Calculate profit factor
    if summary['total_losses'] > 0:
        summary['profit_factor'] = summary['total_gains'] / summary['total_losses']
    else:
        summary['profit_factor'] = float('inf') if summary['total_gains'] > 0 else 0

    print(f"\nResults for {strategy_name}:")
    print(f"  Signals: {summary['total_signals']}")
    print(f"  Win Rate: {summary['win_rate']:.2f}%")
    print(f"  Avg Return: {summary['avg_return']:+.2f}%")
    print(f"  Profit Factor: {summary['profit_factor']:.2f}")
    print(f"  Saved to: {output_file}")

    return summary


def print_comparison_report(summaries: List[Dict], output_file: str, top_n: int = 10):
    """
    Print and save comprehensive comparison report.

    Args:
        summaries: List of summary dictionaries
        output_file: Path to save text report
        top_n: Number of top strategies to highlight
    """
    # Sort by profit factor
    sorted_summaries = sorted(summaries, key=lambda x: x['profit_factor'], reverse=True)

    report_lines = []
    report_lines.append("\n" + "="*100)
    report_lines.append("COMPREHENSIVE STRATEGY COMPARISON")
    report_lines.append("="*100)
    report_lines.append(f"\nTotal strategies tested: {len(summaries)}")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Top performers
    report_lines.append(f"\n{'='*100}")
    report_lines.append(f"TOP {top_n} STRATEGIES (by Profit Factor)")
    report_lines.append("="*100)
    report_lines.append(f"\n{'Rank':<6} {'Strategy':<30} {'Signals':<10} {'Win Rate':<12} {'Avg Ret':<12} {'PF':<10}")
    report_lines.append("-"*100)

    for i, summary in enumerate(sorted_summaries[:top_n], 1):
        report_lines.append(
            f"{i:<6} {summary['name']:<30} {summary['total_signals']:<10} "
            f"{summary['win_rate']:.2f}%{'':<6} {summary['avg_return']:+.2f}%{'':<6} "
            f"{summary['profit_factor']:.2f}{'':<4}"
        )

    # Full results table
    report_lines.append(f"\n{'='*100}")
    report_lines.append("ALL STRATEGIES (sorted by Profit Factor)")
    report_lines.append("="*100)
    report_lines.append(
        f"\n{'Strategy':<35} {'Signals':<10} {'Win%':<10} {'AvgRet':<10} "
        f"{'AvgWin':<10} {'AvgLoss':<10} {'PF':<10} {'CSV File':<30}"
    )
    report_lines.append("-"*100)

    for summary in sorted_summaries:
        report_lines.append(
            f"{summary['name']:<35} "
            f"{summary['total_signals']:<10} "
            f"{summary['win_rate']:.2f}%{'':<5} "
            f"{summary['avg_return']:+.2f}%{'':<5} "
            f"{summary['avg_winner']:+.2f}%{'':<5} "
            f"{summary['avg_loser']:+.2f}%{'':<5} "
            f"{summary['profit_factor']:.2f}{'':<5} "
            f"{Path(summary['csv_file']).name:<30}"
        )

    # Key insights
    report_lines.append(f"\n{'='*100}")
    report_lines.append("KEY INSIGHTS")
    report_lines.append("="*100)

    # Best win rate
    best_wr = max(summaries, key=lambda x: x['win_rate'])
    report_lines.append(f"\nBest Win Rate: {best_wr['name']} ({best_wr['win_rate']:.2f}%)")

    # Best average return
    best_ar = max(summaries, key=lambda x: x['avg_return'])
    report_lines.append(f"Best Avg Return: {best_ar['name']} ({best_ar['avg_return']:+.2f}%)")

    # Best profit factor
    best_pf = sorted_summaries[0]
    report_lines.append(f"Best Profit Factor: {best_pf['name']} ({best_pf['profit_factor']:.2f})")

    # Most signals
    most_signals = max(summaries, key=lambda x: x['total_signals'])
    report_lines.append(f"Most Signals: {most_signals['name']} ({most_signals['total_signals']} signals)")

    # Recommendations
    report_lines.append(f"\n{'='*100}")
    report_lines.append("RECOMMENDATIONS")
    report_lines.append("="*100)

    # Find strategies with good balance
    good_strategies = [
        s for s in sorted_summaries
        if s['win_rate'] >= 55 and s['profit_factor'] >= 1.8 and s['total_signals'] >= 100
    ]

    if good_strategies:
        report_lines.append(f"\nStrategies with Win Rate ≥ 55%, Profit Factor ≥ 1.8, and ≥ 100 signals:")
        for s in good_strategies[:5]:
            report_lines.append(
                f"  • {s['name']}: {s['win_rate']:.1f}% WR, "
                f"{s['profit_factor']:.2f} PF, {s['total_signals']} signals"
            )
    else:
        report_lines.append("\nNo strategies met the ideal criteria (WR≥55%, PF≥1.8, Signals≥100)")
        report_lines.append(f"Best available: {best_pf['name']} with {best_pf['profit_factor']:.2f} PF")

    # CSV files for top strategies
    report_lines.append(f"\n{'='*100}")
    report_lines.append("CSV FILES FOR TOP STRATEGIES")
    report_lines.append("="*100)
    for i, summary in enumerate(sorted_summaries[:top_n], 1):
        report_lines.append(f"{i}. {summary['csv_file']}")

    report_lines.append("\n" + "="*100)

    # Print to console
    report_text = "\n".join(report_lines)
    print(report_text)

    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(report_text)

    print(f"\nReport saved to: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Batch test multiple strategy configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--config',
        default='config/strategy_test_config.yaml',
        help='Path to config file (default: config/strategy_test_config.yaml)'
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Initialize cache
    cache_dir = config['execution']['cache_dir']
    cache_manager = CacheManager(cache_dir=cache_dir)

    # Determine tickers
    if config['execution']['run_sp500']:
        print("Loading S&P 500 tickers...")
        tickers = load_sp500_tickers()
        print(f"Loaded {len(tickers)} tickers")
    else:
        tickers = config['execution']['test_tickers']
        print(f"Using test tickers: {tickers}")

    # Run all strategies
    all_summaries = []
    results_dir = config['output']['results_dir']

    print(f"\n{'='*100}")
    print("STARTING BATCH TESTING")
    print(f"{'='*100}")
    print(f"Total strategies to test: ", end='')

    # Count total parameter sets
    total_tests = sum(
        len(strategy_config['parameter_sets'])
        for strategy_config in config['strategies'].values()
        if strategy_config.get('enabled', True)
    )
    print(total_tests)

    start_time = datetime.now()
    test_num = 0

    # Run each strategy
    for strategy_key, strategy_config in config['strategies'].items():
        if not strategy_config.get('enabled', True):
            print(f"\nSkipping {strategy_key} (disabled)")
            continue

        script_path = strategy_config['script']

        for param_set in strategy_config['parameter_sets']:
            test_num += 1
            param_name = param_set.pop('name')
            output_file = os.path.join(results_dir, f"{param_name}.csv")

            print(f"\n[{test_num}/{total_tests}] Testing {param_name}...")

            summary = run_strategy(
                param_name, script_path, param_set,
                tickers, cache_manager, output_file
            )

            if summary:
                all_summaries.append(summary)

    # Generate comparison report
    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\n{'='*100}")
    print("BATCH TESTING COMPLETE")
    print(f"{'='*100}")
    print(f"Duration: {duration}")
    print(f"Successful tests: {len(all_summaries)}/{total_tests}")

    if all_summaries:
        comparison_file = config['output']['comparison_file']
        top_n = config['output']['top_n_strategies']

        print_comparison_report(all_summaries, comparison_file, top_n)

        # Also create CSV comparison for easy import to Excel
        summary_df = pd.DataFrame(all_summaries)
        summary_df = summary_df.sort_values('profit_factor', ascending=False)
        summary_csv = comparison_file.replace('.txt', '.csv')
        summary_df.to_csv(summary_csv, index=False)
        print(f"Summary CSV saved to: {summary_csv}")
    else:
        print("ERROR: No successful tests completed!")


if __name__ == '__main__':
    main()
