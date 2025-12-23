#!/usr/bin/env python3
"""
Stock Screener - Main Entry Point

A modular CLI stock screener that filters US stocks based on dynamic user-defined rules.
"""

import sys
import os
from typing import List, Dict

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.cli_parser import CLIParser
from src.utils.config_loader import ConfigLoader
from src.engine.rule_parser import RuleParser
from src.engine.screener import ScreeningEngine
from src.data.fetcher import DataFetcher
from src.presentation.formatter import ResultFormatter
from src.models.result import ScreeningResult


def main():
    """Main entry point for the stock screener."""

    # 1. Parse CLI arguments
    cli_parser = CLIParser()
    args = cli_parser.parse_args()

    if args.verbose:
        print("Stock Screener - Starting...\n")

    # 2. Load settings
    try:
        settings = ConfigLoader.load_settings(args.settings)
        if args.verbose:
            print(f"Loaded settings from {args.settings}")
    except Exception as e:
        print(f"Warning: Could not load settings: {e}")
        print("Using default settings.\n")
        settings = {
            'data': {'period': '1y', 'interval': '1d'},
            'display': {'show_summary': True}
        }

    # 3. Load or create rules configuration
    try:
        if args.rules:
            # CLI rules override config file
            if args.verbose:
                print(f"Using {len(args.rules)} rule(s) from command line")

            rule_parser = RuleParser()
            rules_dicts = [rule_parser.parse_cli_rule_string(r) for r in args.rules]
            rules_config = {'rules': rules_dicts, 'indicators': {}}
        else:
            # Load from config file
            rules_config = ConfigLoader.load_rules(args.config)
            if args.verbose:
                print(f"Loaded {len(rules_config['rules'])} rule(s) from {args.config}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading rules: {e}")
        sys.exit(1)

    # 4. Load tickers
    try:
        if args.tickers:
            # Use tickers from command line
            ticker_list = args.tickers
            # Create minimal metadata
            ticker_metadata = {
                t: {'ticker': t, 'company_name': t, 'sector': 'Unknown'}
                for t in ticker_list
            }
            if args.verbose:
                print(f"Using {len(ticker_list)} ticker(s) from command line")
        else:
            # Load from file
            ticker_metadata = ConfigLoader.load_tickers(args.ticker_file)
            ticker_list = list(ticker_metadata.keys())
            if args.verbose:
                print(f"Loaded {len(ticker_list)} ticker(s) from {args.ticker_file}")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease create a data/tickers.json file with the following format:")
        print("""
{
  "AAPL": {
    "company_name": "Apple Inc.",
    "ticker": "AAPL",
    "sector": "Technology"
  },
  "MSFT": {
    "company_name": "Microsoft Corporation",
    "ticker": "MSFT",
    "sector": "Technology"
  }
}
        """)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading tickers: {e}")
        sys.exit(1)

    # 5. Initialize components
    rule_parser = RuleParser()

    try:
        rules = rule_parser.parse_ruleset(rules_config['rules'])
        print(f"\n[DEBUG] Parsed {len(rules)} rules:")
        for i, rule in enumerate(rules, 1):
            print(f"  {i}. {rule} (value_type={rule.value_type})")
    except Exception as e:
        print(f"Error parsing rules: {e}")
        sys.exit(1)

    # Get calculator config
    calculator_config = {}
    if 'indicators' in rules_config and 'avg_volume' in rules_config['indicators']:
        avg_vol_config = rules_config['indicators']['avg_volume']
        if isinstance(avg_vol_config, dict):
            calculator_config['avg_volume_window'] = avg_vol_config.get('window', 14)
        else:
            calculator_config['avg_volume_window'] = 14
    else:
        calculator_config['avg_volume_window'] = 14

    screener = ScreeningEngine(rules, calculator_config)

    # 6. Initialize data fetcher
    data_config = settings.get('data', {})
    fetcher = DataFetcher(
        period=data_config.get('period', '3mo'),
        interval=data_config.get('interval', '1d')
    )

    # 7. Fetch data for all tickers
    print(f"\nFetching data for {len(ticker_list)} tickers...")
    ticker_data = fetcher.fetch_batch(ticker_list)
    print("Data fetch complete.\n")

    # 8. Screen each ticker
    if args.verbose:
        print("Screening tickers against rules...\n")

    results: List[ScreeningResult] = []

    for ticker, data in ticker_data.items():
        if data['error']:
            # Data fetch failed - create error result
            results.append(ScreeningResult(
                ticker=ticker,
                passed=False,
                enriched_data={'ticker': ticker},
                failed_rules=[],
                error=data['error']
            ))
            continue

        try:
            # Get ticker metadata
            metadata = ticker_metadata.get(ticker, {})

            # Enrich ticker data with indicators
            enriched = screener.enrich_ticker_data(
                ticker,
                data['data'],
                data['info'],
                metadata
            )

            # Screen against rules
            result = screener.screen(enriched)
            results.append(result)

        except Exception as e:
            # Screening failed - create error result
            if args.verbose:
                print(f"Error screening {ticker}: {e}")
            results.append(ScreeningResult(
                ticker=ticker,
                passed=False,
                enriched_data={'ticker': ticker},
                failed_rules=[],
                error=str(e)
            ))

    # 9. Display results
    formatter = ResultFormatter()

    # Filter to only passing results
    passing_results = [r for r in results if r.passed]
    failed_results = [r for r in results if not r.passed and not r.error]
    error_results = [r for r in results if r.error]

    # Display passing results
    formatter.display_results(passing_results)

    # Display summary if enabled
    show_summary = settings.get('display', {}).get('show_summary', True)
    if show_summary and not args.no_summary:
        formatter.display_summary(
            total_tickers=len(results),
            passed=len(passing_results),
            failed=len(failed_results),
            errors=len(error_results)
        )

    if args.verbose:
        print("Stock Screener - Complete.")

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nScreening interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)
