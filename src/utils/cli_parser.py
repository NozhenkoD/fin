import argparse
from typing import List, Optional


class CLIParser:
    """
    Parses command-line arguments for the stock screener.

    Supports:
    - Custom ticker lists
    - Custom rule files
    - Rule overrides
    - Display options
    """

    def __init__(self):
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser."""
        parser = argparse.ArgumentParser(
            description='Stock Screener - Filter US stocks based on dynamic rules',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Use default config
  python src/main.py

  # Override tickers
  python src/main.py --tickers AAPL MSFT GOOGL

  # Override rules (replaces config rules)
  python src/main.py --rule "price > 150" --rule "volume > 50000000"

  # Use custom config
  python src/main.py --config config/aggressive_rules.yaml

  # Combine: custom tickers + default rules
  python src/main.py --tickers TSLA NVDA AMD
            """
        )

        # Ticker selection
        parser.add_argument(
            '--tickers',
            nargs='+',
            metavar='TICKER',
            help='List of ticker symbols to screen (overrides tickers.json)'
        )

        # Config file
        parser.add_argument(
            '--config',
            type=str,
            metavar='PATH',
            default='config/rules.yaml',
            help='Path to rules config file (default: config/rules.yaml)'
        )

        # Rule overrides
        parser.add_argument(
            '--rule',
            action='append',
            metavar='RULE',
            dest='rules',
            help='Add a rule (can be used multiple times). '
                 'Format: "field operator value" (e.g., "price > 100")'
        )

        # Ticker file
        parser.add_argument(
            '--ticker-file',
            type=str,
            metavar='PATH',
            default='data/tickers.json',
            help='Path to tickers.json file (default: data/tickers.json)'
        )

        # Settings file
        parser.add_argument(
            '--settings',
            type=str,
            metavar='PATH',
            default='config/settings.yaml',
            help='Path to settings file (default: config/settings.yaml)'
        )

        # Display options
        parser.add_argument(
            '--no-summary',
            action='store_true',
            help='Disable summary statistics display'
        )

        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Enable verbose output'
        )

        return parser

    def parse_args(self, args: Optional[List[str]] = None):
        """
        Parse command-line arguments.

        Args:
            args: Optional list of arguments (for testing). If None, uses sys.argv

        Returns:
            Namespace with parsed arguments
        """
        return self.parser.parse_args(args)

    def print_help(self):
        """Print help message."""
        self.parser.print_help()
