import pandas as pd
from typing import List
from src.models.result import ScreeningResult


class ResultFormatter:
    """
    Formats screening results for console display using pandas.

    Uses M/B suffixes for large numbers and configures pandas for clean output.
    """

    def __init__(self):
        """Initialize formatter and configure pandas display options."""
        # Configure pandas for clean console display
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 20)
        pd.set_option('display.precision', 2)

    def format_number(self, value: float, decimal_places: int = 2) -> str:
        """
        Format numbers with M/B/T suffixes.

        Args:
            value: Number to format
            decimal_places: Number of decimal places

        Returns:
            Formatted string with suffix

        Examples:
            2400000000000 -> '2.40T'
            2400000000 -> '2.40B'
            75000000 -> '75.00M'
            150000 -> '150.00K'
            150 -> '150'
        """
        if value is None:
            return 'N/A'

        abs_value = abs(value)

        if abs_value >= 1_000_000_000_000:  # Trillions
            return f"{value / 1_000_000_000_000:.{decimal_places}f}T"
        elif abs_value >= 1_000_000_000:  # Billions
            return f"{value / 1_000_000_000:.{decimal_places}f}B"
        elif abs_value >= 1_000_000:  # Millions
            return f"{value / 1_000_000:.{decimal_places}f}M"
        elif abs_value >= 1_000:  # Thousands
            return f"{value / 1_000:.{decimal_places}f}K"
        else:
            return f"{value:.{decimal_places}f}"

    def format_percentage(self, value: float, decimal_places: int = 2) -> str:
        """
        Format percentage with + or - sign.

        Args:
            value: Percentage value
            decimal_places: Number of decimal places

        Returns:
            Formatted string

        Examples:
            2.5 -> '+2.50%'
            -1.3 -> '-1.30%'
            0.0 -> '0.00%'
        """
        if value is None:
            return 'N/A'

        if value > 0:
            return f"+{value:.{decimal_places}f}%"
        else:
            return f"{value:.{decimal_places}f}%"

    def format_currency(self, value: float, decimal_places: int = 2) -> str:
        """
        Format currency with $ sign.

        Args:
            value: Currency value
            decimal_places: Number of decimal places

        Returns:
            Formatted string

        Example:
            150.25 -> '$150.25'
        """
        if value is None:
            return 'N/A'

        return f"${value:.{decimal_places}f}"

    def create_results_dataframe(self, results: List[ScreeningResult]) -> pd.DataFrame:
        """
        Convert ScreeningResults to a display DataFrame.

        Creates a table with columns:
        | Name | Last | CHANGE | Price/(SMA200) | Volume | AVG VOLUME | MARKET CAP |

        Args:
            results: List of ScreeningResult objects

        Returns:
            Formatted pandas DataFrame
        """
        rows = []

        for result in results:
            ed = result.enriched_data

            # Build the row
            row = {
                'Stock': ed.get('ticker', ed.get('ticker', 'N/A')),
                'Last': self.format_currency(ed.get('price')),
                'CHANGE': self.format_percentage(ed.get('change')),
            }

            # Add Price/SMA200 ratio if available
            distance_sma200 = ed.get('distance_from_sma200')
            if distance_sma200 is not None:
                row['Price/(SMA200)'] = f"{distance_sma200:.3f}"
            else:
                row['Price/(SMA200)'] = 'N/A'

            # Add volume metrics
            row['Volume'] = self.format_number(ed.get('volume', 0))
            row['AVG VOLUME'] = self.format_number(ed.get('avg_volume', 0))

            # Add market cap
            row['MARKET CAP'] = self.format_number(ed.get('market_cap', 0))

            rows.append(row)

        return pd.DataFrame(rows)

    def display_results(self, results: List[ScreeningResult], show_count: bool = True):
        """
        Print formatted table to console.

        Args:
            results: List of ScreeningResult objects
            show_count: Whether to show result count header
        """
        if not results:
            print("\nNo tickers passed screening.\n")
            return

        if show_count:
            print(f"\nResults: {len(results)} ticker(s) passed screening\n")

        # Create and display DataFrame
        df = self.create_results_dataframe(results)

        # Use pandas to_string for clean console output
        print(df.to_string(index=False))
        print()  # Add blank line after table

    def display_summary(self, total_tickers: int, passed: int, failed: int, errors: int):
        """
        Display summary statistics.

        Args:
            total_tickers: Total number of tickers processed
            passed: Number that passed screening
            failed: Number that failed screening
            errors: Number with fetch/calculation errors
        """
        print("\n" + "=" * 50)
        print("SCREENING SUMMARY")
        print("=" * 50)
        print(f"Total Tickers Processed: {total_tickers}")
        print(f"Passed Screening:        {passed}")
        print(f"Failed Screening:        {failed}")
        print(f"Errors:                  {errors}")
        print("=" * 50 + "\n")
