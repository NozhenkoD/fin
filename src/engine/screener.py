import pandas as pd
import numpy as np
from typing import List, Dict
from src.models.rule import Rule
from src.models.result import ScreeningResult
from src.engine import calculator


class ScreeningEngine:
    """
    Stateless screening engine that evaluates tickers against rules.

    Key design: This class has NO internal state about ticker data.
    It only holds rules and configuration, making it compatible with
    both batch processing and real-time streaming.
    """

    def __init__(self, rules: List[Rule], calculator_config: dict = None):
        """
        Initialize the screening engine.

        Args:
            rules: List of Rule objects to evaluate
            calculator_config: Configuration for calculators (e.g., avg_volume window)
        """
        self.rules = rules
        self.config = calculator_config or {}
        self.avg_volume_window = self.config.get('avg_volume_window', 14)

    def enrich_ticker_data(
        self,
        ticker: str,
        df: pd.DataFrame,
        info: dict,
        ticker_metadata: dict = None
    ) -> dict:
        """
        Add calculated indicators to ticker data.

        This function is the key to the WebSocket-ready design:
        it works on ANY DataFrame, whether from batch fetch or streaming buffer.

        Args:
            ticker: Ticker symbol
            df: OHLCV DataFrame (from yfinance or streaming buffer)
            info: Ticker metadata dictionary (market cap, etc.)
            ticker_metadata: Optional metadata from tickers.json

        Returns:
            Dictionary with all enriched data:
            {
                'ticker': 'AAPL',
                'company_name': 'Apple Inc.',
                'sector': 'Technology',
                'price': 150.25,
                'change': 2.5,
                'volume': 75000000,
                'avg_volume': 68000000,
                'market_cap': 2400000000000,
                'SMA50': 148.20,
                'SMA200': 145.30,
                'distance_from_sma200': 1.034,
                ...
            }
        """
        enriched = {
            'ticker': ticker,
            'company_name': ticker_metadata.get('company_name', ticker) if ticker_metadata else ticker,
            'sector': ticker_metadata.get('sector', 'Unknown') if ticker_metadata else 'Unknown'
        }

        # Get basic price and volume data
        enriched['price'] = calculator.get_latest_price(df)
        enriched['volume'] = calculator.get_latest_volume(df)

        # Calculate price change
        enriched['change'] = calculator.calculate_price_change(df)

        # Calculate average volume
        enriched['avg_volume'] = calculator.calculate_avg_volume(df, self.avg_volume_window)

        # Get market cap from info
        enriched['market_cap'] = info.get('marketCap', None)

        # Determine which indicators to calculate based on rules
        required_indicators = self._get_required_indicators()
        print(f"[DEBUG] {ticker}: Required indicators: {required_indicators}")

        # Calculate required SMAs
        for window in required_indicators.get('sma', []):
            sma_key = f'SMA{window}'
            sma_series = calculator.calculate_sma(df, window)
            print(f"[DEBUG] {ticker}: Calculating {sma_key}, series empty: {sma_series.empty}, df length: {len(df)}")

            if not sma_series.empty:
                sma_value = sma_series.iloc[-1]
                # Check for NaN (pandas returns np.nan for missing values)
                if pd.isna(sma_value):
                    print(f"[DEBUG] {ticker}: {sma_key} is NaN (insufficient data: {len(df)} days < {window} days required)")
                    enriched[sma_key] = None
                else:
                    print(f"[DEBUG] {ticker}: {sma_key} = {float(sma_value)}")
                    enriched[sma_key] = float(sma_value)

                    # Calculate distance ratio
                    if enriched['price'] is not None:
                        distance_key = f'distance_from_sma{window}'
                        enriched[distance_key] = calculator.calculate_distance_from_sma(
                            enriched['price'],
                            float(sma_value)
                        )
            else:
                print(f"[DEBUG] {ticker}: WARNING - Cannot calculate {sma_key}: only {len(df)} days of data, need at least {window} days")
                print(f"[DEBUG] {ticker}: Increase the data period in settings.yaml (currently may be too short for {sma_key})")

        # Calculate required EMAs
        for window in required_indicators.get('ema', []):
            ema_key = f'EMA{window}'
            ema_series = calculator.calculate_ema(df, window)

            if not ema_series.empty:
                ema_value = ema_series.iloc[-1]
                # Check for NaN (pandas returns np.nan for missing values)
                if pd.isna(ema_value):
                    enriched[ema_key] = None
                else:
                    enriched[ema_key] = float(ema_value)

                    # Calculate distance ratio
                    if enriched['price'] is not None:
                        distance_key = f'distance_from_ema{window}'
                        enriched[distance_key] = calculator.calculate_distance_from_ema(
                            enriched['price'],
                            float(ema_value)
                        )

        return enriched

    def evaluate_rule(self, enriched_data: dict, rule: Rule) -> bool:
        """
        Evaluate a single rule against enriched data.

        Args:
            enriched_data: Dictionary with all calculated values
            rule: Rule object to evaluate

        Returns:
            True if rule passes, False otherwise
        """
        ticker = enriched_data.get('ticker', 'UNKNOWN')

        # Get the field value
        field_value = enriched_data.get(rule.field)

        if field_value is None:
            # Missing data fails the rule
            print(f"[DEBUG] {ticker}: Rule {rule} FAILED - field '{rule.field}' is None")
            return False

        # Resolve the comparison value
        if rule.value_type == 'indicator':
            # rule.value is something like 'SMA200'
            # Look it up in enriched_data
            compare_value = enriched_data.get(rule.value)
            if compare_value is None:
                # Missing indicator fails the rule
                print(f"[DEBUG] {ticker}: Rule {rule} FAILED - indicator '{rule.value}' not found in enriched_data")
                print(f"[DEBUG] {ticker}: Available keys: {list(enriched_data.keys())}")
                return False
        else:
            # rule.value is a literal number
            compare_value = rule.value

        # Evaluate the operator
        try:
            result = False
            if rule.operator == '>':
                result = field_value > compare_value
            elif rule.operator == '<':
                result = field_value < compare_value
            elif rule.operator == '==':
                result = field_value == compare_value
            elif rule.operator == '>=':
                result = field_value >= compare_value
            elif rule.operator == '<=':
                result = field_value <= compare_value
            elif rule.operator == '!=':
                result = field_value != compare_value
            else:
                # Should not happen due to validation, but handle it
                result = False

            status = "PASSED" if result else "FAILED"
            print(f"[DEBUG] {ticker}: Rule {rule} {status} - {field_value} {rule.operator} {compare_value} = {result}")
            return result

        except (TypeError, ValueError) as e:
            # Comparison failed (e.g., comparing incompatible types)
            print(f"[DEBUG] {ticker}: Rule {rule} FAILED - Comparison error: {e}")
            return False

    def screen(self, enriched_data: dict) -> ScreeningResult:
        """
        Run all rules against enriched data.

        This is completely stateless - just evaluates rules on the provided data.

        Args:
            enriched_data: Dictionary with all calculated values

        Returns:
            ScreeningResult with passed/failed status and details
        """
        ticker = enriched_data.get('ticker', 'UNKNOWN')
        failed_rules = []

        # Evaluate each rule
        for rule in self.rules:
            if not self.evaluate_rule(enriched_data, rule):
                failed_rules.append(str(rule))

        # All rules must pass (implicit AND logic)
        passed = len(failed_rules) == 0

        return ScreeningResult(
            ticker=ticker,
            passed=passed,
            enriched_data=enriched_data,
            failed_rules=failed_rules
        )

    def _get_required_indicators(self) -> Dict[str, List[int]]:
        """
        Determine which indicators need to be calculated based on rules.

        Returns:
            Dictionary like {'sma': [50, 200], 'ema': [20]}
        """
        indicators = {
            'sma': set(),
            'ema': set()
        }

        print(f"[DEBUG] _get_required_indicators: Examining {len(self.rules)} rules")
        for rule in self.rules:
            print(f"[DEBUG] _get_required_indicators: Rule {rule}, value_type={rule.value_type}, value={rule.value}")
            # Check if rule.value references an indicator
            if rule.value_type == 'indicator':
                indicator_name = str(rule.value)
                print(f"[DEBUG] _get_required_indicators: Found indicator reference: {indicator_name}")

                if indicator_name.startswith('SMA'):
                    try:
                        window = int(indicator_name[3:])
                        indicators['sma'].add(window)
                        print(f"[DEBUG] _get_required_indicators: Added SMA window: {window}")
                    except ValueError as e:
                        print(f"[DEBUG] _get_required_indicators: Failed to parse SMA window from '{indicator_name}': {e}")

                elif indicator_name.startswith('EMA'):
                    try:
                        window = int(indicator_name[3:])
                        indicators['ema'].add(window)
                        print(f"[DEBUG] _get_required_indicators: Added EMA window: {window}")
                    except ValueError as e:
                        print(f"[DEBUG] _get_required_indicators: Failed to parse EMA window from '{indicator_name}': {e}")

        result = {
            'sma': sorted(list(indicators['sma'])),
            'ema': sorted(list(indicators['ema']))
        }
        print(f"[DEBUG] _get_required_indicators: Final result: {result}")
        return result
