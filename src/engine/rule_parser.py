import re
from typing import List, Dict
from src.models.rule import Rule


class RuleParser:
    """
    Parses and validates screening rules.
    Extracts required indicators from rules.
    """

    def __init__(self):
        self.valid_fields = Rule.VALID_FIELDS
        self.valid_operators = Rule.VALID_OPERATORS

    def parse_rule(self, rule_dict: dict) -> Rule:
        """
        Validate and convert a dictionary to a Rule object.

        Args:
            rule_dict: Dictionary with 'field', 'operator', 'value' keys

        Returns:
            Rule object

        Raises:
            ValueError: If rule is invalid
        """
        required_keys = {'field', 'operator', 'value'}
        if not all(key in rule_dict for key in required_keys):
            raise ValueError(
                f"Rule must contain keys: {required_keys}. "
                f"Got: {set(rule_dict.keys())}"
            )

        return Rule(
            field=rule_dict['field'],
            operator=rule_dict['operator'],
            value=rule_dict['value']
        )

    def parse_ruleset(self, rules: List[dict]) -> List[Rule]:
        """
        Parse multiple rules with validation.

        Args:
            rules: List of rule dictionaries

        Returns:
            List of Rule objects

        Raises:
            ValueError: If any rule is invalid
        """
        if not rules:
            raise ValueError("Rules list cannot be empty")

        parsed_rules = []
        for i, rule_dict in enumerate(rules):
            try:
                parsed_rule = self.parse_rule(rule_dict)
                parsed_rules.append(parsed_rule)
            except ValueError as e:
                raise ValueError(f"Error in rule {i + 1}: {str(e)}")

        return parsed_rules

    def extract_required_indicators(self, rules: List[Rule]) -> Dict[str, List[int]]:
        """
        Analyze rules to determine which indicators need to be calculated.

        Args:
            rules: List of Rule objects

        Returns:
            Dictionary with indicator types and their windows:
            {'sma': [50, 200], 'ema': [20], 'avg_volume': [14]}
        """
        indicators = {
            'sma': set(),
            'ema': set()
        }

        for rule in rules:
            # Check if rule.value references an indicator
            if rule.value_type == 'indicator':
                indicator_name = str(rule.value)

                if indicator_name.startswith('SMA'):
                    # Extract window from 'SMA200' -> 200
                    window = self._extract_window(indicator_name, 'SMA')
                    if window:
                        indicators['sma'].add(window)

                elif indicator_name.startswith('EMA'):
                    # Extract window from 'EMA50' -> 50
                    window = self._extract_window(indicator_name, 'EMA')
                    if window:
                        indicators['ema'].add(window)

            # Check if the field itself is comparing against an indicator field
            # (for future expansion)
            if rule.field in ['sma', 'ema']:
                # This would handle rules like: field='sma', operator='>', value=150
                # For now, we'll skip this as it's not in the initial spec
                pass

        # Convert sets to sorted lists
        result = {
            'sma': sorted(list(indicators['sma'])),
            'ema': sorted(list(indicators['ema']))
        }

        return result

    def _extract_window(self, indicator_name: str, prefix: str) -> int:
        """
        Extract the window size from an indicator name.

        Args:
            indicator_name: String like 'SMA200' or 'EMA50'
            prefix: The indicator prefix ('SMA' or 'EMA')

        Returns:
            Window size as integer, or None if invalid
        """
        try:
            # Remove prefix and parse remaining as integer
            window_str = indicator_name[len(prefix):]
            return int(window_str)
        except (ValueError, IndexError):
            return None

    def parse_cli_rule_string(self, rule_string: str) -> dict:
        """
        Parse a CLI rule string into a rule dictionary.

        Examples:
            "price > 100" -> {'field': 'price', 'operator': '>', 'value': 100}
            "volume >= 1000000" -> {'field': 'volume', 'operator': '>=', 'value': 1000000}

        Args:
            rule_string: Rule in string format

        Returns:
            Rule dictionary

        Raises:
            ValueError: If rule string is invalid
        """
        # Pattern: field operator value
        # Supports: >, <, ==, >=, <=, !=
        pattern = r'^\s*(\w+)\s*(>=|<=|!=|>|<|==)\s*(.+)\s*$'
        match = re.match(pattern, rule_string)

        if not match:
            raise ValueError(
                f"Invalid rule format: '{rule_string}'. "
                f"Expected format: 'field operator value' (e.g., 'price > 100')"
            )

        field = match.group(1).strip()
        operator = match.group(2).strip()
        value_str = match.group(3).strip()

        # Try to convert value to number, otherwise keep as string
        try:
            # Try integer first
            if '.' not in value_str:
                value = int(value_str)
            else:
                value = float(value_str)
        except ValueError:
            # Keep as string (could be indicator reference like 'SMA200')
            value = value_str

        return {
            'field': field,
            'operator': operator,
            'value': value
        }
