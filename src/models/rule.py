from dataclasses import dataclass
from typing import Union


@dataclass
class Rule:
    """
    Represents a screening rule for stock filtering.

    Attributes:
        field: The field to evaluate ('price', 'volume', 'market_cap', 'change', etc.)
        operator: The comparison operator ('>', '<', '==', '>=', '<=', '!=')
        value: The value to compare against (number or indicator name like 'SMA200')
        value_type: Type of value - 'literal' for numbers, 'indicator' for references
    """
    field: str
    operator: str
    value: Union[float, int, str]
    value_type: str = 'literal'

    VALID_FIELDS = {
        'price', 'volume', 'market_cap', 'change',
        'avg_volume', 'sma', 'ema'
    }

    VALID_OPERATORS = {'>', '<', '==', '>=', '<=', '!='}

    def __post_init__(self):
        """Validate rule fields after initialization."""
        # Validate operator
        if self.operator not in self.VALID_OPERATORS:
            raise ValueError(
                f"Invalid operator '{self.operator}'. "
                f"Must be one of: {', '.join(self.VALID_OPERATORS)}"
            )

        # Validate field
        if self.field not in self.VALID_FIELDS:
            raise ValueError(
                f"Invalid field '{self.field}'. "
                f"Must be one of: {', '.join(self.VALID_FIELDS)}"
            )

        # Auto-detect value_type if it's a string reference to an indicator
        if isinstance(self.value, str):
            # Check if it's an indicator reference (e.g., 'SMA200', 'EMA50')
            if any(self.value.startswith(ind) for ind in ['SMA', 'EMA']):
                self.value_type = 'indicator'
            else:
                # If it's a string but not an indicator, try to convert to float
                try:
                    self.value = float(self.value)
                    self.value_type = 'literal'
                except ValueError:
                    raise ValueError(
                        f"Invalid value '{self.value}'. Must be a number or "
                        f"indicator reference (e.g., 'SMA200')"
                    )
        else:
            self.value_type = 'literal'

    def __repr__(self):
        return f"Rule({self.field} {self.operator} {self.value})"
