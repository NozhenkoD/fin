from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class ScreeningResult:
    """
    Represents the result of screening a single ticker against rules.

    Attributes:
        ticker: The stock ticker symbol
        passed: Whether the ticker passed all screening rules
        enriched_data: Dictionary containing all calculated values (price, volume, indicators, etc.)
        failed_rules: List of rule descriptions that failed (for debugging)
        error: Optional error message if data fetch/calculation failed
    """
    ticker: str
    passed: bool
    enriched_data: Dict
    failed_rules: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def __repr__(self):
        if self.error:
            return f"ScreeningResult({self.ticker}: ERROR - {self.error})"
        elif self.passed:
            return f"ScreeningResult({self.ticker}: PASSED)"
        else:
            return f"ScreeningResult({self.ticker}: FAILED - {len(self.failed_rules)} rules)"
