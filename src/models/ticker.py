from dataclasses import dataclass


@dataclass
class TickerMetadata:
    """
    Represents metadata for a stock ticker.

    This corresponds to the data stored in tickers.json.

    Attributes:
        ticker: The stock ticker symbol (e.g., 'AAPL')
        company_name: Full company name (e.g., 'Apple Inc.')
        sector: Business sector (e.g., 'Technology')
    """
    ticker: str
    company_name: str
    sector: str

    @classmethod
    def from_dict(cls, data: dict):
        """
        Create TickerMetadata from a dictionary.

        Args:
            data: Dictionary with ticker metadata

        Returns:
            TickerMetadata instance
        """
        return cls(
            ticker=data.get('ticker', ''),
            company_name=data.get('company_name', ''),
            sector=data.get('sector', 'Unknown')
        )

    def to_dict(self) -> dict:
        """
        Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            'ticker': self.ticker,
            'company_name': self.company_name,
            'sector': self.sector
        }
