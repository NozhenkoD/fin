# Stock Screener

A modular Python CLI Stock Screener that filters US stocks based on dynamic user-defined rules.

## Features

- **Dynamic Rule Configuration**: Define custom screening rules via YAML config files or CLI
- **Technical Indicators**: Supports SMA, EMA, and customizable moving average windows
- **Flexible Data Sources**: Currently uses yfinance, architected for easy WebSocket integration
- **Clean CLI Output**: Pandas-based table formatting with M/B/T suffixes for large numbers
- **Error Resilient**: Per-ticker error handling - failures don't stop the screening process
- **Modular Architecture**: Separation of data, logic, and presentation layers

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create your `data/tickers.json` file with your ticker list:

**Option A: Automatically fetch all NASDAQ tickers**

```bash
# RECOMMENDED: Skip metadata to avoid rate limits (much faster!)
python -m src.utils.ticker_fetcher --skip-metadata

# Or fetch all NASDAQ with metadata (slow, may hit rate limits)
python -m src.utils.ticker_fetcher --delay 1.0

# Test with just 50 tickers first
python -m src.utils.ticker_fetcher --limit 50 --skip-metadata

# Include both NASDAQ and NYSE
python -m src.utils.ticker_fetcher --nasdaq --nyse --skip-metadata
```

**Option B: Manually create the file**

```json
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
```

## Usage

### Basic Usage

Run with default configuration:
```bash
python src/main.py
```

### Command-Line Options

**Override ticker list:**
```bash
python src/main.py --tickers AAPL MSFT GOOGL TSLA
```

**Use custom rules config:**
```bash
python src/main.py --config config/custom_rules.yaml
```

**Add rules from command line:**
```bash
python src/main.py --rule "price > 150" --rule "volume > 50000000"
```

**Combine options:**
```bash
python src/main.py --tickers NVDA AMD --rule "change > 5"
```

**Verbose output:**
```bash
python src/main.py --verbose
```

**Disable summary:**
```bash
python src/main.py --no-summary
```

### Full CLI Reference

```
usage: main.py [-h] [--tickers TICKER [TICKER ...]] [--config PATH]
               [--rule RULE] [--ticker-file PATH] [--settings PATH]
               [--no-summary] [--verbose]

Stock Screener - Filter US stocks based on dynamic rules

optional arguments:
  -h, --help            show this help message and exit
  --tickers TICKER [TICKER ...]
                        List of ticker symbols to screen
  --config PATH         Path to rules config file (default: config/rules.yaml)
  --rule RULE           Add a rule (can be used multiple times)
  --ticker-file PATH    Path to tickers.json (default: data/tickers.json)
  --settings PATH       Path to settings file (default: config/settings.yaml)
  --no-summary          Disable summary statistics
  --verbose             Enable verbose output
```

## Configuration

### Default Rules (config/rules.yaml)

The default configuration includes three rules:

```yaml
rules:
  # Price change must be greater than +2%
  - field: change
    operator: '>'
    value: 2.0

  # Current price must be above 200-day SMA
  - field: price
    operator: '>'
    value: SMA200

  # Market cap must be greater than $2 billion
  - field: market_cap
    operator: '>'
    value: 2000000000

indicators:
  sma: [50, 200]
  ema: [20]
  avg_volume:
    window: 14
```

### Rule Format

Each rule consists of three components:

- **field**: The metric to evaluate
  - `price` - Current stock price
  - `volume` - Current volume
  - `market_cap` - Market capitalization
  - `change` - Percentage price change
  - `avg_volume` - Average volume over window

- **operator**: Comparison operator
  - `>`, `<`, `==`, `>=`, `<=`, `!=`

- **value**: Comparison value
  - Numeric literal (e.g., `100`, `2000000000`)
  - Indicator reference (e.g., `SMA200`, `EMA50`)

### Creating Custom Rules

Create a new YAML file (e.g., `config/aggressive_rules.yaml`):

```yaml
rules:
  - field: change
    operator: '>'
    value: 5.0

  - field: price
    operator: '>'
    value: SMA50

  - field: volume
    operator: '>'
    value: 10000000

  - field: market_cap
    operator: '>'
    value: 5000000000  # 5B

indicators:
  sma: [50, 200]
  ema: [20, 50]
  avg_volume:
    window: 30
```

Run with:
```bash
python src/main.py --config config/aggressive_rules.yaml
```

## Output Format

The screener displays results in a clean table:

```
Results: 12 ticker(s) passed screening

Name                 Last      CHANGE    Price/(SMA200)    Volume     AVG VOLUME    MARKET CAP
Apple Inc.           $150.25   +2.50%    1.034             75.00M     68.00M        2.40T
Microsoft Corp.      $378.90   +3.20%    1.012             42.50M     38.20M        2.82T
...

==================================================
SCREENING SUMMARY
==================================================
Total Tickers Processed: 50
Passed Screening:        12
Failed Screening:        35
Errors:                  3
==================================================
```

## Architecture

### Modular Three-Layer Design

```
┌─────────────────────────────────────────┐
│         Presentation Layer              │
│   (formatter.py - Display & Formatting) │
└─────────────────────────────────────────┘
                    ↑
┌─────────────────────────────────────────┐
│           Engine Layer                  │
│  (screener.py, calculator.py, rules)    │
│       Stateless Processing Logic        │
└─────────────────────────────────────────┘
                    ↑
┌─────────────────────────────────────────┐
│          Data Layer                     │
│    (fetcher.py - yfinance Adapter)      │
└─────────────────────────────────────────┘
```

### Key Design Principles

1. **Stateless Engine**: The `ScreeningEngine` has no internal state, making it compatible with both batch and streaming data sources

2. **DataFrame Interface**: All processing uses pandas DataFrames as the universal data format

3. **Modular Components**: Each layer can be modified independently without affecting others

4. **WebSocket-Ready**: Architecture designed for easy integration with real-time data streams in Iteration 2

## Project Structure

```
fin/
├── config/
│   ├── rules.yaml              # Default screening rules
│   └── settings.yaml           # App settings
├── data/
│   └── tickers.json            # Ticker metadata (user-provided)
├── src/
│   ├── main.py                 # Entry point
│   ├── data/
│   │   └── fetcher.py          # Data acquisition layer
│   ├── engine/
│   │   ├── calculator.py       # Technical indicator calculations
│   │   ├── rule_parser.py      # Rule validation and parsing
│   │   └── screener.py         # Core screening logic
│   ├── models/
│   │   ├── rule.py             # Rule data model
│   │   ├── result.py           # Screening result model
│   │   └── ticker.py           # Ticker metadata model
│   ├── presentation/
│   │   └── formatter.py        # Output formatting
│   └── utils/
│       ├── cli_parser.py       # CLI argument parsing
│       └── config_loader.py    # Config file loading
├── requirements.txt
└── README.md
```

## Supported Technical Indicators

- **SMA** (Simple Moving Average): `SMA50`, `SMA200`, etc.
- **EMA** (Exponential Moving Average): `EMA20`, `EMA50`, etc.
- **Average Volume**: Configurable window (default: 14 days)
- **Price Change**: Percentage change from previous close
- **Distance from SMA/EMA**: Ratio of price to moving average

### Important: Data Requirements

To calculate technical indicators, you need sufficient historical data:

- **SMA50/EMA50**: Requires at least ~50 trading days (~2.5 months)
- **SMA200/EMA200**: Requires at least ~200 trading days (~10 months)

The default data period is set to **1 year** (`1y`) in `config/settings.yaml` to support SMA200. If you use shorter-term indicators (like SMA20), you can reduce this to `3mo` for faster data fetching.

If you see warnings like "Cannot calculate SMA200: only 65 days of data", increase the `period` setting in `config/settings.yaml`.

## Error Handling

The screener is resilient to individual ticker failures:

- **Per-ticker error handling**: Failures for one ticker don't stop the entire batch
- **Graceful degradation**: Missing data fields are handled appropriately
- **Clear error reporting**: Summary shows which tickers had errors

## Ticker Fetcher Utility

The project includes a utility to automatically fetch all tickers from major exchanges and save them to `data/tickers.json` with company metadata.

### Usage

```bash
# RECOMMENDED: Fetch all NASDAQ tickers without metadata (fast, no rate limits!)
python -m src.utils.ticker_fetcher --skip-metadata

# Fetch with company names/sectors (slow, may hit rate limits)
python -m src.utils.ticker_fetcher --delay 1.0

# Test with limited tickers first
python -m src.utils.ticker_fetcher --limit 50 --skip-metadata

# Include multiple exchanges
python -m src.utils.ticker_fetcher --nasdaq --nyse --skip-metadata

# Custom output file
python -m src.utils.ticker_fetcher --output data/my_tickers.json --skip-metadata
```

### Options

- `--nasdaq` - Include NASDAQ tickers (default: enabled)
- `--nyse` - Include NYSE tickers
- `--amex` - Include AMEX tickers
- `--no-nasdaq` - Exclude NASDAQ tickers
- `--output PATH` - Output file path (default: data/tickers.json)
- `--delay SECONDS` - Delay between API calls (default: 0.5, increase to 1.0+ if rate limited)
- `--limit N` - Limit to first N tickers (for testing)
- `--skip-metadata` - Skip fetching company names/sectors (RECOMMENDED for large lists)

### Notes

- **Data source**: Fetches ticker symbols from official NASDAQ FTP site (ftp://ftp.nasdaqtrader.com)
- **Rate limiting**: yfinance has strict rate limits when fetching metadata (company names/sectors)
  - **With `--skip-metadata`**: ~10 seconds for 3,000+ tickers (FAST!)
  - **Without `--skip-metadata`**: 30-60 minutes, high risk of rate limit errors
- **Recommendation**: Use `--skip-metadata` for all tickers, then manually add company names for specific stocks you care about
- **Interruption**: Press Ctrl+C to stop; partial results will be saved
- **Testing**: Use `--limit 50 --skip-metadata` to test first

### Output Format

The utility saves tickers in the required format:

```json
{
  "AAPL": {
    "ticker": "AAPL",
    "company_name": "Apple Inc.",
    "sector": "Technology"
  },
  "MSFT": {
    "ticker": "MSFT",
    "company_name": "Microsoft Corporation",
    "sector": "Technology"
  }
}
```

## Future Roadmap (Iteration 2)

The architecture is designed for minimal refactoring when adding real-time features:

- **WebSocket Streaming**: Add `src/data/websocket_stream.py` adapter
- **Rolling Buffers**: Maintain per-ticker DataFrame buffers for real-time calculations
- **Live Screening**: Screen stocks as new data arrives
- **No Core Changes**: Engine, calculator, and formatter layers remain unchanged

## Examples

### Conservative Growth Stocks
```yaml
rules:
  - field: change
    operator: '>'
    value: 1.0
  - field: price
    operator: '>'
    value: SMA200
  - field: market_cap
    operator: '>'
    value: 10000000000  # 10B+
```

### High Volume Momentum
```yaml
rules:
  - field: change
    operator: '>'
    value: 3.0
  - field: volume
    operator: '>'
    value: 50000000
  - field: price
    operator: '>'
    value: SMA50
```

### Large Cap Value
```yaml
rules:
  - field: market_cap
    operator: '>'
    value: 100000000000  # 100B+
  - field: price
    operator: '<'
    value: SMA200
```

## Contributing

This project follows a modular architecture. When adding features:

1. Keep the stateless design of the screening engine
2. Use DataFrames as the universal data interface
3. Add new indicators to `calculator.py` as pure functions
4. Extend rule fields in `Rule.VALID_FIELDS` as needed

## License

MIT License

## Support

For issues and questions, please refer to the documentation or create an issue in the repository.
