# Claude Context - Trading Strategy Backtesting Framework

This document provides context for Claude Code AI assistants working on this project.

## Project Overview

This is a **Trading Strategy Backtesting Framework** for evaluating algorithmic trading strategies on historical stock data. It's designed for educational and research purposes, allowing users to test various technical analysis strategies on S&P 500 stocks.

**NOT a Stock Screener** - While the project originally started as a stock screener, it has evolved into a full-featured backtesting framework with 9 distinct trading strategies.

## Architecture

### Core Design Principles

1. **Unified Interface**: All strategies follow the same pattern
2. **Modular Components**: Indicators, data management, and strategies are separate
3. **Cached Data**: Historical OHLCV data is cached locally for fast repeated analysis
4. **Consistent Reporting**: All strategies use the unified `summary.py` module

### Directory Structure

```
fin/
├── src/
│   ├── analysis/          # Trading strategies (main focus)
│   │   ├── *.py           # 9 strategy implementations
│   │   └── summary.py     # Unified summary/reporting module
│   ├── data/              # Data management
│   │   ├── cache.py       # CacheManager for OHLCV data
│   │   ├── download_history.py  # Initial data download
│   │   ├── sp500_loader.py      # Ticker list loader (S&P 500, custom, all)
│   │   └── update_cache.py      # Update cached data
│   ├── indicators/        # Technical indicators
│   │   └── technical.py   # SMA, RSI, ATR, MACD, ADX, BBands, etc.
│   ├── engine/            # Legacy screener components (kept for compatibility)
│   ├── models/            # Legacy data models
│   └── presentation/      # Legacy formatters
├── data/
│   ├── sp500.csv          # S&P 500 ticker list
│   ├── custom_tickers.csv # Custom watchlist (user-editable)
│   └── cache/ohlcv/       # Cached historical price data
└── results/               # Exported CSV results
```

### Custom Ticker Lists

The framework supports three ticker sources:
1. **S&P 500**: `data/sp500.csv` - 500+ stocks
2. **Custom Watchlist**: `data/custom_tickers.csv` - User-defined tickers (TSLA, NVDA, COIN, etc.)
3. **All Combined**: Both lists merged (duplicates removed)

**Loading Functions (src/data/sp500_loader.py):**
```python
from src.data.sp500_loader import load_sp500_tickers, load_custom_tickers, load_all_tickers

# Load S&P 500 only
sp500 = load_sp500_tickers()  # Returns list of ticker symbols

# Load custom watchlist only
custom = load_custom_tickers()  # Returns list from custom_tickers.csv

# Load all tickers (S&P 500 + custom)
all_tickers = load_all_tickers()  # Returns combined unique list
```

**CLI Flags (used in download_history.py, update_cache.py, strategies):**
- `--sp500`: Use S&P 500 tickers only
- `--custom`: Use custom watchlist only
- `--all`: Use both lists combined
- `--tickers AAPL MSFT`: Use specific tickers

**Example Usage:**
```bash
# Download only custom watchlist
python -m src.data.download_history --custom

# Update all tickers (S&P 500 + custom)
python -m src.data.update_cache --all

# Run strategy on custom watchlist
python -m src.analysis.rsi_mean_reversion --custom
```

## Strategy Implementation Pattern

All strategies follow this standard structure:

```python
#!/usr/bin/env python3
"""
Strategy Name

Brief description of strategy logic and philosophy.
"""

# 1. Signal Detection Function
def detect_signals(df: pd.DataFrame, **params) -> List[int]:
    """Detects entry signals and returns list of indices."""
    signals = []
    # ... signal logic
    return signals

# 2. Outcome Calculation
def calculate_outcome(df: pd.DataFrame, entry_idx: int, **params) -> Optional[Dict]:
    """Calculates trade outcome (exit type, price, %, etc.)."""
    # ... exit logic (stop-loss, take-profit, etc.)
    return outcome_dict

# 3. Forward Analysis
def analyze_forward_days(df: pd.DataFrame, entry_idx: int, **params) -> Optional[Dict]:
    """Wrapper that adds metadata and verification metrics."""
    outcome = calculate_outcome(df, entry_idx, **params)
    # ... add entry metadata, verification metrics
    return analysis_dict

# 4. Run Analysis
def run_analysis(ticker: str, cache_manager: CacheManager, **params) -> pd.DataFrame:
    """Main analysis function for a single ticker."""
    df = cache_manager.get_ticker_data(ticker)
    # ... calculate indicators
    signals = detect_signals(df, **params)
    results = [analyze_forward_days(df, idx, **params) for idx in signals]
    return pd.DataFrame(results)

# 5. CLI Entry Point
def main():
    """CLI with argparse for parameters."""
    parser = argparse.ArgumentParser(...)
    # ... add arguments
    args = parser.parse_args()

    # Run on ticker(s)
    for ticker in tickers:
        results_df = run_analysis(ticker, cache_manager, **params)
        if show_individual:
            print_summary(results_df, ...)  # From summary.py

    # Aggregate results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        print_aggregate_summary(combined_df, ...)  # From summary.py
```

## Current Strategies

1. **rsi_mean_reversion.py** - RSI < 30 in uptrends
2. **atr_mean_reversion.py** - ATR-based dynamic stops
3. **ma_pullback.py** - SMA20 pullback in SMA200 uptrend
4. **breakout_momentum.py** - 20-day high breakouts with volume
5. **triple_filter.py** - RSI + MA + Volume combination
6. **macd_rsi_confluence.py** - MACD crossover + RSI confirmation
7. **bb_squeeze_breakout.py** - Bollinger Band squeeze detection
8. **atr_filtered_breakout.py** - False breakout pattern recognition
9. **sma200_crossover_analysis.py** - SMA200 crossover analysis

## Key Components

### CacheManager (src/data/cache.py)

```python
cache_manager = CacheManager(cache_dir='data/cache/ohlcv')
df = cache_manager.get_ticker_data('AAPL')  # Returns OHLCV DataFrame

# Check cache status
last_date = cache_manager.get_last_cached_date('AAPL')  # Returns datetime
gap_info = cache_manager.detect_gaps('AAPL')  # Returns gap information

# Incremental update (only fetches new data)
fetcher = DataFetcher(period='max', interval='1d')
results = cache_manager.batch_update(
    tickers=['AAPL', 'MSFT'],
    fetcher=fetcher,
    force_update=False  # Skip if already up-to-date
)
```

**Key Features:**
- Manages local Parquet cache of historical price data (compressed, fast columnar storage)
- **Incremental updates**: Only fetches data since last cached date
- Gap detection: Automatically identifies missing date ranges
- Metadata tracking: Stores last update time, date range, row count for each ticker
- Thread-safe operations with metadata locking
- Subsequent accesses are instant (reads from Parquet files)

**Incremental Update Workflow:**
1. `detect_gaps()` checks the last cached date for a ticker
2. `batch_update()` passes the last date to DataFetcher as start parameter
3. DataFetcher only downloads data from that date to today
4. New data is merged with existing cache (duplicates removed)
5. Metadata is updated with new date range

**For Weekly Updates:**
Use `python -m src.data.update_cache --sp500` to update all tickers incrementally. This is much faster than re-downloading all historical data.

### Summary Module (src/analysis/summary.py)

```python
from src.analysis.summary import print_summary, print_aggregate_summary

# Single ticker summary
print_summary(results_df, strategy_name="...", ticker="AAPL", ...)

# Multi-ticker aggregate summary
print_aggregate_summary(combined_df, strategy_name="...")
```

- **CRITICAL**: All new strategies MUST use this module for consistent output
- Provides win rate, avg return, risk/reward, exit type distribution
- Handles both single-ticker and aggregate reporting

### Technical Indicators (src/indicators/technical.py)

```python
from src.indicators.technical import (
    calculate_sma, calculate_rsi, calculate_atr,
    calculate_macd, calculate_adx, calculate_bbands
)

df['SMA200'] = calculate_sma(df, window=200)
df['RSI'] = calculate_rsi(df, window=14)
df['ATR'] = calculate_atr(df, window=14)
```

## Common Patterns

### Entry Signal Detection

- Always check for sufficient data: `if len(df) < required_period: return []`
- Skip NaN values: `if pd.isna(value): continue`
- Use lookback windows for trend confirmation
- Return list of integer indices: `signals.append(i)`

### Exit Logic

- Support multiple exit types: stop_loss, take_profit, rsi_exit, period_end
- Use bar structure heuristic when both SL/TP hit same day
- Always calculate verification metrics (min/max price in forward window)

### CLI Arguments

Standard arguments all strategies should support:
- `--ticker` / `--tickers` / `--sp500`
- `--export` for CSV output
- `--cache-dir` for custom cache location
- Strategy-specific parameters with sensible defaults

## Data Expectations

### Input DataFrame Format

All strategies expect OHLCV DataFrames with DatetimeIndex:

```python
df.index     # DatetimeIndex
df['Open']   # float64
df['High']   # float64
df['Low']    # float64
df['Close']  # float64
df['Volume'] # int64/float64
```

### Output DataFrame Format

All strategies should return DataFrames with these columns:

```python
results_df = pd.DataFrame([
    {
        'date': entry_date,
        'ticker': ticker,
        'strategy': 'strategy_name',
        'entry_price': float,
        # ... strategy-specific entry metrics
        'exit_type': str,      # stop_loss, take_profit, etc.
        'exit_day': int,       # Days held
        'exit_price': float,
        'exit_pct': float,     # Percentage return
        'is_winner': bool,
        # ... verification metrics (min/max prices)
    }
])
```

## Adding New Strategies

1. **Copy existing strategy** as template (e.g., `rsi_mean_reversion.py`)
2. **Update docstring** with strategy description and logic
3. **Implement signal detection** function
4. **Implement exit logic** with multiple exit types
5. **Add CLI arguments** with `argparse`
6. **Use summary module** for output: `print_summary()` and `print_aggregate_summary()`
7. **Test on single ticker** before running on S&P 500
8. **Update README.md** with new strategy description and example

## Testing Workflow

```bash
# 1. Test on single ticker with verbose output
python -m src.analysis.your_strategy --ticker AAPL --verbose

# 2. Test on multiple tickers
python -m src.analysis.your_strategy --tickers AAPL MSFT GOOGL

# 3. Run on S&P 500 with export
python -m src.analysis.your_strategy --sp500 --export results/your_strategy.csv
```

## Common Gotchas

1. **NaN Handling**: Always check for NaN in indicators before comparisons
2. **Index vs Position**: Use `.iloc[]` for integer positions, not `.loc[]`
3. **Forward Window**: Ensure sufficient data exists before analyzing forward period
4. **Column Names**: Stick to convention (SMA200, RSI, ATR, etc.)
5. **Summary Module**: Don't implement custom summaries - use `summary.py`

## Legacy Components

The following directories contain legacy code from the original screener:
- `src/engine/` - Rule-based screening engine (not used by strategies)
- `src/models/` - Data models for screener
- `src/presentation/formatter.py` - Legacy output formatting
- `config/` - YAML configs for screener (not strategies)

**These are kept for backward compatibility but are not used by the backtesting framework.**

## Performance Considerations

1. **Cache First**: Always run `download_history.py` once to populate cache
2. **Indicator Calculation**: Calculate indicators once per ticker, not per signal
3. **DataFrame Operations**: Use vectorized operations when possible
4. **S&P 500 Runs**: Expect 5-15 minutes for full S&P 500 backtest

## Code Style

- Use type hints: `def function(param: int) -> List[int]:`
- Docstrings for all functions
- Default parameter values in argparse
- Clear variable names: `entry_price`, `exit_type`, not `ep`, `et`
- Comments for complex logic only (code should be self-documenting)

## Git Workflow

- Development happens on feature branches
- Branch naming: `claude/feature-name-xxxxx`
- Commit messages: Clear, descriptive, explain "why" not just "what"
- Push to origin with: `git push -u origin branch-name`

## Resources

- **yfinance**: Historical data source
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **pandas_ta**: Some technical indicators (though we mostly use custom implementations)

## Questions to Ask When Uncertain

1. "Should this follow the standard strategy pattern?"
2. "Am I using the summary module correctly?"
3. "Does this need to be added to README.md?"
4. "Have I tested on a single ticker first?"
5. "Are all exit types handled (SL, TP, period_end, etc.)?"

## When Making Changes

- **Always read the existing code first** before suggesting changes
- **Maintain consistency** with existing strategies
- **Test thoroughly** on single ticker before S&P 500
- **Update documentation** (README.md, this file) when adding features
- **Use the summary module** - don't create custom reporting

---

**Last Updated**: 2025-12-25
**Project Status**: Active development - backtesting framework with 9 strategies
**Main Focus**: Adding new strategies, improving existing ones, analyzing results
