# Trading Strategy Backtesting Framework

A Python-based backtesting framework for evaluating algorithmic trading strategies on historical stock data. Test multiple proven strategies on S&P 500 stocks with configurable parameters and comprehensive performance metrics.

## Features

- **9 Battle-Tested Strategies**: Mean reversion, momentum, breakout, and multi-indicator approaches
- **Flexible Backtesting**: Run on single tickers, custom lists, or entire S&P 500
- **Performance Metrics**: Win rate, avg return, risk/reward, hold times, exit type distribution
- **Cached Data**: Local OHLCV data cache for fast repeated analysis
- **CSV Export**: Export results for further analysis in Excel/Python
- **Unified Summary**: Consistent reporting across all strategies

## Installation

```bash
# Clone repository
git clone <repository-url>
cd fin

# Install dependencies
pip install -r requirements.txt

# Download historical data (one-time setup)
python -m src.data.download_history
```

## Available Strategies

Each strategy has its own detailed documentation with usage examples, parameter guides, and optimization tips.

### 1. [RSI Mean Reversion](src/analysis/rsi_mean_reversion/README.md)
Buys oversold stocks (RSI < 30) in strong uptrends (above SMA200). High win rate strategy for range-bound markets.
- **Win Rate**: 60-70%
- **Avg Hold**: 5-15 days
- **Type**: Mean Reversion

### 2. [ATR Mean Reversion](src/analysis/atr_mean_reversion/README.md)
Volatility-adaptive mean reversion using ATR-based dynamic stops. Better risk management for volatile stocks.
- **Win Rate**: Similar to RSI MR, better R:R
- **Avg Hold**: 10-15 days
- **Type**: Adaptive Mean Reversion

### 3. [MA Pullback](src/analysis/ma_pullback/README.md)
Buys pullbacks to SMA20 in stocks above SMA200. One of the most reliable swing trading strategies.
- **Win Rate**: 60-70%
- **Avg Hold**: 5-10 days
- **Type**: Mean Reversion / Swing Trading

### 4. [Breakout Momentum](src/analysis/breakout_momentum/README.md)
Trend-following strategy that buys strength on 20-day high breakouts with volume confirmation.
- **Win Rate**: 52-58%
- **Avg Hold**: 10-20 days
- **Type**: Trend Following / Momentum

### 5. [Triple Filter](src/analysis/triple_filter/README.md)
Combines RSI, moving averages, and volume for high-probability setups.
- **Win Rate**: 65-75%
- **Avg Hold**: 5-10 days
- **Type**: Multi-Filter Swing Trading

### 6. [MACD + RSI Confluence](src/analysis/macd_rsi_confluence/README.md)
Requires both MACD bullish crossover and RSI confirmation. Filters false MACD signals.
- **Win Rate**: 55-60%
- **Avg Hold**: 10-15 days
- **Type**: Momentum Confirmation

### 7. [Bollinger Band Squeeze Breakout](src/analysis/bb_squeeze_breakout/README.md)
Detects volatility contraction and enters on breakout. Catches explosive moves after consolidation.
- **Win Rate**: 50-55%
- **Avg Hold**: 15-25 days
- **Type**: Volatility Breakout

### 8. [ATR-Filtered False Breakout](src/analysis/atr_filtered_breakout/README.md)
Detects false breakouts below swing lows and enters expecting mean reversion. Advanced pattern recognition.
- **Win Rate**: 50-60%
- **Avg Hold**: 10-15 days
- **Type**: Contrarian Mean Reversion

### 9. [SMA200 Crossover Analysis](src/analysis/sma200_crossover_analysis/README.md)
Analyzes what happens when price crosses above/below SMA200. Simple trend-following approach.
- **Win Rate**: 50-65% (varies by type)
- **Avg Hold**: Variable
- **Type**: Trend Following

## Quick Start Examples

### Single Ticker Analysis
```bash
# Run any strategy on a single stock
python -m src.analysis.rsi_mean_reversion --ticker AAPL
python -m src.analysis.atr_mean_reversion --ticker NVDA
python -m src.analysis.ma_pullback --ticker TSLA
```

### Show Detailed Signal Table
```bash
# New feature: View entry/exit dates, hold periods, and returns
python -m src.analysis.atr_mean_reversion \
  --ticker PLTR \
  --show-signal-details \
  --max-signals 20
```

### Multiple Tickers
```bash
python -m src.analysis.triple_filter --tickers AAPL MSFT GOOGL
```

### S&P 500 Backtest with Export
```bash
python -m src.analysis.breakout_momentum \
  --sp500 \
  --export results/breakout.csv
```

## Common Options

All strategies support these options:

```bash
--ticker AAPL                  # Single ticker
--tickers AAPL MSFT GOOGL      # Multiple tickers
--sp500                        # All S&P 500 stocks
--export results/file.csv      # Export to CSV
--cache-dir data/cache/ohlcv   # Custom cache directory
--period 15                    # Max hold period in days
--show-signal-details          # Show detailed signal table with entry/exit dates
--max-signals 20               # Max signals to display in details table
```

## Output Format

All strategies provide consistent summary output with optional detailed signal tables:

```
============================================================
RSI MEAN REVERSION - AAPL
============================================================
Period: 2020-01-01 to 2024-12-25

PERFORMANCE METRICS
  Winners:             31  (68.9%)
  Losers:              14  (31.1%)
  Avg Return:        +2.34%
  Avg Winner:        +4.12%
  Avg Loser:         -1.89%
  Avg Hold Days:     6.2

ADVANCED METRICS
  Profit Factor:     2.18
  Risk/Reward:       2.18:1
  Expectancy ($100): $+2.34

EXIT TYPE BREAKDOWN
  Take Profit:        23  (51.1%)
  Stop Loss:          12  (26.7%)
  Rsi Exit:            7  (15.6%)
  Period End:          3  (6.7%)

SIGNAL DETAILS (with --show-signal-details)
  Entry Date   Exit Date      Hold Exit Type         Return
  ------------ ------------ ------ --------------- --------
  2020-03-23   2020-03-27      4d Rsi Exit          +5.23%
  2020-09-08   2020-09-14      6d Take Profit       +5.00%
  2020-10-28   2020-10-30      2d Stop Loss         -3.00%
  ...
============================================================
```

## Project Structure

```
fin/
├── src/
│   ├── analysis/              # Trading strategies (each in own folder)
│   │   ├── rsi_mean_reversion/
│   │   │   ├── __init__.py
│   │   │   ├── __main__.py
│   │   │   ├── strategy.py    # Strategy implementation
│   │   │   └── README.md      # Strategy documentation
│   │   ├── atr_mean_reversion/
│   │   ├── ma_pullback/
│   │   ├── breakout_momentum/
│   │   ├── triple_filter/
│   │   ├── macd_rsi_confluence/
│   │   ├── bb_squeeze_breakout/
│   │   ├── atr_filtered_breakout/
│   │   ├── sma200_crossover_analysis/
│   │   └── summary.py         # Unified summary module (shared)
│   ├── data/
│   │   ├── cache.py           # OHLCV data caching
│   │   ├── download_history.py
│   │   ├── update_cache.py    # Incremental data updates
│   │   └── sp500_loader.py
│   └── indicators/
│       └── technical.py       # Technical indicators (SMA, RSI, ATR, etc.)
├── data/
│   ├── sp500.csv             # S&P 500 ticker list
│   ├── custom_tickers.csv    # User watchlist
│   └── cache/ohlcv/          # Cached historical data
├── results/                   # Exported CSV files
└── requirements.txt
```

## Data Management

### Custom Ticker Watchlist

In addition to the S&P 500, you can maintain a custom watchlist of tickers in `data/custom_tickers.csv`. This is perfect for:
- Testing strategies on specific stocks (TSLA, NVDA, COIN, etc.)
- Tracking stocks not in the S&P 500
- Maintaining a personal portfolio/watchlist

**Edit `data/custom_tickers.csv` to add your tickers:**
```csv
Symbol,Name,Sector,GICS Sub-Industry,Headquarters Location,Date added,CIK,Founded
TSLA,Tesla Inc.,Consumer Discretionary,Automobile Manufacturers,"Austin, Texas",2023-12-18,1318605,2003
NVDA,NVIDIA Corporation,Information Technology,Semiconductors,"Santa Clara, California",2024-06-24,1045810,1993
```

### Initial Data Download
```bash
# Download S&P 500 historical data (one-time, ~10-20 min)
python -m src.data.download_history --sp500

# Download custom watchlist only
python -m src.data.download_history --custom

# Download ALL tickers (S&P 500 + custom combined)
python -m src.data.download_history --all

# Or download specific tickers
python -m src.data.download_history --tickers AAPL MSFT GOOGL

# Download with custom date range
python -m src.data.download_history --sp500 --start 2020-01-01 --end 2023-12-31
```

### Incremental Updates (Recommended for Weekly Use)
The framework now supports **incremental data fetching** - only downloading new data since the last cached date instead of re-downloading all historical data.

```bash
# Update all cached tickers (only fetches new data)
python -m src.data.update_cache

# Update S&P 500 tickers only
python -m src.data.update_cache --sp500

# Update custom watchlist only
python -m src.data.update_cache --custom

# Update ALL tickers (S&P 500 + custom)
python -m src.data.update_cache --all

# Update specific tickers
python -m src.data.update_cache --tickers AAPL MSFT GOOGL

# Force update even if up-to-date
python -m src.data.update_cache --force
```

**How it works:**
1. The system checks the last cached date for each ticker
2. Only fetches data from that date to today (e.g., if last date is 2024-12-20, only fetches Dec 21-25)
3. Merges new data with existing cache
4. Much faster than re-downloading all historical data

**Weekly Automation (Cron Job):**
```bash
# Update S&P 500 every Monday at 6 AM
0 6 * * 1 cd /path/to/fin && python3 -m src.data.update_cache --sp500

# Update ALL tickers (S&P 500 + custom) every day at market close (6 PM EST)
0 18 * * * cd /path/to/fin && python3 -m src.data.update_cache --all

# Update custom watchlist only (faster for small lists)
0 18 * * * cd /path/to/fin && python3 -m src.data.update_cache --custom
```

**Test Incremental Update:**
```bash
# Test on a single ticker to see incremental fetching in action
python3 test_incremental_update.py
```

## Performance Tips

1. **Use cached data**: First run downloads data, subsequent runs are instant
2. **Start small**: Test on single ticker before running on S&P 500
3. **Export results**: Use `--export` to save results for later analysis
4. **Parallel testing**: Run multiple strategies simultaneously in different terminals

## Strategy Comparison

Use `compare_strategies.py` to compare multiple strategies side-by-side:

```bash
python -m src.analysis.compare_strategies --ticker AAPL
```

## Requirements

- Python 3.8+
- pandas
- numpy
- yfinance
- pandas_ta (for technical indicators)

## Disclaimer

This framework is for **educational and research purposes only**. Past performance does not guarantee future results. Always do your own research and consult with financial professionals before trading real money.

## License

MIT License
