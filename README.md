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

### 1. RSI Mean Reversion
Buys oversold stocks (RSI < 30) in strong uptrends (above SMA200). High win rate strategy for range-bound markets.

**Key Parameters:**
- `--rsi-threshold`: RSI entry level (default: 30)
- `--rsi-exit`: RSI exit level (default: 65)
- `--min-days-above`: Min days above SMA200 (default: 15)

**Example:**
```bash
# Single ticker
python -m src.analysis.rsi_mean_reversion --ticker AAPL

# Multiple tickers
python -m src.analysis.rsi_mean_reversion --tickers AAPL MSFT GOOGL

# S&P 500 with export
python -m src.analysis.rsi_mean_reversion --sp500 --export results/rsi.csv
```

### 2. ATR Mean Reversion
Similar to RSI mean reversion but uses ATR-based dynamic stops that adapt to stock volatility. Better risk management for volatile stocks.

**Key Parameters:**
- `--atr-sl`: ATR multiplier for stop-loss (default: 2.0)
- `--atr-tp`: ATR multiplier for take-profit (default: 3.0)
- `--rsi-threshold`: RSI entry level (default: 30)

**Example:**
```bash
python -m src.analysis.atr_mean_reversion --ticker NVDA --atr-sl 1.5 --atr-tp 3.0
```

### 3. MA Pullback
Buys pullbacks to SMA20 in stocks above SMA200. One of the most reliable swing trading strategies with 60-70% win rate.

**Key Parameters:**
- `--short-ma`: Short MA period (default: 20)
- `--long-ma`: Long MA period (default: 200)
- `--volume-mult`: Volume multiplier (default: 1.5)
- `--stop-loss`: Stop-loss % (default: -3.0)
- `--take-profit`: Take-profit % (default: 6.0)

**Example:**
```bash
python -m src.analysis.ma_pullback --ticker AAPL --stop-loss -4.0 --take-profit 8.0
```

### 4. Breakout Momentum
Trend-following strategy that buys strength. Enters on 20-day high breakouts with volume confirmation and RSI > 60.

**Key Parameters:**
- `--breakout-period`: Lookback for high (default: 20)
- `--volume-mult`: Volume multiplier (default: 2.0)
- `--rsi-threshold`: Min RSI for entry (default: 60)
- `--trailing-stop`: Trailing stop % (default: -8.0)
- `--take-profit`: Take-profit % (default: 15.0)

**Example:**
```bash
python -m src.analysis.breakout_momentum --ticker TSLA --rsi-threshold 65
```

### 5. Triple Filter
Combines RSI, moving averages, and volume for high-probability setups. Typically 65-75% win rate.

**Key Parameters:**
- `--rsi-threshold`: RSI threshold (default: 40)
- `--short-ma`: Short MA (default: 20)
- `--long-ma`: Long MA (default: 200)
- `--volume-mult`: Volume multiplier (default: 1.5)

**Example:**
```bash
python -m src.analysis.triple_filter --sp500 --export results/triple.csv
```

### 6. MACD + RSI Confluence
Requires both MACD bullish crossover and RSI confirmation. Filters false MACD signals for cleaner entries.

**Key Parameters:**
- `--rsi-entry-min`: Min RSI for entry (default: 40)
- `--rsi-entry-max`: Max RSI for entry (default: 60)
- `--adx-threshold`: Min ADX for trend (default: 20)

**Example:**
```bash
python -m src.analysis.macd_rsi_confluence --ticker AAPL
```

### 7. Bollinger Band Squeeze Breakout
Detects volatility contraction (squeeze) and enters on breakout. Catches explosive moves after consolidation.

**Key Parameters:**
- `--squeeze-lookback`: Days for squeeze detection (default: 20)
- `--volume-mult`: Volume multiplier (default: 1.5)
- `--rsi-threshold`: Min RSI (default: 50)

**Example:**
```bash
python -m src.analysis.bb_squeeze_breakout --ticker AAPL
```

### 8. ATR-Filtered False Breakout
Detects false breakouts below swing lows and enters expecting mean reversion. Advanced pattern recognition.

**Key Parameters:**
- `--swing-strength`: Candles on each side (default: 2)
- `--min-atr-mult`: Min ATR multiplier for candle filter (default: 1.0)
- `--max-atr-mult`: Max ATR multiplier for candle filter (default: 2.0)
- `--sl-atr-mult`: Stop-loss ATR multiplier (default: 1.0)
- `--risk-reward`: Risk/reward ratio (default: 1.0)
- `--lookback-candles`: Swing low lookback (default: 200)

**Example with ALL arguments:**
```bash
python -m src.analysis.atr_filtered_breakout \
  --ticker AAPL \
  --atr-period 14 \
  --swing-strength 2 \
  --min-atr-mult 1.0 \
  --max-atr-mult 2.0 \
  --sl-atr-mult 1.0 \
  --risk-reward 1.5 \
  --lookback-candles 200 \
  --period 15 \
  --cache-dir data/cache/ohlcv \
  --export results/atr_filtered.csv \
  --verbose
```

### 9. SMA200 Crossover
Analyzes what happens when price crosses above SMA200. Simple trend-following approach.

**Example:**
```bash
python -m src.analysis.sma200_crossover_analysis --ticker AAPL
```

## Common Options

All strategies support these options:

```bash
--ticker AAPL              # Single ticker
--tickers AAPL MSFT GOOGL  # Multiple tickers
--sp500                    # All S&P 500 stocks
--export results/file.csv  # Export to CSV
--cache-dir data/cache/ohlcv  # Custom cache directory
--period 15                # Max hold period in days
```

## Output Format

All strategies provide consistent summary output:

```
==================================================
STRATEGY: RSI Mean Reversion - AAPL
==================================================

Total Signals:    45
Winners:          31 (68.9%)
Losers:           14 (31.1%)

Returns:
  Avg Return:     +2.34%
  Avg Winner:     +4.12%
  Avg Loser:      -1.89%

Risk/Reward:      2.18:1
Avg Hold Days:    6.2

Exit Types:
  take_profit     23 (51.1%)
  stop_loss       12 (26.7%)
  rsi_exit         7 (15.6%)
  period_end       3 (6.7%)
==================================================
```

## Project Structure

```
fin/
├── src/
│   ├── analysis/              # Trading strategies
│   │   ├── rsi_mean_reversion.py
│   │   ├── atr_mean_reversion.py
│   │   ├── ma_pullback.py
│   │   ├── breakout_momentum.py
│   │   ├── triple_filter.py
│   │   ├── macd_rsi_confluence.py
│   │   ├── bb_squeeze_breakout.py
│   │   ├── atr_filtered_breakout.py
│   │   ├── sma200_crossover_analysis.py
│   │   └── summary.py         # Unified summary module
│   ├── data/
│   │   ├── cache.py           # OHLCV data caching
│   │   ├── download_history.py
│   │   └── sp500_loader.py
│   └── indicators/
│       └── technical.py       # Technical indicators (SMA, RSI, ATR, etc.)
├── data/
│   └── cache/ohlcv/          # Cached historical data
├── results/                   # Exported CSV files
└── requirements.txt
```

## Data Management

### Initial Data Download
```bash
# Download S&P 500 historical data (one-time, ~10-20 min)
python -m src.data.download_history --sp500

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

# Update specific tickers
python -m src.data.update_cache --tickers AAPL MSFT GOOGL

# Update S&P 500 tickers
python -m src.data.update_cache --sp500

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
# Add to your crontab (runs every Monday at 6 AM)
0 6 * * 1 cd /path/to/fin && python3 -m src.data.update_cache --sp500

# Or run every day at market close (6 PM EST)
0 18 * * * cd /path/to/fin && python3 -m src.data.update_cache --sp500
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
