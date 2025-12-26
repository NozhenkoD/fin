# RSI Mean Reversion Strategy

## Overview

Buy oversold stocks (RSI < 30) that are in long-term uptrends (above SMA200). Exit when RSI recovers to overbought (RSI > 70) or hit stop-loss/take-profit targets.

This is a **mean reversion strategy** that capitalizes on temporary pullbacks in strong trending stocks.

## Strategy Logic

### Entry Conditions
1. ✅ Price above SMA200 (uptrend filter)
2. ✅ Price sustained above SMA200 for minimum days (established trend)
3. ✅ RSI crosses below threshold (oversold condition, default: 30)

### Exit Conditions
1. **Take Profit**: Price reaches profit target (default: +5.0%)
2. **Stop Loss**: Price hits stop-loss level (default: -3.0%)
3. **RSI Exit**: RSI recovers above exit threshold (default: 70)
4. **Period End**: Maximum hold period reached (default: 10 days)

## Expected Characteristics

- **Win Rate**: 60-70%
- **Avg Hold Time**: 5-15 days
- **Trade Frequency**: Higher (more signals than trend strategies)
- **Profit per Trade**: Lower but consistent
- **Best For**: Range-bound markets with clear trends

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--rsi-threshold` | 30 | RSI oversold entry level |
| `--rsi-exit` | 70 | RSI overbought exit level |
| `--min-days-above` | 15 | Minimum days price must be above SMA200 |
| `--stop-loss` | -3.0 | Stop-loss percentage |
| `--take-profit` | 5.0 | Take-profit percentage |
| `--period` | 10 | Maximum hold period in days |

## Usage Examples

### Single Ticker Analysis
```bash
python -m src.analysis.rsi_mean_reversion --ticker AAPL
```

### With Custom Parameters
```bash
python -m src.analysis.rsi_mean_reversion \
  --ticker AAPL \
  --rsi-threshold 25 \
  --rsi-exit 75 \
  --stop-loss -4.0 \
  --take-profit 6.0
```

### Multiple Tickers
```bash
python -m src.analysis.rsi_mean_reversion --tickers AAPL MSFT GOOGL
```

### S&P 500 Backtest with Export
```bash
python -m src.analysis.rsi_mean_reversion \
  --sp500 \
  --export results/rsi_mean_reversion.csv
```

### Show Detailed Signal Table
```bash
python -m src.analysis.rsi_mean_reversion \
  --ticker AAPL \
  --show-signal-details \
  --max-signals 20
```

## Output Example

```
============================================================
RSI MEAN REVERSION - AAPL
============================================================
Period: 2020-01-01 to 2024-12-25

STRATEGY
  Buy when RSI < 30 + above SMA200, exit RSI >= 70 or SL/TP

SETTINGS
  RSI threshold: 30
  RSI exit: 70
  Stop-loss: -3.0%
  Take-profit: +5.0%
  Max hold: 10 days

SIGNALS
  Total signals: 45

PERFORMANCE METRICS
  Winners:              31  (68.9%)
  Losers:               14  (31.1%)
  Avg Return:        +2.34%
  Avg Winner:        +4.12%
  Avg Loser:         -1.89%
  Avg Hold Days:     6.2

SIGNAL DETAILS
  Entry Date   Exit Date      Hold Exit Type         Return
  ------------ ------------ ------ --------------- --------
  2020-03-23   2020-03-27      4d Rsi Exit          +5.23%
  2020-09-08   2020-09-14      6d Take Profit       +5.00%
  2020-10-28   2020-10-30      2d Stop Loss         -3.00%
  ...
============================================================
```

## When to Use This Strategy

✅ **Good For:**
- Stocks in established uptrends with periodic pullbacks
- Market conditions with moderate volatility
- Swing trading (5-15 day holds)
- High-volume stocks with clear RSI patterns

❌ **Avoid When:**
- Stock is below SMA200 (downtrend)
- Extreme market volatility (RSI whipsaws)
- Very low volume stocks (poor RSI reliability)
- Strongly trending markets (fewer pullbacks)

## Tips for Optimization

1. **Adjust RSI Threshold**: Lower values (20-25) for stronger oversold signals
2. **Widen Stops on Volatile Stocks**: Use -4% to -5% for high beta names
3. **Tighten Exits in Weak Trends**: Exit at RSI 60-65 instead of 70
4. **Combine with Volume**: Look for volume spikes on entry signals
5. **Test Different Timeframes**: Some stocks work better with RSI 14 vs RSI 7

## Related Strategies

- [ATR Mean Reversion](../atr_mean_reversion/README.md) - Uses volatility-adaptive stops
- [MA Pullback](../ma_pullback/README.md) - Similar concept using MA instead of RSI
- [Triple Filter](../triple_filter/README.md) - Combines RSI + MA + Volume filters

## References

- **Technical Indicator**: RSI (Relative Strength Index) by J. Welles Wilder
- **Trend Filter**: SMA200 (Simple Moving Average, 200 periods)
- **Strategy Type**: Mean Reversion + Trend Following Hybrid
