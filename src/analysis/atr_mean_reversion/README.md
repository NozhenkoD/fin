# ATR-Based Mean Reversion Strategy

## Overview

A **volatility-adaptive** mean reversion strategy that uses ATR (Average True Range) for dynamic stop-loss and take-profit levels instead of fixed percentages. This approach adapts to each stock's individual volatility characteristics.

## Key Innovation

Traditional fixed-percentage stops don't account for stock volatility:
- **Volatile stock** (high ATR): Fixed 3% stop may be too tight → frequent stop-outs
- **Stable stock** (low ATR): Fixed 3% stop may be too wide → excessive risk

**ATR-based solution:**
- Stop-loss: Entry - (ATR × multiplier)
- Take-profit: Entry + (ATR × multiplier)
- Automatically adjusts to stock volatility

## Strategy Logic

### Entry Conditions
1. ✅ Price above SMA200 (uptrend filter)
2. ✅ Price sustained above SMA200 for N days (established trend)
3. ✅ RSI crosses below threshold (oversold condition, default: 30)
4. ⚙️ ADX > 20 (optional - disabled by default, can enable with `--enable-adx`)

### Exit Conditions
1. **Stop Loss**: Entry - (ATR × sl_multiplier) [default: 2.0x]
2. **Take Profit**: Entry + (ATR × tp_multiplier) [default: 3.0x]
3. **RSI Recovery**: RSI crosses above exit threshold (default: 65)
4. **Period End**: Maximum hold period reached (default: 15 days)

## Expected Improvements Over Fixed Stops

- ✅ Better adaptation to individual stock volatility
- ✅ Improved risk/reward ratio (targeting 2:1+)
- ✅ Fewer whipsaw stops on volatile stocks
- ✅ Similar win rate with better average returns
- ✅ More consistent position sizing across different volatility profiles

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--rsi-threshold` | 30 | RSI oversold entry level |
| `--rsi-exit` | 65 | RSI recovery exit level |
| `--min-days-above` | 15 | Min days price must be above SMA200 |
| `--atr-sl` | 2.0 | ATR multiplier for stop-loss (2x ATR below entry) |
| `--atr-tp` | 3.0 | ATR multiplier for take-profit (3x ATR above entry) |
| `--adx-threshold` | 20 | Minimum ADX for trend strength (only used with --enable-adx) |
| `--enable-adx` | - | Enable ADX filter (disabled by default for more signals) |
| `--period` | 15 | Maximum hold period in days |

## Usage Examples

### Basic Usage
```bash
python -m src.analysis.atr_mean_reversion --ticker PLTR
```

### Custom ATR Multipliers (Wider Stops)
```bash
python -m src.analysis.atr_mean_reversion \
  --ticker PLTR \
  --atr-sl 4.0 \
  --atr-tp 6.0 \
  --rsi-exit 60 \
  --period 25
```

### Enable ADX Filter (Optional)
```bash
# ADX filter disabled by default for more signals
# Enable it to only trade in strong trends
python -m src.analysis.atr_mean_reversion \
  --ticker NVDA \
  --enable-adx \
  --adx-threshold 25
```

### S&P 500 Backtest
```bash
python -m src.analysis.atr_mean_reversion \
  --sp500 \
  --export results/atr_mean_reversion.csv
```

### Show Detailed Signals
```bash
python -m src.analysis.atr_mean_reversion \
  --ticker PLTR \
  --show-signal-details \
  --max-signals 10
```

## Output Example

```
============================================================
ATR MEAN REVERSION - PLTR
============================================================
Period: 2023-08-09 to 2025-08-20

STRATEGY
  Volatility-adaptive stops: SL=4.0xATR, TP=6.0xATR

SETTINGS
  RSI threshold: 30
  RSI exit: 60
  ATR SL multiplier: 4.0x
  ATR TP multiplier: 6.0x
  ADX threshold: disabled (use --enable-adx to enable)
  Max hold: 25 days

PERFORMANCE METRICS
  Winners:             14  (93.3%)
  Losers:               1  (6.7%)
  Avg Return:        +11.10%
  Avg Winner:        +12.04%
  Avg Loser:         -2.16%
  Avg Hold Days:     10.7

ADVANCED METRICS
  Profit Factor:     78.07
  Risk/Reward:       5.58:1

SIGNAL DETAILS
  Entry Date   Exit Date      Hold Exit Type         Return
  ------------ ------------ ------ --------------- --------
  2023-08-09   2023-08-24     15d Rsi Exit          +7.08%
  2023-08-17   2023-08-26      9d Rsi Exit         +15.41%
  ...
============================================================
```

## ATR Multiplier Guidelines

### Conservative (Lower Risk)
- SL: 1.5x ATR
- TP: 2.5x ATR
- Risk/Reward: ~1.7:1

### Moderate (Balanced)
- SL: 2.0x ATR
- TP: 3.0x ATR
- Risk/Reward: ~1.5:1

### Aggressive (Higher Volatility Tolerance)
- SL: 3.0x ATR
- TP: 5.0x ATR
- Risk/Reward: ~1.7:1

### Very Wide (Volatile Stocks like PLTR, TSLA)
- SL: 4.0x ATR
- TP: 6.0x ATR
- Risk/Reward: ~1.5:1

## When to Use This Strategy

✅ **Best For:**
- Stocks with varying volatility (tech stocks, growth names)
- Portfolio with mixed volatility profiles
- Avoiding arbitrary fixed percentage stops
- Adaptive risk management

❌ **Avoid When:**
- Extremely low volatility stocks (ATR too small, stops may be too tight)
- Gap-prone stocks (stops may be bypassed)
- Penny stocks or illiquid names

## Comparison: ATR vs Fixed Percentage

### Example: NVDA Entry at $100

**Fixed 3% Stop:**
- Stop-loss: $97.00 (always same)
- Problem: If ATR is $8, this is very tight (< 0.5 ATR)

**ATR-Based 2x Stop (ATR = $8):**
- Stop-loss: $84.00 (2 × $8 = $16 below entry)
- Benefit: Respects stock's natural price movement

## Tips for Optimization

1. **Backtest Different Multipliers**: Test 1.5x-4.0x ranges for your universe
2. **Adjust for Volatility Regime**: Use wider multipliers during high VIX periods
3. **Consider Stock Beta**: High-beta stocks may need 3x-4x ATR
4. **Test RSI Exit Levels**: Lower RSI exits (50-60) can lock in profits earlier
5. **ADX Filter**: Enable for cleaner signals, disable for more opportunities

## Related Strategies

- [RSI Mean Reversion](../rsi_mean_reversion/README.md) - Fixed percentage stops
- [MA Pullback](../ma_pullback/README.md) - Moving average based approach
- [ATR Filtered Breakout](../atr_filtered_breakout/README.md) - Uses ATR for pattern detection

## References

- **ATR**: Average True Range by J. Welles Wilder
- **Volatility-Adaptive Positioning**: Professional risk management technique
- **Strategy Type**: Mean Reversion with Dynamic Risk Management
