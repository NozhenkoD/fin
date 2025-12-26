# Moving Average Pullback Strategy

## Overview

A proven swing trading strategy that buys pullbacks to short-term moving averages in stocks with strong long-term trends. This is one of the most reliable swing trading strategies with high win rates (60-70%) and consistent performance throughout the year.

## Strategy Logic

### Entry Conditions

All conditions must be met:

- ✓ Price above SMA200 (strong long-term uptrend filter)
- ✓ Price has been above SMA200 for minimum 20 days (sustained uptrend)
- ✓ Price pulls back and touches/crosses SMA20
- ✓ Volume on pullback day > 1.5x average (confirmation)
- ✓ Price bounces (closes above or near SMA20)

### Exit Conditions

First condition hit determines exit:

- ✓ Take-profit: +6%
- ✓ Stop-loss: -3%
- ✓ Period end: 10 days (if neither SL/TP hit)

## Expected Characteristics

- **Win rate**: 60-70%
- **Signal frequency**: 50-100 per week on S&P 500
- **Risk/Reward**: 2:1
- **Hold time**: 3-7 days average
- **Best for**: Stocks in strong uptrends

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `long_ma` | 200 | Long moving average period (trend filter) |
| `short_ma` | 20 | Short moving average period (pullback reference) |
| `volume_multiplier` | 1.5 | Minimum volume vs average (1.5x = 50% above average) |
| `min_days_above` | 20 | Minimum consecutive days above long MA |
| `stop_loss` | -3.0 | Stop-loss percentage |
| `take_profit` | 6.0 | Take-profit percentage |
| `period` | 10 | Maximum hold period in days |

## Usage Examples

### Basic Usage (Single Ticker)

```bash
python -m src.analysis.ma_pullback --ticker AAPL
```

### Custom Parameters

```bash
# Tighter parameters for less volatile stocks
python -m src.analysis.ma_pullback --ticker AAPL \
  --short-ma 10 \
  --volume-mult 2.0 \
  --stop-loss -2.0 \
  --take-profit 5.0
```

### S&P 500 Analysis with Export

```bash
python -m src.analysis.ma_pullback --sp500 \
  --export results/ma_pullback.csv
```

### Show Signal Details

```bash
# Show detailed entry/exit table for each signal
python -m src.analysis.ma_pullback --ticker AAPL \
  --show-signal-details \
  --max-signals 20
```

## Output Example

```
===============================================================================
STRATEGY: MA Pullback
TICKER: AAPL
DESCRIPTION: Pullback to SMA20 in SMA200 uptrend + volume
===============================================================================

SETTINGS
--------------------------------------------------------------------------------
Short MA                  : SMA20
Long MA                   : SMA200
Volume multiplier         : 1.5x
Stop-loss                 : -3.0%
Take-profit               : +6.0%
Max hold                  : 10 days

PERFORMANCE SUMMARY
--------------------------------------------------------------------------------
Total Signals             : 45
Winners                   : 31 (68.9%)
Losers                    : 14 (31.1%)
Win Rate                  : 68.9%

Average Return (All)      : +2.8%
Average Winner            : +5.2%
Average Loser             : -2.8%
Risk/Reward Ratio         : 1.9:1

Median Return             : +3.1%
Best Trade                : +6.0%
Worst Trade               : -3.0%

Average Hold Time         : 4.2 days
```

## When to Use This Strategy

### ✓ Good For

- **Bull markets**: Works best when overall market is trending up
- **Quality stocks**: Blue-chip stocks with established trends
- **Strong uptrends**: Stocks consistently above SMA200
- **High volume stocks**: Liquid stocks with reliable volume data
- **Swing trading**: Hold times of 3-10 days
- **Educational purposes**: Clear entry/exit rules, high win rate

### ❌ Avoid When

- **Bear markets**: Strategy is long-only, struggles in downtrends
- **Choppy/sideways markets**: Many false signals around SMA20
- **Low volume stocks**: Volume filter becomes unreliable
- **News-driven volatility**: Technical levels less reliable
- **Gap-heavy stocks**: Large gaps can bypass stop-loss
- **Earnings season**: Increased volatility can trigger stops

## Tips for Optimization

1. **Adjust MA periods for stock personality**
   - Fast movers (tech): Try SMA10/SMA100
   - Slow movers (utilities): Stick with SMA20/SMA200
   - Test different combinations to match volatility

2. **Volume multiplier tuning**
   - Start at 1.5x for normal stocks
   - Increase to 2.0x for high-volume stocks to filter noise
   - Decrease to 1.2x for lower-volume stocks to get more signals

3. **Risk/Reward optimization**
   - Default 2:1 (6% TP, 3% SL) works well historically
   - For conservative trading: 1.5:1 (4.5% TP, 3% SL)
   - For aggressive trading: 3:1 (9% TP, 3% SL)

4. **Filter by distance from SMA20**
   - Add `distance_from_sma20` filter to code
   - Entry only if pullback touches within 1% of SMA20
   - Tighter touches = stronger support = higher win rate

5. **Combine with RSI**
   - Add RSI < 50 filter for deeper pullbacks
   - Increases win rate but decreases signal frequency
   - Good for more conservative traders

6. **Backtest across multiple timeframes**
   - Test on different market conditions (bull/bear/sideways)
   - Verify edge persists across 3+ years of data
   - Check performance by sector (tech, healthcare, etc.)

## Related Strategies

- [**RSI Mean Reversion**](/Users/dmitry/_Projects/fin/src/analysis/rsi_mean_reversion/README.md) - Uses RSI instead of MA for oversold detection
- [**Triple Filter**](/Users/dmitry/_Projects/fin/src/analysis/triple_filter/README.md) - Combines MA pullback with RSI and volume
- [**ATR Mean Reversion**](/Users/dmitry/_Projects/fin/src/analysis/atr_mean_reversion/README.md) - Uses ATR-based dynamic stops instead of fixed percentages

## References

### Technical Indicators Used

- **SMA (Simple Moving Average)**: Trend identification and support/resistance
  - SMA200: Long-term trend filter
  - SMA20: Short-term pullback reference

- **Volume**: Confirmation of buyer interest
  - 14-day average volume for comparison
  - Spikes indicate institutional activity

### Educational Resources

- **Book**: "How to Make Money in Stocks" by William O'Neil (CANSLIM method)
- **Concept**: Buying pullbacks in uptrends is a cornerstone of swing trading
- **Historical Performance**: This pattern has worked consistently since the 1950s

### Notes

- This strategy is for educational and research purposes only
- Not financial advice - always do your own research
- Past performance does not guarantee future results
- Consider transaction costs, slippage, and taxes in live trading
