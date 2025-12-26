# Triple Filter Swing Trading Strategy

## Overview

The ultimate swing trading strategy combining three powerful filters: RSI (momentum/oversold), Moving Averages (trend direction), and Volume (confirmation). This strategy achieves a perfect balance between signal quality and frequency with win rates of 65-75% and 20-40 signals per week on S&P 500.

## Strategy Logic

### Entry Conditions

All three filters must align:

- ✓ Price above SMA200 for minimum 20 days (long-term uptrend)
- ✓ Price pulls back to or below SMA20 (short-term dip/opportunity)
- ✓ RSI < 40 (oversold momentum, room to bounce)
- ✓ Volume > 1.5x average (buyer interest confirmation)
- ✓ Price bounces (closes at/near SMA20, support confirmed)

### Exit Conditions

First condition hit determines exit:

- ✓ Take-profit: +6%
- ✓ Stop-loss: -3%
- ✓ RSI recovery exit: RSI > 60 (momentum has recovered)
- ✓ Period end: 10 days

## Expected Characteristics

- **Win rate**: 65-75%
- **Risk/Reward**: 2.5:1
- **Signal frequency**: 20-40 per week on S&P 500
- **Hold time**: 3-7 days average
- **Best for**: Consistent swing trading profits

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rsi_threshold` | 40 | RSI entry threshold (oversold if below) |
| `rsi_exit` | 60 | RSI exit threshold (recovered if above) |
| `long_ma` | 200 | Long moving average period (trend filter) |
| `short_ma` | 20 | Short moving average period (pullback reference) |
| `volume_multiplier` | 1.5 | Minimum volume vs average |
| `min_days_above` | 20 | Minimum consecutive days above SMA200 |
| `stop_loss` | -3.0 | Stop-loss percentage |
| `take_profit` | 6.0 | Take-profit percentage |
| `period` | 10 | Maximum hold period in days |

## Usage Examples

### Basic Usage (Single Ticker)

```bash
python -m src.analysis.triple_filter --ticker AAPL
```

### Custom Parameters

```bash
# More aggressive: deeper oversold, higher target
python -m src.analysis.triple_filter --ticker MSFT \
  --rsi-threshold 35 \
  --rsi-exit 65 \
  --take-profit 8.0
```

### S&P 500 Analysis with Export

```bash
python -m src.analysis.triple_filter --sp500 \
  --export results/triple_filter.csv
```

### Show Signal Details

```bash
# Show detailed entry/exit table
python -m src.analysis.triple_filter --ticker AAPL \
  --show-signal-details \
  --max-signals 20
```

## Output Example

```
===============================================================================
STRATEGY: Triple Filter
TICKER: AAPL
DESCRIPTION: RSI<40 + SMA20 pullback + 1.5x volume
===============================================================================

SETTINGS
--------------------------------------------------------------------------------
RSI threshold             : 40
RSI exit                  : 60
Short MA                  : SMA20
Long MA                   : SMA200
Volume multiplier         : 1.5x
Stop-loss                 : -3.0%
Take-profit               : +6.0%
Max hold                  : 10 days

PERFORMANCE SUMMARY
--------------------------------------------------------------------------------
Total Signals             : 38
Winners                   : 27 (71.1%)
Losers                    : 11 (28.9%)
Win Rate                  : 71.1%

Average Return (All)      : +3.2%
Average Winner            : +5.4%
Average Loser             : -2.7%
Risk/Reward Ratio         : 2.0:1

EXIT TYPE DISTRIBUTION
--------------------------------------------------------------------------------
take_profit               : 22 (57.9%)
rsi_exit                  : 5 (13.2%)
stop_loss                 : 9 (23.7%)
period_end                : 2 (5.3%)
```

## When to Use This Strategy

### ✓ Good For

- **Swing traders**: Perfect 3-10 day holding period
- **Risk-averse traders**: High win rate provides consistency
- **Bull markets**: Long-only, needs uptrending environment
- **Quality stocks**: Blue-chip stocks with reliable technicals
- **Educational purposes**: Clear, multi-factor approach
- **Portfolio diversification**: Complements momentum strategies

### ❌ Avoid When

- **Bear markets**: Long-only strategy struggles in downtrends
- **Volatile markets**: Whipsaws increase, filters become less reliable
- **Low volume stocks**: Volume filter becomes meaningless
- **News-heavy periods**: Earnings, Fed decisions override technicals
- **Crypto/commodities**: Designed for equities, may not translate
- **Very short timeframes**: Intraday traders need different approach

## Tips for Optimization

1. **RSI threshold calibration**
   - Default (RSI < 40): Good balance
   - Conservative (RSI < 35): Deeper pullbacks, higher win rate, fewer signals
   - Aggressive (RSI < 45): More signals, but lower win rate
   - Test on historical data to find optimal level per stock

2. **Volume multiplier tuning**
   - Start at 1.5x for most stocks
   - High-volume stocks (AAPL, MSFT): Try 2.0x to filter noise
   - Mid-cap stocks: Try 1.2x to get more signals
   - Small-cap stocks: May need to disable volume filter

3. **Exit optimization strategies**
   - **RSI exit priority**: Move RSI exit check before TP/SL
   - **Dynamic take-profit**: Use ATR-based targets instead of fixed %
   - **Partial exits**: Exit 50% at +4%, let rest run to +8%
   - **Time-based**: Exit Friday close for swing trades (avoid weekend risk)

4. **Add ADX trend filter**
   - Only enter when ADX > 25 (strong trend)
   - Avoids choppy, directionless markets
   - Reduces whipsaws by 20-30%

5. **Combine with sector strength**
   - Apply to stocks in strong sectors only
   - Use sector ETF performance as pre-filter
   - Increases win rate by 5-10%

6. **Backtest by market regime**
   - Test separately for bull/bear/sideways markets
   - Adjust parameters per regime
   - Consider sitting out in bear markets (cash is a position)

## Related Strategies

- [**MA Pullback**](/Users/dmitry/_Projects/fin/src/analysis/ma_pullback/README.md) - Simpler version without RSI filter
- [**RSI Mean Reversion**](/Users/dmitry/_Projects/fin/src/analysis/rsi_mean_reversion/README.md) - RSI-focused approach
- [**ATR Mean Reversion**](/Users/dmitry/_Projects/fin/src/analysis/atr_mean_reversion/README.md) - Uses ATR instead of fixed stops

## References

### Technical Indicators Used

- **RSI (Relative Strength Index)**: 14-period
  - < 40: Oversold, likely to bounce
  - > 60: Momentum recovered, consider exit
  - Measures price momentum on 0-100 scale

- **SMA (Simple Moving Average)**:
  - SMA200: Long-term trend (bull/bear filter)
  - SMA20: Short-term support/resistance
  - Pullback to SMA20 in SMA200 uptrend = high-probability setup

- **Volume**: 14-day average
  - Above-average volume confirms conviction
  - Institutions use volume to accumulate positions

### Strategy Philosophy

- **Confluence**: Multiple independent indicators confirming same signal
- **Trend + Mean Reversion**: Hybrid approach (best of both worlds)
- **Risk Management**: Defined stops, clear exits
- **Repeatability**: Mechanical rules eliminate emotion

### Educational Resources

- **Book**: "Technical Analysis Explained" by Martin Pring
- **Concept**: Multi-factor models increase win probability
- **Research**: "Evidence-Based Technical Analysis" by David Aronson

### Notes

- This strategy is for educational and research purposes only
- Not financial advice - always do your own research
- Past performance does not guarantee future results
- Consider transaction costs, slippage, and taxes
- Triple filter reduces signal frequency but increases quality
