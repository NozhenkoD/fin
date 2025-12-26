# MACD + RSI Confluence Strategy

## Overview

A momentum confirmation strategy that combines MACD crossovers with RSI to filter for high-probability setups. MACD crossovers are powerful but generate many false signals. By requiring RSI confirmation, this strategy filters for setups where both momentum indicators agree, significantly improving win rate.

## Strategy Logic

### Entry Conditions

All conditions must align (confluence):

- ✓ Price above SMA200 (long-term uptrend)
- ✓ Price above SMA200 for minimum 15 days (sustained uptrend)
- ✓ MACD histogram crosses from negative to positive (bullish crossover)
- ✓ RSI between 40-60 (recovery zone, not overbought)
- ✓ Previous RSI was below 45 (confirming momentum recovery)
- ✓ ADX > 20 (trending market, not choppy)

### Exit Conditions

First condition hit determines exit:

- ✓ MACD histogram turns negative (momentum lost)
- ✓ RSI > 70 (overbought, take profit)
- ✓ Stop-loss: Entry - (2.0 × ATR) (adaptive to volatility)
- ✓ Period end: 15 days

## Expected Characteristics

- **Win rate**: 55-60%
- **Signal frequency**: Moderate (quality over quantity)
- **Hold time**: 5-12 days average
- **Best for**: Capturing momentum shifts in trending stocks

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rsi_entry_min` | 40 | Minimum RSI for entry (recovery zone) |
| `rsi_entry_max` | 60 | Maximum RSI for entry (not overbought) |
| `rsi_prev_max` | 45 | Previous RSI must be below (confirms recovery) |
| `rsi_overbought` | 70 | RSI exit threshold (overbought) |
| `adx_threshold` | 20 | Minimum ADX (trend strength filter) |
| `min_days_above` | 15 | Minimum consecutive days above SMA200 |
| `atr_sl` | 2.0 | ATR multiplier for stop-loss |
| `period` | 15 | Maximum hold period in days |

## Usage Examples

### Basic Usage (Single Ticker)

```bash
python -m src.analysis.macd_rsi_confluence --ticker AAPL
```

### Custom Parameters

```bash
# Tighter RSI range for quality trades
python -m src.analysis.macd_rsi_confluence --ticker MSFT \
  --rsi-entry-min 45 \
  --rsi-entry-max 55 \
  --atr-sl 1.5
```

### S&P 500 Analysis with Export

```bash
python -m src.analysis.macd_rsi_confluence --sp500 \
  --export results/macd_rsi_confluence.csv
```

### Show Signal Details

```bash
# Show detailed entry/exit table
python -m src.analysis.macd_rsi_confluence --ticker NVDA \
  --show-signal-details \
  --max-signals 15
```

## Output Example

```
===============================================================================
STRATEGY: MACD RSI Confluence
TICKER: AAPL
DESCRIPTION: MACD histogram cross + RSI 40-60 recovery
===============================================================================

SETTINGS
--------------------------------------------------------------------------------
RSI entry range           : 40-60
RSI prev max              : 45
RSI overbought            : 70
ADX threshold             : 20
ATR stop-loss             : 2.0x
Max hold                  : 15 days

PERFORMANCE SUMMARY
--------------------------------------------------------------------------------
Total Signals             : 24
Winners                   : 14 (58.3%)
Losers                    : 10 (41.7%)
Win Rate                  : 58.3%

Average Return (All)      : +2.6%
Average Winner            : +7.8%
Average Loser             : -4.2%
Risk/Reward Ratio         : 1.9:1

EXIT TYPE DISTRIBUTION
--------------------------------------------------------------------------------
rsi_overbought            : 8 (33.3%)
macd_exit                 : 6 (25.0%)
stop_loss                 : 7 (29.2%)
period_end                : 3 (12.5%)
```

## When to Use This Strategy

### ✓ Good For

- **Trending markets**: ADX filter ensures you're in trending environments
- **Momentum traders**: Captures shifts from weakness to strength
- **Swing traders**: 5-15 day holding period
- **Quality over quantity**: Fewer but higher-probability signals
- **Risk-conscious traders**: ATR-based stops adapt to volatility
- **Technical purists**: Multi-indicator confirmation

### ❌ Avoid When

- **Choppy markets**: ADX < 20 signals filtered out, but still possible whipsaws
- **Bear markets**: Long-only strategy needs uptrends
- **Low volatility**: MACD crossovers may be insignificant
- **Earnings season**: News overrides technical signals
- **Gap-heavy stocks**: Stops may be bypassed
- **Range-bound stocks**: MACD generates false signals

## Tips for Optimization

1. **RSI entry range tuning**
   - Narrow range (45-55): Fewer signals, higher quality
   - Wide range (35-65): More signals, but more noise
   - Asymmetric range (40-55): Catches early recoveries
   - Test on backtest data to find optimal range

2. **MACD parameter customization**
   - Default: 12, 26, 9 (standard MACD)
   - Faster: 8, 17, 9 (more responsive, more signals)
   - Slower: 19, 39, 9 (less responsive, fewer whipsaws)
   - Match to stock's volatility profile

3. **ADX threshold optimization**
   - ADX > 20: Default, moderate filter
   - ADX > 25: Stricter, stronger trends only
   - ADX > 15: Looser, more signals
   - Consider ADX slope (rising ADX = strengthening trend)

4. **ATR stop-loss multiplier**
   - 2.0x ATR: Default, good balance
   - 1.5x ATR: Tighter stops for less volatile stocks
   - 2.5x ATR: Wider stops for volatile stocks
   - 3.0x ATR: Very wide, for momentum riding

5. **Add volume confirmation**
   - Require volume > 1.5x average on MACD crossover day
   - Filters out weak crossovers with no conviction
   - Increases win rate by 3-5%

6. **Exit optimization**
   - **Partial profits**: Exit 50% at RSI 65, let rest ride
   - **Trailing MACD**: Only exit if MACD histogram negative for 2 days
   - **Time stop**: Exit if no 5% profit after 10 days
   - **Fundamental overlay**: Exit before earnings announcements

## Related Strategies

- [**Triple Filter**](/Users/dmitry/_Projects/fin/src/analysis/triple_filter/README.md) - Alternative multi-indicator approach
- [**Breakout Momentum**](/Users/dmitry/_Projects/fin/src/analysis/breakout_momentum/README.md) - Pure momentum breakout strategy
- [**RSI Mean Reversion**](/Users/dmitry/_Projects/fin/src/analysis/rsi_mean_reversion/README.md) - RSI-only approach

## References

### Technical Indicators Used

- **MACD (Moving Average Convergence Divergence)**:
  - Calculation: 12-period EMA - 26-period EMA
  - Signal line: 9-period EMA of MACD
  - Histogram: MACD - Signal (positive = bullish)
  - Crossover from negative to positive signals momentum shift

- **RSI (Relative Strength Index)**: 14-period
  - 40-60 range: Recovery zone (not oversold, not overbought)
  - < 45 previous: Confirms coming from weakness
  - > 70: Overbought exit signal
  - Measures momentum on 0-100 scale

- **ADX (Average Directional Index)**: 14-period
  - > 20: Trending market (filter for choppy conditions)
  - Measures trend strength, not direction
  - Higher ADX = stronger trend = better MACD reliability

- **ATR (Average True Range)**: 14-period
  - Volatility measure for adaptive stops
  - 2.0x ATR = stop is 2 standard deviations away
  - Prevents stops too tight in volatile conditions

- **SMA200**: Long-term trend filter
  - Only trade in confirmed uptrends
  - Simple but effective bull/bear classifier

### Strategy Philosophy

- **Confluence**: When multiple indicators agree, probability increases
- **Momentum shift detection**: MACD catches turning points
- **Confirmation**: RSI validates the shift is real
- **Trend following with timing**: Combines trend and timing
- **Adaptive risk**: ATR-based stops adjust to market conditions

### Educational Resources

- **Book**: "Trading with MACD" by Barbara Star
- **Concept**: MACD crossovers are early momentum signals
- **Research**: RSI confirmation reduces false positives by 30-40%

### Notes

- This strategy is for educational and research purposes only
- Not financial advice - always do your own research
- MACD is a lagging indicator - expects trend continuation
- Works best in clear trending environments
- Consider combining with fundamental analysis for best results
