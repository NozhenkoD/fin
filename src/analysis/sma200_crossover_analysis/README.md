# SMA200 Crossover Analysis

## Overview

Analyzes two distinct strategies based on the SMA200 level: **Resistance Break** (price breaks above SMA200 from below) and **Support Bounce** (price touches SMA200 from above and bounces). The SMA200 is one of the most widely watched technical levels, acting as dynamic support/resistance that institutions monitor.

## Strategy Logic

### Strategy 1: Resistance Break

Price breaking above SMA200 after sustained downtrend - resistance becomes support.

**Entry Conditions:**
- ✓ Previous close below SMA200
- ✓ Price was below SMA200 for minimum 15 days (sustained below)
- ✓ Current open above SMA200
- ✓ Current close above SMA200
- ✓ Open within 1% of SMA200 (filters large gap-ups)

**Exit Conditions:**
- ✓ Take-profit: +10%
- ✓ Stop-loss: -5%
- ✓ Period end: 20 days

### Strategy 2: Support Bounce

Price in uptrend dips to touch SMA200 and bounces - support holds.

**Entry Conditions:**
- ✓ Price above SMA200 for minimum 15 days (sustained uptrend)
- ✓ Low touches or gets within 2% of SMA200 (support test)
- ✓ Close back above SMA200 (bounce confirmed)

**Exit Conditions:**
- ✓ Take-profit: +10%
- ✓ Stop-loss: -5%
- ✓ Period end: 20 days

## Expected Characteristics

### Resistance Break
- **Win rate**: 50-55%
- **Best for**: Trend reversal traders
- **Market regime**: Transition from bear to bull

### Support Bounce
- **Win rate**: 60-65%
- **Best for**: Pullback traders in uptrends
- **Market regime**: Established bull markets

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `strategy` | resistance_break | Strategy type: resistance_break or support_bounce |
| `min_days` | 15 | Minimum consecutive days above/below SMA200 |
| `proximity_pct` | 0.01 (resistance)<br>0.02 (support) | Maximum distance from SMA200 |
| `stop_loss` | -5.0 | Stop-loss percentage |
| `take_profit` | 10.0 | Take-profit percentage |
| `period` | 20 | Maximum hold period in days |

## Usage Examples

### Resistance Break Strategy

```bash
# Single ticker
python -m src.analysis.sma200_crossover_analysis \
  --ticker AAPL \
  --strategy resistance_break
```

### Support Bounce Strategy

```bash
# Single ticker with custom parameters
python -m src.analysis.sma200_crossover_analysis \
  --ticker MSFT \
  --strategy support_bounce \
  --min-days 20 \
  --stop-loss -3.0 \
  --take-profit 8.0
```

### S&P 500 Analysis

```bash
# Resistance breaks across S&P 500
python -m src.analysis.sma200_crossover_analysis --sp500 \
  --strategy resistance_break \
  --export results/sp500_resistance_break.csv

# Support bounces across S&P 500
python -m src.analysis.sma200_crossover_analysis --sp500 \
  --strategy support_bounce \
  --export results/sp500_support_bounce.csv
```

### Multiple Tickers

```bash
# Test both strategies on watchlist
python -m src.analysis.sma200_crossover_analysis \
  --tickers AAPL MSFT GOOGL NVDA \
  --strategy support_bounce
```

### Show Signal Details

```bash
# Show detailed entry/exit table
python -m src.analysis.sma200_crossover_analysis \
  --ticker AAPL \
  --strategy resistance_break \
  --show-signal-details \
  --max-signals 20
```

## Output Example

```
===============================================================================
STRATEGY: SMA200 Resistance Break
TICKER: AAPL
DESCRIPTION: Price breaks above SMA200
===============================================================================

SETTINGS
--------------------------------------------------------------------------------
Min days                  : 15
Stop-loss                 : -5.0%
Take-profit               : +10.0%
Analysis period           : 20 days

PERFORMANCE SUMMARY
--------------------------------------------------------------------------------
Total Signals             : 12
Winners                   : 7 (58.3%)
Losers                    : 5 (41.7%)
Win Rate                  : 58.3%

Average Return (All)      : +3.2%
Average Winner            : +8.4%
Average Loser             : -4.1%
Risk/Reward Ratio         : 2.0:1

EXIT TYPE DISTRIBUTION
--------------------------------------------------------------------------------
take_profit               : 6 (50.0%)
stop_loss                 : 4 (33.3%)
period_end                : 2 (16.7%)
```

## When to Use This Strategy

### ✓ Good For (Resistance Break)

- **Trend reversal trading**: Catching transition from bear to bull
- **Long-term investors**: SMA200 breaks signal major shifts
- **Patient traders**: Signals are infrequent but meaningful
- **Risk-on environments**: Market sentiment improving
- **Post-correction entries**: Getting in after pullback ends
- **Institutional following**: Many funds watch SMA200

### ✓ Good For (Support Bounce)

- **Pullback traders**: Buying dips in uptrends
- **Trend followers**: Staying with the trend
- **Swing traders**: 10-20 day holding period
- **Bull markets**: Strategy designed for uptrending environment
- **Risk management**: Well-defined support level for stops
- **High probability setups**: SMA200 as support has high success rate

### ❌ Avoid When

- **Choppy markets**: Whipsaws around SMA200 common in ranges
- **Extreme volatility**: Large gaps can bypass levels
- **Earnings season**: News overrides technical levels
- **Low volume periods**: Moves lack conviction
- **Contrarian plays needed**: Both strategies are trend-following
- **Very short timeframes**: SMA200 is longer-term indicator

## Tips for Optimization

1. **Min days filter optimization**
   - Default 15 days: Good balance
   - Increase to 20-30: More significant levels, fewer whipsaws
   - Decrease to 10: More signals, but more false crossovers
   - Zero: Allow all crossovers (much lower quality)

2. **Proximity percentage tuning**
   - Resistance break (default 1%): Filters gap-ups
   - Tighten to 0.5%: Only clean crossovers
   - Widen to 2%: Allow more entry points
   - Support bounce (default 2%): Allows near-touches
   - Adjust based on stock volatility

3. **Stop-loss and take-profit**
   - Default (5% SL, 10% TP): 2:1 risk/reward
   - Conservative (3% SL, 6% TP): Tighter for less volatile stocks
   - Aggressive (7% SL, 15% TP): Wider for trend riders
   - Asymmetric: Different values for each strategy

4. **Add volume confirmation**
   - Require volume > 1.5x average on crossover day
   - High volume = institutional participation
   - Filters weak crossovers
   - Increases win rate significantly

5. **Combine strategies**
   - Resistance break followed by support bounce = strong trend
   - Track both on same stock for full picture
   - Portfolio approach: Some stocks breaking up, others bouncing

6. **Sector rotation overlay**
   - Apply to sector ETFs first
   - Identify sectors with SMA200 breaks
   - Then focus on individual stocks in those sectors
   - Improves context and probability

## Related Strategies

- [**MA Pullback**](/Users/dmitry/_Projects/fin/src/analysis/ma_pullback/README.md) - Uses SMA20 pullback in SMA200 uptrend
- [**Triple Filter**](/Users/dmitry/_Projects/fin/src/analysis/triple_filter/README.md) - Multi-filter approach including MA
- [**Breakout Momentum**](/Users/dmitry/_Projects/fin/src/analysis/breakout_momentum/README.md) - Momentum breakouts with SMA200 filter

## References

### Technical Indicators Used

- **SMA200 (Simple Moving Average, 200-period)**:
  - Most widely watched long-term moving average
  - Acts as dynamic support/resistance
  - Below SMA200 = bearish, Above = bullish
  - Crossovers signal significant trend changes
  - Used by institutions, creates self-fulfilling prophecy

### Strategy Philosophy

**Resistance Break:**
- Price breaking above resistance = buyers in control
- Sustained time below creates "coiled spring" effect
- Breakout signals trend reversal
- Former resistance becomes new support

**Support Bounce:**
- Price respecting support = trend intact
- Pullbacks to SMA200 create buying opportunities
- Bounce confirms support is holding
- Higher probability in established uptrends

### Market Psychology

- **200-day MA**: Psychological level watched globally
- **Institutional usage**: Triggers for funds and algorithms
- **Self-fulfilling**: So many watch it, it becomes real
- **Trend classification**: Simple bull/bear decision tool

### Educational Resources

- **Book**: "Technical Analysis of the Financial Markets" by John Murphy
- **Concept**: Moving averages as dynamic support/resistance
- **Research**: SMA200 crossovers have worked for decades
- **Historical**: Golden cross (SMA50 > SMA200) is famous variant

### Statistical Context

- **Resistance breaks**: ~50% success rate historically
- **Support bounces**: ~60% success rate in bull markets
- **Whipsaws**: Common in sideways markets (30-40% of signals)
- **Best performance**: Clear trending environments

### Notes

- This strategy is for educational and research purposes only
- Not financial advice - always do your own research
- SMA200 is lagging indicator - confirms trends, doesn't predict
- Works best combined with fundamental analysis
- Consider market regime when interpreting signals
- Both strategies are long-only (bullish bias)
- Results vary significantly by market environment
