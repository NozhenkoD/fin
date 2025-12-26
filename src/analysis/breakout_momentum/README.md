# Breakout + Momentum Strategy

## Overview

A trend-following strategy that "buys high, sells higher" by entering on strength rather than weakness. Designed for trending markets where mean reversion strategies fail. Uses 20-day high breakouts combined with volume and RSI momentum filters to ride institutional buying waves.

## Strategy Logic

### Entry Conditions

All conditions must be met:

- ✓ Price breaks above 20-day high (new breakout)
- ✓ Volume > 2x average (strong buying pressure)
- ✓ Price > SMA200 (long-term uptrend filter)
- ✓ Price above SMA200 for minimum 20 days (sustained uptrend)
- ✓ RSI > 60 (momentum confirmation)
- ✓ Close > Open (bullish candle)

### Exit Conditions

First condition hit determines exit:

- ✓ Trailing stop: -8% from highest high since entry
- ✓ Take-profit: +15%
- ✓ Period end: 30 days (if neither SL/TP hit)

**Note**: Trailing stop locks in profits as price makes new highs - this is how momentum traders protect gains while letting winners run.

## Expected Characteristics

- **Win rate**: 52-58% (lower than mean reversion, but bigger wins)
- **Average winner**: +12-18%
- **Average loser**: -6-8%
- **Risk/Reward**: 2:1 to 3:1
- **Hold time**: 8-15 days
- **Best for**: Bull markets, trending stocks, growth sectors

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `breakout_period` | 20 | Days for high lookback (20-day high) |
| `volume_multiplier` | 2.0 | Minimum volume vs average (2x = 100% above average) |
| `rsi_threshold` | 60 | Minimum RSI for momentum confirmation |
| `min_days_above` | 20 | Minimum consecutive days above SMA200 |
| `trailing_stop` | -8.0 | Trailing stop percentage from highest high |
| `take_profit` | 15.0 | Fixed take-profit percentage |
| `period` | 30 | Maximum hold period in days |

## Usage Examples

### Basic Usage (Single Ticker)

```bash
python -m src.analysis.breakout_momentum --ticker NVDA
```

### Custom Parameters

```bash
# More aggressive: shorter breakout, tighter stops
python -m src.analysis.breakout_momentum --ticker TSLA \
  --breakout-period 10 \
  --trailing-stop -6.0 \
  --take-profit 20.0
```

### S&P 500 Analysis with Export

```bash
python -m src.analysis.breakout_momentum --sp500 \
  --export results/breakout_momentum.csv
```

### Show Signal Details

```bash
# Show detailed entry/exit table
python -m src.analysis.breakout_momentum --ticker NVDA \
  --show-signal-details \
  --max-signals 15
```

## Output Example

```
===============================================================================
STRATEGY: Breakout Momentum
TICKER: NVDA
DESCRIPTION: 20-day high breakout + RSI>60 + 2.0x volume
===============================================================================

SETTINGS
--------------------------------------------------------------------------------
Breakout period           : 20 days
RSI threshold             : >60
Volume multiplier         : 2.0x
Trailing stop             : -8.0%
Take-profit               : 15.0%
Max hold                  : 30 days

PERFORMANCE SUMMARY
--------------------------------------------------------------------------------
Total Signals             : 28
Winners                   : 16 (57.1%)
Losers                    : 12 (42.9%)
Win Rate                  : 57.1%

Average Return (All)      : +4.2%
Average Winner            : +13.8%
Average Loser             : -6.2%
Risk/Reward Ratio         : 2.2:1

Median Return             : +5.6%
Best Trade                : +15.0%
Worst Trade               : -8.0%

Average Hold Time         : 11.3 days
```

## When to Use This Strategy

### ✓ Good For

- **Bull markets**: Momentum strategies thrive when market is trending up
- **Trending stocks**: Works best with stocks making higher highs
- **Growth sectors**: Tech, biotech, and other high-momentum sectors
- **Institutional buying**: High volume confirms smart money participation
- **Strong trends**: When SMA50 > SMA200 (golden cross)
- **Breakout traders**: Those comfortable buying at new highs

### ❌ Avoid When

- **Bear markets**: Long-only strategy, will underperform in downtrends
- **Choppy markets**: False breakouts lead to whipsaws
- **Low volatility**: Breakouts may lack follow-through
- **Low volume periods**: Summer doldrums, holiday weeks
- **Overextended stocks**: RSI > 80 may indicate exhaustion
- **News uncertainty**: Earnings season, Fed meetings

## Tips for Optimization

1. **Breakout period customization**
   - Shorter period (10-15 days): More signals, more noise
   - Longer period (30-50 days): Fewer signals, higher quality
   - Test multiple periods to find sweet spot for each stock

2. **Volume filter strength**
   - Start with 2.0x for broad market stocks
   - Increase to 3.0x for high-volume stocks (filters false breakouts)
   - Reduce to 1.5x for mid-cap stocks (more signals)

3. **RSI threshold tuning**
   - RSI > 60: Default, good balance
   - RSI > 65: More selective, higher win rate, fewer signals
   - RSI > 55: More signals, but more false breakouts

4. **Trailing stop optimization**
   - Default -8%: Good for volatile stocks
   - -6%: Tighter for less volatile stocks
   - -10%: Wider for giving trends room to breathe
   - Consider ATR-based stops for adaptive risk management

5. **Take-profit target adjustment**
   - Conservative: 10% (quicker exits, more consistent)
   - Moderate: 15% (default, good balance)
   - Aggressive: 20%+ (let winners run, but more period-end exits)

6. **Combine with sector rotation**
   - Run on sector ETFs to identify hot sectors
   - Then apply to individual stocks within those sectors
   - Increases probability of catching strong trends

## Related Strategies

- [**Triple Filter**](/Users/dmitry/_Projects/fin/src/analysis/triple_filter/README.md) - Combines multiple confirmation filters
- [**MACD RSI Confluence**](/Users/dmitry/_Projects/fin/src/analysis/macd_rsi_confluence/README.md) - Alternative momentum confirmation approach
- [**BB Squeeze Breakout**](/Users/dmitry/_Projects/fin/src/analysis/bb_squeeze_breakout/README.md) - Volatility-based breakout detection

## References

### Technical Indicators Used

- **Price Breakout**: 20-day rolling high
  - Indicates institutional accumulation
  - New highs attract momentum buyers

- **Volume**: 14-day average volume
  - 2x average indicates strong conviction
  - Professional traders use volume to confirm breakouts

- **RSI (Relative Strength Index)**: 14-period
  - RSI > 60 shows momentum is strong
  - Filters out weak breakouts that fail immediately

- **SMA200**: Long-term trend filter
  - Only trade breakouts in confirmed uptrends
  - Avoids counter-trend trades

### Strategy Philosophy

- **"Buy high, sell higher"**: Counter-intuitive but powerful
- **Momentum begets momentum**: Strong moves attract more buyers
- **Trailing stops lock in profits**: Key to asymmetric risk/reward
- **Volume is confirmation**: Without volume, breakouts fail

### Educational Resources

- **Book**: "Momentum Masters" by Mark Minervini
- **Concept**: SEPA (Specific Entry Point Analysis)
- **Research**: "Momentum Crashes" by Kent Daniel & Tobias Moskowitz

### Notes

- This strategy is for educational and research purposes only
- Not financial advice - always do your own research
- Momentum strategies can experience sharp drawdowns
- Consider transaction costs - more trades than mean reversion
- Best used in trending market environments
