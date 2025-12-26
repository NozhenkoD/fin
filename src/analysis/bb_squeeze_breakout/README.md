# Bollinger Band Squeeze Breakout Strategy

## Overview

A volatility contraction/expansion strategy that identifies periods of unusually low volatility (the "squeeze") and enters on the breakout. When Bollinger Bands contract to historically narrow levels, energy builds up for a significant price move. The strategy combines squeeze detection with volume and momentum confirmation to catch explosive moves.

## Strategy Logic

### Entry Conditions

All conditions must be met:

- ✓ Price above SMA200 (long-term uptrend)
- ✓ Price above SMA200 for minimum 15 days (sustained uptrend)
- ✓ Bollinger Band Width at 20-day low (squeeze detected)
- ✓ Price breaks above upper Bollinger Band (breakout direction confirmed)
- ✓ Volume > 1.5x average (confirms conviction)
- ✓ RSI > 50 (momentum supports breakout direction)

### Exit Conditions

First condition hit determines exit:

- ✓ Trailing stop: 2.0 × ATR from highest high (locks in profits)
- ✓ Middle band touch after profit (mean reversion exit)
- ✓ Period end: 20 days

**Key Concept**: The squeeze is detected when Bollinger Band Width (BBW) reaches its lowest point in 20 days, signaling volatility contraction before expansion.

## Expected Characteristics

- **Win rate**: 50-55%
- **Average winner**: Large (rides volatility expansion)
- **Risk/Reward**: 2:1 to 3:1
- **Signal frequency**: Lower than mean reversion (quality over quantity)
- **Hold time**: 8-18 days average
- **Best for**: Catching explosive moves after consolidation

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `squeeze_lookback` | 20 | Days to detect BB Width minimum (squeeze period) |
| `volume_multiplier` | 1.5 | Minimum volume vs average |
| `rsi_threshold` | 50 | Minimum RSI for momentum confirmation |
| `min_days_above` | 15 | Minimum consecutive days above SMA200 |
| `trailing_stop_atr` | 2.0 | ATR multiplier for trailing stop |
| `use_middle_band_exit` | True | Enable middle band touch exit (after profit) |
| `period` | 20 | Maximum hold period in days |

## Usage Examples

### Basic Usage (Single Ticker)

```bash
python -m src.analysis.bb_squeeze_breakout --ticker AAPL
```

### Custom Parameters

```bash
# Tighter squeeze, wider stops
python -m src.analysis.bb_squeeze_breakout --ticker NVDA \
  --squeeze-lookback 30 \
  --trailing-stop-atr 3.0 \
  --rsi-threshold 55
```

### S&P 500 Analysis with Export

```bash
python -m src.analysis.bb_squeeze_breakout --sp500 \
  --export results/bb_squeeze.csv
```

### Disable Middle Band Exit

```bash
# Use only trailing stop (let winners run longer)
python -m src.analysis.bb_squeeze_breakout --ticker TSLA \
  --no-middle-band-exit
```

### Show Signal Details

```bash
# Show detailed entry/exit table
python -m src.analysis.bb_squeeze_breakout --ticker AAPL \
  --show-signal-details \
  --max-signals 15
```

## Output Example

```
===============================================================================
STRATEGY: BB Squeeze Breakout
TICKER: AAPL
DESCRIPTION: BB Width at 20-day low + upper band breakout
===============================================================================

SETTINGS
--------------------------------------------------------------------------------
Squeeze lookback          : 20 days
RSI threshold             : >50
Volume multiplier         : 1.5x
Trailing stop             : 2.0x ATR
Middle band exit          : enabled
Max hold                  : 20 days

PERFORMANCE SUMMARY
--------------------------------------------------------------------------------
Total Signals             : 18
Winners                   : 10 (55.6%)
Losers                    : 8 (44.4%)
Win Rate                  : 55.6%

Average Return (All)      : +3.8%
Average Winner            : +11.2%
Average Loser             : -5.4%
Risk/Reward Ratio         : 2.1:1

EXIT TYPE DISTRIBUTION
--------------------------------------------------------------------------------
trailing_stop             : 6 (33.3%)
middle_band               : 4 (22.2%)
period_end                : 8 (44.4%)
```

## When to Use This Strategy

### ✓ Good For

- **Consolidation breakouts**: Designed to catch explosive moves after tight ranges
- **Trending stocks**: Works best when breakout has room to run
- **Volatility traders**: Capitalizes on volatility expansion
- **Patient traders**: Signals are less frequent but higher quality
- **Swing to position trading**: 10-20 day holding period
- **Growth stocks**: Tech and biotech often have dramatic squeezes

### ❌ Avoid When

- **Constant high volatility**: Squeeze rarely forms in always-volatile stocks
- **Bear markets**: Long-only, needs uptrends for breakout follow-through
- **Low volume stocks**: Volume filter becomes unreliable
- **Range-bound markets**: Breakouts fail without trend
- **News-driven stocks**: Fundamentals override technical squeeze
- **Pre-earnings**: Squeeze may resolve with earnings gap

## Tips for Optimization

1. **Squeeze detection refinement**
   - Default 20-day lookback works for most stocks
   - Increase to 30-40 days for more selective squeezes
   - Decrease to 10-15 days for faster-moving stocks
   - Add requirement: BBW must be in bottom 20th percentile of 100-day range

2. **Volume confirmation strength**
   - Start at 1.5x for standard confirmation
   - Increase to 2.0x for high-volume stocks (filters noise)
   - Combine with volume trend: require 3-day volume increase
   - Check volume at breakout AND during squeeze formation

3. **RSI threshold tuning**
   - RSI > 50: Default, neutral/bullish filter
   - RSI > 55: More selective, stronger momentum
   - RSI > 60: Very selective, catches only strong breakouts
   - RSI 45-60 range: Allow some mean-reversion plays

4. **Trailing stop optimization**
   - 2.0x ATR: Default, good balance
   - 1.5x ATR: Tighter for less volatile stocks
   - 2.5-3.0x ATR: Wider for volatile stocks, let winners run
   - Consider volatility-adjusted: tighten as BBW expands

5. **Middle band exit strategy**
   - Enable for mean-reversion exit (default)
   - Disable to let momentum plays run further
   - Partial exit: 50% at middle band, rest trails
   - Only use after minimum profit (e.g., +5%)

6. **Combine with other squeeze indicators**
   - Add Keltner Channels squeeze (BB inside KC)
   - Monitor TTM Squeeze indicator
   - Check for decrease in Average True Range
   - Confluence of squeeze indicators = higher probability

## Related Strategies

- [**Breakout Momentum**](/Users/dmitry/_Projects/fin/src/analysis/breakout_momentum/README.md) - Alternative breakout approach without squeeze
- [**ATR Filtered Breakout**](/Users/dmitry/_Projects/fin/src/analysis/atr_filtered_breakout/README.md) - ATR-based breakout validation
- [**MACD RSI Confluence**](/Users/dmitry/_Projects/fin/src/analysis/macd_rsi_confluence/README.md) - Momentum confirmation alternative

## References

### Technical Indicators Used

- **Bollinger Bands**: 20-period, 2 standard deviations
  - Upper Band (BBU): SMA20 + (2 × std dev)
  - Middle Band (BBM): SMA20
  - Lower Band (BBL): SMA20 - (2 × std dev)
  - Band Width (BBW): (BBU - BBL) / BBM
  - Squeeze: BBW at multi-week lows

- **RSI (Relative Strength Index)**: 14-period
  - > 50: Bullish momentum supports upside breakout
  - Filters breakouts that immediately fail
  - Directional confirmation

- **ATR (Average True Range)**: 14-period
  - Volatility measure for adaptive trailing stops
  - 2.0x ATR gives room for normal volatility
  - Locks in profits as price makes new highs

- **Volume**: 14-day average
  - Breakout on high volume = institutional participation
  - Low volume breakouts often fail
  - Confirms conviction behind the move

- **SMA200**: Long-term trend filter
  - Only trade squeezes in uptrends
  - Breakouts have room to run

### Strategy Philosophy

- **Volatility cycles**: Periods of low volatility precede high volatility
- **Energy coiling**: Tight ranges build pressure for explosive moves
- **Direction matters**: Wait for breakout direction confirmation
- **Volume validates**: Without volume, breakouts are false
- **Let winners run**: Trailing stops maximize profit on big moves

### Educational Resources

- **Book**: "Bollinger on Bollinger Bands" by John Bollinger
- **Concept**: "The Squeeze" - John Carter, Simpler Trading
- **Research**: Volatility contraction followed by expansion is well-documented
- **Tool**: TTM Squeeze indicator popularized this concept

### Notes

- This strategy is for educational and research purposes only
- Not financial advice - always do your own research
- Squeeze setups are less frequent than other strategies
- When they hit, they can produce outsized returns
- Patience required - don't force trades when no squeeze exists
- Best combined with fundamental catalysts (product launches, etc.)
