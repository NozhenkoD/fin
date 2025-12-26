# ATR-Filtered False Breakout Strategy

## Overview

A contrarian mean reversion strategy that detects false breakouts below swing lows and enters long positions expecting a bounce. Uses ATR-based candle size validation to filter for genuine failed breakout patterns. The strategy capitalizes on stop-loss hunts and institutional accumulation below key support levels.

## Strategy Logic

### Entry Conditions

All conditions must be met:

- ✓ Swing low identified (low lower than N previous AND N subsequent lows)
- ✓ Swing low is "naked" (untouched within lookback period)
- ✓ Price closes BELOW swing low (false breakout trigger)
- ✓ Breakout candle range (High - Low) within ATR bounds (1.0x - 2.0x ATR)
- ✓ Enter long at close (expecting bounce back)

**Key Concept**: False breakouts occur when price briefly breaks below support (triggering stops) but quickly reverses, trapping sellers. This creates buying opportunities.

### Exit Conditions

First condition hit determines exit:

- ✓ Stop-loss: Breakout candle low - (1.0 × ATR)
- ✓ Take-profit: Entry + (Risk × Risk/Reward ratio, default 1:1)
- ✓ Period end: 15 days

## Expected Characteristics

- **Win rate**: 50-60% (depends on market conditions)
- **Risk/Reward**: Adjustable (default 1:1, recommend testing 1.5:1)
- **Signal frequency**: Moderate (10-30 per month on S&P 500)
- **Hold time**: 3-10 days average
- **Best for**: Mean reversion traders, range-bound markets

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `atr_period` | 14 | ATR calculation period |
| `swing_strength` | 2 | Candles on each side for swing detection (2 = 5-candle pattern) |
| `min_atr_mult` | 1.0 | Minimum ATR multiplier for breakout candle validation |
| `max_atr_mult` | 2.0 | Maximum ATR multiplier for breakout candle validation |
| `sl_atr_mult` | 1.0 | ATR multiplier for stop-loss distance |
| `risk_reward` | 1.0 | Risk-to-reward ratio (1.0 = 1:1, 1.5 = 1.5:1) |
| `lookback_candles` | 200 | How far back to track swing lows |
| `period` | 15 | Maximum hold period in days |

## Usage Examples

### Basic Usage (Single Ticker)

```bash
python -m src.analysis.atr_filtered_breakout --ticker AAPL --verbose
```

### Custom Parameters

```bash
# Tighter filters, better risk/reward
python -m src.analysis.atr_filtered_breakout --ticker MSFT \
  --swing-strength 3 \
  --min-atr-mult 1.2 \
  --max-atr-mult 1.8 \
  --risk-reward 1.5
```

### S&P 500 Analysis with Export

```bash
python -m src.analysis.atr_filtered_breakout --sp500 \
  --export results/atr_filtered_breakout.csv
```

### Multiple Tickers

```bash
# Test on specific watchlist
python -m src.analysis.atr_filtered_breakout \
  --tickers AAPL MSFT GOOGL NVDA TSLA \
  --verbose
```

## Output Example

```
===============================================================================
Analyzing AAPL
===============================================================================
Loaded 1825 candles from 2018-01-02 to 2025-12-25

Detecting false breakout signals...
  Swing Strength: 2 (=5-candle pattern)
  ATR Filter: 1.0x - 2.0x ATR
  Lookback: 200 candles
Found 23 signals

===============================================================================
STRATEGY: ATR Filtered Breakout
TICKER: AAPL
DESCRIPTION: False breakout detection with ATR filtering
===============================================================================

PERFORMANCE SUMMARY
--------------------------------------------------------------------------------
Total Signals             : 23
Winners                   : 13 (56.5%)
Losers                    : 10 (43.5%)
Win Rate                  : 56.5%

Average Return (All)      : +1.8%
Average Winner            : +6.2%
Average Loser             : -4.1%
Risk/Reward Ratio         : 1.5:1

EXIT TYPE DISTRIBUTION
--------------------------------------------------------------------------------
take_profit               : 11 (47.8%)
stop_loss                 : 9 (39.1%)
period_end                : 3 (13.0%)
```

## When to Use This Strategy

### ✓ Good For

- **Range-bound markets**: False breakouts are common in consolidations
- **Mean reversion traders**: Buying weakness, selling strength
- **High-volume stocks**: Liquid stocks with defined swing levels
- **Institutional stop hunts**: Catching accumulation below support
- **Contrarian traders**: Comfortable going against initial direction
- **Short-term trading**: 3-15 day holding period

### ❌ Avoid When

- **Strong trends**: Breakouts in trends are often real, not false
- **Low volume stocks**: Swing levels less reliable
- **News-driven volatility**: Fundamentals override technicals
- **Gap-heavy stocks**: Swing low detection becomes unreliable
- **Extremely volatile markets**: ATR filter may be too wide
- **Bear markets with no bounces**: Mean reversion requires some stability

## Tips for Optimization

1. **Swing strength calibration**
   - Swing strength = 2 (default): 5-candle pattern (2 before, low, 2 after)
   - Swing strength = 3: 7-candle pattern (more significant levels, fewer signals)
   - Swing strength = 1: 3-candle pattern (more sensitive, more signals, more noise)
   - Higher swing strength = more significant support levels

2. **ATR filter tuning**
   - Default (1.0x - 2.0x): Good balance
   - Tighter (1.2x - 1.8x): Filters out extreme candles, higher quality
   - Wider (0.8x - 2.5x): More signals, but more noise
   - Purpose: Validates breakout candle is "normal" size, not extreme

3. **Risk/Reward optimization**
   - 1:1 (default): Conservative, quick exits
   - 1.5:1 or 2:1: Let winners run longer, fewer winners but bigger
   - 0.75:1: Very conservative, prioritize capital preservation
   - Backtest to find optimal ratio for each stock

4. **Stop-loss placement**
   - Default (1.0x ATR below breakout low): Standard
   - Tighter (0.75x ATR): For less volatile stocks
   - Wider (1.5x ATR): For volatile stocks, avoid premature stops
   - Consider using swing low itself as stop (no ATR buffer)

5. **Lookback period tuning**
   - 200 candles (default): ~10 months of daily data
   - Shorter (100): Focus on recent support levels only
   - Longer (300-400): Include older, potentially stronger levels
   - Trade-off: More data = more levels = more signals (but older levels may be stale)

6. **Add volume confirmation**
   - Require volume spike on false breakout day
   - High volume = more stop-loss triggers = better setup
   - Filter: Volume > 1.5x average on breakout day
   - Increases win rate by filtering weak breakouts

## Related Strategies

- [**RSI Mean Reversion**](/Users/dmitry/_Projects/fin/src/analysis/rsi_mean_reversion/README.md) - Alternative mean reversion approach
- [**ATR Mean Reversion**](/Users/dmitry/_Projects/fin/src/analysis/atr_mean_reversion/README.md) - ATR-based dynamic stops for mean reversion
- [**BB Squeeze Breakout**](/Users/dmitry/_Projects/fin/src/analysis/bb_squeeze_breakout/README.md) - Contrasts with this (real vs false breakouts)

## References

### Technical Indicators Used

- **Swing Lows**: Support levels identified by local minima
  - Detected using rolling window comparison
  - "Naked" swing lows haven't been touched since formation
  - More significant than arbitrary price levels

- **ATR (Average True Range)**: 14-period
  - Measures volatility for candle size validation
  - Ensures breakout candle is normal-sized (not extreme gap)
  - Used for adaptive stop-loss placement

### Strategy Philosophy

- **False breakouts trap traders**: Stop-loss hunts create opportunities
- **Support levels matter**: Swing lows represent prior decision points
- **Mean reversion**: Price tends to revert after brief violations
- **Risk management**: ATR-based stops adapt to volatility
- **Patience required**: Not all breakouts are false - need filters

### Pattern Recognition

```
Price action visualization:

      (Swing Low)
          ↓
    ------x------
          |  ← Price briefly breaks below
          |     (false breakout)
          ↓
          █  ← Breakout candle
             (closes below swing low)
          ↑
          |  ← Price bounces back
    ------x------ (entry at close)
```

### Educational Resources

- **Concept**: "Stop Hunting" - institutional accumulation below support
- **Research**: False breakouts occur 30-40% of the time at swing levels
- **Book**: "Market Wizards" - Many traders discuss fading false breakouts
- **Tool**: Market structure analysis

### Notes

- This strategy is for educational and research purposes only
- Not financial advice - always do your own research
- False breakout detection requires clean data and defined swings
- Works best in stocks with visible support/resistance levels
- Consider market regime - false breakouts more common in ranges
- Combine with volume analysis for best results
- Ticket ID: ALGO-001 (original implementation reference)
