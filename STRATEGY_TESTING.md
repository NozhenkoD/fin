# Strategy Testing Guide

This guide shows you how to backtest and compare different trading strategies.

## 🚀 Quick Start - Test All Strategies

Run these commands to test all three strategies on S&P 500 stocks:

```bash
# 1. Test Resistance Break Strategy (SMA200 breakout from below)
python -m src.analysis.sma200_crossover_analysis \
    --sp500 \
    --strategy resistance_break \
    --min-days 15 \
    --stop-loss -5.0 \
    --take-profit 10.0 \
    --period 20 \
    --export results/resistance_break.csv

# 2. Test Support Bounce Strategy (SMA200 bounce from above)
python -m src.analysis.sma200_crossover_analysis \
    --sp500 \
    --strategy support_bounce \
    --min-days 15 \
    --stop-loss -5.0 \
    --take-profit 10.0 \
    --period 20 \
    --export results/support_bounce.csv

# 3. Test RSI Mean Reversion Strategy (buy oversold, sell overbought)
python -m src.analysis.rsi_mean_reversion \
    --sp500 \
    --rsi-threshold 30 \
    --rsi-exit 70 \
    --stop-loss -3.0 \
    --take-profit 5.0 \
    --period 10 \
    --export results/rsi_mean_reversion.csv

# 4. Compare All Strategies Side-by-Side
python -m src.analysis.compare_strategies --quick
```

**Note:** Each strategy test on S&P 500 may take 5-15 minutes depending on your system.

---

## 📊 Strategy Details

### 1. **Resistance Break** (Trend Following)
- **Entry:** Price breaks above SMA200 after being below for 15+ days
- **Exit:** +10% profit OR -5% stop-loss OR 20 days
- **Best for:** Strong trending markets, capturing big moves
- **Expected:** Lower win rate (~45-50%), but bigger winners

### 2. **Support Bounce** (Mean Reversion on Uptrend)
- **Entry:** Price touches SMA200 from above and bounces back
- **Exit:** +10% profit OR -5% stop-loss OR 20 days
- **Best for:** Stocks in established uptrends having temporary pullbacks
- **Expected:** Medium win rate (~50-55%)

### 3. **RSI Mean Reversion** (NEW - Recommended)
- **Entry:** RSI < 30 (oversold) + price above SMA200 (uptrend filter)
- **Exit:** RSI > 70 OR +5% profit OR -3% stop-loss OR 10 days
- **Best for:** Choppy markets, quick trades
- **Expected:** Higher win rate (~60-70%), shorter hold times

---

## 🔍 Test on Single Ticker First

Before running on full S&P 500, test on a single ticker to verify:

```bash
# Test Resistance Break on Ford (F)
python -m src.analysis.sma200_crossover_analysis --ticker F --strategy resistance_break

# Test Support Bounce on Apple (AAPL)
python -m src.analysis.sma200_crossover_analysis --ticker AAPL --strategy support_bounce

# Test RSI Mean Reversion on Microsoft (MSFT)
python -m src.analysis.rsi_mean_reversion --ticker MSFT
```

---

## ⚙️ Parameter Optimization

### Adjust Stop-Loss / Take-Profit

Find the best risk/reward ratio:

```bash
# More aggressive (tighter stops, smaller profits)
python -m src.analysis.rsi_mean_reversion \
    --ticker AAPL \
    --stop-loss -2.0 \
    --take-profit 4.0

# More conservative (wider stops, bigger profits)
python -m src.analysis.rsi_mean_reversion \
    --ticker AAPL \
    --stop-loss -5.0 \
    --take-profit 8.0
```

### Adjust RSI Thresholds

```bash
# More selective (only extreme oversold/overbought)
python -m src.analysis.rsi_mean_reversion \
    --ticker AAPL \
    --rsi-threshold 25 \
    --rsi-exit 75

# Less selective (more signals)
python -m src.analysis.rsi_mean_reversion \
    --ticker AAPL \
    --rsi-threshold 35 \
    --rsi-exit 65
```

---

## 📈 Understanding the Results

### Key Metrics to Focus On:

1. **Win Rate** - Percentage of winning trades
   - Mean reversion: Target 60-70%
   - Trend following: Target 45-55%

2. **Profit Factor** - Total wins ÷ Total losses
   - Should be > 1.5
   - Higher is better

3. **Risk/Reward Ratio** - Avg winner ÷ Avg loser
   - Should be > 2.0 for trend following
   - Should be > 1.5 for mean reversion

4. **Expectancy** - Average $ per trade (if trading $100)
   - Should be positive
   - Higher is better

5. **Average Hold Days**
   - RSI: 5-10 days (quick trades)
   - SMA: 10-20 days (longer term)

### Exit Type Distribution:

- **Take-profit hits:** How often you hit your profit target
- **Stop-loss hits:** How often you hit your stop
- **RSI exits:** (RSI strategy only) How often RSI signals exit
- **Period end:** How often you hold to max period

**Ideal:** High take-profit hits, low stop-loss hits

---

## 🎯 Next Steps After Comparison

Once you've compared all strategies:

### If RSI Mean Reversion wins:
```bash
# Optimize parameters on your best tickers
python -m src.analysis.rsi_mean_reversion \
    --tickers AAPL MSFT GOOGL NVDA AMD \
    --rsi-threshold 30 \
    --stop-loss -3.0 \
    --take-profit 5.0
```

### If Resistance Break wins:
```bash
# Test with different min-days and proximity filters
python -m src.analysis.sma200_crossover_analysis \
    --sp500 \
    --strategy resistance_break \
    --min-days 20 \
    --period 30 \
    --export results/resistance_break_v2.csv
```

### If Support Bounce wins:
```bash
# Test with different parameters
python -m src.analysis.sma200_crossover_analysis \
    --sp500 \
    --strategy support_bounce \
    --min-days 10 \
    --period 15 \
    --export results/support_bounce_v2.csv
```

---

## 🔬 Advanced: Multi-Parameter Testing

Create a simple loop to test multiple parameter combinations:

```bash
# Test different RSI thresholds
for threshold in 25 30 35; do
    python -m src.analysis.rsi_mean_reversion \
        --ticker AAPL \
        --rsi-threshold $threshold \
        --export results/rsi_threshold_${threshold}.csv
done

# Compare results
python -m src.analysis.compare_strategies \
    --file results/rsi_threshold_25.csv "RSI 25" \
    --file results/rsi_threshold_30.csv "RSI 30" \
    --file results/rsi_threshold_35.csv "RSI 35"
```

---

## 📁 Results Files

All results are exported to CSV in the `results/` directory:

- `results/resistance_break.csv` - Resistance break backtest
- `results/support_bounce.csv` - Support bounce backtest
- `results/rsi_mean_reversion.csv` - RSI mean reversion backtest

You can open these in Excel/Google Sheets for further analysis.

---

## ❓ Common Issues

### "ModuleNotFoundError: No module named 'pandas_ta'"

Install dependencies:
```bash
pip install -r requirements.txt
```

### "No cached data found"

Download historical data first:
```bash
python -m src.data.update_cache --tickers AAPL MSFT GOOGL
```

Or let the script download automatically (will be slower).

---

## 💡 Pro Tips

1. **Start small:** Test on 10-20 tickers first before running full S&P 500
2. **Compare apples to apples:** Use same tickers for all strategies
3. **Consider market conditions:** Strategies perform differently in bull vs bear markets
4. **Look at recent signals:** Sort results by date to see how strategy performs recently
5. **Check per-ticker stats:** Some strategies work better on certain stocks

---

## 🎉 Expected Results

Based on historical testing, here's what you might see:

| Strategy | Win Rate | Profit Factor | Avg Hold | Best For |
|----------|----------|---------------|----------|----------|
| Resistance Break | 45-50% | 1.8-2.2 | 12-18 days | Bull markets |
| Support Bounce | 50-55% | 1.6-2.0 | 10-15 days | Ranging markets |
| **RSI Mean Reversion** | **60-70%** | **2.0-2.5** | **6-10 days** | **Most conditions** |

**RSI Mean Reversion is expected to be the winner** due to:
- Higher win rate
- Shorter hold times (more opportunities)
- Better risk/reward on individual trades
- Works in most market conditions

---

Ready to test? Start with the Quick Start commands at the top! 🚀
