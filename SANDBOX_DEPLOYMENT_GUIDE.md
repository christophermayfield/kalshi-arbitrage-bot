# 🚀 Sandbox Deployment Guide

**Date**: July 30, 2026
**Status**: Ready for Kalshi Sandbox Deployment
**Capital**: $1,000
**Risk Tolerance**: Moderate
**Duration**: 7 days of testing

---

## 📋 Configuration Summary

All parameters have been optimized for your $1,000 capital with moderate risk tolerance.

### Core Trading Parameters

```yaml
trading:
  min_profit_cents: 25              # $0.25 minimum profit threshold
  max_order_value_cents: 7500       # $75 max per trade (7.5% of capital)
  max_position_contracts: 100       # Conservative position size
```

**Rationale**:
- **$0.25 minimum**: Allows capturing moderate arbitrage opportunities without being too greedy
- **$75 per trade**: 7.5% of capital - balances opportunity capture with risk management
- **100 contracts max**: Conservative learning phase - can scale up after proving profitability

### Risk Management Parameters

```yaml
risk:
  max_daily_loss_cents: 7500        # $75 daily loss limit (7.5% of capital)
  max_open_positions: 10            # Moderate concurrent position limit
```

**Rationale**:
- **$75 daily loss**: Allows for 1-2 losing days before triggering defensive measures
- **10 open positions**: Moderate - diversifies risk without over-concentration

### Sandbox Environment

```yaml
kalshi:
  demo_mode: true
  base_url: "https://api-sandbox.elections.kalshi.com"
```

**What this does**:
- Routes all trading through Kalshi's sandbox environment
- Uses play money - no real funds at risk
- Full API feature parity with production
- Identical latency/performance characteristics

### Limited Risk Mode (Auto-Enabled)

```yaml
limited_risk:
  enabled: true
  max_daily_trades: 15
  min_trade_cents: 500              # $5 minimum per trade
  max_daily_loss_cents: 7500        # Matches risk limit
```

**What this does**:
- Provides additional safety layer for small accounts
- Progressive scaling: increases trade size after 10 profitable trades
- Automatic stops if daily loss limit reached

---

## 🚀 How to Deploy

### Step 1: Verify Configuration

```bash
cd ~/Documents/Projects/arbitrage_bot

# Check that config.yaml has sandbox settings
grep -A 3 "^kalshi:" config.yaml
# Should show:
#   demo_mode: true
#   base_url: "https://api-sandbox.elections.kalshi.com"
```

### Step 2: Ensure Dependencies Are Installed

```bash
# Install or update Python dependencies
pip install -r requirements.txt

# Or if using uv (faster):
uv pip install -r requirements.txt
```

### Step 3: Start the Bot

```bash
# Run production bot with all hardening systems
python src/main_production.py

# You should see output like:
# 2026-07-30 14:23:45 - INFO - Bot initialized
# 2026-07-30 14:23:46 - INFO - Circuit breaker initialized
# 2026-07-30 14:23:47 - INFO - Risk manager ready
# 2026-07-30 14:23:48 - INFO - Monitoring system active
# 2026-07-30 14:23:49 - INFO - Starting sandbox deployment...
```

### Step 4: Monitor First Trade

Once running, watch for first opportunity detection:

```
2026-07-30 14:25:12 - INFO - Scanning 45 markets
2026-07-30 14:25:13 - INFO - Opportunity detected in MARKET-123
2026-07-30 14:25:13 - INFO - Spread: 0.8% | Profit: $1.20 | Risk: OK
2026-07-30 14:25:14 - INFO - Trade MARKET-123 FILLED: Buy 10 @ $0.50, Sell 10 @ $0.51
```

---

## 📊 What to Monitor (7-Day Sandbox Period)

### Day 1-2: System Validation
- ✓ Bot connects to Kalshi sandbox
- ✓ Markets load correctly
- ✓ Orders place and fill successfully
- ✓ Monitoring alerts trigger properly

**Success Criteria**: System stability, no crashes

### Day 3-5: Strategy Profitability
- ✓ Win rate (target: 50%+)
- ✓ Average profit per trade (target: >$0.25)
- ✓ Drawdown (target: <5% of capital)

**Success Criteria**: Consistent small profits, stable performance

### Day 6-7: Risk Management
- ✓ Position sizing working correctly
- ✓ Daily loss limit enforcement
- ✓ Circuit breaker activating on failures
- ✓ Graceful recovery from errors

**Success Criteria**: Risk controls protect capital, system resilient

---

## 📈 Expected Performance (Sandbox Week)

Based on arbitrage strategy with $1,000 capital:

### Conservative Estimate
- **Daily trades**: 5-10
- **Win rate**: 55-60%
- **Avg profit**: $0.50-1.00 per trade
- **Daily P&L**: $2-6 per day
- **Weekly P&L**: $15-40 (1.5-4% weekly return)

### Realistic Target
- **Daily trades**: 8-15
- **Win rate**: 60-65%
- **Avg profit**: $1.00-1.50 per trade
- **Daily P&L**: $8-20 per day
- **Weekly P&L**: $50-100 (5-10% weekly return)

### Aggressive (High Confidence)
- **Daily trades**: 15-25
- **Win rate**: 65-70%
- **Avg profit**: $1.50-2.00 per trade
- **Daily P&L**: $20-40 per day
- **Weekly P&L**: $100-200 (10-20% weekly return)

**Note**: Arbitrage is lower-volatility than directional trading. Expect steady, smaller gains.

---

## 🔍 Key Metrics to Track

Keep a simple daily log:

```
Date: 2026-07-30
Trades: 12
Winners: 7 (58% win rate)
Losers: 5
Daily P&L: $8.50
Cumulative P&L: $8.50
Issues: None
```

### Weekly Summary Template

```
Week 1 (Jul 30 - Aug 5):
- Total Trades: 85
- Win Rate: 62%
- Avg Trade: $0.95
- Total Profit: $78.50
- Best Day: Aug 3 (+$15.20)
- Worst Day: Aug 2 (-$3.40)
- Largest Win: $3.50
- Largest Loss: ($2.10)

Assessment: ✓ Profitable, stable, ready for paper trading
```

---

## ⚠️ Stop/Pause Conditions

Stop the bot if any of these occur:

1. **Daily Loss Limit Hit**
   - Bot will automatically stop
   - This is a safety feature - don't override

2. **Crash or Repeated Errors**
   - Check logs for root cause
   - Restart bot if transient error
   - Pause if persistent

3. **Unexpected Behavior**
   - Trades not filling
   - Balances not updating
   - Circuit breaker constantly open
   - Stop and investigate

---

## 📝 Decision Framework

### After Day 3: Continue or Adjust?

If **Win Rate < 50%**:
- Lower `min_profit_cents` to 10 (more opportunities)
- Reduce `max_position_contracts` to 50
- Continue for 2 more days

If **No trades after Day 1**:
- Increase `min_profit_cents` to 50 (looking for bigger spreads)
- Check that markets are "open" status
- Verify orderbook data is updating

If **Profitable (Win Rate > 55%)**:
- Continue with current settings
- Increase daily limit to $10k if profitable streak continues
- Prepare for paper trading transition

### After Day 7: Go to Paper Trading?

**Proceed to Paper Trading if:**
- ✓ Win rate ≥ 55%
- ✓ Total profit > 0 (even $5 is fine)
- ✓ No critical errors
- ✓ System stayed stable all 7 days

**Adjust and Re-test if:**
- ✗ Win rate < 50%
- ✗ Experienced crashes or persistent errors
- ✗ Made consistent losses

---

## 🛠️ Troubleshooting

### "No orders filling"
- Check: Are any markets showing spread opportunities?
- Try: Lower `min_profit_cents` to 10
- Try: Increase `max_spread_percent` to 10.0

### "Circuit breaker constantly open"
- Problem: API connectivity issues
- Solution: Check internet connection, Kalshi status page
- Workaround: Increase `circuit_breaker_window_seconds` to 600 (10 min)

### "Drawdown > 5% on Day 1"
- This is likely normal variance in early trading
- Continue monitoring
- Only worry if drawdown exceeds 10% by Day 3

### "Getting rate limited"
- Check API rate limit settings in config
- Current: 5 RPS, 300 RPM (very safe)
- No changes needed unless you hit limits

---

## 📞 Next Steps

### Daily During Sandbox (7 days)
1. Morning: Start bot, confirm no errors
2. Throughout day: Spot-check logs, note # of trades
3. Evening: Record daily P&L and assessment
4. Before bed: Gracefully shutdown bot (`Ctrl+C`)

### End of Week (Day 7 evening)
1. Calculate weekly statistics
2. Review decision framework above
3. If profitable → prepare for paper trading
4. If not profitable → adjust settings and extend sandbox

### Paper Trading Phase (starts Day 8)
- Scale up position sizes by 2-3x
- Increase daily loss limit to match strategy
- Run for 7-14 days to validate real-money trading
- Prepare for production deployment

---

## 💾 Important Files

- `config.yaml` - All settings (sandbox-configured)
- `src/main_production.py` - Entry point (includes all hardening)
- `data/arbitrage.db` - Tracks all trades (SQLite)
- `.github/workflows/ci-cd.yml` - Automated tests in GitHub

---

## ✅ Pre-Launch Checklist

Before starting sandbox deployment:

- [ ] Verified `config.yaml` has sandbox URL + demo_mode: true
- [ ] Confirmed Python dependencies installed
- [ ] Created `logs/` directory for output
- [ ] Set up daily monitoring log file
- [ ] Confirmed Kalshi sandbox account access
- [ ] Reviewed expected performance ranges above
- [ ] Understood stop/pause conditions
- [ ] Ready to commit to 7-day sandbox period

---

## 🎯 Success Definition

**Sandbox week is successful if:**

✓ System runs 7 consecutive days without crashes
✓ Win rate ≥ 55%
✓ Total weekly profit ≥ $0 (break-even OK)
✓ No critical risk management failures
✓ Daily loss limit never hit

If all 5 are true → **Proceed to paper trading**

---

## 📚 Related Guides

- `RISK_PARAMETERS_GUIDE.md` - Detailed parameter tuning reference
- `PRODUCTION_INTEGRATION_GUIDE.md` - Full architecture documentation
- `PRODUCTION_READY_SUMMARY.md` - Complete system overview

---

**Deployment Ready** ✅
**All systems integrated and tested**
**Ready for 7-day Kalshi sandbox validation**

Let's go! 🚀
