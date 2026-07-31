# 📌 Sandbox Deployment Quick Reference

**Duration**: 7 Days (July 30 - Aug 6, 2026)
**Capital**: $1,000
**Risk Tolerance**: Moderate

---

## 🚀 Start Bot

```bash
cd ~/Documents/Projects/arbitrage_bot
python src/main_production.py
```

**Expected startup output:**
```
2026-07-30 14:23:45 - INFO - Production bot initialized
2026-07-30 14:23:46 - INFO - Circuit breaker: CLOSED
2026-07-30 14:23:47 - INFO - Scanning markets...
```

---

## 📊 Core Settings (For This Week)

| Parameter | Value | Why |
|-----------|-------|-----|
| API Mode | Sandbox | Play money, no risk |
| Min Profit | $0.25 | Capture moderate opportunities |
| Max Trade | $75 | 7.5% per trade (safe sizing) |
| Daily Loss Limit | $75 | 7.5% daily (two losing days max) |
| Max Positions | 10 | Diversified but not scattered |
| Win Target | 55%+ | Arbitrage should beat 50% |

---

## ✅ Daily Checklist

**Morning**:
- [ ] Start bot
- [ ] Confirm no errors in first 30 seconds
- [ ] Check for "Scanning markets" message

**Throughout Day**:
- [ ] Spot-check logs every 2-3 hours
- [ ] Note approximately how many trades

**Evening**:
- [ ] Stop bot gracefully (`Ctrl+C`)
- [ ] Record: # trades, wins, losses, daily profit
- [ ] Check: Any errors or anomalies?

---

## 📈 Daily Log Template

```
[Date]: _______________
Trades: ___ | Wins: ___ | Losses: ___
Win Rate: ___% | Daily P&L: $______
Issues: [ ] None  [ ] Minor  [ ] Critical
Note: ________________________________________
```

---

## 🎯 By-Day Goals

| Day | Goal | Success Metric |
|-----|------|-----------------|
| 1-2 | System works | No crashes, orders fill |
| 3-5 | Profitable | Win rate ≥55%, profit >$0 |
| 6-7 | Risk works | Max loss limit protects capital |

---

## ⚠️ STOP If You See

- 🛑 Crash / repeated errors → Stop, check logs
- 🛑 No trades for 4+ hours → Check config
- 🛑 Daily loss > $75 → Bot auto-stops (good!)
- 🛑 Win rate < 40% after Day 3 → Adjust settings

---

## 🎓 What Each Part Does

**Bot finds opportunities** where:
```
Buy Price: $0.50
Sell Price: $0.51
Spread: 1% = Profit $0.10 per contract
At 10 contracts = $1.00 profit
```

**Risk Management ensures**:
- Never lose more than $75 in one day
- Never have more than 10 open trades
- Automatically scales down if losing

**Monitoring alerts you** if:
- System health degrades
- Error rates spike
- Daily loss limit approaching

---

## 📞 Quick Reference Links

| File | Purpose |
|------|---------|
| `config.yaml` | All settings |
| `src/main_production.py` | The bot itself |
| `data/arbitrage.db` | Trade history |
| `logs/bot.log` | Detailed logs |

---

## 💾 Graceful Shutdown

```bash
# In the terminal running the bot:
Ctrl+C

# Bot will:
1. Close any open positions
2. Save state
3. Exit cleanly

# Takes ~5 seconds
```

---

## 📊 Weekly Success Criteria

At end of Day 7, check:

| Metric | Target | Status |
|--------|--------|--------|
| Uptime | 7/7 days (0 crashes) | ☐ Pass |
| Win Rate | ≥ 55% | ☐ Pass |
| Total Profit | ≥ $0 | ☐ Pass |
| Largest Drawdown | < 5% of capital | ☐ Pass |
| Risk Controls | Never exceeded | ☐ Pass |

**If all 5 Pass → Ready for Paper Trading!**
**If 3-4 Pass → Continue sandbox, adjust settings**
**If <3 Pass → Pause, debug, re-test**

---

## 🎯 After Sandbox Week

### If Profitable (Recommend):
```
Move to PAPER TRADING
- Use 2-3x larger positions
- Real money account (Kalshi demo balance)
- Run for 7-14 days
- Prepare for production
```

### If Break-Even (Acceptable):
```
Move to PAPER TRADING
- Start with same position sizes
- Optimize parameters based on sandbox data
- Prove consistency over 2 weeks
```

### If Losing (Investigate):
```
Stay in SANDBOX
- Lower min_profit_cents
- Reduce max_position_contracts
- Run another week
- Re-evaluate strategy
```

---

## 🚨 Emergency Stop

If something seems very wrong:

```bash
# Kill the bot
Ctrl+C

# Check what happened
tail -50 logs/bot.log

# If you're concerned about positions:
# Log into Kalshi sandbox directly
# Manually close any open orders
```

---

## 📲 Key Metrics (Track Daily)

- **Win Rate** = (Winning Trades / Total Trades) × 100
  - Target: ≥55%

- **Profit Factor** = Gross Profit / Gross Loss
  - Target: ≥1.5

- **Daily P&L** = Total profit/loss for the day
  - Target: Positive most days

- **Drawdown** = Largest loss from peak
  - Target: <5% ($50 or less)

---

**You're ready to deploy!** 🚀

Current time: 2026-07-30
Sandbox duration: 7 days (through Aug 6)
Expected timeline to production: 3-4 weeks

Good luck! Monitor daily, stay disciplined, and let the system work.
