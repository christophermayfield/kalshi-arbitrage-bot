# 📊 Risk Parameters Tuning Guide

**Purpose**: Configure bot risk management before Kalshi sandbox deployment
**Current Settings**: Based on balance of profitability and safety
**Adjustable**: After sandbox testing and validation

---

## 🎯 Key Risk Parameters

### 1. **Daily Loss Limit** (Most Important)
```yaml
risk:
  max_daily_loss_cents: 10000  # $100 per day
```

**What it does**: Bot stops trading after losing $X in a day
**Recommended values**:
- Conservative: $50 (5000)
- Standard: $100 (10000) ← Current
- Aggressive: $250 (25000)

**How to choose**:
- Start with: 2-5% of trading capital
- Sandbox testing: Use $100 limit
- Paper trading: Match your comfort level
- Production: Based on sandbox performance

**Formula**:
```
Daily Loss Limit = Capital × Risk %
Example: $10,000 capital × 2% = $200 daily limit
```

---

### 2. **Maximum Open Positions**
```yaml
risk:
  max_open_positions: 50
```

**What it does**: Never have more than N concurrent trades
**Recommended values**:
- Conservative: 5-10 positions
- Standard: 20-50 positions ← Current
- Aggressive: 50+ positions

**How to choose**:
- Sandbox: Start with 10-20
- Paper: Increase to 30-50 if profitable
- Production: Match capital and risk appetite

**Why it matters**:
- Prevents over-concentration
- Limits correlation risk
- Reduces capital tied up

---

### 3. **Maximum Position Value**
```yaml
trading:
  max_order_value_cents: 10000  # $100 max per order
```

**What it does**: Single trade can't exceed $X
**Recommended values**:
- Conservative: $50 (5000)
- Standard: $100-$250 (10000-25000) ← Current
- Aggressive: $500+ (50000)

**How to choose**:
- Sandbox: Start with $100
- Paper: Increase to $250-500 if stable
- Production: 1-3% of trading capital

---

### 4. **Minimum Profit Threshold**
```yaml
trading:
  min_profit_cents: 10  # $0.10 minimum profit
```

**What it does**: Only trade opportunities with profit > $X
**Recommended values**:
- Tight markets: $0.10 (10)
- Normal markets: $0.50 (50)
- Conservative: $1.00 (100)

**How to choose**:
- Sandbox: Start with $0.10
- Paper: Increase if unprofitable
- Production: Match market conditions

**Trade-off**:
- Lower = More trades but smaller profits
- Higher = Fewer trades but better margins

---

### 5. **Maximum Position Contracts**
```yaml
trading:
  max_position_contracts: 1000
```

**What it does**: Never buy more than N contracts in one market
**Recommended values**:
- Conservative: 100-500 contracts
- Standard: 500-1000 contracts ← Current
- Aggressive: 1000+ contracts

**Market context**: Kalshi markets typically have 100-contract minimums

---

## 🎲 Risk Tier System

The bot automatically adjusts risk based on recent performance:

```
CONSERVATIVE TIER
├─ Triggered by: Recent losses or drawdown > 20%
├─ Behavior: Reduce position sizes by 50%
├─ Duration: Until profitable again
└─ Min profit requirement: 2x normal

NORMAL TIER  ← Default
├─ Behavior: Standard position sizes
├─ Daily loss limit: Full limit ($100)
└─ Min profit requirement: Normal

AGGRESSIVE TIER
├─ Triggered by: Winning streak (>70% win rate)
├─ Behavior: Increase position sizes by 25%
├─ Duration: While winning streak continues
└─ Enhanced profitability focus
```

---

## 📋 Recommended Configurations

### For Sandbox Testing (First 7 days)
```yaml
trading:
  min_profit_cents: 10           # Low threshold to test opportunities
  max_order_value_cents: 10000   # $100 max per trade
  max_position_contracts: 100    # Conservative size

risk:
  max_daily_loss_cents: 10000    # $100 daily limit
  max_open_positions: 20         # Limit concurrent trades

orderbook:
  min_fill_probability: 0.8      # 80% fill probability minimum
  max_spread_percent: 5.0        # 5% max spread
```

**Goal**: Validate all systems work without big losses

---

### For Paper Trading (Next 7-14 days)
```yaml
trading:
  min_profit_cents: 50           # Slightly higher threshold
  max_order_value_cents: 25000   # $250 max per trade
  max_position_contracts: 500    # More aggressive

risk:
  max_daily_loss_cents: 50000    # $500 daily limit
  max_open_positions: 30         # More concurrent positions

orderbook:
  min_fill_probability: 0.85     # Stricter fill probability
  max_spread_percent: 3.0        # Stricter spread limit
```

**Goal**: Test strategy profitability and fine-tune parameters

---

### For Production (After validation)
```yaml
trading:
  min_profit_cents: 100          # Higher threshold
  max_order_value_cents: 50000   # $500 max per trade
  max_position_contracts: 1000   # Full capacity

risk:
  max_daily_loss_cents: 100000   # $1,000 daily limit
  max_open_positions: 50         # Max concurrent positions

orderbook:
  min_fill_probability: 0.90     # Strict fill probability
  max_spread_percent: 2.0        # Tight spread limit
```

**Goal**: Maximize profitability while maintaining safety

---

## 🔢 Capital-Based Recommendations

### If Trading Capital = $1,000
```yaml
trading:
  max_order_value_cents: 5000    # $50 per trade (5% of capital)
  min_profit_cents: 25           # $0.25 minimum

risk:
  max_daily_loss_cents: 5000     # $50 daily (5% of capital)
  max_open_positions: 5          # Very conservative
```

### If Trading Capital = $10,000
```yaml
trading:
  max_order_value_cents: 25000   # $250 per trade (2.5% of capital)
  min_profit_cents: 50           # $0.50 minimum

risk:
  max_daily_loss_cents: 10000    # $100 daily (1% of capital)
  max_open_positions: 20         # Conservative
```

### If Trading Capital = $50,000
```yaml
trading:
  max_order_value_cents: 50000   # $500 per trade (1% of capital)
  min_profit_cents: 100          # $1.00 minimum

risk:
  max_daily_loss_cents: 50000    # $500 daily (1% of capital)
  max_open_positions: 50         # Full capacity
```

### If Trading Capital = $100,000+
```yaml
trading:
  max_order_value_cents: 100000+ # Larger sizes possible
  min_profit_cents: 100          # $1.00 minimum

risk:
  max_daily_loss_cents: 100000+  # Flexible limits
  max_open_positions: 50         # Max concurrent positions
```

---

## ⚙️ Advanced Parameters

### Order Execution
```yaml
trading:
  order_timeout_seconds: 30      # Wait max 30s for order fill
  retry_attempts: 3              # Retry failed orders 3x
  retry_delay_seconds: 1         # 1 second between retries
```

### Orderbook Quality
```yaml
orderbook:
  min_liquidity_score: 50        # Minimum liquidity requirement
  max_slippage_percent: 2.0      # Max 2% slippage acceptable
  min_fill_probability: 0.8      # 80% estimated fill rate
  max_spread_percent: 5.0        # Max 5% bid-ask spread
```

### Circuit Breaker
```yaml
risk:
  circuit_breaker_threshold: 5   # Open after 5 consecutive failures
  circuit_breaker_window_seconds: 300  # Reset after 5 minutes
```

---

## 🧪 Testing Strategy

### Phase 1: Sandbox (Days 1-7)
```
Start Conservative:
  Daily Loss: $100
  Max Position: $100
  Min Profit: $0.10
  Max Open: 10

Monitor:
  ✓ Win rate
  ✓ Average profit per trade
  ✓ Drawdown profile
  ✓ Error rates

Adjust if:
  - Win rate < 40% → Lower min_profit_cents
  - Win rate > 70% → Can be more aggressive
  - Frequent timeouts → Increase order_timeout_seconds
```

### Phase 2: Paper Trading (Days 8-21)
```
Increase Gradually:
  Daily Loss: $500
  Max Position: $250
  Min Profit: $0.50
  Max Open: 30

Monitor:
  ✓ Consistent profitability
  ✓ Risk-adjusted returns
  ✓ Drawdown management
  ✓ System stability

Adjust if:
  - Consistent losses → Reduce position sizes
  - High win rate → Increase position sizes
  - Slippage issues → Adjust spread limits
```

### Phase 3: Production (Day 22+)
```
Go Live with Tuned Parameters:
  Based on paper trading results
  Conservative increase from paper
  Close monitoring first week

Success Metrics:
  ✓ 50%+ win rate
  ✓ Positive daily P&L
  ✓ Drawdown < 10%
  ✓ Stable performance
```

---

## 📊 Monitoring Key Metrics

Track these during testing:

1. **Win Rate**: % of profitable trades
   - Target: 50%+ (arbitrage should be >60%)
   - Red flag: <40%

2. **Profit Factor**: Gross profit / Gross loss
   - Target: >1.5 (1.5:1 profit to loss ratio)
   - Red flag: <1.0 (losses exceed profits)

3. **Average Trade**: Average profit per trade
   - Target: >$1.00
   - Red flag: <$0.10

4. **Drawdown**: Largest loss from peak
   - Target: <10% of capital
   - Red flag: >20% of capital

5. **Daily P&L**: Daily profit/loss
   - Target: Positive most days
   - Red flag: Negative streaks >2-3 days

---

## 🚨 Risk Adjustment Rules

Automatically adjust based on performance:

```
If Win Rate < 40%:
  → Reduce min_profit_cents by 50%
  → Reduce max_order_value_cents by 25%
  → Reduce max_open_positions by 25%

If Win Rate > 70%:
  → Increase max_order_value_cents by 10%
  → Increase max_open_positions by 10%
  → Can add new strategies

If Drawdown > 20%:
  → Reduce all position sizes by 50%
  → Increase min_profit_cents by 100%
  → Close less profitable opportunities
```

---

## ✅ Tuning Checklist

Before Sandbox Deployment:
- [ ] Determine your trading capital
- [ ] Calculate appropriate position sizes (1-3% per trade)
- [ ] Set daily loss limit (1-5% of capital)
- [ ] Choose profit threshold (start conservative)
- [ ] Configure max open positions (10-50)
- [ ] Set order timeout based on market conditions
- [ ] Adjust liquidity requirements for your markets

Before Paper Trading:
- [ ] Review sandbox results
- [ ] Increase position sizes by 2-3x
- [ ] Increase daily loss limit to match capital
- [ ] Fine-tune profit thresholds

Before Production:
- [ ] Validate paper trading results
- [ ] Ensure consistent profitability
- [ ] Start small (half of paper limits)
- [ ] Monitor closely first week

---

## 💡 Key Principles

1. **Start Conservative** - Too strict is better than too loose
2. **Test Gradually** - Increase limits after validation
3. **Monitor Metrics** - Track all important stats
4. **Adjust Data** - Use results to refine parameters
5. **Risk First** - Protect capital above all else

---

## 📞 Next: Sandbox Deployment

Once parameters are set, we'll:
1. Update config.yaml with your chosen parameters
2. Switch to Kalshi sandbox environment
3. Deploy main_production.py
4. Monitor for 7 days
5. Collect performance data
6. Adjust for paper trading

Ready to deploy? Let's go! 🚀
