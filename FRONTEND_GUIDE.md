# Arbitrage Bot Front-End User Guide

## 🎯 **HOW TO ACCESS THE FRONT-END**

### **Option 1: Web Dashboard (Recommended for Most Users)**

#### **Quick Start:**
```bash
# Navigate to project directory
cd /Users/christophermayfield/Documents/Projects/arbitrage_bot

# Start the web dashboard
./start_dashboard.sh
```

#### **Manual Start:**
```bash
python3 -m src.api.main_api
```

#### **Access:**
- **Main Dashboard**: http://localhost:8002/dashboard
- **API Documentation**: http://localhost:8002/docs
- **Direct Access**: http://localhost:8002

#### **What You'll See:**
📊 **Real-time Trading Dashboard** with:
- Live P&L charts showing performance over time
- Current arbitrage opportunities table
- Strategy toggle controls (Statistical Arbitrage, Mean Reversion, etc.)
- Portfolio metrics and position information
- Risk monitoring and alerts
- Market data streaming with sub-second updates

### **Option 2: Enterprise REST API (For Developers/Institutions)**

#### **Start Professional API:**
```bash
python3 -m src.api.rest_api
```

#### **Access:**
- **API Documentation**: http://localhost:8000/api/docs
- **Interactive Swagger UI**: http://localhost:8000/redoc

#### **Key Features:**
- 50+ professional API endpoints
- Authentication with Bearer tokens
- Background task processing
- Advanced backtesting, ML model management
- Multi-exchange support

### **Option 3: Command Line Interface (For Power Users)**

#### **Basic Bot:**
```bash
python3 -m src.main
```

#### **Enhanced Bot with Advanced Features:**
```bash
python3 -m src.main_enhanced
```

---

## 🚀 **WEB DASHBOARD FEATURES**

### **📈 Real-Time Charts**
- **P&L Performance**: Live profit/loss tracking
- **Portfolio Value**: Asset allocation over time
- **Strategy Performance**: Individual strategy returns
- **Risk Metrics**: Drawdown, VaR, exposure charts

### **⚡ Trading Controls**
- **Start/Stop Bot**: Instant control over trading
- **Strategy Toggles**: Enable/disable specific strategies
- **Parameter Adjustment**: Real-time config changes
- **Manual Trade Execution**: Override automatic trading

### **🎯 Opportunity Monitoring**
- **Live Opportunities Table**: Current arbitrage opportunities
- **Filter Options**: Sort by profit, probability, risk
- **Execution Controls**: Manual trade approval
- **Historical Tracking**: Past opportunities and outcomes

### **🛡️ Risk Management**
- **Position Monitoring**: Real-time position sizes
- **Risk Metrics**: Current portfolio risk score
- **Alert System**: Instant notifications for breaches
- **Circuit Breakers**: Automatic trading suspension

### **📊 Portfolio Analytics**
- **Performance Attribution**: Strategy-by-strategy analysis
- **Win Rate Statistics**: Success rate tracking
- **Profit Distribution**: Trade outcome visualization
- **Risk-Adjusted Returns**: Sharpe, Sortino ratios

---

## 🎮 **DASHBOARD NAVIGATION**

### **Main Sections:**

#### **📊 Overview Panel (Left Side)**
- Total Portfolio Value
- Daily/Weekly/Monthly P&L
- Active Positions Count
- Current Win Rate
- Risk Score Indicator

#### **📈 Charts Area (Center)**
- Real-time P&L Chart
- Portfolio Composition
- Strategy Performance Comparison
- Risk Metrics Over Time

#### **🎛️ Control Panel (Right Side)**
- Strategy Toggles
- Parameter Sliders
- Start/Stop Controls
- Manual Trade Buttons

#### **📋 Opportunities Table (Bottom)**
- Current Arbitrage Opportunities
- Profit Potential
- Execution Probability
- Risk Assessment

---

## 🔧 **CUSTOMIZATION OPTIONS**

### **Strategy Configuration:**
```yaml
# In config.yaml
statistical:
  enabled: true
  lookback_period: 30
  z_score_threshold: 2.0

mean_reversion:
  enabled: true
  rsi_period: 14
  oversold_threshold: 30
  overbought_threshold: 70
```

### **Risk Management:**
```yaml
risk:
  max_daily_loss_cents: 10000
  max_position_contracts: 1000
  circuit_breaker_threshold: 5
```

### **Trading Parameters:**
```yaml
trading:
  paper_mode: true  # Set to false for live trading
  min_profit_cents: 10
  max_order_value_cents: 10000
```

---

## 📱 **MOBILE ACCESS**

The web dashboard is **fully responsive** and works on:
- **Desktop browsers** (Chrome, Firefox, Safari, Edge)
- **Tablet devices** (iPad, Android tablets)
- **Mobile phones** (iOS Safari, Android Chrome)

### **Mobile Features:**
- Touch-friendly controls
- Swipe gestures for chart navigation
- Compact layout optimization
- Push notifications support

---

## 🔄 **REAL-TIME FEATURES**

### **WebSocket Updates:**
- Sub-second market data updates
- Live trade executions
- Instant opportunity alerts
- Real-time portfolio changes

### **Auto-Refresh:**
- Chart updates every second
- Opportunities table refresh
- Portfolio metrics update
- Risk monitoring continuous

---

## 🚨 **ALERTS & NOTIFICATIONS**

### **Dashboard Alerts:**
- Visual alert banners
- Sound notifications
- Color-coded risk indicators
- Pop-up confirmations

### **External Notifications:**
- Slack integration
- Discord webhooks
- Telegram bot support
- Email alerts

---

## 🎯 **TROUBLESHOOTING**

### **Common Issues:**

#### **Dashboard Not Loading:**
```bash
# Check if server is running
curl http://localhost:8002/health

# Check logs
tail logs/arbitrage_bot.log
```

#### **No Live Data:**
- Check Kalshi API credentials in `config.yaml`
- Verify internet connection
- Check WebSocket connection status

#### **Strategy Not Working:**
- Verify strategy is enabled in config
- Check minimum profit thresholds
- Review risk management limits

### **Performance Issues:**
- Reduce chart update frequency
- Limit historical data range
- Close other browser tabs
- Check system resources

---

## 📚 **NEXT STEPS**

### **For Beginners:**
1. Start with **paper trading mode**
2. Monitor the dashboard for a day
3. Review strategy performance
4. Gradually enable more strategies

### **For Advanced Users:**
1. Use the **REST API** for custom integrations
2. Deploy with **Kubernetes** for production
3. Set up **monitoring alerts**
4. Configure **compliance settings**

### **For Institutions:**
1. Set up **multi-exchange** trading
2. Implement **risk management** policies
3. Configure **audit logging**
4. Deploy with **auto-scaling**

---

## 🎉 **GETTING STARTED RIGHT NOW**

### **1. Quick Test Run:**
```bash
# Start dashboard
./start_dashboard.sh

# Open browser to: http://localhost:8002/dashboard
```

### **2. Configure Your API Keys:**
```bash
# Edit config.yaml
nano config.yaml
# Add your Kalshi API credentials
```

### **3. Start Paper Trading:**
```bash
# In the dashboard, toggle strategies ON
# Monitor performance for 1-2 hours
# Review opportunity detection
```

**Your enterprise-grade arbitrage platform is ready to use! 🚀**