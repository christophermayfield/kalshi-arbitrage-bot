# Kalshi Arbitrage Bot

![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Tests](https://img.shields.io/badge/tests-37%20passed-green.svg)

An automated trading bot designed to find and execute arbitrage opportunities on [Kalshi](https://kalshi.com), the CFTC-regulated prediction market exchange.

## What is Kalshi Arbitrage?

Arbitrage on Kalshi involves exploiting price differences between:
- **Cross-market arbitrage**: Price differences between related prediction markets
- **Internal arbitrage**: Bid-ask spread within a single market

The bot continuously scans order books, identifies profitable opportunities, and executes trades while managing risk.

## Features

- **Real-time Market Scanning**: WebSocket-powered continuous monitoring
- **Cross-Market Arbitrage**: Detects price discrepancies between related markets
- **Internal Arbitrage**: Exploits wide bid-ask spreads within a single market
- **Event-Driven Architecture**: Fast reaction to market changes
- **Risk Management**: 
  - Daily loss limits
  - Maximum position limits
  - Circuit breaker for consecutive losses
  - Position sizing controls
- **Paper Trading Mode**: Test strategies without risking real money
- **Database Persistence**: SQLite storage for trades and positions
- **Position Reconciliation**: Automatic sync with exchange
- **Prometheus Metrics**: Built-in monitoring for performance tracking
- **Notifications**: Alerts for opportunities and executions (Slack, Discord, Telegram)
- **Comprehensive Logging**: Detailed logs for debugging and audit trails

## Quick Start

### Prerequisites

- Python 3.12 or higher
- pip
- A Kalshi API account ([get API keys here](https://docs.kalshi.com/getting_started/api_keys))

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/arbitrage_bot.git
cd arbitrage_bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create configuration:
```bash
cp config.yaml.example config.yaml
```

4. Configure your API credentials in `config.yaml`:
```yaml
kalshi:
  api_key_id: "YOUR_API_KEY_ID"
  private_key_path: "~/.kalshi/private-key.pem"
  base_url: "https://demo-api.kalshi.co"  # Use demo for testing
  demo_mode: true
```

5. Get your API keys:
   - Sign up at [kalshi.com](https://kalshi.com)
   - Go to API Access in your account settings
   - Download your private key to `~/.kalshi/private-key.pem`

### Usage

#### Start in Paper Trading Mode (Recommended for First Run)
```bash
python3 -m src.main
```

#### Start in Live Trading Mode
```yaml
# In config.yaml, set:
trading:
  paper_mode: false
```
```bash
python3 -m src.main
```

#### Run with Docker
```bash
docker build -t kalshi-arbitrage-bot .
docker run -v $(pwd)/config.yaml:/app/config.yaml kalshi-arbitrage-bot
```

### Configuration Options

| Setting | Description | Default |
|---------|-------------|---------|
| `paper_mode` | Run without real money | `true` |
| `min_profit_cents` | Minimum profit per trade | `10` |
| `scan_interval` | Market scan interval (seconds) | `1` |
| `max_daily_loss_cents` | Daily loss limit | `10000` |
| `order_timeout_seconds` | Order timeout | `30` |

## Configuration

All configuration is managed through `config.yaml`:

### Kalshi API Settings
```yaml
kalshi:
  api_key_id: "your-api-key-id"
  private_key_path: "~/.kalshi/private-key.pem"
  base_url: "https://demo-api.kalshi.co"  # Use demo for testing
  demo_mode: true
```

### Trading Parameters
```yaml
trading:
  paper_mode: true              # Set to false for live trading
  arbitrage_threshold: 0.99     # Minimum confidence threshold
  min_profit_cents: 10          # Minimum profit per trade
  max_position_contracts: 1000  # Maximum contracts per position
  max_order_value_cents: 10000  # Maximum order size
  order_timeout_seconds: 30     # Order timeout
  retry_attempts: 3            # Retry failed orders
  scan_interval_seconds: 1     # How often to scan markets
```

### Database Settings
```yaml
database:
  path: "data/arbitrage.db"    # SQLite database path
```

### Risk Management
```yaml
risk:
  max_daily_loss_cents: 10000   # Daily loss limit
  max_open_positions: 50        # Maximum open positions
  circuit_breaker_threshold: 5  # Consecutive losses before stop
  circuit_breaker_window_seconds: 300
```

## Project Structure

```
arbitrage_bot/
├── src/
│   ├── clients/
│   │   └── kalshi_client.py     # Kalshi API client
│   ├── core/
│   │   ├── arbitrage.py         # Arbitrage detection algorithms
│   │   ├── orderbook.py         # Order book analysis
│   │   └── portfolio.py         # Position & risk management
│   ├── execution/
│   │   └── trading.py           # Order execution engine
│   ├── monitoring/
│   │   └── monitoring.py        # Metrics & notifications
│   ├── utils/
│   │   ├── config.py            # Configuration loader
│   │   └── logging_utils.py     # Logging setup
│   └── main.py                  # Bot entry point
├── tests/
│   ├── unit/                    # Unit tests
│   └── integration/             # Integration tests
├── config.yaml                  # Configuration file
├── requirements.txt             # Python dependencies
├── pytest.ini                   # Pytest configuration
└── README.md                    # This file
```

## How It Works

### 1. Market Data Collection
The bot connects to Kalshi via WebSocket for real-time updates:
- Subscribes to orderbook updates for top markets
- Falls back to REST polling if WebSocket fails
- Maintains local orderbook cache

### 2. Arbitrage Detection
For each market pair, the detector identifies:
- **Cross-market opportunities**: Buy on market A, sell on market B
- **Internal opportunities**: Buy at ask, sell at bid in same market

Opportunities are scored by:
- Profit potential (in cents)
- Fill probability
- Market liquidity
- Historical confidence

### 3. Execution
Valid opportunities are executed atomically:
1. Place buy and sell orders simultaneously
2. Wait for both to fill (with timeout)
3. Emergency cancel if timeout occurs
4. Calculate realized profit

### 4. Risk Management
Before each trade, the bot checks:
- Available cash balance
- Open position count
- Daily loss limit
- Circuit breaker status

### 5. Position Reconciliation
Every 60 seconds, the bot syncs:
- Open positions with exchange
- Account balance

## API Reference

### KalshiClient
```python
from src.clients.kalshi_client import KalshiClient

client = KalshiClient(config)
client.get_markets(status="open")
client.get_market_orderbook(market_id)
client.create_order(market_id, side, order_type, price, count)
client.cancel_order(order_id)
```

### ArbitrageDetector
```python
from src.core.arbitrage import ArbitrageDetector

detector = ArbitrageDetector(
    min_profit_cents=10,
    fee_rate=0.01
)
opportunities = detector.scan_for_opportunities(orderbooks)
```

### TradingExecutor
```python
from src.execution.trading import TradingExecutor

executor = TradingExecutor(client, paper_mode=True)
success, profit = await executor.execute_arbitrage(opportunity)
```

## Monitoring

### Prometheus Metrics
The bot exposes metrics on port 8000:
- `arbitrage_opportunities_total` - Total opportunities found
- `arbitrage_executed_total` - Total executed trades
- `arbitrage_profit_cents_total` - Total profit in cents
- `orderbook_scans_total` - Total market scans
- `api_errors_total` - Total API errors

### Health Check
```bash
curl http://localhost:8000
```

## Testing

Run the test suite:
```bash
pytest tests/unit/ -v
```

Run with coverage:
```bash
pytest --cov=src tests/
```

## Troubleshooting

### Common Issues

**"Failed to connect to WebSocket"**
- Check your internet connection
- Verify API credentials are correct
- The bot will fall back to REST polling automatically

**"Failed to sync balance"**
- Verify your API key has correct permissions
- Check private key file exists at configured path

**"No arbitrage opportunities found"**
- This is normal - arbitrage opportunities are fleeting
- Try lowering `min_profit_cents` in config
- Check that markets have sufficient liquidity

**Database errors**
- Ensure `data/` directory exists and is writable
- Or set a different path in `config.yaml`

### Logs

Logs are written to stdout and can be viewed with:
```bash
python3 -m src.main 2>&1 | tee bot.log
```

Set log level in config:
```yaml
monitoring:
  log_level: "DEBUG"  # DEBUG, INFO, WARNING, ERROR
```

## Safety & Risk Disclosure

**IMPORTANT**: Trading prediction markets involves significant risk.

- Start with paper trading mode to understand the bot's behavior
- Never trade with money you cannot afford to lose
- Set appropriate loss limits and monitor the bot regularly
- The bot may experience losses due to:
  - Market volatility
  - Slippage
  - API errors or delays
  - Liquidity issues

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This software is provided for educational purposes only. The authors are not responsible for any financial losses incurred while using this bot. Always do your own research and understand the risks before trading.

## Resources

- [Kalshi API Documentation](https://docs.kalshi.com/)
- [Kalshi API Reference](https://docs.kalshi.com/api-reference)
- [Kalshi Discord](https://discord.gg/kalshi)
- [Prediction Market Basics](https://help.kalshi.com/)

---

**Happy Trading!**
