# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-01-22

### Added
- Initial release of Kalshi Arbitrage Bot
- Core arbitrage detection (cross-market and internal)
- Kalshi API client with RSA authentication
- Order book analysis and liquidity scoring
- Portfolio management with risk limits
- Trading execution with retry logic
- Prometheus metrics integration
- Logging with rotation
- Docker and Docker Compose support
- GitHub Actions CI/CD pipeline
- Unit tests (37 passing)

### Features
- Paper trading mode for safe testing
- Configurable arbitrage thresholds
- Circuit breaker for risk management
- Notifications (Slack, Discord, Telegram)
- Grafana dashboard for monitoring

### Components
- `src/clients/kalshi_client.py` - Kalshi API integration
- `src/core/orderbook.py` - Order book analysis
- `src/core/arbitrage.py` - Arbitrage detection
- `src/core/portfolio.py` - Position management
- `src/execution/trading.py` - Order execution
- `src/monitoring/monitoring.py` - Metrics & alerts
- `src/api/main.py` - REST API (FastAPI)

### Docker Services
- Bot container
- Redis for caching
- Prometheus for metrics
- Grafana for visualization
