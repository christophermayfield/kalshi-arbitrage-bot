# Arbitrage Bot API Documentation

## Base URL
```
http://localhost:8000
```

## Swagger UI
Access interactive documentation at: `/api/docs`

## ReDoc
Alternative documentation at: `/api/redoc`

---

## Authentication

Most endpoints require a Bearer token. Include in requests:
```
Authorization: Bearer <your-token>
```

---

## Endpoints

### Health Check

#### GET /api/v2/health
Get comprehensive health status of the bot.

**Response:**
```json
{
  "status": "healthy",
  "running": true,
  "components": {
    "exchange": {"healthy": true},
    "websocket": {"healthy": true, "subscribed_markets": 50},
    "paper_trading": {
      "balance": 100000,
      "total_pnl": 500,
      "executed_trades": 10
    }
  },
  "stats": {
    "executed_count": 10,
    "subscribed_markets": 50
  }
}
```

#### GET /api/v2/health/live
Kubernetes-style liveness probe.

#### GET /api/v2/health/ready
Kubernetes-style readiness probe.

---

### Risk Analytics

#### GET /api/v2/risk/metrics
Get current risk metrics including VaR, CVaR, Sharpe ratio.

**Response:**
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "var_95": 2.5,
  "var_99": 4.2,
  "cvar_95": 3.8,
  "cvar_99": 5.5,
  "max_drawdown": 5.2,
  "sharpe_ratio": 1.8,
  "volatility": 12.5
}
```

#### GET /api/v2/risk/stress-tests
Run stress tests against market scenarios.

**Response:**
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "scenarios": {
    "market_crash_20": -20.0,
    "market_crash_30": -30.0,
    "volatility_spike_2x": -15.0,
    "liquidity_crisis": -25.0,
    "black_swan": -50.0
  }
}
```

#### GET /api/v2/risk/rolling
Get rolling window metrics.

**Query Parameters:**
- `window` (int): Rolling window size (5-100, default: 20)

#### POST /api/v2/risk/returns
Add a return observation for risk calculations.

**Body:**
```json
{
  "return_pct": 0.01  // 1% return
}
```

---

### Paper Trading

#### GET /api/v2/paper/balance
Get current paper trading balance.

#### GET /api/v2/paper/stats
Get paper trading statistics.

**Response:**
```json
{
  "balance": 102500,
  "total_pnl": 2500,
  "realized_pnl": 2000,
  "unrealized_pnl": 500,
  "filled_orders": 25,
  "total_volume": 500000,
  "total_commission": 500
}
```

#### POST /api/v2/paper/reset
Reset paper trading balance.

**Body (optional):**
```json
{
  "initial_balance": 100000  // in cents
}
```

---

### Dashboard

#### GET /api/v2/dashboard/orderbooks
Get orderbook data for visualization.

**Query Parameters:**
- `market_id` (str, optional): Filter by market
- `limit` (int): Max markets to return (default: 10)

**Response:**
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "orderbooks": [
    {
      "market_id": "MARKET-1",
      "best_bid": 55,
      "best_ask": 56,
      "mid_price": 55.5,
      "spread": 1,
      "spread_percent": 1.8,
      "liquidity_score": 85.5,
      "bid_depth": [{"price": 55, "size": 100}],
      "ask_depth": [{"price": 56, "size": 100}]
    }
  ]
}
```

#### GET /api/v2/dashboard/summary
Get summary data for dashboard.

---

### Analytics

#### GET /api/v2/analytics/correlations
Get correlation analysis between markets.

**Query Parameters:**
- `market_id` (str, optional): Filter by market
- `limit` (int): Max pairs to return (default: 10)

#### GET /api/v2/analytics/correlation-matrix
Get full correlation matrix.

**Query Parameters:**
- `markets` (str, optional): Comma-separated market IDs

#### POST /api/v2/analytics/prices
Add price observation for correlation analysis.

**Body:**
```json
{
  "market_id": "MARKET-1",
  "price": 55.5
}
```

#### GET /api/v2/analytics/stats
Get analytics engine statistics.

---

## Rate Limits

- **Requests**: 100 per minute per IP
- **Response**: 429 Too Many Requests when exceeded

---

## Error Responses

All errors follow this format:
```json
{
  "detail": "Error message description"
}
```

Common status codes:
- `200` - Success
- `400` - Bad Request
- `401` - Unauthorized
- `429` - Rate Limit Exceeded
- `500` - Internal Server Error

---

## WebSocket

For real-time updates, connect to:
```
ws://localhost:8000/ws
```

See WebSocket documentation for message formats.
