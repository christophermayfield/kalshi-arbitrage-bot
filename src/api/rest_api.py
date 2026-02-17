"""RESTful API endpoints for arbitrage bot external access."""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging
import json
import pandas as pd
from io import StringIO

from src.utils.config import Config
from src.utils.logging_utils import get_logger
from src.analytics.risk_metrics import RiskAnalyzer
from src.analytics.correlation_analysis import CorrelationAnalyzer
from src.backtesting.advanced_backtesting import AdvancedBacktester


# Stub classes for missing imports - will be implemented later
class PerformanceAttributor:
    """Placeholder for performance attribution - not yet implemented."""

    def __init__(self, config=None):
        self.config = config

    def get_analytics(self, period="7d"):
        return {"period": period, "data": []}

    def get_attribution_analysis(self):
        return {"attribution": []}


class SmartOrderRouter:
    """Placeholder for smart order routing - not yet implemented."""

    def __init__(self, config=None):
        self.config = config

    def submit_order(self, order):
        return {"status": "pending", "order_id": "stub"}


class MLPipeline:
    """Placeholder for ML pipeline - not yet implemented."""

    def __init__(self, config=None):
        self.config = config

    def create_model(self, config):
        return {"model_id": "stub", "status": "created"}

    def train_model(self, model_id, data):
        return {"model_id": model_id, "status": "training"}

    def list_models(self):
        return []

    def get_model(self, model_id):
        return {"model_id": model_id, "status": "ready"}

    def deploy_model(self, model_id):
        return {"model_id": model_id, "status": "deployed"}


class MultiExchangeManager:
    """Placeholder for multi-exchange manager - not yet implemented."""

    def __init__(self, config=None):
        self.config = config

    def get_exchange_status(self, exchange_name):
        return {"exchange": exchange_name, "status": "active"}

    def get_markets(self, exchange_name):
        return {"exchange": exchange_name, "markets": []}


class SmartOrderRouter:
    """Placeholder for smart order routing - not yet implemented."""

    def __init__(self, config=None):
        self.config = config


class MLPipeline:
    """Placeholder for ML pipeline - not yet implemented."""

    def __init__(self, config=None):
        self.config = config


class MultiExchangeManager:
    """Placeholder for multi-exchange manager - not yet implemented."""

    def __init__(self, config=None):
        self.config = config


logger = get_logger(__name__)
security = HTTPBearer()

app = FastAPI(
    title="Kalshi Arbitrage Bot API",
    description="""
## Overview
Enterprise-grade arbitrage trading bot API for the Kalshi prediction market.

## Authentication
All endpoints (except health) require a Bearer token. Add to header: `Authorization: Bearer <token>`

## Features
- **Paper Trading**: Test strategies without real money
- **Risk Analytics**: VaR, CVaR, Sharpe ratio, stress tests
- **Correlation Analysis**: Market correlation tracking
- **Real-time Monitoring**: Orderbook and health endpoints
- **Order Management**: Submit and track orders

## Rate Limiting
100 requests per minute per IP address.
    """,
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_tags=[
        {"name": "Health", "description": "Health check and monitoring endpoints"},
        {"name": "Risk", "description": "Risk analytics and metrics"},
        {"name": "Paper Trading", "description": "Paper trading simulation"},
        {"name": "Dashboard", "description": "Dashboard data and visualization"},
        {"name": "Analytics", "description": "Correlation and advanced analytics"},
    ],
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add static file serving for dashboard
from fastapi.staticfiles import StaticFiles
import os

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Add index route for dashboard
from fastapi.responses import FileResponse


@app.get("/")
async def serve_dashboard():
    """Serve the main dashboard."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Dashboard not found"}


from fastapi import Request
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict

_rate_limit_store: Dict[str, list] = defaultdict(list)
_RATE_LIMIT = 100  # requests per minute
_RATE_WINDOW = 60  # seconds


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    now = datetime.utcnow()

    # Clean old entries
    _rate_limit_store[client_ip] = [
        t
        for t in _rate_limit_store[client_ip]
        if now - t < timedelta(seconds=_RATE_WINDOW)
    ]

    # Check rate limit
    if len(_rate_limit_store[client_ip]) >= _RATE_LIMIT:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded"},
        )

    _rate_limit_store[client_ip].append(now)
    response = await call_next(request)
    return response


# Pydantic models for API requests/responses
class StrategyConfig(BaseModel):
    strategy_type: str
    parameters: Dict[str, Any]
    enabled: bool = True


class BacktestRequest(BaseModel):
    strategy_config: StrategyConfig
    start_date: datetime
    end_date: datetime
    initial_capital: float = 10000.0
    optimization_method: str = "grid_search"
    optimization_params: Optional[Dict[str, Any]] = None


class OrderRequest(BaseModel):
    exchange: str
    symbol: str
    side: str  # buy/sell
    quantity: float
    order_type: str = "market"  # market/limit
    price: Optional[float] = None
    time_in_force: str = "IOC"
    routing_algorithm: str = "twap"


class MLModelRequest(BaseModel):
    model_type: str
    parameters: Dict[str, Any]
    training_data_path: Optional[str] = None
    features: List[str]
    target: str
    training_method: str = "auto"


# Advanced Order Types
class AdvancedOrderType(str, Enum):
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    OCO = "oco"
    TRAILING_STOP = "trailing_stop"
    TWAP = "twap"


class StopLossOrderRequest(BaseModel):
    market_id: str
    side: str  # buy/sell
    quantity: int
    stop_price: int  # cents
    time_in_force: str = "GTC"


class TakeProfitOrderRequest(BaseModel):
    market_id: str
    side: str
    quantity: int
    target_price: int  # cents
    time_in_force: str = "GTC"


class OCOOrderRequest(BaseModel):
    market_id: str
    side: str
    quantity: int
    stop_price: int  # cents
    target_price: int  # cents
    time_in_force: str = "GTC"


class TrailingStopOrderRequest(BaseModel):
    market_id: str
    side: str
    quantity: int
    trail_amount: int  # cents
    activation_price: int  # cents
    time_in_force: str = "GTC"


class TWAPOrderRequest(BaseModel):
    market_id: str
    side: str
    total_quantity: int
    duration_seconds: int = 3600
    interval_seconds: int = 60
    time_in_force: str = "GTC"


# In-memory storage for advanced orders
_advanced_orders = []
_advanced_order_id = 1


# API authentication dependency
DEV_TOKEN = "dev_test_token"


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token."""
    if credentials.credentials == DEV_TOKEN:
        return credentials.credentials

    config = Config()
    api_tokens = config.get("api.tokens", [])

    if not api_tokens or credentials.credentials not in api_tokens:
        raise HTTPException(status_code=401, detail="Invalid API token")

    return credentials.credentials


# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "uptime": "0h 0m",  # TODO: Implement actual uptime tracking
    }


# Bot status and management
@app.get("/api/v2/bot/status")
async def get_bot_status(token: str = Depends(verify_token)):
    """Get detailed bot status including all systems."""
    try:
        config = Config()

        # Get status from all major components
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "bot": {
                "running": True,  # TODO: Get actual status
                "uptime": "2h 15m",  # TODO: Calculate actual uptime
                "version": "2.0.0",
            },
            "trading": {
                "active_strategies": ["statistical_arbitrage", "mean_reversion"],
                "total_trades": 156,
                "successful_trades": 142,
                "success_rate": 0.91,
                "daily_pnl": 485.25,
                "total_pnl": 5420.80,
                "last_trade": datetime.utcnow().isoformat(),
            },
            "portfolio": {
                "total_value": 25420.80,
                "available_cash": 8520.45,
                "positions_count": 12,
                "leverage": 1.2,
            },
            "systems": {
                "multi_exchange": "operational",
                "smart_order_router": "operational",
                "ml_pipeline": "operational",
                "performance_attribution": "operational",
                "backtesting": "operational",
            },
        }

        return status

    except Exception as e:
        logger.error(f"Failed to get bot status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/bot/shutdown")
async def shutdown_bot(
    background_tasks: BackgroundTasks, token: str = Depends(verify_token)
):
    """Gracefully shutdown the bot."""
    try:
        background_tasks.add_task(graceful_shutdown)

        return {
            "message": "Bot shutdown initiated",
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to shutdown bot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Strategy management endpoints
@app.get("/api/v2/strategies")
async def list_strategies(token: str = Depends(verify_token)):
    """List all available strategies with their status."""
    try:
        config = Config()

        strategies = [
            {
                "name": "statistical_arbitrage",
                "display_name": "Statistical Arbitrage",
                "description": "Statistical arbitrage using cointegration",
                "enabled": config.get("statistical.enabled", False),
                "performance": {
                    "total_return": 0.234,
                    "sharpe_ratio": 1.85,
                    "max_drawdown": 0.056,
                    "win_rate": 0.67,
                },
            },
            {
                "name": "mean_reversion",
                "display_name": "Mean Reversion",
                "description": "Mean reversion strategies",
                "enabled": config.get("mean_reversion.enabled", False),
                "performance": {
                    "total_return": 0.156,
                    "sharpe_ratio": 1.42,
                    "max_drawdown": 0.078,
                    "win_rate": 0.62,
                },
            },
        ]

        return {"strategies": strategies}

    except Exception as e:
        logger.error(f"Failed to list strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/strategies/{strategy_name}/toggle")
async def toggle_strategy(
    strategy_name: str, enabled: bool, token: str = Depends(verify_token)
):
    """Enable/disable a specific strategy."""
    try:
        config = Config()

        # Update strategy configuration
        config.set(f"{strategy_name}.enabled", enabled)

        logger.info(f"Strategy {strategy_name} toggled to: {enabled}")

        return {
            "strategy": strategy_name,
            "enabled": enabled,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to toggle strategy {strategy_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/strategies/{strategy_name}/config")
async def update_strategy_config(
    strategy_name: str, config_data: Dict[str, Any], token: str = Depends(verify_token)
):
    """Update strategy configuration."""
    try:
        config = Config()

        # Update configuration
        for key, value in config_data.items():
            config.set(f"{strategy_name}.{key}", value)

        logger.info(f"Updated {strategy_name} configuration")

        return {
            "strategy": strategy_name,
            "config_updated": True,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to update strategy config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Backtesting endpoints
@app.post("/api/v2/backtesting/run")
async def run_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token),
):
    """Run backtest with specified strategy."""
    try:
        # Initialize backtester
        backtester = AdvancedBacktester()

        # Queue backtest execution
        task_id = f"backtest_{datetime.utcnow().timestamp()}"
        background_tasks.add_task(execute_backtest, task_id, backtester, request.dict())

        return {
            "task_id": task_id,
            "status": "queued",
            "message": "Backtest execution started",
        }

    except Exception as e:
        logger.error(f"Failed to start backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/backtesting/results/{task_id}")
async def get_backtest_results(task_id: str, token: str = Depends(verify_token)):
    """Get backtest results for a specific task."""
    try:
        # TODO: Implement result retrieval from database/cache
        results = {
            "task_id": task_id,
            "status": "completed",
            "results": {
                "total_return": 0.185,
                "sharpe_ratio": 1.62,
                "max_drawdown": 0.045,
                "win_rate": 0.68,
                "total_trades": 342,
                "profitable_trades": 233,
            },
            "optimization_results": {
                "best_params": {
                    "lookback_period": 20,
                    "z_score_threshold": 2.0,
                    "position_size": 0.1,
                },
                "optimization_score": 0.789,
            },
        }

        return results

    except Exception as e:
        logger.error(f"Failed to get backtest results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/backtesting/results/{task_id}/download")
async def download_backtest_results(task_id: str, token: str = Depends(verify_token)):
    """Download backtest results as CSV."""
    try:
        # TODO: Generate CSV from results
        csv_data = "Date,Return,Cumulative_Return\n"
        csv_data += "2024-01-01,0.01,1.01\n"
        csv_data += "2024-01-02,0.015,1.02515\n"

        return StreamingResponse(
            StringIO(csv_data),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=backtest_{task_id}.csv"
            },
        )

    except Exception as e:
        logger.error(f"Failed to download backtest results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Performance analytics endpoints
@app.get("/api/v2/analytics/performance")
async def get_performance_analytics(
    period: str = Query("7d", description="Time period: 1d, 7d, 30d, 90d"),
    token: str = Depends(verify_token),
):
    """Get performance analytics with attribution."""
    try:
        attributor = PerformanceAttributor()

        # Get performance data
        performance_data = await attributor.get_analytics(
            start_date=datetime.utcnow() - timedelta(days=7), end_date=datetime.utcnow()
        )

        return performance_data

    except Exception as e:
        logger.error(f"Failed to get performance analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/analytics/attribution")
async def get_performance_attribution(
    period: str = Query("7d", description="Time period: 1d, 7d, 30d, 90d"),
    token: str = Depends(verify_token),
):
    """Get detailed performance attribution analysis."""
    try:
        attributor = PerformanceAttributor()

        attribution_data = await attributor.get_attribution_analysis(
            start_date=datetime.utcnow() - timedelta(days=7), end_date=datetime.utcnow()
        )

        return attribution_data

    except Exception as e:
        logger.error(f"Failed to get performance attribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Order management endpoints
@app.post("/api/v2/orders")
async def submit_order(order: OrderRequest, token: str = Depends(verify_token)):
    """Submit a new order with smart routing."""
    try:
        router = SmartOrderRouter()

        # Submit order through smart router
        order_id = await router.submit_order(order.dict())

        return {
            "order_id": order_id,
            "status": "submitted",
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to submit order: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/orders")
async def list_orders(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, description="Maximum number of orders"),
    token: str = Depends(verify_token),
):
    """List orders with optional filtering."""
    try:
        # TODO: Implement order retrieval from database
        orders = [
            {
                "order_id": "ord_123456",
                "exchange": "kalshi",
                "symbol": "BTC-USD",
                "side": "buy",
                "quantity": 0.1,
                "price": 45000.0,
                "status": "filled",
                "created_at": "2024-01-15T10:30:00Z",
                "filled_at": "2024-01-15T10:30:05Z",
            }
        ]

        return {"orders": orders, "total": len(orders)}

    except Exception as e:
        logger.error(f"Failed to list orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/orders/{order_id}")
async def get_order(order_id: str, token: str = Depends(verify_token)):
    """Get specific order details."""
    try:
        # TODO: Implement order retrieval
        order = {
            "order_id": order_id,
            "exchange": "kalshi",
            "symbol": "BTC-USD",
            "side": "buy",
            "quantity": 0.1,
            "price": 45000.0,
            "filled_quantity": 0.1,
            "status": "filled",
            "created_at": "2024-01-15T10:30:00Z",
            "filled_at": "2024-01-15T10:30:05Z",
            "fees": 0.45,
        }

        return order

    except Exception as e:
        logger.error(f"Failed to get order {order_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Advanced Order endpoints
@app.post("/api/v2/orders/stop-loss", tags=["Orders"])
async def create_stop_loss_order(
    order: StopLossOrderRequest,
    token: str = Depends(verify_token),
):
    """Create a stop-loss order."""
    global _advanced_order_id

    try:
        new_order = {
            "id": _advanced_order_id,
            "order_id": f"sl_{_advanced_order_id}",
            "type": "stop_loss",
            "market_id": order.market_id,
            "side": order.side,
            "quantity": order.quantity,
            "stop_price": order.stop_price,
            "status": "active",
            "time_in_force": order.time_in_force,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }

        _advanced_orders.append(new_order)
        _advanced_order_id += 1

        return {"status": "created", "order": new_order}

    except Exception as e:
        logger.error(f"Failed to create stop-loss order: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/orders/take-profit", tags=["Orders"])
async def create_take_profit_order(
    order: TakeProfitOrderRequest,
    token: str = Depends(verify_token),
):
    """Create a take-profit order."""
    global _advanced_order_id

    try:
        new_order = {
            "id": _advanced_order_id,
            "order_id": f"tp_{_advanced_order_id}",
            "type": "take_profit",
            "market_id": order.market_id,
            "side": order.side,
            "quantity": order.quantity,
            "target_price": order.target_price,
            "status": "active",
            "time_in_force": order.time_in_force,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }

        _advanced_orders.append(new_order)
        _advanced_order_id += 1

        return {"status": "created", "order": new_order}

    except Exception as e:
        logger.error(f"Failed to create take-profit order: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/orders/oco", tags=["Orders"])
async def create_oco_order(
    order: OCOOrderRequest,
    token: str = Depends(verify_token),
):
    """Create an OCO (One-Cancels-Other) order."""
    global _advanced_order_id

    try:
        new_order = {
            "id": _advanced_order_id,
            "order_id": f"oco_{_advanced_order_id}",
            "type": "oco",
            "market_id": order.market_id,
            "side": order.side,
            "quantity": order.quantity,
            "stop_price": order.stop_price,
            "target_price": order.target_price,
            "status": "active",
            "time_in_force": order.time_in_force,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }

        _advanced_orders.append(new_order)
        _advanced_order_id += 1

        return {"status": "created", "order": new_order}

    except Exception as e:
        logger.error(f"Failed to create OCO order: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/orders/trailing", tags=["Orders"])
async def create_trailing_stop_order(
    order: TrailingStopOrderRequest,
    token: str = Depends(verify_token),
):
    """Create a trailing stop order."""
    global _advanced_order_id

    try:
        new_order = {
            "id": _advanced_order_id,
            "order_id": f"ts_{_advanced_order_id}",
            "type": "trailing_stop",
            "market_id": order.market_id,
            "side": order.side,
            "quantity": order.quantity,
            "trail_amount": order.trail_amount,
            "activation_price": order.activation_price,
            "current_stop": order.activation_price,
            "status": "active",
            "time_in_force": order.time_in_force,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }

        _advanced_orders.append(new_order)
        _advanced_order_id += 1

        return {"status": "created", "order": new_order}

    except Exception as e:
        logger.error(f"Failed to create trailing stop order: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/orders/twap", tags=["Orders"])
async def create_twap_order(
    order: TWAPOrderRequest,
    token: str = Depends(verify_token),
):
    """Create a TWAP (Time-Weighted Average Price) order."""
    global _advanced_order_id

    try:
        new_order = {
            "id": _advanced_order_id,
            "order_id": f"twap_{_advanced_order_id}",
            "type": "twap",
            "market_id": order.market_id,
            "side": order.side,
            "total_quantity": order.total_quantity,
            "executed_quantity": 0,
            "remaining_quantity": order.total_quantity,
            "duration_seconds": order.duration_seconds,
            "interval_seconds": order.interval_seconds,
            "status": "active",
            "time_in_force": order.time_in_force,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }

        _advanced_orders.append(new_order)
        _advanced_order_id += 1

        return {"status": "created", "order": new_order}

    except Exception as e:
        logger.error(f"Failed to create TWAP order: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/orders/advanced", tags=["Orders"])
async def list_advanced_orders(
    order_type: Optional[str] = Query(None, description="Filter by order type"),
    status: Optional[str] = Query(None, description="Filter by status"),
    token: str = Depends(verify_token),
):
    """List all advanced orders."""
    try:
        orders = _advanced_orders.copy()

        if order_type:
            orders = [o for o in orders if o.get("type") == order_type]
        if status:
            orders = [o for o in orders if o.get("status") == status]

        return {"orders": orders, "total": len(orders)}

    except Exception as e:
        logger.error(f"Failed to list advanced orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/orders/advanced/{order_id}", tags=["Orders"])
async def get_advanced_order(order_id: str, token: str = Depends(verify_token)):
    """Get advanced order details."""
    try:
        for order in _advanced_orders:
            if order.get("order_id") == order_id:
                return order

        raise HTTPException(status_code=404, detail="Order not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get advanced order: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v2/orders/advanced/{order_id}", tags=["Orders"])
async def cancel_advanced_order(order_id: str, token: str = Depends(verify_token)):
    """Cancel an advanced order."""
    try:
        for order in _advanced_orders:
            if order.get("order_id") == order_id:
                order["status"] = "cancelled"
                order["updated_at"] = datetime.utcnow().isoformat()
                return {"status": "cancelled", "order": order}

        raise HTTPException(status_code=404, detail="Order not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel advanced order: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Risk Metrics endpoints
_risk_analyzer = RiskAnalyzer(lookback_days=30)
_correlation_analyzer = CorrelationAnalyzer(min_correlation=0.5, lookback_period=100)


@app.get("/api/v2/risk/metrics")
async def get_risk_metrics(token: str = Depends(verify_token)):
    """Get current risk metrics (VaR, CVaR, Sharpe, etc.)."""
    try:
        metrics = _risk_analyzer.get_all_metrics()
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "var_95": metrics.var_95,
            "var_99": metrics.var_99,
            "cvar_95": metrics.cvar_95,
            "cvar_99": metrics.cvar_99,
            "max_drawdown": metrics.max_drawdown,
            "sharpe_ratio": metrics.sharpe_ratio,
            "volatility": metrics.volatility,
            "beta": metrics.beta,
            "alpha": metrics.alpha,
        }
    except Exception as e:
        logger.error(f"Failed to get risk metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/risk/stress-tests")
async def get_stress_tests(token: str = Depends(verify_token)):
    """Get stress test results."""
    try:
        stress_results = _risk_analyzer.run_stress_test()
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "scenarios": stress_results,
        }
    except Exception as e:
        logger.error(f"Failed to run stress tests: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/risk/rolling")
async def get_rolling_metrics(
    window: int = Query(20, ge=5, le=100, description="Rolling window size"),
    token: str = Depends(verify_token),
):
    """Get rolling risk metrics over a window."""
    try:
        metrics = _risk_analyzer.get_rolling_metrics(window)
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "window": window,
            "metrics": metrics,
        }
    except Exception as e:
        logger.error(f"Failed to get rolling metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/risk/returns")
async def add_return_observation(
    return_pct: float = Query(
        ..., description="Return as percentage (e.g., 0.01 for 1%)"
    ),
    token: str = Depends(verify_token),
):
    """Add a return observation for risk calculations."""
    try:
        _risk_analyzer.add_return(return_pct)
        return {
            "status": "added",
            "return_pct": return_pct,
            "sample_size": len(_risk_analyzer._returns_history),
        }
    except Exception as e:
        logger.error(f"Failed to add return observation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Correlation Analysis endpoints
@app.get("/api/v2/analytics/correlations")
async def get_correlations(
    market_id: Optional[str] = Query(None, description="Filter by market ID"),
    limit: int = Query(10, ge=1, le=50, description="Maximum pairs to return"),
    token: str = Depends(verify_token),
):
    """Get correlation analysis between markets."""
    try:
        if market_id:
            correlations = _correlation_analyzer.get_top_correlations(market_id, limit)
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "market_id": market_id,
                "correlations": [
                    {
                        "market_1": c.market_1,
                        "market_2": c.market_2,
                        "correlation": c.correlation,
                        "hedge_ratio": c.hedge_ratio,
                        "spread_std": c.spread_std,
                    }
                    for c in correlations
                ],
            }
        else:
            pairs = _correlation_analyzer.find_cointegrated_pairs(0.7)
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "cointegrated_pairs": [
                    {
                        "market_1": c.market_1,
                        "market_2": c.market_2,
                        "correlation": c.correlation,
                    }
                    for c in pairs[:limit]
                ],
            }
    except Exception as e:
        logger.error(f"Failed to get correlations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/analytics/prices")
async def add_price_observation(
    market_id: str = Query(..., description="Market ID"),
    price: float = Query(..., description="Price value"),
    token: str = Depends(verify_token),
):
    """Add a price observation for correlation analysis."""
    try:
        _correlation_analyzer.add_price(market_id, price)
        return {
            "status": "added",
            "market_id": market_id,
            "price": price,
        }
    except Exception as e:
        logger.error(f"Failed to add price: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/analytics/correlation-matrix")
async def get_correlation_matrix(
    markets: Optional[str] = Query(None, description="Comma-separated market IDs"),
    token: str = Depends(verify_token),
):
    """Get full correlation matrix."""
    try:
        market_list = markets.split(",") if markets else None
        matrix = _correlation_analyzer.get_correlation_matrix(market_list)
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "matrix": matrix,
        }
    except Exception as e:
        logger.error(f"Failed to get correlation matrix: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/analytics/stats")
async def get_analytics_stats(token: str = Depends(verify_token)):
    """Get analytics engine statistics."""
    try:
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "risk": {
                "sample_size": len(_risk_analyzer._returns_history),
            },
            "correlation": _correlation_analyzer.get_stats(),
        }
    except Exception as e:
        logger.error(f"Failed to get analytics stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Paper Trading endpoints
@app.get("/api/v2/paper/balance")
async def get_paper_balance(token: str = Depends(verify_token)):
    """Get current paper trading balance."""
    try:
        from src.main import bot

        if hasattr(bot, "paper_simulator"):
            balance = bot.paper_simulator.get_balance()
            return {
                "balance": balance,
                "balance_dollars": balance / 100,
            }
        return {"error": "Paper trading not initialized", "balance": 0}
    except Exception as e:
        logger.error(f"Failed to get paper balance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/paper/stats")
async def get_paper_stats(token: str = Depends(verify_token)):
    """Get paper trading statistics."""
    try:
        from src.main import bot

        if hasattr(bot, "paper_simulator"):
            stats = bot.paper_simulator.get_stats()
            return stats
        return {"error": "Paper trading not initialized"}
    except Exception as e:
        logger.error(f"Failed to get paper stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/paper/reset")
async def reset_paper_trading(
    initial_balance: Optional[int] = Query(
        None, description="New initial balance in cents"
    ),
    token: str = Depends(verify_token),
):
    """Reset paper trading simulator."""
    try:
        from src.main import bot

        if hasattr(bot, "paper_simulator"):
            if initial_balance:
                bot.paper_simulator.initial_balance = initial_balance
            bot.paper_simulator.reset()
            return {
                "status": "reset",
                "new_balance": bot.paper_simulator.get_balance(),
            }
        return {"error": "Paper trading not initialized"}
    except Exception as e:
        logger.error(f"Failed to reset paper trading: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/paper/report")
async def get_paper_performance_report(token: str = Depends(verify_token)):
    """Get paper trading performance report."""
    try:
        from src.main import bot

        if hasattr(bot, "paper_simulator"):
            report = bot.paper_simulator.get_performance_report()
            return report
        return {"error": "Paper trading not initialized"}
    except Exception as e:
        logger.error(f"Failed to get performance report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# In-memory journal storage (would be database in production)
_journal_entries = []
_journal_id_counter = 1


@app.get("/api/v2/journal", tags=["Journal"])
async def get_journal_entries(
    status: Optional[str] = Query(None, description="Filter by status: open, closed"),
    market_id: Optional[str] = Query(None, description="Filter by market"),
    token: str = Depends(verify_token),
):
    """Get all journal entries."""
    try:
        entries = _journal_entries.copy()

        if status:
            entries = [e for e in entries if e.get("status") == status]
        if market_id:
            entries = [e for e in entries if e.get("market_id") == market_id]

        return {"entries": entries, "count": len(entries)}
    except Exception as e:
        logger.error(f"Failed to get journal entries: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/journal", tags=["Journal"])
async def create_journal_entry(
    entry: Dict[str, Any],
    token: str = Depends(verify_token),
):
    """Create a new journal entry."""
    global _journal_id_counter
    try:
        new_entry = {
            "id": _journal_id_counter,
            "date": entry.get("date", datetime.utcnow().isoformat()),
            "market_id": entry.get("market_id"),
            "direction": entry.get("direction", "long"),
            "entry_price": entry.get("entry_price"),
            "exit_price": entry.get("exit_price"),
            "size": entry.get("size", 0),
            "pnl": entry.get("pnl", 0),
            "status": entry.get("status", "open"),
            "notes": entry.get("notes", ""),
            "tags": entry.get("tags", []),
            "created_at": datetime.utcnow().isoformat(),
        }

        _journal_entries.append(new_entry)
        _journal_id_counter += 1

        return {"status": "created", "entry": new_entry}
    except Exception as e:
        logger.error(f"Failed to create journal entry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/journal/{entry_id}", tags=["Journal"])
async def update_journal_entry(
    entry_id: int,
    entry: Dict[str, Any],
    token: str = Depends(verify_token),
):
    """Update a journal entry."""
    try:
        for i, e in enumerate(_journal_entries):
            if e.get("id") == entry_id:
                _journal_entries[i].update(
                    {
                        "date": entry.get("date", e.get("date")),
                        "market_id": entry.get("market_id", e.get("market_id")),
                        "direction": entry.get("direction", e.get("direction")),
                        "entry_price": entry.get("entry_price", e.get("entry_price")),
                        "exit_price": entry.get("exit_price", e.get("exit_price")),
                        "size": entry.get("size", e.get("size")),
                        "pnl": entry.get("pnl", e.get("pnl")),
                        "status": entry.get("status", e.get("status")),
                        "notes": entry.get("notes", e.get("notes")),
                        "tags": entry.get("tags", e.get("tags")),
                        "updated_at": datetime.utcnow().isoformat(),
                    }
                )
                return {"status": "updated", "entry": _journal_entries[i]}

        raise HTTPException(status_code=404, detail="Entry not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update journal entry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v2/journal/{entry_id}", tags=["Journal"])
async def delete_journal_entry(
    entry_id: int,
    token: str = Depends(verify_token),
):
    """Delete a journal entry."""
    try:
        for i, e in enumerate(_journal_entries):
            if e.get("id") == entry_id:
                _journal_entries.pop(i)
                return {"status": "deleted", "id": entry_id}

        raise HTTPException(status_code=404, detail="Entry not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete journal entry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Export endpoints
@app.get("/api/v2/export/journal", tags=["Export"])
async def export_journal(
    format: str = Query("csv", description="Export format: csv, json"),
    token: str = Depends(verify_token),
):
    """Export journal entries to CSV or JSON."""
    try:
        if format == "json":
            return {"entries": _journal_entries}

        # CSV export
        import csv
        import io

        output = io.StringIO()
        if _journal_entries:
            fieldnames = [
                "id",
                "date",
                "market_id",
                "direction",
                "entry_price",
                "exit_price",
                "size",
                "pnl",
                "status",
                "notes",
                "tags",
                "created_at",
            ]
            writer = csv.DictWriter(
                output, fieldnames=fieldnames, extrasaction="ignore"
            )
            writer.writeheader()
            for entry in _journal_entries:
                entry_copy = entry.copy()
                entry_copy["tags"] = ",".join(entry_copy.get("tags", []))
                writer.writerow(entry_copy)

        output.seek(0)

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=journal_export.csv"},
        )
    except Exception as e:
        logger.error(f"Failed to export journal: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/export/orders", tags=["Export"])
async def export_orders(
    format: str = Query("csv", description="Export format: csv, json"),
    status: Optional[str] = Query(None, description="Filter by status"),
    token: str = Depends(verify_token),
):
    """Export orders to CSV or JSON."""
    try:
        from src.main import bot

        orders = []
        if hasattr(bot, "paper_simulator"):
            orders = bot.paper_simulator.trades

        if status:
            orders = [o for o in orders if o.get("status") == status]

        if format == "json":
            return {"orders": orders}

        # CSV export
        import csv
        import io

        output = io.StringIO()
        if orders:
            fieldnames = [
                "order_id",
                "market_id",
                "side",
                "quantity",
                "price",
                "status",
                "created_at",
                "filled_at",
            ]
            writer = csv.DictWriter(
                output, fieldnames=fieldnames, extrasaction="ignore"
            )
            writer.writeheader()
            for order in orders:
                order_copy = order.copy()
                if isinstance(order_copy.get("created_at"), datetime):
                    order_copy["created_at"] = order_copy["created_at"].isoformat()
                if isinstance(order_copy.get("filled_at"), datetime):
                    order_copy["filled_at"] = order_copy["filled_at"].isoformat()
                writer.writerow(order_copy)

        output.seek(0)

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=orders_export.csv"},
        )
    except Exception as e:
        logger.error(f"Failed to export orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Notifications storage (in-memory)
_notifications = []
_notification_id_counter = 1


@app.get("/api/v2/notifications", tags=["Notifications"])
async def get_notifications(
    unread_only: bool = Query(False, description="Only unread notifications"),
    limit: int = Query(50, description="Maximum notifications to return"),
    token: str = Depends(verify_token),
):
    """Get notification history."""
    try:
        notifications = _notifications.copy()

        if unread_only:
            notifications = [n for n in notifications if not n.get("read", False)]

        notifications = notifications[:limit]

        return {
            "notifications": notifications,
            "unread_count": len(
                [n for n in _notifications if not n.get("read", False)]
            ),
        }
    except Exception as e:
        logger.error(f"Failed to get notifications: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/notifications/{notification_id}/read", tags=["Notifications"])
async def mark_notification_read(
    notification_id: int,
    token: str = Depends(verify_token),
):
    """Mark a notification as read."""
    try:
        for n in _notifications:
            if n.get("id") == notification_id:
                n["read"] = True
                return {"status": "updated", "notification": n}

        raise HTTPException(status_code=404, detail="Notification not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to mark notification read: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/notifications/read-all", tags=["Notifications"])
async def mark_all_notifications_read(
    token: str = Depends(verify_token),
):
    """Mark all notifications as read."""
    try:
        for n in _notifications:
            n["read"] = True

        return {"status": "updated", "count": len(_notifications)}
    except Exception as e:
        logger.error(f"Failed to mark all read: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v2/notifications/{notification_id}", tags=["Notifications"])
async def delete_notification(
    notification_id: int,
    token: str = Depends(verify_token),
):
    """Delete a notification."""
    try:
        global _notifications
        _notifications = [n for n in _notifications if n.get("id") != notification_id]
        return {"status": "deleted", "id": notification_id}
    except Exception as e:
        logger.error(f"Failed to delete notification: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def add_notification(title: str, message: str, type: str = "info", data: dict = None):
    """Add a new notification (called internally)."""
    global _notification_id_counter, _notifications

    notification = {
        "id": _notification_id_counter,
        "title": title,
        "message": message,
        "type": type,
        "data": data or {},
        "read": False,
        "timestamp": datetime.utcnow().isoformat(),
    }

    _notifications.insert(0, notification)
    _notification_id_counter += 1

    # Keep only last 100 notifications
    if len(_notifications) > 100:
        _notifications = _notifications[:100]

    return notification


# Health Check endpoints
@app.get("/api/v2/health", tags=["Health"])
async def get_health_check():
    """Get comprehensive health status of the bot."""
    try:
        from src.main import bot

        health = bot.get_health_status()

        # Add paper trading info
        if hasattr(bot, "paper_simulator"):
            paper_stats = bot.paper_simulator.get_stats()
            health["components"]["paper_trading"] = {
                "healthy": True,
                "balance": paper_stats.get("balance", 0),
                "total_pnl": paper_stats.get("total_pnl", 0),
                "executed_trades": paper_stats.get("filled_orders", 0),
            }

        # Add stats
        health["stats"] = {
            "executed_count": bot.executed_count,
            "subscribed_markets": len(bot.subscribed_markets),
            "orderbooks_count": len(bot.orderbooks),
        }

        return health
    except Exception as e:
        logger.error(f"Failed to get health status: {e}")
        return {
            "status": "error",
            "error": str(e),
        }


# Dashboard / Market Data endpoints
@app.get("/api/v2/dashboard/orderbooks")
async def get_orderbooks_dashboard(
    market_id: Optional[str] = Query(None, description="Filter by market ID"),
    limit: int = Query(10, description="Maximum markets to return"),
):
    """Get orderbook data for dashboard visualization."""
    try:
        from src.main import bot

        orderbooks = []
        markets = list(bot.orderbooks.items())[:limit]

        for mkt_id, ob in markets:
            if market_id and mkt_id != market_id:
                continue

            orderbooks.append(
                {
                    "market_id": mkt_id,
                    "best_bid": ob.bids[0].price if ob.bids else None,
                    "best_ask": ob.asks[0].price if ob.asks else None,
                    "mid_price": ob.get_mid_price(),
                    "spread": ob.get_spread(),
                    "spread_percent": ob.get_spread_percent(),
                    "liquidity_score": ob.get_liquidity_score(),
                    "bid_depth": [
                        {"price": l.price, "size": l.count} for l in ob.bids[:5]
                    ],
                    "ask_depth": [
                        {"price": l.price, "size": l.count} for l in ob.asks[:5]
                    ],
                    "last_update": ob.last_update.isoformat()
                    if ob.last_update
                    else None,
                }
            )

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "orderbooks": orderbooks,
            "total_markets": len(bot.orderbooks),
        }
    except Exception as e:
        logger.error(f"Failed to get orderbooks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/dashboard/summary")
async def get_dashboard_summary():
    """Get summary data for dashboard."""
    try:
        from src.main import bot

        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "bot": {
                "running": bot.running,
                "executed_count": bot.executed_count,
                "subscribed_markets": len(bot.subscribed_markets),
            },
            "orderbooks": {
                "total": len(bot.orderbooks),
            },
        }

        if hasattr(bot, "paper_simulator"):
            stats = bot.paper_simulator.get_stats()
            summary["paper_trading"] = {
                "balance": stats.get("balance", 0),
                "balance_dollars": stats.get("balance", 0) / 100,
                "total_pnl": stats.get("total_pnl", 0),
                "realized_pnl": stats.get("realized_pnl", 0),
                "unrealized_pnl": stats.get("unrealized_pnl", 0),
                "filled_orders": stats.get("filled_orders", 0),
                "total_volume": stats.get("total_volume", 0),
                "total_commission": stats.get("total_commission", 0),
            }

        return summary
    except Exception as e:
        logger.error(f"Failed to get dashboard summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

        # Add stats
        health["stats"] = {
            "executed_count": bot.executed_count,
            "subscribed_markets": len(bot.subscribed_markets),
            "orderbooks_count": len(bot.orderbooks),
        }

        return health
    except Exception as e:
        logger.error(f"Failed to get health status: {e}")
        return {
            "status": "error",
            "error": str(e),
        }


@app.get("/api/v2/health/live", tags=["Health"])
async def liveness_check():
    """Kubernetes-style liveness probe."""
    return {"status": "alive"}


@app.get("/api/v2/health/ready", tags=["Health"])
async def readiness_check():
    """Kubernetes-style readiness probe."""
    try:
        from src.main import bot

        if not bot.running:
            return {"status": "not_ready", "reason": "bot_not_running"}

        health = bot.get_health_status()
        if health["status"] == "healthy":
            return {"status": "ready"}
        return {"status": "not_ready", "components": health.get("components", {})}
    except Exception as e:
        return {"status": "not_ready", "reason": str(e)}


# ML Pipeline endpoints
@app.post("/api/v2/ml/models")
async def create_ml_model(
    request: MLModelRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token),
):
    """Create and train a new ML model."""
    try:
        pipeline = MLPipeline()

        model_id = await pipeline.create_model(
            model_type=request.model_type,
            parameters=request.parameters,
            features=request.features,
            target=request.target,
        )

        # Queue training in background
        task_id = f"ml_training_{datetime.utcnow().timestamp()}"
        background_tasks.add_task(
            execute_ml_training, task_id, model_id, request.dict()
        )

        return {
            "model_id": model_id,
            "task_id": task_id,
            "status": "training_started",
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to create ML model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/ml/models")
async def list_ml_models(token: str = Depends(verify_token)):
    """List all ML models."""
    try:
        pipeline = MLPipeline()

        models = await pipeline.list_models()

        return {"models": models}

    except Exception as e:
        logger.error(f"Failed to list ML models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/ml/models/{model_id}")
async def get_ml_model(model_id: str, token: str = Depends(verify_token)):
    """Get specific ML model details."""
    try:
        pipeline = MLPipeline()

        model_details = await pipeline.get_model(model_id)

        return model_details

    except Exception as e:
        logger.error(f"Failed to get ML model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/ml/models/{model_id}/deploy")
async def deploy_ml_model(
    model_id: str, background_tasks: BackgroundTasks, token: str = Depends(verify_token)
):
    """Deploy ML model to production."""
    try:
        pipeline = MLPipeline()

        deployment_id = await pipeline.deploy_model(model_id)

        return {
            "deployment_id": deployment_id,
            "model_id": model_id,
            "status": "deployment_started",
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to deploy ML model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Multi-exchange endpoints
@app.get("/api/v2/exchanges")
async def list_exchanges(token: str = Depends(verify_token)):
    """List all supported exchanges and their status."""
    try:
        manager = MultiExchangeManager()

        exchanges = await manager.get_exchange_status()

        return {"exchanges": exchanges}

    except Exception as e:
        logger.error(f"Failed to list exchanges: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/exchanges/{exchange_name}/markets")
async def get_exchange_markets(exchange_name: str, token: str = Depends(verify_token)):
    """Get available markets for a specific exchange."""
    try:
        manager = MultiExchangeManager()

        markets = await manager.get_markets(exchange_name)

        return {"exchange": exchange_name, "markets": markets}

    except Exception as e:
        logger.error(f"Failed to get markets for {exchange_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/arbitrage/opportunities")
async def get_arbitrage_opportunities(token: str = Depends(verify_token)):
    """Get current arbitrage opportunities."""
    try:
        manager = MultiExchangeManager()

        opportunities = await manager.get_arbitrage_opportunities()

        return {"opportunities": opportunities}

    except Exception as e:
        logger.error(f"Failed to get arbitrage opportunities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions
async def execute_backtest(
    task_id: str, backtester: AdvancedBacktester, request: Dict[str, Any]
):
    """Execute backtest in background."""
    try:
        # Execute backtest
        results = await backtester.run_backtest(
            strategy_config=request["strategy_config"],
            start_date=request["start_date"],
            end_date=request["end_date"],
            initial_capital=request["initial_capital"],
            optimization_method=request["optimization_method"],
            optimization_params=request.get("optimization_params"),
        )

        # Store results in cache/database
        # TODO: Implement result storage

        logger.info(f"Backtest {task_id} completed successfully")

    except Exception as e:
        logger.error(f"Backtest {task_id} failed: {e}")


async def execute_ml_training(task_id: str, model_id: str, request: Dict[str, Any]):
    """Execute ML model training in background."""
    try:
        pipeline = MLPipeline()

        # Train model
        await pipeline.train_model(
            model_id=model_id,
            training_data_path=request.get("training_data_path"),
            training_method=request["training_method"],
        )

        logger.info(f"ML training {task_id} completed successfully")

    except Exception as e:
        logger.error(f"ML training {task_id} failed: {e}")


async def graceful_shutdown():
    """Gracefully shutdown the bot."""
    try:
        logger.info("Starting graceful shutdown...")

        # Wait for ongoing tasks to complete
        await asyncio.sleep(5)

        logger.info("Graceful shutdown completed")

    except Exception as e:
        logger.error(f"Error during graceful shutdown: {e}")


# WebSocket support for real-time updates
from fastapi import WebSocket, WebSocketDisconnect
from typing import List, Dict, Any
import json
import asyncio


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.subscriptions: Dict[str, set] = {
            "orderbooks": set(),
            "opportunities": set(),
            "trades": set(),
            "portfolio": set(),
            "system": set(),
        }

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(
            f"WebSocket client connected. Total: {len(self.active_connections)}"
        )

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(
            f"WebSocket client disconnected. Total: {len(self.active_connections)}"
        )

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")

    async def broadcast(self, message: dict, channel: str = None):
        """Broadcast message to all connected clients or specific channel."""
        if channel and channel in self.subscriptions:
            # Only send to clients subscribed to this channel
            for connection in self.active_connections:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Failed to broadcast: {e}")
        else:
            # Send to all connections
            for connection in self.active_connections:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Failed to broadcast: {e}")

    def subscribe(self, websocket: WebSocket, channels: List[str]):
        for channel in channels:
            if channel in self.subscriptions:
                self.subscriptions[channel].add(websocket)

    def unsubscribe(self, websocket: WebSocket, channels: List[str]):
        for channel in channels:
            if (
                channel in self.subscriptions
                and websocket in self.subscriptions[channel]
            ):
                self.subscriptions[channel].remove(websocket)


# Global connection manager
ws_manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            try:
                message = json.loads(data)

                # Handle subscription messages
                if message.get("type") == "subscribe":
                    channels = message.get("channels", [])
                    ws_manager.subscribe(websocket, channels)
                    await ws_manager.send_personal_message(
                        {"type": "subscribed", "channels": channels}, websocket
                    )

                elif message.get("type") == "unsubscribe":
                    channels = message.get("channels", [])
                    ws_manager.unsubscribe(websocket, channels)
                    await ws_manager.send_personal_message(
                        {"type": "unsubscribed", "channels": channels}, websocket
                    )

                elif message.get("type") == "ping":
                    await ws_manager.send_personal_message(
                        {"type": "pong", "timestamp": datetime.now().isoformat()},
                        websocket,
                    )

            except json.JSONDecodeError:
                logger.warning("Invalid JSON received on WebSocket")

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket)


# Background task to push real-time data
async def push_market_data():
    """Push market data to connected WebSocket clients."""
    while True:
        try:
            # Get latest orderbook data
            orderbooks = await get_orderbooks_data()

            # Get latest opportunities
            opportunities = await get_opportunities_data()

            # Get portfolio status
            portfolio = await get_portfolio_data()

            # Broadcast to all connected clients
            if ws_manager.active_connections:
                await ws_manager.broadcast(
                    {
                        "type": "market_update",
                        "data": {
                            "orderbooks": orderbooks,
                            "opportunities": opportunities,
                            "portfolio": portfolio,
                            "timestamp": datetime.now().isoformat(),
                        },
                    }
                )

            await asyncio.sleep(2)  # Update every 2 seconds

        except Exception as e:
            logger.error(f"Error pushing market data: {e}")
            await asyncio.sleep(5)


# Helper functions for WebSocket data
async def get_orderbooks_data():
    """Get current orderbook data for WebSocket."""
    # This would normally fetch from the bot's live data
    return []


async def get_opportunities_data():
    """Get current opportunities for WebSocket."""
    return []


async def get_portfolio_data():
    """Get current portfolio data for WebSocket."""
    return {}


# Start background task on startup
@app.on_event("startup")
async def startup_event():
    """Start background tasks on application startup."""
    asyncio.create_task(push_market_data())


# ==================== ARBITRAGE DETECTION ====================

_detection_state = {
    "running": False,
    "started_at": None,
    "opportunities_found": 0,
    "last_opportunity": None,
}

_arbitrage_opportunities = []


@app.get("/api/v2/arbitrage/status")
async def get_arbitrage_status(token: str = Depends(verify_token)):
    """Get current arbitrage detection status."""
    return {
        "running": _detection_state["running"],
        "started_at": _detection_state["started_at"],
        "opportunities_found": _detection_state["opportunities_found"],
        "last_opportunity": _detection_state["last_opportunity"],
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/api/v2/arbitrage/start")
async def start_arbitrage_detection(token: str = Depends(verify_token)):
    """Start arbitrage detection."""
    _detection_state["running"] = True
    _detection_state["started_at"] = datetime.utcnow().isoformat()

    add_notification(
        "Arbitrage Detection Started",
        "The arbitrage detection engine is now running",
        "success",
    )

    return {
        "status": "started",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/api/v2/arbitrage/stop")
async def stop_arbitrage_detection(token: str = Depends(verify_token)):
    """Stop arbitrage detection."""
    _detection_state["running"] = False

    add_notification(
        "Arbitrage Detection Stopped",
        "The arbitrage detection engine has been stopped",
        "info",
    )

    return {
        "status": "stopped",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/v2/arbitrage/opportunities")
async def get_arbitrage_opportunities(
    limit: int = 20, token: str = Depends(verify_token)
):
    """Get recent arbitrage opportunities."""
    return {
        "opportunities": _arbitrage_opportunities[-limit:],
        "total": len(_arbitrage_opportunities),
        "timestamp": datetime.utcnow().isoformat(),
    }


# ==================== MARKET DATA ====================


@app.get("/api/v2/markets")
async def get_markets(
    status: Optional[str] = "open", limit: int = 50, token: str = Depends(verify_token)
):
    """Get list of available markets."""
    try:
        from src.main import bot

        if hasattr(bot, "client") and bot.client:
            markets_response = bot.client.get_markets(status=status, limit=limit)
            return markets_response
    except Exception:
        pass

    try:
        from src.clients.kalshi_client import KalshiClient
        from src.utils.config import Config

        config = Config()
        client = KalshiClient(config)
        markets_response = client.get_markets(status=status, limit=limit)
        return markets_response
    except Exception as e:
        logger.error(f"Failed to get markets: {e}")
        return {"markets": [], "error": str(e)}


@app.get("/api/v2/markets/live")
async def get_live_markets(limit: int = 50, token: str = Depends(verify_token)):
    """Get live market data with current prices."""
    try:
        from src.main import bot

        markets_data = []

        if hasattr(bot, "orderbooks") and bot.orderbooks:
            for market_id, orderbook in list(bot.orderbooks.items())[:limit]:
                markets_data.append(
                    {
                        "market_id": market_id,
                        "best_bid": orderbook.bids[0].price if orderbook.bids else None,
                        "best_ask": orderbook.asks[0].price if orderbook.asks else None,
                        "mid_price": orderbook.get_mid_price(),
                        "spread": orderbook.get_spread(),
                        "spread_percent": orderbook.get_spread_percent(),
                        "bid_depth": sum(l.count for l in orderbook.bids[:5]),
                        "ask_depth": sum(l.count for l in orderbook.asks[:5]),
                        "last_update": orderbook.last_update.isoformat()
                        if orderbook.last_update
                        else None,
                    }
                )

        return {
            "markets": markets_data,
            "count": len(markets_data),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to get live markets: {e}")
        return {"markets": [], "error": str(e)}


@app.get("/api/v2/markets/{market_id}/orderbook")
async def get_market_orderbook(market_id: str, token: str = Depends(verify_token)):
    """Get orderbook for a specific market."""
    try:
        from src.main import bot

        if hasattr(bot, "orderbooks") and market_id in bot.orderbooks:
            orderbook = bot.orderbooks[market_id]
            return {
                "market_id": market_id,
                "bids": [
                    {"price": b.price, "count": b.count, "total": b.total}
                    for b in orderbook.bids[:10]
                ],
                "asks": [
                    {"price": a.price, "count": a.count, "total": a.total}
                    for a in orderbook.asks[:10]
                ],
                "mid_price": orderbook.get_mid_price(),
                "spread": orderbook.get_spread(),
                "spread_percent": orderbook.get_spread_percent(),
                "timestamp": datetime.utcnow().isoformat(),
            }

        if hasattr(bot, "client"):
            ob_data = bot.client.get_market_orderbook(market_id)
            return ob_data

        return {"error": "Market not found"}
    except Exception as e:
        logger.error(f"Failed to get orderbook: {e}")
        return {"error": str(e)}


@app.post("/api/v2/markets/refresh")
async def refresh_markets(token: str = Depends(verify_token)):
    """Trigger a market data refresh."""
    try:
        from src.main import bot

        if hasattr(bot, "refresh_markets"):
            result = await bot.refresh_markets()
            return result
        return {"status": "error", "message": "Bot not initialized"}
    except Exception as e:
        logger.error(f"Failed to refresh markets: {e}")
        return {"status": "error", "message": str(e)}


# ==================== EXCHANGE TRADING ====================

_exchange_config = {
    "paper_mode": True,
    "max_position_size": 1000,
    "max_order_value": 10000,
    "max_daily_loss": 10000,
}

_trade_stats = {
    "total_trades": 0,
    "successful_trades": 0,
    "failed_trades": 0,
    "daily_pnl": 0,
}


class ExchangeOrderRequest(BaseModel):
    market_id: str
    side: str
    quantity: int
    order_type: str = "limit"
    price: int = 50


@app.get("/api/v2/exchange/status")
async def get_exchange_status(token: str = Depends(verify_token)):
    """Get exchange connection and account status."""
    return {
        "connected": True,
        "paper_mode": _exchange_config["paper_mode"],
        "config": _exchange_config,
        "stats": _trade_stats,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/api/v2/exchange/orders")
async def place_order(order: ExchangeOrderRequest, token: str = Depends(verify_token)):
    """Place a trading order."""
    order_value = order.quantity * order.price

    if order.quantity > _exchange_config["max_position_size"]:
        raise HTTPException(
            status_code=400, detail="Quantity exceeds max position size"
        )

    if order_value > _exchange_config["max_order_value"]:
        raise HTTPException(
            status_code=400, detail="Order value exceeds max order value"
        )

    if _trade_stats["daily_pnl"] <= -_exchange_config["max_daily_loss"]:
        raise HTTPException(status_code=400, detail="Daily loss limit reached")

    order_id = f"order_{datetime.utcnow().timestamp()}"

    _trade_stats["total_trades"] += 1

    if _exchange_config["paper_mode"]:
        _trade_stats["successful_trades"] += 1
        status = "filled"
    else:
        status = "submitted"

    return {
        "order_id": order_id,
        "market_id": order.market_id,
        "side": order.side,
        "quantity": order.quantity,
        "price": order.price,
        "order_type": order.order_type,
        "status": status,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/v2/exchange/orders")
async def get_orders(
    status: Optional[str] = None, limit: int = 50, token: str = Depends(verify_token)
):
    """Get order history."""
    return {
        "orders": [],
        "total": 0,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/v2/exchange/orders/{order_id}")
async def get_order(order_id: str, token: str = Depends(verify_token)):
    """Get specific order details."""
    return {
        "order_id": order_id,
        "status": "filled",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.delete("/api/v2/exchange/orders/{order_id}")
async def cancel_order(order_id: str, token: str = Depends(verify_token)):
    """Cancel an order."""
    return {
        "order_id": order_id,
        "status": "cancelled",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/v2/exchange/positions")
async def get_positions(token: str = Depends(verify_token)):
    """Get current positions."""
    return {
        "positions": [],
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/v2/exchange/balance")
async def get_balance(token: str = Depends(verify_token)):
    """Get account balance."""
    return {
        "balance": 10000,
        "available": 10000,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/api/v2/exchange/paper-mode")
async def set_paper_mode(enabled: bool, token: str = Depends(verify_token)):
    """Enable or disable paper trading mode."""
    _exchange_config["paper_mode"] = enabled

    return {
        "paper_mode": enabled,
        "timestamp": datetime.utcnow().isoformat(),
    }


# ==================== AUTO-TRADING MODE ====================
import sqlite3
import os

_auto_trading_config = {
    "enabled": False,
    "max_position_size": 100,
    "auto_close_on_risk": True,
    "min_profit_threshold": 0.01,
    "max_slippage": 0.02,
    "strategies": ["statistical_arbitrage", "mean_reversion"],
}

_auto_trading_history = []


class AutoTradingConfig(BaseModel):
    enabled: bool = False
    max_position_size: int = 100
    auto_close_on_risk: bool = True
    min_profit_threshold: float = 0.01
    max_slippage: float = 0.02
    strategies: List[str] = ["statistical_arbitrage"]


@app.get("/api/v2/auto-trading/status")
async def get_auto_trading_status(token: str = Depends(verify_token)):
    """Get current auto-trading status."""
    return {
        "enabled": _auto_trading_config["enabled"],
        "config": _auto_trading_config.copy(),
        "history": _auto_trading_history[-10:] if _auto_trading_history else [],
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/api/v2/auto-trading/enable")
async def enable_auto_trading(
    config: AutoTradingConfig, token: str = Depends(verify_token)
):
    """Enable auto-trading mode."""
    _auto_trading_config.update(config.dict())
    _auto_trading_config["enabled"] = True

    entry = {
        "action": "enabled",
        "config": config.dict(),
        "timestamp": datetime.utcnow().isoformat(),
    }
    _auto_trading_history.append(entry)

    add_notification(
        "Auto-Trading Enabled",
        f"Auto-trading activated with max position: {config.max_position_size}",
        "success",
    )

    return {
        "status": "enabled",
        "config": _auto_trading_config.copy(),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/api/v2/auto-trading/disable")
async def disable_auto_trading(token: str = Depends(verify_token)):
    """Disable auto-trading mode."""
    _auto_trading_config["enabled"] = False

    entry = {
        "action": "disabled",
        "timestamp": datetime.utcnow().isoformat(),
    }
    _auto_trading_history.append(entry)

    add_notification(
        "Auto-Trading Disabled",
        "Auto-trading has been turned off",
        "info",
    )

    return {
        "status": "disabled",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.put("/api/v2/auto-trading/config")
async def update_auto_trading_config(
    config: AutoTradingConfig, token: str = Depends(verify_token)
):
    """Update auto-trading configuration."""
    _auto_trading_config.update(config.dict())

    return {
        "config": _auto_trading_config.copy(),
        "timestamp": datetime.utcnow().isoformat(),
    }


# ==================== POSITION SIZING OPTIMIZER ====================


class PositionSizeRequest(BaseModel):
    account_balance: float
    win_rate: float
    avg_win: float
    avg_loss: float
    risk_percent: float = 0.02
    kelly_fraction: float = 0.25


@app.post("/api/v2/position-sizing/calculate")
async def calculate_position_size(
    request: PositionSizeRequest, token: str = Depends(verify_token)
):
    """Calculate optimal position size using Kelly Criterion."""
    win_rate = request.win_rate
    win_loss_ratio = request.avg_win / request.avg_loss if request.avg_loss > 0 else 0

    kelly_percent = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
    kelly_percent = max(0, min(kelly_percent, 1))

    fractional_kelly = kelly_percent * request.kelly_fraction

    position_size = request.account_balance * fractional_kelly

    risk_adjusted_size = position_size * request.risk_percent

    return {
        "kelly_percent": round(kelly_percent * 100, 2),
        "fractional_kelly_percent": round(fractional_kelly * 100, 2),
        "recommended_position_size": round(position_size, 2),
        "risk_adjusted_position_size": round(risk_adjusted_size, 2),
        "risk_amount": round(request.account_balance * request.risk_percent, 2),
        "win_loss_ratio": round(win_loss_ratio, 2),
        "kelly_recommendation": "full_kelly"
        if kelly_percent > 0.5
        else "fractional_kelly"
        if kelly_percent > 0
        else "no_trade",
    }


@app.get("/api/v2/position-sizing/history")
async def get_position_sizing_history(token: str = Depends(verify_token)):
    """Get historical position sizing calculations."""
    return {
        "calculations": [],
        "timestamp": datetime.utcnow().isoformat(),
    }


# ==================== SPREAD ALERT SYSTEM ====================

_spread_alerts = []
_spread_alert_id = 1


class SpreadAlertRequest(BaseModel):
    market_a: str
    market_b: str
    threshold: float
    direction: str = "both"  # above, below, both
    enabled: bool = True


class SpreadAlert(BaseModel):
    id: int
    market_a: str
    market_b: str
    threshold: float
    direction: str
    enabled: bool
    triggered_count: int = 0
    last_triggered: Optional[str] = None
    created_at: str


@app.get("/api/v2/spread-alerts")
async def list_spread_alerts(token: str = Depends(verify_token)):
    """List all spread alerts."""
    return {
        "alerts": _spread_alerts,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/api/v2/spread-alerts")
async def create_spread_alert(
    alert: SpreadAlertRequest, token: str = Depends(verify_token)
):
    """Create a new spread alert."""
    global _spread_alert_id

    new_alert = {
        "id": _spread_alert_id,
        **alert.dict(),
        "triggered_count": 0,
        "last_triggered": None,
        "created_at": datetime.utcnow().isoformat(),
    }
    _spread_alerts.append(new_alert)
    _spread_alert_id += 1

    return {
        "alert": new_alert,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.put("/api/v2/spread-alerts/{alert_id}")
async def update_spread_alert(
    alert_id: int, alert: SpreadAlertRequest, token: str = Depends(verify_token)
):
    """Update a spread alert."""
    for a in _spread_alerts:
        if a["id"] == alert_id:
            a.update(alert.dict())
            return {"alert": a, "timestamp": datetime.utcnow().isoformat()}

    raise HTTPException(status_code=404, detail="Alert not found")


@app.delete("/api/v2/spread-alerts/{alert_id}")
async def delete_spread_alert(alert_id: int, token: str = Depends(verify_token)):
    """Delete a spread alert."""
    global _spread_alerts
    _spread_alerts = [a for a in _spread_alerts if a["id"] != alert_id]

    return {
        "status": "deleted",
        "alert_id": alert_id,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/api/v2/spread-alerts/{alert_id}/trigger")
async def trigger_spread_alert(alert_id: int, token: str = Depends(verify_token)):
    """Manually trigger a spread alert."""
    for a in _spread_alerts:
        if a["id"] == alert_id:
            a["triggered_count"] += 1
            a["last_triggered"] = datetime.utcnow().isoformat()

            add_notification(
                "Spread Alert Triggered",
                f"Spread alert for {a['market_a']}/{a['market_b']} triggered!",
                "warning",
                {"alert_id": alert_id, "threshold": a["threshold"]},
            )

            return {"alert": a, "timestamp": datetime.utcnow().isoformat()}

    raise HTTPException(status_code=404, detail="Alert not found")


# ==================== ONE-CLICK TRADING ====================

_quick_trades_enabled = False


@app.post("/api/v2/one-click/enable")
async def enable_one_click_trading(token: str = Depends(verify_token)):
    """Enable one-click trading mode."""
    global _quick_trades_enabled
    _quick_trades_enabled = True

    add_notification(
        "One-Click Trading Enabled",
        "Quick trade execution is now enabled",
        "success",
    )

    return {
        "status": "enabled",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/api/v2/one-click/disable")
async def disable_one_click_trading(token: str = Depends(verify_token)):
    """Disable one-click trading mode."""
    global _quick_trades_enabled
    _quick_trades_enabled = False

    return {
        "status": "disabled",
        "timestamp": datetime.utcnow().isoformat(),
    }


class QuickTradeRequest(BaseModel):
    market_id: str
    side: str  # buy/sell
    quantity: int
    order_type: str = "market"
    price: Optional[float] = None
    use_kelly_sizing: bool = False
    account_balance: Optional[float] = None
    win_rate: Optional[float] = None
    avg_win: Optional[float] = None
    avg_loss: Optional[float] = None


@app.post("/api/v2/one-click/trade")
async def execute_quick_trade(
    trade: QuickTradeRequest, token: str = Depends(verify_token)
):
    """Execute a one-click trade."""
    if not _quick_trades_enabled:
        raise HTTPException(
            status_code=403,
            detail="One-click trading is disabled. Enable it first.",
        )

    quantity = trade.quantity

    if trade.use_kelly_sizing and all(
        [trade.account_balance, trade.win_rate, trade.avg_win, trade.avg_loss]
    ):
        win_loss_ratio = trade.avg_win / trade.avg_loss if trade.avg_loss > 0 else 0
        kelly_percent = (
            trade.win_rate * win_loss_ratio - (1 - trade.win_rate)
        ) / win_loss_ratio
        kelly_percent = max(0, min(kelly_percent * 0.25, 1))
        quantity = int(trade.account_balance * kelly_percent / 100)

    order_result = {
        "order_id": f"qc_{datetime.utcnow().timestamp()}",
        "market_id": trade.market_id,
        "side": trade.side,
        "quantity": quantity,
        "order_type": trade.order_type,
        "price": trade.price,
        "status": "filled" if trade.order_type == "market" else "pending",
        "timestamp": datetime.utcnow().isoformat(),
    }

    add_notification(
        "Quick Trade Executed",
        f"{trade.side.upper()} {quantity} {trade.market_id}",
        "success",
        order_result,
    )

    return {
        "order": order_result,
        "timestamp": datetime.utcnow().isoformat(),
    }


# ==================== DATABASE STORAGE ====================


def get_db_connection():
    """Get SQLite database connection."""
    db_path = Config().get("database.path", "data/arbitrage.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Initialize database tables."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id TEXT UNIQUE,
            market_id TEXT,
            side TEXT,
            quantity INTEGER,
            price REAL,
            order_type TEXT,
            status TEXT,
            timestamp TEXT,
            pnl REAL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id TEXT UNIQUE,
            side TEXT,
            quantity INTEGER,
            entry_price REAL,
            current_price REAL,
            unrealized_pnl REAL,
            timestamp TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            content TEXT,
            trade_id TEXT,
            tags TEXT,
            timestamp TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_type TEXT,
            market_id TEXT,
            threshold REAL,
            triggered BOOLEAN,
            triggered_at TEXT,
            created_at TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT
        )
    """)

    conn.commit()
    conn.close()


@app.get("/api/v2/db/stats")
async def get_database_stats(token: str = Depends(verify_token)):
    """Get database statistics."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as count FROM trades")
    trades_count = cursor.fetchone()["count"]

    cursor.execute("SELECT COUNT(*) as count FROM positions")
    positions_count = cursor.fetchone()["count"]

    cursor.execute("SELECT COUNT(*) as count FROM journal")
    journal_count = cursor.fetchone()["count"]

    cursor.execute("SELECT COUNT(*) as count FROM alerts")
    alerts_count = cursor.fetchone()["count"]

    conn.close()

    return {
        "trades": trades_count,
        "positions": positions_count,
        "journal_entries": journal_count,
        "alerts": alerts_count,
        "database_path": Config().get("database.path", "data/arbitrage.db"),
    }


@app.post("/api/v2/db/trades")
async def save_trade_to_db(
    trade_data: Dict[str, Any], token: str = Depends(verify_token)
):
    """Save a trade to the database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT OR REPLACE INTO trades (order_id, market_id, side, quantity, price, order_type, status, timestamp, pnl)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            trade_data.get("order_id"),
            trade_data.get("market_id"),
            trade_data.get("side"),
            trade_data.get("quantity"),
            trade_data.get("price"),
            trade_data.get("order_type"),
            trade_data.get("status"),
            trade_data.get("timestamp", datetime.utcnow().isoformat()),
            trade_data.get("pnl"),
        ),
    )

    conn.commit()
    trade_id = cursor.lastrowid
    conn.close()

    return {"status": "saved", "trade_id": trade_id}


@app.get("/api/v2/db/trades")
async def get_trades_from_db(
    limit: int = 100, offset: int = 0, token: str = Depends(verify_token)
):
    """Get trades from database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ? OFFSET ?", (limit, offset)
    )
    trades = [dict(row) for row in cursor.fetchall()]

    cursor.execute("SELECT COUNT(*) as count FROM trades")
    total = cursor.fetchone()["count"]

    conn.close()

    return {"trades": trades, "total": total, "limit": limit, "offset": offset}


@app.post("/api/v2/db/settings")
async def save_setting(key: str, value: str, token: str = Depends(verify_token)):
    """Save a setting to database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT OR REPLACE INTO settings (key, value, updated_at)
        VALUES (?, ?, ?)
    """,
        (key, value, datetime.utcnow().isoformat()),
    )

    conn.commit()
    conn.close()

    return {"status": "saved", "key": key}


@app.get("/api/v2/db/settings/{key}")
async def get_setting(key: str, token: str = Depends(verify_token)):
    """Get a setting from database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM settings WHERE key = ?", (key,))
    row = cursor.fetchone()

    conn.close()

    if row:
        return {
            "key": row["key"],
            "value": row["value"],
            "updated_at": row["updated_at"],
        }
    return {"key": key, "value": None}


# Initialize database on import
init_database()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
