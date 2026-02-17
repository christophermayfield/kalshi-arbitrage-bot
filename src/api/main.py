from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from typing import Any, Dict, Optional
from contextlib import asynccontextmanager

from src.utils.config import Config
from src.utils.logging_utils import setup_logging
from src.core.portfolio import PortfolioManager
from src.core.circuit_breaker import CircuitBreakerManager
from src.core.backtest import Backtester, BacktestConfig
from src.clients.kalshi_client import KalshiClient

logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting API server")
    yield
    logger.info("Shutting down API server")


app = FastAPI(
    title="Kalshi Arbitrage Bot API",
    description="API for monitoring and controlling the Kalshi arbitrage bot",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_global_state: Dict[str, Any] = {}


def get_state() -> Dict[str, Any]:
    if not _global_state:
        _global_state['config'] = Config()
        _global_state['portfolio'] = PortfolioManager()
        _global_state['circuit_breaker'] = CircuitBreakerManager()
        _global_state['client'] = None
        _global_state['bot_running'] = False
    return _global_state


@app.get("/")
async def root():
    return {"status": "ok", "message": "Kalshi Arbitrage Bot API"}


@app.get("/health")
async def health():
    state = get_state()
    return {
        "status": "healthy",
        "bot_running": state.get('bot_running', False),
        "portfolio": state['portfolio'].get_stats().__dict__
    }


@app.get("/metrics")
async def metrics():
    return StreamingResponse(
        iter([generate_latest().decode()]),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/portfolio")
async def get_portfolio():
    state = get_state()
    stats = state['portfolio'].get_stats()
    return stats.__dict__


@app.get("/positions")
async def get_positions():
    state = get_state()
    return {
        market_id: pos.__dict__
        for market_id, pos in state['portfolio'].positions.items()
    }


@app.get("/orders")
async def get_orders():
    state = get_state()
    client = state.get('client')
    if not client:
        return {"error": "Bot not connected to Kalshi"}
    try:
        orders = client.get_orders()
        return orders
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/circuit-breakers")
async def get_circuit_breakers():
    state = get_state()
    return state['circuit_breaker'].get_all_stats()


@app.post("/circuit-breaker/{name}/reset")
async def reset_circuit_breaker(name: str):
    state = get_state()
    breaker = state['circuit_breaker'].get(name)
    if breaker:
        breaker.reset()
        return {"status": "reset"}
    raise HTTPException(status_code=404, detail="Circuit breaker not found")


@app.post("/backtest")
async def run_backtest(
    config: Optional[Dict[str, Any]] = None,
    background_tasks: BackgroundTasks = None
):
    bt_config = BacktestConfig(
        initial_balance=config.get('initial_balance', 100000) if config else 100000,
        fee_rate=config.get('fee_rate', 0.01) if config else 0.01
    )

    backtester = Backtester(bt_config)

    sample_data = {
        'market-1': [
            {
                'market_id': 'market-1',
                'bids': [{'price': '60', 'count': 10, 'total': 600}],
                'asks': [{'price': '55', 'count': 10, 'total': 550}]
            }
        ],
        'market-2': [
            {
                'market_id': 'market-2',
                'bids': [{'price': '50', 'count': 10, 'total': 500}],
                'asks': [{'price': '65', 'count': 10, 'total': 650}]
            }
        ]
    }
    timestamps = ['2024-01-01T00:00:00Z']

    stats = backtester.run(sample_data, timestamps)
    report = backtester.generate_report(stats)

    return report


@app.get("/config")
async def get_config():
    state = get_state()
    return state['config']._config


@app.post("/config/reload")
async def reload_config():
    state = get_state()
    state['config'].reload()
    return {"status": "reloaded"}


@app.post("/start")
async def start_bot():
    state = get_state()
    if state.get('bot_running'):
        return {"status": "already_running"}
    state['bot_running'] = True
    return {"status": "started"}


@app.post("/stop")
async def stop_bot():
    state = get_state()
    if not state.get('bot_running'):
        return {"status": "not_running"}
    state['bot_running'] = False
    return {"status": "stopped"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
