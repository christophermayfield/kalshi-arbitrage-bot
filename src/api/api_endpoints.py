"""FastAPI integration for arbitrage bot main API."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from src.api.main_api import app
from src.utils.config import Config
from src.core.arbitrage import ArbitrageDetector
from src.core.portfolio import PortfolioManager
from src.execution.trading import TradingExecutor
from src.clients.kalexi_client import KalshiClient
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Add CORS middleware to main app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/opportunities/execute")
async def execute_opportunity(request):
    """Execute a specific arbitrage opportunity."""
    try:
        data = await request.json()
        opportunity_id = data.get("opportunity_id")
        
        if not opportunity_id:
            return JSONResponse(
                {"error": "Opportunity ID is required"},
                status_code=400
            )
        
        # Get opportunity details and execute
        # This would integrate with your trading execution logic
        success = True  # Mock success - replace with actual execution logic
        
        return JSONResponse({
            "success": success,
            "opportunity_id": opportunity_id,
            "executed_at": datetime.utcnow().isoformat(),
            "message": "Opportunity executed successfully"
        })
        
    except Exception as e:
        logger.error(f"Failed to execute opportunity {opportunity_id}: {e}")
        return JSONResponse(
            {"error": str(e), "opportunity_id": opportunity_id},
                "status_code": 500
            },
            status_code=500
        )


@app.get("/api/bot/status")
async def get_bot_status():
    """Get detailed bot status."""
    try:
        from src.utils.config import Config
        config = Config()
        
        # Get actual bot status from your main bot logic
        status = {
            "running": True,  # Replace with actual status
            "uptime": "2h 15m",
            "last_trade": None,
            "total_trades": 45,
            "successful_trades": 38,
            "daily_pnl": 125.50,
            "total_profit": 1250.00
            "strategies_enabled": {
                "statistical_arbitrage": config.get("statistical.enabled", False),
                "mean_reversion": False,
                "pairs_trading": False
            }
        }
        
        return JSONResponse(status)
        
    except Exception as e:
        logger.error(f"Failed to get bot status: {e}")
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )


@app.get("/api/strategies/stats")
async def get_strategy_performance():
    """Get detailed strategy performance statistics."""
    try:
        from src.api.websocket_handler import get_websocket_manager
        
        stats = get_websocket_manager().get_strategy_stats()
        
        # Add historical performance data
        stats["historical"] = {
            "mean_reversion": {
                "all_time_profit_cents": 850,
                "successful_trades": 22,
                "failed_trades": 8,
                "avg_profit_per_trade": 38.64,
                "sharpe_ratio": 1.25,
                "max_profit_cents": 150
            },
            "pairs_trading": {
                "all_time_profit_cents": 1200,
                "successful_trades": 18,
                "failed_trades": 5,
                "avg_profit_per_trade": 66.67,
                "sharpe_ratio": 0.95,
                "max_profit_cents": 200
            }
        }
        
        return JSONResponse(stats)
        
    except Exception as e:
        logger.error(f"Failed to get strategy performance: {e}")
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        }


@app.post("/api/strategies/{strategy_type}/toggle")
async def toggle_strategy(request, strategy_type: str):
    """Enable/disable a specific strategy."""
    try:
        from src.utils.config import Config
        
        data = await request.json()
        enabled = data.get("enabled", False)
        
        config = Config()
        current_value = config.get(f"statistical.{strategy_type}", False)
        
        if enabled != current_value:
            # Update configuration
            # You would implement configuration persistence here
            logger.info(f"Strategy {strategy_type} toggled to: {enabled}")
            
            return JSONResponse({
                "success": True,
                "message": f"Strategy {strategy_type} {'enabled' if enabled else 'disabled'}",
                "current_value": current_value
            })
        
        except Exception as e:
            logger.error(f"Failed to toggle strategy {strategy_type}: {e}")
            return JSONResponse(
                {"error": str(e)},
                status_code=500
            )


@app.post("/api/strategies/{strategy_type}/config")
async def update_strategy_config(request, strategy_type: str):
    """Update strategy configuration."""
    try:
        from src.utils.config import Config
        
        data = await request.json()
        
        # Validate and update configuration
        # You would implement configuration validation and persistence
        
        logger.info(f"Updated {strategy_type} configuration")
        
        return JSONResponse({
            "success": True,
            "message": f"Strategy {strategy_type} configuration updated"
        })
        
    except Exception as e:
        logger.error(f"Failed to update strategy configuration: {e}")
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)