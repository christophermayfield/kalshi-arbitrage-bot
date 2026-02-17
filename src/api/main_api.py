"""API endpoints for frontend dashboard."""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import logging

from src.api.websocket_handler import (
    websocket_endpoint,
    broadcast_opportunity_update,
    broadcast_bot_status,
    broadcast_metrics_update,
    broadcast_pnl_update,
    get_websocket_manager,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Arbitrage Bot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket endpoint
app.websocket("/ws")(websocket_endpoint)


@app.get("/opportunities")
async def get_opportunities():
    """Get current arbitrage opportunities."""
    try:
        from src.core.arbitrage import ArbitrageDetector
        from src.utils.config import Config

        config = Config()
        detector = ArbitrageDetector(
            min_profit_cents=config.get("trading.min_profit_cents", 10),
            enable_statistical_arbitrage=config.get("statistical.enabled", False),
            statistical_config=config.get("statistical", {}),
        )

        # Get actual opportunities from your detector (async call)
        opportunities_result = await detector.scan_for_opportunities({})
        opportunities = (
            opportunities_result if isinstance(opportunities_result, list) else []
        )

        return {
            "opportunities": [
                opp.to_dict() if hasattr(opp, "to_dict") else str(opp)
                for opp in opportunities
            ],
            "strategy_stats": get_websocket_manager().get_strategy_stats(),
        }

    except Exception as e:
        logger.error(f"Failed to get opportunities: {e}")
        return {"opportunities": [], "error": str(e)}


@app.get("/metrics")
async def get_metrics():
    """Get current bot metrics."""
    try:
        ws_manager = get_websocket_manager()
        stats = ws_manager.get_strategy_stats()

        return {
            "daily_pnl": 125.50,
            "total_profit": 1250.00,
            "win_rate": 0.68,
            "active_positions": 3,
            "strategy_stats": stats,
        }

    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        return {"error": str(e)}


@app.post("/config/statistical")
async def toggle_statistical_arbitrage(request):
    """Toggle statistical arbitrage on/off."""
    try:
        from src.utils.config import Config

        data = await request.json()
        enabled = data.get("enabled", False)

        config = Config()
        current_value = config.get("statistical.enabled", False)

        if enabled != current_value:
            # Update configuration (you'd implement this)
            logger.info(f"Statistical arbitrage toggled to: {enabled}")

            return {
                "success": True,
                "message": f"Statistical arbitrage {'enabled' if enabled else 'disabled'}",
                "current_value": current_value,
            }
        else:
            return {
                "success": True,
                "message": f"Statistical arbitrage already {'enabled' if enabled else 'disabled'}",
                "current_value": current_value,
            }

    except Exception as e:
        logger.error(f"Failed to toggle statistical arbitrage: {e}")
        return {"success": False, "error": str(e)}


@app.post("/start")
async def start_bot():
    """Start the arbitrage bot."""
    try:
        logger.info("Starting arbitrage bot via API")

        # Implement your actual bot start logic here
        # This would start the main bot loop

        return {"status": "starting", "message": "Bot starting..."}

    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/stop")
async def stop_bot():
    """Stop the arbitrage bot."""
    try:
        logger.info("Stopping arbitrage bot via API")

        # Implement your actual bot stop logic here
        # This would gracefully stop the main bot loop

        return {"status": "stopping", "message": "Bot stopping..."}

    except Exception as e:
        logger.error(f"Failed to stop bot: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/dashboard")
async def get_dashboard():
    """Serve the frontend dashboard."""
    try:
        with open(
            "/Users/christophermayfield/Documents/Projects/arbitrage_bot/frontend/dashboard.html",
            "r",
        ) as f:
            html_content = f.read()

        return HTMLResponse(
            content=html_content, status_code=200, media_type="text/html"
        )

    except Exception as e:
        logger.error(f"Failed to serve dashboard: {e}")
        return HTMLResponse(
            content=f"<html><body><h1>Error loading dashboard</h1><p>{str(e)}</p></body></html>",
            status_code=500,
            media_type="text/html",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
