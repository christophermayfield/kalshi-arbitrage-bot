"""WebSocket handler for real-time data streaming to frontend."""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketDisconnect

logger = logging.getLogger(__name__)


class WebSocketManager:
    def __init__(self):
        """Initialize WebSocket manager."""
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscribers: Dict[str, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        """Handle new WebSocket connection."""
        logger.info(f"WebSocket client connected: {client_id}")
        self.active_connections[client_id] = websocket

        # Send current bot status
        await self.send_to_client(
            client_id,
            {
                "type": "bot_status",
                "status": {
                    "running": True,
                    "connected_at": datetime.utcnow().isoformat(),
                },
            },
        )

        # Send current opportunities
        opportunities = self.get_recent_opportunities()
        await self.send_to_client(
            client_id, {"type": "opportunities", "opportunities": opportunities}
        )

        # Send strategy performance stats
        stats = self.get_strategy_stats()
        await self.send_to_client(client_id, {"type": "strategy_stats", "stats": stats})

        # Send P&L data
        pnl_data = self.get_pnl_data()
        await self.send_to_client(
            client_id, {"type": "pnl_update", "pnl_data": pnl_data}
        )

        # Add to subscribers
        self.subscribers[client_id] = {
            "websocket": websocket,
            "connected_at": datetime.utcnow().isoformat(),
        }

        logger.info(f"Total active connections: {len(self.active_connections)}")

    async def disconnect(self, client_id: str):
        """Handle WebSocket disconnection."""
        logger.info(f"WebSocket client disconnected: {client_id}")

        if client_id in self.active_connections:
            del self.active_connections[client_id]

        if client_id in self.subscribers:
            del self.subscribers[client_id]

        logger.info(f"Total active connections: {len(self.active_connections)}")

    async def broadcast_to_all(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        if not self.active_connections:
            return

        message_with_timestamp = {**message, "timestamp": datetime.utcnow().isoformat()}

        # Send to all active connections
        for client_id, websocket in list(self.active_connections.items()):
            try:
                await websocket.send_text(json.dumps(message_with_timestamp))
            except Exception as e:
                logger.error(f"Failed to send to client {client_id}: {e}")
                # Clean up failed connection
                if client_id in self.active_connections:
                    await self.disconnect(client_id)

    async def send_to_client(self, client_id: str, message: Dict[str, Any]):
        """Send message to specific client."""
        if client_id not in self.active_connections:
            return

        message_with_timestamp = {**message, "timestamp": datetime.utcnow().isoformat()}

        websocket = self.active_connections[client_id]["websocket"]
        try:
            await websocket.send_text(json.dumps(message_with_timestamp))
        except Exception as e:
            logger.error(f"Failed to send to client {client_id}: {e}")
            await self.disconnect(client_id)

    def get_recent_opportunities(self) -> List[Dict[str, Any]]:
        """Get recent arbitrage opportunities."""
        try:
            from src.core.arbitrage import ArbitrageDetector
            from src.utils.config import Config

            config = Config()
            detector = ArbitrageDetector(
                min_profit_cents=config.get("trading.min_profit_cents", 10),
                enable_statistical_arbitrage=config.get("statistical.enabled", False),
                statistical_config=config.get("statistical", {}),
            )

            # Get opportunities (this would integrate with your actual detector)
            opportunities = []

            # For demo, return mock opportunities
            for i in range(5):
                opp = {
                    "id": f"demo_opp_{i}",
                    "type": "mean_reversion" if i % 2 == 0 else "pairs_trading",
                    "market_id_1": f"demo_market_{i}",
                    "expected_profit_cents": (i + 1) * 15,
                    "confidence": 0.8 + (i * 0.1),
                    "z_score": 1.5 + (i * 0.3),
                    "strategy_signal": f"signal_{i}",
                    "timestamp": datetime.utcnow().isoformat(),
                }
                opportunities.append(opp)

            return opportunities

        except Exception as e:
            logger.error(f"Failed to get opportunities: {e}")
            return []

    def get_strategy_stats(self) -> Dict[str, Any]:
        """Get strategy performance statistics."""
        try:
            from src.core.arbitrage import ArbitrageDetector
            from src.utils.config import Config

            config = Config()
            stats = {}

            # Mean Reversion Stats
            stats["mean_reversion"] = {
                "enabled": config.get("statistical.enabled", False),
                "opportunities_found": 12,
                "success_rate": 0.65,
                "avg_profit_cents": 8.5,
            }

            # Pairs Trading Stats
            stats["pairs_trading"] = {
                "enabled": config.get("statistical.enabled", False),
                "opportunities_found": 8,
                "success_rate": 0.72,
                "avg_profit_cents": 12.3,
            }

            return stats

        except Exception as e:
            logger.error(f"Failed to get strategy stats: {e}")
            return {}

    def get_pnl_data(self) -> List[Dict[str, Any]]:
        """Get P&L data for charting."""
        try:
            # Mock P&L data - replace with actual implementation
            import random

            pnl_data = []
            cumulative_pnl = 0

            for i in range(24):  # Last 24 hours
                pnl = random.uniform(-50, 150)  # Random profit/loss
                cumulative_pnl += pnl
                pnl_data.append(
                    {
                        "timestamp": f"{23 - i:02d}:00:00",
                        "pnl": pnl,
                        "cumulative_pnl": cumulative_pnl,
                    }
                )

            return pnl_data

        except Exception as e:
            logger.error(f"Failed to get P&L data: {e}")
            return []

    async def handle_message(self, websocket: WebSocket, client_id: str, message: str):
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message)

            logger.info(f"Received message from {client_id}: {data}")

            # Handle different message types
            if data.get("type") == "subscribe_opportunities":
                # Subscribe to opportunity updates
                self.subscribers[client_id]["opportunities"] = True
                logger.info(f"Client {client_id} subscribed to opportunities")

            elif data.get("type") == "unsubscribe_opportunities":
                # Unsubscribe from opportunity updates
                self.subscribers[client_id].pop("opportunities", None)
                logger.info(f"Client {client_id} unsubscribed from opportunities")

            elif data.get("type") == "get_opportunities":
                # Send current opportunities
                opportunities = self.get_recent_opportunities()
                await self.send_to_client(
                    client_id, {"type": "opportunities", "opportunities": opportunities}
                )

            # Add more message handlers as needed
            else:
                logger.warning(
                    f"Unknown message type from {client_id}: {data.get('type')}"
                )

        except Exception as e:
            logger.error(f"Error handling message from {client_id}: {e}")


# Global WebSocket manager
ws_manager = WebSocketManager()


async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint."""
    # Extract client ID from query parameters or headers
    client_id = websocket.query_params.get(
        "client_id", f"client_{datetime.utcnow().timestamp()}"
    )

    await ws_manager.connect(websocket, client_id)

    try:
        while True:
            try:
                message = await websocket.receive_text()
                if message:
                    await ws_manager.handle_message(websocket, client_id, message)
            except WebSocketDisconnect:
                logger.info(f"Client {client_id} disconnected normally")
                await ws_manager.disconnect(client_id)
                break
            except Exception as e:
                logger.error(f"Error handling WebSocket for {client_id}: {e}")
                await ws_manager.disconnect(client_id)

    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")


async def broadcast_opportunity_update(opportunity: Dict[str, Any]):
    """Broadcast new opportunity to all subscribers."""
    message = {"type": "opportunity", "opportunity": opportunity}

    await ws_manager.broadcast_to_all(message)


async def broadcast_bot_status(status: Dict[str, Any]):
    """Broadcast bot status update."""
    message = {"type": "bot_status", "status": status}

    await ws_manager.broadcast_to_all(message)


async def broadcast_metrics_update():
    """Broadcast metrics update to all subscribers."""
    stats = ws_manager.get_strategy_stats()
    message = {"type": "strategy_stats", "stats": stats}

    await ws_manager.broadcast_to_all(message)


async def broadcast_pnl_update():
    """Broadcast P&L update to all subscribers."""
    pnl_data = ws_manager.get_pnl_data()
    message = {"type": "pnl_update", "pnl_data": pnl_data}

    await ws_manager.broadcast_to_all(message)


def get_websocket_manager() -> WebSocketManager:
    """Get the global WebSocket manager instance."""
    return ws_manager
