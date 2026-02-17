import asyncio
import os
from typing import Dict, Set, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

from src.clients.websocket_client import KalshiWebSocketClient, WebSocketMessage
from src.clients.kalshi_client import KalshiClient
from src.core.orderbook import OrderBook
from src.utils.logging_utils import get_logger
from src.utils.config import Config

logger = get_logger("realtime_market_data")


@dataclass
class MarketUpdate:
    market_id: str
    timestamp: str
    orderbook_updates: Optional[Dict[str, Any]] = None
    trade_updates: Optional[Dict[str, Any]] = None
    ticker_updates: Optional[Dict[str, Any]] = None


class RealTimeMarketDataManager:
    """
    Manages real-time market data using WebSocket connections for low-latency
    arbitrage detection.
    """

    def __init__(self, config: Config, client: KalshiClient):
        self.config = config
        self.client = client
        self.ws_client: Optional[KalshiWebSocketClient] = None
        self.orderbooks: Dict[str, OrderBook] = {}
        self.subscribed_markets: Set[str] = set()
        self.update_callbacks: Set[Callable[[MarketUpdate], None]] = set()
        self.error_callbacks: Set[Callable[[Exception], None]] = set()
        self.running = False
        self.reconnect_task: Optional[asyncio.Task] = None

        # WebSocket configuration
        self.ws_enabled = config.get("websocket.enabled", False)
        self.reconnect_delay = config.get("websocket.reconnect_delay_seconds", 5)
        self.heartbeat_interval = config.get("websocket.heartbeat_interval_seconds", 30)

    async def start(self) -> None:
        """Start the real-time market data manager"""
        if not self.ws_enabled:
            logger.info("WebSocket disabled, using polling fallback")
            return

        logger.info("Starting real-time market data manager")
        self.running = True

        # Initialize WebSocket client with credentials
        kalshi_config = self.config.kalshi
        api_key_id = kalshi_config.get("api_key_id")
        private_key_path = kalshi_config.get("private_key_path")

        if not api_key_id or not private_key_path:
            raise ValueError("API credentials not configured for WebSocket")

        # Read private key
        try:
            with open(private_key_path.replace("~", os.path.expanduser("~")), "r") as f:
                private_key = f.read()
        except Exception as e:
            logger.error(f"Failed to read private key: {e}")
            raise

        # Create WebSocket client
        self.ws_client = KalshiWebSocketClient(
            base_url=kalshi_config.get("base_url", "https://api.kalshi.com"),
            demo=kalshi_config.get("demo_mode", True),
            reconnect_delay=self.reconnect_delay,
            heartbeat_interval=self.heartbeat_interval,
        )

        self.ws_client.set_credentials(api_key_id, private_key)

        # Register message handlers
        self.ws_client.on("orderbook")(self._handle_orderbook_update)
        self.ws_client.on("trades")(self._handle_trade_update)
        self.ws_client.on("ticker")(self._handle_ticker_update)
        self.ws_client.on("error")(self._handle_error)

        # Start WebSocket connection
        await self._connect_websocket()

    async def stop(self) -> None:
        """Stop the real-time market data manager"""
        logger.info("Stopping real-time market data manager")
        self.running = False

        if self.reconnect_task:
            self.reconnect_task.cancel()

        if self.ws_client:
            await self.ws_client.disconnect()

    async def subscribe_to_market(self, market_id: str) -> None:
        """Subscribe to real-time updates for a specific market"""
        if market_id in self.subscribed_markets:
            return

        if self.ws_client and self.ws_client.is_connected:
            try:
                # Subscribe to orderbook and trades for this market
                await self.ws_client.subscribe_orderbook(market_id)
                await self.ws_client.subscribe_trades(market_id)
                await self.ws_client.subscribe_ticker(market_id)
                self.subscribed_markets.add(market_id)
                logger.info(f"Subscribed to real-time data for market {market_id}")
            except Exception as e:
                logger.error(f"Failed to subscribe to market {market_id}: {e}")
                raise
        else:
            # Fall back to REST API for initial orderbook
            await self._fetch_initial_orderbook(market_id)
            self.subscribed_markets.add(market_id)

    async def unsubscribe_from_market(self, market_id: str) -> None:
        """Unsubscribe from real-time updates for a specific market"""
        if market_id not in self.subscribed_markets:
            return

        if self.ws_client and self.ws_client.is_connected:
            try:
                await self.ws_client.unsubscribe(f"orderbook:{market_id}")
                await self.ws_client.unsubscribe(f"trades:{market_id}")
                await self.ws_client.unsubscribe(f"ticker:{market_id}")
            except Exception as e:
                logger.error(f"Failed to unsubscribe from market {market_id}: {e}")

        self.subscribed_markets.discard(market_id)
        self.orderbooks.pop(market_id, None)
        logger.info(f"Unsubscribed from market {market_id}")

    def add_update_callback(self, callback: Callable[[MarketUpdate], None]) -> None:
        """Add a callback for market updates"""
        self.update_callbacks.add(callback)

    def remove_update_callback(self, callback: Callable[[MarketUpdate], None]) -> None:
        """Remove a market update callback"""
        self.update_callbacks.discard(callback)

    def add_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Add a callback for errors"""
        self.error_callbacks.add(callback)

    def get_orderbook(self, market_id: str) -> Optional[OrderBook]:
        """Get current orderbook for a market"""
        return self.orderbooks.get(market_id)

    def get_all_orderbooks(self) -> Dict[str, OrderBook]:
        """Get all current orderbooks"""
        return self.orderbooks.copy()

    async def _connect_websocket(self) -> None:
        """Connect to WebSocket and authenticate"""
        while self.running:
            try:
                await self.ws_client.connect()
                await self.ws_client.authenticate()
                logger.info("WebSocket connected and authenticated")

                # Re-subscribe to existing markets
                for market_id in self.subscribed_markets:
                    await self.ws_client.subscribe_orderbook(market_id)
                    await self.ws_client.subscribe_trades(market_id)
                    await self.ws_client.subscribe_ticker(market_id)

                # Start message processing
                asyncio.create_task(self._message_loop())
                break

            except Exception as e:
                logger.error(f"WebSocket connection failed: {e}")
                if self.running:
                    await asyncio.sleep(self.reconnect_delay)

    async def _message_loop(self) -> None:
        """Process WebSocket messages"""
        while self.running and self.ws_client and self.ws_client.is_connected:
            try:
                message = await self.ws_client.get_message(timeout=1.0)
                if message:
                    # Process message is handled by registered handlers
                    pass
            except Exception as e:
                logger.debug(f"Message loop error: {e}")

    async def _handle_orderbook_update(self, message: WebSocketMessage) -> None:
        """Handle orderbook update messages"""
        try:
            data = message.data
            market_id = data.get("market_id")
            if not market_id:
                return

            # Update or create orderbook
            orderbook = self.orderbooks.get(market_id)
            if not orderbook:
                orderbook = OrderBook(market_id=market_id)
                self.orderbooks[market_id] = orderbook

            # Update orderbook with new data
            orderbook.update_from_ws_message(data)

            # Create market update and notify callbacks
            update = MarketUpdate(
                market_id=market_id, timestamp=message.timestamp, orderbook_updates=data
            )

            for callback in self.update_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(update)
                    else:
                        callback(update)
                except Exception as e:
                    logger.error(f"Update callback error: {e}")

        except Exception as e:
            logger.error(f"Orderbook update error: {e}")

    async def _handle_trade_update(self, message: WebSocketMessage) -> None:
        """Handle trade update messages"""
        try:
            data = message.data
            market_id = data.get("market_id")
            if not market_id:
                return

            update = MarketUpdate(
                market_id=market_id, timestamp=message.timestamp, trade_updates=data
            )

            for callback in self.update_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(update)
                    else:
                        callback(update)
                except Exception as e:
                    logger.error(f"Trade update callback error: {e}")

        except Exception as e:
            logger.error(f"Trade update error: {e}")

    async def _handle_ticker_update(self, message: WebSocketMessage) -> None:
        """Handle ticker update messages"""
        try:
            data = message.data
            market_id = data.get("market_id")
            if not market_id:
                return

            update = MarketUpdate(
                market_id=market_id, timestamp=message.timestamp, ticker_updates=data
            )

            for callback in self.update_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(update)
                    else:
                        callback(update)
                except Exception as e:
                    logger.error(f"Ticker update callback error: {e}")

        except Exception as e:
            logger.error(f"Ticker update error: {e}")

    async def _handle_error(self, message: WebSocketMessage) -> None:
        """Handle WebSocket error messages"""
        error_data = message.data
        error_msg = error_data.get("message", "Unknown WebSocket error")
        error = Exception(f"WebSocket error: {error_msg}")

        logger.error(f"WebSocket error: {error_msg}")

        for callback in self.error_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(error)
                else:
                    callback(error)
            except Exception as e:
                logger.error(f"Error callback error: {e}")

    async def _fetch_initial_orderbook(self, market_id: str) -> None:
        """Fetch initial orderbook via REST API as fallback"""
        try:
            orderbook_data = self.client.get_market_orderbook(market_id)
            orderbook = OrderBook.from_api_response(orderbook_data)
            self.orderbooks[market_id] = orderbook
            logger.debug(f"Fetched initial orderbook for {market_id} via REST API")
        except Exception as e:
            logger.error(f"Failed to fetch initial orderbook for {market_id}: {e}")


import os
