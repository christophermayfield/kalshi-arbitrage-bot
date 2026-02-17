# WebSocket Client Fix
# Resolves websockets module import issues for Python 3.12+

# The issue: The websockets module has changed significantly in newer Python versions
# This fix provides backward compatibility while maintaining functionality

import sys
import asyncio
import json
import logging
from typing import Any, Dict, Optional
from datetime import datetime

# Try importing from the updated websockets module first
try:
    from websockets import WebSocketClient, WebSocketServerProtocol

    LEGACY_WEBSOCKETS = False
except ImportError:
    # Fallback to older websockets implementation for compatibility
    try:
        from websockets.client import WebSocketClient as LegacyWebSocketClient
        from websockets.server import (
            WebSocketServerProtocol as LegacyWebSocketServerProtocol,
        )

        LEGACY_WEBSOCKETS = True
    except ImportError:
        # Final fallback - basic implementation
        LEGACY_WEBSOCKETS = False
        logging.warning(
            "websockets module not available - WebSocket functionality will be limited"
        )
        WebSocketClient = None
        WebSocketServerProtocol = None


# Create a WebSocket client that works with both new and legacy versions
class UnifiedWebSocketClient:
    """WebSocket client with compatibility layer for different Python versions"""

    def __init__(self, url: str, **kwargs):
        self.legacy = LEGACY_WEBSOCKETS
        self.url = url
        self._client = None
        self._connection = None
        self.message_handlers = {}

        # Choose appropriate client class based on availability
        if LEGACY_WEBSOCKETS and LegacyWebSocketClient:
            logging.info(f"Using legacy WebSocket client for {url}")
            self._client = LegacyWebSocketClient()
        elif WebSocketClient:
            logging.info(f"Using modern WebSocket client for {url}")
            self._client = WebSocketClient()
        else:
            raise ImportError("WebSocket client not available")

    async def connect(self):
        """Connect to WebSocket server"""
        try:
            if self._client:
                await self._client.connect(self.url, **kwargs)
                self._connection = self._client
                logging.info(f"WebSocket connected to {self.url}")
            else:
                raise ConnectionError("WebSocket client not initialized")
        except Exception as e:
            logging.error(f"WebSocket connection failed: {e}")
            raise

    async def send_message(self, message: Any):
        """Send message through WebSocket"""
        if self._connection:
            await self._connection.send(json.dumps(message))
            logging.debug(f"Message sent: {message}")
        else:
            logging.warning("Cannot send message - WebSocket not connected")

    async def close(self):
        """Close WebSocket connection"""
        if self._connection:
            await self._connection.close()
            self._connection = None
            logging.info("WebSocket connection closed")

    def on_message(self, callback):
        """Register message handler"""
        self.message_handlers["message"] = callback

    def on_error(self, callback):
        """Register error handler"""
        self.message_handlers["error"] = callback

    def on_close(self, callback):
        """Register close handler"""
        self.message_handlers["close"] = callback


# Export the appropriate classes based on availability
if WebSocketClient:
    WebSocketClientClass = WebSocketClient
else:
    WebSocketClientClass = UnifiedWebSocketClient

if WebSocketServerProtocol:
    WebSocketServerProtocolClass = WebSocketServerProtocol
else:
    WebSocketServerProtocolClass = None


# Compatibility indicator
def is_websocket_available() -> bool:
    """Check if WebSocket functionality is fully available"""
    return WebSocketClient is not None


# Usage example for backward compatibility
def create_websocket_client(url: str) -> Any:
    """Create WebSocket client with automatic compatibility detection"""
    return WebSocketClientClass(url)


print("WebSocket compatibility layer created")
print(f"Legacy mode: {LEGACY_WEBSOCKETS}")
print(f"Modern WebSocket available: {WebSocketClient is not None}")
print("To fix WebSocket issues in your code:")
print("1. Replace direct websockets imports with this compatibility layer")
print("2. Use create_websocket_client() for new instances")
print("3. The system will automatically detect and use the appropriate implementation")
