import asyncio
import json
import base64
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Set, TYPE_CHECKING
from datetime import datetime

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from src.utils.logging_utils import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger("websocket")


@dataclass
class WebSocketMessage:
    type: str
    data: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class WebSocketClient:
    def __init__(
        self,
        uri: str,
        reconnect_delay: float = 5.0,
        heartbeat_interval: float = 30.0,
        subscriptions: Optional[Set[str]] = None
    ):
        self.uri = uri
        self.reconnect_delay = reconnect_delay
        self.heartbeat_interval = heartbeat_interval
        self.subscriptions = subscriptions or set()
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._reconnect_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._handlers: Dict[str, Set[Callable]] = {}
        self._message_queue: asyncio.Queue = asyncio.Queue()

    async def connect(self) -> None:
        if self._websocket:
            await self._websocket.close()

        try:
            self._websocket = await websockets.connect(
                self.uri,
                ping_interval=self.heartbeat_interval,
                ping_timeout=10.0
            )
            self._running = True
            logger.info(f"Connected to {self.uri}")

            for subscription in self.subscriptions:
                await self.subscribe(subscription)

            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            asyncio.create_task(self._message_loop())

        except WebSocketException as e:
            logger.error(f"Failed to connect: {e}")
            raise

    async def disconnect(self) -> None:
        self._running = False

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        if self._reconnect_task:
            self._reconnect_task.cancel()

        if self._websocket:
            await self._websocket.close()
            self._websocket = None
            logger.info("Disconnected")

    async def _message_loop(self) -> None:
        while self._running and self._websocket:
            try:
                message = await self._websocket.recv()
                await self._handle_message(message)
            except ConnectionClosed:
                if self._running:
                    logger.warning("Connection closed, attempting to reconnect...")
                    await self._reconnect()
            except WebSocketException as e:
                logger.error(f"WebSocket error: {e}")
                if self._running:
                    await self._reconnect()
            except Exception as e:
                logger.error(f"Unexpected error in message loop: {e}")
                if self._running:
                    await self._reconnect()

    async def _handle_message(self, raw_message: str) -> None:
        try:
            message = json.loads(raw_message)
            msg_type = message.get('type', 'unknown')
            data = message.get('data', {})

            ws_message = WebSocketMessage(type=msg_type, data=data)

            await self._message_queue.put(ws_message)

            if msg_type in self._handlers:
                for handler in self._handlers[msg_type]:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(ws_message)
                        else:
                            handler(ws_message)
                    except Exception as e:
                        logger.error(f"Handler error for {msg_type}: {e}")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message: {e}")

    async def _reconnect(self) -> None:
        if self._reconnect_task and not self._reconnect_task.done():
            return

        self._reconnect_task = asyncio.create_task(self._reconnect_loop())

    async def _reconnect_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.reconnect_delay)
            try:
                await self.connect()
                logger.info("Reconnected successfully")
                break
            except Exception as e:
                logger.error(f"Reconnection failed: {e}")

    async def _heartbeat_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.heartbeat_interval)
            try:
                if self._websocket and self._websocket.open:
                    await self._websocket.ping()
            except Exception as e:
                logger.warning(f"Heartbeat failed: {e}")

    async def subscribe(self, channel: str) -> bool:
        if channel not in self.subscriptions:
            self.subscriptions.add(channel)

        if self._websocket and self._websocket.open:
            await self._websocket.send(json.dumps({
                'type': 'subscribe',
                'channel': channel
            }))
            logger.info(f"Subscribed to {channel}")
            return True
        return False

    async def unsubscribe(self, channel: str) -> bool:
        if channel in self.subscriptions:
            self.subscriptions.discard(channel)

        if self._websocket and self._websocket.open:
            await self._websocket.send(json.dumps({
                'type': 'unsubscribe',
                'channel': channel
            }))
            logger.info(f"Unsubscribed from {channel}")
            return True
        return False

    def on(self, message_type: str) -> Callable:
        def decorator(func: Callable) -> Callable:
            if message_type not in self._handlers:
                self._handlers[message_type] = set()
            self._handlers[message_type].add(func)
            return func
        return decorator

    def off(self, message_type: str, handler: Callable) -> None:
        if message_type in self._handlers:
            self._handlers[message_type].discard(handler)

    async def get_message(self, timeout: float = 1.0) -> Optional[WebSocketMessage]:
        try:
            return await asyncio.wait_for(
                self._message_queue.get(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return None

    @property
    def is_connected(self) -> bool:
        return self._websocket is not None and self._websocket.open


class KalshiWebSocketClient(WebSocketClient):
    MARKET_DATA_URI = "wss://api.kalshi.com/trade-api/ws/v2/market"
    USER_DATA_URI = "wss://api.kalshi.com/trade-api/ws/v2/user"

    def __init__(
        self,
        base_url: str = "https://api.kalshi.com",
        demo: bool = True,
        **kwargs
    ):
        uri = f"{base_url}/trade-api/ws/v2/market"
        if demo:
            uri = f"{base_url.replace('api', 'demo-api')}/trade-api/ws/v2/market"

        super().__init__(uri=uri, **kwargs)
        self.base_url = base_url
        self.demo = demo
        self._api_key_id: Optional[str] = None
        self._private_key: Optional[str] = None

    def set_credentials(self, api_key_id: str, private_key: str) -> None:
        self._api_key_id = api_key_id
        self._private_key = private_key

    async def authenticate(self) -> None:
        if not self._api_key_id or not self._private_key:
            raise ValueError("API credentials not set")

        auth_message = {
            'type': 'auth',
            'api_key_id': self._api_key_id,
            'timestamp': str(int(datetime.utcnow().timestamp() * 1000))
        }

        message_str = auth_message['timestamp']
        private_key = self._private_key

        try:
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import padding
            from cryptography.hazmat.backends import default_backend

            private_key_obj = serialization.load_pem_private_key(
                private_key.encode(),
                password=None,
                backend=default_backend()
            )

            if not hasattr(private_key_obj, 'sign'):
                raise ValueError("Private key does not support signing")

            from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
            if isinstance(private_key_obj, RSAPrivateKey):
                signature = private_key_obj.sign(
                    message_str.encode(),
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
                auth_message['signature'] = base64.b64encode(signature).decode()
            else:
                raise ValueError("Unsupported private key type for signing")

        except Exception as e:
            logger.error(f"Failed to sign auth message: {e}")
            raise

        if self._websocket and self._websocket.open:
            await self._websocket.send(json.dumps(auth_message))
            logger.info("Authentication sent")

    async def subscribe_orderbook(self, market_id: str) -> bool:
        return await self.subscribe(f"orderbook:{market_id}")

    async def subscribe_trades(self, market_id: Optional[str] = None) -> bool:
        channel = f"trades:{market_id}" if market_id else "trades:all"
        return await self.subscribe(channel)

    async def subscribe_ticker(self, market_id: str) -> bool:
        return await self.subscribe(f"ticker:{market_id}")
