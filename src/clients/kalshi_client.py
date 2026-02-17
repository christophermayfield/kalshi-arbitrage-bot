import time
import base64
import json
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from src.utils.config import Config
from src.utils.logging_utils import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger("kalshi_client")


class KalshiClient:
    BASE_URL = "https://api.kalshi.com"
    DEMO_URL = "https://demo-api.kalshi.co"

    def __init__(self, config: Config):
        self.config = config
        self.api_key_id = config.get("kalshi.api_key_id", "")
        self.private_key_path = config.get("kalshi.private_key_path", "")
        self.base_url = config.get("kalshi.base_url", self.DEMO_URL)
        self.demo_mode = config.get("kalshi.demo_mode", True)

        self.session = requests.Session()
        self.session.headers.update(
            {"Content-Type": "application/json", "Accept": "application/json"}
        )

        self._private_key: Optional[str] = None
        self._load_private_key()

    def _load_private_key(self) -> None:
        if self.private_key_path:
            path = Path(self.private_key_path).expanduser()
            if path.exists():
                with open(path, "r") as f:
                    self._private_key = f.read()
            else:
                logger.warning(f"Private key file not found: {path}")

    def _sign_message(self, message: str) -> str:
        if not self._private_key:
            return ""

        try:
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import padding
            from cryptography.hazmat.backends import default_backend

            private_key = serialization.load_pem_private_key(
                self._private_key.encode(), password=None, backend=default_backend()
            )

            if not hasattr(private_key, "sign"):
                logger.error("Private key does not support signing")
                return ""

            from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

            if isinstance(private_key, RSAPrivateKey):
                signature = private_key.sign(
                    message.encode(),
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH,
                    ),
                    hashes.SHA256(),
                )
                return base64.b64encode(signature).decode()

            logger.error("Unsupported private key type for signing")
            return ""
        except Exception as e:
            logger.error(f"Failed to sign message: {e}")
            return ""

    def _authenticate(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        timestamp = str(int(time.time() * 1000))
        # Signature: timestamp + HTTP method + request path (without query params)
        signature_input = f"{timestamp}{method.upper()}{path}"
        signature = self._sign_message(signature_input)

        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": signature,
        }

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        body_str = json.dumps(body) if body else ""

        auth_headers = self._authenticate(method, path, body_str)
        headers = {**self.session.headers, **auth_headers}

        try:
            response = self.session.request(
                method=method, url=url, headers=headers, params=params, data=body_str
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise

    def get_exchange_status(self) -> Dict[str, Any]:
        return self._request("GET", "/trade-api/v2/exchange/status")

    def get_markets(
        self,
        series_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        params = {"limit": limit}
        if series_id:
            params["series_id"] = series_id
        if status:
            params["status"] = status
        return self._request("GET", "/trade-api/v2/markets", params=params)

    def get_market(self, market_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/trade-api/v2/markets/{market_id}")

    def get_market_orderbook(self, market_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/trade-api/v2/markets/{market_id}/orderbook")

    def get_market_candlesticks(
        self, market_id: str, interval: str = "1m", limit: int = 100
    ) -> Dict[str, Any]:
        params = {"interval": interval, "limit": limit}
        return self._request(
            "GET", f"/trade-api/v2/markets/{market_id}/candlesticks", params=params
        )

    def get_trades(
        self, market_id: Optional[str] = None, limit: int = 100
    ) -> Dict[str, Any]:
        params = {"limit": limit}
        if market_id:
            params["market_id"] = market_id
        return self._request("GET", "/trade-api/v2/trades", params=params)

    def get_events(
        self,
        series_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        params = {"limit": limit}
        if series_id:
            params["series_id"] = series_id
        if status:
            params["status"] = status
        return self._request("GET", "/trade-api/v2/events", params=params)

    def get_series(self, series_id: Optional[str] = None) -> Dict[str, Any]:
        path = (
            f"/trade-api/v2/series/{series_id}" if series_id else "/trade-api/v2/series"
        )
        return self._request("GET", path)

    def create_order(
        self, market_id: str, side: str, order_type: str, price: int, count: int
    ) -> Dict[str, Any]:
        body = {
            "market_id": market_id,
            "side": side,
            "order_type": order_type,
            "price": price,
            "count": count,
        }
        return self._request("POST", "/trade-api/v2/orders", body=body)

    def batch_create_orders(self, orders: List[Dict]) -> Dict[str, Any]:
        return self._request(
            "POST", "/trade-api/v2/orders/batch", body={"orders": orders}
        )

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        return self._request("DELETE", f"/trade-api/v2/orders/{order_id}")

    def get_orders(
        self,
        market_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        params = {"limit": limit}
        if market_id:
            params["market_id"] = market_id
        if status:
            params["status"] = status
        return self._request("GET", "/trade-api/v2/orders", params=params)

    def get_order(self, order_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/trade-api/v2/orders/{order_id}")

    def get_positions(self) -> Dict[str, Any]:
        return self._request("GET", "/trade-api/v2/portfolio/positions")

    def get_balance(self) -> Dict[str, Any]:
        return self._request("GET", "/trade-api/v2/portfolio/balance")

    def get_fills(
        self, market_id: Optional[str] = None, limit: int = 100
    ) -> Dict[str, Any]:
        params = {"limit": limit}
        if market_id:
            params["market_id"] = market_id
        return self._request("GET", "/trade-api/v2/portfolio/fills", params=params)
