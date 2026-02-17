from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class OrderBookLevel:
    price: int
    count: int
    total: int


@dataclass
class OrderBook:
    market_id: str
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)
    timestamp: Optional[str] = None
    sequence: Optional[int] = None
    last_update: Optional[datetime] = None

    def update_bid(self, price: int, size: int) -> None:
        for level in self.bids:
            if level.price == price:
                if size == 0:
                    self.bids.remove(level)
                else:
                    level.count = size
                return
        if size > 0:
            self.bids.append(OrderBookLevel(price=price, count=size, total=size))
            self.bids.sort(key=lambda x: x.price, reverse=True)

    def update_ask(self, price: int, size: int) -> None:
        for level in self.asks:
            if level.price == price:
                if size == 0:
                    self.asks.remove(level)
                else:
                    level.count = size
                return
        if size > 0:
            self.asks.append(OrderBookLevel(price=price, count=size, total=size))
            self.asks.sort(key=lambda x: x.price)

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "OrderBook":
        bids = [
            OrderBookLevel(
                price=int(b["price"]), count=int(b["count"]), total=int(b["total"])
            )
            for b in data.get("bids", [])
        ]
        asks = [
            OrderBookLevel(
                price=int(a["price"]), count=int(a["count"]), total=int(a["total"])
            )
            for a in data.get("asks", [])
        ]
        return cls(
            market_id=data.get("market_id", ""),
            bids=bids,
            asks=asks,
            timestamp=data.get("ts"),
            sequence=data.get("seq"),
        )

    def get_best_bid(self) -> Optional[OrderBookLevel]:
        return self.bids[0] if self.bids else None

    def get_best_ask(self) -> Optional[OrderBookLevel]:
        return self.asks[0] if self.asks else None

    def get_mid_price(self) -> Optional[float]:
        return (
            (self.bids[0].price + self.asks[0].price) / 2.0
            if self.bids and self.asks
            else None
        )

    @property
    def mid_price(self) -> Optional[float]:
        """Property alias for get_mid_price()"""
        return self.get_mid_price()

    def get_spread(self) -> Optional[int]:
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid and best_ask:
            return best_ask.price - best_bid.price
        return None

    def get_spread_percent(self) -> Optional[float]:
        spread = self.get_spread()
        mid = self.get_mid_price()
        if spread and mid:
            return (spread / mid) * 100
        return None

    def get_bid_depth(self, levels: int = 3) -> int:
        return sum(b.total for b in self.bids[:levels])

    def get_ask_depth(self, levels: int = 3) -> int:
        return sum(a.total for a in self.asks[:levels])

    def get_liquidity_score(self, min_price: int = 1, max_price: int = 99) -> float:
        bid_liquidity = sum(
            b.total for b in self.bids if min_price <= b.price <= max_price
        )
        ask_liquidity = sum(
            a.total for a in self.asks if min_price <= a.price <= max_price
        )
        total_liquidity = bid_liquidity + ask_liquidity
        if total_liquidity == 0:
            return 0.0
        balance = abs(bid_liquidity - ask_liquidity)
        imbalance_penalty = balance / total_liquidity
        return max(0.0, (1.0 - imbalance_penalty) * 100)

    def estimate_slippage(self, side: OrderSide, quantity: int) -> Tuple[int, float]:
        if side == OrderSide.BUY:
            levels = self.asks
        else:
            levels = self.bids

        total_cost: float = 0
        remaining = quantity
        avg_price = 0

        for level in levels:
            if remaining <= 0:
                break
            take = min(remaining, level.count)
            total_cost += take * level.price
            remaining -= take

        if quantity > 0:
            avg_price = total_cost / quantity
            weighted_price = avg_price
            if side == OrderSide.BUY:
                reference = self.asks[0].price if self.asks else weighted_price
            else:
                reference = self.bids[0].price if self.bids else weighted_price

            if reference > 0:
                slippage = abs(weighted_price - reference) / reference * 100
            else:
                slippage = 0.0
            return (total_cost, slippage)

        return (0, 0.0)

    def get_fill_probability(self, side: OrderSide, quantity: int, price: int) -> float:
        if side == OrderSide.BUY:
            if not self.asks or self.asks[0].price > price:
                return 0.0
            available = sum(a.count for a in self.asks if a.price <= price)
        else:
            if not self.bids or self.bids[0].price < price:
                return 0.0
            available = sum(b.count for b in self.bids if b.price >= price)

        if quantity <= available:
            return 1.0
        if available == 0:
            return 0.0
        return min(1.0, available / quantity)

    def update_from_ws_message(self, data: Dict[str, Any]) -> None:
        """Update orderbook with WebSocket message data"""
        try:
            # Update sequence and timestamp
            self.sequence = data.get("seq")
            self.timestamp = data.get("ts")

            # Update bids
            if "bids" in data:
                new_bids = []
                for bid_data in data["bids"]:
                    if isinstance(bid_data, list) and len(bid_data) >= 3:
                        # Format: [price, count, total]
                        new_bids.append(
                            OrderBookLevel(
                                price=int(bid_data[0]),
                                count=int(bid_data[1]),
                                total=int(bid_data[2]),
                            )
                        )
                    elif isinstance(bid_data, dict):
                        # Format: {price: X, count: Y, total: Z}
                        new_bids.append(
                            OrderBookLevel(
                                price=int(bid_data["price"]),
                                count=int(bid_data["count"]),
                                total=int(bid_data["total"]),
                            )
                        )
                self.bids = new_bids

            # Update asks
            if "asks" in data:
                new_asks = []
                for ask_data in data["asks"]:
                    if isinstance(ask_data, list) and len(ask_data) >= 3:
                        # Format: [price, count, total]
                        new_asks.append(
                            OrderBookLevel(
                                price=int(ask_data[0]),
                                count=int(ask_data[1]),
                                total=int(ask_data[2]),
                            )
                        )
                    elif isinstance(ask_data, dict):
                        # Format: {price: X, count: Y, total: Z}
                        new_asks.append(
                            OrderBookLevel(
                                price=int(ask_data["price"]),
                                count=int(ask_data["count"]),
                                total=int(ask_data["total"]),
                            )
                        )
                self.asks = new_asks

        except Exception as e:
            # If update fails, keep existing data
            pass

    def is_healthy(
        self, max_spread_percent: float = 5.0, min_liquidity: float = 50.0
    ) -> tuple[bool, str]:
        """Validate orderbook health for trading (alias for validate_health)"""
        return self.validate_health(max_spread_percent, min_liquidity)

    def validate_health(
        self, max_spread_percent: float = 5.0, min_liquidity: float = 50.0
    ) -> tuple[bool, str]:
        """Validate orderbook health for trading"""
        if not self.bids or not self.asks:
            return False, "Empty orderbook"

        spread_pct = self.get_spread_percent()
        if spread_pct is None:
            return False, "Could not calculate spread"

        if spread_pct > max_spread_percent:
            return False, f"Spread too wide: {spread_pct:.2f}%"

        liquidity = self.get_liquidity_score()
        if liquidity < min_liquidity:
            return False, f"Low liquidity: {liquidity:.2f}"

        return True, "Healthy"
