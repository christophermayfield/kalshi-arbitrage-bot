from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum

from src.utils.logging_utils import get_logger

logger = get_logger("portfolio")


class PositionSide(Enum):
    YES = "yes"
    NO = "no"


@dataclass
class Position:
    market_id: str
    side: PositionSide
    quantity: int
    avg_cost: int
    current_price: int = 0
    unrealized_pnl: int = 0
    realized_pnl: int = 0

    @property
    def market_value(self) -> int:
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> int:
        return self.quantity * self.avg_cost

    @property
    def total_pnl(self) -> int:
        return self.unrealized_pnl + self.realized_pnl

    def update_price(self, price: int) -> None:
        self.current_price = price
        self.unrealized_pnl = (price - self.avg_cost) * self.quantity


@dataclass
class Trade:
    id: str
    market_id: str
    side: PositionSide
    quantity: int
    price: int
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    pnl: int = 0


@dataclass
class PortfolioStats:
    total_value: int = 0
    cash_balance: int = 0
    positions_value: int = 0
    total_pnl: int = 0
    daily_pnl: int = 0
    open_positions: int = 0
    completed_trades: int = 0
    win_rate: float = 0.0
    avg_win: int = 0
    avg_loss: int = 0


class PortfolioManager:
    def can_open_position(self, quantity: int, price: int) -> bool:
        """Check if we can open a position given current balance and limits"""
        # Calculate position cost
        position_cost = quantity * price
        
        # Check against available balance
        if position_cost > self.cash_balance:
            logger.warning(f"Insufficient balance: need ${position_cost}, have ${self.cash_balance}")
            return False
        
        # Check max position size
        if quantity > self.max_position_contracts:
            logger.warning(f"Position size exceeds maximum: {quantity} > {self.max_position_contracts}")
            return False
        
        # Check portfolio heat
        current_positions_value = sum(pos['quantity'] * pos['avg_price'] for pos in self.positions.values())
        portfolio_value = current_positions_value + position_cost + self.cash_balance
        
        if portfolio_value > self.max_portfolio_value:
            logger.warning(f"Position exceeds portfolio limit: portfolio value ${portfolio_value} > ${self.max_portfolio_value}")
            return False
        
        return True



    def __init__(
        self,
        max_daily_loss: int = 10000,
        max_open_positions: int = 50,
        circuit_breaker_threshold: int = 5
    ):
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.daily_stats: Dict[str, int] = {}
        self.max_daily_loss = max_daily_loss
        self.max_open_positions = max_open_positions
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_count = 0
        self.daily_loss_count = 0
        self.last_reset_date = datetime.utcnow().date()

    def get_balance(self) -> int:
        return self.cash_balance

    def set_balance(self, balance: int) -> None:
        self.cash_balance = balance

    def update_positions(self, positions_data: List[Dict]) -> None:
        for pos_data in positions_data:
            market_id = pos_data.get('market_id', '')
            side = PositionSide.YES if pos_data.get('side') == 'yes' else PositionSide.NO
            quantity = int(pos_data.get('count', 0))
            avg_cost = int(pos_data.get('avg_price', 0))

            if market_id in self.positions:
                existing = self.positions[market_id]
                if existing.side == side:
                    total_quantity = existing.quantity + quantity
                    if total_quantity > 0:
                        new_avg = (existing.avg_cost * existing.quantity + avg_cost * quantity) / total_quantity
                        existing.quantity = total_quantity
                        existing.avg_cost = int(new_avg)
                    else:
                        del self.positions[market_id]
                else:
                    if quantity <= existing.quantity:
                        existing.quantity -= quantity
                    else:
                        new_quantity = quantity - existing.quantity
                        self.positions[market_id] = Position(
                            market_id=market_id,
                            side=side,
                            quantity=new_quantity,
                            avg_cost=avg_cost
                        )
            elif quantity > 0:
                self.positions[market_id] = Position(
                    market_id=market_id,
                    side=side,
                    quantity=quantity,
                    avg_cost=avg_cost
                )

    def record_trade(
        self,
        market_id: str,
        side: PositionSide,
        quantity: int,
        price: int,
        pnl: int = 0
    ) -> Trade:
        trade = Trade(
            id=f"trade_{datetime.utcnow().timestamp()}",
            market_id=market_id,
            side=side,
            quantity=quantity,
            price=price,
            pnl=pnl
        )
        self.trades.append(trade)

        if pnl > 0:
            self.daily_stats['wins'] = self.daily_stats.get('wins', 0) + 1
        else:
            self.daily_stats['losses'] = self.daily_stats.get('losses', 0) + 1

        return trade

    def get_stats(self) -> PortfolioStats:
        positions_value = sum(p.market_value for p in self.positions.values())
        total_pnl = sum(p.total_pnl for p in self.positions.values())

        completed_trades = len([t for t in self.trades if t.pnl != 0])
        wins = self.daily_stats.get('wins', 0)
        losses = self.daily_stats.get('losses', 0)
        total_trades = wins + losses

        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]
        avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0

        return PortfolioStats(
            total_value=self.cash_balance + positions_value,
            cash_balance=self.cash_balance,
            positions_value=positions_value,
            total_pnl=total_pnl,
            daily_pnl=self.daily_stats.get('daily_pnl', 0),
            open_positions=len(self.positions),
            completed_trades=completed_trades,
            win_rate=win_rate,
            avg_win=int(avg_win),
            avg_loss=int(avg_loss)
        )

    def check_risk_limits(self) -> Tuple[bool, str]:
        stats = self.get_stats()

        if stats.cash_balance < 0:
            return False, "Negative cash balance"

        if len(self.positions) >= self.max_open_positions:
            return False, "Max open positions reached"

        daily_pnl = self.daily_stats.get('daily_pnl', 0)
        if daily_pnl < -self.max_daily_loss:
            self.daily_loss_count += 1
            if self.daily_loss_count >= self.circuit_breaker_threshold:
                return False, "Circuit breaker triggered - daily loss limit"
            return False, "Daily loss limit exceeded"

        return True, "OK"

    def update_prices(self, prices: Dict[str, int]) -> None:
        for market_id, price in prices.items():
            if market_id in self.positions:
                self.positions[market_id].update_price(price)

    def reset_daily_stats(self) -> None:
        today = datetime.utcnow().date()
        if today != self.last_reset_date:
            self.daily_stats = {}
            self.last_reset_date = today
            self.daily_loss_count = 0

    def can_open_position(self, quantity: int, price: int) -> Tuple[bool, str]:
        cost = quantity * price
        if cost > self.cash_balance:
            return False, "Insufficient balance"

        if len(self.positions) >= self.max_open_positions:
            return False, "Max positions reached"

        return True, "OK"

    def close_position(self, market_id: str, price: int) -> Optional[Trade]:
        if market_id not in self.positions:
            return None

        position = self.positions[market_id]
        pnl = (price - position.avg_cost) * position.quantity

        trade = self.record_trade(
            market_id=market_id,
            side=position.side,
            quantity=position.quantity,
            price=price,
            pnl=pnl
        )

        self.cash_balance += position.market_value
        del self.positions[market_id]

        return trade
