"""Backtesting module for testing trading strategies on historical data."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_balance_cents: int = 100000
    max_position_contracts: int = 1000
    max_order_value_cents: int = 10000
    min_liquidity_score: float = 50.0
    max_slippage_percent: float = 5.0
    min_fill_probability: float = 0.5
    max_spread_percent: float = 10.0
    arbitrage_threshold: float = 0.95
    paper_mode: bool = True
    fee_percent: float = 0.01


@dataclass
class BacktestTrade:
    """Represents a single trade in the backtest."""
    timestamp: str
    market_id: str
    side: str
    price: int
    count: int
    total_cents: int
    fee_cents: int


@dataclass
class BacktestStats:
    """Statistics from a backtest run."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit_cents: int = 0
    total_fees_cents: int = 0
    final_balance_cents: int = 0
    max_drawdown_percent: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    avg_win_cents: int = 0
    avg_loss_cents: int = 0
    profit_factor: float = 0.0
    trades: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class BacktestResult:
    """Result of a backtest run."""
    config: BacktestConfig
    stats: BacktestStats
    market_data: List[Dict[str, Any]]
    start_time: str
    end_time: str
    duration_seconds: float


class Backtester:
    """Backtesting engine for trading strategies."""

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.balance_cents = self.config.initial_balance_cents
        self.positions: Dict[str, int] = {}
        self.trades: List[BacktestTrade] = []
        self.daily_pnl: List[int] = []
        self.peak_balance = self.balance_cents

    def run(
        self,
        market_data: List[Dict[str, Any]],
        opportunities: Optional[List[Dict[str, Any]]] = None
    ) -> BacktestResult:
        """Run backtest on market data.

        Args:
            market_data: List of market snapshots with orderbook data
            opportunities: Detected arbitrage opportunities

        Returns:
            BacktestResult with statistics and trades
        """
        start_time = datetime.utcnow().isoformat()
        self.balance_cents = self.config.initial_balance_cents
        self.positions = {}
        self.trades = []
        self.daily_pnl = []
        self.peak_balance = self.balance_cents

        for snapshot in market_data:
            self._process_snapshot(snapshot)

        stats = self._calculate_stats()
        end_time = datetime.utcnow().isoformat()
        duration = (datetime.fromisoformat(end_time) - datetime.fromisoformat(start_time)).total_seconds()

        return BacktestResult(
            config=self.config,
            stats=stats,
            market_data=market_data,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration
        )

    def _process_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Process a single market snapshot."""
        market_id = snapshot.get('market_id', 'unknown')
        orderbook = snapshot.get('orderbook', {})

        bids = orderbook.get('bids', [])[:5]
        asks = orderbook.get('asks', [])[:5]

        if not bids or not asks:
            return

        best_bid = bids[0].get('price', 0)
        best_ask = asks[0].get('price', 0)

        spread_percent = (best_ask - best_bid) / best_ask * 100 if best_ask > 0 else 0

        if spread_percent < self.config.max_spread_percent:
            self._execute_trades(market_id, bids, asks)

    def _execute_trades(
        self,
        market_id: str,
        bids: List[Dict],
        asks: List[Dict]
    ) -> None:
        """Execute simulated trades based on arbitrage opportunities."""
        for level in asks[:3]:
            if self.balance_cents >= level.get('price', 0) * level.get('count', 0):
                price = level.get('price', 0)
                count = min(level.get('count', 1), self.config.max_position_contracts)
                fee = int(price * count * self.config.fee_percent / 100)
                total = price * count + fee

                if total <= self.balance_cents and count > 0:
                    trade = BacktestTrade(
                        timestamp=datetime.utcnow().isoformat(),
                        market_id=market_id,
                        side='buy',
                        price=price,
                        count=count,
                        total_cents=total,
                        fee_cents=fee
                    )
                    self.trades.append(trade)
                    self.balance_cents -= total
                    self.positions[market_id] = self.positions.get(market_id, 0) + count
                    break

    def _calculate_stats(self) -> BacktestStats:
        """Calculate statistics from completed trades."""
        if not self.trades:
            return BacktestStats()

        total_profit = 0
        winning = 0
        losing = 0
        wins = []
        losses = []

        for trade in self.trades:
            if trade.side == 'buy':
                current_positions = self.positions.get(trade.market_id, 0)
                if current_positions > 0:
                    sell_price = trade.price + int(trade.price * 0.01)
                    profit = (sell_price - trade.price) * trade.count - trade.fee_cents
                    total_profit += profit
                    self.balance_cents += trade.price * trade.count - trade.fee_cents

                    if profit > 0:
                        winning += 1
                        wins.append(profit)
                    else:
                        losing += 1
                        losses.append(profit)

                    self.positions[trade.market_id] -= trade.count

        final_balance = self.balance_cents
        total_return = (final_balance - self.config.initial_balance_cents) / self.config.initial_balance_cents * 100

        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 1

        profit_factor = sum(wins) / avg_loss if avg_loss > 0 else 0

        trades_data = [
            {
                'timestamp': t.timestamp,
                'market_id': t.market_id,
                'side': t.side,
                'price': t.price,
                'count': t.count,
                'total_cents': t.total_cents,
                'fee_cents': t.fee_cents
            }
            for t in self.trades
        ]

        return BacktestStats(
            total_trades=len(self.trades),
            winning_trades=winning,
            losing_trades=losing,
            total_profit_cents=total_profit,
            total_fees_cents=sum(t.fee_cents for t in self.trades),
            final_balance_cents=final_balance,
            win_rate=winning / len(self.trades) * 100 if self.trades else 0,
            avg_win_cents=int(avg_win),
            avg_loss_cents=int(avg_loss),
            profit_factor=profit_factor,
            trades=trades_data
        )
