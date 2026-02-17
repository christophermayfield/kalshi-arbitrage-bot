import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.portfolio import PortfolioManager, PositionSide


class TestPortfolioManager:
    def test_set_and_get_balance(self):
        pm = PortfolioManager()
        pm.set_balance(10000)

        assert pm.get_balance() == 10000

    def test_update_positions_add(self):
        pm = PortfolioManager()
        pm.set_balance(10000)

        pm.update_positions([
            {'market_id': 'test-1', 'side': 'yes', 'count': 10, 'avg_price': 50}
        ])

        assert 'test-1' in pm.positions
        assert pm.positions['test-1'].quantity == 10
        assert pm.positions['test-1'].avg_cost == 50

    def test_update_positions_update_existing(self):
        pm = PortfolioManager()
        pm.set_balance(10000)

        pm.update_positions([
            {'market_id': 'test-1', 'side': 'yes', 'count': 10, 'avg_price': 50}
        ])
        pm.update_positions([
            {'market_id': 'test-1', 'side': 'yes', 'count': 10, 'avg_price': 60}
        ])

        assert pm.positions['test-1'].quantity == 20
        assert pm.positions['test-1'].avg_cost == 55

    def test_record_trade(self):
        pm = PortfolioManager()
        pm.set_balance(10000)

        trade = pm.record_trade(
            market_id='test-1',
            side=PositionSide.YES,
            quantity=10,
            price=50,
            pnl=100
        )

        assert trade is not None
        assert trade.pnl == 100
        assert len(pm.trades) == 1

    def test_get_stats(self):
        pm = PortfolioManager()
        pm.set_balance(10000)

        pm.record_trade('test-1', PositionSide.YES, 10, 50, 100)
        pm.record_trade('test-2', PositionSide.NO, 5, 50, -50)

        stats = pm.get_stats()

        assert stats.completed_trades == 2
        assert stats.win_rate == 50.0

    def test_check_risk_limits_ok(self):
        pm = PortfolioManager(max_daily_loss=10000, max_open_positions=5)
        pm.set_balance(10000)

        can_trade, reason = pm.can_open_position(100, 50)

        assert can_trade
        assert reason == "OK"

    def test_check_risk_limits_insufficient_balance(self):
        pm = PortfolioManager()
        pm.set_balance(1000)

        can_trade, reason = pm.can_open_position(100, 50)

        assert not can_trade
        assert "Insufficient" in reason

    def test_check_risk_limits_max_positions(self):
        pm = PortfolioManager(max_open_positions=2)
        pm.set_balance(10000)

        pm.update_positions([
            {'market_id': 'test-1', 'side': 'yes', 'count': 10, 'avg_price': 50}
        ])
        pm.update_positions([
            {'market_id': 'test-2', 'side': 'yes', 'count': 10, 'avg_price': 50}
        ])

        can_trade, reason = pm.can_open_position(10, 50)

        assert not can_trade
        assert "Max positions" in reason

    def test_update_prices(self):
        pm = PortfolioManager()
        pm.set_balance(10000)

        pm.update_positions([
            {'market_id': 'test-1', 'side': 'yes', 'count': 10, 'avg_price': 50}
        ])

        pm.update_prices({'test-1': 60})

        assert pm.positions['test-1'].current_price == 60
        assert pm.positions['test-1'].unrealized_pnl == 100

    def test_close_position(self):
        pm = PortfolioManager()
        pm.set_balance(10000)

        pm.update_positions([
            {'market_id': 'test-1', 'side': 'yes', 'count': 10, 'avg_price': 50}
        ])

        trade = pm.close_position('test-1', 60)

        assert trade is not None
        assert 'test-1' not in pm.positions

    def test_close_position_nonexistent(self):
        pm = PortfolioManager()

        trade = pm.close_position('nonexistent', 50)

        assert trade is None

    def test_reset_daily_stats(self):
        pm = PortfolioManager()
        pm.daily_stats = {'wins': 5, 'losses': 3}
        pm.last_reset_date = pm.last_reset_date  # Keep same date

        pm.reset_daily_stats()

        assert pm.daily_stats == {'wins': 5, 'losses': 3}
