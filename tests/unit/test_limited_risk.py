"""
Tests for Limited Risk Trading Mode

Tests the core functionality of the $10-$15 trading constraints.
"""

import pytest
from datetime import datetime, timedelta

from src.core.limited_risk_manager import (
    LimitedRiskManager,
    LimitedRiskConfig,
    RiskStatus,
)
from src.analytics.limited_position_sizing import LimitedRiskPositionSizer


class TestLimitedRiskConfig:
    """Test LimitedRiskConfig dataclass."""

    def test_default_values(self):
        config = LimitedRiskConfig()
        assert config.min_trade_cents == 1000
        assert config.max_trade_cents == 1500
        assert config.max_daily_trades == 10
        assert config.max_daily_loss_cents == 5000
        assert config.cooldown_seconds == 60
        assert config.progressive_scaling_enabled is True


class TestLimitedRiskManager:
    """Test LimitedRiskManager functionality."""

    @pytest.fixture
    def manager(self):
        config = LimitedRiskConfig(
            enabled=True,
            min_trade_cents=1000,
            max_trade_cents=1500,
            max_daily_trades=10,
            max_daily_loss_cents=5000,
            cooldown_seconds=60,
        )
        return LimitedRiskManager(config)

    def test_auto_enable_below_threshold(self, manager):
        """Test auto-enable when balance is below threshold."""
        manager.enabled = False
        result = manager.check_auto_enable(50000)  # $500
        assert result is True
        assert manager.enabled is True

    def test_auto_enable_above_threshold(self, manager):
        """Test no auto-enable when balance is above threshold."""
        manager.enabled = False
        result = manager.check_auto_enable(200000)  # $2000
        assert result is False
        assert manager.enabled is False

    def test_validate_trade_size_valid(self, manager):
        """Test validation of valid trade sizes."""
        # 20 contracts at 50 cents = $10 (valid)
        is_valid, msg, qty = manager.validate_trade_size(20, 50)
        assert is_valid is True
        assert qty == 20

    def test_validate_trade_size_below_minimum(self, manager):
        """Test validation when trade is below minimum."""
        # 5 contracts at 50 cents = $2.50 (too small)
        is_valid, msg, qty = manager.validate_trade_size(5, 50)
        # Should adjust to minimum
        assert is_valid is True
        assert qty >= 20  # Should be at least 20 to reach $10

    def test_validate_trade_size_above_maximum(self, manager):
        """Test validation when trade is above maximum."""
        # 40 contracts at 50 cents = $20 (too large)
        is_valid, msg, qty = manager.validate_trade_size(40, 50)
        # Should adjust to maximum
        assert is_valid is True
        assert qty <= 30  # Should be at most 30 to stay at $15

    def test_can_execute_trade_allowed(self, manager):
        """Test trade execution allowed."""
        can_trade, status, reason = manager.can_execute_trade()
        assert can_trade is True
        assert status == RiskStatus.ALLOWED

    def test_can_execute_trade_daily_limit_reached(self, manager):
        """Test trade blocked when daily limit reached."""
        manager.daily_stats.trades_executed = 10
        can_trade, status, reason = manager.can_execute_trade()
        assert can_trade is False
        assert status == RiskStatus.DAILY_LIMIT

    def test_can_execute_trade_loss_limit_reached(self, manager):
        """Test trade blocked when loss limit reached."""
        manager.daily_stats.daily_pnl_cents = -5000  # $50 loss
        can_trade, status, reason = manager.can_execute_trade()
        assert can_trade is False
        assert status == RiskStatus.LOSS_LIMIT

    def test_can_execute_trade_cooldown_active(self, manager):
        """Test trade blocked during cooldown."""
        manager.daily_stats.last_trade_time = datetime.utcnow()
        can_trade, status, reason = manager.can_execute_trade()
        assert can_trade is False
        assert status == RiskStatus.COOLDOWN

    def test_calculate_fee_impact(self, manager):
        """Test fee calculation using Kalshi formula."""
        # 20 contracts @ 50 cents with 5 cent spread
        result = manager.calculate_fee_impact(20, 50, 5)

        assert result["contracts"] == 20
        assert result["price_cents"] == 50
        assert result["spread_cents"] == 5
        assert result["gross_profit_cents"] == 100  # 20 * 5
        assert result["total_fee_cents"] > 0
        assert "net_profit_cents" in result
        assert "fee_percent_of_profit" in result
        assert "is_viable" in result
        assert "fee_too_high" in result

    def test_is_market_eligible(self, manager):
        """Test market eligibility check."""
        # Eligible market with sufficient volume
        is_eligible, reason = manager.is_market_eligible("MARKET-1", 200000)
        assert is_eligible is True

        # Ineligible market with low volume
        is_eligible, reason = manager.is_market_eligible("MARKET-2", 50000)
        assert is_eligible is False

    def test_record_trade_win(self, manager):
        """Test recording a winning trade."""
        initial_trades = manager.daily_stats.trades_executed
        manager.record_trade(pnl_cents=100, fee_cents=35, gross_profit_cents=135)

        assert manager.daily_stats.trades_executed == initial_trades + 1
        assert manager.daily_stats.trades_won == 1
        assert manager.daily_stats.daily_pnl_cents == 100
        assert manager.consecutive_wins == 1

    def test_record_trade_loss(self, manager):
        """Test recording a losing trade."""
        manager.record_trade(pnl_cents=-50, fee_cents=35, gross_profit_cents=0)

        assert manager.daily_stats.trades_lost == 1
        assert manager.daily_stats.daily_pnl_cents == -50
        assert manager.consecutive_wins == 0

    def test_progressive_scaling(self, manager):
        """Test progressive scaling after profitable trades."""
        # Simulate 10 winning trades
        for i in range(10):
            manager.record_trade(pnl_cents=100, fee_cents=35, gross_profit_cents=135)

        # Should have scaled up
        assert manager._scaling_level >= 1
        assert manager.current_max_trade_cents > manager.config.max_trade_cents

    def test_reset_daily_stats(self, manager):
        """Test daily stats reset."""
        manager.record_trade(pnl_cents=100, fee_cents=35)
        manager.reset_daily_stats(force=True)

        assert manager.daily_stats.trades_executed == 0
        assert manager.daily_stats.daily_pnl_cents == 0
        assert manager.consecutive_wins == 0


class TestLimitedRiskPositionSizer:
    """Test LimitedRiskPositionSizer functionality."""

    @pytest.fixture
    def sizer(self):
        config = LimitedRiskConfig(min_trade_cents=1000, max_trade_cents=1500)
        manager = LimitedRiskManager(config)
        return LimitedRiskPositionSizer(manager)

    def test_calculate_contracts_for_target(self, sizer):
        """Test contract calculation for target dollar amount."""
        # $10 target at 50 cents = 20 contracts
        contracts = sizer.calculate_contracts_for_target(50, 1000)
        assert contracts == 20

    def test_find_optimal_size(self, sizer):
        """Test finding optimal position size."""
        contracts, analysis = sizer.find_optimal_size(50, 5)  # 50 cents, 5 cent spread

        if contracts > 0:
            assert contracts >= 20  # At least $10
            assert contracts <= 30  # At most $15
            assert analysis["is_viable"] is True

    def test_recommend_price_range_for_fees(self, sizer):
        """Test price range recommendations."""
        recommendations = sizer.recommend_price_range_for_fees()

        assert "optimal_ranges" in recommendations
        assert "acceptable_ranges" in recommendations
        assert "avoid_ranges" in recommendations
        assert len(recommendations["optimal_ranges"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
