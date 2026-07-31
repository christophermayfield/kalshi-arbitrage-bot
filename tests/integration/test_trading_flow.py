"""Integration tests for complete trading flows."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'fixtures'))

from mock_kalshi_api import MockKalshiAPI


class TestCompleteArbitrageTradingFlow:
    """Test complete arbitrage trading flow."""

    @pytest.fixture
    def mock_api(self):
        api = MockKalshiAPI()
        yield api
        api.reset()

    @pytest.mark.asyncio
    async def test_full_trading_cycle(self, mock_api):
        """Test complete cycle: scan -> detect -> execute -> reconcile."""
        markets = await mock_api.get_markets(status="open")
        assert len(markets) > 0

        market_id = markets[0]["market_id"]
        orderbook = await mock_api.get_market_orderbook(market_id)
        assert orderbook is not None

        initial_balance = (await mock_api.get_portfolio())["balance_cents"]

        # Buy
        buy_order = await mock_api.create_order(
            market_id=market_id,
            side="BUY",
            order_type="MARKET",
            price=7500,
            count=10,
        )
        assert buy_order["status"] == "FILLED"

        # Sell
        sell_order = await mock_api.create_order(
            market_id=market_id,
            side="SELL",
            order_type="MARKET",
            price=7600,
            count=10,
        )
        assert sell_order["status"] == "FILLED"

        # Verify profit
        final = await mock_api.get_portfolio()
        profit = 10 * (7600 - 7500)
        assert final["balance_cents"] == initial_balance + profit

    @pytest.mark.asyncio
    async def test_insufficient_balance_error(self, mock_api):
        """Test error handling for insufficient balance."""
        with pytest.raises(ValueError, match="Insufficient balance"):
            await mock_api.create_order(
                market_id="market_001",
                side="BUY",
                order_type="MARKET",
                price=1000000,
                count=1000,
            )

    @pytest.mark.asyncio
    async def test_order_cancellation(self, mock_api):
        """Test order cancellation."""
        order = await mock_api.create_order(
            market_id="market_001",
            side="BUY",
            order_type="LIMIT",
            price=7000,
            count=5,
        )
        assert order["status"] == "PENDING"

        cancelled = await mock_api.cancel_order(order["order_id"])
        assert cancelled["status"] == "CANCELLED"
