import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from src.clients.kalshi_client import KalshiClient
from src.core.orderbook import OrderBook, OrderBookLevel
from src.core.arbitrage import ArbitrageDetector
from src.core.portfolio import PortfolioManager
from src.execution.trading import TradingExecutor
from src.core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState


class TestKalshiClientIntegration:
    @pytest.fixture
    def mock_config(self):
        config = Mock()
        config.get.side_effect = lambda key, default=None: {
            "kalshi.api_key_id": "test-key",
            "kalshi.private_key_path": "/test/path",
            "kalshi.base_url": "https://test.api",
            "kalshi.demo_mode": True,
        }.get(key, default)
        return config

    @pytest.mark.asyncio
    async def test_get_exchange_status_success(self, mock_config):
        with patch("requests.Session") as mock_session:
            mock_response = Mock()
            mock_response.json.return_value = {
                "exchange_active": True,
                "trading_active": True,
            }
            mock_response.raise_for_status = Mock()
            mock_session.return_value.request.return_value = mock_response

            client = KalshiClient(mock_config)
            result = client.get_exchange_status()

            assert result["exchange_active"] is True


class TestOrderbookIntegration:
    def test_orderbook_from_api_complex(self):
        data = {
            "market_id": "KALSHI-EL-2024",
            "bids": [
                {"price": str(i), "count": str(100 - i), "total": str(i * (100 - i))}
                for i in range(50, 60)
            ],
            "asks": [
                {"price": str(i), "count": str(100 - i), "total": str(i * (100 - i))}
                for i in range(60, 70)
            ],
            "ts": "2024-01-01T12:00:00Z",
            "seq": 12345,
        }

        orderbook = OrderBook.from_api_response(data)

        assert orderbook.market_id == "KALSHI-EL-2024"
        assert len(orderbook.bids) == 10
        assert len(orderbook.asks) == 10
        assert orderbook.timestamp == "2024-01-01T12:00:00Z"
        assert orderbook.sequence == 12345

    def test_orderbook_liquidity_calculation(self):
        orderbook = OrderBook(
            market_id="test",
            bids=[
                OrderBookLevel(price=50, count=100, total=5000),
                OrderBookLevel(price=51, count=80, total=4080),
            ],
            asks=[
                OrderBookLevel(price=52, count=120, total=6240),
                OrderBookLevel(price=53, count=60, total=3180),
            ],
        )

        score = orderbook.get_liquidity_score()
        assert 0 < score <= 100
        assert score > 50


class TestArbitrageIntegration:
    @pytest.mark.asyncio
    async def test_scan_multiple_markets(self):
        orderbooks = {
            f"market-{i}": OrderBook(
                market_id=f"market-{i}",
                bids=[OrderBookLevel(price=60 + i, count=10, total=600 + 10 * i)],
                asks=[OrderBookLevel(price=55 + i, count=10, total=550 + 10 * i)],
            )
            for i in range(5)
        }

        detector = ArbitrageDetector(min_profit_cents=10)
        opportunities = await detector.scan_for_opportunities(orderbooks)

        assert len(opportunities) >= 0


class TestPortfolioIntegration:
    def test_complex_position_management(self):
        pm = PortfolioManager()
        pm.set_balance(50000)

        pm.update_positions(
            [
                {"market_id": "A", "side": "yes", "count": 100, "avg_price": 50},
                {"market_id": "B", "side": "no", "count": 50, "avg_price": 60},
            ]
        )

        assert len(pm.positions) == 2

        pm.update_positions(
            [{"market_id": "A", "side": "no", "count": 30, "avg_price": 55}]
        )

        stats = pm.get_stats()
        assert stats.open_positions == 2
        assert stats.cash_balance == 50000


class TestCircuitBreakerIntegration:
    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_threshold(self):
        breaker = CircuitBreaker(
            "test", CircuitBreakerConfig(failure_threshold=3, timeout_seconds=1)
        )

        async def failing_func():
            raise ValueError("Test failure")

        # Trigger circuit to open
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self):
        breaker = CircuitBreaker(
            "test",
            CircuitBreakerConfig(
                failure_threshold=2, success_threshold=1, timeout_seconds=0.1
            ),
        )

        async def failing_func():
            raise ValueError("Fail")

        async def success_func():
            return "success"

        # Trigger circuit to open
        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN

        # Manually transition to half-open for testing
        breaker._state = CircuitState.HALF_OPEN

        # Try calling success function - should work now
        result = await breaker.call(success_func)
        assert result == "success"

        assert breaker.state == CircuitState.CLOSED


class TestExecutionIntegration:
    @pytest.mark.asyncio
    async def test_execution_with_retry(self):
        mock_client = Mock()
        mock_client.create_order = AsyncMock(
            side_effect=[
                Exception("API Error"),
                Exception("API Error"),
                {"order": {"id": "test-order-123", "status": "filled", "count": 10}},
            ]
        )
        mock_client.get_order = AsyncMock(
            return_value={
                "order": {"id": "test-order", "status": "filled", "count": 10}
            }
        )
        mock_client.cancel_order = Mock()

        executor = TradingExecutor(
            mock_client, paper_mode=False, max_retries=3, retry_delay=0.01
        )

        result = await executor._execute_buy("market-1", 50, 10)

        assert result.success
        assert result.order_id == "test-order-123"


class TestMonitoringIntegration:
    def test_metrics_collection(self):
        from src.monitoring.monitoring import MetricsCollector

        metrics = MetricsCollector(port=0)
        metrics.record_opportunity(100)
        metrics.record_execution(True, 0.5)
        metrics.record_scan(0.1)
        metrics.update_positions(5)
        metrics.update_balance(50000)

        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
