"""
Limited Risk Position Sizing

Calculates position sizes constrained to $10-$15 trade values
with fee-aware optimization for small account trading.
"""

from typing import Dict, Tuple, Optional, Any
import logging

from src.core.arbitrage import ArbitrageOpportunity, ArbitrageType
from src.core.limited_risk_manager import LimitedRiskManager

logger = logging.getLogger(__name__)


class LimitedRiskPositionSizer:
    """
    Position sizer for $10-$15 trade mode with Kalshi fee optimization.

    Key insight: Kalshi fees are lower when contracts are priced
    near $0 or $1 (extremes) vs $0.50 (middle).
    Fee formula: 0.07 * C * P * (1-P) where P is price in dollars
    """

    def __init__(self, risk_manager: LimitedRiskManager):
        self.risk_manager = risk_manager

    def calculate_contracts_for_target(
        self, price_cents: int, target_value_cents: int, min_contracts: int = 1
    ) -> int:
        """
        Calculate number of contracts to achieve target dollar value.

        Args:
            price_cents: Price per contract in cents
            target_value_cents: Target trade value in cents
            min_contracts: Minimum contracts required

        Returns:
            Number of contracts
        """
        if price_cents <= 0:
            return 0

        contracts = target_value_cents // price_cents
        return max(contracts, min_contracts)

    def find_optimal_size(
        self, price_cents: int, spread_cents: int
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Find optimal position size within $10-$15 range.

        Tries different sizes and picks the one with best profit/fee ratio.

        Args:
            price_cents: Entry price in cents
            spread_cents: Expected profit per contract in cents

        Returns:
            Tuple of (optimal_contracts, analysis_dict)
        """
        config = self.risk_manager.config
        best_contracts = 0
        best_score = -1
        best_analysis = None

        # Try different contract quantities
        for contracts in range(1, 100):
            trade_value = contracts * price_cents

            # Skip if outside $10-$15 range
            if trade_value < config.min_trade_cents:
                continue
            if trade_value > self.risk_manager.current_max_trade_cents:
                break

            # Calculate fee impact
            analysis = self.risk_manager.calculate_fee_impact(
                contracts, price_cents, spread_cents
            )

            if not analysis["is_viable"]:
                continue

            if analysis["fee_too_high"]:
                continue

            # Score by net profit (prefer higher profit)
            score = analysis["net_profit_cents"]

            if score > best_score:
                best_score = score
                best_contracts = contracts
                best_analysis = analysis

        if best_contracts == 0:
            return 0, {"error": "No viable position size found"}

        return best_contracts, best_analysis

    def adjust_opportunity_for_limited_risk(
        self, opportunity: ArbitrageOpportunity
    ) -> Optional[ArbitrageOpportunity]:
        """
        Adjust an arbitrage opportunity to fit within $10-$15 constraints.

        Args:
            opportunity: Original arbitrage opportunity

        Returns:
            Adjusted opportunity or None if not viable
        """
        if not self.risk_manager.enabled:
            return opportunity

        # Calculate spread per contract
        spread_cents = opportunity.sell_price - opportunity.buy_price

        # Find optimal size
        contracts, analysis = self.find_optimal_size(
            opportunity.buy_price, spread_cents
        )

        if contracts == 0:
            logger.debug(f"Opportunity {opportunity.id} cannot fit in $10-$15 range")
            return None

        # Scale profit estimates
        scale_factor = (
            contracts / opportunity.quantity if opportunity.quantity > 0 else 1
        )

        adjusted = ArbitrageOpportunity(
            id=f"{opportunity.id}_limited",
            type=opportunity.type,
            market_id_1=opportunity.market_id_1,
            market_id_2=opportunity.market_id_2,
            buy_market_id=opportunity.buy_market_id,
            sell_market_id=opportunity.sell_market_id,
            buy_price=opportunity.buy_price,
            sell_price=opportunity.sell_price,
            quantity=contracts,
            profit_cents=int(opportunity.profit_cents * scale_factor),
            profit_percent=opportunity.profit_percent,
            confidence=opportunity.confidence,
            fees=analysis["total_fee_cents"],
            net_profit_cents=analysis["net_profit_cents"],
            risk_level="low",
            execution_window_seconds=30,
        )

        logger.info(
            f"Adjusted opportunity {opportunity.id}: {contracts} contracts, "
            f"net profit ${adjusted.net_profit_cents / 100:.2f}, "
            f"fees {analysis['fee_percent_of_profit']:.1f}%"
        )

        return adjusted

    def get_market_liquidity_estimate(self, orderbook_depth: int) -> str:
        """
        Estimate market liquidity category.

        Args:
            orderbook_depth: Total contracts available in orderbook

        Returns:
            Liquidity category string
        """
        if orderbook_depth < 10:
            return "very_illiquid"
        elif orderbook_depth < 50:
            return "illiquid"
        elif orderbook_depth < 200:
            return "moderate"
        elif orderbook_depth < 1000:
            return "liquid"
        else:
            return "very_liquid"

    def recommend_price_range_for_fees(self) -> Dict[str, Any]:
        """
        Recommend optimal price ranges to minimize fee impact.

        Returns:
            Dictionary with price range recommendations
        """
        return {
            "optimal_ranges": [
                {
                    "min": 1,
                    "max": 20,
                    "description": "Very low prices (1-20¢)",
                    "fee_efficiency": "excellent",
                },
                {
                    "min": 80,
                    "max": 99,
                    "description": "Very high prices (80-99¢)",
                    "fee_efficiency": "excellent",
                },
            ],
            "acceptable_ranges": [
                {
                    "min": 20,
                    "max": 40,
                    "description": "Low prices (20-40¢)",
                    "fee_efficiency": "good",
                },
                {
                    "min": 60,
                    "max": 80,
                    "description": "High prices (60-80¢)",
                    "fee_efficiency": "good",
                },
            ],
            "avoid_ranges": [
                {
                    "min": 40,
                    "max": 60,
                    "description": "Mid prices (40-60¢)",
                    "fee_efficiency": "poor",
                    "reason": "Maximum fees at 50¢",
                },
            ],
            "note": "Fees are lowest when contracts are priced near $0 or $1",
        }
