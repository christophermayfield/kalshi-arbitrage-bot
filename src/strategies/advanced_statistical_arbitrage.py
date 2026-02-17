"""
Advanced Statistical Arbitrage Strategies
Multiple statistical arbitrage approaches with ML enhancement
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
import redis.asyncio as redis

from ..utils.performance_cache import PerformanceCache

logger = logging.getLogger(__name__)


class ArbitrageStrategy(Enum):
    """Types of statistical arbitrage strategies"""

    PAIRS_TRADING = "pairs_trading"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    MARKET_NEUTRAL = "market_neutral"
    VARIANCE_SWAP = "variance_swap"
    CORRELATION_TRADING = "correlation_trading"


class SignalType(Enum):
    """Types of trading signals"""

    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"
    CLOSE = "close"


@dataclass
class StatisticalSignal:
    """Statistical arbitrage trading signal"""

    strategy: ArbitrageStrategy
    assets: List[str]
    signal_type: SignalType
    confidence: float
    z_score: float
    spread: float
    expected_return: float
    risk_score: float
    entry_price: List[float]
    stop_loss: List[float]
    take_profit: List[float]
    position_size: List[float]
    correlation: float = 0.0
    half_life: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PairsTradingModel:
    """Model for pairs trading strategy"""

    asset1: str
    asset2: str
    hedge_ratio: float
    beta: float
    alpha: float
    spread_mean: float
    spread_std: float
    half_life: float
    cointegration_pvalue: float
    correlation: float
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class MeanReversionModel:
    """Model for mean reversion strategy"""

    asset: str
    lookback_period: int
    mean_reversion_rate: float
    volatility: float
    upper_band: float
    lower_band: float
    middle_band: float
    band_width: float
    z_score_threshold: float


class AdvancedStatisticalArbitrage:
    """
    Advanced statistical arbitrage system with multiple strategies
    and machine learning enhancement
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.stat_config = config.get("statistical_arbitrage", {})

        # Strategy parameters
        self.enabled_strategies = [
            ArbitrageStrategy(s)
            for s in self.stat_config.get(
                "enabled_strategies", ["pairs_trading", "mean_reversion", "momentum"]
            )
        ]

        # Trading parameters
        self.min_confidence = self.stat_config.get("min_confidence", 0.7)
        self.max_position_size = self.stat_config.get("max_position_size", 10000)
        self.risk_limit = self.stat_config.get("risk_limit", 0.02)  # 2% risk per trade

        # Statistical parameters
        self.lookback_periods = self.stat_config.get("lookback_periods", [20, 50, 100])
        self.z_score_threshold = self.stat_config.get("z_score_threshold", 2.0)
        self.cointegration_threshold = self.stat_config.get(
            "cointegration_threshold", 0.05
        )
        self.correlation_threshold = self.stat_config.get("correlation_threshold", 0.7)

        # Market data
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.return_data: Dict[str, pd.DataFrame] = {}
        self.volume_data: Dict[str, pd.DataFrame] = {}

        # Strategy models
        self.pairs_models: Dict[str, PairsTradingModel] = {}
        self.mean_reversion_models: Dict[str, MeanReversionModel] = {}

        # Signal tracking
        self.active_signals: Dict[str, StatisticalSignal] = {}
        self.signal_history: List[StatisticalSignal] = []
        self.performance_metrics: Dict[str, Any] = defaultdict(float)

        # Machine learning enhancement
        self.anomaly_detector: Optional[IsolationForest] = None
        self.predictive_models: Dict[str, LinearRegression] = {}

        # Performance cache
        self.cache = PerformanceCache(
            redis_url=config.get("redis_url", "redis://localhost:6379"),
            default_ttl=30,  # 30 second TTL for statistical data
        )

        # Asset universe
        self.asset_universe = self.stat_config.get(
            "asset_universe", ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        )
        self.pair_combinations = self._generate_pair_combinations()

        logger.info("Advanced Statistical Arbitrage initialized")

    async def initialize(self) -> None:
        """Initialize the statistical arbitrage system"""
        try:
            # Initialize ML models
            await self._initialize_ml_models()

            # Load historical data
            await self._load_historical_data()

            # Initialize strategy models
            await self._initialize_pairs_models()
            await self._initialize_mean_reversion_models()

            # Start signal generation
            asyncio.create_task(self._signal_generation_loop())

            logger.info("Advanced Statistical Arbitrage initialized successfully")

        except Exception as e:
            logger.error(f"Statistical Arbitrage initialization failed: {e}")
            raise

    async def generate_signals(self) -> List[StatisticalSignal]:
        """Generate trading signals from all enabled strategies"""
        signals = []

        try:
            # Update market data
            await self._update_market_data()

            # Generate signals for each strategy
            for strategy in self.enabled_strategies:
                if strategy == ArbitrageStrategy.PAIRS_TRADING:
                    pairs_signals = await self._generate_pairs_trading_signals()
                    signals.extend(pairs_signals)

                elif strategy == ArbitrageStrategy.MEAN_REVERSION:
                    mean_reversion_signals = (
                        await self._generate_mean_reversion_signals()
                    )
                    signals.extend(mean_reversion_signals)

                elif strategy == ArbitrageStrategy.MOMENTUM:
                    momentum_signals = await self._generate_momentum_signals()
                    signals.extend(momentum_signals)

                elif strategy == ArbitrageStrategy.STATISTICAL_ARBITRAGE:
                    stat_arb_signals = (
                        await self._generate_statistical_arbitrage_signals()
                    )
                    signals.extend(stat_arb_signals)

                elif strategy == ArbitrageStrategy.MARKET_NEUTRAL:
                    market_neutral_signals = (
                        await self._generate_market_neutral_signals()
                    )
                    signals.extend(market_neutral_signals)

                elif strategy == ArbitrageStrategy.CORRELATION_TRADING:
                    correlation_signals = (
                        await self._generate_correlation_trading_signals()
                    )
                    signals.extend(correlation_signals)

            # Filter and rank signals
            filtered_signals = await self._filter_and_rank_signals(signals)

            # Update active signals
            for signal in filtered_signals:
                signal_key = f"{signal.strategy.value}_{'_'.join(signal.assets)}"
                self.active_signals[signal_key] = signal
                self.signal_history.append(signal)

            # Cache signals
            await self.cache.set(
                "statistical_signals",
                [
                    {
                        "strategy": s.strategy.value,
                        "assets": s.assets,
                        "signal_type": s.signal_type.value,
                        "confidence": s.confidence,
                        "z_score": s.z_score,
                        "expected_return": s.expected_return,
                        "timestamp": s.timestamp.isoformat(),
                    }
                    for s in filtered_signals
                ],
                ttl=60,
            )

            return filtered_signals

        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            return []

    async def _generate_pairs_trading_signals(self) -> List[StatisticalSignal]:
        """Generate pairs trading signals"""
        signals = []

        try:
            for pair_key, model in self.pairs_models.items():
                if (
                    model.asset1 not in self.price_data
                    or model.asset2 not in self.price_data
                ):
                    continue

                # Get current prices
                asset1_price = self.price_data[model.asset1]["close"].iloc[-1]
                asset2_price = self.price_data[model.asset2]["close"].iloc[-1]

                # Calculate current spread
                current_spread = asset1_price - model.hedge_ratio * asset2_price

                # Calculate Z-score
                z_score = (
                    (current_spread - model.spread_mean) / model.spread_mean
                    if model.spread_mean > 0
                    else 0
                )

                # Generate signal based on Z-score
                if abs(z_score) > self.z_score_threshold:
                    if z_score > 0:
                        # Short the spread (short asset1, long asset2)
                        signal_type = SignalType.SHORT
                        position_sizes = [
                            -self.max_position_size / 2,
                            self.max_position_size / 2,
                        ]
                    else:
                        # Long the spread (long asset1, short asset2)
                        signal_type = SignalType.LONG
                        position_sizes = [
                            self.max_position_size / 2,
                            -self.max_position_size / 2,
                        ]

                    # Calculate expected return (mean reversion)
                    expected_return = (
                        -abs(z_score) * model.spread_std * 0.5
                    )  # Conservative estimate

                    # Calculate confidence
                    confidence = min(1.0, (abs(z_score) - self.z_score_threshold) / 2.0)
                    if model.cointegration_pvalue < 0.01:  # Strong cointegration
                        confidence *= 1.2

                    # Calculate risk score
                    risk_score = abs(z_score) / 5.0  # Normalize to 0-1

                    signal = StatisticalSignal(
                        strategy=ArbitrageStrategy.PAIRS_TRADING,
                        assets=[model.asset1, model.asset2],
                        signal_type=signal_type,
                        confidence=min(1.0, confidence),
                        z_score=z_score,
                        spread=current_spread,
                        expected_return=expected_return,
                        risk_score=min(1.0, risk_score),
                        entry_price=[asset1_price, asset2_price],
                        stop_loss=[
                            asset1_price * (1 - 0.05)
                            if signal_type == SignalType.LONG
                            else asset1_price * (1 + 0.05),
                            asset2_price * (1 - 0.05)
                            if signal_type == SignalType.SHORT
                            else asset2_price * (1 + 0.05),
                        ],
                        take_profit=[
                            asset1_price * (1 + 0.02)
                            if signal_type == SignalType.LONG
                            else asset1_price * (1 - 0.02),
                            asset2_price * (1 - 0.02)
                            if signal_type == SignalType.SHORT
                            else asset2_price * (1 + 0.02),
                        ],
                        position_size=position_sizes,
                        correlation=model.correlation,
                        half_life=model.half_life,
                        metadata={
                            "hedge_ratio": model.hedge_ratio,
                            "cointegration_pvalue": model.cointegration_pvalue,
                            "beta": model.beta,
                            "alpha": model.alpha,
                        },
                    )

                    if signal.confidence >= self.min_confidence:
                        signals.append(signal)

            return signals

        except Exception as e:
            logger.error(f"Pairs trading signal generation failed: {e}")
            return []

    async def _generate_mean_reversion_signals(self) -> List[StatisticalSignal]:
        """Generate mean reversion signals"""
        signals = []

        try:
            for asset, model in self.mean_reversion_models.items():
                if asset not in self.price_data:
                    continue

                # Get current price
                current_price = self.price_data[asset]["close"].iloc[-1]

                # Calculate distance from bands
                distance_from_upper = (
                    current_price - model.upper_band
                ) / model.upper_band
                distance_from_lower = (
                    model.lower_band - current_price
                ) / model.lower_band

                # Generate signals based on band penetration
                if current_price > model.upper_band:
                    # Price is above upper band - short signal
                    z_score = (current_price - model.middle_band) / model.band_width
                    signal_type = SignalType.SHORT
                    position_size = -self.max_position_size / 3

                elif current_price < model.lower_band:
                    # Price is below lower band - long signal
                    z_score = (current_price - model.middle_band) / model.band_width
                    signal_type = SignalType.LONG
                    position_size = self.max_position_size / 3

                else:
                    continue

                # Calculate expected return (reversion to mean)
                expected_return = (
                    abs(current_price - model.middle_band) * model.mean_reversion_rate
                )

                # Calculate confidence
                confidence = min(1.0, abs(z_score) / model.z_score_threshold)
                if model.volatility < np.std(
                    self.price_data[asset]["close"].iloc[-50:]
                ):  # Low volatility increases confidence
                    confidence *= 1.1

                # Calculate risk score
                risk_score = abs(z_score) / 4.0

                signal = StatisticalSignal(
                    strategy=ArbitrageStrategy.MEAN_REVERSION,
                    assets=[asset],
                    signal_type=signal_type,
                    confidence=min(1.0, confidence),
                    z_score=z_score,
                    spread=0,  # Not applicable for single asset
                    expected_return=expected_return,
                    risk_score=min(1.0, risk_score),
                    entry_price=[current_price],
                    stop_loss=[
                        current_price * (1 + 0.03)
                        if signal_type == SignalType.SHORT
                        else current_price * (1 - 0.03)
                    ],
                    take_profit=[
                        current_price * (1 - 0.02)
                        if signal_type == SignalType.SHORT
                        else current_price * (1 + 0.02)
                    ],
                    position_size=[position_size],
                    metadata={
                        "upper_band": model.upper_band,
                        "lower_band": model.lower_band,
                        "middle_band": model.middle_band,
                        "band_width": model.band_width,
                        "mean_reversion_rate": model.mean_reversion_rate,
                    },
                )

                if signal.confidence >= self.min_confidence:
                    signals.append(signal)

            return signals

        except Exception as e:
            logger.error(f"Mean reversion signal generation failed: {e}")
            return []

    async def _generate_momentum_signals(self) -> List[StatisticalSignal]:
        """Generate momentum-based signals"""
        signals = []

        try:
            for asset in self.asset_universe:
                if asset not in self.price_data:
                    continue

                prices = self.price_data[asset]["close"]
                if len(prices) < 50:
                    continue

                # Calculate momentum indicators
                returns = prices.pct_change().dropna()

                # Short-term momentum (10-day)
                mom_10 = (prices.iloc[-1] - prices.iloc[-10]) / prices.iloc[-10]

                # Long-term momentum (30-day)
                mom_30 = (prices.iloc[-1] - prices.iloc[-30]) / prices.iloc[-30]

                # RSI-like momentum
                momentum_score = (mom_10 + mom_30) / 2

                # Calculate confidence
                volatility = returns.rolling(20).std().iloc[-1]
                confidence = abs(momentum_score) / volatility if volatility > 0 else 0

                # Generate signals
                if confidence > 0.5:  # Minimum momentum threshold
                    if momentum_score > 0:
                        signal_type = SignalType.LONG
                        position_size = self.max_position_size / 4
                    else:
                        signal_type = SignalType.SHORT
                        position_size = -self.max_position_size / 4

                    # Expected return based on momentum persistence
                    expected_return = (
                        momentum_score * prices.iloc[-1] * 0.7
                    )  # Conservative

                    # Risk score based on volatility
                    risk_score = min(1.0, volatility * 50)  # Normalize volatility

                    signal = StatisticalSignal(
                        strategy=ArbitrageStrategy.MOMENTUM,
                        assets=[asset],
                        signal_type=signal_type,
                        confidence=min(
                            1.0, confidence / 2
                        ),  # Reduce confidence for momentum
                        z_score=momentum_score / volatility if volatility > 0 else 0,
                        spread=0,
                        expected_return=expected_return,
                        risk_score=min(1.0, risk_score),
                        entry_price=[prices.iloc[-1]],
                        stop_loss=[
                            prices.iloc[-1] * (1 - 0.04)
                            if signal_type == SignalType.LONG
                            else prices.iloc[-1] * (1 + 0.04)
                        ],
                        take_profit=[
                            prices.iloc[-1] * (1 + 0.06)
                            if signal_type == SignalType.LONG
                            else prices.iloc[-1] * (1 - 0.06)
                        ],
                        position_size=[position_size],
                        metadata={
                            "mom_10": mom_10,
                            "mom_30": mom_30,
                            "volatility": volatility,
                        },
                    )

                    if signal.confidence >= self.min_confidence:
                        signals.append(signal)

            return signals

        except Exception as e:
            logger.error(f"Momentum signal generation failed: {e}")
            return []

    async def _generate_statistical_arbitrage_signals(self) -> List[StatisticalSignal]:
        """Generate general statistical arbitrage signals"""
        signals = []

        try:
            # Analyze cross-asset statistical relationships
            for asset1 in self.asset_universe:
                for asset2 in self.asset_universe:
                    if (
                        asset1 == asset2
                        or asset1 not in self.price_data
                        or asset2 not in self.price_data
                    ):
                        continue

                    # Get price series
                    prices1 = self.price_data[asset1]["close"]
                    prices2 = self.price_data[asset2]["close"]

                    # Calculate correlation
                    correlation = prices1.corr(prices2)

                    if abs(correlation) < self.correlation_threshold:
                        continue

                    # Calculate beta (slope of regression)
                    X = sm.add_constant(prices2)
                    model = sm.OLS(prices1, X).fit()
                    beta = model.params[1]

                    # Calculate residuals
                    residuals = model.resid
                    residual_std = residuals.std()

                    # Current residual
                    current_residual = prices1.iloc[-1] - (
                        model.params[0] + beta * prices2.iloc[-1]
                    )
                    z_score = current_residual / residual_std if residual_std > 0 else 0

                    # Generate signal if residual is significant
                    if abs(z_score) > self.z_score_threshold:
                        if current_residual > 0:
                            # Asset1 is overpriced relative to asset2
                            signal_type = SignalType.SHORT  # Short asset1, long asset2
                            position_sizes = [
                                -self.max_position_size / 4,
                                self.max_position_size / 4,
                            ]
                        else:
                            # Asset1 is underpriced relative to asset2
                            signal_type = SignalType.LONG  # Long asset1, short asset2
                            position_sizes = [
                                self.max_position_size / 4,
                                -self.max_position_size / 4,
                            ]

                        # Expected return (residual mean reversion)
                        expected_return = -current_residual * 0.5  # Conservative

                        # Confidence based on correlation and z-score
                        confidence = (
                            abs(correlation) * abs(z_score) / self.z_score_threshold
                        ) / 2

                        # Risk score
                        risk_score = abs(z_score) / 4.0

                        signal = StatisticalSignal(
                            strategy=ArbitrageStrategy.STATISTICAL_ARBITRAGE,
                            assets=[asset1, asset2],
                            signal_type=signal_type,
                            confidence=min(1.0, confidence),
                            z_score=z_score,
                            spread=current_residual,
                            expected_return=expected_return,
                            risk_score=min(1.0, risk_score),
                            entry_price=[prices1.iloc[-1], prices2.iloc[-1]],
                            stop_loss=[
                                prices1.iloc[-1] * (1 + 0.03)
                                if signal_type == SignalType.SHORT
                                else prices1.iloc[-1] * (1 - 0.03),
                                prices2.iloc[-1] * (1 - 0.03)
                                if signal_type == SignalType.LONG
                                else prices2.iloc[-1] * (1 + 0.03),
                            ],
                            take_profit=[
                                prices1.iloc[-1] * (1 - 0.02)
                                if signal_type == SignalType.SHORT
                                else prices1.iloc[-1] * (1 + 0.02),
                                prices2.iloc[-1] * (1 + 0.02)
                                if signal_type == SignalType.LONG
                                else prices2.iloc[-1] * (1 - 0.02),
                            ],
                            position_size=position_sizes,
                            correlation=correlation,
                            metadata={
                                "beta": beta,
                                "residual_std": residual_std,
                                "current_residual": current_residual,
                            },
                        )

                        if signal.confidence >= self.min_confidence:
                            signals.append(signal)

            return signals

        except Exception as e:
            logger.error(f"Statistical arbitrage signal generation failed: {e}")
            return []

    async def _generate_market_neutral_signals(self) -> List[StatisticalSignal]:
        """Generate market neutral signals"""
        signals = []

        try:
            # Simple market neutral strategy using index futures concept
            # For crypto, use BTC as market proxy

            market_proxy = "BTC/USDT"
            if market_proxy not in self.price_data:
                return signals

            market_price = self.price_data[market_proxy]["close"].iloc[-1]
            market_returns = (
                self.price_data[market_proxy]["close"].pct_change().dropna()
            )

            for asset in self.asset_universe:
                if asset == market_proxy or asset not in self.price_data:
                    continue

                # Calculate asset beta to market
                asset_returns = self.price_data[asset]["close"].pct_change().dropna()

                if len(asset_returns) < 20 or len(market_returns) < 20:
                    continue

                # Align returns
                min_length = min(len(asset_returns), len(market_returns))
                asset_returns_aligned = asset_returns.iloc[-min_length:]
                market_returns_aligned = market_returns.iloc[-min_length:]

                # Calculate beta
                covariance = np.cov(asset_returns_aligned, market_returns_aligned)[0, 1]
                market_variance = np.var(market_returns_aligned)
                beta = covariance / market_variance if market_variance > 0 else 1.0

                # Calculate alpha
                asset_return = asset_returns_aligned.mean()
                market_return = market_returns_aligned.mean()
                alpha = asset_return - beta * market_return

                # Current asset price
                asset_price = self.price_data[asset]["close"].iloc[-1]

                # Expected excess return
                expected_excess_return = alpha + beta * market_returns_aligned.iloc[-1]

                # Generate signal based on alpha
                if abs(alpha) > 0.001:  # 0.1% alpha threshold
                    if alpha > 0:
                        signal_type = SignalType.LONG
                        position_size = self.max_position_size / 5
                    else:
                        signal_type = SignalType.SHORT
                        position_size = -self.max_position_size / 5

                    # Hedge with market proxy
                    hedge_size = -beta * position_size

                    # Confidence based on alpha significance
                    alpha_std = np.std(
                        asset_returns_aligned - beta * market_returns_aligned
                    )
                    confidence = abs(alpha) / alpha_std if alpha_std > 0 else 0

                    signal = StatisticalSignal(
                        strategy=ArbitrageStrategy.MARKET_NEUTRAL,
                        assets=[asset, market_proxy],
                        signal_type=signal_type,
                        confidence=min(
                            1.0, confidence / 3
                        ),  # Lower confidence for market neutral
                        z_score=alpha / alpha_std if alpha_std > 0 else 0,
                        spread=0,
                        expected_return=expected_excess_return,
                        risk_score=abs(beta) / 2.0,
                        entry_price=[asset_price, market_price],
                        stop_loss=[
                            asset_price * (1 - 0.04)
                            if signal_type == SignalType.LONG
                            else asset_price * (1 + 0.04),
                            market_price * (1 - 0.02)
                            if signal_type == SignalType.LONG
                            else market_price * (1 + 0.02),
                        ],
                        take_profit=[
                            asset_price * (1 + 0.03)
                            if signal_type == SignalType.LONG
                            else asset_price * (1 - 0.03),
                            market_price * (1 + 0.01)
                            if signal_type == SignalType.SHORT
                            else market_price * (1 - 0.01),
                        ],
                        position_size=[position_size, hedge_size],
                        metadata={
                            "beta": beta,
                            "alpha": alpha,
                            "expected_excess_return": expected_excess_return,
                        },
                    )

                    if signal.confidence >= self.min_confidence:
                        signals.append(signal)

            return signals

        except Exception as e:
            logger.error(f"Market neutral signal generation failed: {e}")
            return []

    async def _generate_correlation_trading_signals(self) -> List[StatisticalSignal]:
        """Generate correlation-based trading signals"""
        signals = []

        try:
            # Find highly correlated pairs
            correlation_matrix = {}

            for asset1 in self.asset_universe:
                for asset2 in self.asset_universe:
                    if asset1 >= asset2:  # Avoid duplicates
                        continue

                    if asset1 not in self.price_data or asset2 not in self.price_data:
                        continue

                    # Calculate rolling correlation
                    prices1 = self.price_data[asset1]["close"]
                    prices2 = self.price_data[asset2]["close"]

                    correlation = prices1.rolling(30).corr(prices2).iloc[-1]

                    if not np.isnan(correlation) and abs(correlation) > 0.8:
                        correlation_matrix[f"{asset1}-{asset2}"] = correlation

                        # Check for correlation divergence
                        recent_correlation = prices1.rolling(10).corr(prices2).iloc[-1]

                        if (
                            abs(recent_correlation - correlation) > 0.2
                        ):  # Correlation breakdown
                            # Calculate price ratio deviation
                            price_ratio = prices1.iloc[-1] / prices2.iloc[-1]
                            ratio_mean = (prices1 / prices2).rolling(30).mean().iloc[-1]
                            ratio_deviation = (price_ratio - ratio_mean) / ratio_mean

                            if abs(ratio_deviation) > 0.02:  # 2% deviation
                                # Generate mean reversion signal
                                if price_ratio > ratio_mean:
                                    signal_type = (
                                        SignalType.SHORT
                                    )  # Short asset1, long asset2
                                    position_sizes = [
                                        -self.max_position_size / 6,
                                        self.max_position_size / 6,
                                    ]
                                else:
                                    signal_type = (
                                        SignalType.LONG
                                    )  # Long asset1, short asset2
                                    position_sizes = [
                                        self.max_position_size / 6,
                                        -self.max_position_size / 6,
                                    ]

                                # Expected return from ratio reversion
                                expected_return = (
                                    abs(price_ratio - ratio_mean) * ratio_mean * 0.5
                                )

                                signal = StatisticalSignal(
                                    strategy=ArbitrageStrategy.CORRELATION_TRADING,
                                    assets=[asset1, asset2],
                                    signal_type=signal_type,
                                    confidence=min(
                                        1.0, abs(ratio_deviation) * 20
                                    ),  # Scale deviation
                                    z_score=ratio_deviation
                                    / 0.01,  # Normalize to 1% = 1 Z-score
                                    spread=price_ratio - ratio_mean,
                                    expected_return=expected_return,
                                    risk_score=abs(recent_correlation - correlation)
                                    / 0.5,
                                    entry_price=[prices1.iloc[-1], prices2.iloc[-1]],
                                    stop_loss=[
                                        prices1.iloc[-1] * (1 + 0.03)
                                        if signal_type == SignalType.SHORT
                                        else prices1.iloc[-1] * (1 - 0.03),
                                        prices2.iloc[-1] * (1 - 0.03)
                                        if signal_type == SignalType.LONG
                                        else prices2.iloc[-1] * (1 + 0.03),
                                    ],
                                    take_profit=[
                                        prices1.iloc[-1] * (1 - 0.02)
                                        if signal_type == SignalType.SHORT
                                        else prices1.iloc[-1] * (1 + 0.02),
                                        prices2.iloc[-1] * (1 + 0.02)
                                        if signal_type == SignalType.LONG
                                        else prices2.iloc[-1] * (1 - 0.02),
                                    ],
                                    position_size=position_sizes,
                                    correlation=correlation,
                                    metadata={
                                        "recent_correlation": recent_correlation,
                                        "historical_correlation": correlation,
                                        "price_ratio": price_ratio,
                                        "ratio_mean": ratio_mean,
                                    },
                                )

                                if signal.confidence >= self.min_confidence:
                                    signals.append(signal)

            return signals

        except Exception as e:
            logger.error(f"Correlation trading signal generation failed: {e}")
            return []

    async def _filter_and_rank_signals(
        self, signals: List[StatisticalSignal]
    ) -> List[StatisticalSignal]:
        """Filter and rank trading signals by quality"""
        try:
            # Filter by minimum confidence
            filtered = [s for s in signals if s.confidence >= self.min_confidence]

            # Filter by risk limits
            filtered = [s for s in filtered if s.risk_score <= 0.8]

            # Rank by composite score
            def score_signal(signal: StatisticalSignal) -> float:
                # Composite score weighting confidence, expected return, and risk
                confidence_weight = 0.4
                return_weight = 0.4
                risk_weight = 0.2

                return (
                    signal.confidence * confidence_weight
                    + abs(signal.expected_return) * return_weight
                    + (1 - signal.risk_score) * risk_weight
                )

            filtered.sort(key=score_signal, reverse=True)

            # Limit to top 10 signals
            return filtered[:10]

        except Exception as e:
            logger.error(f"Signal filtering and ranking failed: {e}")
            return []

    def _generate_pair_combinations(self) -> List[Tuple[str, str]]:
        """Generate all possible asset pairs"""
        combinations = []
        for i, asset1 in enumerate(self.asset_universe):
            for asset2 in self.asset_universe[i + 1 :]:
                combinations.append((asset1, asset2))
        return combinations

    async def _update_market_data(self) -> None:
        """Update market data for analysis"""
        try:
            # This would fetch real-time data from exchanges
            # For now, simulate market data updates

            for asset in self.asset_universe:
                if asset not in self.price_data:
                    self.price_data[asset] = pd.DataFrame()

                # Simulate price movement
                if len(self.price_data[asset]) == 0:
                    base_price = np.random.uniform(100, 10000)
                else:
                    last_price = self.price_data[asset]["close"].iloc[-1]
                    change = np.random.normal(0, 0.01)  # 1% volatility
                    base_price = last_price * (1 + change)

                # Generate OHLCV data
                timestamp = datetime.now()
                open_price = base_price * np.random.uniform(0.99, 1.01)
                high_price = base_price * np.random.uniform(1.0, 1.02)
                low_price = base_price * np.random.uniform(0.98, 1.0)
                close_price = base_price
                volume = np.random.uniform(100000, 1000000)

                new_data = pd.DataFrame(
                    {
                        "timestamp": [timestamp],
                        "open": [open_price],
                        "high": [high_price],
                        "low": [low_price],
                        "close": [close_price],
                        "volume": [volume],
                    }
                )

                self.price_data[asset] = pd.concat(
                    [self.price_data[asset], new_data], ignore_index=True
                )

                # Keep only last 500 data points
                if len(self.price_data[asset]) > 500:
                    self.price_data[asset] = self.price_data[asset].iloc[-500:]

        except Exception as e:
            logger.error(f"Market data update failed: {e}")

    async def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        try:
            # Calculate strategy performance
            strategy_performance = defaultdict(
                lambda: {"signals": 0, "avg_confidence": 0, "avg_return": 0}
            )

            for signal in self.signal_history[-100:]:  # Last 100 signals
                strategy = signal.strategy.value
                strategy_performance[strategy]["signals"] += 1
                strategy_performance[strategy]["avg_confidence"] += signal.confidence
                strategy_performance[strategy]["avg_return"] += signal.expected_return

            # Calculate averages
            for strategy in strategy_performance:
                if strategy_performance[strategy]["signals"] > 0:
                    strategy_performance[strategy]["avg_confidence"] /= (
                        strategy_performance[strategy]["signals"]
                    )
                    strategy_performance[strategy]["avg_return"] /= (
                        strategy_performance[strategy]["signals"]
                    )

            return {
                "active_signals": len(self.active_signals),
                "total_signals_generated": len(self.signal_history),
                "strategy_performance": dict(strategy_performance),
                "top_performing_strategies": sorted(
                    strategy_performance.items(),
                    key=lambda x: x[1]["avg_return"],
                    reverse=True,
                )[:5],
                "last_updated": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Performance report generation failed: {e}")
            return {}


# Utility functions
async def create_statistical_arbitrage(
    config: Dict[str, Any],
) -> AdvancedStatisticalArbitrage:
    """Create and initialize statistical arbitrage system"""
    arb_system = AdvancedStatisticalArbitrage(config)
    await arb_system.initialize()
    return arb_system


def calculate_pairs_hedge_ratio(prices1: pd.Series, prices2: pd.Series) -> float:
    """Calculate optimal hedge ratio for pairs trading"""
    try:
        # Linear regression of prices1 on prices2
        X = sm.add_constant(prices2)
        model = sm.OLS(prices1, X).fit()
        return model.params[1]  # Slope coefficient
    except:
        return 1.0


def calculate_half_life(series: pd.Series) -> float:
    """Calculate half-life of mean reversion"""
    try:
        # Calculate half-life using AR(1) model
        delta_series = series.diff().dropna()
        lagged_series = series.shift(1).dropna()

        # Align series
        min_length = min(len(delta_series), len(lagged_series))
        delta_aligned = delta_series.iloc[-min_length:]
        lagged_aligned = lagged_series.iloc[-min_length:]

        X = sm.add_constant(lagged_aligned)
        model = sm.OLS(delta_aligned, X).fit()

        # Half-life = -ln(2) / ln(1 + coefficient)
        coefficient = model.params[1]
        if coefficient >= 0:
            return float("inf")  # No mean reversion

        half_life = -np.log(2) / np.log(1 + coefficient)
        return half_life

    except:
        return 30.0  # Default 30 days
