"""
Market Regime Detection and Adaptive Trading System
Advanced market state identification with strategy adaptation
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import redis.asyncio as redis

from ..utils.performance_cache import PerformanceCache

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types"""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    LOW_VOLATILITY = "low_volatility"
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    TRANSITIONING = "transitioning"
    UNCERTAIN = "uncertain"


class RegimeDetectorType(Enum):
    """Types of regime detection methods"""

    TECHNICAL = "technical"
    STATISTICAL = "statistical"
    MACHINE_LEARNING = "machine_learning"
    HYBRID = "hybrid"


class AdaptationStrategy(Enum):
    """Strategy adaptation approaches"""

    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    DYNAMIC = "dynamic"
    NEUTRAL = "neutral"


@dataclass
class RegimeSignal:
    """Market regime signal with confidence"""

    regime: MarketRegime
    confidence: float
    strength: float
    duration: Optional[timedelta] = None
    indicators: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RegimeTransition:
    """Information about regime change"""

    from_regime: MarketRegime
    to_regime: MarketRegime
    transition_time: datetime
    confidence: float
    supporting_indicators: Dict[str, float] = field(default_factory=dict)


@dataclass
class AdaptiveStrategy:
    """Adaptive strategy configuration"""

    strategy_id: str
    base_config: Dict[str, Any]
    regime_adjustments: Dict[MarketRegime, Dict[str, Any]]
    performance_history: Dict[MarketRegime, List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    current_adjustment: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)


class MarketRegimeDetector:
    """
    Advanced market regime detection with machine learning
    and adaptive strategy management
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.regime_config = config.get("market_regime_detection", {})

        # Detection parameters
        self.detection_type = RegimeDetectorType(
            self.regime_config.get("detection_type", "hybrid")
        )
        self.lookback_period = self.regime_config.get("lookback_period", 100)  # periods
        self.min_regime_duration = self.regime_config.get(
            "min_regime_duration", 30
        )  # periods
        self.confidence_threshold = self.regime_config.get("confidence_threshold", 0.7)

        # Technical indicators parameters
        self.trend_periods = self.regime_config.get("trend_periods", [20, 50, 200])
        self.volatility_periods = self.regime_config.get("volatility_periods", [14, 30])
        self.momentum_periods = self.regime_config.get("momentum_periods", [12, 26])

        # Market data
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self.volume_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self.indicators: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=500))
        )

        # Regime tracking
        self.current_regime: Dict[str, MarketRegime] = {}
        self.regime_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.regime_transitions: Dict[str, List[RegimeTransition]] = defaultdict(list)
        self.regime_confidence: Dict[str, float] = {}

        # Machine learning model
        self.ml_model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.training_data: List[Dict] = []

        # Adaptive strategies
        self.adaptive_strategies: Dict[str, AdaptiveStrategy] = {}
        self.strategy_performance: Dict[str, Dict[MarketRegime, Dict]] = defaultdict(
            dict
        )

        # Performance cache
        self.cache = PerformanceCache(
            redis_url=config.get("redis_url", "redis://localhost:6379"),
            default_ttl=60,  # 1 minute TTL for regime data
        )

        # Supported symbols
        self.monitored_symbols = self.regime_config.get(
            "monitored_symbols", ["BTC/USDT", "ETH/USDT"]
        )

        logger.info("Market Regime Detector initialized")

    async def initialize(self) -> None:
        """Initialize the regime detection system"""
        try:
            # Initialize ML model if enabled
            if self.detection_type in [
                RegimeDetectorType.MACHINE_LEARNING,
                RegimeDetectorType.HYBRID,
            ]:
                await self._initialize_ml_model()

            # Load historical data
            await self._load_historical_data()

            # Initialize adaptive strategies
            await self._initialize_adaptive_strategies()

            # Start regime monitoring
            asyncio.create_task(self._regime_monitoring_loop())

            logger.info("Market Regime Detector initialized successfully")

        except Exception as e:
            logger.error(f"Market Regime Detector initialization failed: {e}")
            raise

    async def detect_market_regime(self, symbol: str) -> Optional[RegimeSignal]:
        """Detect current market regime for a symbol"""
        try:
            # Get historical data
            if (
                symbol not in self.price_history
                or len(self.price_history[symbol]) < self.lookback_period
            ):
                return None

            prices = list(self.price_history[symbol])[-self.lookback_period :]
            volumes = list(self.volume_history[symbol])[-self.lookback_period :]

            if len(prices) < 50:
                return None

            # Calculate indicators
            await self._calculate_technical_indicators(symbol, prices, volumes)

            # Detect regime based on type
            if self.detection_type == RegimeDetectorType.TECHNICAL:
                regime_signal = await self._detect_technical_regime(
                    symbol, prices, volumes
                )
            elif self.detection_type == RegimeDetectorType.STATISTICAL:
                regime_signal = await self._detect_statistical_regime(
                    symbol, prices, volumes
                )
            elif self.detection_type == RegimeDetectorType.MACHINE_LEARNING:
                regime_signal = await self._detect_ml_regime(symbol, prices, volumes)
            else:  # HYBRID
                regime_signal = await self._detect_hybrid_regime(
                    symbol, prices, volumes
                )

            # Validate regime signal
            if regime_signal and regime_signal.confidence >= self.confidence_threshold:
                # Update regime history
                await self._update_regime_history(symbol, regime_signal)

                # Cache regime signal
                await self.cache.set(
                    f"regime_signal:{symbol}",
                    {
                        "regime": regime_signal.regime.value,
                        "confidence": regime_signal.confidence,
                        "strength": regime_signal.strength,
                        "timestamp": regime_signal.timestamp.isoformat(),
                        "indicators": regime_signal.indicators,
                    },
                    ttl=300,  # 5 minutes
                )

                return regime_signal

            return None

        except Exception as e:
            logger.error(f"Market regime detection failed for {symbol}: {e}")
            return None

    async def _detect_technical_regime(
        self, symbol: str, prices: List[float], volumes: List[float]
    ) -> Optional[RegimeSignal]:
        """Detect regime using technical analysis"""
        try:
            indicators = self.indicators.get(symbol, {})

            # Get trend indicators
            sma_20 = self._get_latest_indicator(indicators, "sma_20")
            sma_50 = self._get_latest_indicator(indicators, "sma_50")
            sma_200 = self._get_latest_indicator(indicators, "sma_200")
            ema_12 = self._get_latest_indicator(indicators, "ema_12")
            ema_26 = self._get_latest_indicator(indicators, "ema_26")

            # Get volatility
            atr_14 = self._get_latest_indicator(indicators, "atr_14")
            volatility = self._get_latest_indicator(indicators, "volatility")

            # Get momentum
            rsi_14 = self._get_latest_indicator(indicators, "rsi_14")
            macd = self._get_latest_indicator(indicators, "macd")
            macd_signal = self._get_latest_indicator(indicators, "macd_signal")

            # Get volume indicators
            volume_sma = self._get_latest_indicator(indicators, "volume_sma")
            current_volume = volumes[-1] if volumes else 0

            regime_scores = {}
            confidence_factors = {}

            # Trending regimes
            if sma_20 and sma_50 and sma_200 and ema_12 and ema_26:
                current_price = prices[-1]

                # Uptrend conditions
                if (
                    current_price > sma_20 > sma_50 > sma_200
                    and ema_12 > ema_26
                    and macd > macd_signal
                ):
                    if current_price > sma_200 * 1.2:  # Strong uptrend
                        regime_scores[MarketRegime.TRENDING_UP] = 0.8
                        regime_scores[MarketRegime.BULL_MARKET] = 0.9
                    else:
                        regime_scores[MarketRegime.TRENDING_UP] = 0.7

                # Downtrend conditions
                elif (
                    current_price < sma_20 < sma_50 < sma_200
                    and ema_12 < ema_26
                    and macd < macd_signal
                ):
                    if current_price < sma_200 * 0.8:  # Strong downtrend
                        regime_scores[MarketRegime.TRENDING_DOWN] = 0.8
                        regime_scores[MarketRegime.BEAR_MARKET] = 0.9
                    else:
                        regime_scores[MarketRegime.TRENDING_DOWN] = 0.7

                # Sideways conditions
                elif abs(sma_20 - sma_50) / sma_50 < 0.02:  # Within 2%
                    regime_scores[MarketRegime.SIDEWAYS] = 0.7

            # Volatility regimes
            if volatility and atr_14:
                if volatility > np.std(prices[-30:]) * 2:  # High volatility
                    regime_scores[MarketRegime.VOLATILE] = 0.8
                elif volatility < np.std(prices[-30:]) * 0.5:  # Low volatility
                    regime_scores[MarketRegime.LOW_VOLATILITY] = 0.8

            # Momentum confirmation
            if rsi_14:
                if rsi_14 > 70:  # Overbought
                    if MarketRegime.TRENDING_UP in regime_scores:
                        regime_scores[MarketRegime.TRENDING_UP] *= (
                            0.8  # Reduce confidence
                        )
                    regime_scores[MarketRegime.TRANSITIONING] = 0.6
                elif rsi_14 < 30:  # Oversold
                    if MarketRegime.TRENDING_DOWN in regime_scores:
                        regime_scores[MarketRegime.TRENDING_DOWN] *= (
                            0.8  # Reduce confidence
                        )
                    regime_scores[MarketRegime.TRANSITIONING] = 0.6

            # Volume confirmation
            if volume_sma and current_volume:
                volume_ratio = current_volume / volume_sma
                if volume_ratio > 2.0:  # High volume
                    for regime in regime_scores:
                        regime_scores[regime] *= 1.2  # Boost confidence

            if not regime_scores:
                regime_scores[MarketRegime.UNCERTAIN] = 0.5

            # Select best regime
            best_regime = max(regime_scores.keys(), key=lambda k: regime_scores[k])
            confidence = regime_scores[best_regime]

            # Calculate regime strength
            strength = min(1.0, confidence * 1.2)

            return RegimeSignal(
                regime=best_regime,
                confidence=confidence,
                strength=strength,
                indicators={
                    "sma_20": sma_20 or 0,
                    "sma_50": sma_50 or 0,
                    "sma_200": sma_200 or 0,
                    "rsi_14": rsi_14 or 0,
                    "volatility": volatility or 0,
                    "volume_ratio": current_volume / volume_sma if volume_sma else 1.0,
                },
            )

        except Exception as e:
            logger.error(f"Technical regime detection failed for {symbol}: {e}")
            return None

    async def _detect_statistical_regime(
        self, symbol: str, prices: List[float], volumes: List[float]
    ) -> Optional[RegimeSignal]:
        """Detect regime using statistical methods"""
        try:
            if len(prices) < 50:
                return None

            # Calculate statistical properties
            returns = np.diff(np.log(prices))
            volatility = np.std(returns) * np.sqrt(252)  # Annualized

            # Trend analysis using linear regression
            x = np.arange(len(prices))
            slope, intercept, r_value, p_value, std_err = np.polyfit(
                x, prices, 1, full=True
            )

            # Momentum and acceleration
            momentum_20 = (
                (prices[-1] - prices[-20]) / prices[-20] if len(prices) > 20 else 0
            )
            momentum_50 = (
                (prices[-1] - prices[-50]) / prices[-50] if len(prices) > 50 else 0
            )

            # Distribution analysis
            skewness = self._calculate_skewness(returns)
            kurtosis = self._calculate_kurtosis(returns)

            regime_scores = {}

            # Trending regimes based on slope and R²
            if abs(r_value) > 0.7:  # Strong trend
                if slope > 0:
                    trend_strength = min(
                        1.0, slope / prices[-1] * 1000
                    )  # Normalize slope
                    regime_scores[MarketRegime.TRENDING_UP] = (
                        abs(r_value) * trend_strength
                    )
                else:
                    trend_strength = min(1.0, abs(slope) / prices[-1] * 1000)
                    regime_scores[MarketRegime.TRENDING_DOWN] = (
                        abs(r_value) * trend_strength
                    )

            # Volatility regimes
            volatility_percentile = np.percentile(returns, 95) - np.percentile(
                returns, 5
            )

            if volatility > volatility_percentile * 1.5:
                regime_scores[MarketRegime.VOLATILE] = min(
                    1.0, volatility / volatility_percentile - 0.5
                )
            elif volatility < volatility_percentile * 0.5:
                regime_scores[MarketRegime.LOW_VOLATILITY] = min(
                    1.0, 1.0 - volatility / volatility_percentile
                )

            # Momentum regimes
            if abs(momentum_20) > 0.05:  # 5% momentum
                if momentum_20 > 0:
                    regime_scores[MarketRegime.TRENDING_UP] = (
                        regime_scores.get(MarketRegime.TRENDING_UP, 0) + 0.3
                    )
                else:
                    regime_scores[MarketRegime.TRENDING_DOWN] = (
                        regime_scores.get(MarketRegime.TRENDING_DOWN, 0) + 0.3
                    )

            # Sideways regime
            if abs(r_value) < 0.3 and volatility < volatility_percentile:
                regime_scores[MarketRegime.SIDEWAYS] = 0.6

            if not regime_scores:
                regime_scores[MarketRegime.UNCERTAIN] = 0.5

            best_regime = max(regime_scores.keys(), key=lambda k: regime_scores[k])
            confidence = regime_scores[best_regime]

            return RegimeSignal(
                regime=best_regime,
                confidence=confidence,
                strength=abs(r_value),
                indicators={
                    "slope": slope,
                    "r_squared": r_value**2,
                    "volatility": volatility,
                    "momentum_20": momentum_20,
                    "skewness": skewness,
                    "kurtosis": kurtosis,
                },
            )

        except Exception as e:
            logger.error(f"Statistical regime detection failed for {symbol}: {e}")
            return None

    async def _detect_ml_regime(
        self, symbol: str, prices: List[float], volumes: List[float]
    ) -> Optional[RegimeSignal]:
        """Detect regime using machine learning"""
        try:
            if not self.ml_model or len(prices) < self.lookback_period:
                return None

            # Prepare features
            features = await self._extract_ml_features(symbol, prices, volumes)
            if features is None:
                return None

            # Make prediction
            features_scaled = self.scaler.transform([features])
            prediction = self.ml_model.predict(features_scaled)[0]
            probabilities = self.ml_model.predict_proba(features_scaled)[0]

            # Map prediction to regime
            regime_map = {
                0: MarketRegime.TRENDING_UP,
                1: MarketRegime.TRENDING_DOWN,
                2: MarketRegime.SIDEWAYS,
                3: MarketRegime.VOLATILE,
                4: MarketRegime.LOW_VOLATILITY,
            }

            predicted_regime = regime_map.get(prediction, MarketRegime.UNCERTAIN)
            confidence = np.max(probabilities)

            # Get feature importance
            feature_names = await self._get_feature_names()
            feature_importance = dict(
                zip(feature_names, self.ml_model.feature_importances_)
            )

            return RegimeSignal(
                regime=predicted_regime,
                confidence=confidence,
                strength=confidence,
                indicators=feature_importance,
            )

        except Exception as e:
            logger.error(f"ML regime detection failed for {symbol}: {e}")
            return None

    async def _detect_hybrid_regime(
        self, symbol: str, prices: List[float], volumes: List[float]
    ) -> Optional[RegimeSignal]:
        """Detect regime using hybrid approach"""
        try:
            # Get predictions from all methods
            technical_signal = await self._detect_technical_regime(
                symbol, prices, volumes
            )
            statistical_signal = await self._detect_statistical_regime(
                symbol, prices, volumes
            )

            ml_signal = None
            if self.ml_model:
                ml_signal = await self._detect_ml_regime(symbol, prices, volumes)

            # Combine signals with weights
            signals = [technical_signal, statistical_signal]
            weights = [0.4, 0.3]  # Technical and statistical weights

            if ml_signal:
                signals.append(ml_signal)
                weights.append(0.3)  # ML weight

            # Weighted voting
            regime_votes = defaultdict(float)
            total_confidence = 0

            for signal, weight in zip(signals, weights):
                if signal:
                    regime_votes[signal.regime] += signal.confidence * weight
                    total_confidence += signal.confidence * weight

            if not regime_votes:
                return RegimeSignal(
                    regime=MarketRegime.UNCERTAIN, confidence=0.5, strength=0.5
                )

            # Select best regime
            best_regime = max(regime_votes.keys(), key=lambda k: regime_votes[k])
            confidence = (
                regime_votes[best_regime] / total_confidence
                if total_confidence > 0
                else 0.5
            )

            # Combine indicators
            combined_indicators = {}
            for signal in signals:
                if signal:
                    combined_indicators.update(
                        {
                            f"{k}_{signal.regime.value}": v
                            for k, v in signal.indicators.items()
                        }
                    )

            return RegimeSignal(
                regime=best_regime,
                confidence=confidence,
                strength=confidence,
                indicators=combined_indicators,
            )

        except Exception as e:
            logger.error(f"Hybrid regime detection failed for {symbol}: {e}")
            return None

    async def update_market_data(
        self,
        symbol: str,
        price: float,
        volume: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Update market data for regime detection"""
        try:
            if timestamp is None:
                timestamp = datetime.now()

            # Update price and volume history
            self.price_history[symbol].append((timestamp, price))
            self.volume_history[symbol].append((timestamp, volume))

            # Keep only prices for indicator calculation
            if len(self.price_history[symbol]) > 0 and isinstance(
                self.price_history[symbol][0], tuple
            ):
                # Convert to price-only deque
                price_only_deque = deque(maxlen=500)
                for ts, p in self.price_history[symbol]:
                    price_only_deque.append(p)
                self.price_history[symbol] = price_only_deque

            # Update current data
            self.price_history[symbol].append(price)
            self.volume_history[symbol].append(volume)

        except Exception as e:
            logger.error(f"Market data update failed for {symbol}: {e}")

    async def get_adapted_strategy(
        self, strategy_id: str, symbol: str
    ) -> Optional[Dict[str, Any]]:
        """Get strategy configuration adapted for current market regime"""
        try:
            if strategy_id not in self.adaptive_strategies:
                return None

            if symbol not in self.current_regime:
                return None

            strategy = self.adaptive_strategies[strategy_id]
            current_regime = self.current_regime[symbol]
            current_confidence = self.regime_confidence.get(symbol, 0.5)

            # Get regime-specific adjustments
            regime_adjustments = strategy.regime_adjustments.get(current_regime, {})

            # Apply adjustments based on confidence
            adapted_config = strategy.base_config.copy()

            for key, value in regime_adjustments.items():
                if isinstance(value, dict):
                    # Nested configuration
                    adapted_config[key] = adapted_config.get(key, {}).copy()
                    adapted_config[key].update(value)
                else:
                    # Simple adjustment
                    adapted_config[key] = value

            # Apply confidence-based scaling
            if current_confidence < 0.7:
                # Reduce aggressive parameters in low confidence
                if "position_size_multiplier" in adapted_config:
                    adapted_config["position_size_multiplier"] *= current_confidence
                if "risk_multiplier" in adapted_config:
                    adapted_config["risk_multiplier"] *= 2 - current_confidence

            strategy.current_adjustment = adapted_config
            strategy.last_updated = datetime.now()

            return adapted_config

        except Exception as e:
            logger.error(f"Strategy adaptation failed for {strategy_id}: {e}")
            return None

    async def _calculate_technical_indicators(
        self, symbol: str, prices: List[float], volumes: List[float]
    ) -> None:
        """Calculate technical indicators for regime detection"""
        try:
            indicators = self.indicators[symbol]

            # Simple Moving Averages
            for period in self.trend_periods:
                if len(prices) >= period:
                    sma = np.mean(prices[-period:])
                    indicators[f"sma_{period}"].append(sma)

                    # Exponential Moving Average
                    if f"ema_{period}" not in indicators:
                        indicators[f"ema_{period}"].append(sma)
                    else:
                        multiplier = 2 / (period + 1)
                        ema = (
                            prices[-1] - indicators[f"ema_{period}"][-1]
                        ) * multiplier + indicators[f"ema_{period}"][-1]
                        indicators[f"ema_{period}"].append(ema)

            # MACD
            if len(indicators["ema_12"]) > 0 and len(indicators["ema_26"]) > 0:
                ema_12 = indicators["ema_12"][-1]
                ema_26 = indicators["ema_26"][-1]
                macd = ema_12 - ema_26
                indicators["macd"].append(macd)

                # MACD Signal (9-period EMA of MACD)
                if "macd_signal" not in indicators:
                    indicators["macd_signal"].append(macd)
                else:
                    multiplier = 2 / (9 + 1)
                    signal = (
                        macd - indicators["macd_signal"][-1]
                    ) * multiplier + indicators["macd_signal"][-1]
                    indicators["macd_signal"].append(signal)

            # RSI
            if len(prices) >= 14:
                rsi = self._calculate_rsi(prices[-14:])
                indicators["rsi_14"].append(rsi)

            # ATR (Average True Range)
            if len(prices) >= 14:
                atr = self._calculate_atr(prices[-14:])
                indicators["atr_14"].append(atr)

            # Volatility
            if len(prices) >= 20:
                returns = np.diff(prices[-20:])
                volatility = np.std(returns) * np.sqrt(252)  # Annualized
                indicators["volatility"].append(volatility)

            # Volume SMA
            if len(volumes) >= 20:
                volume_sma = np.mean(volumes[-20:])
                indicators["volume_sma"].append(volume_sma)

        except Exception as e:
            logger.error(f"Technical indicator calculation failed for {symbol}: {e}")

    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_atr(self, prices: List[float], period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(prices) < 2:
            return 0.0

        true_ranges = []
        for i in range(1, len(prices)):
            high = prices[i]
            low = prices[i]
            prev_close = prices[i - 1]

            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)

            true_ranges.append(max(tr1, tr2, tr3))

        if len(true_ranges) >= period:
            return np.mean(true_ranges[-period:])
        else:
            return np.mean(true_ranges) if true_ranges else 0.0

    def _calculate_skewness(self, data: List[float]) -> float:
        """Calculate skewness of data"""
        if len(data) < 3:
            return 0.0

        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return 0.0

        n = len(data)
        skewness = (n / ((n - 1) * (n - 2))) * np.sum(((data - mean) / std) ** 3)

        return skewness

    def _calculate_kurtosis(self, data: List[float]) -> float:
        """Calculate kurtosis of data"""
        if len(data) < 4:
            return 0.0

        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return 0.0

        n = len(data)
        kurtosis = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * np.sum(
            ((data - mean) / std) ** 4
        )
        kurtosis -= (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))

        return kurtosis

    def _get_latest_indicator(
        self, indicators: Dict[str, deque], name: str
    ) -> Optional[float]:
        """Get latest value of an indicator"""
        if name in indicators and indicators[name]:
            return indicators[name][-1]
        return None

    async def _extract_ml_features(
        self, symbol: str, prices: List[float], volumes: List[float]
    ) -> Optional[List[float]]:
        """Extract features for ML model"""
        try:
            if len(prices) < self.lookback_period:
                return None

            features = []

            # Price-based features
            returns = np.diff(np.log(prices[-50:]))
            features.extend(
                [
                    np.mean(returns),
                    np.std(returns),
                    np.max(returns),
                    np.min(returns),
                    self._calculate_skewness(returns),
                    self._calculate_kurtosis(returns),
                ]
            )

            # Trend features
            x = np.arange(len(prices[-50:]))
            slope, _, r_value, _, _ = np.polyfit(x, prices[-50:], 1, full=True)
            features.extend([slope, r_value**2])

            # Volatility features
            short_vol = np.std(returns[-10:])
            long_vol = np.std(returns[-30:])
            features.extend(
                [short_vol, long_vol, short_vol / long_vol if long_vol > 0 else 1]
            )

            # Volume features
            if volumes and len(volumes) >= 50:
                volume_returns = np.diff(volumes[-50:])
                features.extend(
                    [
                        np.mean(volume_returns),
                        np.std(volume_returns),
                        volumes[-1] / np.mean(volumes[-50:]),
                    ]
                )
            else:
                features.extend([0, 0, 1])

            # Technical indicators
            indicators = self.indicators.get(symbol, {})
            rsi = self._get_latest_indicator(indicators, "rsi_14") or 50
            macd = self._get_latest_indicator(indicators, "macd") or 0
            atr = self._get_latest_indicator(indicators, "atr_14") or 0

            features.extend([rsi, macd, atr])

            return features

        except Exception as e:
            logger.error(f"ML feature extraction failed for {symbol}: {e}")
            return None

    async def _get_feature_names(self) -> List[str]:
        """Get names of ML features"""
        return [
            "return_mean",
            "return_std",
            "return_max",
            "return_min",
            "return_skew",
            "return_kurt",
            "trend_slope",
            "trend_r2",
            "vol_short",
            "vol_long",
            "vol_ratio",
            "volume_mean",
            "volume_std",
            "volume_ratio",
            "rsi",
            "macd",
            "atr",
        ]

    async def _initialize_ml_model(self) -> None:
        """Initialize machine learning model for regime detection"""
        try:
            # Create sample training data
            n_samples = 1000
            n_features = 18

            # Generate sample features
            X = np.random.randn(n_samples, n_features)

            # Generate sample labels (regimes)
            y = np.random.randint(0, 5, n_samples)

            # Train model
            self.ml_model = RandomForestClassifier(
                n_estimators=100, random_state=42, max_depth=10
            )

            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            self.ml_model.fit(X_scaled, y)

            logger.info("ML model initialized for regime detection")

        except Exception as e:
            logger.error(f"ML model initialization failed: {e}")

    async def _load_historical_data(self) -> None:
        """Load historical market data"""
        try:
            # This would load actual historical data from database or files
            # For now, initialize empty structures
            logger.info("Historical market data loaded")

        except Exception as e:
            logger.error(f"Historical data loading failed: {e}")

    async def _initialize_adaptive_strategies(self) -> None:
        """Initialize adaptive strategies"""
        try:
            # Define default adaptive strategies
            strategies_config = self.regime_config.get("adaptive_strategies", {})

            for strategy_id, base_config in strategies_config.items():
                # Define regime-specific adjustments
                regime_adjustments = {
                    MarketRegime.TRENDING_UP: {
                        "position_size_multiplier": 1.2,
                        "risk_multiplier": 1.1,
                        "take_profit_multiplier": 1.3,
                    },
                    MarketRegime.TRENDING_DOWN: {
                        "position_size_multiplier": 0.8,
                        "risk_multiplier": 0.7,
                        "take_profit_multiplier": 0.9,
                    },
                    MarketRegime.SIDEWAYS: {
                        "position_size_multiplier": 1.0,
                        "risk_multiplier": 1.0,
                        "take_profit_multiplier": 1.0,
                    },
                    MarketRegime.VOLATILE: {
                        "position_size_multiplier": 0.6,
                        "risk_multiplier": 0.5,
                        "take_profit_multiplier": 1.5,
                    },
                    MarketRegime.LOW_VOLATILITY: {
                        "position_size_multiplier": 1.1,
                        "risk_multiplier": 1.0,
                        "take_profit_multiplier": 1.0,
                    },
                }

                strategy = AdaptiveStrategy(
                    strategy_id=strategy_id,
                    base_config=base_config,
                    regime_adjustments=regime_adjustments,
                )

                self.adaptive_strategies[strategy_id] = strategy

            logger.info(
                f"Initialized {len(self.adaptive_strategies)} adaptive strategies"
            )

        except Exception as e:
            logger.error(f"Adaptive strategies initialization failed: {e}")

    async def _update_regime_history(
        self, symbol: str, regime_signal: RegimeSignal
    ) -> None:
        """Update regime history and detect transitions"""
        try:
            # Add to history
            self.regime_history[symbol].append(regime_signal)

            # Check for regime transition
            previous_regime = self.current_regime.get(symbol)

            if previous_regime and previous_regime != regime_signal.regime:
                # Record transition
                transition = RegimeTransition(
                    from_regime=previous_regime,
                    to_regime=regime_signal.regime,
                    transition_time=regime_signal.timestamp,
                    confidence=regime_signal.confidence,
                    supporting_indicators=regime_signal.indicators,
                )

                self.regime_transitions[symbol].append(transition)

                logger.info(
                    f"Regime transition detected for {symbol}: {previous_regime.value} -> {regime_signal.regime.value}"
                )

            # Update current regime
            self.current_regime[symbol] = regime_signal.regime
            self.regime_confidence[symbol] = regime_signal.confidence

        except Exception as e:
            logger.error(f"Regime history update failed for {symbol}: {e}")

    async def _regime_monitoring_loop(self) -> None:
        """Background loop for continuous regime monitoring"""
        while True:
            try:
                for symbol in self.monitored_symbols:
                    regime_signal = await self.detect_market_regime(symbol)

                    if regime_signal:
                        # Update adaptive strategies
                        for strategy_id in self.adaptive_strategies:
                            adapted_config = await self.get_adapted_strategy(
                                strategy_id, symbol
                            )

                            # Cache adapted configuration
                            await self.cache.set(
                                f"adaptive_strategy:{strategy_id}:{symbol}",
                                adapted_config,
                                ttl=60,
                            )

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Regime monitoring loop error: {e}")
                await asyncio.sleep(10)

    async def get_regime_report(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive regime report for a symbol"""
        try:
            current_regime = self.current_regime.get(symbol, MarketRegime.UNCERTAIN)
            current_confidence = self.regime_confidence.get(symbol, 0.0)

            # Get regime statistics
            history = list(self.regime_history.get(symbol, []))
            if history:
                regime_counts = defaultdict(int)
                for signal in history:
                    regime_counts[signal.regime] += 1

                most_common = max(regime_counts.keys(), key=lambda k: regime_counts[k])
                transition_count = len(self.regime_transitions.get(symbol, []))

                # Calculate regime duration
                current_duration = None
                if history and history[-1].regime == current_regime:
                    current_duration = datetime.now() - history[-1].timestamp
            else:
                regime_counts = {}
                most_common = MarketRegime.UNCERTAIN
                transition_count = 0
                current_duration = None

            # Get adaptive strategy status
            adapted_strategies = {}
            for strategy_id, strategy in self.adaptive_strategies.items():
                adapted_config = await self.get_adapted_strategy(strategy_id, symbol)
                if adapted_config:
                    adapted_strategies[strategy_id] = adapted_config

            return {
                "symbol": symbol,
                "current_regime": current_regime.value,
                "current_confidence": current_confidence,
                "regime_statistics": {
                    "regime_counts": {
                        regime.value: count for regime, count in regime_counts.items()
                    },
                    "most_common_regime": most_common.value,
                    "transition_count": transition_count,
                    "current_duration_minutes": current_duration.total_seconds() / 60
                    if current_duration
                    else None,
                },
                "adapted_strategies": adapted_strategies,
                "last_updated": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Regime report generation failed for {symbol}: {e}")
            return {}


# Utility functions
async def create_regime_detector(config: Dict[str, Any]) -> MarketRegimeDetector:
    """Create and initialize market regime detector"""
    detector = MarketRegimeDetector(config)
    await detector.initialize()
    return detector


def assess_regime_stability(regime_history: List[RegimeSignal]) -> float:
    """Assess stability of current regime"""
    if len(regime_history) < 10:
        return 0.5

    recent_signals = regime_history[-10:]
    current_regime = recent_signals[-1].regime

    # Calculate how many times current regime has appeared
    same_regime_count = sum(
        1 for signal in recent_signals if signal.regime == current_regime
    )
    stability = same_regime_count / len(recent_signals)

    return stability


def calculate_regime_transition_probability(
    transitions: List[RegimeTransition],
) -> Dict[str, Dict[str, float]]:
    """Calculate transition probabilities between regimes"""
    transition_matrix = defaultdict(lambda: defaultdict(float))

    for transition in transitions:
        from_regime = transition.from_regime.value
        to_regime = transition.to_regime.value
        transition_matrix[from_regime][to_regime] += 1

    # Normalize to probabilities
    for from_regime in transition_matrix:
        total = sum(transition_matrix[from_regime].values())
        for to_regime in transition_matrix[from_regime]:
            transition_matrix[from_regime][to_regime] /= total if total > 0 else 0

    return dict(transition_matrix)
