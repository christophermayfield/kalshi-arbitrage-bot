"""
Enhanced Circuit Breaker System with Predictive Triggers
Advanced protection against market anomalies and system failures
"""
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from collections import deque, defaultdict
import redis.asyncio as redis
from scipy import stats

from ..utils.performance_cache import PerformanceCache

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Trading halted
    HALF_OPEN = "half_open"  # Testing recovery

class TriggerType(Enum):
    """Types of circuit breaker triggers"""
    PRICE_VOLATILITY = "price_volatility"
    VOLUME_ANOMALY = "volume_anomaly"
    SYSTEM_FAILURE = "system_failure"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    CORRELATION_SPIKE = "correlation_spike"
    LATENCY_SPIKE = "latency_spike"
    ERROR_RATE = "error_rate"
    DRAWDOWN_LIMIT = "drawdown_limit"
    PREDICTIVE_RISK = "predictive_risk"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    # Basic thresholds
    price_volatility_threshold: float = 0.05      # 5% price change
    volume_anomaly_threshold: float = 3.0         # 3x normal volume
    error_rate_threshold: float = 0.1             # 10% error rate
    latency_threshold: float = 5.0                 # 5 seconds
    max_drawdown_threshold: float = 0.1           # 10% drawdown
    
    # Time windows
    short_window: int = 60                         # 1 minute
    medium_window: int = 300                       # 5 minutes
    long_window: int = 900                         # 15 minutes
    
    # Recovery settings
    recovery_timeout: int = 300                    # 5 minutes
    test_requests: int = 5                         # Test requests in half-open
    success_threshold: float = 0.8                 # 80% success for recovery
    
    # Predictive settings
    enable_predictive: bool = True
    prediction_window: int = 120                   # 2 minutes
    confidence_threshold: float = 0.7
    
    # Multi-level thresholds
    warning_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'price_volatility': 0.03,
        'volume_anomaly': 2.0,
        'error_rate': 0.05,
        'latency': 2.0
    })
    
    critical_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'price_volatility': 0.08,
        'volume_anomaly': 5.0,
        'error_rate': 0.15,
        'latency': 10.0
    })

@dataclass
class TriggerEvent:
    """Circuit breaker trigger event"""
    trigger_type: TriggerType
    severity: AlertSeverity
    symbol: Optional[str]
    value: float
    threshold: float
    timestamp: datetime
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CircuitMetrics:
    """Circuit breaker performance metrics"""
    total_triggers: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    average_downtime: float = 0.0
    total_downtime: float = 0.0
    last_triggered: Optional[datetime] = None
    trigger_history: deque = field(default_factory=lambda: deque(maxlen=1000))

class PredictiveModel:
    """Simple predictive model for circuit breaker"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.price_history: deque = deque(maxlen=window_size)
        self.volume_history: deque = deque(maxlen=window_size)
        self.latency_history: deque = deque(maxlen=window_size)
    
    def update(self, price: float, volume: float, latency: float) -> None:
        """Update model with new data"""
        self.price_history.append(price)
        self.volume_history.append(volume)
        self.latency_history.append(latency)
    
    def predict_volatility_spike(self, confidence_threshold: float) -> Tuple[bool, float]:
        """Predict if volatility spike is likely"""
        if len(self.price_history) < 20:
            return False, 0.0
        
        prices = list(self.price_history)
        recent_prices = prices[-10:]
        older_prices = prices[-20:-10]
        
        # Calculate recent vs older volatility
        recent_vol = np.std(np.diff(np.log(recent_prices))) if len(recent_prices) > 1 else 0
        older_vol = np.std(np.diff(np.log(older_prices))) if len(older_prices) > 1 else 0
        
        if older_vol == 0:
            return False, 0.0
        
        vol_ratio = recent_vol / older_vol
        confidence = min(1.0, vol_ratio / 2.0)  # Normalize
        
        return vol_ratio > 1.5 and confidence > confidence_threshold, confidence
    
    def predict_volume_anomaly(self, confidence_threshold: float) -> Tuple[bool, float]:
        """Predict if volume anomaly is likely"""
        if len(self.volume_history) < 20:
            return False, 0.0
        
        volumes = list(self.volume_history)
        recent_avg = np.mean(volumes[-5:])
        historical_avg = np.mean(volumes[-20:-5])
        
        if historical_avg == 0:
            return False, 0.0
        
        volume_ratio = recent_avg / historical_avg
        confidence = min(1.0, volume_ratio / 3.0)  # Normalize
        
        return volume_ratio > 2.0 and confidence > confidence_threshold, confidence
    
    def predict_latency_spike(self, confidence_threshold: float) -> Tuple[bool, float]:
        """Predict if latency spike is likely"""
        if len(self.latency_history) < 20:
            return False, 0.0
        
        latencies = list(self.latency_history)
        recent_trend = np.polyfit(range(len(latencies[-5:])), latencies[-5:], 1)[0]
        recent_avg = np.mean(latencies[-5:])
        
        # Predict if latency is trending up and above average
        confidence = min(1.0, recent_trend / 0.1)  # Normalize by expected trend
        
        return recent_trend > 0.05 and recent_avg > 2.0 and confidence > confidence_threshold, confidence

class EnhancedCircuitBreaker:
    """
    Enhanced circuit breaker with predictive triggers and multi-level protection
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.breaker_config = CircuitBreakerConfig(**config.get('circuit_breaker', {}))
        
        # Circuit breaker state
        self.state = CircuitState.CLOSED
        self.state_change_time = datetime.now()
        self.triggered_circuits: Set[str] = set()
        
        # Metrics tracking
        self.metrics = CircuitMetrics()
        self.active_triggers: List[TriggerEvent] = []
        self.trigger_history: deque = deque(maxlen=1000)
        
        # Data tracking
        self.price_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.volume_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.latency_data: deque = deque(maxlen=1000))
        self.error_data: deque = deque(maxlen=1000))
        
        # Predictive models
        self.predictive_models: Dict[str, PredictiveModel] = {}
        
        # Performance cache
        self.cache = PerformanceCache(
            redis_url=config.get('redis_url', 'redis://localhost:6379'),
            default_ttl=60
        )
        
        # Circuit-specific configurations
        self.circuit_configs: Dict[str, CircuitBreakerConfig] = {}
        
        logger.info("Enhanced Circuit Breaker initialized with predictive capabilities")
    
    async def initialize(self) -> None:
        """Initialize circuit breaker with existing data"""
        try:
            # Load state from cache
            await self._load_circuit_state()
            
            # Initialize predictive models for known symbols
            known_symbols = self.config.get('monitored_symbols', [])
            for symbol in known_symbols:
                self.predictive_models[symbol] = PredictiveModel()
            
            logger.info(f"Circuit Breaker initialized with {len(known_symbols)} monitored symbols")
            
        except Exception as e:
            logger.error(f"Circuit Breaker initialization failed: {e}")
            raise
    
    async def check_triggers(self, symbol: str, price: float, volume: float, 
                           latency: float = 0.0, error_occurred: bool = False) -> List[TriggerEvent]:
        """
        Check all circuit breaker triggers for the given data
        
        Returns:
            List of triggered events
        """
        triggers = []
        
        try:
            # Skip checks if circuit is open and not in recovery
            if self.state == CircuitState.OPEN:
                if datetime.now() - self.state_change_time < timedelta(seconds=self.breaker_config.recovery_timeout):
                    return []
                else:
                    # Try to transition to half-open
                    await self._attempt_recovery()
            
            # Update data tracking
            await self._update_data(symbol, price, volume, latency, error_occurred)
            
            # Update predictive model
            if symbol not in self.predictive_models:
                self.predictive_models[symbol] = PredictiveModel()
            self.predictive_models[symbol].update(price, volume, latency)
            
            # Check predictive triggers
            if self.breaker_config.enable_predictive:
                predictive_triggers = await self._check_predictive_triggers(symbol)
                triggers.extend(predictive_triggers)
            
            # Check traditional triggers
            traditional_triggers = await self._check_traditional_triggers(symbol)
            triggers.extend(traditional_triggers)
            
            # Process triggers
            if triggers:
                await self._process_triggers(triggers)
            
            return triggers
            
        except Exception as e:
            logger.error(f"Trigger check failed for {symbol}: {e}")
            return []
    
    async def _check_predictive_triggers(self, symbol: str) -> List[TriggerEvent]:
        """Check predictive triggers"""
        triggers = []
        
        try:
            model = self.predictive_models.get(symbol)
            if not model:
                return triggers
            
            # Predict volatility spike
            is_likely, confidence = model.predict_volatility_spike(
                self.breaker_config.confidence_threshold
            )
            if is_likely:
                triggers.append(TriggerEvent(
                    trigger_type=TriggerType.PREDICTIVE_RISK,
                    severity=AlertSeverity.WARNING,
                    symbol=symbol,
                    value=confidence,
                    threshold=self.breaker_config.confidence_threshold,
                    timestamp=datetime.now(),
                    message=f"Predictive: Volatility spike likely (confidence: {confidence:.2f})",
                    metadata={'prediction_type': 'volatility_spike'}
                ))
            
            # Predict volume anomaly
            is_likely, confidence = model.predict_volume_anomaly(
                self.breaker_config.confidence_threshold
            )
            if is_likely:
                triggers.append(TriggerEvent(
                    trigger_type=TriggerType.PREDICTIVE_RISK,
                    severity=AlertSeverity.WARNING,
                    symbol=symbol,
                    value=confidence,
                    threshold=self.breaker_config.confidence_threshold,
                    timestamp=datetime.now(),
                    message=f"Predictive: Volume anomaly likely (confidence: {confidence:.2f})",
                    metadata={'prediction_type': 'volume_anomaly'}
                ))
            
            # Predict latency spike
            is_likely, confidence = model.predict_latency_spike(
                self.breaker_config.confidence_threshold
            )
            if is_likely:
                triggers.append(TriggerEvent(
                    trigger_type=TriggerType.PREDICTIVE_RISK,
                    severity=AlertSeverity.WARNING,
                    symbol=symbol,
                    value=confidence,
                    threshold=self.breaker_config.confidence_threshold,
                    timestamp=datetime.now(),
                    message=f"Predictive: Latency spike likely (confidence: {confidence:.2f})",
                    metadata={'prediction_type': 'latency_spike'}
                ))
            
        except Exception as e:
            logger.error(f"Predictive trigger check failed for {symbol}: {e}")
        
        return triggers
    
    async def _check_traditional_triggers(self, symbol: str) -> List[TriggerEvent]:
        """Check traditional circuit breaker triggers"""
        triggers = []
        
        try:
            # Price volatility trigger
            price_triggers = await self._check_price_volatility(symbol)
            triggers.extend(price_triggers)
            
            # Volume anomaly trigger
            volume_triggers = await self._check_volume_anomaly(symbol)
            triggers.extend(volume_triggers)
            
            # System latency trigger
            latency_triggers = await self._check_latency_spike()
            triggers.extend(latency_triggers)
            
            # Error rate trigger
            error_triggers = await self._check_error_rate()
            triggers.extend(error_triggers)
            
            # Drawdown trigger
            drawdown_triggers = await self._check_drawdown_limit()
            triggers.extend(drawdown_triggers)
            
            # Liquidity trigger
            liquidity_triggers = await self._check_liquidity_crisis(symbol)
            triggers.extend(liquidity_triggers)
            
        except Exception as e:
            logger.error(f"Traditional trigger check failed for {symbol}: {e}")
        
        return triggers
    
    async def _check_price_volatility(self, symbol: str) -> List[TriggerEvent]:
        """Check price volatility triggers"""
        triggers = []
        
        try:
            prices = list(self.price_data[symbol])
            if len(prices) < 2:
                return triggers
            
            # Calculate price changes
            price_changes = [abs(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices)) if prices[i-1] > 0]
            
            if not price_changes:
                return triggers
            
            # Recent volatility (last 10 changes)
            recent_vol = np.mean(price_changes[-10:])
            
            # Check warning threshold
            warning_threshold = self.breaker_config.warning_thresholds['price_volatility']
            if recent_vol > warning_threshold:
                triggers.append(TriggerEvent(
                    trigger_type=TriggerType.PRICE_VOLATILITY,
                    severity=AlertSeverity.WARNING,
                    symbol=symbol,
                    value=recent_vol,
                    threshold=warning_threshold,
                    timestamp=datetime.now(),
                    message=f"Price volatility warning: {recent_vol:.3f} > {warning_threshold:.3f}"
                ))
            
            # Check critical threshold
            critical_threshold = self.breaker_config.critical_thresholds['price_volatility']
            if recent_vol > critical_threshold:
                triggers.append(TriggerEvent(
                    trigger_type=TriggerType.PRICE_VOLATILITY,
                    severity=AlertSeverity.CRITICAL,
                    symbol=symbol,
                    value=recent_vol,
                    threshold=critical_threshold,
                    timestamp=datetime.now(),
                    message=f"Price volatility critical: {recent_vol:.3f} > {critical_threshold:.3f}"
                ))
            
        except Exception as e:
            logger.error(f"Price volatility check failed for {symbol}: {e}")
        
        return triggers
    
    async def _check_volume_anomaly(self, symbol: str) -> List[TriggerEvent]:
        """Check volume anomaly triggers"""
        triggers = []
        
        try:
            volumes = list(self.volume_data[symbol])
            if len(volumes) < 10:
                return triggers
            
            # Calculate volume ratios
            recent_avg = np.mean(volumes[-5:])
            historical_avg = np.mean(volumes[-20:-5]) if len(volumes) >= 20 else np.mean(volumes[:-5])
            
            if historical_avg == 0:
                return triggers
            
            volume_ratio = recent_avg / historical_avg
            
            # Check warning threshold
            warning_threshold = self.breaker_config.warning_thresholds['volume_anomaly']
            if volume_ratio > warning_threshold:
                triggers.append(TriggerEvent(
                    trigger_type=TriggerType.VOLUME_ANOMALY,
                    severity=AlertSeverity.WARNING,
                    symbol=symbol,
                    value=volume_ratio,
                    threshold=warning_threshold,
                    timestamp=datetime.now(),
                    message=f"Volume anomaly warning: {volume_ratio:.1f}x > {warning_threshold:.1f}x"
                ))
            
            # Check critical threshold
            critical_threshold = self.breaker_config.critical_thresholds['volume_anomaly']
            if volume_ratio > critical_threshold:
                triggers.append(TriggerEvent(
                    trigger_type=TriggerType.VOLUME_ANOMALY,
                    severity=AlertSeverity.CRITICAL,
                    symbol=symbol,
                    value=volume_ratio,
                    threshold=critical_threshold,
                    timestamp=datetime.now(),
                    message=f"Volume anomaly critical: {volume_ratio:.1f}x > {critical_threshold:.1f}x"
                ))
            
        except Exception as e:
            logger.error(f"Volume anomaly check failed for {symbol}: {e}")
        
        return triggers
    
    async def _check_latency_spike(self) -> List[TriggerEvent]:
        """Check latency spike triggers"""
        triggers = []
        
        try:
            latencies = list(self.latency_data)
            if len(latencies) < 5:
                return triggers
            
            recent_avg = np.mean(latencies[-5:])
            
            # Check warning threshold
            warning_threshold = self.breaker_config.warning_thresholds['latency']
            if recent_avg > warning_threshold:
                triggers.append(TriggerEvent(
                    trigger_type=TriggerType.LATENCY_SPIKE,
                    severity=AlertSeverity.WARNING,
                    symbol=None,
                    value=recent_avg,
                    threshold=warning_threshold,
                    timestamp=datetime.now(),
                    message=f"Latency warning: {recent_avg:.2f}s > {warning_threshold:.2f}s"
                ))
            
            # Check critical threshold
            critical_threshold = self.breaker_config.critical_thresholds['latency']
            if recent_avg > critical_threshold:
                triggers.append(TriggerEvent(
                    trigger_type=TriggerType.LATENCY_SPIKE,
                    severity=AlertSeverity.CRITICAL,
                    symbol=None,
                    value=recent_avg,
                    threshold=critical_threshold,
                    timestamp=datetime.now(),
                    message=f"Latency critical: {recent_avg:.2f}s > {critical_threshold:.2f}s"
                ))
            
        except Exception as e:
            logger.error(f"Latency spike check failed: {e}")
        
        return triggers
    
    async def _check_error_rate(self) -> List[TriggerEvent]:
        """Check error rate triggers"""
        triggers = []
        
        try:
            errors = list(self.error_data)
            if len(errors) < 10:
                return triggers
            
            recent_errors = sum(errors[-10:])
            error_rate = recent_errors / 10
            
            # Check warning threshold
            warning_threshold = self.breaker_config.warning_thresholds['error_rate']
            if error_rate > warning_threshold:
                triggers.append(TriggerEvent(
                    trigger_type=TriggerType.ERROR_RATE,
                    severity=AlertSeverity.WARNING,
                    symbol=None,
                    value=error_rate,
                    threshold=warning_threshold,
                    timestamp=datetime.now(),
                    message=f"Error rate warning: {error_rate:.2%} > {warning_threshold:.2%}"
                ))
            
            # Check critical threshold
            critical_threshold = self.breaker_config.critical_thresholds['error_rate']
            if error_rate > critical_threshold:
                triggers.append(TriggerEvent(
                    trigger_type=TriggerType.ERROR_RATE,
                    severity=AlertSeverity.CRITICAL,
                    symbol=None,
                    value=error_rate,
                    threshold=critical_threshold,
                    timestamp=datetime.now(),
                    message=f"Error rate critical: {error_rate:.2%} > {critical_threshold:.2%}"
                ))
            
        except Exception as e:
            logger.error(f"Error rate check failed: {e}")
        
        return triggers
    
    async def _check_drawdown_limit(self) -> List[TriggerEvent]:
        """Check drawdown limit triggers"""
        triggers = []
        
        try:
            # This would integrate with portfolio P&L tracking
            # For now, implement a simple check using cached data
            cache_key = "portfolio_drawdown"
            drawdown_data = await self.cache.get(cache_key)
            
            if drawdown_data:
                current_drawdown = float(drawdown_data)
                
                if current_drawdown > self.breaker_config.max_drawdown_threshold:
                    triggers.append(TriggerEvent(
                        trigger_type=TriggerType.DRAWDOWN_LIMIT,
                        severity=AlertSeverity.CRITICAL,
                        symbol=None,
                        value=current_drawdown,
                        threshold=self.breaker_config.max_drawdown_threshold,
                        timestamp=datetime.now(),
                        message=f"Drawdown limit exceeded: {current_drawdown:.2%} > {self.breaker_config.max_drawdown_threshold:.2%}"
                    ))
            
        except Exception as e:
            logger.error(f"Drawdown limit check failed: {e}")
        
        return triggers
    
    async def _check_liquidity_crisis(self, symbol: str) -> List[TriggerEvent]:
        """Check liquidity crisis triggers"""
        triggers = []
        
        try:
            # Simple liquidity check: very low volume with high volatility
            volumes = list(self.volume_data[symbol])
            prices = list(self.price_data[symbol])
            
            if len(volumes) < 10 or len(prices) < 10:
                return triggers
            
            recent_volume = np.mean(volumes[-5:])
            recent_prices = prices[-10:]
            
            if recent_prices and recent_volume < 100:  # Low volume threshold
                # Calculate price volatility
                price_changes = [abs(recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1] 
                               for i in range(1, len(recent_prices)) if recent_prices[i-1] > 0]
                
                if price_changes:
                    volatility = np.mean(price_changes)
                    
                    # Low volume + high volatility = liquidity crisis
                    if volatility > 0.02:  # 2% volatility threshold
                        triggers.append(TriggerEvent(
                            trigger_type=TriggerType.LIQUIDITY_CRISIS,
                            severity=AlertSeverity.CRITICAL,
                            symbol=symbol,
                            value=volatility,
                            threshold=0.02,
                            timestamp=datetime.now(),
                            message=f"Liquidity crisis: Low volume {recent_volume:.0f} with high volatility {volatility:.3f}"
                        ))
            
        except Exception as e:
            logger.error(f"Liquidity crisis check failed for {symbol}: {e}")
        
        return triggers
    
    async def _update_data(self, symbol: str, price: float, volume: float, 
                          latency: float, error_occurred: bool) -> None:
        """Update tracking data"""
        timestamp = datetime.now()
        
        self.price_data[symbol].append((timestamp, price))
        self.volume_data[symbol].append((timestamp, volume))
        self.latency_data.append((timestamp, latency))
        self.error_data.append((timestamp, 1 if error_occurred else 0))
    
    async def _process_triggers(self, triggers: List[TriggerEvent]) -> None:
        """Process triggered events"""
        try:
            # Add to active triggers
            self.active_triggers.extend(triggers)
            self.trigger_history.extend(triggers)
            
            # Update metrics
            self.metrics.total_triggers += len(triggers)
            self.metrics.last_triggered = datetime.now()
            
            # Check if circuit should be opened
            critical_triggers = [t for t in triggers if t.severity == AlertSeverity.CRITICAL]
            warning_count = len([t for t in triggers if t.severity == AlertSeverity.WARNING])
            
            should_open = False
            reason = ""
            
            if critical_triggers:
                should_open = True
                reason = f"Critical triggers: {len(critical_triggers)}"
            elif warning_count >= 3:
                should_open = True
                reason = f"Multiple warnings: {warning_count}"
            
            if should_open and self.state == CircuitState.CLOSED:
                await self._open_circuit(reason)
            
            # Log triggers
            for trigger in triggers:
                log_level = {
                    AlertSeverity.INFO: logging.INFO,
                    AlertSeverity.WARNING: logging.WARNING,
                    AlertSeverity.CRITICAL: logging.ERROR,
                    AlertSeverity.EMERGENCY: logging.CRITICAL
                }.get(trigger.severity, logging.WARNING)
                
                logger.log(log_level, f"Circuit Breaker Trigger [{trigger.severity.value.upper()}] {trigger.message}")
            
            # Cache recent triggers
            await self.cache.set(
                f"circuit_triggers:{datetime.now().isoformat()}",
                [t.message for t in triggers],
                ttl=3600
            )
            
        except Exception as e:
            logger.error(f"Trigger processing failed: {e}")
    
    async def _open_circuit(self, reason: str) -> None:
        """Open the circuit breaker"""
        try:
            self.state = CircuitState.OPEN
            self.state_change_time = datetime.now()
            
            logger.critical(f"CIRCUIT BREAKER OPENED: {reason}")
            
            # Store in cache
            await self.cache.set(
                "circuit_state",
                {
                    'state': self.state.value,
                    'timestamp': self.state_change_time.isoformat(),
                    'reason': reason
                },
                ttl=86400
            )
            
            # Send alert (would integrate with alerting system)
            # await self._send_alert(reason)
            
        except Exception as e:
            logger.error(f"Circuit opening failed: {e}")
    
    async def _attempt_recovery(self) -> None:
        """Attempt circuit recovery"""
        try:
            self.state = CircuitState.HALF_OPEN
            self.state_change_time = datetime.now()
            
            logger.info("Circuit Breaker transitioning to HALF_OPEN for recovery testing")
            
            # Store state change
            await self.cache.set(
                "circuit_state",
                {
                    'state': self.state.value,
                    'timestamp': self.state_change_time.isoformat()
                },
                ttl=86400
            )
            
        except Exception as e:
            logger.error(f"Recovery attempt failed: {e}")
    
    async def record_recovery_result(self, success: bool) -> None:
        """Record result of recovery test"""
        try:
            if self.state != CircuitState.HALF_OPEN:
                return
            
            if success:
                # Successful recovery
                self.state = CircuitState.CLOSED
                self.state_change_time = datetime.now()
                self.metrics.successful_recoveries += 1
                
                # Calculate downtime
                downtime = (datetime.now() - self.state_change_time).total_seconds()
                self.metrics.total_downtime += downtime
                self.metrics.average_downtime = self.metrics.total_downtime / self.metrics.successful_recoveries
                
                logger.info("Circuit Breaker RECOVERED - returning to CLOSED state")
                
                # Clear active triggers
                self.active_triggers.clear()
                
            else:
                # Failed recovery, keep open
                self.state = CircuitState.OPEN
                self.state_change_time = datetime.now()
                self.metrics.failed_recoveries += 1
                
                logger.warning("Circuit Breaker recovery FAILED - staying OPEN")
            
            # Update cache
            await self.cache.set(
                "circuit_state",
                {
                    'state': self.state.value,
                    'timestamp': self.state_change_time.isoformat()
                },
                ttl=86400
            )
            
        except Exception as e:
            logger.error(f"Recovery result recording failed: {e}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status"""
        try:
            return {
                'state': self.state.value,
                'state_change_time': self.state_change_time.isoformat(),
                'active_triggers': [
                    {
                        'type': t.trigger_type.value,
                        'severity': t.severity.value,
                        'symbol': t.symbol,
                        'message': t.message,
                        'timestamp': t.timestamp.isoformat()
                    }
                    for t in self.active_triggers[-10:]  # Last 10 active triggers
                ],
                'metrics': {
                    'total_triggers': self.metrics.total_triggers,
                    'successful_recoveries': self.metrics.successful_recoveries,
                    'failed_recoveries': self.metrics.failed_recoveries,
                    'average_downtime': self.metrics.average_downtime,
                    'last_triggered': self.metrics.last_triggered.isoformat() if self.metrics.last_triggered else None
                },
                'predictive_enabled': self.breaker_config.enable_predictive,
                'monitored_symbols': list(self.predictive_models.keys()),
                'config': {
                    'recovery_timeout': self.breaker_config.recovery_timeout,
                    'success_threshold': self.breaker_config.success_threshold,
                    'confidence_threshold': self.breaker_config.confidence_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"Status retrieval failed: {e}")
            return {}
    
    async def _load_circuit_state(self) -> None:
        """Load circuit state from cache"""
        try:
            cached_state = await self.cache.get("circuit_state")
            if cached_state:
                state_data = cached_state
                self.state = CircuitState(state_data['state'])
                self.state_change_time = datetime.fromisoformat(state_data['timestamp'])
                
                logger.info(f"Circuit state loaded: {self.state.value}")
            
        except Exception as e:
            logger.error(f"Circuit state loading failed: {e}")
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            # Save final state
            await self.cache.set(
                "circuit_state",
                {
                    'state': self.state.value,
                    'timestamp': self.state_change_time.isoformat()
                },
                ttl=86400
            )
            
            await self.cache.close()
            logger.info("Circuit Breaker cleaned up")
            
        except Exception as e:
            logger.error(f"Circuit Breaker cleanup failed: {e}")

# Utility functions
async def create_circuit_breaker(config: Dict[str, Any]) -> EnhancedCircuitBreaker:
    """Create and initialize circuit breaker"""
    breaker = EnhancedCircuitBreaker(config)
    await breaker.initialize()
    return breaker

def calculate_circuit_heatmap(triggers: List[TriggerEvent]) -> str:
    """Generate circuit breaker heatmap"""
    if not triggers:
        return "🟢 All Systems Normal"
    
    critical_count = len([t for t in triggers if t.severity == AlertSeverity.CRITICAL])
    warning_count = len([t for t in triggers if t.severity == AlertSeverity.WARNING])
    
    if critical_count > 0:
        return f"🔴 {critical_count} Critical Alerts"
    elif warning_count >= 3:
        return f"🟠 {warning_count} Warning Alerts"
    elif warning_count > 0:
        return f"🟡 {warning_count} Warning Alerts"
    else:
        return "🟢 Minor Alerts"