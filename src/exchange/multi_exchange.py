"""
Multi-Exchange Arbitrage Infrastructure
Support for multiple prediction markets with unified arbitrage detection and execution
"""

from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json
import uuid
from collections import defaultdict, deque
import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger
from src.utils.config import Config
from src.core.arbitrage import ArbitrageOpportunity, ArbitrageType
from src.core.orderbook import OrderBook
from src.clients.kalshi_client import KalshiClient
from src.execution.smart_order_routing import SmartOrderRouter, OrderRequest, OrderSide, OrderType

logger = get_logger("multi_exchange")


class ExchangeType(Enum):
    """Types of supported exchanges"""
    KALSHI = "kalshi"
    POLYMARKET = "polymarket"
    PREDICTIT = "predictit"
    AUGUR = "augur"
    OLYMPUS = "olympus"
    BETFAIR = "betfair"


class ArbitrageType(Enum):
    """Types of cross-exchange arbitrage"""
    DIRECT_PRICE = "direct_price"  # Price difference across exchanges
    IMPLIED_PROBABILITY = "implied_probability"  # Implied probability differences
    LIQUIDITY_ARBITRAGE = "liquidity_arbitrage"  # Cross-exchange liquidity differences
    TEMPORAL_ARBITRAGE = "temporal_arbitrage"  # Time-based arbitrage
    CORRELATION_ARBITRAGE = "correlation_arbitrage"  # Correlation-based arbitrage


@dataclass
class ExchangeConfig:
    """Configuration for an exchange"""
    
    exchange_id: str
    exchange_type: ExchangeType
    exchange_name: str
    
    # API configuration
    api_base_url: str
    api_key: Optional[str] = None
    private_key_path: Optional[str] = None
    demo_mode: bool = True
    
    # Rate limiting
    rate_limit_rps: int = 10
    rate_limit_rpm: int = 600
    rate_limit_burst: int = 20
    
    # Market data
    websocket_url: Optional[str] = None
    market_data_timeout: int = 30
    
    # Trading configuration
    min_order_size: int = 1
    max_order_size: int = 10000
    commission_rate: float = 0.01
    
    # Supported features
    supports_websocket: bool = False
    supports_market_data: bool = True
    supports_trading: bool = True
    supports_orderbook: bool = True
    
    # Risk parameters
    max_position_size: int = 5000
    max_daily_trades: int = 100
    max_exposure: float = 10000.0
    
    # Status
    is_active: bool = True
    last_heartbeat: Optional[datetime] = None


@dataclass
class MarketMapping:
    """Mapping of equivalent markets across exchanges"""
    
    base_event: str  # Canonical event identifier
    exchanges: Dict[str, str]  # exchange_id -> market_id
    
    # Event metadata
    event_type: str  # yes/no, numeric, categorical
    event_description: str
    event_end_time: Optional[datetime] = None
    
    # Mapping quality
    mapping_confidence: float = 1.0  # Confidence in mapping accuracy
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Arbitrage potential
    historical_arbitrage_count: int = 0
    avg_arbitrage_profit: float = 0.0
    max_arbitrage_profit: float = 0.0


@dataclass
class CrossExchangeOpportunity:
    """Cross-exchange arbitrage opportunity"""
    
    opportunity_id: str
    arbitrage_type: ArbitrageType
    
    # Exchange information
    exchange_1: str
    exchange_2: str
    market_1: str
    market_2: str
    
    # Pricing information
    price_1: float
    price_2: float
    spread: float
    spread_percent: float
    
    # Implied probabilities (for prediction markets)
    implied_prob_1: Optional[float] = None
    implied_prob_2: Optional[float] = None
    prob_difference: Optional[float] = None
    
    # Volume and liquidity
    volume_1: int = 0
    volume_2: int = 0
    liquidity_score_1: float = 0.0
    liquidity_score_2: float = 0.0
    
    # Execution parameters
    recommended_quantity: int = 0
    max_quantity: int = 0
    execution_window_seconds: int = 30
    
    # Risk metrics
    confidence: float = 0.0
    risk_score: float = 0.0
    execution_risk: float = 0.0
    
    # Timing
    timestamp: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExchangeClient:
    """Base class for exchange clients"""
    
    def __init__(self, config: ExchangeConfig):
        self.config = config
        self.client = None
        self.is_connected = False
        
    async def connect(self) -> bool:
        """Connect to exchange"""
        raise NotImplementedError
    
    async def disconnect(self) -> None:
        """Disconnect from exchange"""
        raise NotImplementedError
    
    async def get_markets(self, **kwargs) -> List[Dict[str, Any]]:
        """Get available markets"""
        raise NotImplementedError
    
    async def get_market_orderbook(self, market_id: str) -> Optional[OrderBook]:
        """Get orderbook for a market"""
        raise NotImplementedError
    
    async def create_order(self, **kwargs) -> Dict[str, Any]:
        """Create an order"""
        raise NotImplementedError
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        raise NotImplementedError
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        raise NotImplementedError


class KalshiExchangeClient(ExchangeClient):
    """Kalshi exchange client implementation"""
    
    def __init__(self, config: ExchangeConfig):
        super().__init__(config)
        self.kalshi_client = KalshiClient(config)
        
    async def connect(self) -> bool:
        """Connect to Kalshi"""
        try:
            # Test connection
            status = self.kalshi_client.get_exchange_status()
            self.is_connected = status.get("exchange_active", False)
            
            if self.is_connected:
                logger.info(f"Connected to Kalshi: {self.config.exchange_name}")
            else:
                logger.warning(f"Kalshi not active: {status}")
            
            return self.is_connected
            
        except Exception as e:
            logger.error(f"Failed to connect to Kalshi: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Kalshi"""
        self.is_connected = False
        logger.info("Disconnected from Kalshi")
    
    async def get_markets(self, status: str = "open", limit: int = 100) -> List[Dict[str, Any]]:
        """Get available markets from Kalshi"""
        try:
            response = self.kalshi_client.get_markets(status=status, limit=limit)
            return response.get("markets", [])
            
        except Exception as e:
            logger.error(f"Failed to get Kalshi markets: {e}")
            return []
    
    async def get_market_orderbook(self, market_id: str) -> Optional[OrderBook]:
        """Get orderbook for a Kalshi market"""
        try:
            orderbook_data = self.kalshi_client.get_market_orderbook(market_id)
            return OrderBook.from_api_response(orderbook_data)
            
        except Exception as e:
            logger.error(f"Failed to get Kalshi orderbook for {market_id}: {e}")
            return None
    
    async def create_order(self, market_id: str, side: str, order_type: str, 
                         price: int, count: int) -> Dict[str, Any]:
        """Create an order on Kalshi"""
        try:
            return self.kalshi_client.create_order(market_id, side, order_type, price, count)
            
        except Exception as e:
            logger.error(f"Failed to create Kalshi order: {e}")
            return {}
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a Kalshi order"""
        try:
            self.kalshi_client.cancel_order(order_id)
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel Kalshi order {order_id}: {e}")
            return False


class PolymarketExchangeClient(ExchangeClient):
    """Polymarket exchange client implementation"""
    
    def __init__(self, config: ExchangeConfig):
        super().__init__(config)
        # Initialize Polymarket client (simplified)
        self.markets_cache: Dict[str, Dict[str, Any]] = {}
        
    async def connect(self) -> bool:
        """Connect to Polymarket"""
        try:
            # Simulate connection
            self.is_connected = True
            logger.info(f"Connected to Polymarket: {self.config.exchange_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Polymarket: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Polymarket"""
        self.is_connected = False
        logger.info("Disconnected from Polymarket")
    
    async def get_markets(self, **kwargs) -> List[Dict[str, Any]]:
        """Get available markets from Polymarket"""
        try:
            # Simulate Polymarket markets
            markets = [
                {
                    "id": "presidential-election-winner-2024",
                    "question": "Who will win the 2024 US Presidential Election?",
                    "description": "US Presidential Election 2024",
                    "outcome_type": "binary",
                    "volume": 1000000,
                    "liquidity": 0.8
                },
                {
                    "id": "bitcoin-price-end-of-year",
                    "question": "What will be Bitcoin price at end of 2024?",
                    "description": "Bitcoin price prediction",
                    "outcome_type": "numeric",
                    "volume": 500000,
                    "liquidity": 0.6
                }
            ]
            
            # Cache markets
            for market in markets:
                self.markets_cache[market["id"]] = market
            
            return markets
            
        except Exception as e:
            logger.error(f"Failed to get Polymarket markets: {e}")
            return []
    
    async def get_market_orderbook(self, market_id: str) -> Optional[OrderBook]:
        """Get orderbook for a Polymarket market"""
        try:
            # Simulate orderbook
            market = self.markets_cache.get(market_id)
            if not market:
                return None
            
            # Create simulated orderbook
            base_price = 50  # Base price for binary markets
            
            bids = []
            asks = []
            
            # Generate orderbook levels
            for i in range(5):
                bid_price = base_price - (i + 1) * 2
                ask_price = base_price + (i + 1) * 2
                volume = 1000 - i * 200
                
                bids.append(OrderBookLevel(price=bid_price, count=volume, total=volume))
                asks.append(OrderBookLevel(price=ask_price, count=volume, total=volume))
            
            return OrderBook(
                market_id=market_id,
                bids=bids,
                asks=asks,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Failed to get Polymarket orderbook for {market_id}: {e}")
            return None
    
    async def create_order(self, **kwargs) -> Dict[str, Any]:
        """Create an order on Polymarket"""
        # Simulate order creation
        return {
            "id": str(uuid.uuid4()),
            "status": "filled",
            "created_at": datetime.now().isoformat()
        }
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a Polymarket order"""
        return True


class ExchangeManager:
    """Manages multiple exchange connections"""
    
    def __init__(self, config: Config):
        self.config = config
        self.multi_exchange_config = config.get("multi_exchange", {})
        
        # Exchange clients
        self.clients: Dict[str, ExchangeClient] = {}
        self.exchanges: Dict[str, ExchangeConfig] = {}
        
        # Market mappings
        self.market_mappings: Dict[str, MarketMapping] = {}
        self.event_mappings: Dict[str, MarketMapping] = {}
        
        # Status tracking
        self.is_connected = False
        self.last_heartbeat = datetime.now()
        
        # Load exchange configurations
        self._load_exchange_configs()
        
    def _load_exchange_configs(self) -> None:
        """Load exchange configurations"""
        try:
            # Load Kalshi configuration
            kalshi_config = ExchangeConfig(
                exchange_id="kalshi",
                exchange_type=ExchangeType.KALSHI,
                exchange_name="Kalshi",
                api_base_url=self.config.get("kalshi.base_url", "https://demo-api.kalshi.co"),
                api_key_id=self.config.get("kalshi.api_key_id"),
                private_key_path=self.config.get("kalshi.private_key_path"),
                demo_mode=self.config.get("kalshi.demo_mode", True),
                supports_websocket=True,
                supports_market_data=True,
                supports_trading=True,
                supports_orderbook=True
            )
            
            self.exchanges["kalshi"] = kalshi_config
            
            # Load Polymarket configuration
            polymarket_config = ExchangeConfig(
                exchange_id="polymarket",
                exchange_type=ExchangeType.POLYMARKET,
                exchange_name="Polymarket",
                api_base_url="https://api.polymarket.com",
                demo_mode=True,
                supports_websocket=False,
                supports_market_data=True,
                supports_trading=True,
                supports_orderbook=True
            )
            
            self.exchanges["polymarket"] = polymarket_config
            
            logger.info(f"Loaded {len(self.exchanges)} exchange configurations")
            
        except Exception as e:
            logger.error(f"Failed to load exchange configs: {e}")
    
    async def connect_all(self) -> bool:
        """Connect to all configured exchanges"""
        try:
            success_count = 0
            
            for exchange_id, config in self.exchanges.items():
                if not config.is_active:
                    logger.info(f"Skipping inactive exchange: {exchange_id}")
                    continue
                
                # Create client
                if config.exchange_type == ExchangeType.KALSHI:
                    client = KalshiExchangeClient(config)
                elif config.exchange_type == ExchangeType.POLYMARKET:
                    client = PolymarketExchangeClient(config)
                else:
                    logger.warning(f"Unsupported exchange type: {config.exchange_type}")
                    continue
                
                # Connect
                if await client.connect():
                    self.clients[exchange_id] = client
                    success_count += 1
                    logger.info(f"Connected to {exchange_id}")
                else:
                    logger.error(f"Failed to connect to {exchange_id}")
            
            self.is_connected = success_count > 0
            logger.info(f"Connected to {success_count}/{len(self.exchanges)} exchanges")
            
            return self.is_connected
            
        except Exception as e:
            logger.error(f"Failed to connect exchanges: {e}")
            return False
    
    async def disconnect_all(self) -> None:
        """Disconnect from all exchanges"""
        try:
            for exchange_id, client in self.clients.items():
                await client.disconnect()
            
            self.clients.clear()
            self.is_connected = False
            logger.info("Disconnected from all exchanges")
            
        except Exception as e:
            logger.error(f"Failed to disconnect exchanges: {e}")
    
    def get_client(self, exchange_id: str) -> Optional[ExchangeClient]:
        """Get client for an exchange"""
        return self.clients.get(exchange_id)
    
    def get_active_exchanges(self) -> List[str]:
        """Get list of active exchanges"""
        return [exchange_id for exchange_id, config in self.exchanges.items() 
                if config.is_active and exchange_id in self.clients]
    
    async def get_all_markets(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get markets from all active exchanges"""
        try:
            all_markets = {}
            
            for exchange_id in self.get_active_exchanges():
                client = self.clients[exchange_id]
                markets = await client.get_markets()
                all_markets[exchange_id] = markets
            
            return all_markets
            
        except Exception as e:
            logger.error(f"Failed to get all markets: {e}")
            return {}
    
    async def get_all_orderbooks(self, market_ids: Dict[str, str]) -> Dict[str, OrderBook]:
        """Get orderbooks for multiple markets"""
        try:
            orderbooks = {}
            
            for exchange_id, market_id in market_ids.items():
                client = self.clients.get(exchange_id)
                if client:
                    orderbook = await client.get_market_orderbook(market_id)
                    if orderbook:
                        orderbooks[f"{exchange_id}:{market_id}"] = orderbook
            
            return orderbooks
            
        except Exception as e:
            logger.error(f"Failed to get all orderbooks: {e}")
            return {}


class CrossExchangeArbitrageDetector:
    """Detects arbitrage opportunities across multiple exchanges"""
    
    def __init__(self, config: Config, exchange_manager: ExchangeManager):
        self.config = config
        self.exchange_manager = exchange_manager
        self.arbitrage_config = config.get("cross_exchange_arbitrage", {})
        
        # Detection parameters
        self.min_spread_threshold = self.arbitrage_config.get("min_spread_threshold", 2.0)
        self.min_profit_cents = self.arbitrage_config.get("min_profit_cents", 10)
        self.max_slippage_percent = self.arbitrage_config.get("max_slippage_percent", 5.0)
        self.confidence_threshold = self.arbitrage_config.get("confidence_threshold", 0.7)
        
        # Market mappings
        self._load_market_mappings()
        
        # Opportunity tracking
        self.active_opportunities: Dict[str, CrossExchangeOpportunity] = {}
        self.opportunity_history: deque = deque(maxlen=10000)
        
        logger.info("Cross-exchange arbitrage detector initialized")
    
    def _load_market_mappings(self) -> None:
        """Load market mappings between exchanges"""
        try:
            # Example mappings (in production, this would be loaded from database)
            mappings = [
                MarketMapping(
                    base_event="us_presidential_2024",
                    exchanges={
                        "kalshi": "PRESIDENTIAL-2024-DEM",
                        "polymarket": "presidential-election-winner-2024"
                    },
                    event_type="binary",
                    event_description="US Presidential Election 2024",
                    mapping_confidence=0.9
                ),
                MarketMapping(
                    base_event="bitcoin_price_2024",
                    exchanges={
                        "kalshi": "BITCOIN-PRICE-2024",
                        "polymarket": "bitcoin-price-end-of-year"
                    },
                    event_type="numeric",
                    event_description="Bitcoin price at end of 2024",
                    mapping_confidence=0.8
                )
            ]
            
            for mapping in mappings:
                self.market_mappings[mapping.base_event] = mapping
                self.event_mappings[mapping.base_event] = mapping
            
            logger.info(f"Loaded {len(mappings)} market mappings")
            
        except Exception as e:
            logger.error(f"Failed to load market mappings: {e}")
    
    async def scan_for_opportunities(self) -> List[CrossExchangeOpportunity]:
        """Scan for cross-exchange arbitrage opportunities"""
        try:
            opportunities = []
            
            # Get all markets
            all_markets = await self.exchange_manager.get_all_markets()
            
            # Scan each mapped event
            for base_event, mapping in self.market_mappings.items():
                # Get markets for this event
                event_markets = {}
                for exchange_id, market_id in mapping.exchanges.items():
                    if exchange_id in all_markets:
                        # Find market in exchange markets
                        for market in all_markets[exchange_id]:
                            if market["id"] == market_id:
                                event_markets[exchange_id] = market
                                break
                
                if len(event_markets) < 2:
                    continue
                
                # Get orderbooks
                market_ids = {exchange_id: market["id"] for exchange_id, market in event_markets.items()}
                orderbooks = await self.exchange_manager.get_all_orderbooks(market_ids)
                
                # Detect opportunities for this event
                event_opportunities = await self._detect_event_opportunities(
                    base_event, mapping, event_markets, orderbooks
                )
                
                opportunities.extend(event_opportunities)
            
            # Filter and rank opportunities
            filtered_opportunities = self._filter_opportunities(opportunities)
            ranked_opportunities = self._rank_opportunities(filtered_opportunities)
            
            # Update active opportunities
            self._update_active_opportunities(ranked_opportunities)
            
            logger.info(f"Found {len(ranked_opportunities)} cross-exchange opportunities")
            return ranked_opportunities
            
        except Exception as e:
            logger.error(f"Failed to scan for opportunities: {e}")
            return []
    
    async def _detect_event_opportunities(
        self,
        base_event: str,
        mapping: MarketMapping,
        event_markets: Dict[str, Dict[str, Any]],
        orderbooks: Dict[str, OrderBook]
    ) -> List[CrossExchangeOpportunity]:
        """Detect opportunities for a specific event"""
        opportunities = []
        
        try:
            exchange_ids = list(event_markets.keys())
            
            # Check all exchange pairs
            for i, exchange_1 in enumerate(exchange_ids):
                for exchange_2 in exchange_ids[i+1:]:
                    market_1 = event_markets[exchange_1]
                    market_2 = event_markets[exchange_2]
                    
                    orderbook_1 = orderbooks.get(f"{exchange_1}:{market_1['id']}")
                    orderbook_2 = orderbooks.get(f"{exchange_2}:{market_2['id']}")
                    
                    if not orderbook_1 or not orderbook_2:
                        continue
                    
                    # Detect different types of arbitrage
                    price_opportunity = self._detect_price_arbitrage(
                        exchange_1, exchange_2, market_1, market_2,
                        orderbook_1, orderbook_2, mapping
                    )
                    
                    if price_opportunity:
                        opportunities.append(price_opportunity)
                    
                    # Detect implied probability arbitrage for prediction markets
                    if mapping.event_type == "binary":
                        prob_opportunity = self._detect_probability_arbitrage(
                            exchange_1, exchange_2, market_1, market_2,
                            orderbook_1, orderbook_2, mapping
                        )
                        
                        if prob_opportunity:
                            opportunities.append(prob_opportunity)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Failed to detect event opportunities: {e}")
            return []
    
    def _detect_price_arbitrage(
        self,
        exchange_1: str,
        exchange_2: str,
        market_1: Dict[str, Any],
        market_2: Dict[str, Any],
        orderbook_1: OrderBook,
        orderbook_2: OrderBook,
        mapping: MarketMapping
    ) -> Optional[CrossExchangeOpportunity]:
        """Detect price arbitrage opportunity"""
        try:
            # Get best prices
            best_bid_1 = orderbook_1.get_best_bid()
            best_ask_1 = orderbook_1.get_best_ask()
            best_bid_2 = orderbook_2.get_best_bid()
            best_ask_2 = orderbook_2.get_best_ask()
            
            if not all([best_bid_1, best_ask_1, best_bid_2, best_ask_2]):
                return None
            
            # Calculate arbitrage opportunities
            # Buy on exchange with lower ask, sell on exchange with higher bid
            if best_ask_1.price < best_bid_2.price:
                # Buy on exchange_1, sell on exchange_2
                spread = best_bid_2.price - best_ask_1.price
                spread_percent = (spread / best_ask_1.price) * 100
                
                if spread_percent >= self.min_spread_threshold:
                    profit_cents = int(spread * 100)  # Convert to cents
                    
                    if profit_cents >= self.min_profit_cents:
                        return CrossExchangeOpportunity(
                            opportunity_id=str(uuid.uuid4()),
                            arbitrage_type=ArbitrageType.DIRECT_PRICE,
                            exchange_1=exchange_1,
                            exchange_2=exchange_2,
                            market_1=market_1["id"],
                            market_2=market_2["id"],
                            price_1=best_ask_1.price,
                            price_2=best_bid_2.price,
                            spread=spread,
                            spread_percent=spread_percent,
                            volume_1=best_ask_1.total,
                            volume_2=best_bid_2.total,
                            confidence=mapping.mapping_confidence,
                            recommended_quantity=min(best_ask_1.total, best_bid_2.total) // 2,
                            max_quantity=min(best_ask_1.total, best_bid_2.total),
                            metadata={"base_event": mapping.base_event}
                        )
            
            elif best_ask_2.price < best_bid_1.price:
                # Buy on exchange_2, sell on exchange_1
                spread = best_bid_1.price - best_ask_2.price
                spread_percent = (spread / best_ask_2.price) * 100
                
                if spread_percent >= self.min_spread_threshold:
                    profit_cents = int(spread * 100)
                    
                    if profit_cents >= self.min_profit_cents:
                        return CrossExchangeOpportunity(
                            opportunity_id=str(uuid.uuid4()),
                            arbitrage_type=ArbitrageType.DIRECT_PRICE,
                            exchange_1=exchange_2,
                            exchange_2=exchange_1,
                            market_1=market_2["id"],
                            market_2=market_1["id"],
                            price_1=best_ask_2.price,
                            price_2=best_bid_1.price,
                            spread=spread,
                            spread_percent=spread_percent,
                            volume_1=best_ask_2.total,
                            volume_2=best_bid_1.total,
                            confidence=mapping.mapping_confidence,
                            recommended_quantity=min(best_ask_2.total, best_bid_1.total) // 2,
                            max_quantity=min(best_ask_2.total, best_bid_1.total),
                            metadata={"base_event": mapping.base_event}
                        )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to detect price arbitrage: {e}")
            return None
    
    def _detect_probability_arbitrage(
        self,
        exchange_1: str,
        exchange_2: str,
        market_1: Dict[str, Any],
        market_2: Dict[str, Any],
        orderbook_1: OrderBook,
        orderbook_2: OrderBook,
        mapping: MarketMapping
    ) -> Optional[CrossExchangeOpportunity]:
        """Detect implied probability arbitrage"""
        try:
            # Calculate implied probabilities from orderbooks
            implied_prob_1 = self._calculate_implied_probability(orderbook_1)
            implied_prob_2 = self._calculate_implied_probability(orderbook_2)
            
            if implied_prob_1 is None or implied_prob_2 is None:
                return None
            
            # Calculate probability difference
            prob_diff = abs(implied_prob_1 - implied_prob_2)
            
            # Check if arbitrage opportunity exists
            if prob_diff >= 0.05:  # 5% difference threshold
                # Calculate potential profit
                profit_cents = int(prob_diff * 1000)  # Simplified profit calculation
                
                if profit_cents >= self.min_profit_cents:
                    return CrossExchangeOpportunity(
                        opportunity_id=str(uuid.uuid4()),
                        arbitrage_type=ArbitrageType.IMPLIED_PROBABILITY,
                        exchange_1=exchange_1,
                        exchange_2=exchange_2,
                        market_1=market_1["id"],
                        market_2=market_2["id"],
                        price_1=implied_prob_1 * 100,  # Convert to price-like format
                        price_2=implied_prob_2 * 100,
                        spread=abs(implied_prob_1 - implied_prob_2) * 100,
                        spread_percent=prob_diff * 100,
                        implied_prob_1=implied_prob_1,
                        implied_prob_2=implied_prob_2,
                        prob_difference=prob_diff,
                        confidence=mapping.mapping_confidence * 0.8,  # Lower confidence for prob arbitrage
                        recommended_quantity=100,  # Fixed quantity for prob arbitrage
                        max_quantity=200,
                        metadata={"base_event": mapping.base_event}
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to detect probability arbitrage: {e}")
            return None
    
    def _calculate_implied_probability(self, orderbook: OrderBook) -> Optional[float]:
        """Calculate implied probability from orderbook"""
        try:
            best_bid = orderbook.get_best_bid()
            best_ask = orderbook.get_best_ask()
            
            if not best_bid or not best_ask:
                return None
            
            # For binary markets, implied probability = bid / (bid + ask)
            implied_prob = best_bid.price / (best_bid.price + best_ask.price)
            
            return implied_prob
            
        except Exception:
            return None
    
    def _filter_opportunities(self, opportunities: List[CrossExchangeOpportunity]) -> List[CrossExchangeOpportunity]:
        """Filter opportunities based on criteria"""
        try:
            filtered = []
            
            for opp in opportunities:
                # Filter by confidence
                if opp.confidence < self.confidence_threshold:
                    continue
                
                # Filter by risk
                if opp.risk_score > 0.8:  # High risk threshold
                    continue
                
                # Filter by execution risk
                if opp.execution_risk > 0.7:
                    continue
                
                # Filter by spread
                if opp.spread_percent < self.min_spread_threshold:
                    continue
                
                filtered.append(opp)
            
            return filtered
            
        except Exception as e:
            logger.error(f"Failed to filter opportunities: {e}")
            return opportunities
    
    def _rank_opportunities(self, opportunities: List[CrossExchangeOpportunity]) -> List[CrossExchangeOpportunity]:
        """Rank opportunities by quality"""
        try:
            # Calculate composite score
            def opportunity_score(opp: CrossExchangeOpportunity) -> float:
                score = 0.0
                
                # Profit factor (40%)
                profit_score = min(1.0, opp.spread_percent / 10.0)
                score += profit_score * 0.4
                
                # Confidence factor (30%)
                score += opp.confidence * 0.3
                
                # Volume/liquidity factor (20%)
                volume_score = min(1.0, (opp.volume_1 + opp.volume_2) / 10000.0))
                score += volume_score * 0.2
                
                # Risk factor (10%)
                risk_score = max(0.0, 1.0 - opp.risk_score)
                score += risk_score * 0.1
                
                return score
            
            # Sort by score
            opportunities.sort(key=opportunity_score, reverse=True)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Failed to rank opportunities: {e}")
            return opportunities
    
    def _update_active_opportunities(self, opportunities: List[CrossExchangeOpportunity]) -> None:
        """Update active opportunities"""
        try:
            # Clear expired opportunities
            current_time = datetime.now()
            expired_ids = []
            
            for opp_id, opp in self.active_opportunities.items():
                if opp.expires_at and current_time > opp.expires_at:
                    expired_ids.append(opp_id)
            
            for opp_id in expired_ids:
                del self.active_opportunities[opp_id]
            
            # Add new opportunities
            for opp in opportunities:
                # Set expiration time
                opp.expires_at = current_time + timedelta(seconds=opp.execution_window_seconds)
                
                # Remove existing opportunity for same event
                existing_ids = [
                    oid for oid, o in self.active_opportunities.items()
                    if o.metadata.get("base_event") == opp.metadata.get("base_event")
                ]
                
                for oid in existing_ids:
                    del self.active_opportunities[oid]
                
                # Add new opportunity
                self.active_opportunities[opp.opportunity_id] = opp
            
            # Add to history
            for opp in opportunities:
                self.opportunity_history.append(opp)
            
        except Exception as e:
            logger.error(f"Failed to update active opportunities: {e}")


class MultiExchangeArbitrageEngine:
    """Main engine for multi-exchange arbitrage"""
    
    def __init__(self, config: Config):
        self.config = config
        self.multi_exchange_config = config.get("multi_exchange", {})
        
        # Components
        self.exchange_manager = ExchangeManager(config)
        self.arbitrage_detector = CrossExchangeArbitrageDetector(config, self.exchange_manager)
        self.order_router = SmartOrderRouter(config)
        
        # State
        self.is_running = False
        self.scanning_task: Optional[asyncio.Task] = None
        
        # Configuration
        self.enable_auto_execution = self.multi_exchange_config.get("enable_auto_execution", False)
        self.scanning_interval_seconds = self.multi_exchange_config.get("scanning_interval_seconds", 30)
        self.max_concurrent_arbitrages = self.multi_exchange_config.get("max_concurrent_arbitrages", 5)
        
        # Tracking
        self.active_arbitrages: Dict[str, asyncio.Task] = {}
        self.arbitrage_history: deque = deque(maxlen=1000)
        
        logger.info("Multi-exchange arbitrage engine initialized")
    
    async def start(self) -> None:
        """Start the multi-exchange arbitrage engine"""
        try:
            # Connect to exchanges
            if not await self.exchange_manager.connect_all():
                raise Exception("Failed to connect to exchanges")
            
            # Start scanning
            self.is_running = True
            self.scanning_task = asyncio.create_task(self._scanning_loop())
            
            logger.info("Multi-exchange arbitrage engine started")
            
        except Exception as e:
            logger.error(f"Failed to start arbitrage engine: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the multi-exchange arbitrage engine"""
        try:
            self.is_running = False
            
            # Cancel scanning task
            if self.scanning_task:
                self.scanning_task.cancel()
            
            # Cancel active arbitrages
            for arbitrage_id, task in self.active_arbitrages.items():
                task.cancel()
            
            # Disconnect from exchanges
            await self.exchange_manager.disconnect_all()
            
            logger.info("Multi-exchange arbitrage engine stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop arbitrage engine: {e}")
    
    async def _scanning_loop(self) -> None:
        """Background scanning loop"""
        while self.is_running:
            try:
                # Scan for opportunities
                opportunities = await self.arbitrage_detector.scan_for_opportunities()
                
                if opportunities:
                    logger.info(f"Found {len(opportunities)} cross-exchange opportunities")
                    
                    # Auto-execute if enabled
                    if self.enable_auto_execution:
                        await self._auto_execute_opportunities(opportunities[:3])  # Top 3
                
                # Wait for next scan
                await asyncio.sleep(self.scanning_interval_seconds)
                
            except Exception as e:
                logger.error(f"Scanning loop error: {e}")
                await asyncio.sleep(5)
    
    async def _auto_execute_opportunities(self, opportunities: List[CrossExchangeOpportunity]) -> None:
        """Automatically execute arbitrage opportunities"""
        try:
            for opp in opportunities:
                if len(self.active_arbitrages) >= self.max_concurrent_arbitrages:
                    logger.warning("Maximum concurrent arbitrages reached")
                    break
                
                # Execute arbitrage
                task = asyncio.create_task(self._execute_arbitrage(opp))
                self.active_arbitrages[opp.opportunity_id] = task
                
        except Exception as e:
            logger.error(f"Auto-execution failed: {e}")
    
    async def _execute_arbitrage(self, opportunity: CrossExchangeOpportunity) -> Dict[str, Any]:
        """Execute a cross-exchange arbitrage opportunity"""
        try:
            execution_id = str(uuid.uuid4())
            start_time = datetime.now()
            
            logger.info(f"Executing arbitrage: {opportunity.opportunity_id}")
            
            # Create orders for both legs
            buy_order = OrderRequest(
                request_id=f"{execution_id}_buy",
                market_id=opportunity.market_1,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=opportunity.recommended_quantity,
                price=int(opportunity.price_1),
                algorithm=ExecutionAlgorithm.SIMPLE,
                time_limit_seconds=opportunity.execution_window_seconds
            )
            
            sell_order = OrderRequest(
                request_id=f"{execution_id}_sell",
                market_id=opportunity.market_2,
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=opportunity.recommended_quantity,
                price=int(opportunity.price_2),
                algorithm=ExecutionAlgorithm.SIMPLE,
                time_limit_seconds=opportunity.execution_window_seconds
            )
            
            # Execute orders concurrently
            buy_result, sell_result = await asyncio.gather(
                self.order_router.execute_order(buy_order),
                self.order_router.execute_order(sell_order)
            )
            
            # Calculate execution metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            total_filled = buy_result.filled_quantity + sell_result.filled_quantity
            total_cost = buy_result.total_cost + sell_result.total_cost
            
            # Calculate realized profit
            if buy_result.filled_quantity > 0 and sell_result.filled_quantity > 0:
                realized_profit = (sell_result.average_price - buy_result.average_price) * min(buy_result.filled_quantity, sell_result.filled_quantity)
            else:
                realized_profit = 0
            
            # Create execution result
            result = {
                "execution_id": execution_id,
                "opportunity_id": opportunity.opportunity_id,
                "status": "completed" if total_filled >= opportunity.recommended_quantity else "partial",
                "execution_time_seconds": execution_time,
                "total_filled": total_filled,
                "total_cost": total_cost,
                "realized_profit": realized_profit,
                "expected_profit": opportunity.spread * opportunity.recommended_quantity,
                "buy_result": buy_result,
                "sell_result": sell_result,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add to history
            self.arbitrage_history.append(result)
            
            # Clean up active arbitrage
            if opportunity.opportunity_id in self.active_arbitrages:
                del self.active_arbitrages[opportunity.opportunity_id]
            
            logger.info(f"Arbitrage executed: {opportunity.opportunity_id}, profit: {realized_profit}")
            
            return result
            
        except Exception as e:
            logger.error(f"Arbitrage execution failed: {e}")
            return {
                "execution_id": str(uuid.uuid4()),
                "opportunity_id": opportunity.opportunity_id,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get engine status"""
        try:
            return {
                "running": self.is_running,
                "connected_exchanges": len(self.exchange_manager.get_active_exchanges()),
                "total_exchanges": len(self.exchange_manager.exchanges),
                "active_opportunities": len(self.active_arbitrages),
                "active_arbitrages": len(self.active_arbitrages),
                "arbitrage_history_count": len(self.arbitrage_history),
                "auto_execution": self.enable_auto_execution,
                "last_scan": None  # Would track last scan time
            }
            
        except Exception as e:
            logger.error(f"Failed to get engine status: {e}")
            return {}


# Utility functions
def create_multi_exchange_engine(config: Config) -> MultiExchangeArbitrageEngine:
    """Create and return multi-exchange arbitrage engine"""
    return MultiExchangeArbitrageEngine(config)


async def execute_cross_exchange_arbitrage(
    engine: MultiExchangeArbitrageEngine,
    opportunity: CrossExchangeOpportunity
) -> Dict[str, Any]:
    """Execute a cross-exchange arbitrage opportunity"""
    return await engine._execute_arbitrage(opportunity)