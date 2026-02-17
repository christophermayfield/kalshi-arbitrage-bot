"""
Event-Driven Architecture with Message Queues
Decoupled microservices architecture with asynchronous message processing
"""

from typing import Dict, List, Any, Optional, Callable, Union, TypeVar, Type
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json
import uuid
import pickle
import threading
import queue
from collections import defaultdict, deque
import weakref
import logging
from pathlib import Path

from src.utils.logging_utils import get_logger
from src.utils.config import Config

logger = get_logger("event_driven")


class MessageType(Enum):
    """Message types for the event system"""
    MARKET_DATA = "market_data"
    ORDERBOOK_UPDATE = "orderbook_update"
    TRADE_EXECUTION = "trade_execution"
    ARBITRAGE_OPPORTUNITY = "arbitrage_opportunity"
    PERFORMANCE_UPDATE = "performance_update"
    RISK_ALERT = "risk_alert"
    SYSTEM_EVENT = "system_event"
    USER_COMMAND = "user_command"
    MODEL_PREDICTION = "model_prediction"
    STRATEGY_SIGNAL = "strategy_signal"


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"
    URGENT = "urgent"


@dataclass
class EventMessage:
    """Base event message"""
    
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType = MessageType.SYSTEM_EVENT
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    destination: Optional[str] = None
    correlation_id: Optional[str] = None
    
    # Message content
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Routing information
    routing_key: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 30
    
    # Processing information
    processed: bool = False
    processing_start: Optional[datetime] = None
    processing_end: Optional[datetime] = None
    processing_duration_ms: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "destination": self.destination,
            "correlation_id": self.correlation_id,
            "data": self.data,
            "metadata": self.metadata,
            "routing_key": self.routing_key,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "processed": self.processed,
            "processing_start": self.processing_start.isoformat() if self.processing_start else None,
            "processing_end": self.processing_end.isoformat() if self.processing_end else None,
            "processing_duration_ms": self.processing_duration_ms,
            "error": self.error
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EventMessage":
        """Create message from dictionary"""
        return cls(
            message_id=data.get("message_id", str(uuid.uuid4())),
            message_type=MessageType(data.get("message_type", "system_event")),
            priority=MessagePriority(data.get("priority", "normal")),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            source=data.get("source", ""),
            destination=data.get("destination"),
            correlation_id=data.get("correlation_id"),
            data=data.get("data", {}),
            metadata=data.get("metadata", {}),
            routing_key=data.get("routing_key"),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            timeout_seconds=data.get("timeout_seconds", 30),
            processed=data.get("processed", False),
            processing_start=datetime.fromisoformat(data["processing_start"]) if data.get("processing_start") else None,
            processing_end=datetime.fromisoformat(data["processing_end"]) if data.get("processing_end") else None,
            processing_duration_ms=data.get("processing_duration_ms", 0.0),
            error=data.get("error")
        )


class MessageHandler:
    """Base class for message handlers"""
    
    def __init__(self, handler_id: str, message_types: List[MessageType]):
        self.handler_id = handler_id
        self.message_types = message_types
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.is_active = True
        
    async def handle_message(self, message: EventMessage) -> bool:
        """Handle an incoming message"""
        try:
            if message.message_type not in self.message_types:
                return False
            
            # Call subscribers
            for subscriber_id, callback in self.subscribers.get(message.message_type.value, []):
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(message)
                    else:
                        callback(message)
                except Exception as e:
                    logger.error(f"Handler {subscriber_id} error: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Message handler error: {e}")
            return False
    
    def subscribe(self, message_type: MessageType, callback: Callable) -> str:
        """Subscribe to a message type"""
        subscriber_id = str(uuid.uuid4())
        self.subscribers[message_type.value].append((subscriber_id, callback))
        return subscriber_id
    
    def unsubscribe(self, message_type: MessageType, subscriber_id: str) -> bool:
        """Unsubscribe from a message type"""
        try:
            callbacks = self.subscribers[message_type.value]
            for i, (sid, _) in enumerate(callbacks):
                if sid == subscriber_id:
                    del callbacks[i]
                    return True
            return False
        except Exception:
            return False
    
    def get_subscriber_count(self, message_type: MessageType) -> int:
        """Get number of subscribers for a message type"""
        return len(self.subscribers.get(message_type.value, []))


class MessageQueue:
    """In-memory message queue implementation"""
    
    def __init__(self, queue_id: str, max_size: int = 10000):
        self.queue_id = queue_id
        self.max_size = max_size
        self._queue = queue.Queue(maxsize=max_size)
        self._closed = False
        
        # Statistics
        self.total_messages = 0
        self.processed_messages = 0
        self.failed_messages = 0
        self.dropped_messages = 0
        
        # Priority queues
        self._priority_queues = {
            MessagePriority.CRITICAL: queue.Queue(maxsize=1000),
            MessagePriority.URGENT: queue.Queue(maxsize=2000),
            MessagePriority.HIGH: queue.Queue(maxsize=3000),
            MessagePriority.NORMAL: queue.Queue(maxsize=5000),
            MessagePriority.LOW: queue.Queue(maxsize=10000)
        }
        
    async def put(self, message: EventMessage) -> bool:
        """Put a message in the queue"""
        try:
            if self._closed:
                return False
            
            # Check queue size
            if self._queue.full():
                self.dropped_messages += 1
                logger.warning(f"Queue {self.queue_id} full, dropping message")
                return False
            
            # Add to appropriate priority queue
            priority_queue = self._priority_queues[message.priority]
            if priority_queue.full():
                self.dropped_messages += 1
                logger.warning(f"Priority queue {message.priority.value} full, dropping message")
                return False
            
            priority_queue.put(message)
            self.total_messages += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to put message in queue {self.queue_id}: {e}")
            return False
    
    async def get(self, timeout: float = 1.0) -> Optional[EventMessage]:
        """Get a message from the queue"""
        try:
            if self._closed:
                return None
            
            # Try priority queues in order
            for priority in [MessagePriority.CRITICAL, MessagePriority.URGENT, MessagePriority.HIGH, MessagePriority.NORMAL, MessagePriority.LOW]:
                priority_queue = self._priority_queues[priority]
                if not priority_queue.empty():
                    message = priority_queue.get_nowait(timeout=timeout)
                    if message:
                        self.processed_messages += 1
                        return message
            
            # Try main queue
            if not self._queue.empty():
                message = self._queue.get_nowait(timeout=timeout)
                if message:
                    self.processed_messages += 1
                    return message
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get message from queue {self.queue_id}: {e}")
            return None
    
    def size(self) -> int:
        """Get queue size"""
        return self._queue.qsize()
    
    def empty(self) -> bool:
        """Check if queue is empty"""
        return self._queue.empty()
    
    def close(self) -> None:
        """Close the queue"""
        self._closed = True
        
        # Clear all queues
        while not self._queue.empty():
            self._queue.get()
        
        for priority_queue in self._priority_queues.values():
            while not priority_queue.empty():
                priority_queue.get()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            "queue_id": self.queue_id,
            "total_messages": self.total_messages,
            "processed_messages": self.processed_messages,
            "failed_messages": self.failed_messages,
            "dropped_messages": self.dropped_messages,
            "size": self.size(),
            "empty": self.empty(),
            "priority_queue_sizes": {
                priority.value: queue.qsize()
                for priority, queue in self._priority_queues.items()
            }
        }


class MessageBroker:
    """Message broker for routing messages between handlers and queues"""
    
    def __init__(self, config: Config):
        self.config = config
        self.broker_config = config.get("event_driven", {})
        
        # Message handlers
        self.handlers: Dict[str, MessageHandler] = {}
        self.queues: Dict[str, MessageQueue] = {}
        
        # Routing table
        self.routing_table: Dict[str, str] = {}  # routing_key -> queue_id
        
        # Statistics
        self.total_messages = 0
        self.total_handlers = 0
        self.total_queues = 0
        
        # Configuration
        self.default_queue_size = self.broker_config.get("default_queue_size", 10000)
        self.enable_persistence = self.broker_config.get("enable_persistence", False)
        self.persistence_path = self.broker_config.get("persistence_path", "data/events")
        
        # State
        self.is_running = False
        self.persistence_task: Optional[asyncio.Task] = None
        
        logger.info("Message broker initialized")
    
    def register_handler(self, handler: MessageHandler) -> None:
        """Register a message handler"""
        try:
            self.handlers[handler.handler_id] = handler
            self.total_handlers += 1
            
            logger.info(f"Registered handler: {handler.handler_id}")
            
        except Exception as e:
            logger.error(f"Failed to register handler: {e}")
    
    def unregister_handler(self, handler_id: str) -> bool:
        """Unregister a message handler"""
        try:
            if handler_id in self.handlers:
                del self.handlers[handler_id]
                self.total_handlers -= 1
                logger.info(f"Unregistered handler: {handler_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to unregister handler: {e}")
            return False
    
    def create_queue(self, queue_id: str, max_size: Optional[int] = None) -> str:
        """Create a message queue"""
        try:
            queue_size = max_size or self.default_queue_size
            queue = MessageQueue(queue_id, queue_size)
            
            self.queues[queue_id] = queue
            self.total_queues += 1
            
            logger.info(f"Created queue: {queue_id} (size: {queue_size})")
            
            return queue_id
            
        except Exception as e:
            logger.error(f"Failed to create queue: {e}")
            raise
    
    def delete_queue(self, queue_id: str) -> bool:
        """Delete a message queue"""
        try:
            if queue_id in self.queues:
                queue = self.queues[queue_id]
                queue.close()
                del self.queues[queue_id]
                self.total_queues -= 1
                
                logger.info(f"Deleted queue: {queue_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete queue: {e}")
            return False
    
    def add_routing_rule(self, routing_key: str, queue_id: str) -> None:
        """Add a routing rule"""
        try:
            if queue_id not in self.queues:
                raise ValueError(f"Queue {queue_id} does not exist")
            
            self.routing_table[routing_key] = queue_id
            logger.info(f"Added routing rule: {routing_key} -> {queue_id}")
            
        except Exception as e:
            logger.error(f"Failed to add routing rule: {e}")
    
    def remove_routing_rule(self, routing_key: str) -> bool:
        """Remove a routing rule"""
        try:
            if routing_key in self.routing_table:
                del self.routing_table[routing_key]
                logger.info(f"Removed routing rule: {routing_key}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove routing rule: {e}")
            return False
    
    async def publish(self, message: EventMessage) -> bool:
        """Publish a message to the appropriate queue"""
        try:
            # Determine destination queue
            queue_id = self._get_destination_queue(message)
            
            if queue_id is None:
                logger.warning(f"No routing rule for message: {message.message_type} from {message.source}")
                return False
            
            queue = self.queues.get(queue_id)
            if not queue:
                logger.warning(f"Queue {queue_id} not found")
                return False
            
            # Add to queue
            success = await queue.put(message)
            
            if success:
                self.total_messages += 1
                logger.debug(f"Published message {message.message_id} to queue {queue_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            return False
    
    async def publish_to_queue(self, queue_id: str, message: EventMessage) -> bool:
        """Publish a message to a specific queue"""
        try:
            queue = self.queues.get(queue_id)
            if not queue:
                logger.error(f"Queue {queue_id} not found")
                return False
            
            success = await queue.put(message)
            
            if success:
                self.total_messages += 1
                logger.debug(f"Published message {message.message_id} to queue {queue_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to publish to queue {queue_id}: {e}")
            return False
    
    async def subscribe(self, queue_id: str, callback: Callable[[EventMessage], None]) -> str:
        """Subscribe to a queue"""
        try:
            queue = self.queues.get(queue_id)
            if not queue:
                raise ValueError(f"Queue {queue_id} not found")
            
            # Create a simple subscriber
            subscriber_id = str(uuid.uuid4())
            
            # Start consumer task
            consumer_task = asyncio.create_task(self._consume_messages(queue_id, callback))
            
            # Store task reference
            if not hasattr(queue, '_consumer_tasks'):
                queue._consumer_tasks = []
            queue._consumer_tasks.append(consumer_task)
            
            logger.info(f"Subscribed to queue {queue_id} with subscriber {subscriber_id}")
            
            return subscriber_id
            
        except Exception as e:
            logger.error(f"Failed to subscribe to queue {queue_id}: {e}")
            raise
    
    async def _consume_messages(self, queue_id: str, callback: Callable[[EventMessage], None]) -> None:
        """Consume messages from a queue"""
        try:
            queue = self.queues.get(queue_id)
            if not queue:
                logger.error(f"Queue {queue_id} not found")
                return
            
            while True:
                # Get message from queue
                message = await queue.get(timeout=1.0)
                
                if message is None:
                    await asyncio.sleep(0.1)
                    continue
                
                # Mark as processed
                message.processed = True
                message.processing_start = datetime.now()
                
                # Call callback
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(message)
                    else:
                        callback(message)
                except Exception as e:
                    logger.error(f"Consumer callback error: {e}")
                    message.error = str(e)
                
                # Update processing info
                message.processing_end = datetime.now()
                message.processing_duration_ms = (message.processing_end - message.processing_start).total_seconds() * 1000
                
                # Mark as processed
                message.processed = True
                
        except Exception as e:
            logger.error(f"Consumer error for queue {queue_id}: {e}")
    
    def get_broker_status(self) -> Dict[str, Any]:
        """Get broker status"""
        try:
            return {
                "running": self.is_running,
                "total_handlers": self.total_handlers,
                "total_queues": self.total_queues,
                "total_messages": self.total_messages,
                "queue_stats": {
                    queue_id: queue.get_stats()
                    for queue_id, queue in self.queues.items()
                },
                "routing_rules": len(self.routing_table),
                "persistence_enabled": self.enable_persistence,
                "persistence_path": self.persistence_path
            }
            
        except Exception as e:
            logger.error(f"Failed to get broker status: {e}")
            return {}
    
    async def start(self) -> None:
        """Start the message broker"""
        try:
            self.is_running = True
            
            # Start persistence task if enabled
            if self.enable_persistence:
                self.persistence_task = asyncio.create_task(self._persistence_loop())
            
            logger.info("Message broker started")
            
        except Exception as e:
            logger.error(f"Failed to start message broker: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the message broker"""
        try:
            self.is_running = False
            
            # Cancel persistence task
            if self.persistence_task:
                self.persistence_task.cancel()
                try:
                    await self.persistence_task
                except asyncio.CancelledError:
                    pass
            
            # Close all queues
            for queue in self.queues.values():
                queue.close()
            
            logger.info("Message broker stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop message broker: {e}")
    
    async def _persistence_loop(self) -> None:
        """Background persistence loop"""
        try:
            while self.is_running:
                await self._persist_state()
                    await asyncio.sleep(60)  # Persist every minute
                    
        except Exception as e:
            logger.error(f"Persistence loop error: {e}")
            await asyncio.sleep(5)
    
    def _persist_state(self) -> None:
        """Persist broker state to disk"""
        try:
            if not self.enable_persistence:
                return
            
            # Create persistence directory
            persistence_path = Path(self.persistence_path)
            persistence_path.mkdir(parents=True, exist_ok=True)
            
            # Persist broker state
            state = {
                "handlers": list(self.handlers.keys()),
                "queues": list(self.queues.keys()),
                "routing_table": self.routing_table,
                "statistics": self.get_broker_status(),
                "timestamp": datetime.now().isoformat()
            }
            
            state_file = persistence_path / "broker_state.json"
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Failed to persist state: {e}")


class EventProcessor:
    """Event processor for handling system events"""
    
    def __init__(self, broker: MessageBroker):
        self.broker = broker
        self.processors: Dict[str, Callable] = {}
        self.is_running = False
        self.processing_task: Optional[asyncio.Task] = None
        
        # Register system event handlers
        self._register_system_handlers()
        
    def _register_system_handlers(self) -> None:
        """Register system event handlers"""
        try:
            # Market data handler
            self.register_processor(
                "market_data",
                self._handle_market_data,
                [MessageType.MARKET_DATA]
            )
            
            # Orderbook update handler
            self.register_processor(
                "orderbook_update",
                self._handle_orderbook_update,
                [MessageType.ORDERBOOK_UPDATE]
            )
            
            # Trade execution handler
            self.register_processor(
                "trade_execution",
                self._handle_trade_execution,
                [MessageType.TRADE_EXECUTION]
            )
            
            # Performance update handler
            self.register_processor(
                "performance_update",
                self._handle_performance_update,
                [MessageType.PERFORMANCE_UPDATE]
            )
            
            # Risk alert handler
            self.register_processor(
                "risk_alert",
                self._handle_risk_alert,
                [MessageType.RISK_ALERT]
            )
            
            logger.info(f"Registered {len(self.processors)} system processors")
            
        except Exception as e:
            logger.error(f"Failed to register system handlers: {e}")
    
    def register_processor(self, processor_id: str, processor: Callable, message_types: List[MessageType]) -> None:
        """Register an event processor"""
        try:
            self.processors[processor_id] = processor
            logger.info(f"Registered processor: {processor_id}")
            
        except Exception as e:
            logger.error(f"Failed to register processor: {e}")
    
    async def start(self) -> None:
        """Start the event processor"""
        try:
            self.is_running = True
            
            # Start processing loop
            self.processing_task = asyncio.create_task(self._processing_loop())
            
            logger.info("Event processor started")
            
        except Exception as e:
            logger.error(f"Failed to start event processor: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the event processor"""
        try:
            self.is_running = False
            
            # Cancel processing task
            if self.processing_task:
                self.processing_task.cancel()
                try:
                    await self.processing_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Event processor stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop event processor: {e}")
    
    async def _processing_loop(self) -> None:
        """Main processing loop"""
        try:
            while self.is_running:
                # Process messages from all queues
                for queue_id, queue in self.broker.queues.items():
                    while not queue.empty() and self.is_running:
                        message = await queue.get(timeout=0.1)
                        if message:
                            await self._process_message(message)
                        
                        # Small delay to prevent busy waiting
                        await asyncio.sleep(0.01)
                
                # Wait before next iteration
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Processing loop error: {e}")
            await asyncio.sleep(1)
    
    async def _process_message(self, message: EventMessage) -> None:
        """Process a single message"""
        try:
            # Find processor for this message type
            processor_id = self.processors.get(message.message_type.value)
            
            if processor_id:
                processor = self.processors[processor_id]
                
                # Mark processing start
                message.processing_start = datetime.now()
                
                # Call processor
                if asyncio.iscoroutinefunction(processor):
                    await processor(message)
                else:
                    processor(message)
                
                # Mark processing end
                message.processed = True
                message.processing_end = datetime.now()
                message.processing_duration_ms = (message.processing_end - message.processing_start).total_seconds() * 1000
                
                logger.debug(f"Processed message {message.message_id} with {processor_id}")
            
        except Exception as e:
            logger.error(f"Message processing error: {e}")
    
    async def _handle_market_data(self, message: EventMessage) -> None:
        """Handle market data events"""
        try:
            # Process market data
            market_data = message.data
            
            # Update internal state
            # This would update market data caches, trigger strategy updates, etc.
            
            logger.debug(f"Processed market data: {market_data}")
            
        except Exception as e:
            logger.error(f"Market data handler error: {e}")
    
    async def _handle_orderbook_update(self, message: EventMessage) -> None:
        """Handle orderbook update events"""
        try:
            # Process orderbook update
            orderbook_data = message.data
            
            # Update orderbook caches
            # This would update orderbook data structures
            
            logger.debug(f"Processed orderbook update: {orderbook_data}")
            
        except Exception as e:
            logger.error(f"Orderbook update handler error: {e}")
    
    async def _handle_trade_execution(self, message: EventMessage) -> None:
        """Handle trade execution events"""
        try:
            # Process trade execution
            execution_data = message.data
            
            # Update trading state
            # This would update position tracking, P&L calculations, etc.
            
            logger.debug(f"Processed trade execution: {execution_data}")
            
        except Exception as e:
            logger.error(f"Trade execution handler error: {e}")
    
    async def _handle_performance_update(self, message: EventMessage) -> None:
        """Handle performance update events"""
        try:
            # Process performance update
            performance_data = message.data
            
            # Update performance metrics
            # This would update performance tracking systems
            
            logger.debug(f"Processed performance update: {performance_data}")
            
        except Exception as e:
            logger.error(f"Performance update handler error: {e}")
    
    async def _handle_risk_alert(self, message: EventMessage) -> None:
        """Handle risk alert events"""
        try:
            # Process risk alert
            alert_data = message.data
            
            # Trigger risk management actions
            # This would trigger position closures, reduce exposure, etc.
            
            logger.warning(f"Risk alert: {alert_data}")
            
        except Exception as e:
            logger.error(f"Risk alert handler error: {e}")


class EventDrivenArchitecture:
    """Main event-driven architecture coordinator"""
    
    def __init__(self, config: Config):
        self.config = config
        self.event_config = config.get("event_driven", {})
        
        # Core components
        self.broker = MessageBroker(config)
        self.processor = EventProcessor(self.broker)
        
        # Event sources
        self.event_sources: Dict[str, Any] = {}
        
        # State
        self.is_running = False
        self.startup_task: Optional[asyncio.Task] = None
        
        # Configuration
        self.enable_persistence = self.event_config.get("enable_persistence", True)
        self.enable_monitoring = self.event_config.get("enable_monitoring", True)
        self.monitoring_interval_seconds = self.event_config.get("monitoring_interval_seconds", 30)
        
        # Performance tracking
        self.message_count = 0
        self.processing_time_ms = 0.0
        self.error_count = 0
        
        logger.info("Event-driven architecture initialized")
    
    async def start(self) -> None:
        """Start the event-driven architecture"""
        try:
            # Start components
            await self.broker.start()
            await self.processor.start()
            
            # Start event sources
            await self._start_event_sources()
            
            # Start monitoring
            if self.enable_monitoring:
                self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            self.is_running = True
            self.startup_task = asyncio.create_task(self._startup_loop())
            
            logger.info("Event-driven architecture started")
            
        except Exception as e:
            logger.error(f"Failed to start event-driven architecture: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the event-driven architecture"""
        try:
            self.is_running = False
            
            # Cancel startup task
            if self.startup_task:
                self.startup_task.cancel()
                try:
                    await self.startup_task
                except asyncio.CancelledError:
                    pass
            
            # Stop components
            await self.processor.stop()
            await self.broker.stop()
            
            # Stop event sources
            await self._stop_event_sources()
            
            # Stop monitoring
            if self.enable_monitoring and self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Event-driven architecture stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop event-driven architecture: {e}")
    
    async def _start_event_sources(self) -> None:
        """Start event sources"""
        try:
            # Start WebSocket connections for real-time data
            # This would start WebSocket clients for market data, etc.
            
            logger.info("Started event sources")
            
        except Exception as e:
            logger.error(f"Failed to start event sources: {e}")
    
    async def _stop_event_sources(self) -> None:
        """Stop event sources"""
        try:
            # Stop WebSocket connections
            # This would close WebSocket clients
            
            logger.info("Stopped event sources")
            
        except Exception as e:
            logger.error(f"Failed to stop event sources: {e}")
    
    async def _startup_loop(self) -> None:
        """Startup loop for initialization"""
        try:
            # Perform startup tasks
            await self._initialize_connections()
            await self._validate_connections()
            
            logger.info("Event-driven architecture startup completed")
            
            # Wait for stop signal
            while self.is_running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Startup loop error: {e}")
            await asyncio.sleep(5)
    
    async def _initialize_connections(self) -> None:
        """Initialize connections and validate"""
        try:
            # Validate broker connections
            if not self.broker.is_running:
                raise Exception("Message broker not running")
            
            # Validate processor
            if not self.processor.is_running:
                raise Exception("Event processor not running")
            
            # Validate event sources
            # This would validate WebSocket connections, etc.
            
            logger.info("Connections initialized and validated")
            
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            raise
    
    async def _validate_connections(self) -> None:
        """Validate all connections"""
        try:
            # Check broker
            if not self.broker.is_running:
                raise Exception("Message broker not running")
            
            # Check processor
            if not self.processor.is_running:
                raise Exception("Event processor not running")
            
            # Check queues
            if not self.broker.queues:
                raise Exception("No queues available")
            
            logger.info("All connections validated")
            
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            raise
    
    async def publish_event(self, message_type: MessageType, data: Dict[str, Any], **kwargs) -> bool:
        """Publish an event"""
        try:
            # Create message
            message = EventMessage(
                message_type=message_type,
                data=data,
                source="system",
                **kwargs
            )
            
            # Publish message
            return await self.broker.publish(message)
            
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
            return False
    
    def get_architecture_status(self) -> Dict[str, Any]:
        """Get architecture status"""
        try:
            return {
                "running": self.is_running,
                "broker": self.broker.get_broker_status(),
                "processor": {
                    "running": self.processor.is_running,
                    "handlers": self.processor.total_handlers,
                    "message_types": len(self.processor.message_types)
                },
                "queues": {
                    "total": self.broker.total_queues,
                    "active": len([q for q in self.broker.queues.values() if not q.empty()]),
                    "stats": {
                        queue_id: queue.get_stats()
                        for queue_id, queue in self.broker.queues.items()
                    }
                },
                "event_sources": len(self.event_sources),
                "message_count": self.message_count,
                "processing_time_ms": self.processing_time_ms,
                "error_count": self.error_count,
                "persistence_enabled": self.enable_persistence,
                "monitoring_enabled": self.enable_monitoring
            }
            
        except Exception as e:
            logger.error(f"Failed to get architecture status: {e}")
            return {}


# Utility functions
def create_event_driven_architecture(config: Config) -> EventDrivenArchitecture:
    """Create and return event-driven architecture"""
    return EventDrivenArchitecture(config)


def create_event_message(
    message_type: MessageType,
    data: Dict[str, Any],
    **kwargs
) -> EventMessage:
    """Create an event message"""
    return EventMessage(
        message_type=message_type,
        data=data,
        **kwargs
    )


def create_message_broker(config: Config) -> MessageBroker:
    """Create and return message broker"""
    return MessageBroker(config)


def create_event_processor(broker: MessageBroker) -> EventProcessor:
    """Create and return event processor"""
    return EventProcessor(broker)