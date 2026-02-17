"""
Event-driven architecture with message queues
"""

from .event_driven import (
    MessageType,
    MessagePriority,
    EventMessage,
    MessageHandler,
    MessageQueue,
    MessageBroker,
    EventProcessor,
    EventDrivenArchitecture
    create_event_driven_architecture
)

from src.utils.logging_utils import get_logger

logger = get_logger("event_driven")

__all__ = [
    'MessageType',
    'MessagePriority',
    'EventMessage',
    'MessageHandler',
    'MessageQueue',
    'MessageBroker',
    'EventProcessor',
    'EventDrivenArchitecture',
    'create_event_driven_architecture'
]