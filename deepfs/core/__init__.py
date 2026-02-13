"""
Deep Feature Selection - Core Module
"""

from .base import BaseSelector, GateBase, EncoderBase
from .types import SelectionResult, SparsityLoss, TemperatureSchedule
from .registry import (
    register_gate, 
    register_encoder, 
    get_gate, 
    get_encoder, 
    create_gate,
    create_encoder,
    list_gates, 
    list_encoders
)

__all__ = [
    # Base classes
    "BaseSelector",
    "GateBase", 
    "EncoderBase",
    # Types
    "SelectionResult",
    "SparsityLoss",
    "TemperatureSchedule",
    # Registry
    "register_gate",
    "register_encoder", 
    "get_gate",
    "get_encoder",
    "create_gate",
    "create_encoder",
    "list_gates",
    "list_encoders",
]
