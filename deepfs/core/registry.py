"""
Module registry for extensible feature selection components.
"""
from __future__ import annotations

from typing import Dict, Type, Callable, Any, Optional

from .base import GateBase, EncoderBase


# Global registries
_GATE_REGISTRY: Dict[str, Type[GateBase]] = {}
_ENCODER_REGISTRY: Dict[str, Type[EncoderBase]] = {}


def register_gate(name: str) -> Callable[[Type[GateBase]], Type[GateBase]]:
    """
    Decorator to register a gate class.
    
    Parameters
    ----------
    name : str
        Name to register the gate under
        
    Returns
    -------
    Callable
        Decorator function
        
    Example
    -------
    @register_gate("stg")
    class StochasticGate(GateBase):
        ...
    """
    def decorator(cls: Type[GateBase]) -> Type[GateBase]:
        if name in _GATE_REGISTRY:
            raise ValueError(f"Gate '{name}' is already registered")
        if not issubclass(cls, GateBase):
            raise TypeError(f"Registered class must be a subclass of GateBase")
        _GATE_REGISTRY[name] = cls
        return cls
    return decorator


def register_encoder(name: str) -> Callable[[Type[EncoderBase]], Type[EncoderBase]]:
    """
    Decorator to register an encoder class.
    
    Parameters
    ----------
    name : str
        Name to register the encoder under
        
    Returns
    -------
    Callable
        Decorator function
        
    Example
    -------
    @register_encoder("cae")
    class ConcreteEncoder(EncoderBase):
        ...
    """
    def decorator(cls: Type[EncoderBase]) -> Type[EncoderBase]:
        if name in _ENCODER_REGISTRY:
            raise ValueError(f"Encoder '{name}' is already registered")
        if not issubclass(cls, EncoderBase):
            raise TypeError(f"Registered class must be a subclass of EncoderBase")
        _ENCODER_REGISTRY[name] = cls
        return cls
    return decorator


def get_gate(name: str) -> Type[GateBase]:
    """
    Get a registered gate class by name.
    
    Parameters
    ----------
    name : str
        Name of the gate
        
    Returns
    -------
    Type[GateBase]
        The gate class
        
    Raises
    ------
    KeyError
        If gate is not registered
    """
    if name not in _GATE_REGISTRY:
        available = list(_GATE_REGISTRY.keys())
        raise KeyError(f"Gate '{name}' not found. Available: {available}")
    return _GATE_REGISTRY[name]


def get_encoder(name: str) -> Type[EncoderBase]:
    """
    Get a registered encoder class by name.
    
    Parameters
    ----------
    name : str
        Name of the encoder
        
    Returns
    -------
    Type[EncoderBase]
        The encoder class
        
    Raises
    ------
    KeyError
        If encoder is not registered
    """
    if name not in _ENCODER_REGISTRY:
        available = list(_ENCODER_REGISTRY.keys())
        raise KeyError(f"Encoder '{name}' not found. Available: {available}")
    return _ENCODER_REGISTRY[name]


def list_gates() -> list:
    """List all registered gate names."""
    return list(_GATE_REGISTRY.keys())


def list_encoders() -> list:
    """List all registered encoder names."""
    return list(_ENCODER_REGISTRY.keys())


def create_gate(name: str, **kwargs) -> GateBase:
    """
    Create a gate instance by name.
    
    Parameters
    ----------
    name : str
        Name of the gate
    **kwargs
        Arguments to pass to the gate constructor
        
    Returns
    -------
    GateBase
        Gate instance
    """
    cls = get_gate(name)
    return cls(**kwargs)


def create_encoder(name: str, **kwargs) -> EncoderBase:
    """
    Create an encoder instance by name.
    
    Parameters
    ----------
    name : str
        Name of the encoder
    **kwargs
        Arguments to pass to the encoder constructor
        
    Returns
    -------
    EncoderBase
        Encoder instance
    """
    cls = get_encoder(name)
    return cls(**kwargs)