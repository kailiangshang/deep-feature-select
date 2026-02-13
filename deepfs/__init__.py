"""
DeepFS - Deep Feature Selection Library

A modular library for differentiable feature selection using neural networks.

Components:
- core: Base classes, types, and utilities
- gates: Stochastic gates (STG, GSG, GumbelSoftmax, HCG)
- encoders: Feature encoders (CAE, IPCAE)
- selectors: Composite selectors combining gates and encoders
- training: Training utilities and callbacks

Example
-------
>>> from deepfs import StochasticGate, ConcreteEncoder
>>> from deepfs import GateEncoderSelector, FeatureSelectionTrainer
>>> 
>>> # Create selector
>>> gate = StochasticGate(input_dim=10)
>>> encoder = ConcreteEncoder(input_dim=100, output_dim=10)
>>> selector = GateEncoderSelector(gate, encoder)
>>> 
>>> # Use in model
>>> class MyModel(nn.Module):
...     def __init__(self):
...         super().__init__()
...         self.selector = selector
...         self.fc = nn.Linear(10, 1)
...     
...     def forward(self, x):
...         x = self.selector(x)
...         return self.fc(x)
"""

# Version
__version__ = "0.1.0"

# Core components
from deepfs.core import (
    # Base classes
    GateBase,
    EncoderBase,
    BaseSelector,
    # Types
    SparsityLoss,
    SelectionResult,
    TemperatureSchedule,
    # Registry
    register_gate,
    register_encoder,
    get_gate,
    get_encoder,
    create_gate,
    create_encoder,
    list_gates,
    list_encoders,
)

# Gates
from deepfs.gates import (
    StochasticGate,
    GumbelSigmoidGate,
    GumbelSoftmaxGate,
    HardConcreteGate,
)

# Encoders
from deepfs.encoders import (
    ConcreteEncoder,
    IndirectConcreteEncoder,
)

# Selectors
from deepfs.selectors import (
    CompositeSelector,
    GateEncoderSelector,
)

# Training
from deepfs.training import (
    FeatureSelectionTrainer,
    TemperatureCallback,
    LoggingCallback,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "GateBase",
    "EncoderBase",
    "BaseSelector",
    "SparsityLoss",
    "SelectionResult",
    "TemperatureSchedule",
    "register_gate",
    "register_encoder",
    "get_gate",
    "get_encoder",
    "create_gate",
    "create_encoder",
    "list_gates",
    "list_encoders",
    # Gates
    "StochasticGate",
    "GumbelSigmoidGate",
    "GumbelSoftmaxGate",
    "HardConcreteGate",
    # Encoders
    "ConcreteEncoder",
    "IndirectConcreteEncoder",
    # Selectors
    "CompositeSelector",
    "GateEncoderSelector",
    # Training
    "FeatureSelectionTrainer",
    "TemperatureCallback",
    "LoggingCallback",
]