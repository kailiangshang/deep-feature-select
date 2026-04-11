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

__all__ = ["__version__"]