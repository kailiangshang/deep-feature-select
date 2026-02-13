"""
Gate-based feature selection modules.

Available gates:
- stochastic_gate: Stochastic Gate (STG)
- gumbel_sigmoid_gate: Gumbel Sigmoid Gate (GSG)  
- gumbel_softmax_gate: Gumbel Softmax Gate
- hard_concrete_gate: Hard Concrete Gate (HCG)
"""
from .stochastic import StochasticGate
from .gumbel_sigmoid import GumbelSigmoidGate
from .gumbel_softmax import GumbelSoftmaxGate
from .hard_concrete import HardConcreteGate

# Import base for subclassing
from deepfs.core import GateBase, register_gate, register_encoder

# Register all gates with full names
register_gate("stochastic_gate")(StochasticGate)
register_gate("gumbel_sigmoid_gate")(GumbelSigmoidGate)
register_gate("gumbel_softmax_gate")(GumbelSoftmaxGate)
register_gate("hard_concrete_gate")(HardConcreteGate)

__all__ = [
    "StochasticGate",
    "GumbelSigmoidGate", 
    "GumbelSoftmaxGate",
    "HardConcreteGate",
    "GateBase",
]