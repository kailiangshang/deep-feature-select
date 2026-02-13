"""
Encoder-based feature selection modules.

Available encoders:
- concrete_encoder: Concrete Autoencoder (direct selection matrix)
- indirect_concrete_encoder: Indirect Concrete Autoencoder (low-rank embedding)
"""
from .concrete import ConcreteEncoder
from .indirect_concrete import IndirectConcreteEncoder

# Import base for subclassing
from deepfs.core import EncoderBase, register_encoder

# Register all encoders with full names
register_encoder("concrete_encoder")(ConcreteEncoder)
register_encoder("indirect_concrete_encoder")(IndirectConcreteEncoder)

__all__ = [
    "ConcreteEncoder",
    "IndirectConcreteEncoder",
    "EncoderBase",
]