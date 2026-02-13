"""
Composite selectors combining Gate + Encoder.

This enables flexible combinations of any gate with any encoder:
- STG + CAE, STG + IPCAE
- GSG + CAE, GSG + IPCAE
- GumbelSoftmax + CAE, GumbelSoftmax + IPCAE
- HCG + CAE, HCG + IPCAE
"""
from .composite import (
    CompositeSelector,
    GateEncoderSelector,
)

__all__ = [
    "CompositeSelector",
    "GateEncoderSelector",
]