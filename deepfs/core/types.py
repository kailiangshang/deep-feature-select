"""
Type definitions for Deep Feature Selection
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional, List, Dict, Any

import torch
import numpy as np


class SparsityLoss(NamedTuple):
    """
    Container for sparsity loss values.
    
    Attributes
    ----------
    names : List[str]
        Names of each loss component
    values : List[torch.Tensor]
        Loss tensor values
    weights : Optional[List[float]]
        Optional weights for each loss component
    """
    names: List[str]
    values: List[torch.Tensor]
    weights: Optional[List[float]] = None
    
    @property
    def total(self) -> torch.Tensor:
        """Calculate weighted total loss."""
        if not self.values:
            return torch.tensor(0.0)
        
        if self.weights is None:
            return sum(self.values)
        
        return sum(w * v for w, v in zip(self.weights, self.values))
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary with scalar values."""
        return {name: value.item() for name, value in zip(self.names, self.values)}


@dataclass
class SelectionResult:
    """
    Result of feature selection.
    
    Attributes
    ----------
    selected_indices : np.ndarray
        Indices of selected features (-1 means not selected)
    selected_mask : np.ndarray
        Boolean mask of selected features
    gate_probs : Optional[np.ndarray]
        Gate probabilities for each feature (if applicable)
    num_selected : int
        Number of selected features
    """
    selected_indices: np.ndarray
    selected_mask: np.ndarray
    gate_probs: Optional[np.ndarray] = None
    num_selected: int = 0
    
    def __post_init__(self):
        if self.num_selected == 0:
            self.num_selected = int(self.selected_mask.sum())


class HardGateType(str):
    """Hard gate type for inference."""
    HARD_ZERO = "hard_zero"  # Gate > 0 means selected
    HARD_ONE = "hard_one"    # Gate == 1 means selected


@dataclass
class TemperatureSchedule:
    """
    Temperature schedule for Gumbel-Softmax based methods.
    
    Attributes
    ----------
    initial : float
        Initial temperature
    final : float
        Final temperature  
    total_epochs : int
        Total number of epochs for annealing
    """
    initial: float = 10.0
    final: float = 0.01
    total_epochs: int = 100
    
    def get_temperature(self, epoch: int) -> float:
        """Get temperature for current epoch using exponential decay."""
        if epoch >= self.total_epochs:
            return self.final
        return self.initial * (self.final / self.initial) ** (epoch / self.total_epochs)


@dataclass  
class GateConfig:
    """Base configuration for gate modules."""
    input_dim: int
    device: str = "cpu"
    hard_gate_type: str = "hard_zero"


@dataclass
class EncoderConfig:
    """Base configuration for encoder modules."""
    input_dim: int
    output_dim: int  # k features to select
    device: str = "cpu"