"""
Base classes for Deep Feature Selection
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np

from .types import SparsityLoss, SelectionResult


class BaseSelector(nn.Module, ABC):
    """
    Abstract base class for all feature selectors.
    """
    
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        self._selection_result: Optional[SelectionResult] = None
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - apply feature selection.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim)
            
        Returns
        -------
        torch.Tensor
            Selected features
        """
        pass
    
    @abstractmethod
    def get_selection_result(self) -> SelectionResult:
        """
        Get the feature selection result.
        
        Returns
        -------
        SelectionResult
            Contains selected indices, mask, and probabilities
        """
        pass
    
    def to(self, device: Union[str, torch.device]) -> "BaseSelector":
        """Move module to device."""
        self.device = str(device)
        return super().to(device)


class GateBase(BaseSelector):
    """
    Abstract base class for gate-based feature selectors.
    
    Gate methods learn a probability for each feature independently,
    then apply element-wise multiplication to select features.
    The number of selected features is controlled by sparsity loss.
    
    Examples: STG, GSG, Gumbel Softmax Gate, HCG
    """
    
    def __init__(
        self,
        input_dim: int,
        device: str = "cpu",
        hard_gate_type: str = "hard_zero"
    ):
        super().__init__(device=device)
        self.input_dim = input_dim
        self.hard_gate_type = hard_gate_type
        self._gate_probs: Optional[torch.Tensor] = None
    
    @abstractmethod
    def sparsity_loss(self) -> SparsityLoss:
        """
        Compute sparsity loss to encourage feature selection.
        
        Returns
        -------
        SparsityLoss
            Named tuple containing loss names and values
        """
        pass
    
    @property
    @abstractmethod
    def num_selected(self) -> int:
        """Get the number of selected features."""
        pass
    
    @property
    @abstractmethod
    def gate_probs(self) -> torch.Tensor:
        """Get current gate probabilities."""
        pass
    
    def _apply_hard_gate(self, gate_logits: torch.Tensor) -> torch.Tensor:
        """
        Apply hard gate during inference.
        
        Parameters
        ----------
        gate_logits : torch.Tensor
            Gate logits or probabilities
            
        Returns
        -------
        torch.Tensor
            Binary gate values (0 or 1)
        """
        if self.hard_gate_type == "hard_zero":
            return torch.where(gate_logits > 0, 1.0, 0.0)
        elif self.hard_gate_type == "hard_one":
            return torch.where(gate_logits == 1, 1.0, 0.0)
        else:
            raise ValueError(f"Unknown hard_gate_type: {self.hard_gate_type}")
    
    def get_selection_result(self) -> SelectionResult:
        """Get the feature selection result."""
        if self._gate_probs is None:
            raise RuntimeError("Gate probabilities not available. Run forward pass first.")
        
        probs = self._gate_probs.detach().cpu()
        
        if self.training:
            # During training, use soft selection
            mask = probs > 0.5
        else:
            # During inference, use hard selection
            mask = self._apply_hard_gate(probs).bool()
        
        # Create index array (-1 for not selected)
        indices = torch.arange(self.input_dim)
        indices[~mask.squeeze()] = -1
        
        return SelectionResult(
            selected_indices=indices.numpy(),
            selected_mask=mask.squeeze().numpy(),
            gate_probs=probs.squeeze().numpy(),
            num_selected=int(mask.sum().item())
        )


class EncoderBase(BaseSelector):
    """
    Abstract base class for encoder-based feature selectors.
    
    Encoder methods explicitly select exactly k features using
    a learned selection matrix with Gumbel-Softmax relaxation.
    
    Examples: Concrete Encoder (CAE), Indirect Concrete Encoder (IPCAE)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,  # k features to select
        device: str = "cpu"
    ):
        super().__init__(device=device)
        self.input_dim = input_dim
        self.output_dim = output_dim  # k
    
    # Note: logits should be implemented as either a Parameter or property
    # depending on the encoder type
    
    @property
    @abstractmethod
    def selected_indices(self) -> torch.Tensor:
        """Get indices of selected features."""
        pass
    
    def get_selection_result(self) -> SelectionResult:
        """Get the feature selection result."""
        indices = self.selected_indices.detach().cpu().numpy()
        
        # Create mask
        mask = np.zeros(self.input_dim, dtype=bool)
        valid_indices = indices[indices >= 0]
        mask[valid_indices] = True
        
        return SelectionResult(
            selected_indices=indices,
            selected_mask=mask,
            gate_probs=None,
            num_selected=len(valid_indices)
        )