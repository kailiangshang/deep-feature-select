"""
Composite selectors combining Gate + Encoder for feature selection.

The gate controls which of the k selected features are active,
providing additional sparsity control.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfs.core import (
    BaseSelector, 
    GateBase, 
    EncoderBase,
    SparsityLoss,
    SelectionResult
)


class CompositeSelector(BaseSelector):
    """
    Base class for composite feature selectors.
    """
    pass


class GateEncoderSelector(CompositeSelector):
    """
    Combines a Gate with an Encoder for flexible feature selection.
    
    The encoder selects k candidate features, and the gate determines
    which of those k features are actually used. This provides:
    - Exact control over maximum features (k) via encoder
    - Additional sparsity via gate
    
    Parameters
    ----------
    gate : GateBase
        Gate module for controlling feature activation
    encoder : EncoderBase
        Encoder module for selecting candidate features
        
    Attributes
    ----------
    gate : GateBase
        The gate module
    encoder : EncoderBase
        The encoder module
        
    Examples
    --------
    >>> from deepfs.gates import StochasticGate
    >>> from deepfs.encoders import ConcreteEncoder
    >>> gate = StochasticGate(input_dim=10)  # k=10 candidates
    >>> encoder = ConcreteEncoder(input_dim=100, output_dim=10)
    >>> selector = GateEncoderSelector(gate, encoder)
    >>> selected = selector(x)  # Combines gate + encoder
    """
    
    def __init__(
        self,
        gate: GateBase,
        encoder: EncoderBase
    ):
        super().__init__(device=encoder.device)
        
        if gate.input_dim != encoder.output_dim:
            raise ValueError(
                f"Gate input_dim ({gate.input_dim}) must match "
                f"encoder output_dim ({encoder.output_dim})"
            )
        
        self.gate = gate
        self.encoder = encoder
        
        self._gate_probs: Optional[torch.Tensor] = None
        self._selection_probs: Optional[torch.Tensor] = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply gate + encoder feature selection.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim)
            
        Returns
        -------
        torch.Tensor
            Selected features
        """
        # Encoder selects k candidate features
        encoded = self.encoder(x)  # (batch, k)
        
        # Gate determines which of k features are active
        gated = self.gate(encoded)  # (batch, k)
        
        self._gate_probs = self.gate._gate_probs
        self._selection_probs = self.encoder._selection_probs
        
        return gated
    
    def sparsity_loss(self) -> SparsityLoss:
        """
        Compute combined sparsity loss from gate and encoder.
        
        Returns
        -------
        SparsityLoss
            Combined sparsity losses
        """
        gate_loss = self.gate.sparsity_loss()
        
        # Combine loss names and values
        return SparsityLoss(
            names=[f"gate_{name}" for name in gate_loss.names],
            values=gate_loss.values
        )
    
    def update_temperature(self, epoch: int) -> None:
        """
        Update temperature for both gate and encoder.
        
        Parameters
        ----------
        epoch : int
            Current training epoch
        """
        if hasattr(self.gate, 'update_temperature'):
            self.gate.update_temperature(epoch)
        if hasattr(self.encoder, 'update_temperature'):
            self.encoder.update_temperature(epoch)
    
    @property
    def selected_indices(self) -> torch.Tensor:
        """
        Get indices of selected features from encoder.
        
        Returns
        -------
        torch.Tensor
            Tensor of shape (k,) containing encoder-selected feature indices
        """
        return self.encoder.selected_indices
    
    @property
    def gate_probs(self) -> torch.Tensor:
        """Get current gate probabilities."""
        return self.gate.gate_probs
    
    @property
    def num_selected(self) -> int:
        """Get number of features selected by gate."""
        return self.gate.num_selected
    
    def get_selection_result(self) -> SelectionResult:
        """
        Get detailed selection result.
        
        Returns
        -------
        SelectionResult
            Contains both encoder indices and gate probabilities
        """
        encoder_indices = self.encoder.selected_indices.detach().cpu().numpy()
        gate_probs = self.gate.gate_probs.detach().cpu().numpy()
        
        # Apply gate to encoder indices
        if self.training:
            active_mask = gate_probs > 0.5
        else:
            active_mask = self.gate._apply_hard_gate(
                torch.from_numpy(gate_probs)
            ).numpy().astype(bool)
        
        # Mark inactive features with -1
        final_indices = encoder_indices.copy()
        final_indices[~active_mask] = -1
        
        # Create full mask
        full_mask = torch.zeros(self.encoder.input_dim, dtype=torch.bool)
        valid_indices = final_indices[final_indices >= 0]
        if len(valid_indices) > 0:
            full_mask[valid_indices] = True
        
        return SelectionResult(
            selected_indices=final_indices,
            selected_mask=full_mask.numpy(),
            gate_probs=gate_probs,
            num_selected=int(active_mask.sum())
        )
    
    def hard_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with hard (discrete) selection.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim)
            
        Returns
        -------
        torch.Tensor
            Hard selected features
        """
        # Hard encoder selection
        encoded = self.encoder.hard_forward(x)
        # Hard gate selection
        return self.gate._apply_hard_gate(self.gate.gate_probs) * encoded