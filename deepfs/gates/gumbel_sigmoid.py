"""
Gumbel Sigmoid Gate (GSG) for feature selection.

Uses Gumbel noise with sigmoid activation for differentiable binary gates.
"""
from __future__ import annotations

import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfs.core import GateBase, SparsityLoss, TemperatureSchedule


class GumbelSigmoidGate(GateBase):
    """
    Gumbel Sigmoid Gate for feature selection.
    
    Uses Gumbel noise with sigmoid activation to create relaxed binary gates.
    Temperature annealing controls the discretization.
    
    Parameters
    ----------
    input_dim : int
        Number of input features
    initial_temperature : float, default=10.0
        Initial temperature for Gumbel-Softmax
    final_temperature : float, default=0.01
        Final temperature after annealing
    total_epochs : int, default=100
        Total epochs for temperature annealing
    hard_gate_type : str, default="hard_zero"
        Type of hard gate for inference
    device : str, default="cpu"
        Device to run the module on
        
    Attributes
    ----------
    logits : nn.Parameter
        Learnable gate logits
    temperature : torch.Tensor
        Current temperature value
        
    Examples
    --------
    >>> gate = GumbelSigmoidGate(input_dim=100, total_epochs=50)
    >>> for epoch in range(50):
    ...     selected = gate(x)
    ...     loss = gate.sparsity_loss()
    ...     gate.update_temperature(epoch)
    """
    
    def __init__(
        self,
        input_dim: int,
        initial_temperature: float = 10.0,
        final_temperature: float = 0.01,
        total_epochs: int = 100,
        hard_gate_type: str = "hard_zero",
        device: str = "cpu"
    ):
        super().__init__(
            input_dim=input_dim,
            device=device,
            hard_gate_type=hard_gate_type
        )
        
        self.schedule = TemperatureSchedule(
            initial=initial_temperature,
            final=final_temperature,
            total_epochs=total_epochs
        )
        
        # Learnable gate logits
        self.logits = nn.Parameter(torch.randn(input_dim))
        
        # Current temperature
        self.register_buffer(
            "temperature", 
            torch.tensor(initial_temperature)
        )
    
    def _generate_gumbel_noise(self) -> torch.Tensor:
        """Generate Gumbel noise."""
        u = torch.rand_like(self.logits)
        return -torch.log(-torch.log(u + 1e-10) + 1e-10)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Gumbel Sigmoid gate to input features.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim)
            
        Returns
        -------
        torch.Tensor
            Gated output of shape (batch_size, input_dim)
        """
        if self.training:
            # Gumbel-Sigmoid relaxation
            noise = self._generate_gumbel_noise()
            gate = torch.sigmoid((self.logits + noise) / self.temperature)
            self._gate_probs = gate
        else:
            # Hard gate during inference
            probs = torch.sigmoid(self.logits)
            gate = self._apply_hard_gate(probs)
            self._gate_probs = probs
        
        return x * self._gate_probs
    
    def update_temperature(self, epoch: int) -> None:
        """
        Update temperature based on current epoch.
        
        Parameters
        ----------
        epoch : int
            Current training epoch
        """
        self.temperature.fill_(self.schedule.get_temperature(epoch))
    
    def sparsity_loss(self) -> SparsityLoss:
        """
        Compute sparsity loss (L1 + Entropy).
        
        Returns
        -------
        SparsityLoss
            Contains L1 and entropy loss components
        """
        gate_probs = torch.sigmoid(self.logits)
        
        # L1 loss on gate probabilities
        l1_loss = torch.mean(gate_probs)
        
        # Entropy loss for binary gates
        eps = 1e-10
        entropy = -torch.mean(
            gate_probs * torch.log(gate_probs + eps) + 
            (1 - gate_probs) * torch.log(1 - gate_probs + eps)
        )
        
        return SparsityLoss(
            names=["l1_loss", "entropy_loss"],
            values=[l1_loss, entropy]
        )
    
    @property
    def num_selected(self) -> int:
        """Get the number of selected features."""
        if self.training:
            warnings.warn("Feature count may vary during training")
        probs = torch.sigmoid(self.logits)
        mask = self._apply_hard_gate(probs)
        return int(mask.sum().item())
    
    @property
    def gate_probs(self) -> torch.Tensor:
        """Get current gate probabilities."""
        if self._gate_probs is None:
            return torch.sigmoid(self.logits)
        return self._gate_probs