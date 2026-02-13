"""
Gumbel Softmax Gate for feature selection.

Uses 2-class Gumbel-Softmax (open/close) with embedding for efficient parameterization.
"""
from __future__ import annotations

import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfs.core import GateBase, SparsityLoss, TemperatureSchedule


class GumbelSoftmaxGate(GateBase):
    """
    Gumbel Softmax Gate for feature selection.
    
    Uses a 2-class Gumbel-Softmax (open/close) with embedding layer
    for efficient parameterization. Each feature has a probability of
    being "open" (selected) or "closed" (not selected).
    
    Parameters
    ----------
    input_dim : int
        Number of input features
    embedding_dim : int
        Dimension of embedding layer for parameter efficiency
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
    logits_embedding : nn.Parameter
        Embedding logits of shape (2, embedding_dim) for open/close classes
    logits_feature : nn.Parameter
        Feature logits of shape (embedding_dim, input_dim)
    temperature : torch.Tensor
        Current temperature value
        
    Examples
    --------
    >>> gate = GumbelSoftmaxGate(input_dim=100, embedding_dim=16)
    >>> selected = gate(x)
    >>> loss = gate.sparsity_loss()
    """
    
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
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
        
        self.embedding_dim = embedding_dim
        
        self.schedule = TemperatureSchedule(
            initial=initial_temperature,
            final=final_temperature,
            total_epochs=total_epochs
        )
        
        # Embedding-based parameterization
        # Shape: (2, embedding_dim) - 2 classes: close (0) and open (1)
        self.logits_embedding = nn.Parameter(torch.randn(2, embedding_dim))
        # Shape: (embedding_dim, input_dim)
        self.logits_feature = nn.Parameter(torch.randn(embedding_dim, input_dim))
        
        # Current temperature
        self.register_buffer(
            "temperature",
            torch.tensor(initial_temperature)
        )
        
        # Store full softmax output for inference
        self._softmax_probs: Optional[torch.Tensor] = None
    
    def _generate_gumbel_noise(self, shape: tuple) -> torch.Tensor:
        """Generate Gumbel noise."""
        u = torch.rand(shape, device=self.device)
        return -torch.log(-torch.log(u + 1e-10) + 1e-10)
    
    def _compute_logits(self) -> torch.Tensor:
        """Compute gate logits: (2, input_dim)"""
        return torch.matmul(self.logits_embedding, self.logits_feature)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Gumbel Softmax gate to input features.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim)
            
        Returns
        -------
        torch.Tensor
            Gated output of shape (batch_size, input_dim)
        """
        logits = self._compute_logits()  # (2, input_dim)
        
        if self.training:
            # Gumbel-Softmax relaxation
            noise = self._generate_gumbel_noise(logits.shape)
            probs = F.softmax((logits + noise) / self.temperature, dim=0)
            self._softmax_probs = probs
            # Take probability of "open" class (index 1)
            gate = probs[1, :]  # (input_dim,)
            self._gate_probs = gate
        else:
            # Hard gate during inference
            probs = F.softmax(logits, dim=0)
            self._softmax_probs = probs
            # Take probability of "open" class
            gate_probs = probs[1, :]
            
            # Apply hard gating
            selected_indices = torch.argmax(probs, dim=0)
            gate = torch.where(selected_indices == 1, 1.0, 0.0)
            self._gate_probs = gate_probs
        
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
        Compute sparsity loss (mean of open probabilities).
        
        Returns
        -------
        SparsityLoss
            Contains L1 loss on open probabilities
        """
        if self._gate_probs is None:
            logits = self._compute_logits()
            probs = F.softmax(logits, dim=0)
            p_open = probs[1, :]
        else:
            p_open = self._gate_probs
        
        l1_loss = torch.mean(p_open)
        
        return SparsityLoss(
            names=["l1_loss"],
            values=[l1_loss]
        )
    
    @property
    def num_selected(self) -> int:
        """Get the number of selected features."""
        if self.training:
            warnings.warn("Feature count may vary during training")
        
        if self._softmax_probs is None:
            logits = self._compute_logits()
            probs = F.softmax(logits, dim=0)
        else:
            probs = self._softmax_probs
        
        selected_indices = torch.argmax(probs, dim=0)
        return int((selected_indices == 1).sum().item())
    
    @property
    def gate_probs(self) -> torch.Tensor:
        """Get current gate probabilities (probability of open class)."""
        if self._gate_probs is None:
            logits = self._compute_logits()
            probs = F.softmax(logits, dim=0)
            return probs[1, :]
        return self._gate_probs