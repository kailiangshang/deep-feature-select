"""
Stochastic Gate (STG) for feature selection.

Reference:
    Yamada, Y., et al. "Feature Selection using Stochastic Gates." ICML 2020.
"""
from __future__ import annotations

import warnings
from typing import Optional

import torch
import torch.nn as nn

from deepfs.core import GateBase, SparsityLoss


class StochasticGate(GateBase):
    """
    Stochastic Gate for feature selection.
    
    Uses Gaussian noise with learnable mean (μ) to create differentiable
    stochastic gates. The gate output is clamped to [0, 1].
    
    Parameters
    ----------
    input_dim : int
        Number of input features
    sigma : float, default=0.5
        Standard deviation of the Gaussian noise
    hard_gate_type : str, default="hard_zero"
        Type of hard gate for inference ("hard_zero" or "hard_one")
    device : str, default="cpu"
        Device to run the module on
        
    Attributes
    ----------
    mu : nn.Parameter
        Learnable gate parameters (mean of the stochastic gate)
        
    Examples
    --------
    >>> gate = StochasticGate(input_dim=100, sigma=0.5)
    >>> x = torch.randn(32, 100)
    >>> selected = gate(x)  # Apply feature selection
    >>> loss = gate.sparsity_loss()  # Get sparsity regularization
    """
    
    def __init__(
        self,
        input_dim: int,
        sigma: float = 0.5,
        hard_gate_type: str = "hard_zero",
        device: str = "cpu"
    ):
        super().__init__(
            input_dim=input_dim,
            device=device,
            hard_gate_type=hard_gate_type
        )
        self.sigma = sigma
        
        # Learnable gate parameters (initialized near 0)
        self.mu = nn.Parameter(0.01 * torch.randn(input_dim))
        
        # Noise buffer (reused during forward pass)
        self.register_buffer("_noise", torch.randn(input_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply stochastic gate to input features.
        
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
            # Sample stochastic gate with Gaussian noise
            self._noise.normal_()  # Refresh noise
            gate = self.mu + self.sigma * self._noise
            gate = torch.clamp(gate + 0.5, min=0.0, max=1.0)
            self._gate_probs = gate
        else:
            # Deterministic gate during inference
            gate = torch.clamp(self.mu + 0.5, min=0.0, max=1.0)
            gate = self._apply_hard_gate(gate)
            self._gate_probs = gate.float()
        
        return x * self._gate_probs
    
    def sparsity_loss(self) -> SparsityLoss:
        """
        Compute sparsity loss using Gaussian CDF.
        
        The loss encourages gates to be closed (μ < -0.5).
        
        Returns
        -------
        SparsityLoss
            Contains the Gaussian CDF loss
        """
        # Gaussian CDF: 0.5 * (1 + erf(h / sqrt(2)))
        h = (self.mu + 0.5) / self.sigma
        cdf = 0.5 * (1 + torch.erf(h / torch.sqrt(torch.tensor(2.0))))
        loss = torch.mean(cdf)
        
        return SparsityLoss(
            names=["gaussian_cdf"],
            values=[loss]
        )
    
    @property
    def num_selected(self) -> int:
        """Get the number of selected features."""
        if self.training:
            warnings.warn("Feature count may vary during training")
        gate = torch.clamp(self.mu + 0.5, min=0.0, max=1.0)
        mask = self._apply_hard_gate(gate)
        return int(mask.sum().item())
    
    @property
    def gate_probs(self) -> torch.Tensor:
        """Get current gate probabilities."""
        if self._gate_probs is None:
            # Return deterministic probabilities
            return torch.clamp(self.mu + 0.5, min=0.0, max=1.0)
        return self._gate_probs