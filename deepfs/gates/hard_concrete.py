"""
Hard Concrete Gate (HCG) for feature selection.

Reference:
    Louizos, C., et al. "Learning Sparse Neural Networks through L0 Regularization." ICLR 2018.
"""
from __future__ import annotations

import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfs.core import GateBase, SparsityLoss


class HardConcreteGate(GateBase):
    """
    Hard Concrete Gate for feature selection.
    
    Uses the Hard Concrete distribution which stretches the Concrete
    distribution and clamps to [0, 1], enabling exact zeros and ones.
    
    Parameters
    ----------
    input_dim : int
        Number of input features
    min_max_scale : tuple, default=(-0.1, 1.1)
        The (gamma, zeta) stretching parameters
    temperature : float, default=0.5
        Temperature for Concrete distribution
    hard_gate_type : str, default="hard_zero"
        Type of hard gate for inference
    device : str, default="cpu"
        Device to run the module on
        
    Attributes
    ----------
    logits : nn.Parameter
        Learnable gate logits
    gamma : float
        Minimum stretching parameter
    zeta : float
        Maximum stretching parameter
        
    Examples
    --------
    >>> gate = HardConcreteGate(input_dim=100)
    >>> selected = gate(x)
    >>> loss = gate.sparsity_loss()
    """
    
    def __init__(
        self,
        input_dim: int,
        min_max_scale: Tuple[float, float] = (-0.1, 1.1),
        temperature: float = 0.5,
        hard_gate_type: str = "hard_zero",
        device: str = "cpu"
    ):
        super().__init__(
            input_dim=input_dim,
            device=device,
            hard_gate_type=hard_gate_type
        )
        
        self.gamma = min_max_scale[0]
        self.zeta = min_max_scale[1]
        self.temperature = temperature
        
        # Learnable gate logits
        self.logits = nn.Parameter(torch.randn(input_dim))
    
    def _generate_uniform_noise(self) -> torch.Tensor:
        """Generate uniform noise for Concrete distribution."""
        return torch.rand_like(self.logits)
    
    def _sample_concrete(self, hard: bool = False) -> torch.Tensor:
        """
        Sample from Hard Concrete distribution.
        
        Parameters
        ----------
        hard : bool
            If True, use deterministic sampling (no noise)
            
        Returns
        -------
        torch.Tensor
            Gate values in [0, 1]
        """
        if hard:
            # Deterministic version for inference
            s = torch.sigmoid(self.logits)
        else:
            # Stochastic version with uniform noise
            u = self._generate_uniform_noise()
            s = torch.sigmoid(
                (torch.log(u + 1e-10) - torch.log(1 - u + 1e-10) + self.logits) 
                / self.temperature
            )
        
        # Stretch to [gamma, zeta]
        s = s * (self.zeta - self.gamma) + self.gamma
        
        # Clamp to [0, 1]
        return torch.clamp(s, min=0.0, max=1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Hard Concrete gate to input features.
        
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
            gate = self._sample_concrete(hard=False)
            self._gate_probs = gate
        else:
            gate = self._sample_concrete(hard=True)
            gate = self._apply_hard_gate(gate)
            self._gate_probs = gate.float()
        
        return x * self._gate_probs
    
    def sparsity_loss(self) -> SparsityLoss:
        """
        Compute L0 regularization loss.
        
        The loss represents the probability of a gate being open,
        encouraging sparsity.
        
        Returns
        -------
        SparsityLoss
            Contains the L0 regularizer loss
        """
        # Probability of gate being open
        # P(g > 0) = sigmoid(logits - temperature * log(-gamma / zeta))
        import math
        log_ratio = math.log(-self.gamma / self.zeta + 1e-10)
        l0_loss = torch.sigmoid(self.logits - self.temperature * log_ratio)
        loss = torch.mean(l0_loss)
        
        return SparsityLoss(
            names=["l0_regularizer"],
            values=[loss]
        )
    
    @property
    def num_selected(self) -> int:
        """Get the number of selected features."""
        if self.training:
            warnings.warn("Feature count may vary during training")
        gate = self._sample_concrete(hard=True)
        mask = self._apply_hard_gate(gate)
        return int(mask.sum().item())
    
    @property
    def gate_probs(self) -> torch.Tensor:
        """Get current gate probabilities."""
        if self._gate_probs is None:
            return self._sample_concrete(hard=True)
        return self._gate_probs