"""
Concrete Autoencoder (CAE) for feature selection.

Reference:
    Balin, M.F., et al. "Concrete Autoencoders for Differentiable Feature Selection." ICML 2019.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfs.core import EncoderBase, TemperatureSchedule


class ConcreteEncoder(EncoderBase):
    """
    Concrete Autoencoder encoder for feature selection.
    
    Uses Gumbel-Softmax relaxation to select exactly k features from
    n input features. Each of the k output dimensions selects one input
    feature using a softmax over all input features.
    
    Parameters
    ----------
    input_dim : int
        Number of input features (n)
    output_dim : int
        Number of features to select (k)
    initial_temperature : float, default=10.0
        Initial temperature for Gumbel-Softmax
    final_temperature : float, default=0.01
        Final temperature after annealing
    total_epochs : int, default=100
        Total epochs for temperature annealing
    device : str, default="cpu"
        Device to run the module on
        
    Attributes
    ----------
    logits : nn.Parameter
        Selection logits of shape (input_dim, output_dim)
    temperature : torch.Tensor
        Current temperature value
        
    Examples
    --------
    >>> encoder = ConcreteEncoder(input_dim=100, output_dim=10)
    >>> for epoch in range(100):
    ...     selected = encoder(x)  # Select 10 features
    ...     encoder.update_temperature(epoch)
    >>> indices = encoder.selected_indices  # Get selected feature indices
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        initial_temperature: float = 10.0,
        final_temperature: float = 0.01,
        total_epochs: int = 100,
        device: str = "cpu"
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            device=device
        )
        
        self.schedule = TemperatureSchedule(
            initial=initial_temperature,
            final=final_temperature,
            total_epochs=total_epochs
        )
        
        # Selection logits: each output dimension selects one input
        # Shape: (input_dim, output_dim)
        self.logits = nn.Parameter(torch.randn(input_dim, output_dim))
        
        # Current temperature
        self.register_buffer(
            "temperature",
            torch.tensor(initial_temperature)
        )
        
        # Store softmax output for inference
        self._selection_probs: Optional[torch.Tensor] = None
    
    def _generate_gumbel_noise(self) -> torch.Tensor:
        """Generate Gumbel noise."""
        u = torch.rand_like(self.logits)
        return -torch.log(-torch.log(u + 1e-10) + 1e-10)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply concrete feature selection.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim)
            
        Returns
        -------
        torch.Tensor
            Selected features of shape (batch_size, output_dim)
        """
        if self.training:
            # Gumbel-Softmax relaxation
            noise = self._generate_gumbel_noise()
            self._selection_probs = F.softmax(
                (self.logits + noise) / self.temperature, dim=0
            )
        else:
            # Deterministic selection during inference
            self._selection_probs = F.softmax(self.logits, dim=0)
        
        # Weighted combination of input features
        # x: (batch, input_dim), probs: (input_dim, output_dim)
        return torch.matmul(x, self._selection_probs)
    
    def update_temperature(self, epoch: int) -> None:
        """
        Update temperature based on current epoch.
        
        Parameters
        ----------
        epoch : int
            Current training epoch
        """
        self.temperature.fill_(self.schedule.get_temperature(epoch))
    
    @property
    def selected_indices(self) -> torch.Tensor:
        """
        Get indices of selected features.
        
        Returns
        -------
        torch.Tensor
            Tensor of shape (output_dim,) containing selected feature indices
        """
        if self._selection_probs is None:
            probs = F.softmax(self.logits, dim=0)
        else:
            probs = self._selection_probs
        
        # Argmax over input dimension for each output
        return torch.argmax(probs, dim=0)
    
    def get_one_hot_selection(self) -> torch.Tensor:
        """
        Get one-hot selection matrix.
        
        Returns
        -------
        torch.Tensor
            One-hot matrix of shape (input_dim, output_dim)
        """
        indices = self.selected_indices
        one_hot = torch.zeros_like(self.logits)
        one_hot[indices, torch.arange(self.output_dim)] = 1.0
        return one_hot
    
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
            Selected features of shape (batch_size, output_dim)
        """
        indices = self.selected_indices
        return x[:, indices]