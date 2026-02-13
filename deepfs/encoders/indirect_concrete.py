"""
Indirect Concrete Autoencoder (IPCAE) for feature selection.

A more parameter-efficient version of ConcreteEncoder using low-rank embedding.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfs.core import EncoderBase, TemperatureSchedule


class IndirectConcreteEncoder(EncoderBase):
    """
    Indirect Concrete Autoencoder encoder for feature selection.
    
    Uses a low-rank embedding to parameterize the selection matrix,
    reducing parameters from O(n*k) to O(n*d + d*k) where d << n.
    
    Parameters
    ----------
    input_dim : int
        Number of input features (n)
    output_dim : int
        Number of features to select (k)
    embedding_dim : int
        Dimension of the embedding (d)
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
    logits_input : nn.Parameter
        Input embedding of shape (input_dim, embedding_dim)
    logits_output : nn.Parameter
        Output embedding of shape (embedding_dim, output_dim)
    temperature : torch.Tensor
        Current temperature value
        
    Examples
    --------
    >>> encoder = IndirectConcreteEncoder(input_dim=1000, output_dim=10, embedding_dim=32)
    >>> selected = encoder(x)  # Only needs 1000*32 + 32*10 = 32320 params
    >>> # vs ConcreteEncoder which needs 1000*10 = 10000 params
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        embedding_dim: int,
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
        
        self.embedding_dim = embedding_dim
        
        self.schedule = TemperatureSchedule(
            initial=initial_temperature,
            final=final_temperature,
            total_epochs=total_epochs
        )
        
        # Low-rank parameterization: logits = input_emb @ output_emb
        self.logits_input = nn.Parameter(torch.randn(input_dim, embedding_dim))
        self.logits_output = nn.Parameter(torch.randn(embedding_dim, output_dim))
        
        # Current temperature
        self.register_buffer(
            "temperature",
            torch.tensor(initial_temperature)
        )
        
        # Store softmax output for inference
        self._selection_probs: Optional[torch.Tensor] = None
    
    @property
    def logits(self) -> torch.Tensor:
        """Compute selection logits from embeddings."""
        return torch.matmul(self.logits_input, self.logits_output)
    
    def _generate_gumbel_noise(self) -> torch.Tensor:
        """Generate Gumbel noise."""
        u = torch.rand_like(self.logits)
        return -torch.log(-torch.log(u + 1e-10) + 1e-10)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply indirect concrete feature selection.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim)
            
        Returns
        -------
        torch.Tensor
            Selected features of shape (batch_size, output_dim)
        """
        logits = self.logits
        
        if self.training:
            # Gumbel-Softmax relaxation
            noise = self._generate_gumbel_noise()
            self._selection_probs = F.softmax(
                (logits + noise) / self.temperature, dim=0
            )
        else:
            # Deterministic selection during inference
            self._selection_probs = F.softmax(logits, dim=0)
        
        # Weighted combination of input features
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