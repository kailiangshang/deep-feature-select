from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfs.core.base import GateFeatureModule
from deepfs.core.types import SparsityLoss
from deepfs.core.utils import custom_one_hot


class StochasticGateModel(GateFeatureModule):
    def __init__(
        self,
        input_dim: int,
        sigma: float = 1.0,
        device: str = "cpu",
    ):
        super().__init__(input_dim, temperature_schedule=None, device=device)
        self.sigma = sigma
        self.mu_gate = nn.Parameter(0.01 * torch.randn(input_dim))
        self.noise_gate = torch.randn(input_dim, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            gate_prob = torch.clamp(
                self.mu_gate + self.sigma * self.noise_gate.normal_() + 0.5, 0.0, 1.0
            )
            self._gate_soft_prob = gate_prob
            return x * gate_prob
        else:
            gate_prob = torch.clamp(self.mu_gate + 0.5, 0.0, 1.0)
            indices = torch.arange(0, self.input_dim, device=self.device)
            indices[gate_prob == 0] = -1
            self._selected_indices = indices.cpu().numpy()
            y = custom_one_hot(indices.long(), self.input_dim).T
            return torch.matmul(x, y)

    def sparsity_loss(self) -> SparsityLoss:
        h = (self.mu_gate + 0.5) / self.sigma
        loss = torch.mean(0.5 * (1 + torch.erf(h / torch.sqrt(torch.tensor(2.0)))))
        return SparsityLoss(names=["stg_sparsity"], values=[loss])

    @property
    def selected_indices_candidate(self) -> torch.Tensor:
        gate_prob = torch.clamp(self.mu_gate + 0.5, 0.0, 1.0)
        indices = torch.arange(0, self.input_dim, device=self.device)
        indices[gate_prob == 0] = -1
        return indices
