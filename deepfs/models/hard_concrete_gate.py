from __future__ import annotations

import torch
import torch.nn as nn

from deepfs.core.base import GateFeatureModule
from deepfs.core.types import SparsityLoss
from deepfs.core.utils import custom_one_hot


class HardConcreteGateModel(GateFeatureModule):
    def __init__(
        self,
        input_dim: int,
        min_max_scale: tuple = (-0.1, 1.1),
        temperature: float = 0.5,
        device: str = "cpu",
    ):
        super().__init__(input_dim, temperature_schedule=None, device=device)
        self.gamma = min_max_scale[0]
        self.zeta = min_max_scale[1]
        self.hcg_temperature = temperature
        self.gate_logits = nn.Parameter(torch.randn(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            u = torch.rand_like(self.gate_logits, device=self.device)
            s = torch.sigmoid(
                (torch.log(u + 1e-10) - torch.log(1 - u + 1e-10) + self.gate_logits)
                / self.hcg_temperature
            )
            gate_prob = torch.clamp(s * (self.zeta - self.gamma) + self.gamma, 0.0, 1.0)
            self._gate_soft_prob = gate_prob
            return x * gate_prob
        else:
            s = torch.sigmoid(self.gate_logits)
            gate_prob = torch.clamp(s * (self.zeta - self.gamma) + self.gamma, 0.0, 1.0)
            indices = torch.arange(0, self.input_dim, device=self.device)
            indices[gate_prob == 0] = -1
            self._selected_indices = indices.cpu().numpy()
            y = custom_one_hot(indices.long(), self.input_dim).T
            return torch.matmul(x, y)

    def sparsity_loss(self) -> SparsityLoss:
        loss = torch.mean(torch.sigmoid(
            self.gate_logits - self.hcg_temperature * torch.log(torch.tensor(-self.gamma / self.zeta))
        ))
        return SparsityLoss(names=["hcg_l0"], values=[loss])

    @property
    def selected_indices_candidate(self) -> torch.Tensor:
        s = torch.sigmoid(self.gate_logits)
        gate_prob = torch.clamp(s * (self.zeta - self.gamma) + self.gamma, 0.0, 1.0)
        indices = torch.arange(0, self.input_dim, device=self.device)
        indices[gate_prob == 0] = -1
        return indices
