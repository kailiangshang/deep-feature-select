from __future__ import annotations

import torch
import torch.nn as nn

from deepfs.core.base import GateFeatureModule
from deepfs.core.types import SparsityLoss, TemperatureSchedule
from deepfs.core.utils import custom_one_hot, generate_gumbel_noise


class GumbelSigmoidGateModel(GateFeatureModule):
    def __init__(
        self,
        input_dim: int,
        total_epochs: int,
        initial_temperature: float = 10.0,
        final_temperature: float = 0.01,
        device: str = "cpu",
    ):
        schedule = TemperatureSchedule(
            initial=initial_temperature,
            final=final_temperature,
            total_epochs=total_epochs,
        )
        super().__init__(input_dim, temperature_schedule=schedule, device=device)
        self.gate_logits = nn.Parameter(torch.randn(input_dim))
        self.temperature = torch.tensor(initial_temperature, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            noise = generate_gumbel_noise(self.gate_logits)
            gate_prob = torch.sigmoid((self.gate_logits + noise) / self.temperature)
            self._gate_soft_prob = gate_prob
            return x * gate_prob
        else:
            gate_prob = torch.sigmoid(self.gate_logits)
            indices = torch.arange(0, self.input_dim, device=self.device)
            indices[gate_prob <= 0.5] = -1
            self._selected_indices = indices.cpu().numpy()
            y = custom_one_hot(indices.long(), self.input_dim).T
            return torch.matmul(x, y)

    def sparsity_loss(self) -> SparsityLoss:
        noise = generate_gumbel_noise(self.gate_logits)
        gate_prob = torch.sigmoid((self.gate_logits + noise) / self.temperature)
        l1_loss = torch.mean(torch.abs(gate_prob))
        return SparsityLoss(names=["gsg_sigmoid_l1"], values=[l1_loss])

    @property
    def selected_indices_candidate(self) -> torch.Tensor:
        gate_prob = torch.sigmoid(self.gate_logits)
        indices = torch.arange(0, self.input_dim, device=self.device)
        indices[gate_prob <= 0.5] = -1
        return indices
