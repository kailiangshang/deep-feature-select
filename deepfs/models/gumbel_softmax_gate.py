from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfs.core.base import GateFeatureModule
from deepfs.core.types import SparsityLoss, TemperatureSchedule
from deepfs.core.utils import custom_one_hot, generate_gumbel_noise


class GumbelSoftmaxGateModel(GateFeatureModule):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
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
        self.embedding_dim = embedding_dim
        self.logits_gate_embedding = nn.Parameter(torch.randn(2, embedding_dim))
        self.logits_gate_feature = nn.Parameter(torch.randn(embedding_dim, input_dim))
        self.temperature = torch.tensor(initial_temperature, device=device)
        self._gate_soft_prob_full = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits_gate = (torch.matmul(self.logits_gate_feature.T, self.logits_gate_embedding.T)).T
        if self.training:
            gumbel_noise = generate_gumbel_noise(logits_gate)
            gate_soft_prob = F.softmax((logits_gate + gumbel_noise) / self.temperature, dim=0)
            p_open = gate_soft_prob[1, :]
            self._gate_soft_prob = p_open
            self._gate_soft_prob_full = gate_soft_prob
            return x * p_open
        else:
            selected_open = torch.argmax(logits_gate, dim=0)
            indices = torch.arange(0, self.input_dim, device=self.device)
            indices[selected_open == 0] = -1
            self._selected_indices = indices.cpu().numpy()
            y = custom_one_hot(indices.long(), self.input_dim).T
            return torch.matmul(x, y)

    def sparsity_loss(self) -> SparsityLoss:
        p_open = self._gate_soft_prob
        l1_loss = torch.mean(p_open)
        return SparsityLoss(names=["gsg_softmax_l1"], values=[l1_loss])

    @property
    def selected_indices_candidate(self) -> torch.Tensor:
        logits_gate = (torch.matmul(self.logits_gate_feature.T, self.logits_gate_embedding.T)).T
        selected_open = torch.argmax(logits_gate, dim=0)
        indices = torch.arange(0, self.input_dim, device=self.device)
        indices[selected_open == 0] = -1
        return indices
