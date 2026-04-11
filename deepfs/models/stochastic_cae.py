from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfs.core.base import GateFeatureModule
from deepfs.core.types import EncoderDiagnostics, SparsityLoss, TemperatureSchedule
from deepfs.core.utils import custom_one_hot, generate_gumbel_noise


class StochasticGateConcreteModel(GateFeatureModule):
    def __init__(
        self,
        input_dim: int,
        k: int,
        sigma: float = 1.0,
        total_epochs: int = 100,
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
        self.k = k
        self.sigma = sigma
        self.mu_gate = nn.Parameter(0.01 * torch.randn(k))
        self.noise_gate = torch.randn(k, device=device)
        self.logits_encoder = nn.Parameter(torch.randn(input_dim, k))
        self.temperature = torch.tensor(initial_temperature, device=device)
        self._encoder_soft_prob = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            gate_prob = torch.clamp(
                self.mu_gate + self.sigma * self.noise_gate.normal_() + 0.5, 0.0, 1.0
            )
            self._gate_soft_prob = gate_prob

            gumbel_noise = generate_gumbel_noise(self.logits_encoder)
            enc_soft_prob = F.softmax((self.logits_encoder + gumbel_noise) / self.temperature, dim=0)
            self._encoder_soft_prob = enc_soft_prob.detach().clone()

            combined = enc_soft_prob * gate_prob.unsqueeze(0)
            return torch.matmul(x, combined)
        else:
            gate_prob = torch.clamp(self.mu_gate + 0.5, 0.0, 1.0)
            selected = torch.argmax(self.logits_encoder, dim=0)
            selected[gate_prob == 0] = -1
            self._selected_indices = selected.cpu().numpy()
            y = custom_one_hot(selected, self.input_dim).T
            return torch.matmul(x, y)

    def sparsity_loss(self) -> SparsityLoss:
        h = (self.mu_gate + 0.5) / self.sigma
        loss = torch.mean(0.5 * (1 + torch.erf(h / torch.sqrt(torch.tensor(2.0)))))
        return SparsityLoss(names=["stg_cae_sparsity"], values=[loss])

    @property
    def selected_indices_candidate(self) -> torch.Tensor:
        gate_prob = torch.clamp(self.mu_gate + 0.5, 0.0, 1.0)
        selected = torch.argmax(self.logits_encoder, dim=0)
        selected[gate_prob == 0] = -1
        return selected

    def encoder_diagnostics(self) -> EncoderDiagnostics:
        indices = self.selected_indices_candidate.detach().cpu().numpy()
        valid = indices[indices >= 0]
        unique, counts = np.unique(valid, return_counts=True)
        overlap = int(counts.sum() - len(unique)) if len(counts) > 0 else 0
        entropy = np.zeros(self.k)
        if self._encoder_soft_prob is not None:
            probs = self._encoder_soft_prob.detach().cpu().numpy()
            entropy = -np.sum(probs * np.log(probs + 1e-10), axis=0)
        return EncoderDiagnostics(
            selected_indices=indices,
            selection_entropy=entropy,
            feature_overlap=overlap,
        )
