from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfs.core.base import GateFeatureModule
from deepfs.core.types import EncoderDiagnostics, SparsityLoss, TemperatureSchedule
from deepfs.core.utils import custom_one_hot, generate_gumbel_noise


class GumbelSoftmaxGateIndirectConcreteModel(GateFeatureModule):
    def __init__(
        self,
        input_dim: int,
        k: int,
        embedding_dim_encoder: int,
        embedding_dim_gate: int,
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
        self.k = k
        self.embedding_dim_encoder = embedding_dim_encoder
        self.embedding_dim_gate = embedding_dim_gate
        self.logits_encoder_embedding = nn.Parameter(
            torch.randn(input_dim, embedding_dim_encoder)
        )
        self.logits_encoder_feature = nn.Parameter(
            torch.randn(embedding_dim_encoder, k)
        )
        self.logits_gate_embedding = nn.Parameter(torch.randn(2, embedding_dim_gate))
        self.logits_gate_feature = nn.Parameter(torch.randn(embedding_dim_gate, k))
        self.temperature = torch.tensor(initial_temperature, device=device)
        self._encoder_soft_prob = None

    def _get_encoder_logits(self) -> torch.Tensor:
        return (torch.matmul(self.logits_encoder_feature.T, self.logits_encoder_embedding.T)).T

    def _get_gate_logits(self) -> torch.Tensor:
        return (torch.matmul(self.logits_gate_feature.T, self.logits_gate_embedding.T)).T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            logits_gate = self._get_gate_logits()
            gate_noise = generate_gumbel_noise(logits_gate)
            gate_soft_prob = F.softmax((logits_gate + gate_noise) / self.temperature, dim=0)
            p_open = gate_soft_prob[1, :]
            self._gate_soft_prob = p_open

            logits_enc = self._get_encoder_logits()
            enc_noise = generate_gumbel_noise(logits_enc)
            enc_soft_prob = F.softmax((logits_enc + enc_noise) / self.temperature, dim=0)
            self._encoder_soft_prob = enc_soft_prob.detach().clone()

            combined = enc_soft_prob * p_open.unsqueeze(0)
            return torch.matmul(x, combined)
        else:
            enc_logits = self._get_encoder_logits()
            gate_logits = self._get_gate_logits()
            selected = torch.argmax(enc_logits, dim=0)
            open_slots = torch.argmax(gate_logits, dim=0)
            selected[open_slots == 0] = -1
            self._selected_indices = selected.cpu().numpy()
            y = custom_one_hot(selected, self.input_dim).T
            return torch.matmul(x, y)

    def sparsity_loss(self) -> SparsityLoss:
        p_open = self._gate_soft_prob
        l1_loss = torch.mean(p_open)
        return SparsityLoss(names=["gsg_ipcae_l1"], values=[l1_loss])

    @property
    def selected_indices_candidate(self) -> torch.Tensor:
        enc_logits = self._get_encoder_logits()
        gate_logits = self._get_gate_logits()
        selected = torch.argmax(enc_logits, dim=0)
        open_slots = torch.argmax(gate_logits, dim=0)
        selected[open_slots == 0] = -1
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
