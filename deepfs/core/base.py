from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .types import (
    EncoderDiagnostics,
    GateDiagnostics,
    SelectionResult,
    SparsityLoss,
    TemperatureSchedule,
)


class BaseSelector(nn.Module, ABC):

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def get_selection_result(self) -> SelectionResult:
        ...

    def update_temperature(self, epoch: int) -> None:
        pass

    def to(self, device, **kwargs):
        self.device = str(device) if not isinstance(device, str) else device
        return super().to(device, **kwargs)


class EncoderFeatureModule(BaseSelector):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        temperature_schedule: TemperatureSchedule,
        device: str = "cpu",
    ):
        super().__init__(device=device)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.temperature_schedule = temperature_schedule
        self.temperature = torch.tensor(temperature_schedule.initial)
        self._selected_indices: Optional[torch.Tensor] = None
        self._encoder_soft_prob: Optional[torch.Tensor] = None

    def update_temperature(self, epoch: int) -> None:
        t = self.temperature_schedule.get_temperature(epoch)
        self.temperature = torch.tensor(t)

    @property
    @abstractmethod
    def selected_indices(self) -> torch.Tensor:
        ...

    def get_selection_result(self) -> SelectionResult:
        indices = self.selected_indices
        mask = np.zeros(self.input_dim, dtype=bool)
        indices_np = indices.detach().cpu().numpy()
        valid = indices_np >= 0
        mask[indices_np[valid]] = True
        gate_probs = None
        if self._encoder_soft_prob is not None:
            gate_probs = self._encoder_soft_prob.detach().cpu().numpy()
        return SelectionResult(
            selected_indices=indices_np,
            selected_mask=mask,
            gate_probs=gate_probs,
        )

    def encoder_diagnostics(self) -> EncoderDiagnostics:
        soft_prob = self._encoder_soft_prob
        if soft_prob is None:
            raise ValueError("_encoder_soft_prob is not set")
        prob_np = soft_prob.detach().cpu().numpy()
        indices = self.selected_indices.detach().cpu().numpy()
        entropy = -np.sum(prob_np * np.log(prob_np + 1e-10), axis=-1)
        feature_overlap = int(np.sum(np.bincount(indices[indices >= 0]) > 1)) if len(indices[indices >= 0]) > 0 else 0
        return EncoderDiagnostics(
            selected_indices=indices,
            selection_entropy=entropy,
            feature_overlap=feature_overlap,
        )


class GateFeatureModule(BaseSelector):

    def __init__(
        self,
        input_dim: int,
        temperature_schedule: Optional[TemperatureSchedule] = None,
        device: str = "cpu",
    ):
        super().__init__(device=device)
        self.input_dim = input_dim
        self.temperature_schedule = temperature_schedule
        if temperature_schedule is not None:
            self.temperature = torch.tensor(temperature_schedule.initial)
        else:
            self.temperature = None
        self._selected_indices: Optional[torch.Tensor] = None
        self._gate_soft_prob: Optional[torch.Tensor] = None

    def update_temperature(self, epoch: int) -> None:
        if self.temperature_schedule is not None:
            t = self.temperature_schedule.get_temperature(epoch)
            self.temperature = torch.tensor(t)

    @abstractmethod
    def sparsity_loss(self) -> SparsityLoss:
        ...

    @property
    @abstractmethod
    def selected_indices_candidate(self) -> torch.Tensor:
        ...

    def get_selection_result(self) -> SelectionResult:
        indices = self.selected_indices_candidate
        mask = np.zeros(self.input_dim, dtype=bool)
        indices_np = indices.detach().cpu().numpy()
        valid = indices_np >= 0
        mask[indices_np[valid]] = True
        gate_probs = None
        if self._gate_soft_prob is not None:
            gate_probs = self._gate_soft_prob.detach().cpu().numpy()
        return SelectionResult(
            selected_indices=indices_np,
            selected_mask=mask,
            gate_probs=gate_probs,
        )

    def gate_diagnostics(self) -> GateDiagnostics:
        soft_prob = self._gate_soft_prob
        if soft_prob is None:
            raise ValueError("_gate_soft_prob is not set")
        prob_np = soft_prob.detach().cpu().numpy()
        threshold = 0.5
        open_mask = prob_np > threshold
        num_open = int(open_mask.sum())
        num_closed = int((~open_mask).sum())
        open_ratio = num_open / len(prob_np) if len(prob_np) > 0 else 0.0
        entropy = -np.sum(prob_np * np.log(prob_np + 1e-10) + (1 - prob_np) * np.log(1 - prob_np + 1e-10))
        return GateDiagnostics(
            gate_probs=prob_np,
            num_open=num_open,
            num_closed=num_closed,
            open_ratio=open_ratio,
            threshold=threshold,
            entropy=float(entropy),
        )

    def encoder_diagnostics(self) -> Optional[EncoderDiagnostics]:
        return None


GateBase = GateFeatureModule
EncoderBase = EncoderFeatureModule
