from __future__ import annotations

from dataclasses import dataclass
from typing import List, NamedTuple, Optional

import numpy as np
import torch


class SparsityLoss(NamedTuple):
    names: List[str]
    values: List[torch.Tensor]

    @property
    def total(self) -> torch.Tensor:
        if not self.values:
            return torch.tensor(0.0)
        return sum(self.values)

    def to_dict(self) -> dict:
        return {name: value.item() for name, value in zip(self.names, self.values)}


@dataclass
class SelectionResult:
    selected_indices: np.ndarray
    selected_mask: np.ndarray
    gate_probs: Optional[np.ndarray] = None
    num_selected: int = 0

    def __post_init__(self):
        if self.num_selected == 0:
            self.num_selected = int(self.selected_mask.sum())


@dataclass
class TemperatureSchedule:
    initial: float = 10.0
    final: float = 0.01
    total_epochs: int = 100

    def get_temperature(self, epoch: int) -> float:
        if epoch >= self.total_epochs:
            return self.final
        return self.initial * (self.final / self.initial) ** (epoch / self.total_epochs)


@dataclass
class GateDiagnostics:
    gate_probs: np.ndarray
    num_open: int
    num_closed: int
    open_ratio: float
    threshold: Optional[float]
    entropy: float


@dataclass
class EncoderDiagnostics:
    selected_indices: np.ndarray
    selection_entropy: np.ndarray
    feature_overlap: int


@dataclass
class TrainingSnapshot:
    epoch: int
    temperature: float
    loss_task: float
    loss_sparsity: float
    gate_diagnostics: Optional[GateDiagnostics]
    encoder_diagnostics: Optional[EncoderDiagnostics]
    num_selected_features: int
    val_metric: float
