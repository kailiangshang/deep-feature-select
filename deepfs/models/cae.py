from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfs.core.base import EncoderFeatureModule
from deepfs.core.types import TemperatureSchedule
from deepfs.core.utils import generate_gumbel_noise


class ConcreteAutoencoderModel(EncoderFeatureModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
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
        super().__init__(input_dim, output_dim, schedule, device)
        self.logits = nn.Parameter(torch.randn(input_dim, output_dim))
        self.temperature = torch.tensor(initial_temperature, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            gumbel_noise = generate_gumbel_noise(self.logits)
            y = F.softmax((self.logits + gumbel_noise) / self.temperature, dim=0)
            self._encoder_soft_prob = y.detach().clone()
        else:
            selected = torch.argmax(self.logits, dim=0)
            self._selected_indices = selected.detach().cpu().numpy()
            y = F.one_hot(selected, num_classes=self.input_dim).T.float()
        return torch.matmul(x, y)

    @property
    def selected_indices(self) -> torch.Tensor:
        return torch.argmax(self.logits, dim=0)
