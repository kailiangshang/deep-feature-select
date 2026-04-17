from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import random
import torch
import torch.nn as nn

from anndata.experimental.pytorch import AnnLoader


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device: str = "cpu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim),
        )
        self.to(device)

    def forward(self, x):
        return self.net(x)


class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device: str = "cpu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
        )
        self.to(device)

    def forward(self, x):
        return self.net(x)


class AutoencoderHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device: str = "cpu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.to(device)

    def forward(self, x):
        return self.net(x)


@dataclass
class MetaData:
    name: str
    feature_dim: int
    train_loader: AnnLoader
    test_loader: AnnLoader
    label_mapping: dict
    cls_num: int
    random_state: int
    shuffle: bool
    batch_size: int
