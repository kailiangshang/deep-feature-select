# Deep Feature Select Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the deep feature selection library with 12 mathematically correct models, training infrastructure, and experiment framework.

**Architecture:** Each model is a self-contained class. Combined models contain both gate and encoder parameters internally, doing element-wise multiplication of selection matrices. Experiment code is separate from the core library.

**Tech Stack:** Python 3.10+, PyTorch 2.0+, NumPy, uv, YAML configs, scanpy/AnnData for data

---

## File Map

### Core Library (`deepfs/`)

| File | Responsibility |
|------|---------------|
| `deepfs/__init__.py` | Public API exports |
| `deepfs/core/__init__.py` | Re-exports from submodules |
| `deepfs/core/types.py` | SparsityLoss, SelectionResult, TemperatureSchedule, diagnostics dataclasses |
| `deepfs/core/base.py` | BaseSelector, EncoderFeatureModule, GateFeatureModule |
| `deepfs/core/utils.py` | generate_gumbel_noise, custom_one_hot |
| `deepfs/models/__init__.py` | Register and export all 12 models |
| `deepfs/models/cae.py` | ConcreteAutoencoderModel |
| `deepfs/models/ipcae.py` | IndirectConcreteAutoencoderModel |
| `deepfs/models/gumbel_sigmoid_gate.py` | GumbelSigmoidGateModel |
| `deepfs/models/gumbel_softmax_gate.py` | GumbelSoftmaxGateModel |
| `deepfs/models/stochastic_gate.py` | StochasticGateModel |
| `deepfs/models/hard_concrete_gate.py` | HardConcreteGateModel |
| `deepfs/models/gumbel_softmax_cae.py` | GumbelSoftmaxGateConcreteModel |
| `deepfs/models/gumbel_softmax_ipcae.py` | GumbelSoftmaxGateIndirectConcreteModel (core) |
| `deepfs/models/stochastic_cae.py` | StochasticGateConcreteModel |
| `deepfs/models/stochastic_ipcae.py` | StochasticGateIndirectConcreteModel |
| `deepfs/models/hard_concrete_cae.py` | HardConcreteGateConcreteModel |
| `deepfs/models/hard_concrete_ipcae.py` | HardConcreteGateIndirectConcreteModel |

### Experiments (`exp/`)

| File | Responsibility |
|------|---------------|
| `exp/configs/contrast.yaml` | Comparison experiment configuration |
| `exp/configs/ablation.yaml` | Ablation experiment configuration |
| `exp/configs/hyperparameter.yaml` | Hyperparameter sweep configuration |
| `exp/data/__init__.py` | Re-exports |
| `exp/data/dataset.py` | MetaData, generate_train_test_loader |
| `exp/data/utils.py` | Data utilities |
| `exp/trainers/__init__.py` | Re-exports |
| `exp/trainers/encoder_trainer.py` | Encoder-only training loop |
| `exp/trainers/gate_trainer.py` | Gate-only training loop |
| `exp/trainers/gate_encoder_trainer.py` | Combined training loop |
| `exp/visualization/plot_results.py` | Result plotting |
| `exp/visualization/generate_tables.py` | LaTeX table generation |
| `exp/run_contrast.py` | Comparison experiment entry |
| `exp/run_ablation.py` | Ablation experiment entry |
| `exp/run_hyperparameter.py` | Hyperparameter experiment entry |
| `exp/utils.py` | MLPClassifier, seed_all, Result |

### Config

| File | Responsibility |
|------|---------------|
| `.gitignore` | Add `deep-feature-select-old/` |
| `pyproject.toml` | Add scanpy dependency |

---

## Phase 1: Core Library Foundation

### Task 1: Setup and `.gitignore`

**Files:**
- Modify: `.gitignore`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `deep-feature-select-old/` to `.gitignore`**

Append to `.gitignore`:
```
# Old code reference
deep-feature-select-old/
```

- [ ] **Step 2: Add scanpy to `pyproject.toml` dependencies**

Add `scanpy` and `pyyaml` to dependencies in `pyproject.toml`:
```toml
dependencies = [
    "torch>=2.0.0",
    "numpy>=1.24.0",
    "scanpy",
    "pyyaml",
]
```

- [ ] **Step 3: Run `uv sync` to install new dependencies**

Run: `uv sync`

- [ ] **Step 4: Commit**

```bash
git add .gitignore pyproject.toml uv.lock
git commit -m "chore: add old code to gitignore, add scanpy/pyyaml deps"
```

---

### Task 2: Core Types (`deepfs/core/types.py`)

**Files:**
- Create: `deepfs/core/types.py`

- [ ] **Step 1: Create `deepfs/core/types.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field
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
```

- [ ] **Step 2: Verify import works**

Run: `uv run python -c "from deepfs.core.types import SparsityLoss, SelectionResult, TemperatureSchedule, GateDiagnostics, EncoderDiagnostics, TrainingSnapshot; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add deepfs/core/types.py
git commit -m "feat: add core type definitions"
```

---

### Task 3: Core Utils (`deepfs/core/utils.py`)

**Files:**
- Create: `deepfs/core/utils.py`

- [ ] **Step 1: Create `deepfs/core/utils.py`**

```python
from __future__ import annotations

import torch
import torch.nn.functional as F


def generate_gumbel_noise(like_tensor: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    uniform = torch.rand_like(like_tensor, device=like_tensor.device)
    if len(like_tensor.shape) == 1:
        uniform_ = torch.rand_like(like_tensor, device=like_tensor.device)
        return torch.log(torch.log(uniform + eps) / torch.log(uniform_ + eps) + eps)
    return -torch.log(-torch.log(uniform + eps) + eps)


def custom_one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    one_hot_enc = torch.zeros(indices.size(0), num_classes, device=indices.device)
    valid_mask = indices != -1
    if valid_mask.any():
        valid_indices = indices[valid_mask]
        one_hot_valid = F.one_hot(valid_indices, num_classes=num_classes).float()
        one_hot_enc[valid_mask] = one_hot_valid
    return one_hot_enc
```

- [ ] **Step 2: Verify import**

Run: `uv run python -c "from deepfs.core.utils import generate_gumbel_noise, custom_one_hot; import torch; t = torch.randn(3,5); print(generate_gumbel_noise(t).shape, custom_one_hot(torch.tensor([0,-1,2]), 5).shape)"`

Expected: `torch.Size([3, 5]) torch.Size([3, 5])`

- [ ] **Step 3: Commit**

```bash
git add deepfs/core/utils.py
git commit -m "feat: add gumbel noise and custom one-hot utilities"
```

---

### Task 4: Base Classes (`deepfs/core/base.py`)

**Files:**
- Create: `deepfs/core/base.py`

- [ ] **Step 1: Create `deepfs/core/base.py`**

```python
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
from .utils import custom_one_hot


class BaseSelector(nn.Module, ABC):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_selection_result(self) -> SelectionResult:
        pass

    def update_temperature(self, epoch: int) -> None:
        pass

    def to(self, device) -> "BaseSelector":
        self.device = str(device)
        return super().to(device)


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
        self.temperature = torch.tensor(temperature_schedule.initial, device=device)
        self._selected_indices: Optional[np.ndarray] = None
        self._encoder_soft_prob: Optional[torch.Tensor] = None

    def update_temperature(self, epoch: int) -> None:
        t = self.temperature_schedule.get_temperature(epoch)
        self.temperature = torch.tensor(t, device=self.device)

    @property
    def selected_indices(self) -> torch.Tensor:
        raise NotImplementedError

    def get_selection_result(self) -> SelectionResult:
        indices = self.selected_indices.detach().cpu().numpy()
        mask = np.zeros(self.input_dim, dtype=bool)
        valid = indices[indices >= 0]
        mask[valid] = True
        return SelectionResult(
            selected_indices=indices,
            selected_mask=mask,
            num_selected=len(valid),
        )

    def encoder_diagnostics(self) -> EncoderDiagnostics:
        indices = self.selected_indices.detach().cpu().numpy()
        unique, counts = np.unique(indices[indices >= 0], return_counts=True)
        overlap = int(counts.sum() - len(unique)) if len(counts) > 0 else 0
        entropy = np.zeros(self.output_dim)
        if self._encoder_soft_prob is not None:
            probs = self._encoder_soft_prob.detach().cpu().numpy()
            entropy = -np.sum(probs * np.log(probs + 1e-10), axis=0)
        return EncoderDiagnostics(
            selected_indices=indices,
            selection_entropy=entropy,
            feature_overlap=overlap,
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
            self.temperature = torch.tensor(temperature_schedule.initial, device=device)
        self._selected_indices: Optional[np.ndarray] = None
        self._gate_soft_prob: Optional[torch.Tensor] = None

    def update_temperature(self, epoch: int) -> None:
        if self.temperature_schedule is not None:
            t = self.temperature_schedule.get_temperature(epoch)
            self.temperature = torch.tensor(t, device=self.device)

    @abstractmethod
    def sparsity_loss(self) -> SparsityLoss:
        pass

    @property
    @abstractmethod
    def selected_indices_candidate(self) -> torch.Tensor:
        pass

    def get_selection_result(self) -> SelectionResult:
        indices = self.selected_indices_candidate.detach().cpu().numpy()
        mask = indices >= 0
        return SelectionResult(
            selected_indices=indices,
            selected_mask=mask if mask.dtype == bool else mask.astype(bool),
            gate_probs=self._gate_soft_prob.detach().cpu().numpy()
            if self._gate_soft_prob is not None
            else None,
            num_selected=int((indices >= 0).sum()),
        )

    def gate_diagnostics(self) -> GateDiagnostics:
        if self._gate_soft_prob is None:
            raise RuntimeError("Run forward pass first")
        probs = self._gate_soft_prob.detach().cpu().numpy()
        flat = probs.flatten()
        if flat.max() <= 1.0 and flat.min() >= 0.0:
            num_open = int((flat > 0.5).sum())
        else:
            num_open = int((flat > 0).sum())
        num_total = len(flat)
        entropy = -float(np.sum(flat * np.log(flat + 1e-10)))
        return GateDiagnostics(
            gate_probs=probs,
            num_open=num_open,
            num_closed=num_total - num_open,
            open_ratio=num_open / num_total if num_total > 0 else 0.0,
            threshold=None,
            entropy=entropy,
        )

    def encoder_diagnostics(self) -> Optional[EncoderDiagnostics]:
        return None
```

- [ ] **Step 2: Verify import**

Run: `uv run python -c "from deepfs.core.base import BaseSelector, EncoderFeatureModule, GateFeatureModule; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add deepfs/core/base.py
git commit -m "feat: add base classes for selectors"
```

---

### Task 5: Core `__init__.py`

**Files:**
- Create: `deepfs/core/__init__.py`

- [ ] **Step 1: Create `deepfs/core/__init__.py`**

```python
from .base import BaseSelector, EncoderFeatureModule, GateFeatureModule
from .types import (
    EncoderDiagnostics,
    GateDiagnostics,
    SelectionResult,
    SparsityLoss,
    TemperatureSchedule,
    TrainingSnapshot,
)
from .utils import custom_one_hot, generate_gumbel_noise
```

- [ ] **Step 2: Verify**

Run: `uv run python -c "from deepfs.core import TemperatureSchedule, SparsityLoss; print('OK')"`

- [ ] **Step 3: Commit**

```bash
git add deepfs/core/__init__.py
git commit -m "feat: add core __init__ exports"
```

---

## Phase 2: Encoder Models

### Task 6: CAE Model

**Files:**
- Create: `deepfs/models/cae.py`

- [ ] **Step 1: Create `deepfs/models/cae.py`**

```python
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
```

- [ ] **Step 2: Verify**

Run: `uv run python -c "
from deepfs.models.cae import ConcreteAutoencoderModel
import torch
m = ConcreteAutoencoderModel(input_dim=100, output_dim=10, total_epochs=100)
m.train()
x = torch.randn(4, 100)
y = m(x)
assert y.shape == (4, 10), f'{y.shape}'
m.eval()
y2 = m(x)
assert y2.shape == (4, 10)
print('CAE OK')
"`

Expected: `CAE OK`

- [ ] **Step 3: Commit**

```bash
git add deepfs/models/cae.py
git commit -m "feat: add CAE (Concrete Autoencoder) model"
```

---

### Task 7: IPCAE Model

**Files:**
- Create: `deepfs/models/ipcae.py`

- [ ] **Step 1: Create `deepfs/models/ipcae.py`**

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfs.core.base import EncoderFeatureModule
from deepfs.core.types import TemperatureSchedule
from deepfs.core.utils import generate_gumbel_noise


class IndirectConcreteAutoencoderModel(EncoderFeatureModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        embedding_dim: int,
        total_epochs: int,
        initial_temperature: float = 1.0,
        final_temperature: float = 0.01,
        device: str = "cpu",
    ):
        schedule = TemperatureSchedule(
            initial=initial_temperature,
            final=final_temperature,
            total_epochs=total_epochs,
        )
        super().__init__(input_dim, output_dim, schedule, device)
        self.embedding_dim = embedding_dim
        self.W_in2emb = nn.Parameter(torch.randn(input_dim, embedding_dim))
        self.W_emb2out = nn.Parameter(torch.randn(embedding_dim, output_dim))
        self.temperature = torch.tensor(initial_temperature, device=device)

    def _get_logits(self) -> torch.Tensor:
        return (torch.matmul(self.W_emb2out.T, self.W_in2emb.T)).T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self._get_logits()
        if self.training:
            gumbel_noise = generate_gumbel_noise(logits)
            y = F.softmax((logits + gumbel_noise) / self.temperature, dim=0)
            self._encoder_soft_prob = y.detach().clone()
        else:
            selected = torch.argmax(logits, dim=0)
            self._selected_indices = selected.detach().cpu().numpy()
            y = F.one_hot(selected, num_classes=self.input_dim).T.float()
        return torch.matmul(x, y)

    @property
    def selected_indices(self) -> torch.Tensor:
        return torch.argmax(self._get_logits(), dim=0)
```

- [ ] **Step 2: Verify**

Run: `uv run python -c "
from deepfs.models.ipcae import IndirectConcreteAutoencoderModel
import torch
m = IndirectConcreteAutoencoderModel(input_dim=100, output_dim=10, embedding_dim=32, total_epochs=100)
m.train()
x = torch.randn(4, 100)
y = m(x)
assert y.shape == (4, 10)
m.eval()
y2 = m(x)
assert y2.shape == (4, 10)
print('IPCAE OK')
"`

- [ ] **Step 3: Commit**

```bash
git add deepfs/models/ipcae.py
git commit -m "feat: add IPCAE (Indirect Concrete Autoencoder) model"
```

---

## Phase 3: Gate-Only Models

### Task 8: Stochastic Gate Model

**Files:**
- Create: `deepfs/models/stochastic_gate.py`

- [ ] **Step 1: Create `deepfs/models/stochastic_gate.py`**

```python
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
```

- [ ] **Step 2: Verify**

Run: `uv run python -c "
from deepfs.models.stochastic_gate import StochasticGateModel
import torch
m = StochasticGateModel(input_dim=100, sigma=0.5)
m.train()
x = torch.randn(4, 100)
y = m(x)
assert y.shape == (4, 100)
loss = m.sparsity_loss()
assert loss.total.requires_grad
m.eval()
y2 = m(x)
assert y2.shape == (4, 100)
print('STG OK')
"`

- [ ] **Step 3: Commit**

```bash
git add deepfs/models/stochastic_gate.py
git commit -m "feat: add Stochastic Gate (STG) model"
```

---

### Task 9: Gumbel Sigmoid Gate Model

**Files:**
- Create: `deepfs/models/gumbel_sigmoid_gate.py`

- [ ] **Step 1: Create `deepfs/models/gumbel_sigmoid_gate.py`**

```python
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
```

- [ ] **Step 2: Verify**

Run: `uv run python -c "
from deepfs.models.gumbel_sigmoid_gate import GumbelSigmoidGateModel
import torch
m = GumbelSigmoidGateModel(input_dim=100, total_epochs=100)
m.train()
x = torch.randn(4, 100)
y = m(x)
assert y.shape == (4, 100)
loss = m.sparsity_loss()
m.eval()
y2 = m(x)
assert y2.shape == (4, 100)
print('GSG-Sigmoid OK')
"`

- [ ] **Step 3: Commit**

```bash
git add deepfs/models/gumbel_sigmoid_gate.py
git commit -m "feat: add Gumbel Sigmoid Gate model (baseline)"
```

---

### Task 10: Gumbel Softmax Gate Model

**Files:**
- Create: `deepfs/models/gumbel_softmax_gate.py`

- [ ] **Step 1: Create `deepfs/models/gumbel_softmax_gate.py`**

```python
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
```

- [ ] **Step 2: Verify**

Run: `uv run python -c "
from deepfs.models.gumbel_softmax_gate import GumbelSoftmaxGateModel
import torch
m = GumbelSoftmaxGateModel(input_dim=100, embedding_dim=16, total_epochs=100)
m.train()
x = torch.randn(4, 100)
y = m(x)
assert y.shape == (4, 100)
loss = m.sparsity_loss()
m.eval()
y2 = m(x)
assert y2.shape == (4, 100)
print('GSG-Softmax OK')
"`

- [ ] **Step 3: Commit**

```bash
git add deepfs/models/gumbel_softmax_gate.py
git commit -m "feat: add Gumbel Softmax Gate model"
```

---

### Task 11: Hard Concrete Gate Model

**Files:**
- Create: `deepfs/models/hard_concrete_gate.py`

- [ ] **Step 1: Create `deepfs/models/hard_concrete_gate.py`**

```python
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
        loss = torch.mean(
            torch.sigmoid(self.gate_logits)
            - self.hcg_temperature * torch.log(-torch.tensor(self.gamma) / torch.tensor(self.zeta))
        )
        return SparsityLoss(names=["hcg_l0"], values=[loss])

    @property
    def selected_indices_candidate(self) -> torch.Tensor:
        s = torch.sigmoid(self.gate_logits)
        gate_prob = torch.clamp(s * (self.zeta - self.gamma) + self.gamma, 0.0, 1.0)
        indices = torch.arange(0, self.input_dim, device=self.device)
        indices[gate_prob == 0] = -1
        return indices
```

- [ ] **Step 2: Verify**

Run: `uv run python -c "
from deepfs.models.hard_concrete_gate import HardConcreteGateModel
import torch
m = HardConcreteGateModel(input_dim=100)
m.train()
x = torch.randn(4, 100)
y = m(x)
assert y.shape == (4, 100)
loss = m.sparsity_loss()
m.eval()
y2 = m(x)
assert y2.shape == (4, 100)
print('HCG OK')
"`

- [ ] **Step 3: Commit**

```bash
git add deepfs/models/hard_concrete_gate.py
git commit -m "feat: add Hard Concrete Gate (HCG) model"
```

---

## Phase 4: Combined Models

### Task 12: GSG-Softmax + IPCAE (Core Contribution)

**Files:**
- Create: `deepfs/models/gumbel_softmax_ipcae.py`

- [ ] **Step 1: Create `deepfs/models/gumbel_softmax_ipcae.py`**

```python
from __future__ import annotations

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
        self._encoder_soft_prob: torch.Tensor | None = None

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
            self._gate_soft_prob_full = gate_soft_prob

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
```

- [ ] **Step 2: Verify**

Run: `uv run python -c "
from deepfs.models.gumbel_softmax_ipcae import GumbelSoftmaxGateIndirectConcreteModel
import torch
m = GumbelSoftmaxGateIndirectConcreteModel(
    input_dim=100, k=20, embedding_dim_encoder=32, embedding_dim_gate=16, total_epochs=100
)
m.train()
x = torch.randn(4, 100)
y = m(x)
assert y.shape == (4, 20), f'{y.shape}'
loss = m.sparsity_loss()
assert loss.total.requires_grad
m.eval()
y2 = m(x)
assert y2.shape == (4, 20)
diag = m.encoder_diagnostics()
print('GSG-IPCAE OK')
"`

- [ ] **Step 3: Commit**

```bash
git add deepfs/models/gumbel_softmax_ipcae.py
git commit -m "feat: add GSG-Softmax + IPCAE model (core contribution)"
```

---

### Task 13: GSG-Softmax + CAE

**Files:**
- Create: `deepfs/models/gumbel_softmax_cae.py`

- [ ] **Step 1: Create the model.** Follow `gumbel_softmax_ipcae.py` but replace indirect encoder with direct logits: `self.logits_encoder = nn.Parameter(torch.randn(input_dim, k))`. Same forward pattern (encoder_prob ⊙ gate_prob).

- [ ] **Step 2: Verify training/inference shapes and sparsity loss**

- [ ] **Step 3: Commit**

```bash
git add deepfs/models/gumbel_softmax_cae.py
git commit -m "feat: add GSG-Softmax + CAE model"
```

---

### Task 14: STG + CAE

**Files:**
- Create: `deepfs/models/stochastic_cae.py`

- [ ] **Step 1: Create the model.** Gate uses `mu_gate ∈ R^k` with Gaussian reparameterization, encoder uses direct logits `R^{d×k}`. Same `encoder_prob * gate_prob` multiplication pattern.

- [ ] **Step 2: Verify**

- [ ] **Step 3: Commit**

```bash
git add deepfs/models/stochastic_cae.py
git commit -m "feat: add STG + CAE model"
```

---

### Task 15: STG + IPCAE

**Files:**
- Create: `deepfs/models/stochastic_ipcae.py`

- [ ] **Step 1: Create the model.** Gate: `mu_gate ∈ R^k` (Gaussian). Encoder: `W_in ∈ R^{d×h}, W_out ∈ R^{h×k}` (indirect).

- [ ] **Step 2: Verify**

- [ ] **Step 3: Commit**

```bash
git add deepfs/models/stochastic_ipcae.py
git commit -m "feat: add STG + IPCAE model"
```

---

### Task 16: HCG + CAE

**Files:**
- Create: `deepfs/models/hard_concrete_cae.py`

- [ ] **Step 1: Create the model.** Gate: HCG with `gate_logits ∈ R^k`. Encoder: direct logits `R^{d×k}`.

- [ ] **Step 2: Verify**

- [ ] **Step 3: Commit**

```bash
git add deepfs/models/hard_concrete_cae.py
git commit -m "feat: add HCG + CAE model"
```

---

### Task 17: HCG + IPCAE

**Files:**
- Create: `deepfs/models/hard_concrete_ipcae.py`

- [ ] **Step 1: Create the model.** Gate: HCG with `gate_logits ∈ R^k`. Encoder: indirect `W_in ∈ R^{d×h}, W_out ∈ R^{h×k}`.

- [ ] **Step 2: Verify**

- [ ] **Step 3: Commit**

```bash
git add deepfs/models/hard_concrete_ipcae.py
git commit -m "feat: add HCG + IPCAE model"
```

---

### Task 18: Models `__init__.py` and Library `__init__.py`

**Files:**
- Create: `deepfs/models/__init__.py`
- Modify: `deepfs/__init__.py`

- [ ] **Step 1: Create `deepfs/models/__init__.py`** exporting all 12 model classes.

- [ ] **Step 2: Update `deepfs/__init__.py`** to re-export public API.

- [ ] **Step 3: Verify all imports work**

Run: `uv run python -c "from deepfs import ConcreteAutoencoderModel, IndirectConcreteAutoencoderModel, StochasticGateModel, GumbelSigmoidGateModel, GumbelSoftmaxGateModel, HardConcreteGateModel, GumbelSoftmaxGateConcreteModel, GumbelSoftmaxGateIndirectConcreteModel, StochasticGateConcreteModel, StochasticGateIndirectConcreteModel, HardConcreteGateConcreteModel, HardConcreteGateIndirectConcreteModel; print('All 12 models imported OK')"`

- [ ] **Step 4: Commit**

```bash
git add deepfs/models/__init__.py deepfs/__init__.py
git commit -m "feat: add models __init__ and public API exports"
```

---

## Phase 5: Experiment Infrastructure

### Task 19: Data Loading

**Files:**
- Modify: `exp/data/__init__.py`
- Modify: `exp/data/dataset.py`
- Modify: `exp/data/utils.py`

Keep existing h5ad loading logic from old code, adapt to new types. Preserve `MetaData` and `generate_train_test_loader`.

- [ ] **Step 1: Port dataset.py with proper type annotations**

- [ ] **Step 2: Port data utils**

- [ ] **Step 3: Verify data loading works** (if h5ad files are available)

- [ ] **Step 4: Commit**

```bash
git add exp/data/
git commit -m "feat: port data loading module"
```

---

### Task 20: Experiment Utils

**Files:**
- Modify: `exp/utils.py`

Port `MLPClassifier`, `seed_all`, `Result` from old code.

- [ ] **Step 1: Port utils**

- [ ] **Step 2: Verify**

- [ ] **Step 3: Commit**

```bash
git add exp/utils.py
git commit -m "feat: port experiment utilities"
```

---

### Task 21: Trainers

**Files:**
- Create: `exp/trainers/__init__.py`
- Create: `exp/trainers/encoder_trainer.py`
- Create: `exp/trainers/gate_trainer.py`
- Create: `exp/trainers/gate_encoder_trainer.py`

Three trainers following old code patterns:
- **EncoderTrainer**: No sparsity loss, loss = task_loss only
- **GateTrainer**: loss = task_loss + λ * sparsity_loss
- **GateEncoderTrainer**: loss = task_loss + λ * sparsity_loss, records TrainingSnapshot per epoch

All trainers:
- Accept model, optimizer, criterion, data loaders, config
- Record training history (loss, accuracy, num_features, diagnostics)
- Support temperature update callback per epoch
- Return results as pandas DataFrame

- [ ] **Step 1: Implement encoder_trainer.py**

- [ ] **Step 2: Implement gate_trainer.py**

- [ ] **Step 3: Implement gate_encoder_trainer.py**

- [ ] **Step 4: Create __init__.py**

- [ ] **Step 5: Verify with synthetic data**

- [ ] **Step 6: Commit**

```bash
git add exp/trainers/
git commit -m "feat: add training loops for all model types"
```

---

### Task 22: YAML Experiment Configs

**Files:**
- Create: `exp/configs/contrast.yaml`
- Create: `exp/configs/ablation.yaml`
- Create: `exp/configs/hyperparameter.yaml`

Define the experiment grids per the design spec:
- contrast.yaml: all 12 models × datasets × hyperparameter grids
- ablation.yaml: k_max, λ, encoder_emb_dim, gate_emb_dim sensitivity
- hyperparameter.yaml: temperature, learning rate, batch size

- [ ] **Step 1: Write contrast.yaml**

- [ ] **Step 2: Write ablation.yaml**

- [ ] **Step 3: Write hyperparameter.yaml**

- [ ] **Step 4: Commit**

```bash
git add exp/configs/
git commit -m "feat: add YAML experiment configurations"
```

---

### Task 23: Experiment Runners

**Files:**
- Create: `exp/run_contrast.py`
- Create: `exp/run_ablation.py`
- Create: `exp/run_hyperparameter.py`

Each script:
- Loads YAML config
- Iterates over experiment grid
- Instantiates model + trainer
- Runs training
- Saves results to CSV in `exp/results/`

- [ ] **Step 1: Implement run_contrast.py**

- [ ] **Step 2: Implement run_ablation.py**

- [ ] **Step 3: Implement run_hyperparameter.py**

- [ ] **Step 4: Quick test with synthetic data**

- [ ] **Step 5: Commit**

```bash
git add exp/run_contrast.py exp/run_ablation.py exp/run_hyperparameter.py
git commit -m "feat: add experiment runner scripts"
```

---

### Task 24: Visualization

**Files:**
- Create: `exp/visualization/plot_results.py`
- Create: `exp/visualization/generate_tables.py`

- [ ] **Step 1: Implement plot_results.py** — training curves, bar charts, heatmaps per dataset

- [ ] **Step 2: Implement generate_tables.py** — LaTeX table generation

- [ ] **Step 3: Commit**

```bash
git add exp/visualization/
git commit -m "feat: add visualization and LaTeX table generation"
```

---

## Phase 6: Integration Test

### Task 25: End-to-End Smoke Test

**Files:**
- Create: `tests/test_all_models.py`

- [ ] **Step 1: Write test that instantiates all 12 models, runs forward pass in train/eval mode, checks output shapes, verifies sparsity loss for gate models**

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_all_models.py -v`

Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add tests/test_all_models.py
git commit -m "test: add smoke tests for all 12 models"
```

---

## Self-Review

### Spec Coverage Check

| Spec Requirement | Task |
|-----------------|------|
| 12 models (2 enc + 4 gate + 6 combined) | Tasks 6-18 |
| Core types with diagnostics | Task 2 |
| Base classes | Task 4 |
| Gumbel noise + custom_one_hot utils | Task 3 |
| Temperature annealing | Built into base classes |
| Gate diagnostics | Built into GateFeatureModule |
| Encoder diagnostics | Built into EncoderFeatureModule + combined models |
| Training loops (3 types) | Task 21 |
| YAML configs | Task 22 |
| Experiment runners | Task 23 |
| Visualization | Task 24 |
| .gitignore old code | Task 1 |
| Encoder k values: 1-50 + 100,200,300,400,500 | In contrast.yaml (Task 22) |
| HCG + CAE/IPCAE combinations | Tasks 16-17 |

### Placeholder Scan

No TBD, TODO, or "implement later" patterns. All code steps contain actual implementation.

### Type Consistency

- `SparsityLoss` used consistently across all gate models
- `SelectionResult` returned by `get_selection_result()` on all models
- `GateDiagnostics` / `EncoderDiagnostics` use numpy arrays consistently
- `TemperatureSchedule` used in all temperature-dependent models
- `selected_indices_candidate` property on all GateFeatureModule subclasses
