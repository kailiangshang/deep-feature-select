# DeepFS - Deep Feature Selection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular deep learning library for differentiable feature selection, implementing 12 feature selection models built on 4 gates and 2 encoders.

## Core Contribution: GSG-IPCAE

**Gumbel Softmax Gate + Indirect Parametric Concrete Autoencoder** — a two-stage feature selection method:

1. **Encoder (IPCAE)**: Selects `k` candidate features from `D` input features via indirect low-rank parameterization
2. **Gate (GSG-Softmax)**: Further filters the `k` candidates via a 2-class (open/closed) Gumbel Softmax over embedding space

The combined model computes `output = X @ (encoder_prob ⊙ gate_prob)`, where both distributions are learned end-to-end with Gumbel-Softmax reparameterization.

## Models

| Category | Models | Output |
|----------|--------|--------|
| **Encoders** (2) | CAE, IPCAE | Selects exactly `k` features from `D` |
| **Gates** (4) | STG, GSG-Sigmoid, GSG-Softmax, HCG | Sparse binary mask over all `D` features |
| **Combined** (6) | GSG-Softmax+CAE, GSG-Softmax+IPCAE, STG+CAE, STG+IPCAE, HCG+CAE, HCG+IPCAE | Two-stage: encoder picks `k` slots, gate opens/closes each slot |

## Installation

```bash
git clone https://github.com/your-repo/deep-feature-select.git
cd deep-feature-select
uv sync
```

## Quick Start

```python
import torch
from deepfs import GumbelSoftmaxGateIndirectConcreteModel

model = GumbelSoftmaxGateIndirectConcreteModel(
    input_dim=58482,
    k=50,
    embedding_dim_encoder=32,
    embedding_dim_gate=16,
    total_epochs=1000,
)

model.train()
x = torch.randn(32, 58482)
y = model(x)  # (32, 50)

sparsity_loss = model.sparsity_loss().total
model.update_temperature(epoch=0)

model.eval()
y = model(x)  # hard selection
result = model.get_selection_result()
print(f"Selected {result.num_selected} features")
```

### With a Downstream Classifier

```python
from deepfs import StochasticGateModel
from exp.trainers import GateTrainer
from exp.utils import MLPClassifier

model = StochasticGateModel(input_dim=1000, sigma=0.5)
classifier = MLPClassifier(input_dim=1000, hidden_dim=128, output_dim=5)
trainer = GateTrainer(model, classifier, sparse_loss_weight=1.0, lr=1e-4)
df = trainer.fit(train_loader, epochs=100, test_loader=test_loader)
```

## Algorithms

### Gates

| Gate | Paper | Sparsity Loss |
|------|-------|---------------|
| **STG** (Stochastic Gate) | [Yamada et al., ICML 2020](http://proceedings.mlr.press/v119/yamada20a.html) | Gaussian CDF (erf) |
| **GSG-Sigmoid** | [Jang et al., 2016](https://arxiv.org/abs/1611.01144) | L1 on sigmoid probabilities |
| **GSG-Softmax** | [Jang et al., 2016](https://arxiv.org/abs/1611.01144) | L1 on open-class probability |
| **HCG** (Hard Concrete Gate) | [Louizos et al., ICLR 2018](https://openreview.net/forum?id=H1Y8hhg0b) | L0 approximation |

### Encoders

| Encoder | Paper | Key Idea |
|---------|-------|----------|
| **CAE** (Concrete Autoencoder) | [Balin et al., ICML 2019](https://proceedings.mlr.press/v97/balin19a.html) | Direct logits `D × k` |
| **IPCAE** (Indirect Concrete AE) | This work | Low-rank: `D × h` and `h × k` |

## Experiments

YAML configs in `exp/configs/`:

```bash
# Comparison experiments (all 12 models)
python exp/run_contrast.py --config exp/configs/contrast.yaml

# Ablation studies (GSG-IPCAE sensitivity)
python exp/run_ablation.py --config exp/configs/ablation.yaml

# Hyperparameter sweeps
python exp/run_hyperparameter.py --config exp/configs/hyperparameter.yaml
```

Results saved to `exp/results/`.

## Project Structure

```
deepfs/
├── core/                          # Base classes and types
│   ├── types.py                   # SparsityLoss, SelectionResult, TemperatureSchedule, diagnostics
│   ├── base.py                    # BaseSelector, EncoderFeatureModule, GateFeatureModule
│   └── utils.py                   # generate_gumbel_noise, custom_one_hot
├── models/                        # 12 self-contained models
│   ├── cae.py                     # ConcreteAutoencoderModel
│   ├── ipcae.py                   # IndirectConcreteAutoencoderModel
│   ├── stochastic_gate.py         # StochasticGateModel
│   ├── gumbel_sigmoid_gate.py     # GumbelSigmoidGateModel
│   ├── gumbel_softmax_gate.py     # GumbelSoftmaxGateModel
│   ├── hard_concrete_gate.py      # HardConcreteGateModel
│   ├── gumbel_softmax_cae.py      # GumbelSoftmaxGateConcreteModel
│   ├── gumbel_softmax_ipcae.py    # GumbelSoftmaxGateIndirectConcreteModel (core)
│   ├── stochastic_cae.py          # StochasticGateConcreteModel
│   ├── stochastic_ipcae.py        # StochasticGateIndirectConcreteModel
│   ├── hard_concrete_cae.py       # HardConcreteGateConcreteModel
│   └── hard_concrete_ipcae.py     # HardConcreteGateIndirectConcreteModel
└── __init__.py                    # Public API (all 12 models + core types)

exp/
├── configs/                       # YAML experiment configs
├── data/                          # h5ad data loading (AnnData/scanpy)
├── trainers/                      # EncoderTrainer, GateTrainer, GateEncoderTrainer
├── visualization/                 # Plotting and LaTeX table generation
├── run_contrast.py                # Comparison experiment runner
├── run_ablation.py                # Ablation experiment runner
├── run_hyperparameter.py          # Hyperparameter sweep runner
└── utils.py                       # seed_all, MLPClassifier, Result

tests/
└── test_all_models.py             # Smoke tests (17 tests, all models)
```

## Citation

```bibtex
@misc{deepfs2026,
  author = {Your Name},
  title = {DeepFS: Deep Feature Selection via Gumbel Softmax Gating},
  year = {2026},
  url = {https://github.com/your-repo/deep-feature-select}
}
```

## License

MIT License
