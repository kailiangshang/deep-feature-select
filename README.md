# DeepFS - Deep Feature Selection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular deep learning library for differentiable feature selection, implementing 12 models built on 4 gates and 2 encoders.

## Core Contribution

Our contribution is the **combination** of two existing techniques into a unified two-stage feature selection framework:

- **Gate**: Gumbel Softmax Gate (GSG-Softmax), applying the Gumbel-Softmax trick ([Jang et al., 2016](https://arxiv.org/abs/1611.01144)) to model the open/closed gate decision as 2-class classification
- **Encoder**: Indirect Parametric Concrete Autoencoder (IPCAE) ([Balin et al., ICML 2019](https://proceedings.mlr.press/v97/balin19a.html)), using low-rank parameterization for feature-to-slot assignment

Neither component is novel on its own. The novelty lies in how they work together to solve three fundamental problems in gate-based feature selection.

## Problem: Why Existing Methods Need Thresholds

Existing methods (STG, HCG) model the gate as a continuous value $p_i \in (0,1)$ and require a threshold $\theta$ at test time:

$$S = \{i : p_i > \theta\}$$

This causes three problems:

**1. Threshold sensitivity.** The selected feature set changes with $\theta$. There is no principled way to choose $\theta$.

**2. Gradient ambiguity.** For sigmoid gates, $\nabla_\mu \sigma(\mu) = \sigma(\mu)(1-\sigma(\mu))$, which is maximized at $\mu=0$ (i.e., $p=0.5$). Gates near the decision boundary receive the strongest gradients, which keeps them stuck in the ambiguous zone rather than pushing them toward 0 or 1.

**3. Entangled decisions.** "Which feature to select?" and "Should this slot be open?" are conflated into a single continuous value. There is no separate mechanism to answer each question.

## Solution: Two-Stage Binary Classification

### Key Insight

The gate decision is fundamentally binary — a feature is either selected or not. Prior work used Gumbel Softmax for multi-class feature-to-slot assignment (in CAE) but never for the open/closed decision itself. We apply Gumbel Softmax to both stages:

```
Prior methods:
  [Feature-to-slot assignment] ← Gumbel Softmax (multi-class classification) ✓
  [Open/closed decision]       ← Sigmoid (continuous regression) ✗  →  threshold needed

Our method:
  [Feature-to-slot assignment] ← Gumbel Softmax via IPCAE (multi-class) ✓
  [Open/closed decision]       ← Gumbel Softmax via GSG (2-class binary) ✓  →  argmax, no threshold
```

### Architecture

Given input $X \in \mathbb{R}^{B \times D}$ with $B$ samples and $D$ features, select up to $k$ features:

```
Input X ∈ R^{B×D}
    │
    ├──► Encoder (IPCAE): assigns each of k slots to a feature
    │    Parameters: W_in ∈ R^{D×h}, W_out ∈ R^{h×k}
    │    Logits_enc = (W_out^T @ W_in^T)^T ∈ R^{D×k}
    │
    ├──► Gate (GSG-Softmax): binary open/close per slot
    │    Parameters: G_emb ∈ R^{2×h'}, G_feat ∈ R^{h'×k}
    │    Logits_gate = (G_feat^T @ G_emb^T)^T ∈ R^{2×k}
    │
    └──► Combine: Output = X @ (P ⊙ g) ∈ R^{B×k}
```

### Training (Differentiable)

Both modules use the Gumbel-Softmax reparameterization trick to produce differentiable soft outputs:

**Encoder** — each column of $P$ is a soft distribution over $D$ features for one slot:

$$P = \text{Softmax}\left(\frac{\text{Logits}_{enc} + G_1}{\tau}, \text{dim}=0\right) \in \mathbb{R}^{D \times k}$$

**Gate** — each slot gets a soft 2-class probability (closed vs. open):

$$\hat{g} = \text{Softmax}\left(\frac{\text{Logits}_{gate} + G_2}{\tau}, \text{dim}=0\right) \in \mathbb{R}^{2 \times k}$$

$$g = \hat{g}[1,:] \in \mathbb{R}^k \quad \text{(open-class probability)}$$

where $G_1, G_2$ are i.i.d. Gumbel noise: $G = -\log(-\log(U))$, $U \sim \text{Uniform}(0,1)$.

**Combined output**:

$$Y = X \cdot (P \odot g) \in \mathbb{R}^{B \times k}$$

where $g$ is broadcast along the feature dimension. The element-wise product $P \odot g$ zeros out columns (slots) where the gate is closed.

**Training loss**:

$$\mathcal{L} = \mathcal{L}_{task}(Y, \text{target}) + \lambda \cdot \underbrace{\frac{1}{k}\sum_{j=1}^{k} g_j}_{\text{sparsity (L1 on open probability)}}$$

The sparsity term penalizes the mean open probability, encouraging the gate to close unnecessary slots.

### Inference (Discrete)

At test time, both modules use hard argmax — no threshold parameter:

**Encoder**: each slot selects exactly one feature:

$$\text{slot}_j \mapsto \arg\max_{i} \text{Logits}_{enc}[i, j]$$

**Gate**: each slot is binary open or closed:

$$g_j = \begin{cases} 1 & \text{if } \arg\max_{c} \text{Logits}_{gate}[c, j] = 1 \text{ (open class wins)} \\ 0 & \text{if } \arg\max_{c} \text{Logits}_{gate}[c, j] = 0 \text{ (closed class wins)} \end{cases}$$

Closed slots are masked out. The final selected features are $\{\text{slot}_j : g_j = 1\}$.

### Temperature Annealing

The temperature $\tau$ controls the softness of the Gumbel-Softmax output. It follows an exponential decay schedule:

$$\tau_t = \tau_{init} \cdot \left(\frac{\tau_{final}}{\tau_{init}}\right)^{t/T}$$

- High $\tau$ (early training): soft probabilities → exploration, gradients flow through all features
- Low $\tau$ (late training): near-discrete outputs → exploitation, gates commit to decisions
- At $\tau \to 0$: Gumbel-Softmax converges to one-hot / argmax

### Why IPCAE Instead of CAE

CAE ([Balin et al., 2019](https://proceedings.mlr.press/v97/balin19a.html)) parameterizes the encoder logits directly as $\text{Logits} \in \mathbb{R}^{D \times k}$, requiring $O(Dk)$ parameters. When $D$ is large (e.g., 58,482 genes in single-cell data), this becomes problematic.

IPCAE ([Balin et al., 2019](https://proceedings.mlr.press/v97/balin19a.html)) factorizes through a low-rank embedding:

$$\text{Logits} = (W_{out}^T \cdot W_{in}^T)^T, \quad W_{in} \in \mathbb{R}^{D \times h}, \; W_{out} \in \mathbb{R}^{h \times k}$$

This reduces parameters from $O(Dk)$ to $O(Dh + hk)$ where $h \ll k \ll D$, preventing overfitting in high-dimensional settings.

### Summary: How Each Problem Is Solved

| Problem | Prior methods | GSG-Softmax + IPCAE |
|---------|--------------|---------------------|
| Threshold sensitivity | $S = \{i : p_i > \theta\}$, different $\theta$ → different $S$ | argmax of 2-class softmax → binary by construction, no $\theta$ |
| Gradient ambiguity | $\sigma'(\mu)$ peaks at $p=0.5$, keeping gates stuck | Gumbel noise + $\tau$ annealing → stochastic exploration then commitment |
| Entangled decisions | Single $p_i$ encodes both "which" and "open/close" | Gate (open/close) and encoder (which feature) are separate modules |
| High-dimensional collapse | $O(Dk)$ parameters | IPCAE low-rank: $O(Dh+hk)$ |

## Models

| Category | Models | Output |
|----------|--------|--------|
| **Encoders** (2) | CAE, IPCAE | Selects exactly `k` features from `D` |
| **Gates** (4) | STG, GSG-Sigmoid, GSG-Softmax, HCG | Sparse binary mask over all `D` features |
| **Combined** (6) | GSG-Softmax+CAE, **GSG-Softmax+IPCAE (Ours)**, STG+CAE, STG+IPCAE, HCG+CAE, HCG+IPCAE | Two-stage: encoder picks `k` slots, gate opens/closes each |

## Algorithms

### Gates

| Gate | Paper | Sparsity Loss |
|------|-------|---------------|
| **STG** (Stochastic Gate) | [Yamada et al., ICML 2020](http://proceedings.mlr.press/v119/yamada20a.html) | $\frac{1}{d}\sum \Phi\!\left(\frac{\mu_i+0.5}{\sigma}\right)$ (Gaussian CDF) |
| **GSG-Sigmoid** | [Jang et al., 2016](https://arxiv.org/abs/1611.01144) | $\frac{1}{d}\sum \lvert\sigma(\alpha_i)\rvert$ |
| **GSG-Softmax** | [Jang et al., 2016](https://arxiv.org/abs/1611.01144) | $\frac{1}{k}\sum p_{open,j}$ |
| **HCG** (Hard Concrete Gate) | [Louizos et al., ICLR 2018](https://openreview.net/forum?id=H1Y8hhg0b) | $\frac{1}{d}\sum \sigma(\alpha_i - \beta\log\frac{-\gamma}{\zeta})$ (L0 approx.) |

### Encoders

| Encoder | Paper | Key Idea |
|---------|-------|----------|
| **CAE** (Concrete Autoencoder) | [Balin et al., ICML 2019](https://proceedings.mlr.press/v97/balin19a.html) | Direct logits $\mathbb{R}^{D \times k}$ |
| **IPCAE** (Indirect Parametric Concrete AE) | [Balin et al., ICML 2019](https://proceedings.mlr.press/v97/balin19a.html) | Low-rank factorization $\mathbb{R}^{D \times h} \cdot \mathbb{R}^{h \times k}$ |

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
y = model(x)  # hard selection, no threshold
result = model.get_selection_result()
print(f"Selected {result.num_selected} features out of k=50 slots")
```

### With a Downstream Classifier

```python
from deepfs import GumbelSoftmaxGateIndirectConcreteModel
from exp.trainers import GateEncoderTrainer
from exp.utils import MLPClassifier

model = GumbelSoftmaxGateIndirectConcreteModel(
    input_dim=1000, k=50,
    embedding_dim_encoder=32, embedding_dim_gate=16,
    total_epochs=1000,
)
classifier = MLPClassifier(input_dim=50, hidden_dim=128, output_dim=5)
trainer = GateEncoderTrainer(model, classifier, sparse_loss_weight=1.0, lr=1e-4)
result_df, feature_df = trainer.fit(train_loader, epochs=1000, test_loader=test_loader)
```

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
│   ├── types.py                   # SparsityLoss, SelectionResult, TemperatureSchedule
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

## References

- **Gumbel-Softmax**: Jang, E., Gu, S., & Poole, B. (2016). *Categorical Reparameterization with Gumbel-Softmax*. [arXiv:1611.01144](https://arxiv.org/abs/1611.01144)
- **CAE / IPCAE**: Balin, M. F., Abid, A., & Zou, J. (2019). *Concrete Autoencoders: Differentiable Feature Selection and Reconstruction*. ICML 2019. [Paper](https://proceedings.mlr.press/v97/balin19a.html)
- **STG**: Yamada, Y., Lindenbaum, O., Negahban, S., & Kluger, Y. (2020). *Feature Selection using Stochastic Gates*. ICML 2020. [Paper](http://proceedings.mlr.press/v119/yamada20a.html)
- **Hard Concrete**: Louizos, C., Welling, M., & Kingma, D. P. (2018). *Learning Sparse Neural Networks through L0 Regularization*. ICLR 2018. [Paper](https://openreview.net/forum?id=H1Y8hhg0b)

## Citation

```bibtex
@misc{deepfs2026,
  author = {Your Name},
  title = {DeepFS: Deep Feature Selection via Gumbel Softmax Gating with Indirect Concrete Autoencoder},
  year = {2026},
  url = {https://github.com/your-repo/deep-feature-select}
}
```

## License

MIT License
