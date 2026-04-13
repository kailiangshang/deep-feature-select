# Deep Feature Select Rewrite Design

## 1. Overview

Rewrite the deep feature selection library, keeping mathematical correctness as the top priority. The core contribution is **GSG-IPCAE** (Gumbel Softmax Gate + Indirect Parameterized Concrete Autoencoder), a two-stage feature selection method.

### Key Principles

- **Mathematical correctness first**: Each model file is self-contained with clear forward/inference paths
- **Decouple feature selection from downstream tasks**: The library only does feature selection; experiments handle downstream evaluation
- **Observability**: Every model exposes diagnostics for monitoring training dynamics
- **Reproducibility**: YAML configs, seed management, deterministic training

### Algorithm Scope

| Category | Methods | Role |
|----------|---------|------|
| Encoder-only | CAE, IPCAE | Baselines, select exactly k features |
| Gate-only | STG, GSG-Sigmoid, GSG-Softmax, HCG | Baselines, control sparsity via λ |
| Combined | GSG-Softmax+CAE, GSG-Softmax+IPCAE, STG+CAE, STG+IPCAE, HCG+CAE, HCG+IPCAE | Full comparisons |
| **Core** | **GSG-Softmax+IPCAE** | **Paper contribution** |

Note: GSG has two variants. GSG-Sigmoid is unstable (baseline). GSG-Softmax is the gate used in our combined method.

---

## 2. Algorithm Mathematical Definitions

### 2.1 Shared Utilities

**Temperature Annealing**: `τ_t = τ_init × (τ_final / τ_init)^(t/T)` (exponential decay)

**Gumbel Noise**: `g = -log(-log(U))`, `U ~ Uniform(0,1)`

**`custom_one_hot(indices, num_classes)`**: When index == -1, produces a zero row (closed slot).

### 2.2 CAE (Concrete Autoencoder, Balin et al. ICML 2019)

- **Parameters**: `Logits ∈ R^{d×k}` (direct parameterization)
- **Training**: `X' = X @ Softmax((Logits + Gumbel) / τ, dim=0)` — each column is a soft distribution over d features
- **Inference**: `X' = X @ OneHot(argmax(Logits, dim=0))` — hard selection
- **No sparsity loss**, selects exactly k features

### 2.3 IPCAE (Indirect Parameterized Concrete Autoencoder)

- **Parameters**: `W_in ∈ R^{d×h}, W_out ∈ R^{h×k}` (low-rank, O(dh+hk) vs O(dk))
- **Logits**: `Logits = (W_out^T @ W_in^T)^T`
- **Forward**: Same as CAE but logits come from embedding space

### 2.4 STG (Stochastic Gate, Yamada et al. ICML 2020)

- **Parameters**: `μ ∈ R^d` (one logit per feature)
- **Training**: `z = clamp(μ + σ·N(0,1) + 0.5, 0, 1)`, `X' = X ⊙ z`
- **Inference**: `z = clamp(μ + 0.5, 0, 1)`, features where z==0 are closed
- **Sparsity Loss**: `L = mean(Φ((μ+0.5)/σ))` — Φ is standard normal CDF (probability gate is open)
- **Activation**: Threshold at μ+0.5 ≤ 0 means closed

### 2.5 GSG-Sigmoid (Gumbel Sigmoid Gate, Baseline)

- **Parameters**: `α ∈ R^d`
- **Training**: `z = σ((α + g) / τ)`, `X' = X ⊙ z`
- **Inference**: `z = (σ(α) > 0.5) ? 1 : 0`
- **Sparsity Loss**: `L = mean(z)` (L1)
- **Note**: Unstable in practice, used as baseline comparison

### 2.6 GSG-Softmax (Gumbel Softmax Gate, Jang et al. 2016)

- **Parameters**: `G_emb ∈ R^{2×h}, G_feat ∈ R^{h×d}` (low-rank, 2 classes = open/close)
- **Logits**: `L = (G_feat^T @ G_emb^T)^T ∈ R^{2×d}`
- **Training**: `P = Softmax((L + Gumbel) / τ, dim=0) ∈ R^{2×d}`, `p_open = P[1,:]`
- **Inference**: `open = argmax(L, dim=0)` — each feature independently open/close
- **Sparsity Loss**: `L = mean(p_open)` (L1)
- **Activation**: Binary by construction (argmax of 2-class softmax), no threshold needed

### 2.7 HCG (Hard Concrete Gate, Louizos et al. ICLR 2018)

- **Parameters**: `α ∈ R^d`
- **Training**: `u ~ U(0,1)`, `s = σ((log(u)-log(1-u)+α)/τ)`, `z = clamp(s·(ζ-γ)+γ, 0, 1)`
- **Inference**: `z = (σ(α) > threshold) ? 1 : 0`
- **Sparsity Loss**: `L = mean(σ(α - τ·log(-γ/ζ)))` (L0 regularization)
- **Default**: γ=-0.1, ζ=1.1, τ=0.5

### 2.8 Combined Models (Gate + Encoder)

**Critical Design**: Gate operates on **k selection slots**, not d original features.

#### GSG-Softmax + IPCAE (Core Contribution: GSG-IPCAE)

- **Encoder Parameters**: `W_in ∈ R^{d×h_enc}, W_out ∈ R^{h_enc×k}`
- **Gate Parameters**: `G_emb ∈ R^{2×h_gate}, G_feat ∈ R^{h_gate×k}`
- **Encoder Logits**: `L_enc = (W_out^T @ W_in^T)^T ∈ R^{d×k}`
- **Gate Logits**: `L_gate = (G_feat^T @ G_emb^T)^T ∈ R^{2×k}`

**Training**:
```
P_enc = Softmax((L_enc + gumbel) / τ, dim=0)    # R^{d×k}
P_gate = Softmax((L_gate + gumbel) / τ, dim=0)  # R^{2×k}
p_open = P_gate[1, :]                             # R^k
X' = X @ (P_enc ⊙ p_open)                        # element-wise on selection matrix
```

**Inference**:
```
selected = argmax(L_enc, dim=0)    # k indices
open = argmax(L_gate, dim=0)       # k binary decisions
selected[open == 0] = -1           # close slots
X' = X @ custom_one_hot(selected)  # hard selection
```

The other 5 combined models (GSG-Softmax+CAE, STG+CAE, STG+IPCAE, HCG+CAE, HCG+IPCAE) follow the same structure with their respective gate/encoder components.

---

## 3. Code Architecture

### 3.1 Project Structure

```
deep-feature-select/
├── deepfs/                          # Core feature selection library (public)
│   ├── __init__.py                  # Public API exports
│   ├── core/
│   │   ├── __init__.py
│   │   ├── types.py                 # SparsityLoss, SelectionResult, TemperatureSchedule,
│   │   │                            # GateDiagnostics, EncoderDiagnostics, TrainingSnapshot
│   │   ├── base.py                  # BaseSelector, EncoderFeatureModule, GateFeatureModule
│   │   └── utils.py                 # generate_gumbel_noise, custom_one_hot
│   ├── models/
│   │   ├── __init__.py              # Register and export all models
│   │   ├── cae.py                   # ConcreteAutoencoderModel
│   │   ├── ipcae.py                 # IndirectConcreteAutoencoderModel
│   │   ├── gumbel_sigmoid_gate.py   # GumbelSigmoidGateModel (gate-only, baseline)
│   │   ├── gumbel_softmax_gate.py   # GumbelSoftmaxGateModel (gate-only)
│   │   ├── stochastic_gate.py       # StochasticGateModel (gate-only)
│   │   ├── hard_concrete_gate.py    # HardConcreteGateModel (gate-only)
│   │   ├── gumbel_softmax_cae.py    # GumbelSoftmaxGateConcreteModel
│   │   ├── gumbel_softmax_ipcae.py  # GumbelSoftmaxGateIndirectConcreteModel (core)
│   │   ├── stochastic_cae.py        # StochasticGateConcreteModel
│   │   ├── stochastic_ipcae.py      # StochasticGateIndirectConcreteModel
│   │   ├── hard_concrete_cae.py     # HardConcreteGateConcreteModel
│   │   └── hard_concrete_ipcae.py   # HardConcreteGateIndirectConcreteModel
│   └── ...
├── exp/                             # Experiment code (paper results)
│   ├── configs/
│   │   ├── contrast.yaml            # Comparison experiment config
│   │   ├── ablation.yaml            # Ablation experiment config
│   │   └── hyperparameter.yaml      # Hyperparameter sweep config
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py               # MetaData, generate_train_test_loader (h5ad)
│   │   └── utils.py                 # Data utilities
│   ├── trainers/
│   │   ├── __init__.py
│   │   ├── encoder_trainer.py       # Encoder-only training (no sparsity loss)
│   │   ├── gate_trainer.py          # Gate-only training (with sparsity loss)
│   │   └── gate_encoder_trainer.py  # Combined training (with sparsity loss)
│   ├── visualization/
│   │   ├── plot_results.py          # Result plotting (per dataset)
│   │   └── generate_tables.py       # LaTeX table generation
│   ├── run_contrast.py              # Comparison experiment entry
│   ├── run_ablation.py              # Ablation experiment entry
│   ├── run_hyperparameter.py        # Hyperparameter experiment entry
│   └── utils.py                     # MLPClassifier, seed_all, Result
├── docs/
├── deep-feature-select-old/         # Old code reference (in .gitignore)
├── pyproject.toml
├── README.md
├── .gitignore
└── uv.lock
```

### 3.2 Core Types

```python
# core/types.py

class SparsityLoss(NamedTuple):
    names: List[str]
    values: List[torch.Tensor]
    
    @property
    def total(self) -> torch.Tensor: ...

@dataclass
class SelectionResult:
    selected_indices: np.ndarray      # -1 for closed slots
    selected_mask: np.ndarray         # boolean mask over original features
    gate_probs: Optional[np.ndarray]  # gate probabilities (if applicable)
    num_selected: int

@dataclass
class TemperatureSchedule:
    initial: float = 10.0
    final: float = 0.01
    total_epochs: int = 100
    
    def get_temperature(self, epoch: int) -> float: ...

@dataclass
class GateDiagnostics:
    """Gate monitoring information"""
    gate_probs: np.ndarray            # current gate probabilities
    num_open: int                     # number of open gates
    num_closed: int                   # number of closed gates
    open_ratio: float                 # open / total ratio
    threshold: Optional[float]        # activation threshold (None for GSG-Softmax)
    entropy: float                    # gate distribution entropy

@dataclass
class EncoderDiagnostics:
    """Encoder monitoring information"""
    selected_indices: np.ndarray      # currently selected feature indices
    selection_entropy: np.ndarray     # per-slot selection entropy
    feature_overlap: int              # number of duplicate selections

@dataclass
class TrainingSnapshot:
    """Recorded at end of each epoch"""
    epoch: int
    temperature: float
    loss_task: float
    loss_sparsity: float
    gate_diagnostics: Optional[GateDiagnostics]
    encoder_diagnostics: Optional[EncoderDiagnostics]
    num_selected_features: int
    val_metric: float                 # downstream task metric (accuracy, MSE, etc.)
```

### 3.3 Base Classes

```python
# core/base.py

class BaseSelector(nn.Module, ABC):
    """Abstract base for all feature selectors"""
    
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor: ...
    
    @abstractmethod
    def get_selection_result(self) -> SelectionResult: ...
    
    def update_temperature(self, epoch: int): ...

class EncoderFeatureModule(BaseSelector):
    """Base for encoder-only models (CAE, IPCAE)"""
    input_dim: int
    output_dim: int    # k features to select
    
    @property
    @abstractmethod
    def selected_indices(self) -> Tensor: ...
    
    def get_selection_result(self) -> SelectionResult: ...
    
    def encoder_diagnostics(self) -> EncoderDiagnostics: ...

class GateFeatureModule(BaseSelector):
    """Base for gate and gate+encoder models"""
    input_dim: int
    
    @abstractmethod
    def sparsity_loss(self) -> SparsityLoss: ...
    
    @property
    @abstractmethod
    def selected_indices_candidate(self) -> Tensor: ...
    
    @abstractmethod
    def gate_logits_probs(self) -> Tensor: ...
    
    def gate_diagnostics(self) -> GateDiagnostics: ...
```

### 3.4 Model Implementation Pattern

Each model file follows this pattern (example: `gumbel_softmax_ipcae.py`):

```python
class GumbelSoftmaxGateIndirectConcreteModel(GateFeatureModule):
    def __init__(self, input_dim, k, embedding_dim_encoder, embedding_dim_gate,
                 temperature_schedule, device='cpu'):
        # Initialize all parameters
        # Encoder: W_in (d, h_enc), W_out (h_enc, k)
        # Gate: G_emb (2, h_gate), G_feat (h_gate, k)
    
    def forward(self, x):
        if self.training:
            # Soft selection: encoder_prob ⊙ gate_prob
        else:
            # Hard selection: argmax + custom_one_hot
    
    def sparsity_loss(self) -> SparsityLoss:
        # mean(p_open) for GSG-Softmax
    
    def update_temperature(self, epoch):
        # Anneal temperature
    
    @property
    def selected_indices_candidate(self) -> Tensor:
        # argmax(encoder_logits, dim=0), with -1 for closed slots
    
    def gate_logits_probs(self) -> Tensor:
        # Current gate probabilities
    
    def gate_diagnostics(self) -> GateDiagnostics:
        # Return monitoring info
    
    def encoder_diagnostics(self) -> EncoderDiagnostics:
        # Return monitoring info
```

### 3.5 .gitignore Addition

Add `deep-feature-select-old/` to `.gitignore`.

---

## 4. Experiment Plan

### 4.1 Comparison Experiments

**Datasets**: Pancreas, Lung, Spleen, Tongue (h5ad, single-cell RNA-seq) + new datasets (TBD)

**Experiment Matrix**:

| Category | Methods | Hyperparameter Grid | Groups |
|----------|---------|-------------------|--------|
| Encoder-only | CAE, IPCAE | k ∈ {1..50} (step 1) ∪ {100,200,300,400,500} | 110 groups × 5 seeds |
| Gate-only | STG, GSG-Sigmoid, GSG-Softmax, HCG | λ ∈ {0.001, 0.01, 0.1, 1, 10} | 20 groups × 5 seeds |
| Combined | GSG-Softmax+CAE, GSG-Softmax+IPCAE, STG+CAE, STG+IPCAE, HCG+CAE, HCG+IPCAE | k_max ∈ {50,100}, λ ∈ {0.001, 0.01, 0.1, 1, 10} | 60 groups × 5 seeds |

### 4.2 Ablation Experiments

On GSG-IPCAE (core method) only:

1. **k_max Sensitivity**: Fix λ, vary k_max ∈ {10,20,30,50,100,200}
2. **λ Sensitivity**: Fix k_max, vary λ ∈ {0.001, 0.01, 0.1, 1, 10, 100}
3. **Encoder Embedding Dimension**: Fix gate_emb, vary enc_emb ∈ {16,32,64,128,256}
4. **Gate Embedding Dimension**: Fix enc_emb, vary gate_emb ∈ {8,16,32,64,128}
5. **Joint Embedding**: Vary both dimensions jointly
6. **Temperature Schedule**: Different initial/final temperatures and annealing rates

### 4.3 Evaluation Metrics

**Downstream Task Metrics** (pluggable, not limited to classification):

| Task Type | Metrics |
|-----------|---------|
| Classification | Accuracy, F1 (macro/weighted), Precision, Recall |
| Regression | MSE, MAE, R² |
| Reconstruction | Reconstruction MSE (selected features → original) |

**Feature Selection Metrics**:

| Metric | Description |
|--------|-------------|
| # Selected Features | Final number of selected features |
| Sparsity Ratio | 1 - (#selected / #total) |
| Training Stability | Standard deviation across 5 seeds |
| Parameter Count | Total trainable parameters |
| Convergence Speed | Epochs to reach 95% of final performance |

### 4.4 Training Configuration

```yaml
# Default configuration
training:
  epochs: 1000
  batch_size: 512
  learning_rate: 1e-4
  optimizer: adam
  seed: [0, 1, 2, 3, 4]  # 5 seeds per experiment

model:
  embedding_dim_encoder: 300
  embedding_dim_gate: 64
  temperature:
    initial: 10.0
    final: 0.01
```

### 4.5 Experiment Workflow

Per dataset:
1. Load data → split train/test/val
2. Run comparison experiments → CSV results + plots
3. Run ablation experiments → CSV results + plots
4. Run hyperparameter experiments → CSV results + plots
5. Generate LaTeX tables
6. Visualization includes: training curves (loss, accuracy, gate dynamics), bar charts (comparison), heatmaps (ablation)

---

## 5. Key Design Decisions

1. **Combined models are independent classes**: Not layer composition. Each combined model contains all parameters and implements the correct `encoder_prob ⊙ gate_prob` multiplication directly. This avoids the gate scope confusion that caused bugs in the previous refactored code.

2. **Gate operates on k slots in combined models**: Not on d original features. This is the fundamental difference between gate-only and gate+encoder models.

3. **GSG-Softmax needs no activation threshold**: The 2-class softmax + argmax naturally produces binary open/close decisions. STG and HCG require threshold-based activation.

4. **Embedding is used in both gate and encoder**: Both components can use low-rank parameterization independently. Ablation experiments test both dimensions.

5. **Downstream task evaluation is pluggable**: The experiment framework supports classification, regression, and reconstruction evaluation through a common interface.

6. **Training dynamics monitoring**: Every model exposes `gate_diagnostics()` and `encoder_diagnostics()` for tracking training progress. Snapshots are recorded per epoch.

7. **YAML-driven experiments**: All experiment parameters are in YAML config files for reproducibility.

8. **Package manager**: uv (keeping current setup).
