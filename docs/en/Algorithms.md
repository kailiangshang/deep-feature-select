# Algorithms

This document provides detailed explanations of the feature selection algorithms implemented in DeepFS.

## Algorithm Categories

Algorithms in DeepFS are divided into three categories:

1. **Gate Methods**: Control feature selection through learnable gate values
2. **Encoder Methods**: Directly select a fixed number of features
3. **Composite Methods**: Combine advantages of both gates and encoders

---

## Gate Methods

Gate methods learn an independent probability/weight for each feature, controlling the number of selected features through sparsity loss.

### 1. StochasticGate (STG)

**Paper**: [Feature Selection using Stochastic Gates (ICML 2020)](http://proceedings.mlr.press/v119/yamada20a.html)

**Principle**: Uses Gaussian noise to create differentiable stochastic gates.

**Formula**:
```
z = μ + σ * ε,  ε ~ N(0, 1)
gate = clamp(z + 0.5, 0, 1)
output = input * gate
```

**Sparsity Loss**: Gaussian CDF loss
```
L_sparsity = mean(Φ((μ + 0.5) / σ))
```

**Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dim` | int | - | Number of input features |
| `sigma` | float | 0.5 | Gaussian noise standard deviation |

---

### 2. GumbelSigmoidGate (GSG)

**Principle**: Uses Gumbel noise with Sigmoid activation to create relaxed binary gates.

**Formula**:
```
noise = -log(-log(U)),  U ~ Uniform(0, 1)
gate = σ((logits + noise) / τ)
```

**Sparsity Loss**: L1 loss + Entropy loss
```
L_sparsity = mean(p) - mean(p*log(p) + (1-p)*log(1-p))
```

**Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dim` | int | - | Number of input features |
| `initial_temperature` | float | 10.0 | Initial temperature |
| `final_temperature` | float | 0.01 | Final temperature |
| `total_epochs` | int | 100 | Total epochs for annealing |

---

### 3. GumbelSoftmaxGate (Embedded Gumbel-Softmax Gate)

**Principle**: Uses 2-class Gumbel-Softmax (open/close) with embedding layer for parameter-efficient gating.

**Formula**:
```
logits = embedding @ feature  # (2, D)
noise = Gumbel(0, 1)
probs = softmax((logits + noise) / τ, dim=0)
gate = probs[1, :]  # Probability of "open" class
```

**Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dim` | int | - | Number of input features |
| `embedding_dim` | int | - | Embedding dimension |
| `initial_temperature` | float | 10.0 | Initial temperature |
| `final_temperature` | float | 0.01 | Final temperature |

---

### 4. HardConcreteGate (HCG)

**Paper**: [Learning Sparse Neural Networks through L0 Regularization (ICLR 2018)](https://openreview.net/forum?id=H1Y8hhg0b)

**Principle**: Uses Hard Concrete distribution to enable exact zeros and ones in gate values.

**Formula**:
```
U ~ Uniform(0, 1)
s = σ((log(U) - log(1-U) + logits) / τ)
s_stretched = s * (ζ - γ) + γ
gate = clamp(s_stretched, 0, 1)
```

**Sparsity Loss**: L0 regularization
```
L_sparsity = mean(σ(logits - τ * log(-γ/ζ)))
```

**Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dim` | int | - | Number of input features |
| `min_max_scale` | tuple | (-0.1, 1.1) | Stretch parameters (γ, ζ) |
| `temperature` | float | 0.5 | Concrete distribution temperature |

---

## Encoder Methods

Encoder methods directly select a fixed k features without needing sparsity loss.

### 1. ConcreteEncoder (CAE)

**Paper**: [Concrete Autoencoders for Differentiable Feature Selection (ICML 2019)](https://proceedings.mlr.press/v97/balin19a.html)

**Principle**: Uses Gumbel-Softmax to select one input feature for each output position.

**Formula**:
```
logits ∈ R^{D×k}  # D inputs, k outputs
noise = Gumbel(0, 1)
probs = softmax((logits + noise) / τ, dim=0)
output = input @ probs
```

**Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dim` | int | - | Number of input features D |
| `output_dim` | int | - | Number of output features k |
| `initial_temperature` | float | 10.0 | Initial temperature |
| `final_temperature` | float | 0.01 | Final temperature |

**Parameter Count**: O(D × k)

---

### 2. IndirectConcreteEncoder (IPCAE)

**Principle**: Uses low-rank factorization to reduce parameter count.

**Formula**:
```
logits = input_emb @ output_emb  # (D×d) @ (d×k) = (D×k)
probs = softmax((logits + Gumbel) / τ, dim=0)
output = input @ probs
```

**Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dim` | int | - | Number of input features D |
| `output_dim` | int | - | Number of output features k |
| `embedding_dim` | int | - | Embedding dimension d |
| `initial_temperature` | float | 10.0 | Initial temperature |
| `final_temperature` | float | 0.01 | Final temperature |

**Parameter Count**: O(D × d + d × k), significantly reduced when d << D

---

## Composite Methods

### GateEncoderSelector

Combines gate and encoder for two-stage feature selection:

1. **Stage 1**: Encoder selects k_max candidate features
2. **Stage 2**: Gate further filters from candidates

**Available Combinations**:

| Combination | Gate | Encoder |
|-------------|------|---------|
| GE1 | StochasticGate | ConcreteEncoder |
| GE2 | StochasticGate | IndirectConcreteEncoder |
| GE3 | GumbelSigmoidGate | ConcreteEncoder |
| GE4 | GumbelSigmoidGate | IndirectConcreteEncoder |
| GE5 | GumbelSoftmaxGate | ConcreteEncoder |
| **GE6** | **GumbelSoftmaxGate** | **IndirectConcreteEncoder** |
| GE7 | HardConcreteGate | ConcreteEncoder |
| GE8 | HardConcreteGate | IndirectConcreteEncoder |

**Recommended Combination**: GE6 (GSG + IPCAE)

---

## Algorithm Selection Guide

| Scenario | Recommended Algorithm | Reason |
|----------|----------------------|--------|
| Need exact feature count | Encoder series | Direct k control |
| Parameter-sensitive | IndirectConcreteEncoder | Low-rank parameterization |
| High-dimensional data | GSG + IPCAE | Two-stage filtering |
| Need L0 regularization | HardConcreteGate | Native support |
| Simple tasks | StochasticGate | Simple implementation |

---

## Temperature Annealing

Most algorithms use temperature annealing strategy:

```
τ(t) = τ_final + (τ_initial - τ_final) * (1 - t/T)
```

- High initial temperature: Soft selection, smooth gradients
- Low final temperature: Near-discrete selection

---

[中文版本](../zh-CN/Algorithms.md)
