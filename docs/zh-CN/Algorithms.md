# 算法说明

本文档详细介绍 DeepFS 中实现的各种特征选择算法。

## 算法分类

DeepFS 中的算法分为三类：

1. **门控方法 (Gate)**: 通过可学习的门控值控制特征是否被选中
2. **编码器方法 (Encoder)**: 直接选择固定数量的特征
3. **组合方法 (Composite)**: 结合门控和编码器的优势

---

## 门控方法

门控方法为每个特征学习一个独立的概率/权重，通过稀疏损失控制选择的特征数量。

### 1. StochasticGate (随机门控)

**论文**: [Feature Selection using Stochastic Gates (ICML 2020)](http://proceedings.mlr.press/v119/yamada20a.html)

**原理**: 使用高斯噪声创建可微的随机门控。

**公式**:
```
z = μ + σ * ε,  ε ~ N(0, 1)
gate = clamp(z + 0.5, 0, 1)
output = input * gate
```

**稀疏损失**: 高斯 CDF 损失
```
L_sparsity = mean(Φ((μ + 0.5) / σ))
```

**参数**:
| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `input_dim` | int | - | 输入特征数 |
| `sigma` | float | 0.5 | 高斯噪声标准差 |

---

### 2. GumbelSigmoidGate (Gumbel-Sigmoid 门控)

**原理**: 使用 Gumbel 噪声与 Sigmoid 激活创建松弛的二值门控。

**公式**:
```
noise = -log(-log(U)),  U ~ Uniform(0, 1)
gate = σ((logits + noise) / τ)
```

**稀疏损失**: L1 损失 + 熵损失
```
L_sparsity = mean(p) - mean(p*log(p) + (1-p)*log(1-p))
```

**参数**:
| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `input_dim` | int | - | 输入特征数 |
| `initial_temperature` | float | 10.0 | 初始温度 |
| `final_temperature` | float | 0.01 | 最终温度 |
| `total_epochs` | int | 100 | 温度退火总轮数 |

---

### 3. GumbelSoftmaxGate (嵌入式 Gumbel-Softmax 门控)

**原理**: 使用 2 类 Gumbel-Softmax (开/关) 配合嵌入层实现参数高效的门控。

**公式**:
```
logits = embedding @ feature  # (2, D)
noise = Gumbel(0, 1)
probs = softmax((logits + noise) / τ, dim=0)
gate = probs[1, :]  # "开"类的概率
```

**参数**:
| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `input_dim` | int | - | 输入特征数 |
| `embedding_dim` | int | - | 嵌入维度 |
| `initial_temperature` | float | 10.0 | 初始温度 |
| `final_temperature` | float | 0.01 | 最终温度 |

---

### 4. HardConcreteGate (硬混凝土门控)

**论文**: [Learning Sparse Neural Networks through L0 Regularization (ICLR 2018)](https://openreview.net/forum?id=H1Y8hhg0b)

**原理**: 使用硬混凝土分布实现精确的 0 和 1 门控值。

**公式**:
```
U ~ Uniform(0, 1)
s = σ((log(U) - log(1-U) + logits) / τ)
s_stretched = s * (ζ - γ) + γ
gate = clamp(s_stretched, 0, 1)
```

**稀疏损失**: L0 正则化
```
L_sparsity = mean(σ(logits - τ * log(-γ/ζ)))
```

**参数**:
| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `input_dim` | int | - | 输入特征数 |
| `min_max_scale` | tuple | (-0.1, 1.1) | 拉伸参数 (γ, ζ) |
| `temperature` | float | 0.5 | 混凝土分布温度 |

---

## 编码器方法

编码器方法直接选择固定 k 个特征，不需要稀疏损失。

### 1. ConcreteEncoder (混凝土编码器)

**论文**: [Concrete Autoencoders for Differentiable Feature Selection (ICML 2019)](https://proceedings.mlr.press/v97/balin19a.html)

**原理**: 使用 Gumbel-Softmax 为每个输出位置选择一个输入特征。

**公式**:
```
logits ∈ R^{D×k}  # D 个输入, k 个输出
noise = Gumbel(0, 1)
probs = softmax((logits + noise) / τ, dim=0)
output = input @ probs
```

**参数**:
| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `input_dim` | int | - | 输入特征数 D |
| `output_dim` | int | - | 输出特征数 k |
| `initial_temperature` | float | 10.0 | 初始温度 |
| `final_temperature` | float | 0.01 | 最终温度 |

**参数量**: O(D × k)

---

### 2. IndirectConcreteEncoder (间接混凝土编码器 / IPCAE)

**原理**: 使用低秩分解减少参数量。

**公式**:
```
logits = input_emb @ output_emb  # (D×d) @ (d×k) = (D×k)
probs = softmax((logits + Gumbel) / τ, dim=0)
output = input @ probs
```

**参数**:
| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `input_dim` | int | - | 输入特征数 D |
| `output_dim` | int | - | 输出特征数 k |
| `embedding_dim` | int | - | 嵌入维度 d |
| `initial_temperature` | float | 10.0 | 初始温度 |
| `final_temperature` | float | 0.01 | 最终温度 |

**参数量**: O(D × d + d × k)，当 d << D 时显著减少

---

## 组合方法

### GateEncoderSelector

将门控和编码器组合，实现两阶段特征选择：

1. **第一阶段**: 编码器选择 k_max 个候选特征
2. **第二阶段**: 门控从候选中进一步筛选

**可用组合**:

| 组合 | 门控 | 编码器 |
|------|------|--------|
| GE1 | StochasticGate | ConcreteEncoder |
| GE2 | StochasticGate | IndirectConcreteEncoder |
| GE3 | GumbelSigmoidGate | ConcreteEncoder |
| GE4 | GumbelSigmoidGate | IndirectConcreteEncoder |
| GE5 | GumbelSoftmaxGate | ConcreteEncoder |
| **GE6** | **GumbelSoftmaxGate** | **IndirectConcreteEncoder** |
| GE7 | HardConcreteGate | ConcreteEncoder |
| GE8 | HardConcreteGate | IndirectConcreteEncoder |

**推荐组合**: GE6 (GSG + IPCAE)

---

## 算法选择指南

| 场景 | 推荐算法 | 原因 |
|------|----------|------|
| 需要精确的特征数 | Encoder 系列 | 直接控制 k |
| 参数量敏感 | IndirectConcreteEncoder | 低秩参数化 |
| 高维数据 | GSG + IPCAE | 两阶段筛选 |
| 需要 L0 正则 | HardConcreteGate | 原生支持 |
| 简单任务 | StochasticGate | 实现简单 |

---

## 温度退火

大多数算法使用温度退火策略：

```
τ(t) = τ_final + (τ_initial - τ_final) * (1 - t/T)
```

- 初始高温度: 软选择，梯度平滑
- 最终低温度: 接近离散选择

---

[English Version](../en/Algorithms.md)
