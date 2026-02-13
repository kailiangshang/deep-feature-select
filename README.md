# DeepFS - Deep Feature Selection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个灵活、模块化的深度特征选择库，实现了多种最先进的可微特征选择算法。

## 🌟 特性

- **🏆 核心算法**: IPCAE + 嵌入式 Gumbel-Softmax 门控的组合方法，实现高效的两阶段特征选择
- **模块化设计**: 门控(Gate)、编码器(Encoder)、选择器(Selector)可自由组合
- **多种算法**: 实现 4 种门控算法 + 2 种编码器算法 + 8 种组合方法
- **易于扩展**: 清晰的基类设计，方便实现新算法
- **生产就绪**: 完整的类型注解、文档字符串、单元测试

## 🔥 核心算法: GSG-IPCAE

我们提出的核心算法是 **嵌入式 Gumbel-Softmax 门控 (GSG) + 间接参数化混凝土自编码器 (IPCAE)** 的组合，实现两阶段特征选择：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      GSG-IPCAE 两阶段特征选择架构                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   输入特征 X ∈ R^{N×D}                                                      │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │   Stage 1: IPCAE (间接参数化混凝土编码器)                              │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │  特征嵌入 E ∈ R^{D×d}     选择概率 α ∈ R^{D×k_max}           │   │   │
│  │   │  (低秩参数化)              (Concrete Distribution)           │   │   │
│  │   │       ↓                          ↓                          │   │   │
│  │   │  logits = E @ W          Gumbel-Softmax 采样                 │   │   │
│  │   │       ↓                          ↓                          │   │   │
│  │   │  选择 k_max 个候选特征 ─────────────────► X' ∈ R^{N×k_max}   │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │   Stage 2: GSG (嵌入式 Gumbel-Softmax 门控)                           │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │  门控嵌入 G ∈ R^{k_max×d'}   门控概率 π ∈ R^{k_max}           │   │   │
│  │   │  (低秩参数化)                 (Gumbel-Sigmoid)               │   │   │
│  │   │       ↓                           ↓                         │   │   │
│  │   │  gates = σ(G @ v + ε)     稀疏正则化 L0惩罚                   │   │   │
│  │   │       ↓                           ↓                         │   │   │
│  │   │  从 k_max 候选中筛选 ───────────────► X'' ∈ R^{N×k}          │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       ▼                                                                     │
│   输出: 选择 k 个最优特征 (k < k_max << D)                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 为什么选择 GSG-IPCAE?

| 特性 | 传统方法 | GSG-IPCAE |
|-----|---------|-----------|
| **参数量** | O(D) 或 O(D²) | O(D×d) 低秩参数化 |
| **特征数量** | 固定或不稳定 | 两阶段精确控制 |
| **稀疏性** | 需要 L1 正则 | 天然稀疏 + L0 正则 |
| **可微性** | 部分可微 | 完全可微 |
| **特征关联** | 独立选择 | 考虑特征关联 |

### 快速使用 GSG-IPCAE

```python
import torch
from deepfs import GumbelSoftmaxGate, IndirectConcreteEncoder, GateEncoderSelector

# 创建 GSG-IPCAE 选择器
encoder = IndirectConcreteEncoder(
    input_dim=58482,      # 原始特征数 (如基因数)
    output_dim=100,       # 第一阶段选择 k_max 个候选
    embedding_dim=32,     # 低秩嵌入维度 (节省参数)
)

gate = GumbelSoftmaxGate(
    input_dim=100,        # 从 k_max 个候选中筛选
    embedding_dim=16,     # 门控嵌入维度
)

selector = GateEncoderSelector(gate, encoder)

# 训练
x = torch.randn(32, 58482)  # 32 样本, 58482 特征
selected_x = selector(x)     # -> 32 样本, ~50 特征 (由 λ 控制)

# 获取稀疏损失
sparsity_loss = gate.sparsity_loss().total
total_loss = task_loss + 0.1 * sparsity_loss
```

## 📦 安装

```bash
# 克隆仓库
git clone https://github.com/your-repo/deep-feature-select.git
cd deep-feature-select

# 安装依赖 (使用 uv)
uv sync

# 或使用 pip
pip install -e .
```

## 🚀 快速开始

### 基本使用

```python
import torch
from deepfs import StochasticGate, GumbelSigmoidGate, HardConcreteGate
from deepfs import ConcreteEncoder, IndirectConcreteEncoder
from deepfs import GateEncoderSelector

# 1. 纯门控方法 - 使用稀疏损失控制特征数量
input_dim = 1000
gate = StochasticGate(input_dim=input_dim, sigma=0.5)

# 训练时
x = torch.randn(32, input_dim)  # batch_size=32
selected_x = gate(x)  # 应用特征选择

# 获取稀疏损失
sparsity_loss = gate.sparsity_loss().total
total_loss = task_loss + 0.1 * sparsity_loss  # λ=0.1 控制稀疏程度

# 2. 编码器方法 - 直接选择 k 个特征
encoder = ConcreteEncoder(input_dim=1000, output_dim=50)  # 选择 50 个特征
selected_x = encoder(x)

# 3. 组合方法 - 编码器选候选特征，门控进一步筛选
gate = GumbelSigmoidGate(input_dim=50)  # 从 50 个候选中筛选
selector = GateEncoderSelector(gate, encoder)
selected_x = selector(x)
```

### 完整训练示例

```python
import torch
import torch.nn as nn
from deepfs import StochasticGate

# 定义带特征选择的模型
class FeatureSelectionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.gate = StochasticGate(input_dim=input_dim, sigma=0.5)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        x = self.gate(x)
        return self.classifier(x)
    
    def sparsity_loss(self):
        return self.gate.sparsity_loss().total

# 训练
model = FeatureSelectionClassifier(input_dim=1000, hidden_dim=128, num_classes=5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
lambda_sparsity = 0.1

for epoch in range(100):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        
        # 任务损失 + 稀疏损失
        loss = criterion(outputs, batch_y)
        loss = loss + lambda_sparsity * model.sparsity_loss()
        
        loss.backward()
        optimizer.step()

# 推理
model.eval()
with torch.no_grad():
    outputs = model(test_x)
    predictions = outputs.argmax(dim=1)
    
# 获取选择的特征
num_selected = model.gate.num_selected
print(f"选择了 {num_selected} 个特征")
```

## 🔧 使用自己的数据

DeepFS 支持多种数据格式：

### 方式1: PyTorch Tensor

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

# 准备数据
X = torch.randn(1000, 500)  # 1000 样本, 500 特征
y = torch.randint(0, 3, (1000,))  # 3 分类任务

dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 训练
for batch_x, batch_y in train_loader:
    outputs = model(batch_x)
    loss = criterion(outputs, batch_y) + lambda_sparsity * model.sparsity_loss()
    # ...
```

### 方式2: AnnData (单细胞数据)

```python
import scanpy as sc
from exp.data.dataset_all import generate_train_test_loader

# 加载 h5ad 格式数据
# 数据需包含 'cell_type' 列作为标签
metadata = generate_train_test_loader(
    name="pancreas",  # exp/data/pancreas.h5ad
    test_size=0.2,
    batch_size=32,
    device="cuda"
)

# 使用数据
for batch in metadata.train_loader:
    x = batch.X  # 特征
    y = batch.obs['cell_type']  # 标签
    # ...
```

> 🚧 **更多数据集即将推出**: 我们正在准备更多公开数据集的下载链接和使用示例，包括标准特征选择基准数据集。敬请期待！

### 方式3: NumPy 数组

```python
import numpy as np
import torch

# 从文件加载
X = np.loadtxt("features.csv", delimiter=",")
y = np.loadtxt("labels.csv", delimiter=",")

# 或从其他来源
# X = your_data_pipeline()
# y = your_labels()

# 转换为 PyTorch
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)
```

### 方式4: Pandas DataFrame

```python
import pandas as pd
import torch

df = pd.read_csv("your_data.csv")
X = torch.tensor(df.drop(columns=["label"]).values, dtype=torch.float32)
y = torch.tensor(df["label"].values, dtype=torch.long)
```

## 📊 完整实验流程

我们提供了完整的实验脚本：

```bash
# 运行所有实验 (对比实验 + 消融实验)
python exp/run_experiment.py

# 或单独运行快速测试
python exp/run_quick_test.py
```

实验结果将保存在 `exp/results/` 目录下：
- `contrast/gate_only.csv` - 纯门控实验结果
- `contrast/encoder_only.csv` - 纯编码器实验结果  
- `contrast/gate_encoder.csv` - 组合方法实验结果
- `ablation/ablation.csv` - 消融实验结果

## 📚 算法说明

### 门控方法 (Gate Methods)

| 算法 | 描述 | 论文 |
|-----|------|------|
| **StochasticGate** | 高斯噪声随机门控 | [STG, ICML 2020](http://proceedings.mlr.press/v119/yamada20a.html) |
| **GumbelSigmoidGate** | Gumbel-Sigmoid 松弛门控 | [Gumbel-Softmax, 2016](https://arxiv.org/abs/1611.01144) |
| **GumbelSoftmaxGate** | 嵌入式 Gumbel-Softmax 门控 | [Gumbel-Softmax, 2016](https://arxiv.org/abs/1611.01144) |
| **HardConcreteGate** | 硬混凝土分布门控 | [L0 Regularization, ICLR 2018](https://openreview.net/forum?id=H1Y8hhg0b) |

**特点**: 通过稀疏损失控制特征数量，特征数量不固定

### 编码器方法 (Encoder Methods)

| 算法 | 描述 | 论文 |
|-----|------|------|
| **ConcreteEncoder** | 混凝土自编码器 | [CAE, ICML 2019](https://proceedings.mlr.press/v97/balin19a.html) |
| **IndirectConcreteEncoder** | 间接参数化混凝土编码器 | 基于 [CAE](https://proceedings.mlr.press/v97/balin19a.html) 改进 |

**特点**: 直接选择固定 k 个特征，无需稀疏损失

### 组合方法

编码器先选择 k_max 个候选特征，门控再进一步筛选：

```python
# 8 种组合
StochasticGate + ConcreteEncoder
StochasticGate + IndirectConcreteEncoder
GumbelSigmoidGate + ConcreteEncoder
GumbelSigmoidGate + IndirectConcreteEncoder
GumbelSoftmaxGate + ConcreteEncoder
GumbelSoftmaxGate + IndirectConcreteEncoder
HardConcreteGate + ConcreteEncoder
HardConcreteGate + IndirectConcreteEncoder
```

## ⚙️ 超参数说明

### 门控方法

```python
StochasticGate(
    input_dim=1000,      # 输入特征数
    sigma=0.5,           # 高斯噪声标准差
    hard_gate_type="hard_zero",  # 推理时硬门控类型
)

GumbelSigmoidGate(
    input_dim=1000,
    initial_temperature=10.0,   # 初始温度
    final_temperature=0.01,     # 最终温度
    total_epochs=100,           # 温度退火总轮数
)

GumbelSoftmaxGate(
    input_dim=1000,
    embedding_dim=16,     # 嵌入维度 (参数效率)
    # ... 温度参数同上
)

HardConcreteGate(
    input_dim=1000,
    min_max_scale=(-0.1, 1.1),  # 拉伸参数 (gamma, zeta)
    temperature=0.5,             # 混凝土分布温度
)
```

### 编码器方法

```python
ConcreteEncoder(
    input_dim=1000,       # 输入特征数
    output_dim=50,        # 选择的特征数 k
    initial_temperature=10.0,
    final_temperature=0.01,
    total_epochs=100,
)

IndirectConcreteEncoder(
    input_dim=1000,
    output_dim=50,
    embedding_dim=32,     # 低秩嵌入维度 (参数效率)
    # ... 温度参数同上
)
```

### 稀疏损失权重 λ

- **小 λ (0.001-0.01)**: 选择更多特征，可能过拟合
- **中 λ (0.1-1)**: 平衡稀疏性和性能
- **大 λ (10-100)**: 选择更少特征，可能欠拟合

建议通过网格搜索找到最优值。

## 📁 项目结构

```
deep-feature-select/
├── deepfs/                    # 核心库
│   ├── core/                  # 基础类型和抽象类
│   │   ├── types.py          # 类型定义
│   │   ├── base.py           # 基类
│   │   └── registry.py       # 注册表
│   ├── gates/                 # 门控模块
│   │   ├── stochastic.py     # 随机门控 (STG)
│   │   ├── gumbel_sigmoid.py # Gumbel-Sigmoid 门控
│   │   ├── gumbel_softmax.py # Gumbel-Softmax 门控
│   │   └── hard_concrete.py  # 硬混凝土门控
│   ├── encoders/              # 编码器模块
│   │   ├── concrete.py       # 混凝土编码器 (CAE)
│   │   └── indirect_concrete.py # 间接混凝土编码器
│   ├── selectors/             # 选择器模块
│   │   └── composite.py      # 组合选择器
│   ├── training/              # 训练工具
│   │   ├── trainer.py        # 训练器
│   │   └── callbacks.py      # 回调函数
│   └── __init__.py           # 公共 API
├── exp/                       # 实验代码
│   ├── run_experiment.py     # 完整实验脚本
│   ├── run_quick_test.py     # 快速验证脚本
│   ├── data/                 # 数据加载
│   └── results/              # 实验结果
├── EXPERIMENT_PLAN.md        # 实验计划
├── README.md                 # 本文档
└── pyproject.toml            # 项目配置
```

## 🔬 实验复现

详见 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)

### 实验内容

1. **对比实验** (412 组)
   - 纯门控: 4 算法 × 6 λ值 = 24 组
   - 纯编码: 2 算法 × 50 k值 = 100 组
   - 组合: 8 算法 × 6 k_max × 6 λ = 288 组

2. **消融实验** (22 组)
   - k_max 影响: 6 组
   - λ 影响: 6 组
   - 嵌入维度影响: 5 + 5 组

### 评估指标

- **任务性能**: 分类准确率
- **稀疏性**: 1 - (选择特征数 / 总特征数)
- **参数量**: 可训练参数总数
- **稳定性**: 5 次运行的标准差/均值

## 📖 引用

> 📝 **论文即将发布**: 本项目的学术论文正在准备中，即将在 arXiv 上发布。敬请期待！

如果这个库对您的研究有帮助，请引用：

```bibtex
@misc{deepfs2024,
  author = {Your Name},
  title = {DeepFS: A Modular Deep Feature Selection Library},
  year = {2024},
  url = {https://github.com/your-repo/deep-feature-select},
  note = {arXiv preprint coming soon}
}
```

相关论文：

```bibtex
@inproceedings{yamada2020feature,
  title={Feature Selection Using Stochastic Gates},
  author={Yamada, Yutaro and d'Aspremont, Alexandre and Watanabe, Shinji and Koyama, Shinya and Matsuo, Yutaka},
  booktitle={ICML},
  year={2020}
}

@inproceedings{balin2019concrete,
  title={Concrete Autoencoders for Differentiable Feature Selection},
  author={Balin, Mustafa Furkan and Ozturkler, Berk and Can, Mustafa and Dengiz, Emre and Yoldemir, Aykut and Karaca, Abdullah and Ozturk, Murat and Erkmen, Burak and Erdogan, Alper and Gunduz, Huseyin and others},
  booktitle={ICML},
  year={2019}
}

@inproceedings{louizos2018learning,
  title={Learning Sparse Neural Networks through $L_0$ Regularization},
  author={Louizos, Christos and Welling, Max and Kingma, Diederik P},
  booktitle={ICLR},
  year={2018}
}
```

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件