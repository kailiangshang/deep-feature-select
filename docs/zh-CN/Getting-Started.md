# 快速开始

本指南将帮助您快速上手 DeepFS 库。

## 系统要求

- Python >= 3.10
- PyTorch >= 2.0.0
- NumPy >= 1.24.0

## 安装

### 方式一：使用 uv (推荐)

```bash
# 克隆仓库
git clone https://github.com/your-repo/deep-feature-select.git
cd deep-feature-select

# 使用 uv 安装
uv sync
```

### 方式二：使用 pip

```bash
git clone https://github.com/your-repo/deep-feature-select.git
cd deep-feature-select
pip install -e .
```

## 基本使用

### 1. 纯门控方法

门控方法通过稀疏损失控制特征数量，特征数量不固定：

```python
import torch
from deepfs import StochasticGate

# 创建门控
input_dim = 1000
gate = StochasticGate(input_dim=input_dim, sigma=0.5)

# 训练时
x = torch.randn(32, input_dim)  # batch_size=32
selected_x = gate(x)  # 应用特征选择

# 获取稀疏损失
sparsity_loss = gate.sparsity_loss().total
total_loss = task_loss + 0.1 * sparsity_loss  # lambda=0.1 控制稀疏程度
```

### 2. 编码器方法

编码器方法直接选择固定 k 个特征：

```python
from deepfs import ConcreteEncoder

# 创建编码器
encoder = ConcreteEncoder(
    input_dim=1000,   # 输入特征数
    output_dim=50     # 选择 50 个特征
)

# 使用
selected_x = encoder(x)  # 输出: (batch_size, 50)

# 训练时更新温度
for epoch in range(100):
    # ... 训练代码 ...
    encoder.update_temperature(epoch)
```

### 3. 组合方法 (推荐)

组合门控和编码器实现两阶段特征选择：

```python
from deepfs import GumbelSoftmaxGate, IndirectConcreteEncoder, GateEncoderSelector

# 创建组件
encoder = IndirectConcreteEncoder(
    input_dim=1000,      # 原始特征数
    output_dim=100,      # 第一阶段选择 k_max=100 个候选
    embedding_dim=32     # 低秩嵌入维度
)

gate = GumbelSoftmaxGate(
    input_dim=100,       # 从 100 个候选中筛选
    embedding_dim=16
)

# 创建选择器
selector = GateEncoderSelector(gate, encoder)

# 使用
x = torch.randn(32, 1000)
selected_x = selector(x)  # 两阶段特征选择

# 获取稀疏损失
sparsity_loss = selector.sparsity_loss().total
```

## 完整训练示例

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
    
# 获取选择的特征数
num_selected = model.gate.num_selected
print(f"选择了 {num_selected} 个特征")
```

## 数据格式支持

DeepFS 支持多种数据格式：

### PyTorch Tensor

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

X = torch.randn(1000, 500)  # 1000 样本, 500 特征
y = torch.randint(0, 3, (1000,))  # 3 分类任务

dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### NumPy 数组

```python
import numpy as np
import torch

X = np.loadtxt("features.csv", delimiter=",")
y = np.loadtxt("labels.csv", delimiter=",")

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)
```

### Pandas DataFrame

```python
import pandas as pd
import torch

df = pd.read_csv("your_data.csv")
X = torch.tensor(df.drop(columns=["label"]).values, dtype=torch.float32)
y = torch.tensor(df["label"].values, dtype=torch.long)
```

## 下一步

- 了解各种[算法详情](Algorithms.md)
- 查看完整的 [API 参考](API-Reference.md)
- 探索更多[使用示例](Examples.md)

---

[English Version](../en/Getting-Started.md)
