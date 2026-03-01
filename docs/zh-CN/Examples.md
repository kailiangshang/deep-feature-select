# 使用示例

本文档提供 DeepFS 的详细使用示例。

---

## 示例 1: 基础分类任务

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from deepfs import StochasticGate

# 1. 准备数据
X = torch.randn(1000, 500)  # 1000 样本, 500 特征
y = torch.randint(0, 5, (1000,))  # 5 类分类

train_dataset = TensorDataset(X[:800], y[:800])
test_dataset = TensorDataset(X[800:], y[800:])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# 2. 定义模型
class Classifier(nn.Module):
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

# 3. 训练
model = Classifier(500, 128, 5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(50):
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y) + 0.1 * model.sparsity_loss()
        loss.backward()
        optimizer.step()
    
    # 打印进度
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Selected features: {model.gate.num_selected}")

# 4. 评估
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x)
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()

print(f"Accuracy: {100.*correct/total:.2f}%")
print(f"Final selected features: {model.gate.num_selected}")
```

---

## 示例 2: 两阶段特征选择 (GSG-IPCAE)

```python
import torch
import torch.nn as nn
from deepfs import GumbelSoftmaxGate, IndirectConcreteEncoder, GateEncoderSelector

# 1. 创建两阶段选择器
input_dim = 1000  # 原始特征数
k_max = 100       # 第一阶段选择的候选数

encoder = IndirectConcreteEncoder(
    input_dim=input_dim,
    output_dim=k_max,
    embedding_dim=32
)

gate = GumbelSoftmaxGate(
    input_dim=k_max,
    embedding_dim=16
)

selector = GateEncoderSelector(gate, encoder)

# 2. 定义模型
class TwoStageModel(nn.Module):
    def __init__(self, selector, hidden_dim, num_classes):
        super().__init__()
        self.selector = selector
        self.classifier = nn.Sequential(
            nn.Linear(k_max, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        x = self.selector(x)
        return self.classifier(x)
    
    def sparsity_loss(self):
        return self.selector.sparsity_loss().total
    
    def update_temperature(self, epoch):
        self.selector.update_temperature(epoch)

model = TwoStageModel(selector, 128, 5)

# 3. 训练
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    model.train()
    model.update_temperature(epoch)  # 更新温度
    
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y) + 0.1 * model.sparsity_loss()
        loss.backward()
        optimizer.step()

# 4. 获取选择结果
result = model.selector.get_selection_result()
print(f"Encoder selected indices: {model.selector.selected_indices}")
print(f"Gate selected count: {result.num_selected}")
```

---

## 示例 3: 使用 Trainer

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from deepfs import (
    StochasticGate, 
    FeatureSelectionTrainer, 
    TrainConfig,
    TemperatureCallback
)

# 1. 准备数据
X = torch.randn(1000, 200)
y = torch.randn(1000, 1)  # 回归任务

train_loader = DataLoader(
    TensorDataset(X[:800], y[:800]), 
    batch_size=32, 
    shuffle=True
)
val_loader = DataLoader(
    TensorDataset(X[800:], y[800:]), 
    batch_size=32
)

# 2. 创建模型
class RegressionModel(nn.Module):
    def __init__(self, input_dim, selector):
        super().__init__()
        self.selector = selector
        self.fc = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        x = self.selector(x)
        return self.fc(x)

selector = StochasticGate(input_dim=200)
model = RegressionModel(200, selector)

# 3. 配置训练
config = TrainConfig(
    epochs=50,
    batch_size=32,
    learning_rate=1e-3,
    sparsity_weight=0.1,
    verbose=True,
    print_every=10
)

# 4. 创建训练器
trainer = FeatureSelectionTrainer(model, selector, config)

# 5. 训练
history = trainer.train(
    train_loader, 
    val_loader, 
    criterion=nn.MSELoss()
)

# 6. 获取选择的特征
selected = trainer.get_selected_features()
print(f"Selected features: {selected}")
```

---

## 示例 4: 自定义回调

```python
from deepfs import FeatureSelectionTrainer, TrainConfig

class EarlyStoppingCallback:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False
    
    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        val_loss = metrics.get('val_loss', float('inf'))
        
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.should_stop = True
            print(f"Early stopping at epoch {epoch}")

class SparsityMonitorCallback:
    def __init__(self, log_every=5):
        self.log_every = log_every
    
    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        if (epoch + 1) % self.log_every == 0:
            num_selected = trainer.selector.num_selected
            print(f"Epoch {epoch+1}: {num_selected} features selected")

# 使用回调
trainer = FeatureSelectionTrainer(model, selector, config)
trainer.add_callback(EarlyStoppingCallback(patience=10))
trainer.add_callback(SparsityMonitorCallback(log_every=5))
trainer.train(train_loader, val_loader, criterion=nn.MSELoss())
```

---

## 示例 5: 不同 lambda 值的影响

```python
import torch
from deepfs import StochasticGate

# 测试不同的稀疏系数
lambdas = [0.01, 0.1, 1.0, 10.0]
results = []

for lam in lambdas:
    gate = StochasticGate(input_dim=100)
    optimizer = torch.optim.Adam(gate.parameters())
    
    # 简单训练
    for _ in range(100):
        x = torch.randn(32, 100)
        output = gate(x)
        loss = output.mean() + lam * gate.sparsity_loss().total
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    results.append({
        'lambda': lam,
        'num_selected': gate.num_selected
    })

for r in results:
    print(f"lambda={r['lambda']}: {r['num_selected']} features")

# 预期输出:
# lambda=0.01: ~90 features (弱稀疏)
# lambda=0.1: ~50 features
# lambda=1.0: ~20 features
# lambda=10.0: ~5 features (强稀疏)
```

---

## 示例 6: 获取特征重要性

```python
import torch
import matplotlib.pyplot as plt
from deepfs import GumbelSigmoidGate

# 训练后获取特征重要性
gate = GumbelSigmoidGate(input_dim=50)

# ... 训练代码 ...

# 获取门控概率作为特征重要性
importance = gate.gate_probs.detach().numpy()

# 可视化
plt.figure(figsize=(12, 4))
plt.bar(range(50), importance)
plt.xlabel('Feature Index')
plt.ylabel('Importance (Gate Probability)')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

# 获取 top-k 特征
k = 10
top_k_indices = importance.argsort()[-k:][::-1]
print(f"Top {k} features: {top_k_indices}")
```

---

## 示例 7: 比较不同算法

```python
import torch
import torch.nn as nn
from deepfs import (
    StochasticGate,
    GumbelSigmoidGate,
    GumbelSoftmaxGate,
    HardConcreteGate
)

# 准备数据
X_train = torch.randn(500, 100)
y_train = torch.randint(0, 3, (500,))

gates = {
    'STG': StochasticGate(100),
    'GSG': GumbelSigmoidGate(100),
    'GumbelSoftmax': GumbelSoftmaxGate(100, embedding_dim=16),
    'HCG': HardConcreteGate(100)
}

results = {}

for name, gate in gates.items():
    # 简单分类器
    classifier = nn.Linear(100, 3)
    optimizer = torch.optim.Adam(
        list(gate.parameters()) + list(classifier.parameters())
    )
    criterion = nn.CrossEntropyLoss()
    
    # 训练
    for epoch in range(50):
        if hasattr(gate, 'update_temperature'):
            gate.update_temperature(epoch)
        
        x = gate(X_train)
        loss = criterion(classifier(x), y_train)
        loss = loss + 0.1 * gate.sparsity_loss().total
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    results[name] = {
        'num_selected': gate.num_selected,
        'sparsity': 1 - gate.num_selected / 100
    }

# 打印结果
for name, r in results.items():
    print(f"{name}: {r['num_selected']} features, sparsity={r['sparsity']:.2%}")
```

---

[English Version](../en/Examples.md)
