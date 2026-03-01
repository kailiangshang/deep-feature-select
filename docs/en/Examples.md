# Examples

This document provides detailed usage examples for DeepFS.

---

## Example 1: Basic Classification Task

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from deepfs import StochasticGate

# 1. Prepare data
X = torch.randn(1000, 500)  # 1000 samples, 500 features
y = torch.randint(0, 5, (1000,))  # 5-class classification

train_dataset = TensorDataset(X[:800], y[:800])
test_dataset = TensorDataset(X[800:], y[800:])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# 2. Define model
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

# 3. Train
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
    
    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Selected features: {model.gate.num_selected}")

# 4. Evaluate
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

## Example 2: Two-Stage Feature Selection (GSG-IPCAE)

```python
import torch
import torch.nn as nn
from deepfs import GumbelSoftmaxGate, IndirectConcreteEncoder, GateEncoderSelector

# 1. Create two-stage selector
input_dim = 1000  # Original features
k_max = 100       # Candidates from stage 1

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

# 2. Define model
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

# 3. Train
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    model.train()
    model.update_temperature(epoch)  # Update temperature
    
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y) + 0.1 * model.sparsity_loss()
        loss.backward()
        optimizer.step()

# 4. Get selection result
result = model.selector.get_selection_result()
print(f"Encoder selected indices: {model.selector.selected_indices}")
print(f"Gate selected count: {result.num_selected}")
```

---

## Example 3: Using the Trainer

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

# 1. Prepare data
X = torch.randn(1000, 200)
y = torch.randn(1000, 1)  # Regression task

train_loader = DataLoader(
    TensorDataset(X[:800], y[:800]), 
    batch_size=32, 
    shuffle=True
)
val_loader = DataLoader(
    TensorDataset(X[800:], y[800:]), 
    batch_size=32
)

# 2. Create model
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

# 3. Configure training
config = TrainConfig(
    epochs=50,
    batch_size=32,
    learning_rate=1e-3,
    sparsity_weight=0.1,
    verbose=True,
    print_every=10
)

# 4. Create trainer
trainer = FeatureSelectionTrainer(model, selector, config)

# 5. Train
history = trainer.train(
    train_loader, 
    val_loader, 
    criterion=nn.MSELoss()
)

# 6. Get selected features
selected = trainer.get_selected_features()
print(f"Selected features: {selected}")
```

---

## Example 4: Custom Callbacks

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

# Use callbacks
trainer = FeatureSelectionTrainer(model, selector, config)
trainer.add_callback(EarlyStoppingCallback(patience=10))
trainer.add_callback(SparsityMonitorCallback(log_every=5))
trainer.train(train_loader, val_loader, criterion=nn.MSELoss())
```

---

## Example 5: Effect of Different Lambda Values

```python
import torch
from deepfs import StochasticGate

# Test different sparsity coefficients
lambdas = [0.01, 0.1, 1.0, 10.0]
results = []

for lam in lambdas:
    gate = StochasticGate(input_dim=100)
    optimizer = torch.optim.Adam(gate.parameters())
    
    # Simple training
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

# Expected output:
# lambda=0.01: ~90 features (weak sparsity)
# lambda=0.1: ~50 features
# lambda=1.0: ~20 features
# lambda=10.0: ~5 features (strong sparsity)
```

---

## Example 6: Getting Feature Importance

```python
import torch
import matplotlib.pyplot as plt
from deepfs import GumbelSigmoidGate

# Get feature importance after training
gate = GumbelSigmoidGate(input_dim=50)

# ... training code ...

# Get gate probabilities as feature importance
importance = gate.gate_probs.detach().numpy()

# Visualize
plt.figure(figsize=(12, 4))
plt.bar(range(50), importance)
plt.xlabel('Feature Index')
plt.ylabel('Importance (Gate Probability)')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

# Get top-k features
k = 10
top_k_indices = importance.argsort()[-k:][::-1]
print(f"Top {k} features: {top_k_indices}")
```

---

## Example 7: Comparing Different Algorithms

```python
import torch
import torch.nn as nn
from deepfs import (
    StochasticGate,
    GumbelSigmoidGate,
    GumbelSoftmaxGate,
    HardConcreteGate
)

# Prepare data
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
    # Simple classifier
    classifier = nn.Linear(100, 3)
    optimizer = torch.optim.Adam(
        list(gate.parameters()) + list(classifier.parameters())
    )
    criterion = nn.CrossEntropyLoss()
    
    # Train
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

# Print results
for name, r in results.items():
    print(f"{name}: {r['num_selected']} features, sparsity={r['sparsity']:.2%}")
```

---

[中文版本](../zh-CN/Examples.md)
