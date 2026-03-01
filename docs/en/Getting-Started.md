# Getting Started

This guide will help you get started with the DeepFS library.

## Requirements

- Python >= 3.10
- PyTorch >= 2.0.0
- NumPy >= 1.24.0

## Installation

### Option 1: Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-repo/deep-feature-select.git
cd deep-feature-select

# Install with uv
uv sync
```

### Option 2: Using pip

```bash
git clone https://github.com/your-repo/deep-feature-select.git
cd deep-feature-select
pip install -e .
```

## Basic Usage

### 1. Gate-Only Method

Gate methods control feature count through sparsity loss:

```python
import torch
from deepfs import StochasticGate

# Create gate
input_dim = 1000
gate = StochasticGate(input_dim=input_dim, sigma=0.5)

# Training
x = torch.randn(32, input_dim)  # batch_size=32
selected_x = gate(x)  # Apply feature selection

# Get sparsity loss
sparsity_loss = gate.sparsity_loss().total
total_loss = task_loss + 0.1 * sparsity_loss  # lambda=0.1 controls sparsity
```

### 2. Encoder Method

Encoder methods directly select a fixed number of k features:

```python
from deepfs import ConcreteEncoder

# Create encoder
encoder = ConcreteEncoder(
    input_dim=1000,   # Input features
    output_dim=50     # Select 50 features
)

# Use
selected_x = encoder(x)  # Output: (batch_size, 50)

# Update temperature during training
for epoch in range(100):
    # ... training code ...
    encoder.update_temperature(epoch)
```

### 3. Composite Method (Recommended)

Combine gate and encoder for two-stage feature selection:

```python
from deepfs import GumbelSoftmaxGate, IndirectConcreteEncoder, GateEncoderSelector

# Create components
encoder = IndirectConcreteEncoder(
    input_dim=1000,      # Original features
    output_dim=100,      # Stage 1: select k_max=100 candidates
    embedding_dim=32     # Low-rank embedding dimension
)

gate = GumbelSoftmaxGate(
    input_dim=100,       # Filter from 100 candidates
    embedding_dim=16
)

# Create selector
selector = GateEncoderSelector(gate, encoder)

# Use
x = torch.randn(32, 1000)
selected_x = selector(x)  # Two-stage feature selection

# Get sparsity loss
sparsity_loss = selector.sparsity_loss().total
```

## Complete Training Example

```python
import torch
import torch.nn as nn
from deepfs import StochasticGate

# Define model with feature selection
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

# Training
model = FeatureSelectionClassifier(input_dim=1000, hidden_dim=128, num_classes=5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        
        # Task loss + Sparsity loss
        loss = criterion(outputs, batch_y)
        loss = loss + 0.1 * model.sparsity_loss()
        
        loss.backward()
        optimizer.step()

# Inference
model.eval()
with torch.no_grad():
    outputs = model(test_x)
    predictions = outputs.argmax(dim=1)
    
# Get selected feature count
num_selected = model.gate.num_selected
print(f"Selected {num_selected} features")
```

## Supported Data Formats

DeepFS supports multiple data formats:

### PyTorch Tensor

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

X = torch.randn(1000, 500)  # 1000 samples, 500 features
y = torch.randint(0, 3, (1000,))  # 3-class classification

dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### NumPy Array

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

## Next Steps

- Learn about [Algorithm Details](Algorithms.md)
- Check the complete [API Reference](API-Reference.md)
- Explore more [Examples](Examples.md)

---

[中文版本](../zh-CN/Getting-Started.md)
