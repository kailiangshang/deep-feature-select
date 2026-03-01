# DeepFS Wiki

Welcome to the DeepFS (Deep Feature Selection) project documentation!

## Project Overview

DeepFS is a flexible, modular deep feature selection library that implements multiple state-of-the-art differentiable feature selection algorithms. It is designed for feature selection in high-dimensional data (such as gene expression data).

## Core Features

- **Modular Design**: Gate, Encoder, and Selector components can be freely combined
- **Multiple Algorithms**: 4 gate algorithms + 2 encoder algorithms + 8 combination methods
- **Easy to Extend**: Clear base class design for implementing new algorithms
- **Production Ready**: Complete type annotations and docstrings

## Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](Getting-Started.md) | Installation and basic usage guide |
| [Algorithms](Algorithms.md) | Detailed explanation of feature selection algorithms |
| [API Reference](API-Reference.md) | Complete API documentation |
| [Examples](Examples.md) | Detailed code examples |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DeepFS Architecture                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Gates     │    │  Encoders   │    │  Selectors  │     │
│  ├─────────────┤    ├─────────────┤    ├─────────────┤     │
│  │ Stochastic  │    │  Concrete   │    │  Composite  │     │
│  │ GumbelSig   │    │  Indirect   │    │ GateEncoder │     │
│  │ GumbelSoft  │    │  Concrete   │    │             │     │
│  │ HardConcrete│    │             │    │             │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │             │
│         └──────────────────┼──────────────────┘             │
│                            ▼                                │
│                  ┌─────────────────┐                        │
│                  │    Training     │                        │
│                  │    Utilities    │                        │
│                  └─────────────────┘                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Core Algorithm: GSG-IPCAE

Our core algorithm combines **Embedded Gumbel-Softmax Gate (GSG) + Indirect Parameterized Concrete Autoencoder (IPCAE)** for two-stage feature selection:

1. **Stage 1 (IPCAE)**: Select k_max candidate features from original D dimensions
2. **Stage 2 (GSG)**: Further filter optimal features from k_max candidates

### Advantages

| Feature | Traditional Methods | GSG-IPCAE |
|---------|---------------------|-----------|
| Parameters | O(D) or O(D²) | O(D×d) low-rank |
| Feature Count | Fixed or unstable | Two-stage precise control |
| Sparsity | Requires L1 regularization | Natural sparsity + L0 regularization |
| Differentiability | Partially differentiable | Fully differentiable |

## Quick Links

- [GitHub Repository](https://github.com/your-repo/deep-feature-select)
- [Issue Tracker](https://github.com/your-repo/deep-feature-select/issues)

---

[中文版本](../zh-CN/Home.md)
