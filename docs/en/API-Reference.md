# API Reference

This document provides the complete API reference for the DeepFS library.

---

## Core Module (deepfs.core)

### SparsityLoss

Sparsity loss dataclass.

```python
@dataclass
class SparsityLoss:
    names: List[str]    # Loss name list
    values: List[Tensor] # Loss value list
    
    @property
    def total(self) -> Tensor:
        """Returns sum of all loss values"""
```

### SelectionResult

Feature selection result dataclass.

```python
@dataclass
class SelectionResult:
    selected_indices: np.ndarray  # Selected feature indices
    selected_mask: np.ndarray     # Boolean mask
    gate_probs: Optional[np.ndarray]  # Gate probabilities
    num_selected: int             # Number of selected features
```

### TemperatureSchedule

Temperature annealing scheduler.

```python
class TemperatureSchedule:
    def __init__(
        self,
        initial: float = 10.0,
        final: float = 0.01,
        total_epochs: int = 100
    )
    
    def get_temperature(self, epoch: int) -> float:
        """Get temperature value for specified epoch"""
```

---

## Gates Module (deepfs.gates)

### StochasticGate

```python
class StochasticGate(GateBase):
    def __init__(
        self,
        input_dim: int,
        sigma: float = 0.5,
        hard_gate_type: str = "hard_zero",
        device: str = "cpu"
    )
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply stochastic gate"""
    
    def sparsity_loss(self) -> SparsityLoss:
        """Compute sparsity loss (Gaussian CDF)"""
    
    @property
    def num_selected(self) -> int:
        """Get number of selected features"""
    
    @property
    def gate_probs(self) -> Tensor:
        """Get gate probabilities"""
```

### GumbelSigmoidGate

```python
class GumbelSigmoidGate(GateBase):
    def __init__(
        self,
        input_dim: int,
        initial_temperature: float = 10.0,
        final_temperature: float = 0.01,
        total_epochs: int = 100,
        hard_gate_type: str = "hard_zero",
        device: str = "cpu"
    )
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply Gumbel-Sigmoid gate"""
    
    def update_temperature(self, epoch: int) -> None:
        """Update temperature"""
    
    def sparsity_loss(self) -> SparsityLoss:
        """Compute sparsity loss (L1 + Entropy)"""
```

### GumbelSoftmaxGate

```python
class GumbelSoftmaxGate(GateBase):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        initial_temperature: float = 10.0,
        final_temperature: float = 0.01,
        total_epochs: int = 100,
        hard_gate_type: str = "hard_zero",
        device: str = "cpu"
    )
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply embedded Gumbel-Softmax gate"""
    
    def update_temperature(self, epoch: int) -> None:
        """Update temperature"""
    
    def sparsity_loss(self) -> SparsityLoss:
        """Compute sparsity loss (L1)"""
```

### HardConcreteGate

```python
class HardConcreteGate(GateBase):
    def __init__(
        self,
        input_dim: int,
        min_max_scale: Tuple[float, float] = (-0.1, 1.1),
        temperature: float = 0.5,
        hard_gate_type: str = "hard_zero",
        device: str = "cpu"
    )
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply hard concrete gate"""
    
    def sparsity_loss(self) -> SparsityLoss:
        """Compute L0 regularization loss"""
```

---

## Encoders Module (deepfs.encoders)

### ConcreteEncoder

```python
class ConcreteEncoder(EncoderBase):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        initial_temperature: float = 10.0,
        final_temperature: float = 0.01,
        total_epochs: int = 100,
        device: str = "cpu"
    )
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply concrete feature selection"""
    
    def update_temperature(self, epoch: int) -> None:
        """Update temperature"""
    
    @property
    def selected_indices(self) -> Tensor:
        """Get selected feature indices"""
    
    def hard_forward(self, x: Tensor) -> Tensor:
        """Hard selection forward pass"""
```

### IndirectConcreteEncoder

```python
class IndirectConcreteEncoder(EncoderBase):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        embedding_dim: int,
        initial_temperature: float = 10.0,
        final_temperature: float = 0.01,
        total_epochs: int = 100,
        device: str = "cpu"
    )
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply indirect concrete feature selection"""
    
    @property
    def logits(self) -> Tensor:
        """Compute logits from embeddings"""
    
    def update_temperature(self, epoch: int) -> None:
        """Update temperature"""
    
    @property
    def selected_indices(self) -> Tensor:
        """Get selected feature indices"""
```

---

## Selectors Module (deepfs.selectors)

### GateEncoderSelector

```python
class GateEncoderSelector(CompositeSelector):
    def __init__(
        self,
        gate: GateBase,
        encoder: EncoderBase
    )
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply gate + encoder feature selection"""
    
    def sparsity_loss(self) -> SparsityLoss:
        """Get gate's sparsity loss"""
    
    def update_temperature(self, epoch: int) -> None:
        """Update temperature for both gate and encoder"""
    
    @property
    def selected_indices(self) -> Tensor:
        """Get encoder's selected feature indices"""
    
    @property
    def gate_probs(self) -> Tensor:
        """Get gate probabilities"""
    
    @property
    def num_selected(self) -> int:
        """Get number of features selected by gate"""
    
    def get_selection_result(self) -> SelectionResult:
        """Get detailed selection result"""
    
    def hard_forward(self, x: Tensor) -> Tensor:
        """Hard selection forward pass"""
```

---

## Training Module (deepfs.training)

### TrainConfig

```python
@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    sparsity_weight: float = 0.1
    device: str = "cpu"
    verbose: bool = True
    print_every: int = 10
```

### FeatureSelectionTrainer

```python
class FeatureSelectionTrainer:
    def __init__(
        self,
        model: nn.Module,
        selector: BaseSelector,
        config: Optional[TrainConfig] = None
    )
    
    def add_callback(self, callback: Any) -> None:
        """Add callback"""
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        criterion: Optional[Callable] = None
    ) -> Dict[str, List[float]]:
        """Train the model"""
    
    def get_selected_features(self) -> np.ndarray:
        """Get selected feature indices"""
```

### TemperatureCallback

```python
class TemperatureCallback:
    """Temperature update callback"""
    
    def on_epoch_end(self, trainer, epoch: int, **kwargs) -> None:
        """Update temperature at epoch end"""
```

### LoggingCallback

```python
class LoggingCallback:
    """Logging callback"""
    
    def __init__(self, log_every: int = 10)
    
    def on_epoch_end(self, trainer, epoch: int, metrics: dict, **kwargs) -> None:
        """Log training metrics"""
```

---

## Utility Functions

### Registry Functions

```python
def register_gate(name: str) -> Callable:
    """Decorator to register gate class"""

def register_encoder(name: str) -> Callable:
    """Decorator to register encoder class"""

def get_gate(name: str) -> Type[GateBase]:
    """Get registered gate class"""

def get_encoder(name: str) -> Type[EncoderBase]:
    """Get registered encoder class"""

def create_gate(name: str, **kwargs) -> GateBase:
    """Create gate instance"""

def create_encoder(name: str, **kwargs) -> EncoderBase:
    """Create encoder instance"""

def list_gates() -> List[str]:
    """List all registered gate names"""

def list_encoders() -> List[str]:
    """List all registered encoder names"""
```

---

## Usage Example

```python
from deepfs import (
    # Gates
    StochasticGate,
    GumbelSigmoidGate,
    GumbelSoftmaxGate,
    HardConcreteGate,
    # Encoders
    ConcreteEncoder,
    IndirectConcreteEncoder,
    # Selectors
    GateEncoderSelector,
    # Training
    FeatureSelectionTrainer,
    TrainConfig,
    TemperatureCallback,
)

# Create components
gate = GumbelSoftmaxGate(input_dim=100, embedding_dim=16)
encoder = IndirectConcreteEncoder(input_dim=1000, output_dim=100, embedding_dim=32)
selector = GateEncoderSelector(gate, encoder)

# Get selection result
result = selector.get_selection_result()
print(f"Selected {result.num_selected} features")
print(f"Indices: {result.selected_indices}")
```

---

[中文版本](../zh-CN/API-Reference.md)
