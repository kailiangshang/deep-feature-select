# API 参考

本文档提供 DeepFS 库的完整 API 参考。

---

## 核心模块 (deepfs.core)

### SparsityLoss

稀疏损失数据类。

```python
@dataclass
class SparsityLoss:
    names: List[str]    # 损失名称列表
    values: List[Tensor] # 损失值列表
    
    @property
    def total(self) -> Tensor:
        """返回所有损失值的总和"""
```

### SelectionResult

特征选择结果数据类。

```python
@dataclass
class SelectionResult:
    selected_indices: np.ndarray  # 选中的特征索引
    selected_mask: np.ndarray     # 布尔掩码
    gate_probs: Optional[np.ndarray]  # 门控概率
    num_selected: int             # 选中的特征数
```

### TemperatureSchedule

温度退火调度器。

```python
class TemperatureSchedule:
    def __init__(
        self,
        initial: float = 10.0,
        final: float = 0.01,
        total_epochs: int = 100
    )
    
    def get_temperature(self, epoch: int) -> float:
        """获取指定 epoch 的温度值"""
```

---

## 门控模块 (deepfs.gates)

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
        """应用随机门控"""
    
    def sparsity_loss(self) -> SparsityLoss:
        """计算稀疏损失 (高斯 CDF)"""
    
    @property
    def num_selected(self) -> int:
        """获取选中的特征数"""
    
    @property
    def gate_probs(self) -> Tensor:
        """获取门控概率"""
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
        """应用 Gumbel-Sigmoid 门控"""
    
    def update_temperature(self, epoch: int) -> None:
        """更新温度"""
    
    def sparsity_loss(self) -> SparsityLoss:
        """计算稀疏损失 (L1 + 熵)"""
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
        """应用嵌入式 Gumbel-Softmax 门控"""
    
    def update_temperature(self, epoch: int) -> None:
        """更新温度"""
    
    def sparsity_loss(self) -> SparsityLoss:
        """计算稀疏损失 (L1)"""
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
        """应用硬混凝土门控"""
    
    def sparsity_loss(self) -> SparsityLoss:
        """计算 L0 正则化损失"""
```

---

## 编码器模块 (deepfs.encoders)

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
        """应用混凝土特征选择"""
    
    def update_temperature(self, epoch: int) -> None:
        """更新温度"""
    
    @property
    def selected_indices(self) -> Tensor:
        """获取选中的特征索引"""
    
    def hard_forward(self, x: Tensor) -> Tensor:
        """硬选择前向传播"""
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
        """应用间接混凝土特征选择"""
    
    @property
    def logits(self) -> Tensor:
        """从嵌入计算 logits"""
    
    def update_temperature(self, epoch: int) -> None:
        """更新温度"""
    
    @property
    def selected_indices(self) -> Tensor:
        """获取选中的特征索引"""
```

---

## 选择器模块 (deepfs.selectors)

### GateEncoderSelector

```python
class GateEncoderSelector(CompositeSelector):
    def __init__(
        self,
        gate: GateBase,
        encoder: EncoderBase
    )
    
    def forward(self, x: Tensor) -> Tensor:
        """应用门控 + 编码器特征选择"""
    
    def sparsity_loss(self) -> SparsityLoss:
        """获取门控的稀疏损失"""
    
    def update_temperature(self, epoch: int) -> None:
        """更新门控和编码器的温度"""
    
    @property
    def selected_indices(self) -> Tensor:
        """获取编码器选中的特征索引"""
    
    @property
    def gate_probs(self) -> Tensor:
        """获取门控概率"""
    
    @property
    def num_selected(self) -> int:
        """获取门控选中的特征数"""
    
    def get_selection_result(self) -> SelectionResult:
        """获取详细的选择结果"""
    
    def hard_forward(self, x: Tensor) -> Tensor:
        """硬选择前向传播"""
```

---

## 训练模块 (deepfs.training)

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
        """添加回调"""
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        criterion: Optional[Callable] = None
    ) -> Dict[str, List[float]]:
        """训练模型"""
    
    def get_selected_features(self) -> np.ndarray:
        """获取选中的特征索引"""
```

### TemperatureCallback

```python
class TemperatureCallback:
    """温度更新回调"""
    
    def on_epoch_end(self, trainer, epoch: int, **kwargs) -> None:
        """epoch 结束时更新温度"""
```

### LoggingCallback

```python
class LoggingCallback:
    """日志记录回调"""
    
    def __init__(self, log_every: int = 10)
    
    def on_epoch_end(self, trainer, epoch: int, metrics: dict, **kwargs) -> None:
        """记录训练指标"""
```

---

## 工具函数

### 注册表函数

```python
def register_gate(name: str) -> Callable:
    """注册门控类的装饰器"""

def register_encoder(name: str) -> Callable:
    """注册编码器类的装饰器"""

def get_gate(name: str) -> Type[GateBase]:
    """获取注册的门控类"""

def get_encoder(name: str) -> Type[EncoderBase]:
    """获取注册的编码器类"""

def create_gate(name: str, **kwargs) -> GateBase:
    """创建门控实例"""

def create_encoder(name: str, **kwargs) -> EncoderBase:
    """创建编码器实例"""

def list_gates() -> List[str]:
    """列出所有注册的门控名称"""

def list_encoders() -> List[str]:
    """列出所有注册的编码器名称"""
```

---

## 使用示例

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

# 创建组件
gate = GumbelSoftmaxGate(input_dim=100, embedding_dim=16)
encoder = IndirectConcreteEncoder(input_dim=1000, output_dim=100, embedding_dim=32)
selector = GateEncoderSelector(gate, encoder)

# 获取选择结果
result = selector.get_selection_result()
print(f"Selected {result.num_selected} features")
print(f"Indices: {result.selected_indices}")
```

---

[English Version](../en/API-Reference.md)
