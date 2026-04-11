# 实验指南

## 0. 准备数据

数据文件为 `.h5ad` 格式（scanpy/AnnData），需放在 `exp/data/` 下。当前数据集：

| 数据集 | 文件名 | 说明 |
|--------|--------|------|
| Lung | `Lung.h5ad` | 肺组织单细胞 |
| Pancreas | `pancreas.h5ad` | 胰腺单细胞 |
| Spleen | `Spleen.h5ad` | 脾脏单细胞 |
| Tongue | `Tongue.h5ad` | 舌头单细胞 |

每个 `.h5ad` 文件需包含：
- `.X`：特征矩阵（细胞 × 基因）
- `.obs['cell_type']`：标签列

从旧目录拷贝数据（旧目录在 `.gitignore` 中，不会被提交）：

```bash
cp deep-feature-select-old/exp/data/*.h5ad exp/data/
```

## 1. 项目结构

```
exp/
├── configs/                        # YAML 实验配置
│   ├── contrast.yaml               # 对比实验（全部 12 模型）
│   ├── ablation.yaml               # 消融实验（GSG-IPCAE 各参数敏感性）
│   └── hyperparameter.yaml         # 超参数搜索
├── data/                           # 数据加载
│   ├── dataset_all.py              # generate_train_test_loader()
│   └── utils.py                    # MetaData 数据类
├── trainers/                       # 训练器
│   ├── encoder_trainer.py          # 编码器训练（无稀疏损失）
│   ├── gate_trainer.py             # 门控训练（cls + λ × sparse）
│   └── gate_encoder_trainer.py     # 组合训练（cls + λ × sparse + 诊断）
├── visualization/                  # 可视化
│   ├── plot_results.py             # 训练曲线、柱状图、消融图
│   └── generate_tables.py          # LaTeX 表格生成
├── run_contrast.py                 # 对比实验入口
├── run_ablation.py                 # 消融实验入口
├── run_hyperparameter.py           # 超参搜索入口
└── utils.py                        # seed_all, MLPClassifier, Result
```

## 2. 数据流

```
.h5ad 文件
    ↓  sc.read_h5ad()
AnnData (label encoded)
    ↓  AnnCollection → train_test_split
AnnLoader (train / test)
    ↓  每个 batch:
    │   batch.X              → 特征 (torch.Tensor)
    │   batch.obs['cell_type'] → 标签 (torch.Tensor)
    ↓
模型前向: features = model(batch.X)
分类器:   output = classifier(features)
损失:     loss = cross_entropy(output, target) [+ λ × sparsity_loss]
```

## 3. 训练器说明

三个训练器对应三类模型：

| 训练器 | 适用模型 | 损失函数 | 返回值 |
|--------|---------|---------|--------|
| `EncoderTrainer` | CAE, IPCAE | `cross_entropy` | `DataFrame` |
| `GateTrainer` | STG, GSG-Sigmoid, GSG-Softmax, HCG | `cross_entropy + λ × sparse` | `DataFrame` |
| `GateEncoderTrainer` | 6 个组合模型 | `cross_entropy + λ × sparse` | `(result_df, feature_df)` |

每个 epoch 结束后：
1. 调用 `model.update_temperature(epoch)` 进行温度退火
2. 在测试集上评估 accuracy
3. 打印训练日志

## 4. 运行实验

### 4.1 对比实验

对比全部 12 个模型在不同数据集上的表现。

```bash
# 运行全部（编码器 + 门控 + 组合）
python exp/run_contrast.py --config exp/configs/contrast.yaml --mode all

# 只跑某类模型
python exp/run_contrast.py --mode encoder    # 只跑 CAE, IPCAE
python exp/run_contrast.py --mode gate       # 只跑 4 个门控
python exp/run_contrast.py --mode combined   # 只跑 6 个组合模型
```

实验网格：
- 编码器：2 模型 × 19 k 值 × 4 数据集 × 5 seeds = 760 组
- 门控：4 模型 × 4 λ 值 × 4 数据集 × 5 seeds = 320 组
- 组合：6 模型 × 5 k 值 × 4 λ 值 × 4 数据集 × 5 seeds = 2400 组
- **总计 ≈ 3480 组实验**，每组 1000 epochs

输出：`exp/results/contrast/contrast_results.csv`

列含义：`epoch, loss_cls, loss_sparsity, accuracy, num_selected, model, k, sparse_weight, seed, dataset`

### 4.2 消融实验

固定 GSG-IPCAE 模型，逐个变化超参数观察影响。

```bash
python exp/run_ablation.py --config exp/configs/ablation.yaml
```

消融维度：

| 维度 | 变化值 | 固定其他参数 |
|------|--------|-------------|
| `k` (选择槽数) | 5, 10, 15, 20, 25, 30, 35, 40, 45, 50 | λ=1.0, enc_emb=32, gate_emb=16 |
| `λ` (稀疏权重) | 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0 | k=20 |
| `embedding_dim_encoder` | 8, 16, 32, 64, 128, 256 | k=20 |
| `embedding_dim_gate` | 4, 8, 16, 32, 64 | k=20 |
| `temperature_schedule` | 5 组 (初始/终温组合) | k=20 |

输出：`exp/results/ablation/ablation_results.csv` 及每个消融维度的单独 CSV。

### 4.3 超参数搜索

```bash
python exp/run_hyperparameter.py --config exp/configs/hyperparameter.yaml
```

单维度搜索，每次只变一个参数：
- learning_rate: [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
- batch_size: [64, 128, 256, 512, 1024]
- initial_temperature: [1.0, 5.0, 10.0, 20.0, 50.0]
- embedding_dim_encoder: [8, 16, 32, 64, 128]
- embedding_dim_gate: [4, 8, 16, 32, 64]

输出：`exp/results/hyperparameter/hyperparameter_results.csv`

## 5. 修改 YAML 配置

所有实验参数在 YAML 中定义，无需改代码即可调整实验。

### 修改数据集

```yaml
datasets:
  - Lung
  - pancreas
  # - Spleen    # 注释掉不想跑的数据集
```

### 修改训练参数

```yaml
training:
  epochs: 1000       # 训练轮数
  batch_size: 512    # 批大小
  lr: 1e-4           # 学习率
  device: cpu        # cpu 或 cuda
```

### 修改 k 值网格

```yaml
encoder_models:
  - name: CAE
    k_values: [10, 20, 50]   # 缩小搜索范围加快实验
```

### 修改稀疏损失权重

```yaml
gate_models:
  - name: STG
    sparse_loss_weights: [0.1, 1.0, 10.0]   # 缩小范围
```

## 6. 结果分析

### 可视化

```python
from exp.visualization.plot_results import plot_training_curves, plot_accuracy_bar, plot_ablation
import pandas as pd

df = pd.read_csv("exp/results/contrast/contrast_results.csv")
plot_training_curves(df, "exp/results/contrast/figures")
plot_accuracy_bar(df, "exp/results/contrast/figures")
```

### LaTeX 表格

```python
from exp.visualization.generate_tables import generate_comparison_table, generate_ablation_table

df = pd.read_csv("exp/results/contrast/contrast_results.csv")
generate_comparison_table(df, "exp/results/contrast/tables")

ablation_df = pd.read_csv("exp/results/ablation/ablation_results.csv")
generate_ablation_table(ablation_df, "exp/results/ablation/tables")
```

## 7. 用自己的数据

### 方式 A：h5ad 格式（推荐）

将数据放到 `exp/data/` 下，确保有 `.obs['cell_type']` 列：

```python
import scanpy as sc
import numpy as np

adata = sc.AnnData(X=your_feature_matrix)
adata.obs['cell_type'] = your_labels
adata.write_h5ad("exp/data/MyDataset.h5ad")
```

然后在 YAML 的 `datasets` 中加入 `MyDataset`。

### 方式 B：自定义 DataLoader

不使用 `generate_train_test_loader`，直接用 PyTorch DataLoader：

```python
from torch.utils.data import DataLoader, TensorDataset
from deepfs import GumbelSoftmaxGateIndirectConcreteModel
from exp.trainers import GateEncoderTrainer
from exp.utils import MLPClassifier

train_ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)

model = GumbelSoftmaxGateIndirectConcreteModel(
    input_dim=X_train.shape[1], k=50,
    embedding_dim_encoder=32, embedding_dim_gate=16,
    total_epochs=1000,
)
classifier = MLPClassifier(50, 128, num_classes)

trainer = GateEncoderTrainer(model, classifier, sparse_loss_weight=1.0, lr=1e-4)
result_df, feature_df = trainer.fit(train_loader, epochs=1000)
```

> **注意**：使用普通 DataLoader 时，batch 是 `(data, target)` 元组，而非 AnnData 的 `batch.X / batch.obs`。需要修改 trainer 的数据访问方式，或写一个简单的 wrapper：

```python
class AnnDataLikeBatch:
    def __init__(self, x, y):
        self.X = x
        class Obs:
            def __init__(self, cell_type):
                self.cell_type = cell_type
        self.obs = Obs(y)

class AnnDataLikeLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
    def __iter__(self):
        for x, y in self.dataloader:
            yield AnnDataLikeBatch(x, y)
    def __len__(self):
        return len(self.dataloader)

wrapped_loader = AnnDataLikeLoader(train_loader)
```

## 8. GPU 使用

修改 YAML 中的 `device`：

```yaml
training:
  device: cuda    # 改为 cuda 使用 GPU
```

或运行时覆盖（需修改代码支持，当前从 YAML 读取）。

## 9. 常见问题

**Q: 数据文件在哪里？**
A: `.h5ad` 文件不随 git 提交（在 `.gitignore` 中）。从 `deep-feature-select-old/exp/data/` 拷贝。

**Q: 实验跑多久？**
A: 单组实验（1000 epochs，~1000 features，512 batch）在 CPU 上约 2-5 分钟。全量对比实验 3480 组约需数天，建议用 GPU 或减少网格。

**Q: 如何减少实验规模？**
A: 修改 YAML — 减少 `seeds`、`k_values`、`sparse_loss_weights`、`datasets`。

**Q: 结果 CSV 的列含义？**
A:
- `epoch`: 训练轮次
- `loss_cls`: 分类交叉熵损失
- `loss_sparsity`: 稀疏正则损失
- `accuracy`: 测试集准确率
- `num_selected`: 选择的特征数
- `model`: 模型名称
- `k`: 选择槽数（编码器/组合模型）
- `sparse_weight`: λ 稀疏损失权重
- `seed`: 随机种子
- `dataset`: 数据集名称
