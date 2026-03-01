# DeepFS 实验计划

## 一、实验目标

验证 **GSG-IPCAE** (GumbelSoftmaxGate + IndirectConcreteEncoder) 两阶段特征选择方法的有效性，与基线方法进行对比。

## 二、方法概述

### 2.1 我们的方法: GSG-IPCAE

两阶段特征选择：
1. **Stage 1 (IPCAE)**: 从 D 维特征中选择 k_max 个候选特征
2. **Stage 2 (GSG)**: 通过门控从 k_max 个候选中自动筛选最终特征

**优势**: 无需指定精确的特征数量 k，通过稀疏损失自动控制

### 2.2 基线方法

| 方法 | 类型 | 特点 |
|------|------|------|
| **CAE** | 编码器 | 直接参数化，需指定 k |
| **IPCAE** | 编码器 | 低秩参数化，需指定 k |
| **门控方法** | 门控 | 稀疏损失控制，特征数不固定 |

## 三、实验设计

### 3.1 基线网格搜索

由于 CAE 和 IPCAE 需要指定特征数量 k，对其进行网格搜索以获得最优结果：

| 方法 | 搜索参数 | 搜索范围 |
|------|---------|---------|
| CAE | k | [1, 2, 3, ..., 50] (共 50 组) |
| IPCAE | k | [1, 2, 3, ..., 50] (共 50 组) |

其他超参数固定：
- initial_temperature: 10.0
- final_temperature: 0.01
- embedding_dim (IPCAE): 32

### 3.2 门控方法基线

| 方法 | 搜索参数 | 搜索范围 |
|------|---------|---------|
| StochasticGate | λ | [0.001, 0.01, 0.1, 1, 10] |
| GumbelSigmoidGate | λ | [0.001, 0.01, 0.1, 1, 10] |
| HardConcreteGate | λ | [0.001, 0.01, 0.1, 1, 10] |

### 3.3 我们的方法配置

GSG-IPCAE 使用固定配置，无需搜索 k：

| 参数 | 值 | 说明 |
|------|-----|------|
| k_max | 100 | 候选特征数上限 |
| gate_embedding_dim | 16 | 门控嵌入维度 |
| encoder_embedding_dim | 32 | 编码器嵌入维度 |
| λ | 0.1 | 稀疏损失权重 |

### 3.4 消融实验

| 实验 | 变量 | 搜索范围 | 固定参数 |
|------|------|---------|---------|
| A1 | k_max | [50, 100, 200, 500] | λ=0.1 |
| A2 | λ | [0.01, 0.1, 1, 10] | k_max=100 |

## 四、评估指标

| 指标 | 描述 |
|------|------|
| **Accuracy** | 分类准确率 (mean ± std) |
| **Selected Features** | 最终选择的特征数量 |
| **Sparsity** | 稀疏度 = 1 - (选择数 / 总特征数) |
| **Parameters** | 模型参数量 |

## 五、训练配置

```python
epochs = 100
batch_size = 32
learning_rate = 1e-3
optimizer = Adam
seeds = [42, 123, 456, 789, 1024]  # 5次运行取平均
```

## 六、运行方式

```bash
# 完整实验
python exp/run_experiment.py --dataset pancreas --epochs 100

# 快速测试
python exp/run_experiment.py --dataset pancreas --quick

# 指定参数
python exp/run_experiment.py --dataset pancreas --epochs 50 --seeds 42 123 456
```

## 七、结果输出

```
exp/results/
├── cae_results.csv         # CAE k=1-50 结果
├── ipcae_results.csv       # IPCAE k=1-50 结果
├── gate_results.csv        # 门控方法结果
├── gsg_ipcae_results.csv   # GSG-IPCAE 结果
├── ablation_results.csv    # 消融实验结果
└── all_results.csv         # 汇总结果
```
