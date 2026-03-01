"""
DeepFS 完整实验脚本

按实验计划顺序执行所有实验：
1. 基线实验 - CAE 网格搜索 (k=1-50)
2. 基线实验 - IPCAE 网格搜索 (k=1-50)
3. 基线实验 - 门控方法 (λ 搜索)
4. 我们的方法 - GSG-IPCAE
5. 消融实验
"""
from __future__ import annotations

import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm

# DeepFS 导入
from deepfs import (
    StochasticGate,
    GumbelSigmoidGate,
    GumbelSoftmaxGate,
    HardConcreteGate,
    ConcreteEncoder,
    IndirectConcreteEncoder,
    GateEncoderSelector,
)

# 数据导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dataset_all import generate_train_test_loader
from data.utils import MetaData


# ============================================================================
# 配置
# ============================================================================

@dataclass
class ExperimentConfig:
    """实验配置"""
    # 数据
    dataset_name: str = "pancreas"
    
    # 训练
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    seeds: List[int] = None
    
    # 模型
    hidden_dim: int = 128
    
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 结果保存
    result_dir: str = "exp/results"
    
    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [42, 123, 456, 789, 1024]
        os.makedirs(self.result_dir, exist_ok=True)


@dataclass
class ExperimentResult:
    """实验结果"""
    experiment_id: str
    algorithm: str
    category: str
    hyperparameters: Dict[str, Any]
    accuracy: float
    accuracy_std: float
    sparsity: float
    num_selected: int
    num_params: int
    train_time: float


# ============================================================================
# 模型定义
# ============================================================================

class MLPClassifier(nn.Module):
    """MLP 分类器"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)


class EncoderModel(nn.Module):
    """编码器模型 (CAE/IPCAE)"""
    
    def __init__(self, encoder: nn.Module, hidden_dim: int, output_dim: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = MLPClassifier(encoder.output_dim, hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))
    
    def update_temperature(self, epoch: int) -> None:
        self.encoder.update_temperature(epoch)


class GateModel(nn.Module):
    """门控模型"""
    
    def __init__(self, gate: nn.Module, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.gate = gate
        self.classifier = MLPClassifier(input_dim, hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.gate(x))
    
    def sparsity_loss(self) -> torch.Tensor:
        return self.gate.sparsity_loss().total
    
    def update_temperature(self, epoch: int) -> None:
        if hasattr(self.gate, 'update_temperature'):
            self.gate.update_temperature(epoch)


class GSGIPCAEModel(nn.Module):
    """GSG-IPCAE 模型 (我们的方法)"""
    
    def __init__(self, selector: GateEncoderSelector, hidden_dim: int, output_dim: int):
        super().__init__()
        self.selector = selector
        self.classifier = MLPClassifier(selector.encoder.output_dim, hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.selector(x))
    
    def sparsity_loss(self) -> torch.Tensor:
        return self.selector.sparsity_loss().total
    
    def update_temperature(self, epoch: int) -> None:
        self.selector.update_temperature(epoch)


# ============================================================================
# 工具函数
# ============================================================================

def set_seed(seed: int) -> None:
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def count_parameters(model: nn.Module) -> int:
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# 训练函数
# ============================================================================

def train_epoch(model, train_loader, optimizer, criterion, device, 
                lambda_sparsity=0.0, use_sparsity=False):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch in train_loader:
        x = batch.X.to(device)
        y = batch.obs['cell_type'].to(device)
        
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        
        if use_sparsity and hasattr(model, 'sparsity_loss'):
            loss = loss + lambda_sparsity * model.sparsity_loss()
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        _, predicted = outputs.max(1)
        total_correct += predicted.eq(y).sum().item()
        total_samples += x.size(0)
    
    return total_loss / total_samples, total_correct / total_samples


def evaluate(model, loader, criterion, device, is_anndata=True):
    """评估模型"""
    model.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        if is_anndata:
            for batch in loader:
                x = batch.X.to(device)
                y = batch.obs['cell_type'].to(device)
                outputs = model(x)
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(y).sum().item()
                total_samples += x.size(0)
        else:
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(y).sum().item()
                total_samples += x.size(0)
    
    return total_correct / total_samples


def train_and_evaluate(model, metadata, config, lambda_sparsity=0.0, use_sparsity=False):
    """完整训练和评估流程"""
    model = model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    
    for epoch in range(config.epochs):
        if hasattr(model, 'update_temperature'):
            model.update_temperature(epoch)
        
        train_epoch(model, metadata.train_loader, optimizer, criterion, 
                   config.device, lambda_sparsity, use_sparsity)
    
    train_time = time.time() - start_time
    test_acc = evaluate(model, metadata.test_loader, criterion, config.device, is_anndata=True)
    
    return test_acc, train_time


# ============================================================================
# 实验 1: CAE 网格搜索 (k=1-50)
# ============================================================================

def run_cae_experiments(config: ExperimentConfig, metadata: MetaData) -> List[ExperimentResult]:
    """CAE 基线实验"""
    print("\n" + "="*60)
    print("实验 1: CAE 网格搜索 (k=1-50)")
    print("="*60)
    
    results = []
    k_values = list(range(1, 51))
    
    pbar = tqdm(total=len(k_values) * len(config.seeds), desc="CAE")
    
    for k in k_values:
        all_accs = []
        
        for seed in config.seeds:
            set_seed(seed)
            encoder = ConcreteEncoder(
                input_dim=metadata.feature_dim,
                output_dim=k,
                total_epochs=config.epochs
            )
            model = EncoderModel(encoder, config.hidden_dim, metadata.cls_num)
            acc, train_time = train_and_evaluate(model, metadata, config)
            all_accs.append(acc)
            pbar.update(1)
        
        result = ExperimentResult(
            experiment_id=f"CAE_k{k}",
            algorithm="CAE",
            category="baseline_encoder",
            hyperparameters={"k": k},
            accuracy=np.mean(all_accs),
            accuracy_std=np.std(all_accs),
            sparsity=1.0 - k / metadata.feature_dim,
            num_selected=k,
            num_params=count_parameters(model),
            train_time=train_time
        )
        results.append(result)
        
        if k % 10 == 0:
            print(f"  k={k}: acc={np.mean(all_accs):.4f}±{np.std(all_accs):.4f}")
    
    pbar.close()
    return results


# ============================================================================
# 实验 2: IPCAE 网格搜索 (k=1-50)
# ============================================================================

def run_ipcae_experiments(config: ExperimentConfig, metadata: MetaData) -> List[ExperimentResult]:
    """IPCAE 基线实验"""
    print("\n" + "="*60)
    print("实验 2: IPCAE 网格搜索 (k=1-50)")
    print("="*60)
    
    results = []
    k_values = list(range(1, 51))
    
    pbar = tqdm(total=len(k_values) * len(config.seeds), desc="IPCAE")
    
    for k in k_values:
        all_accs = []
        
        for seed in config.seeds:
            set_seed(seed)
            encoder = IndirectConcreteEncoder(
                input_dim=metadata.feature_dim,
                output_dim=k,
                embedding_dim=32,
                total_epochs=config.epochs
            )
            model = EncoderModel(encoder, config.hidden_dim, metadata.cls_num)
            acc, train_time = train_and_evaluate(model, metadata, config)
            all_accs.append(acc)
            pbar.update(1)
        
        result = ExperimentResult(
            experiment_id=f"IPCAE_k{k}",
            algorithm="IPCAE",
            category="baseline_encoder",
            hyperparameters={"k": k, "embedding_dim": 32},
            accuracy=np.mean(all_accs),
            accuracy_std=np.std(all_accs),
            sparsity=1.0 - k / metadata.feature_dim,
            num_selected=k,
            num_params=count_parameters(model),
            train_time=train_time
        )
        results.append(result)
        
        if k % 10 == 0:
            print(f"  k={k}: acc={np.mean(all_accs):.4f}±{np.std(all_accs):.4f}")
    
    pbar.close()
    return results


# ============================================================================
# 实验 3: 门控方法基线
# ============================================================================

def run_gate_experiments(config: ExperimentConfig, metadata: MetaData) -> List[ExperimentResult]:
    """门控方法基线实验"""
    print("\n" + "="*60)
    print("实验 3: 门控方法基线")
    print("="*60)
    
    results = []
    gates = [
        ("StochasticGate", lambda dim: StochasticGate(input_dim=dim, sigma=0.5)),
        ("GumbelSigmoidGate", lambda dim: GumbelSigmoidGate(input_dim=dim)),
        ("HardConcreteGate", lambda dim: HardConcreteGate(input_dim=dim)),
    ]
    lambdas = [0.001, 0.01, 0.1, 1, 10]
    
    pbar = tqdm(total=len(gates) * len(lambdas) * len(config.seeds), desc="Gates")
    
    for gate_name, gate_fn in gates:
        for lam in lambdas:
            all_accs = []
            all_selected = []
            
            for seed in config.seeds:
                set_seed(seed)
                gate = gate_fn(metadata.feature_dim)
                model = GateModel(gate, metadata.feature_dim, config.hidden_dim, metadata.cls_num)
                acc, train_time = train_and_evaluate(
                    model, metadata, config, lambda_sparsity=lam, use_sparsity=True
                )
                all_accs.append(acc)
                all_selected.append(gate.num_selected)
                pbar.update(1)
            
            result = ExperimentResult(
                experiment_id=f"{gate_name}_lambda{lam}",
                algorithm=gate_name,
                category="baseline_gate",
                hyperparameters={"lambda": lam},
                accuracy=np.mean(all_accs),
                accuracy_std=np.std(all_accs),
                sparsity=1.0 - np.mean(all_selected) / metadata.feature_dim,
                num_selected=int(np.mean(all_selected)),
                num_params=count_parameters(model),
                train_time=train_time
            )
            results.append(result)
        
        print(f"  {gate_name} 完成")
    
    pbar.close()
    return results


# ============================================================================
# 实验 4: GSG-IPCAE (我们的方法)
# ============================================================================

def run_gsg_ipcae_experiments(config: ExperimentConfig, metadata: MetaData) -> List[ExperimentResult]:
    """GSG-IPCAE 实验 (我们的方法)"""
    print("\n" + "="*60)
    print("实验 4: GSG-IPCAE (我们的方法)")
    print("="*60)
    
    results = []
    
    # 主要配置
    k_max = 100
    gate_emb = 16
    enc_emb = 32
    lam = 0.1
    
    all_accs = []
    all_selected = []
    
    pbar = tqdm(total=len(config.seeds), desc="GSG-IPCAE")
    
    for seed in config.seeds:
        set_seed(seed)
        
        gate = GumbelSoftmaxGate(input_dim=k_max, embedding_dim=gate_emb)
        encoder = IndirectConcreteEncoder(
            input_dim=metadata.feature_dim,
            output_dim=k_max,
            embedding_dim=enc_emb,
            total_epochs=config.epochs
        )
        selector = GateEncoderSelector(gate, encoder)
        model = GSGIPCAEModel(selector, config.hidden_dim, metadata.cls_num)
        
        acc, train_time = train_and_evaluate(
            model, metadata, config, lambda_sparsity=lam, use_sparsity=True
        )
        all_accs.append(acc)
        all_selected.append(gate.num_selected)
        pbar.update(1)
    
    pbar.close()
    
    result = ExperimentResult(
        experiment_id="GSG-IPCAE",
        algorithm="GSG-IPCAE",
        category="ours",
        hyperparameters={"k_max": k_max, "lambda": lam, "gate_emb": gate_emb, "enc_emb": enc_emb},
        accuracy=np.mean(all_accs),
        accuracy_std=np.std(all_accs),
        sparsity=1.0 - np.mean(all_selected) / metadata.feature_dim,
        num_selected=int(np.mean(all_selected)),
        num_params=count_parameters(model),
        train_time=train_time
    )
    results.append(result)
    
    print(f"  GSG-IPCAE: acc={np.mean(all_accs):.4f}±{np.std(all_accs):.4f}, selected={int(np.mean(all_selected))}")
    
    return results


# ============================================================================
# 实验 5: 消融实验
# ============================================================================

def run_ablation_experiments(config: ExperimentConfig, metadata: MetaData) -> List[ExperimentResult]:
    """消融实验"""
    print("\n" + "="*60)
    print("实验 5: 消融实验")
    print("="*60)
    
    results = []
    
    # A1: k_max 影响
    print("\n  A1: k_max 影响")
    k_max_values = [50, 100, 200, 500]
    k_max_values = [k for k in k_max_values if k <= metadata.feature_dim]
    
    for k_max in k_max_values:
        all_accs = []
        all_selected = []
        
        for seed in config.seeds:
            set_seed(seed)
            gate = GumbelSoftmaxGate(input_dim=k_max, embedding_dim=16)
            encoder = IndirectConcreteEncoder(metadata.feature_dim, k_max, 32, total_epochs=config.epochs)
            selector = GateEncoderSelector(gate, encoder)
            model = GSGIPCAEModel(selector, config.hidden_dim, metadata.cls_num)
            acc, _ = train_and_evaluate(model, metadata, config, 0.1, True)
            all_accs.append(acc)
            all_selected.append(gate.num_selected)
        
        results.append(ExperimentResult(
            experiment_id=f"ablation_kmax{k_max}",
            algorithm="GSG-IPCAE",
            category="ablation_kmax",
            hyperparameters={"k_max": k_max, "lambda": 0.1},
            accuracy=np.mean(all_accs),
            accuracy_std=np.std(all_accs),
            sparsity=1.0 - np.mean(all_selected) / metadata.feature_dim,
            num_selected=int(np.mean(all_selected)),
            num_params=count_parameters(model),
            train_time=0
        ))
        print(f"    k_max={k_max}: acc={np.mean(all_accs):.4f}")
    
    # A2: lambda 影响
    print("\n  A2: lambda 影响")
    lambdas = [0.01, 0.1, 1, 10]
    k_max = min(100, metadata.feature_dim)
    
    for lam in lambdas:
        all_accs = []
        all_selected = []
        
        for seed in config.seeds:
            set_seed(seed)
            gate = GumbelSoftmaxGate(input_dim=k_max, embedding_dim=16)
            encoder = IndirectConcreteEncoder(metadata.feature_dim, k_max, 32, total_epochs=config.epochs)
            selector = GateEncoderSelector(gate, encoder)
            model = GSGIPCAEModel(selector, config.hidden_dim, metadata.cls_num)
            acc, _ = train_and_evaluate(model, metadata, config, lam, True)
            all_accs.append(acc)
            all_selected.append(gate.num_selected)
        
        results.append(ExperimentResult(
            experiment_id=f"ablation_lambda{lam}",
            algorithm="GSG-IPCAE",
            category="ablation_lambda",
            hyperparameters={"k_max": k_max, "lambda": lam},
            accuracy=np.mean(all_accs),
            accuracy_std=np.std(all_accs),
            sparsity=1.0 - np.mean(all_selected) / metadata.feature_dim,
            num_selected=int(np.mean(all_selected)),
            num_params=count_parameters(model),
            train_time=0
        ))
        print(f"    lambda={lam}: acc={np.mean(all_accs):.4f}")
    
    return results


# ============================================================================
# 结果保存
# ============================================================================

def save_results(results: List[ExperimentResult], filepath: Path) -> None:
    """保存结果到 CSV"""
    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(filepath, index=False)
    print(f"结果已保存: {filepath}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="DeepFS 实验")
    parser.add_argument("--dataset", type=str, default="pancreas", help="数据集名称")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024], help="随机种子")
    parser.add_argument("--quick", action="store_true", help="快速测试模式")
    args = parser.parse_args()
    
    # 快速测试模式
    if args.quick:
        args.epochs = 5
        args.seeds = [42]
    
    print("="*60)
    print("DeepFS 完整实验")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # 配置
    config = ExperimentConfig(
        dataset_name=args.dataset,
        epochs=args.epochs,
        seeds=args.seeds,
    )
    
    print(f"\n配置:")
    print(f"  数据集: {config.dataset_name}")
    print(f"  训练轮数: {config.epochs}")
    print(f"  随机种子: {config.seeds}")
    print(f"  设备: {config.device}")
    
    # 加载数据
    print("\n加载数据...")
    metadata = generate_train_test_loader(
        name=config.dataset_name,
        test_size=0.2,
        batch_size=config.batch_size,
        shuffle=True,
        device=config.device,
        random_state=config.seeds[0],
    )
    metadata.generate_val_loader()
    print(f"  特征数: {metadata.feature_dim}")
    print(f"  类别数: {metadata.cls_num}")
    
    # 创建结果目录
    result_dir = Path(config.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # 实验 1: CAE
    cae_results = run_cae_experiments(config, metadata)
    save_results(cae_results, result_dir / "cae_results.csv")
    all_results.extend(cae_results)
    
    # 实验 2: IPCAE
    ipcae_results = run_ipcae_experiments(config, metadata)
    save_results(ipcae_results, result_dir / "ipcae_results.csv")
    all_results.extend(ipcae_results)
    
    # 实验 3: 门控方法
    gate_results = run_gate_experiments(config, metadata)
    save_results(gate_results, result_dir / "gate_results.csv")
    all_results.extend(gate_results)
    
    # 实验 4: GSG-IPCAE
    gsg_ipcae_results = run_gsg_ipcae_experiments(config, metadata)
    save_results(gsg_ipcae_results, result_dir / "gsg_ipcae_results.csv")
    all_results.extend(gsg_ipcae_results)
    
    # 实验 5: 消融实验
    ablation_results = run_ablation_experiments(config, metadata)
    save_results(ablation_results, result_dir / "ablation_results.csv")
    all_results.extend(ablation_results)
    
    # 保存汇总结果
    save_results(all_results, result_dir / "all_results.csv")
    
    # 汇总
    print("\n" + "="*60)
    print("实验完成汇总")
    print("="*60)
    print(f"  CAE 实验: {len(cae_results)} 组")
    print(f"  IPCAE 实验: {len(ipcae_results)} 组")
    print(f"  门控实验: {len(gate_results)} 组")
    print(f"  GSG-IPCAE 实验: {len(gsg_ipcae_results)} 组")
    print(f"  消融实验: {len(ablation_results)} 组")
    print(f"  总计: {len(all_results)} 组")
    print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)


if __name__ == "__main__":
    main()
