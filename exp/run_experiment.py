"""
DeepFS 完整实验脚本

实现对比实验和消融实验，支持：
- 纯门控实验 (24组)
- 纯编码实验 (100组)  
- 门控+编码实验 (288组)
- 消融实验 (22组)
"""
from __future__ import annotations

import os
import sys
import json
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
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
    data_dir: str = "exp/data"
    
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
    """单次实验结果"""
    experiment_id: str
    algorithm: str
    category: str
    hyperparameters: Dict[str, Any]
    task_performance: float
    sparsity: float
    num_params: int
    train_time: float
    seed: int
    num_selected: int
    stability: float = 0.0  # 多次运行后计算


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


class GateOnlyModel(nn.Module):
    """纯门控模型"""
    
    def __init__(self, gate: nn.Module, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.gate = gate
        self.classifier = MLPClassifier(input_dim, hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gate(x)
        return self.classifier(x)
    
    def sparsity_loss(self) -> torch.Tensor:
        return self.gate.sparsity_loss().total
    
    def update_temperature(self, epoch: int) -> None:
        if hasattr(self.gate, 'update_temperature'):
            self.gate.update_temperature(epoch)


class EncoderOnlyModel(nn.Module):
    """纯编码器模型"""
    
    def __init__(self, encoder: nn.Module, output_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = MLPClassifier(encoder.output_dim, hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))
    
    def update_temperature(self, epoch: int) -> None:
        self.encoder.update_temperature(epoch)


class GateEncoderModel(nn.Module):
    """门控+编码器模型"""
    
    def __init__(self, selector: GateEncoderSelector, output_dim: int, hidden_dim: int):
        super().__init__()
        self.selector = selector
        self.classifier = MLPClassifier(selector.output_dim, hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.selector(x))
    
    def sparsity_loss(self) -> torch.Tensor:
        return self.selector.sparsity_loss().total
    
    def update_temperature(self, epoch: int) -> None:
        self.selector.update_temperature(epoch)


# ============================================================================
# 工具函数
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed: int) -> None:
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def create_gate(gate_name: str, input_dim: int, embedding_dim: int = 16) -> nn.Module:
    """创建门控模块"""
    if gate_name == "StochasticGate":
        return StochasticGate(input_dim=input_dim, sigma=0.5)
    elif gate_name == "GumbelSigmoidGate":
        return GumbelSigmoidGate(input_dim=input_dim)
    elif gate_name == "GumbelSoftmaxGate":
        return GumbelSoftmaxGate(input_dim=input_dim, embedding_dim=embedding_dim)
    elif gate_name == "HardConcreteGate":
        return HardConcreteGate(input_dim=input_dim)
    else:
        raise ValueError(f"Unknown gate: {gate_name}")


def create_encoder(encoder_name: str, input_dim: int, output_dim: int, 
                   embedding_dim: int = 32, total_epochs: int = 100) -> nn.Module:
    """创建编码器模块"""
    if encoder_name == "ConcreteEncoder":
        return ConcreteEncoder(input_dim=input_dim, output_dim=output_dim, total_epochs=total_epochs)
    elif encoder_name == "IndirectConcreteEncoder":
        return IndirectConcreteEncoder(
            input_dim=input_dim, output_dim=output_dim, 
            embedding_dim=embedding_dim, total_epochs=total_epochs
        )
    else:
        raise ValueError(f"Unknown encoder: {encoder_name}")


def get_num_selected(model: nn.Module, model_type: str) -> int:
    """获取选择的特征数量"""
    if model_type == "gate":
        return model.gate.num_selected
    elif model_type == "encoder":
        return model.encoder.output_dim
    elif model_type == "gate_encoder":
        return model.selector.num_selected
    return 0


# ============================================================================
# 数据加载
# ============================================================================

def load_data(config: ExperimentConfig) -> MetaData:
    """加载数据集"""
    metadata = generate_train_test_loader(
        name=config.dataset_name,
        test_size=0.2,
        batch_size=config.batch_size,
        shuffle=True,
        device=config.device,
        random_state=config.seeds[0],
    )
    metadata.generate_val_loader()
    return metadata


# ============================================================================
# 训练函数
# ============================================================================

def train_epoch(model: nn.Module, train_loader, optimizer, criterion, 
                device: str, lambda_sparsity: float = 0.1, model_type: str = "gate") -> Tuple[float, float]:
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
        
        # 添加稀疏损失
        if model_type in ["gate", "gate_encoder"]:
            sparsity_loss = model.sparsity_loss()
            loss = loss + lambda_sparsity * sparsity_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        _, predicted = outputs.max(1)
        total_correct += predicted.eq(y).sum().item()
        total_samples += x.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def evaluate(model: nn.Module, loader, criterion, device: str) -> Tuple[float, float]:
    """评估模型"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in loader:
            x = batch.X.to(device)
            y = batch.obs['cell_type'].to(device)
            
            outputs = model(x)
            loss = criterion(outputs, y)
            
            total_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(y).sum().item()
            total_samples += x.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def evaluate_tensor(model: nn.Module, loader, criterion, device: str) -> Tuple[float, float]:
    """评估模型 (Tensor 数据)"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            outputs = model(x)
            loss = criterion(outputs, y)
            
            total_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(y).sum().item()
            total_samples += x.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


# ============================================================================
# 实验运行函数
# ============================================================================

def run_single_experiment(
    model: nn.Module,
    metadata: MetaData,
    config: ExperimentConfig,
    lambda_sparsity: float,
    model_type: str,
    seed: int,
) -> Tuple[float, float, int, float, int]:
    """运行单次实验
    
    Returns:
        accuracy, sparsity, num_params, train_time, num_selected
    """
    set_seed(seed)
    
    # 重新创建模型以确保参数初始化一致
    model = model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # 计算参数量
    num_params = count_parameters(model)
    
    # 训练
    start_time = time.time()
    best_val_acc = 0.0
    
    for epoch in range(config.epochs):
        # 更新温度
        model.update_temperature(epoch)
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, metadata.train_loader, optimizer, criterion,
            config.device, lambda_sparsity, model_type
        )
        
        # 验证
        val_loss, val_acc = evaluate_tensor(
            model, metadata.val_loader, criterion, config.device
        )
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    train_time = time.time() - start_time
    
    # 测试
    test_loss, test_acc = evaluate(model, metadata.test_loader, criterion, config.device)
    
    # 计算稀疏性
    num_selected = get_num_selected(model, model_type)
    sparsity = 1.0 - num_selected / metadata.feature_dim
    
    return test_acc, sparsity, num_params, train_time, num_selected


# ============================================================================
# 对比实验
# ============================================================================

def run_gate_only_experiments(config: ExperimentConfig, metadata: MetaData) -> List[ExperimentResult]:
    """运行纯门控实验 (24组)"""
    print("\n" + "="*60)
    print("纯门控实验 (Gate-Only)")
    print("="*60)
    
    results = []
    gates = ["StochasticGate", "GumbelSigmoidGate", "GumbelSoftmaxGate", "HardConcreteGate"]
    lambdas = [0.001, 0.01, 0.1, 1, 10, 100]
    
    total = len(gates) * len(lambdas) * len(config.seeds)
    pbar = tqdm(total=total, desc="Gate-Only")
    
    for gate_name in gates:
        for lam in lambdas:
            exp_id = f"G_{gate_name}_lambda{lam}"
            
            all_accs = []
            all_sparities = []
            all_times = []
            all_selected = []
            
            for seed in config.seeds:
                set_seed(seed)
                
                # 创建模型
                gate = create_gate(gate_name, metadata.feature_dim, embedding_dim=16)
                model = GateOnlyModel(gate, metadata.feature_dim, config.hidden_dim, metadata.cls_num)
                
                acc, sparsity, num_params, train_time, num_selected = run_single_experiment(
                    model, metadata, config, lam, "gate", seed
                )
                
                all_accs.append(acc)
                all_sparities.append(sparsity)
                all_times.append(train_time)
                all_selected.append(num_selected)
                
                pbar.update(1)
            
            # 计算稳定性和均值
            mean_acc = np.mean(all_accs)
            std_acc = np.std(all_accs)
            stability = std_acc / mean_acc if mean_acc > 0 else 0
            
            result = ExperimentResult(
                experiment_id=exp_id,
                algorithm=gate_name,
                category="gate_only",
                hyperparameters={"lambda": lam},
                task_performance=mean_acc,
                sparsity=np.mean(all_sparities),
                num_params=num_params,
                train_time=np.mean(all_times),
                seed=0,  # 已聚合
                num_selected=int(np.mean(all_selected)),
                stability=stability
            )
            results.append(result)
            
            print(f"  {exp_id}: acc={mean_acc:.4f}±{std_acc:.4f}, sparsity={np.mean(all_sparities):.4f}")
    
    pbar.close()
    return results


def run_encoder_only_experiments(config: ExperimentConfig, metadata: MetaData) -> List[ExperimentResult]:
    """运行纯编码器实验 (100组)"""
    print("\n" + "="*60)
    print("纯编码器实验 (Encoder-Only)")
    print("="*60)
    
    results = []
    encoders = ["ConcreteEncoder", "IndirectConcreteEncoder"]
    k_values = list(range(1, 51))  # k = 1 to 50
    
    total = len(encoders) * len(k_values) * len(config.seeds)
    pbar = tqdm(total=total, desc="Encoder-Only")
    
    for encoder_name in encoders:
        for k in k_values:
            exp_id = f"E_{encoder_name}_k{k}"
            
            all_accs = []
            all_sparities = []
            all_times = []
            
            for seed in config.seeds:
                set_seed(seed)
                
                # 创建模型
                encoder = create_encoder(encoder_name, metadata.feature_dim, k, 
                                        embedding_dim=32, total_epochs=config.epochs)
                model = EncoderOnlyModel(encoder, metadata.cls_num, config.hidden_dim)
                
                acc, sparsity, num_params, train_time, num_selected = run_single_experiment(
                    model, metadata, config, 0.0, "encoder", seed
                )
                
                all_accs.append(acc)
                all_sparities.append(sparsity)
                all_times.append(train_time)
                
                pbar.update(1)
            
            mean_acc = np.mean(all_accs)
            std_acc = np.std(all_accs)
            stability = std_acc / mean_acc if mean_acc > 0 else 0
            
            result = ExperimentResult(
                experiment_id=exp_id,
                algorithm=encoder_name,
                category="encoder_only",
                hyperparameters={"k": k},
                task_performance=mean_acc,
                sparsity=np.mean(all_sparities),
                num_params=num_params,
                train_time=np.mean(all_times),
                seed=0,
                num_selected=k,
                stability=stability
            )
            results.append(result)
            
            if k % 10 == 0:
                print(f"  {encoder_name} k={k}: acc={mean_acc:.4f}±{std_acc:.4f}")
    
    pbar.close()
    return results


def run_gate_encoder_experiments(config: ExperimentConfig, metadata: MetaData) -> List[ExperimentResult]:
    """运行门控+编码器实验 (288组)"""
    print("\n" + "="*60)
    print("门控+编码器实验 (Gate+Encoder)")
    print("="*60)
    
    results = []
    gates = ["StochasticGate", "GumbelSigmoidGate", "GumbelSoftmaxGate", "HardConcreteGate"]
    encoders = ["ConcreteEncoder", "IndirectConcreteEncoder"]
    k_max_values = [10, 50, 100, 200, 500, 1000]
    lambdas = [0.001, 0.01, 0.1, 1, 10, 100]
    
    # 限制 k_max 不超过特征数
    k_max_values = [k for k in k_max_values if k <= metadata.feature_dim]
    
    total = len(gates) * len(encoders) * len(k_max_values) * len(lambdas) * len(config.seeds)
    pbar = tqdm(total=total, desc="Gate+Encoder")
    
    for gate_name in gates:
        for encoder_name in encoders:
            for k_max in k_max_values:
                for lam in lambdas:
                    exp_id = f"GE_{gate_name}_{encoder_name}_k{k_max}_lambda{lam}"
                    
                    all_accs = []
                    all_sparities = []
                    all_times = []
                    all_selected = []
                    
                    for seed in config.seeds:
                        set_seed(seed)
                        
                        # 创建模型
                        gate = create_gate(gate_name, k_max, embedding_dim=64)
                        encoder = create_encoder(encoder_name, metadata.feature_dim, k_max,
                                               embedding_dim=64, total_epochs=config.epochs)
                        selector = GateEncoderSelector(gate, encoder)
                        model = GateEncoderModel(selector, metadata.cls_num, config.hidden_dim)
                        
                        acc, sparsity, num_params, train_time, num_selected = run_single_experiment(
                            model, metadata, config, lam, "gate_encoder", seed
                        )
                        
                        all_accs.append(acc)
                        all_sparities.append(sparsity)
                        all_times.append(train_time)
                        all_selected.append(num_selected)
                        
                        pbar.update(1)
                    
                    mean_acc = np.mean(all_accs)
                    std_acc = np.std(all_accs)
                    stability = std_acc / mean_acc if mean_acc > 0 else 0
                    
                    result = ExperimentResult(
                        experiment_id=exp_id,
                        algorithm=f"{gate_name}+{encoder_name}",
                        category="gate_encoder",
                        hyperparameters={"k_max": k_max, "lambda": lam},
                        task_performance=mean_acc,
                        sparsity=np.mean(all_sparities),
                        num_params=num_params,
                        train_time=np.mean(all_times),
                        seed=0,
                        num_selected=int(np.mean(all_selected)),
                        stability=stability
                    )
                    results.append(result)
    
    pbar.close()
    return results


# ============================================================================
# 消融实验
# ============================================================================

def run_ablation_experiments(config: ExperimentConfig, metadata: MetaData) -> List[ExperimentResult]:
    """运行消融实验 (22组)"""
    print("\n" + "="*60)
    print("消融实验 (Ablation Study)")
    print("="*60)
    
    results = []
    
    # A1: k_max 影响
    print("\nA1: k_max 影响分析")
    k_max_values = [10, 50, 100, 200, 500, 1000]
    k_max_values = [k for k in k_max_values if k <= metadata.feature_dim]
    
    for k_max in k_max_values:
        exp_id = f"A1_kmax{k_max}"
        all_accs = []
        all_sparities = []
        all_times = []
        
        for seed in config.seeds:
            set_seed(seed)
            gate = GumbelSoftmaxGate(input_dim=k_max, embedding_dim=64)
            encoder = IndirectConcreteEncoder(metadata.feature_dim, k_max, 64, total_epochs=config.epochs)
            selector = GateEncoderSelector(gate, encoder)
            model = GateEncoderModel(selector, metadata.cls_num, config.hidden_dim)
            
            acc, sparsity, num_params, train_time, num_selected = run_single_experiment(
                model, metadata, config, 0.1, "gate_encoder", seed
            )
            
            all_accs.append(acc)
            all_sparities.append(sparsity)
            all_times.append(train_time)
        
        mean_acc = np.mean(all_accs)
        std_acc = np.std(all_accs)
        
        result = ExperimentResult(
            experiment_id=exp_id,
            algorithm="GumbelSoftmaxGate+IndirectConcreteEncoder",
            category="ablation_k_max",
            hyperparameters={"k_max": k_max, "lambda": 0.1, "gate_emb": 64, "enc_emb": 64},
            task_performance=mean_acc,
            sparsity=np.mean(all_sparities),
            num_params=num_params,
            train_time=np.mean(all_times),
            seed=0,
            num_selected=k_max,
            stability=std_acc / mean_acc if mean_acc > 0 else 0
        )
        results.append(result)
        print(f"  k_max={k_max}: acc={mean_acc:.4f}±{std_acc:.4f}")
    
    # A2: lambda 影响
    print("\nA2: lambda 影响分析")
    lambdas = [0.001, 0.01, 0.1, 1, 10, 100]
    k_max = min(100, metadata.feature_dim)
    
    for lam in lambdas:
        exp_id = f"A2_lambda{lam}"
        all_accs = []
        all_sparities = []
        all_times = []
        
        for seed in config.seeds:
            set_seed(seed)
            gate = GumbelSoftmaxGate(input_dim=k_max, embedding_dim=64)
            encoder = IndirectConcreteEncoder(metadata.feature_dim, k_max, 64, total_epochs=config.epochs)
            selector = GateEncoderSelector(gate, encoder)
            model = GateEncoderModel(selector, metadata.cls_num, config.hidden_dim)
            
            acc, sparsity, num_params, train_time, num_selected = run_single_experiment(
                model, metadata, config, lam, "gate_encoder", seed
            )
            
            all_accs.append(acc)
            all_sparities.append(sparsity)
            all_times.append(train_time)
        
        mean_acc = np.mean(all_accs)
        std_acc = np.std(all_accs)
        
        result = ExperimentResult(
            experiment_id=exp_id,
            algorithm="GumbelSoftmaxGate+IndirectConcreteEncoder",
            category="ablation_lambda",
            hyperparameters={"k_max": k_max, "lambda": lam, "gate_emb": 64, "enc_emb": 64},
            task_performance=mean_acc,
            sparsity=np.mean(all_sparities),
            num_params=num_params,
            train_time=np.mean(all_times),
            seed=0,
            num_selected=k_max,
            stability=std_acc / mean_acc if mean_acc > 0 else 0
        )
        results.append(result)
        print(f"  lambda={lam}: acc={mean_acc:.4f}±{std_acc:.4f}")
    
    # A3: gate_embedding_dim 影响
    print("\nA3: gate_embedding_dim 影响分析")
    gate_emb_values = [4, 64, 128, 256, 1024]
    
    for gate_emb in gate_emb_values:
        exp_id = f"A3_gate_emb{gate_emb}"
        all_accs = []
        all_sparities = []
        all_times = []
        
        for seed in config.seeds:
            set_seed(seed)
            gate = GumbelSoftmaxGate(input_dim=k_max, embedding_dim=gate_emb)
            encoder = IndirectConcreteEncoder(metadata.feature_dim, k_max, 64, total_epochs=config.epochs)
            selector = GateEncoderSelector(gate, encoder)
            model = GateEncoderModel(selector, metadata.cls_num, config.hidden_dim)
            
            acc, sparsity, num_params, train_time, num_selected = run_single_experiment(
                model, metadata, config, 0.1, "gate_encoder", seed
            )
            
            all_accs.append(acc)
            all_sparities.append(sparsity)
            all_times.append(train_time)
        
        mean_acc = np.mean(all_accs)
        std_acc = np.std(all_accs)
        
        result = ExperimentResult(
            experiment_id=exp_id,
            algorithm="GumbelSoftmaxGate+IndirectConcreteEncoder",
            category="ablation_gate_emb",
            hyperparameters={"k_max": k_max, "lambda": 0.1, "gate_emb": gate_emb, "enc_emb": 64},
            task_performance=mean_acc,
            sparsity=np.mean(all_sparities),
            num_params=num_params,
            train_time=np.mean(all_times),
            seed=0,
            num_selected=k_max,
            stability=std_acc / mean_acc if mean_acc > 0 else 0
        )
        results.append(result)
        print(f"  gate_emb={gate_emb}: acc={mean_acc:.4f}±{std_acc:.4f}")
    
    # A4: encoder_embedding_dim 影响
    print("\nA4: encoder_embedding_dim 影响分析")
    enc_emb_values = [4, 64, 128, 256, 1024]
    
    for enc_emb in enc_emb_values:
        exp_id = f"A4_enc_emb{enc_emb}"
        all_accs = []
        all_sparities = []
        all_times = []
        
        for seed in config.seeds:
            set_seed(seed)
            gate = GumbelSoftmaxGate(input_dim=k_max, embedding_dim=64)
            encoder = IndirectConcreteEncoder(metadata.feature_dim, k_max, enc_emb, total_epochs=config.epochs)
            selector = GateEncoderSelector(gate, encoder)
            model = GateEncoderModel(selector, metadata.cls_num, config.hidden_dim)
            
            acc, sparsity, num_params, train_time, num_selected = run_single_experiment(
                model, metadata, config, 0.1, "gate_encoder", seed
            )
            
            all_accs.append(acc)
            all_sparities.append(sparsity)
            all_times.append(train_time)
        
        mean_acc = np.mean(all_accs)
        std_acc = np.std(all_accs)
        
        result = ExperimentResult(
            experiment_id=exp_id,
            algorithm="GumbelSoftmaxGate+IndirectConcreteEncoder",
            category="ablation_enc_emb",
            hyperparameters={"k_max": k_max, "lambda": 0.1, "gate_emb": 64, "enc_emb": enc_emb},
            task_performance=mean_acc,
            sparsity=np.mean(all_sparities),
            num_params=num_params,
            train_time=np.mean(all_times),
            seed=0,
            num_selected=k_max,
            stability=std_acc / mean_acc if mean_acc > 0 else 0
        )
        results.append(result)
        print(f"  enc_emb={enc_emb}: acc={mean_acc:.4f}±{std_acc:.4f}")
    
    return results


# ============================================================================
# 结果保存
# ============================================================================

def save_results(results: List[ExperimentResult], filepath: str) -> None:
    """保存结果到 CSV"""
    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(filepath, index=False)
    print(f"结果已保存到: {filepath}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print("="*60)
    print("DeepFS 完整实验")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # 配置
    config = ExperimentConfig(
        dataset_name="pancreas",  # 可切换为 Lung, Spleen, Tongue
        epochs=5,  # 快速测试用 5 个 epoch
        seeds=[42],  # 单次运行快速测试
    )
    
    print(f"\n配置:")
    print(f"  数据集: {config.dataset_name}")
    print(f"  训练轮数: {config.epochs}")
    print(f"  随机种子: {config.seeds}")
    print(f"  设备: {config.device}")
    
    # 加载数据
    print("\n加载数据...")
    metadata = load_data(config)
    print(f"  特征数: {metadata.feature_dim}")
    print(f"  类别数: {metadata.cls_num}")
    
    # 创建结果目录
    contrast_dir = Path(config.result_dir) / "contrast"
    ablation_dir = Path(config.result_dir) / "ablation"
    contrast_dir.mkdir(parents=True, exist_ok=True)
    ablation_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # 1. 纯门控实验
    gate_results = run_gate_only_experiments(config, metadata)
    save_results(gate_results, contrast_dir / "gate_only.csv")
    all_results["gate_only"] = gate_results
    
    # 2. 纯编码器实验
    encoder_results = run_encoder_only_experiments(config, metadata)
    save_results(encoder_results, contrast_dir / "encoder_only.csv")
    all_results["encoder_only"] = encoder_results
    
    # 3. 门控+编码器实验
    ge_results = run_gate_encoder_experiments(config, metadata)
    save_results(ge_results, contrast_dir / "gate_encoder.csv")
    all_results["gate_encoder"] = ge_results
    
    # 4. 消融实验
    ablation_results = run_ablation_experiments(config, metadata)
    save_results(ablation_results, ablation_dir / "ablation.csv")
    all_results["ablation"] = ablation_results
    
    # 汇总
    print("\n" + "="*60)
    print("实验完成汇总")
    print("="*60)
    print(f"  纯门控实验: {len(gate_results)} 组")
    print(f"  纯编码器实验: {len(encoder_results)} 组")
    print(f"  门控+编码器实验: {len(ge_results)} 组")
    print(f"  消融实验: {len(ablation_results)} 组")
    print(f"  总计: {len(gate_results) + len(encoder_results) + len(ge_results) + len(ablation_results)} 组")
    
    print("\n" + "="*60)
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)


if __name__ == "__main__":
    main()