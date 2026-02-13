"""
快速验证实验脚本 - 基于新架构

测试所有算法组合以验证代码正常工作
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

# 新架构导入
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
    TemperatureCallback,
    LoggingCallback,
)

from deepfs.core import SparsityLoss, SelectionResult
from deepfs.training.trainer import TrainConfig


def seed_all(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


class MLPClassifier(torch.nn.Module):
    """Simple MLP classifier for testing."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


class FeatureSelectionModel(torch.nn.Module):
    """Model combining feature selector with classifier."""
    
    def __init__(
        self,
        selector: GateEncoderSelector,
        classifier: MLPClassifier
    ):
        super().__init__()
        self.selector = selector
        self.classifier = classifier
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.selector(x)
        return self.classifier(x)


def create_synthetic_data(
    n_samples: int = 1000,
    n_features: int = 100,
    n_informative: int = 10,
    n_classes: int = 3,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic classification data."""
    np.random.seed(seed)
    
    # Create informative features
    informative = np.random.randn(n_samples, n_informative)
    
    # Create labels from informative features
    labels = np.argmax(informative[:, :n_classes-1], axis=1)
    
    # Create noise features
    noise = np.random.randn(n_samples, n_features - n_informative)
    
    # Combine features
    X = np.hstack([informative, noise])
    
    # Shuffle features
    perm = np.random.permutation(n_features)
    X = X[:, perm]
    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def run_gate_experiment(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    gate_class,
    gate_name: str,
    input_dim: int,
    output_dim: int,
    epochs: int = 10,
    seed: int = 42,
) -> Optional[float]:
    """Run a single gate experiment."""
    print(f"\n  [{gate_name}]")
    
    try:
        # Create gate
        if gate_name == "StochasticGate":
            gate = StochasticGate(input_dim=input_dim, sigma=0.5)
        elif gate_name == "GumbelSigmoidGate":
            gate = GumbelSigmoidGate(input_dim=input_dim)
        elif gate_name == "GumbelSoftmaxGate":
            gate = GumbelSoftmaxGate(input_dim=input_dim, embedding_dim=16)
        elif gate_name == "HardConcreteGate":
            gate = HardConcreteGate(input_dim=input_dim)
        else:
            raise ValueError(f"Unknown gate: {gate_name}")
        
        # Create model (gate only, no encoder)
        classifier = MLPClassifier(input_dim=input_dim, hidden_dim=64, output_dim=output_dim)
        
        # Simple model that just applies gate
        class GateOnlyModel(torch.nn.Module):
            def __init__(self, gate, classifier):
                super().__init__()
                self.gate = gate
                self.classifier = classifier
                self._gate_probs = None
            
            def forward(self, x):
                x = self.gate(x)
                self._gate_probs = self.gate._gate_probs
                return self.classifier(x)
            
            def sparsity_loss(self):
                return self.gate.sparsity_loss()
        
        model = GateOnlyModel(gate, classifier)
        
        # Train
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        X_train_dev = X_train.to(device)
        y_train_dev = y_train.to(device)
        X_val_dev = X_val.to(device)
        y_val_dev = y_val.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_dev)
            loss = criterion(outputs, y_train_dev)
            sparsity = model.sparsity_loss().total
            total_loss = loss + 0.1 * sparsity
            total_loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            outputs = model(X_val_dev)
            preds = outputs.argmax(dim=1)
            acc = (preds == y_val_dev).float().mean().item()
        
        print(f"    准确率: {acc:.4f}")
        return acc
    
    except Exception as e:
        print(f"    错误: {e}")
        return None


def run_encoder_experiment(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    encoder_class,
    encoder_name: str,
    input_dim: int,
    output_dim: int,
    k: int,
    epochs: int = 10,
    seed: int = 42,
) -> Optional[float]:
    """Run a single encoder experiment."""
    print(f"\n  [{encoder_name}]")
    
    try:
        # Create encoder
        if encoder_name == "ConcreteEncoder":
            encoder = ConcreteEncoder(
                input_dim=input_dim,
                output_dim=k,
                total_epochs=epochs,
            )
        elif encoder_name == "IndirectConcreteEncoder":
            encoder = IndirectConcreteEncoder(
                input_dim=input_dim,
                output_dim=k,
                embedding_dim=32,
                total_epochs=epochs,
            )
        else:
            raise ValueError(f"Unknown encoder: {encoder_name}")
        
        # Create model (encoder only)
        classifier = MLPClassifier(input_dim=k, hidden_dim=64, output_dim=output_dim)
        
        class EncoderOnlyModel(torch.nn.Module):
            def __init__(self, encoder, classifier):
                super().__init__()
                self.encoder = encoder
                self.classifier = classifier
            
            def forward(self, x):
                return self.classifier(self.encoder(x))
            
            def update_temperature(self, epoch):
                self.encoder.update_temperature(epoch)
        
        model = EncoderOnlyModel(encoder, classifier)
        
        # Train
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        X_train_dev = X_train.to(device)
        y_train_dev = y_train.to(device)
        X_val_dev = X_val.to(device)
        y_val_dev = y_val.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_dev)
            loss = criterion(outputs, y_train_dev)
            loss.backward()
            optimizer.step()
            model.update_temperature(epoch)
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            outputs = model(X_val_dev)
            preds = outputs.argmax(dim=1)
            acc = (preds == y_val_dev).float().mean().item()
        
        print(f"    准确率: {acc:.4f}")
        return acc
    
    except Exception as e:
        print(f"    错误: {e}")
        return None


def run_gate_encoder_experiment(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    gate_name: str,
    encoder_name: str,
    input_dim: int,
    output_dim: int,
    k: int,
    epochs: int = 10,
    seed: int = 42,
) -> Optional[float]:
    """Run a gate + encoder combined experiment."""
    combo_name = f"{gate_name} + {encoder_name}"
    print(f"\n  [{combo_name}]")
    
    try:
        # Create encoder
        if encoder_name == "ConcreteEncoder":
            encoder = ConcreteEncoder(
                input_dim=input_dim,
                output_dim=k,
                total_epochs=epochs,
            )
        elif encoder_name == "IndirectConcreteEncoder":
            encoder = IndirectConcreteEncoder(
                input_dim=input_dim,
                output_dim=k,
                embedding_dim=32,
                total_epochs=epochs,
            )
        else:
            raise ValueError(f"Unknown encoder: {encoder_name}")
        
        # Create gate (input_dim = k, the encoder output)
        if gate_name == "StochasticGate":
            gate = StochasticGate(input_dim=k, sigma=0.5)
        elif gate_name == "GumbelSigmoidGate":
            gate = GumbelSigmoidGate(input_dim=k)
        elif gate_name == "GumbelSoftmaxGate":
            gate = GumbelSoftmaxGate(input_dim=k, embedding_dim=16)
        elif gate_name == "HardConcreteGate":
            gate = HardConcreteGate(input_dim=k)
        else:
            raise ValueError(f"Unknown gate: {gate_name}")
        
        # Create selector
        selector = GateEncoderSelector(gate, encoder)
        
        # Create model
        classifier = MLPClassifier(input_dim=k, hidden_dim=64, output_dim=output_dim)
        model = FeatureSelectionModel(selector, classifier)
        
        # Train
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        X_train_dev = X_train.to(device)
        y_train_dev = y_train.to(device)
        X_val_dev = X_val.to(device)
        y_val_dev = y_val.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_dev)
            loss = criterion(outputs, y_train_dev)
            sparsity = selector.sparsity_loss().total
            total_loss = loss + 0.1 * sparsity
            total_loss.backward()
            optimizer.step()
            selector.update_temperature(epoch)
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            outputs = model(X_val_dev)
            preds = outputs.argmax(dim=1)
            acc = (preds == y_val_dev).float().mean().item()
        
        print(f"    准确率: {acc:.4f}")
        return acc
    
    except Exception as e:
        print(f"    错误: {e}")
        return None


def main():
    """主函数"""
    print("=" * 60)
    print("DeepFS 新架构快速验证实验")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 配置
    EPOCHS = 10
    SEED = 42
    N_SAMPLES = 1000
    N_FEATURES = 100
    K = 20  # 选择特征数
    N_CLASSES = 3
    
    seed_all(SEED)
    
    # 创建数据
    print(f"\n创建合成数据: {N_SAMPLES} 样本, {N_FEATURES} 特征, {N_CLASSES} 类别")
    X, y = create_synthetic_data(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=15,
        n_classes=N_CLASSES,
        seed=SEED
    )
    
    # 分割数据
    n_train = int(0.8 * N_SAMPLES)
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    
    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}")
    
    # 运行实验
    all_results = {}
    
    # 1. 纯门控实验
    print("\n" + "=" * 60)
    print("纯门控实验")
    print("=" * 60)
    
    gate_names = ["StochasticGate", "GumbelSigmoidGate", "GumbelSoftmaxGate", "HardConcreteGate"]
    for gate_name in gate_names:
        acc = run_gate_experiment(
            X_train, y_train, X_val, y_val,
            gate_class=None,  # 使用名称创建
            gate_name=gate_name,
            input_dim=N_FEATURES,
            output_dim=N_CLASSES,
            epochs=EPOCHS,
            seed=SEED
        )
        all_results[f"Gate:{gate_name}"] = acc
    
    # 2. 纯编码器实验
    print("\n" + "=" * 60)
    print("纯编码器实验")
    print("=" * 60)
    
    encoder_names = ["ConcreteEncoder", "IndirectConcreteEncoder"]
    for encoder_name in encoder_names:
        acc = run_encoder_experiment(
            X_train, y_train, X_val, y_val,
            encoder_class=None,  # 使用名称创建
            encoder_name=encoder_name,
            input_dim=N_FEATURES,
            output_dim=N_CLASSES,
            k=K,
            epochs=EPOCHS,
            seed=SEED
        )
        all_results[f"Encoder:{encoder_name}"] = acc
    
    # 3. 门控+编码器组合实验
    print("\n" + "=" * 60)
    print("门控+编码器组合实验")
    print("=" * 60)
    
    for gate_name in gate_names:
        for encoder_name in encoder_names:
            acc = run_gate_encoder_experiment(
                X_train, y_train, X_val, y_val,
                gate_name=gate_name,
                encoder_name=encoder_name,
                input_dim=N_FEATURES,
                output_dim=N_CLASSES,
                k=K,
                epochs=EPOCHS,
                seed=SEED
            )
            all_results[f"{gate_name}+{encoder_name}"] = acc
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)
    
    print("\n算法 | 准确率")
    print("-" * 50)
    for algo, acc in all_results.items():
        if acc is not None:
            print(f"{algo}: {acc:.4f}")
        else:
            print(f"{algo}: 失败")
    
    # 统计
    success_count = sum(1 for v in all_results.values() if v is not None)
    total_count = len(all_results)
    print(f"\n成功: {success_count}/{total_count}")
    
    # 保存结果
    os.makedirs("exp/results", exist_ok=True)
    results_df = pd.DataFrame([
        {'algorithm': k, 'accuracy': v}
        for k, v in all_results.items()
    ])
    results_df.to_csv("exp/results/quick_test_results.csv", index=False)
    print(f"\n结果已保存到: exp/results/quick_test_results.csv")
    
    print("\n" + "=" * 60)
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()