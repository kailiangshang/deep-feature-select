from __future__ import annotations

import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, r2_score

from deepfs import (
    ConcreteAutoencoderModel,
    GumbelSoftmaxGateConcreteModel,
    GumbelSoftmaxGateIndirectConcreteModel,
    GumbelSoftmaxGateModel,
    IndirectConcreteAutoencoderModel,
    StochasticGateModel,
)
from exp.data import generate_train_test_loader
from exp.trainers import EncoderTrainer, GateTrainer, GateEncoderTrainer
from exp.utils import MLPClassifier, MLPRegressor, AutoencoderHead, seed_all

EPOCHS = 20
LR = 1e-3
DEVICE = "cpu"
HIDDEN = 64


def test_classification_encoder():
    print("\n" + "=" * 60)
    print("TEST 1: CAE Encoder + Classification")
    print("=" * 60)
    data = generate_train_test_loader(
        name="ToyCls", batch_size=64, device=DEVICE, random_state=0,
    )
    k = 50
    model = ConcreteAutoencoderModel(
        input_dim=data.feature_dim, output_dim=k, total_epochs=EPOCHS, device=DEVICE,
    )
    head = MLPClassifier(k, HIDDEN, data.cls_num, DEVICE)
    trainer = EncoderTrainer(model, head, task="classification", lr=LR, device=DEVICE, seed=0)
    df = trainer.fit(data.train_loader, EPOCHS, data.test_loader)
    final_acc = df["accuracy"].iloc[-1]
    print(f"  Final accuracy: {final_acc:.4f}")
    assert len(df) == EPOCHS
    assert "accuracy" in df.columns
    print("  PASS")


def test_classification_gate():
    print("\n" + "=" * 60)
    print("TEST 2: STG Gate + Classification")
    print("=" * 60)
    data = generate_train_test_loader(
        name="ToyCls", batch_size=64, device=DEVICE, random_state=0,
    )
    model = StochasticGateModel(input_dim=data.feature_dim, sigma=0.5, device=DEVICE)
    head = MLPClassifier(data.feature_dim, HIDDEN, data.cls_num, DEVICE)
    trainer = GateTrainer(
        model, head, task="classification",
        sparse_loss_weight=1.0, lr=LR, device=DEVICE, seed=0,
    )
    df = trainer.fit(data.train_loader, EPOCHS, data.test_loader)
    final_acc = df["accuracy"].iloc[-1]
    num_selected = df["num_selected"].iloc[-1]
    print(f"  Final accuracy: {final_acc:.4f}, Features selected: {num_selected}")
    assert "loss_sparsity" in df.columns
    print("  PASS")


def test_classification_combined():
    print("\n" + "=" * 60)
    print("TEST 3: GSG-Softmax+IPCAE + Classification (core model)")
    print("=" * 60)
    data = generate_train_test_loader(
        name="ToyCls", batch_size=64, device=DEVICE, random_state=0,
    )
    k = 30
    model = GumbelSoftmaxGateIndirectConcreteModel(
        input_dim=data.feature_dim, k=k,
        embedding_dim_encoder=16, embedding_dim_gate=8,
        total_epochs=EPOCHS, device=DEVICE,
    )
    head = MLPClassifier(k, HIDDEN, data.cls_num, DEVICE)
    trainer = GateEncoderTrainer(
        model, head, task="classification",
        sparse_loss_weight=1.0, lr=LR, device=DEVICE, seed=0,
    )
    result_df, feature_df = trainer.fit(data.train_loader, EPOCHS, data.test_loader)
    final_acc = result_df["accuracy"].iloc[-1]
    num_selected = result_df["num_selected"].iloc[-1]
    print(f"  Final accuracy: {final_acc:.4f}, Features: {num_selected}/{k}")
    assert feature_df.shape[1] == k
    assert "gate_open_ratio" in result_df.columns
    print("  PASS")


def test_regression_encoder():
    print("\n" + "=" * 60)
    print("TEST 4: IPCAE Encoder + Regression")
    print("=" * 60)
    data = generate_train_test_loader(
        name="ToyReg", batch_size=64, device=DEVICE, random_state=0,
    )
    k = 30
    model = IndirectConcreteAutoencoderModel(
        input_dim=data.feature_dim, output_dim=k, embedding_dim=16,
        total_epochs=EPOCHS, device=DEVICE,
    )
    head = MLPRegressor(k, HIDDEN, 1, DEVICE)
    trainer = EncoderTrainer(model, head, task="regression", lr=LR, device=DEVICE, seed=0)
    df = trainer.fit(data.train_loader, EPOCHS, data.test_loader)
    final_r2 = df["r2"].iloc[-1]
    print(f"  Final R²: {final_r2:.4f}")
    assert "r2" in df.columns
    assert "loss_task" in df.columns
    print("  PASS")


def test_regression_gate():
    print("\n" + "=" * 60)
    print("TEST 5: GSG-Softmax Gate + Regression")
    print("=" * 60)
    data = generate_train_test_loader(
        name="ToyReg", batch_size=64, device=DEVICE, random_state=0,
    )
    model = GumbelSoftmaxGateModel(
        input_dim=data.feature_dim, embedding_dim=8,
        total_epochs=EPOCHS, device=DEVICE,
    )
    head = MLPRegressor(data.feature_dim, HIDDEN, 1, DEVICE)
    trainer = GateTrainer(
        model, head, task="regression",
        sparse_loss_weight=0.1, lr=LR, device=DEVICE, seed=0,
    )
    df = trainer.fit(data.train_loader, EPOCHS, data.test_loader)
    final_r2 = df["r2"].iloc[-1]
    print(f"  Final R²: {final_r2:.4f}")
    assert "r2" in df.columns
    assert "loss_sparsity" in df.columns
    print("  PASS")


def test_regression_combined():
    print("\n" + "=" * 60)
    print("TEST 6: GSG-Softmax+CAE + Regression")
    print("=" * 60)
    data = generate_train_test_loader(
        name="ToyReg", batch_size=64, device=DEVICE, random_state=0,
    )
    k = 20
    model = GumbelSoftmaxGateConcreteModel(
        input_dim=data.feature_dim, k=k, embedding_dim_gate=8,
        total_epochs=EPOCHS, device=DEVICE,
    )
    head = MLPRegressor(k, HIDDEN, 1, DEVICE)
    trainer = GateEncoderTrainer(
        model, head, task="regression",
        sparse_loss_weight=0.5, lr=LR, device=DEVICE, seed=0,
    )
    result_df, feature_df = trainer.fit(data.train_loader, EPOCHS, data.test_loader)
    final_r2 = result_df["r2"].iloc[-1]
    print(f"  Final R²: {final_r2:.4f}, Features: {result_df['num_selected'].iloc[-1]}/{k}")
    assert "r2" in result_df.columns
    assert "loss_sparsity" in result_df.columns
    print("  PASS")


def test_reconstruction_encoder():
    print("\n" + "=" * 60)
    print("TEST 7: CAE Encoder + Reconstruction (Autoencoder)")
    print("=" * 60)
    data = generate_train_test_loader(
        name="ToyAE", batch_size=64, device=DEVICE, random_state=0,
    )
    k = 50
    model = ConcreteAutoencoderModel(
        input_dim=data.feature_dim, output_dim=k, total_epochs=EPOCHS, device=DEVICE,
    )
    head = AutoencoderHead(k, HIDDEN, data.feature_dim, DEVICE)
    trainer = EncoderTrainer(
        model, head, task="reconstruction",
        lr=LR, device=DEVICE, seed=0, input_dim=data.feature_dim,
    )
    df = trainer.fit(data.train_loader, EPOCHS, data.test_loader)
    final_mse = df["neg_mse"].iloc[-1]
    print(f"  Final neg_MSE: {final_mse:.4f}")
    assert "neg_mse" in df.columns
    print("  PASS")


def test_reconstruction_combined():
    print("\n" + "=" * 60)
    print("TEST 8: GSG-Softmax+IPCAE + Reconstruction (core model)")
    print("=" * 60)
    data = generate_train_test_loader(
        name="ToyAE", batch_size=64, device=DEVICE, random_state=0,
    )
    k = 30
    model = GumbelSoftmaxGateIndirectConcreteModel(
        input_dim=data.feature_dim, k=k,
        embedding_dim_encoder=16, embedding_dim_gate=8,
        total_epochs=EPOCHS, device=DEVICE,
    )
    head = AutoencoderHead(k, HIDDEN, data.feature_dim, DEVICE)
    trainer = GateEncoderTrainer(
        model, head, task="reconstruction",
        sparse_loss_weight=0.1, lr=LR, device=DEVICE, seed=0,
        input_dim=data.feature_dim,
    )
    result_df, feature_df = trainer.fit(data.train_loader, EPOCHS, data.test_loader)
    final_mse = result_df["neg_mse"].iloc[-1]
    print(f"  Final neg_MSE: {final_mse:.4f}, Features: {result_df['num_selected'].iloc[-1]}/{k}")
    assert "neg_mse" in result_df.columns
    assert "gate_open_ratio" in result_df.columns
    print("  PASS")


def main():
    tests = [
        ("Classification", [
            test_classification_encoder,
            test_classification_gate,
            test_classification_combined,
        ]),
        ("Regression", [
            test_regression_encoder,
            test_regression_gate,
            test_regression_combined,
        ]),
        ("Reconstruction", [
            test_reconstruction_encoder,
            test_reconstruction_combined,
        ]),
    ]

    passed = 0
    failed = 0
    for group_name, group_tests in tests:
        print(f"\n{'#' * 60}")
        print(f"# {group_name}")
        print(f"{'#' * 60}")
        for test_fn in group_tests:
            try:
                test_fn()
                passed += 1
            except Exception as e:
                print(f"  FAIL: {e}")
                import traceback
                traceback.print_exc()
                failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'=' * 60}")
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
