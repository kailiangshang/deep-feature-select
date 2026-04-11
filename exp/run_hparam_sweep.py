"""Quick hyperparameter sweep for GSG-Softmax+IPCAE on classification."""
from __future__ import annotations

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import torch
from deepfs import GumbelSoftmaxGateIndirectConcreteModel
from exp.data import generate_train_test_loader
from exp.trainers import GateEncoderTrainer
from exp.utils import MLPClassifier, seed_all

DEVICE = "cpu"

def run_one(cls_data, k, enc_dim, gate_dim, sw, lr, epochs, hidden, name):
    seed_all(0)
    m = GumbelSoftmaxGateIndirectConcreteModel(
        cls_data.feature_dim, k, enc_dim, gate_dim,
        total_epochs=epochs, device=DEVICE)
    h = MLPClassifier(k, hidden, cls_data.cls_num, DEVICE)
    trainer = GateEncoderTrainer(m, h, task="classification",
                                 sparse_loss_weight=sw, lr=lr, device=DEVICE, seed=0)
    result_df, _ = trainer.fit(cls_data.train_loader, epochs, cls_data.test_loader)
    last = result_df.iloc[-1]
    acc = last["accuracy"]
    feat = int(last["num_selected"])
    print(f"  {name:55s}  acc={acc:.4f}  features={feat:3d}")
    return {"name": name, "accuracy": acc, "features": feat,
            "k": k, "enc_dim": enc_dim, "gate_dim": gate_dim,
            "sw": sw, "lr": lr, "epochs": epochs, "hidden": hidden}


def main():
    cls_data = generate_train_test_loader("SklearnDigitsCls", batch_size=128, device=DEVICE, random_state=0)

    configs = []

    # ── sparse_loss_weight sweep ──
    print("\n=== sparse_loss_weight (k=20, enc=16, gate=8, lr=1e-3, 200ep) ===")
    for sw in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
        configs.append((20, 16, 8, sw, 1e-3, 200, 64, f"sw={sw}"))

    # ── k sweep ──
    print("\n=== k sweep (enc=16, gate=8, sw=0.1, lr=1e-3, 200ep) ===")
    for k in [10, 15, 20, 30, 40]:
        configs.append((k, 16, 8, 0.1, 1e-3, 200, 64, f"k={k}"))

    # ── embedding dim sweep ──
    print("\n=== embedding_dim sweep (k=20, sw=0.1, lr=1e-3, 200ep) ===")
    for enc_dim, gate_dim in [(8, 4), (16, 8), (32, 16), (64, 32)]:
        configs.append((20, enc_dim, gate_dim, 0.1, 1e-3, 200, 64,
                        f"enc={enc_dim}_gate={gate_dim}"))

    # ── lr sweep ──
    print("\n=== learning rate (k=20, enc=16, gate=8, sw=0.1, 200ep) ===")
    for lr in [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]:
        configs.append((20, 16, 8, 0.1, lr, 200, 64, f"lr={lr}"))

    # ── hidden dim sweep ──
    print("\n=== hidden dim (k=20, enc=16, gate=8, sw=0.1, lr=1e-3, 200ep) ===")
    for hidden in [32, 64, 128, 256]:
        configs.append((20, 16, 8, 0.1, 1e-3, 200, hidden, f"hidden={hidden}"))

    # ── longer training ──
    print("\n=== longer training (k=20, enc=16, gate=8, sw=0.1, lr=1e-3) ===")
    for epochs in [200, 500]:
        configs.append((20, 16, 8, 0.1, 1e-3, epochs, 64, f"epochs={epochs}"))

    results = []
    for k, enc_dim, gate_dim, sw, lr, epochs, hidden, label in configs:
        name = label
        row = run_one(cls_data, k, enc_dim, gate_dim, sw, lr, epochs, hidden, name)
        results.append(row)

    df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    df.to_csv("exp/results/sklearn_demo/hparam_sweep.csv", index=False)

    print("\n" + "=" * 80)
    print("TOP 10 by accuracy:")
    print("=" * 80)
    for _, row in df.head(10).iterrows():
        print(f"  acc={row['accuracy']:.4f}  feat={row['features']:3.0f}  "
              f"k={row['k']}  enc={row['enc_dim']}  gate={row['gate_dim']}  "
              f"sw={row['sw']}  lr={row['lr']}  ep={row['epochs']}  hid={row['hidden']}")
    print("\nBest efficiency (acc / features used):")
    print("-" * 80)
    df["efficiency"] = df["accuracy"] / df["features"].clip(lower=1)
    for _, row in df.sort_values("efficiency", ascending=False).head(5).iterrows():
        print(f"  acc={row['accuracy']:.4f}  feat={row['features']:3.0f}  eff={row['efficiency']:.4f}  "
              f"k={row['k']}  sw={row['sw']}  enc={row['enc_dim']}_gate={row['gate_dim']}  "
              f"lr={row['lr']}  ep={row['epochs']}")


if __name__ == "__main__":
    main()
