"""Track feature quality during training to understand convergence dynamics.

For GSG-Softmax+IPCAE, save features at epoch checkpoints and evaluate
with a fixed MLP to see how feature quality evolves over training.

Usage:
  PYTHONPATH=. python exp/run_feature_evolution.py
"""
from __future__ import annotations

import os
import warnings
warnings.filterwarnings("ignore")

import anndata as ad
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from deepfs import GumbelSoftmaxGateIndirectConcreteModel
from exp.utils import MLPClassifier, seed_all

DATASET = "SklearnDigitsCls"
INPUT_DIM = 64
N_CLASSES = 10
K = 30
SW = 0.01
EPOCHS = 300
CKPT_EPOCHS = [1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 300]
SEEDS = [0, 1, 2]


def _eval_feature_set(features, X_tr, X_te, y_tr, y_te, seed=42):
    feats = sorted(set(f for f in features if 0 <= f < INPUT_DIM))
    if len(feats) == 0:
        return 0.0, 0
    seed_all(seed)
    xtr = torch.tensor(X_tr[:, feats], dtype=torch.float32)
    xte = torch.tensor(X_te[:, feats], dtype=torch.float32)
    ytr = torch.tensor(y_tr, dtype=torch.long)
    yte = torch.tensor(y_te, dtype=torch.long)

    head = MLPClassifier(len(feats), 128, N_CLASSES, "cpu")
    opt = torch.optim.Adam(head.parameters(), lr=2e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    for _ in range(200):
        head.train()
        logits = head(xtr)
        loss = loss_fn(logits, ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()

    head.eval()
    with torch.no_grad():
        acc = (head(xte).argmax(1) == yte).float().mean().item()
    return acc, len(feats)


def _run_single_seed(seed, X_np, y_np):
    seed_all(seed)
    X_tr_np, X_te_np, y_tr_np, y_te_np = train_test_split(
        X_np, y_np, test_size=0.2, random_state=seed, stratify=y_np)

    X_tr = torch.tensor(X_tr_np, dtype=torch.float32)
    y_tr = torch.tensor(y_tr_np, dtype=torch.long)
    X_te = torch.tensor(X_te_np, dtype=torch.float32)
    y_te = torch.tensor(y_te_np, dtype=torch.long)

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=128, shuffle=True)

    model = GumbelSoftmaxGateIndirectConcreteModel(
        input_dim=INPUT_DIM, k=K,
        embedding_dim_encoder=16, embedding_dim_gate=16,
        initial_temperature=10.0, final_temperature=0.01,
        total_epochs=EPOCHS,
    )
    head = MLPClassifier(K, 128, N_CLASSES, "cpu")
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(head.parameters()), lr=2e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    results = []
    ckpt_set = set(CKPT_EPOCHS)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        head.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            output = model(xb)
            logits = head(output)
            task_loss = loss_fn(logits, yb)
            sparse_loss = sum(model.sparsity_loss().values)
            total_loss = task_loss + SW * sparse_loss
            total_loss.backward()
            optimizer.step()

        if epoch in ckpt_set:
            model.eval()
            with torch.no_grad():
                output = model(X_te)
                logits = head(output)
                joint_acc = (logits.argmax(1) == y_te).float().mean().item()

            sel = model.get_selection_result()
            valid_feats = sorted(set(
                f for f in sel.selected_indices if 0 <= f < INPUT_DIM))
            mlp_acc, n_feat = _eval_feature_set(
                valid_feats, X_tr_np, X_te_np, y_tr_np, y_te_np,
                seed=seed + 1000)

            results.append({
                "epoch": epoch,
                "joint_acc": joint_acc,
                "mlp_acc": mlp_acc,
                "n_features": n_feat,
                "features": valid_feats,
                "temperature": model.temperature.item(),
            })
            print(f"    epoch={epoch:3d}  joint={joint_acc:.4f}  "
                  f"mlp={mlp_acc:.4f}  feat={n_feat:2d}  "
                  f"temp={results[-1]['temperature']:.3f}  "
                  f"feats={valid_feats}")

    return results


def main():
    adata = ad.read_h5ad(f"exp/data/{DATASET}.h5ad")
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)
    y = adata.obs["cell_type"].cat.codes.values

    all_results = []
    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")
        results = _run_single_seed(seed, X, y)
        for r in results:
            r["seed"] = seed
        all_results.extend(results)

    df = pd.DataFrame(all_results)

    print("\n" + "=" * 70)
    print("  FEATURE EVOLUTION SUMMARY")
    print("=" * 70)
    for epoch in sorted(df["epoch"].unique()):
        sub = df[df["epoch"] == epoch]
        ja = sub["joint_acc"].mean()
        ma = sub["mlp_acc"].mean()
        nf = sub["n_features"].mean()
        temp = sub["temperature"].mean()

        feat_sets = [frozenset(r["features"]) for _, r in sub.iterrows()]
        inter = len(set().intersection(*feat_sets)) if feat_sets else 0
        union = len(set().union(*feat_sets)) if feat_sets else 0
        jacc = inter / union if union > 0 else 0

        print(f"  ep={epoch:3d}  joint={ja:.4f}  mlp={ma:.4f}  "
              f"feat={nf:.0f}  temp={temp:.3f}  "
              f"cross_seed: u={union} i={inter} j={jacc:.3f}")

    print(f"\n  Feature quality improvement:")
    early = df[df["epoch"] <= 30]["mlp_acc"].mean()
    mid = df[(df["epoch"] > 30) & (df["epoch"] <= 100)]["mlp_acc"].mean()
    late = df[df["epoch"] >= 200]["mlp_acc"].mean()
    print(f"    Early (<=30 ep):  mlp={early:.4f}")
    print(f"    Mid  (30-100 ep): mlp={mid:.4f}")
    print(f"    Late (>=200 ep):  mlp={late:.4f}")
    print(f"    Total gain: {late - early:+.4f}")

    print(f"\n  Convergence pattern:")
    for epoch in [1, 10, 30, 100, 300]:
        sub = df[df["epoch"] == epoch]
        print(f"    ep={epoch:3d}: n_feat={sub['n_features'].mean():.0f}  "
              f"joint={sub['joint_acc'].mean():.4f}  "
              f"mlp={sub['mlp_acc'].mean():.4f}")


if __name__ == "__main__":
    main()
