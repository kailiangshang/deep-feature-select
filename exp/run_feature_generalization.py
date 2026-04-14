"""Evaluate generalization of selected feature subsets.

1. K-fold CV: fixed features, varying train/test splits
2. Cross-task: cls features -> reconstruction, rec features -> classification
3. Cross-model: features from model A evaluated by model B's head
4. Comparison: selected vs full vs random vs LASSO vs IPCAE-grid

Usage:
  PYTHONPATH=. python exp/run_feature_generalization.py
"""
from __future__ import annotations

import os
import warnings
warnings.filterwarnings("ignore")

import json
import anndata as ad
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import TensorDataset, DataLoader

from exp.utils import MLPClassifier, seed_all

INPUT_DIM = 64
N_CLASSES = 10


def _load_data(name):
    adata = ad.read_h5ad(f"exp/data/{name}.h5ad")
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)
    y = adata.obs["cell_type"].cat.codes.values
    return X, y


def _eval_mlp(X, y, features, n_splits=5, epochs=200, seed=42):
    """K-fold CV with a fixed feature subset using MLP."""
    feats = sorted(set(f for f in features if 0 <= f < X.shape[1]))
    if len(feats) == 0:
        return {"acc_mean": 0, "acc_std": 0, "n_features": 0}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for train_idx, test_idx in skf.split(X, y):
        seed_all(seed)
        xtr = torch.tensor(X[train_idx][:, feats], dtype=torch.float32)
        ytr = torch.tensor(y[train_idx], dtype=torch.long)
        xte = torch.tensor(X[test_idx][:, feats], dtype=torch.float32)
        yte = torch.tensor(y[test_idx], dtype=torch.long)

        head = MLPClassifier(len(feats), 128, N_CLASSES, "cpu")
        opt = torch.optim.Adam(head.parameters(), lr=2e-3)
        loss_fn = torch.nn.CrossEntropyLoss()
        for _ in range(epochs):
            head.train()
            loss_fn(head(xtr), ytr).backward()
            opt.step()
            opt.zero_grad()
        head.eval()
        with torch.no_grad():
            acc = (head(xte).argmax(1) == yte).float().mean().item()
        accs.append(acc)
    return {"acc_mean": np.mean(accs), "acc_std": np.std(accs),
            "n_features": len(feats)}


def _collect_features(results_dirs, models):
    records = []
    for rdir in results_dirs:
        fdir = os.path.join(rdir, "per_group")
        if not os.path.exists(fdir):
            continue
        for meta_file in sorted(os.listdir(fdir)):
            if not meta_file.endswith("_meta.json"):
                continue
            with open(os.path.join(fdir, meta_file)) as f:
                meta = json.load(f)
            if meta["model"] not in models:
                continue
            if meta.get("dataset") not in ("SklearnDigitsCls",):
                continue
            feat_path = os.path.join(fdir, meta_file.replace("_meta.json", "_features.csv"))
            if not os.path.exists(feat_path):
                continue
            feat_df = pd.read_csv(feat_path)
            raw = feat_df.iloc[0].dropna().astype(int).tolist()
            selected = sorted(set(f for f in raw if 0 <= f < INPUT_DIM))
            if len(selected) == 0:
                continue
            records.append({
                "model": meta["model"],
                "seed": meta["seed"],
                "k": meta.get("k"),
                "sw": meta.get("sparse_loss_weight"),
                "features": selected,
                "n_features": len(selected),
            })
    return pd.DataFrame(records)


def main():
    X, y = _load_data("SklearnDigitsCls")

    # Collect features from all gate models (k=30, sw=0.01 only)
    feat_df = _collect_features(
        ["exp/results/cls_grid", "exp/results/cls_sw_ext", "exp/results/cls_digits"],
        ["GSG-Softmax+IPCAE", "HCG+IPCAE", "GSG-Softmax+CAE", "HCG+CAE"],
    )
    feat_df = feat_df[(feat_df["k"] == 30) & (feat_df["sw"] == 0.01)]

    # Also get IPCAE grid features for k=30
    ipcae_feats = []
    fdir = "exp/results/cls_grid/per_group"
    for meta_file in sorted(os.listdir(fdir)):
        if not meta_file.endswith("_meta.json") or "IPCAE" not in meta_file:
            continue
        with open(os.path.join(fdir, meta_file)) as f:
            meta = json.load(f)
        if meta.get("k") != 30:
            continue
        feat_path = os.path.join(fdir, meta_file.replace("_meta.json", "_features.csv"))
        if not os.path.exists(feat_path):
            continue
        feat_df_raw = pd.read_csv(feat_path)
        raw = feat_df_raw.iloc[0].dropna().astype(int).tolist()
        selected = sorted(set(f for f in raw if 0 <= f < INPUT_DIM))
        if len(selected) > 0:
            ipcae_feats.append(selected)

    # LASSO features
    seed_all(42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    lasso = Lasso(alpha=0.01, random_state=42, max_iter=5000)
    lasso.fit(X_tr, y_tr)
    lasso_feats_20 = sorted(np.where(np.abs(lasso.coef_) > 0.01)[0].tolist())[:20]
    lasso_feats_30 = sorted(np.where(np.abs(lasso.coef_) > 0.001)[0].tolist())[:30]

    # Random features baseline
    rng = np.random.RandomState(42)
    random_20 = sorted(rng.choice(INPUT_DIM, 20, replace=False).tolist())

    # ================================================================
    # EXP 1: 5-fold CV generalization
    # ================================================================
    print("=" * 70)
    print("  EXP 1: 5-FOLD CV GENERALIZATION")
    print("  Fixed features, varying train/test splits")
    print("=" * 70)

    test_sets = {}

    # Full features baseline
    r = _eval_mlp(X, y, list(range(INPUT_DIM)))
    test_sets["ALL 64 features"] = r
    print(f"  ALL 64 features:      {r['acc_mean']:.4f} +/- {r['acc_std']:.4f}")

    # Random baseline
    r = _eval_mlp(X, y, random_20)
    test_sets["Random 20"] = r
    print(f"  Random 20 features:   {r['acc_mean']:.4f} +/- {r['acc_std']:.4f}")

    # LASSO
    r = _eval_mlp(X, y, lasso_feats_20)
    test_sets["LASSO 20"] = r
    print(f"  LASSO 20 features:    {r['acc_mean']:.4f} +/- {r['acc_std']:.4f}")

    if len(lasso_feats_30) > 0:
        r = _eval_mlp(X, y, lasso_feats_30)
        test_sets["LASSO 30"] = r
        print(f"  LASSO 30 features:    {r['acc_mean']:.4f} +/- {r['acc_std']:.4f}")

    # IPCAE grid (average across seeds)
    if ipcae_feats:
        ipcae_accs = [_eval_mlp(X, y, f)["acc_mean"] for f in ipcae_feats]
        for i, f in enumerate(ipcae_feats):
            r = _eval_mlp(X, y, f)
            test_sets[f"IPCAE grid seed{i}"] = r
            print(f"  IPCAE grid seed{i} ({len(f):2d} feat): {r['acc_mean']:.4f} +/- {r['acc_std']:.4f}")
        print(f"  IPCAE grid mean:      {np.mean(ipcae_accs):.4f}")

    # Gate models
    print()
    for _, row in feat_df.sort_values(["model", "seed"]).iterrows():
        label = f"{row['model']} seed={row['seed']}"
        r = _eval_mlp(X, y, row["features"])
        test_sets[label] = r
        print(f"  {label:35s} ({row['n_features']:2d} feat): "
              f"{r['acc_mean']:.4f} +/- {r['acc_std']:.4f}")

    # ================================================================
    # EXP 2: Cross-seed feature generalization
    # ================================================================
    print("\n" + "=" * 70)
    print("  EXP 2: CROSS-SEED GENERALIZATION")
    print("  Train on seed A split, test on seed B split (same features)")
    print("=" * 70)

    ours = feat_df[feat_df["model"] == "GSG-Softmax+IPCAE"]
    if len(ours) >= 2:
        for i, (_, ri) in enumerate(ours.iterrows()):
            for j, (_, rj) in enumerate(ours.iterrows()):
                if i >= j:
                    continue
                # Train on seed A's split, eval on seed B's split
                feats = ri["features"]
                seed_all(ri["seed"])
                X_tr_a, X_te_a, y_tr_a, y_te_a = train_test_split(
                    X[:, feats], y, test_size=0.2, random_state=int(ri["seed"]), stratify=y)
                seed_all(rj["seed"])
                X_tr_b, X_te_b, y_tr_b, y_te_b = train_test_split(
                    X[:, feats], y, test_size=0.2, random_state=int(rj["seed"]), stratify=y)

                xtr = torch.tensor(X_tr_a, dtype=torch.float32)
                ytr = torch.tensor(y_tr_a, dtype=torch.long)
                head = MLPClassifier(len(feats), 128, N_CLASSES, "cpu")
                opt = torch.optim.Adam(head.parameters(), lr=2e-3)
                loss_fn = torch.nn.CrossEntropyLoss()
                for _ in range(200):
                    head.train()
                    loss_fn(head(xtr), ytr).backward()
                    opt.step()
                    opt.zero_grad()

                # Test on both splits
                head.eval()
                with torch.no_grad():
                    acc_a = (head(torch.tensor(X_te_a, dtype=torch.float32)).argmax(1)
                             == torch.tensor(y_te_a, dtype=torch.long)).float().mean().item()
                    acc_b = (head(torch.tensor(X_te_b, dtype=torch.float32)).argmax(1)
                             == torch.tensor(y_te_b, dtype=torch.long)).float().mean().item()
                drop = acc_a - acc_b
                print(f"  seed {int(ri['seed'])} features, "
                      f"train=split{int(ri['seed'])}: acc_A={acc_a:.4f}  acc_B={acc_b:.4f}  drop={drop:+.4f}")

    # ================================================================
    # EXP 3: Feature stability = generalization?
    # ================================================================
    print("\n" + "=" * 70)
    print("  EXP 3: FEATURE OVERLAP vs GENERALIZATION GAP")
    print("  Do more overlapping features generalize better?")
    print("=" * 70)

    for model in ["GSG-Softmax+IPCAE", "HCG+IPCAE"]:
        sub = feat_df[feat_df["model"] == model]
        if len(sub) < 2:
            continue
        feat_sets = [frozenset(r["features"]) for _, r in sub.iterrows()]
        union = set().union(*feat_sets)
        freq = {}
        for fs in feat_sets:
            for f in fs:
                freq[f] = freq.get(f, 0) + 1

        for min_freq in [1, 2, len(sub)]:
            feats = sorted(f for f, c in freq.items() if c >= min_freq)
            if len(feats) == 0:
                continue
            r = _eval_mlp(X, y, feats)
            print(f"  {model:25s} freq>={min_freq}: {len(feats):2d} feats -> "
                  f"{r['acc_mean']:.4f} +/- {r['acc_std']:.4f}")

    # ================================================================
    # EXP 4: Sklearn baselines on same feature subsets
    # ================================================================
    print("\n" + "=" * 70)
    print("  EXP 4: SKLEARN BASELINES ON SELECTED FEATURES")
    print("  LogisticRegression / RF on the same feature subsets")
    print("=" * 70)

    for _, row in feat_df.sort_values(["model", "seed"]).iterrows():
        feats = row["features"]
        label = f"{row['model']} seed={row['seed']}"
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        X_sub = X[:, feats]

        lr_accs = []
        rf_accs = []
        for train_idx, test_idx in skf.split(X_sub, y):
            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(X_sub[train_idx], y[train_idx])
            lr_accs.append(lr.score(X_sub[test_idx], y[test_idx]))

            rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
            rf.fit(X_sub[train_idx], y[train_idx])
            rf_accs.append(rf.score(X_sub[test_idx], y[test_idx]))

        print(f"  {label:35s} ({row['n_features']:2d} feat): "
              f"LR={np.mean(lr_accs):.4f}  RF={np.mean(rf_accs):.4f}")

    print(f"\n  {'ALL 64 features':35s} (64 feat):  ", end="")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lr_accs, rf_accs = [], []
    for train_idx, test_idx in skf.split(X, y):
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X[train_idx], y[train_idx])
        lr_accs.append(lr.score(X[test_idx], y[test_idx]))
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf.fit(X[train_idx], y[train_idx])
        rf_accs.append(rf.score(X[test_idx], y[test_idx]))
    print(f"LR={np.mean(lr_accs):.4f}  RF={np.mean(rf_accs):.4f}")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("  SUMMARY: GENERALIZATION PERFORMANCE")
    print("=" * 70)

    ours_mean = np.mean([test_sets[f"GSG-Softmax+IPCAE seed={s}"]["acc_mean"]
                         for s in ours["seed"] if f"GSG-Softmax+IPCAE seed={s}" in test_sets])
    print(f"  GSG-Softmax+IPCAE 5-fold CV: {ours_mean:.4f}")
    print(f"  Random 20 features 5-fold CV: {test_sets['Random 20']['acc_mean']:.4f}")
    print(f"  ALL 64 features 5-fold CV:    {test_sets['ALL 64 features']['acc_mean']:.4f}")
    print(f"  Gap vs random: {ours_mean - test_sets['Random 20']['acc_mean']:+.4f}")
    print(f"  Gap vs full:   {test_sets['ALL 64 features']['acc_mean'] - ours_mean:+.4f}")


if __name__ == "__main__":
    main()
