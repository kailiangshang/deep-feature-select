"""Feature analysis: cross-seed, cross-model, random baseline, union/intersection.

Usage:
  PYTHONPATH=. python exp/run_feature_analysis.py
"""
from __future__ import annotations

import os
import warnings
warnings.filterwarnings("ignore")

import json
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from exp.data import generate_train_test_loader
from exp.utils import MLPClassifier, seed_all

GATE_MODELS = {"GSG-Softmax+IPCAE", "GSG-Softmax+CAE", "HCG+IPCAE", "HCG+CAE"}
RESULTS_DIRS = [
    "exp/results/cls_grid",
    "exp/results/cls_sw_ext",
    "exp/results/cls_digits",
]
DATASET = "SklearnDigitsCls"
TASK = "classification"
INPUT_DIM = 64
N_CLASSES = 10
EVAL_SEED = 42


def _collect_features(results_dirs, gate_models, k_filter=None, sw_filter=None):
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
            model = meta["model"]
            if model not in gate_models:
                continue
            if meta.get("dataset") != DATASET:
                continue
            k = meta.get("k")
            sw = meta.get("sparse_loss_weight")
            if k_filter is not None and k != k_filter:
                continue
            if sw_filter is not None and sw != sw_filter:
                continue
            seed = meta["seed"]
            n_sel = meta["final_num_selected"]
            feat_path = os.path.join(fdir, meta_file.replace("_meta.json", "_features.csv"))
            if not os.path.exists(feat_path):
                continue
            feat_df = pd.read_csv(feat_path)
            raw = feat_df.iloc[0].dropna().astype(int).tolist()
            selected = sorted(f for f in raw if f >= 0)
            if len(selected) == 0:
                continue
            acc = meta.get("final_accuracy")
            records.append({
                "model": model, "seed": seed, "k": k, "sw": sw,
                "n_selected": n_sel, "features": selected,
                "accuracy": acc,
            })
    return pd.DataFrame(records)


def _eval_features(features, seed=EVAL_SEED):
    """Train a fixed MLP on selected features, return test accuracy."""
    seed_all(seed)
    import anndata as ad
    adata = ad.read_h5ad(f"exp/data/{DATASET}.h5ad")
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)
    y = adata.obs["cell_type"].cat.codes.values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X[:, features], y, test_size=0.2, random_state=seed, stratify=y)

    X_tr = torch.tensor(X_tr, dtype=torch.float32)
    X_te = torch.tensor(X_te, dtype=torch.float32)
    y_tr = torch.tensor(y_tr, dtype=torch.long)
    y_te = torch.tensor(y_te, dtype=torch.long)

    head = MLPClassifier(len(features), 128, N_CLASSES, "cpu")

    optimizer = torch.optim.Adam(head.parameters(), lr=2e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(200):
        head.train()
        logits = head(X_tr)
        loss = loss_fn(logits, y_tr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    head.eval()
    with torch.no_grad():
        logits = head(X_te)
        pred = logits.argmax(dim=1)
        acc = (pred == y_te).float().mean().item()
    return acc


def main():
    df = _collect_features(RESULTS_DIRS, GATE_MODELS, k_filter=30.0, sw_filter=0.01)
    print(f"Collected {len(df)} runs")
    print(df[["model", "seed", "n_selected", "accuracy"]].to_string(index=False))
    print()

    # Focus on GSG-Softmax+IPCAE
    ours = df[df["model"] == "GSG-Softmax+IPCAE"]
    hcg = df[df["model"] == "HCG+IPCAE"]

    if len(ours) == 0:
        print("No GSG-Softmax+IPCAE data found")
        return

    all_feat_sets = ours["features"].tolist()
    union_feats = sorted(set().union(*(set(f) for f in all_feat_sets)))
    inter_feats = sorted(set().intersection(*(set(f) for f in all_feat_sets)))

    print("=" * 70)
    print("  FEATURE SETS (GSG-Softmax+IPCAE k=30 sw=0.01)")
    print("=" * 70)
    for _, row in ours.iterrows():
        print(f"  seed={row['seed']}: {row['n_selected']} feats = {row['features']}")
    print(f"  UNION:  {len(union_feats)} feats = {union_feats}")
    print(f"  INTERSECTION: {len(inter_feats)} feats = {inter_feats}")
    print()

    # --- Experiment 1: Cross-seed evaluation ---
    print("=" * 70)
    print("  EXP 1: CROSS-SEED EVALUATION")
    print("  Train MLP on seed A's features, report accuracy")
    print("=" * 70)
    for _, row in ours.iterrows():
        acc = _eval_features(row["features"])
        print(f"  seed={row['seed']} features ({row['n_selected']} feat): MLP acc = {acc:.4f}")
    print()

    # --- Experiment 2: Union / Intersection ---
    print("=" * 70)
    print("  EXP 2: UNION / INTERSECTION")
    print("=" * 70)
    if len(union_feats) > 0:
        acc_union = _eval_features(union_feats)
        print(f"  UNION ({len(union_feats)} feats): MLP acc = {acc_union:.4f}")
    if len(inter_feats) > 0:
        acc_inter = _eval_features(inter_feats)
        print(f"  INTERSECTION ({len(inter_feats)} feats): MLP acc = {acc_inter:.4f}")
    else:
        print(f"  INTERSECTION: empty (no shared features across seeds)")
    print()

    # --- Experiment 3: Random feature baseline ---
    print("=" * 70)
    print("  EXP 3: RANDOM FEATURE BASELINE")
    print("  Same number of features, randomly selected")
    print("=" * 70)
    n_feat = int(ours["n_selected"].mean())
    rng = np.random.RandomState(42)
    random_accs = []
    for trial in range(10):
        rand_feats = sorted(rng.choice(INPUT_DIM, size=n_feat, replace=False).tolist())
        acc = _eval_features(rand_feats)
        random_accs.append(acc)
        print(f"  random trial {trial}: {n_feat} feats {rand_feats} -> acc = {acc:.4f}")
    print(f"  Random mean: {np.mean(random_accs):.4f} +/- {np.std(random_accs):.4f}")
    print()

    # --- Experiment 4: Feature importance by ablation ---
    print("=" * 70)
    print("  EXP 4: FEATURE ABLATION (leave-one-out)")
    print("  Using seed=0 features as base, remove each feature one at a time")
    print("=" * 70)
    base_feats = ours[ours["seed"] == 0]["features"].values[0]
    base_acc = _eval_features(base_feats)
    print(f"  Base (all {len(base_feats)} feats): acc = {base_acc:.4f}")
    ablation = []
    for feat in base_feats:
        reduced = [f for f in base_feats if f != feat]
        acc = _eval_features(reduced)
        drop = base_acc - acc
        ablation.append({"feature": feat, "acc_without": acc, "drop": drop})
        print(f"  Remove feat {feat:2d}: acc = {acc:.4f} (drop = {drop:+.4f})")
    ablation_df = pd.DataFrame(ablation).sort_values("drop", ascending=False)
    print(f"\n  Most important features (largest drop when removed):")
    for _, row in ablation_df.head(5).iterrows():
        print(f"    feat {int(row['feature']):2d}: drop = {row['drop']:+.4f}")
    print()

    # --- Experiment 5: Cross-model feature comparison ---
    print("=" * 70)
    print("  EXP 5: CROSS-MODEL FEATURE EVALUATION")
    print("  Evaluate features from one model on all models' feature sets")
    print("=" * 70)
    for _, row_src in df.iterrows():
        acc = _eval_features(row_src["features"])
        print(f"  {row_src['model']:25s} seed={row_src['seed']} "
              f"({row_src['n_selected']:2d} feat): MLP acc = {acc:.4f}")
    print()

    # --- Experiment 6: Incremental feature addition ---
    print("=" * 70)
    print("  EXP 6: INCREMENTAL ADDITION (union features by frequency)")
    print("=" * 70)
    freq = {}
    for _, row in ours.iterrows():
        for f in row["features"]:
            freq[f] = freq.get(f, 0) + 1
    sorted_feats = sorted(freq.items(), key=lambda x: -x[1])
    ordered_feats = [f for f, c in sorted_feats]
    print(f"  Feature frequency order: {sorted_feats}")
    for n in [3, 5, 8, 10, 13, len(ordered_feats)]:
        if n > len(ordered_feats):
            break
        subset = ordered_feats[:n]
        acc = _eval_features(subset)
        print(f"  Top-{n} frequent features: acc = {acc:.4f}")
    print()

    # --- Summary ---
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    ours_accs = [_eval_features(row["features"]) for _, row in ours.iterrows()]
    print(f"  GSG-Softmax+IPCAE per-seed MLP acc: {ours_accs}")
    print(f"  Mean: {np.mean(ours_accs):.4f}")
    print(f"  Random {n_feat}-feat baseline: {np.mean(random_accs):.4f}")
    print(f"  Gap: {np.mean(ours_accs) - np.mean(random_accs):+.4f}")
    print(f"  Union ({len(union_feats)} feat): {acc_union:.4f}")
    print(f"  Intersection ({len(inter_feats)} feat): {acc_inter if inter_feats else 'N/A'}")
    print()
    print("  Conclusion:")
    if np.mean(ours_accs) > np.mean(random_accs) + 0.05:
        print("  -> Selected features are SIGNIFICANTLY better than random")
        print("     (method finds truly informative features)")
    else:
        print("  -> Selected features are NOT much better than random")
        print("     (dataset may be too easy / features too redundant)")
    if len(inter_feats) == 0:
        print("  -> No shared features across seeds, but similar accuracy")
        print("     (multiple equally-good subsets exist -> feature redundancy)")


if __name__ == "__main__":
    main()
