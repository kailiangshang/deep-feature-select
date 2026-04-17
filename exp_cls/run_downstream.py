"""Downstream evaluation: retrain fixed MLP on selected features.

Usage:
  python -m exp_cls.run_downstream --results_dir exp_cls/results --epochs 100
"""

from __future__ import annotations

import glob
import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class FixedMLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(dim_hidden, dim_out),
        )

    def forward(self, x):
        return self.net(x)


def load_features_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    features = []
    for col in df.columns:
        vals = df[col].dropna().values
        features.extend([int(v) for v in vals])
    return sorted(set(features))


def load_data(dataset_name, features=None):
    path = os.path.join(DATA_DIR, f"{dataset_name}.h5ad")
    adata = sc.read_h5ad(path)

    X = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
    X = X.astype(np.float32)

    if features is not None:
        X = X[:, features]

    has_cell_type = "cell_type" in adata.obs.columns
    unique_ct = adata.obs["cell_type"].nunique() if has_cell_type else 0
    is_regression = unique_ct > 20 or "target" in adata.obs.columns

    if is_regression:
        y = (
            adata.obs["target"].values.astype(np.float32)
            if "target" in adata.obs.columns
            else None
        )
        task = "regression"
    else:
        le = LabelEncoder()
        y = le.fit_transform(adata.obs["cell_type"].values).astype(np.int64)
        task = "classification"

    return X, y, task


def train_downstream_cls(X_train, y_train, X_test, y_test, epochs=100, lr=1e-3):
    n_cls = len(set(y_train.tolist() + y_test.tolist()))
    model = FixedMLP(X_train.shape[1], 64, n_cls)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_ds = TensorDataset(
        torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long)
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test)).argmax(dim=1).numpy()
    acc = (preds == y_test).mean()
    return {"accuracy": float(acc), "n_features": X_train.shape[1]}


def train_downstream_reg(X_train, y_train, X_test, y_test, epochs=100, lr=1e-3):
    model = FixedMLP(X_train.shape[1], 64, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_ds = TensorDataset(
        torch.tensor(X_train), torch.tensor(y_train.reshape(-1, 1))
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = F.mse_loss(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test)).numpy().flatten()
    from sklearn.metrics import r2_score

    r2 = r2_score(y_test, preds)
    return {"r2": float(r2), "n_features": X_train.shape[1]}


def run_downstream(results_dir, epochs=100):
    group_dir = os.path.join(results_dir, "per_group")
    feature_files = sorted(glob.glob(os.path.join(group_dir, "*_features.csv")))

    all_results = []
    for fpath in feature_files:
        basename = os.path.basename(fpath).replace("_features.csv", "")
        meta_path = os.path.join(group_dir, f"{basename}_meta.json")
        downstream_path = os.path.join(group_dir, f"{basename}_downstream.json")

        if os.path.exists(downstream_path):
            print(f"  SKIP downstream {basename}")
            continue

        if not os.path.exists(meta_path):
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        dataset = meta["dataset"]
        model_name = meta["model"]
        seed = meta.get("seed", 0)

        features = load_features_from_csv(fpath)
        if not features:
            print(f"  SKIP {basename}: no features")
            continue

        print(f"  Evaluating: {basename} ({len(features)} features)")

        X, y, task = load_data(dataset, features=features)
        stratify = y if task == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )

        if task == "classification":
            result = train_downstream_cls(X_train, y_train, X_test, y_test, epochs)
        else:
            result = train_downstream_reg(X_train, y_train, X_test, y_test, epochs)

        result["model"] = model_name
        result["dataset"] = dataset
        result["seed"] = seed
        result["task"] = task
        result["group"] = basename
        if "k" in meta:
            result["k"] = meta["k"]
        if "sparse_loss_weight" in meta:
            result["sparse_weight"] = meta["sparse_loss_weight"]

        with open(downstream_path, "w") as f:
            json.dump(result, f, indent=2)
        all_results.append(result)

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(os.path.join(results_dir, "downstream_results.csv"), index=False)
        print(f"\nDownstream results saved to {results_dir}/downstream_results.csv")

        task = df["task"].iloc[0] if len(df) > 0 else "unknown"
        metric = "accuracy" if task == "classification" else "r2"
        print(f"\n  Top models by downstream {metric}:")
        for _, row in df.sort_values(metric, ascending=False).head(10).iterrows():
            print(
                f"    {row[metric]:.4f}  feat={row['n_features']}  {row['model']}  {row['group']}"
            )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="exp_cls/results")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    run_downstream(args.results_dir, args.epochs)


if __name__ == "__main__":
    main()
