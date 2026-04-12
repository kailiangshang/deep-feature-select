"""Classic baselines: LASSO + Random Forest feature selection + downstream eval.

Usage:
  python exp/run_baselines.py --config exp/configs/cls_digits.yaml
"""
from __future__ import annotations

import os
import warnings
warnings.filterwarnings("ignore")

import yaml
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class FixedMLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(dim_hidden, dim_hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(dim_hidden, dim_out),
        )

    def forward(self, x):
        return self.net(x)


def load_data(dataset_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, "data", f"{dataset_name}.h5ad")
    adata = sc.read_h5ad(path)
    X = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
    X = X.astype(np.float32)

    has_cell_type = "cell_type" in adata.obs.columns
    unique_ct = adata.obs["cell_type"].nunique() if has_cell_type else 0
    is_regression = unique_ct > 20 or "target" in adata.obs.columns

    if is_regression:
        y = adata.obs["target"].values.astype(np.float32)
        task = "regression"
    else:
        le = LabelEncoder()
        y = le.fit_transform(adata.obs["cell_type"].values).astype(np.int64)
        task = "classification"
    return X, y, task, X.shape[1]


def select_features_lasso(X, y, task, n_features):
    if task == "classification":
        model = LogisticRegressionCV(penalty="l1", solver="saga", max_iter=5000, cv=3)
        model.fit(X, y)
        importance = np.abs(model.coef_).sum(axis=0)
    else:
        model = LassoCV(cv=3, max_iter=5000)
        model.fit(X, y)
        importance = np.abs(model.coef_)
    top_idx = np.argsort(importance)[::-1][:n_features]
    return sorted(top_idx.tolist())


def select_features_rf(X, y, task, n_features):
    if task == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    importance = model.feature_importances_
    top_idx = np.argsort(importance)[::-1][:n_features]
    return sorted(top_idx.tolist())


def eval_downstream(X_train, y_train, X_test, y_test, task, epochs=100):
    if task == "classification":
        n_cls = len(set(y_train.tolist() + y_test.tolist()))
        model = FixedMLP(X_train.shape[1], 64, n_cls)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
        for epoch in range(epochs):
            model.train()
            for xb, yb in DataLoader(train_ds, batch_size=64, shuffle=True):
                optimizer.zero_grad()
                F.cross_entropy(model(xb), yb).backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_test)).argmax(dim=1).numpy()
        return {"accuracy": float(accuracy_score(y_test, preds))}
    else:
        model = FixedMLP(X_train.shape[1], 64, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train.reshape(-1, 1)))
        for epoch in range(epochs):
            model.train()
            for xb, yb in DataLoader(train_ds, batch_size=64, shuffle=True):
                optimizer.zero_grad()
                F.mse_loss(model(xb), yb).backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_test)).numpy().flatten()
        return {"r2": float(r2_score(y_test, preds))}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    task = config["task"]
    results_dir = config.get("results_dir", "exp/results/experiment")
    n_features_list = config.get("baseline_n_features", [10, 20, 30])

    all_results = []
    for dataset_name in config["datasets"]:
        X, y, data_task, n_dim = load_data(dataset_name)
        print(f"\n  Dataset: {dataset_name} ({X.shape[0]} × {X.shape[1]}, task={data_task})")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if data_task == "classification" else None
        )

        for n_feat in n_features_list:
            if n_feat > n_dim:
                continue
            print(f"    n_features={n_feat}")

            for method_name, select_fn in [("LASSO", select_features_lasso), ("RF", select_features_rf)]:
                features = select_fn(X_train, y_train, data_task, n_feat)
                result = eval_downstream(
                    X_train[:, features], y_train,
                    X_test[:, features], y_test,
                    data_task, args.epochs
                )
                result["method"] = method_name
                result["dataset"] = dataset_name
                result["task"] = data_task
                result["n_features"] = n_feat
                result["selected_features"] = features
                all_results.append(result)
                metric = "accuracy" if data_task == "classification" else "r2"
                print(f"      {method_name}: {metric}={result[metric]:.4f}")

    if all_results:
        df = pd.DataFrame(all_results)
        out_path = os.path.join(results_dir, "baseline_results.csv")
        df.to_csv(out_path, index=False)
        print(f"\nBaseline results saved to {out_path}")


if __name__ == "__main__":
    main()
