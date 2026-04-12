"""Threshold sensitivity: sweep threshold for STG/HCG, compare with GSG-Softmax (threshold-free).

Usage:
  python exp/run_threshold_sweep.py --config exp/configs/cls_digits.yaml
"""
from __future__ import annotations

import os
import warnings
warnings.filterwarnings("ignore")

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
from torch.utils.data import DataLoader, TensorDataset

from deepfs import StochasticGateModel, HardConcreteGateModel
from exp.data import generate_train_test_loader
from exp.trainers import GateTrainer
from exp.utils import MLPClassifier, MLPRegressor, seed_all

DEVICE = "cpu"


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


def get_gate_probs(model, model_name):
    with torch.no_grad():
        if model_name == "STG":
            probs = torch.clamp(model.mu_gate + 0.5, 0.0, 1.0).numpy()
        elif model_name == "HCG":
            s = torch.sigmoid(model.gate_logits)
            probs = torch.clamp(s * (model.zeta - model.gamma) + model.gamma, 0.0, 1.0).numpy()
        else:
            raise ValueError(f"Unknown model: {model_name}")
    return probs


def eval_with_features(X_train, y_train, X_test, y_test, features, task, epochs=100):
    if not features:
        return {("accuracy" if task == "classification" else "r2"): 0.0, "n_features": 0}

    X_tr = X_train[:, features]
    X_te = X_test[:, features]

    if task == "classification":
        n_cls = len(set(y_train.tolist() + y_test.tolist()))
        model = FixedMLP(X_tr.shape[1], 64, n_cls)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_train, dtype=torch.long))
        for _ in range(epochs):
            model.train()
            for xb, yb in DataLoader(ds, batch_size=64, shuffle=True):
                opt.zero_grad()
                F.cross_entropy(model(xb), yb).backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_te)).argmax(1).numpy()
        return {"accuracy": float(accuracy_score(y_test, preds)), "n_features": len(features)}
    else:
        model = FixedMLP(X_tr.shape[1], 64, 1)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_train.reshape(-1, 1)))
        for _ in range(epochs):
            model.train()
            for xb, yb in DataLoader(ds, batch_size=64, shuffle=True):
                opt.zero_grad()
                F.mse_loss(model(xb), yb).backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_te)).numpy().flatten()
        return {"r2": float(r2_score(y_test, preds)), "n_features": len(features)}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--downstream_epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    task = config["task"]
    training_cfg = config["training"]
    training_cfg["lr"] = float(training_cfg["lr"])
    training_cfg["epochs"] = int(training_cfg["epochs"])
    training_cfg["batch_size"] = int(training_cfg["batch_size"])
    results_dir = config.get("results_dir", "exp/results/experiment")
    thresholds = np.arange(0.1, 1.0, 0.1)
    sparse_weights = config.get("threshold_sweep_sw", [0.1, 1.0])

    gate_models_cfg = [
        ("STG", StochasticGateModel, {}),
        ("HCG", HardConcreteGateModel, {}),
    ]

    all_results = []
    for dataset_name in config["datasets"]:
        data = generate_train_test_loader(
            name=dataset_name, batch_size=training_cfg["batch_size"],
            device=training_cfg["device"], random_state=args.seed,
        )

        import scanpy as sc
        current_dir = os.path.dirname(os.path.abspath(__file__))
        adata = sc.read_h5ad(os.path.join(current_dir, "data", f"{dataset_name}.h5ad"))
        X_full = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
        X_full = X_full.astype(np.float32)

        if task == "classification":
            le = LabelEncoder()
            y_full = le.fit_transform(adata.obs["cell_type"].values).astype(np.int64)
        else:
            y_full = adata.obs["target"].values.astype(np.float32)

        X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
            X_full, y_full, test_size=0.2, random_state=42,
            stratify=y_full if task == "classification" else None
        )

        for model_name, model_cls, extra_params in gate_models_cfg:
            for sw in sparse_weights:
                print(f"\n  Training {model_name} on {dataset_name}, sw={sw}")
                seed_all(args.seed)
                params = dict(device=training_cfg["device"])
                params.update(extra_params)
                import inspect
                sig = inspect.signature(model_cls)
                valid = {k: v for k, v in params.items() if k in sig.parameters}
                if "total_epochs" in sig.parameters:
                    valid["total_epochs"] = args.epochs
                model = model_cls(data.feature_dim, **valid)

                if task == "classification":
                    head = MLPClassifier(data.feature_dim, 128, data.cls_num, DEVICE)
                else:
                    head = MLPRegressor(data.feature_dim, 128, 1, DEVICE)

                trainer = GateTrainer(model, head, task=task, sparse_loss_weight=sw,
                                      lr=training_cfg["lr"], device=training_cfg["device"], seed=args.seed)
                trainer.fit(data.train_loader, args.epochs, data.test_loader)

                probs = get_gate_probs(model, model_name)

                print(f"    Gate probs: min={probs.min():.4f} max={probs.max():.4f} "
                      f"mean={probs.mean():.4f} in_ambiguous(0.3-0.7)={((probs > 0.3) & (probs < 0.7)).sum()}/{len(probs)}")

                for threshold in thresholds:
                    features = sorted(np.where(probs > threshold)[0].tolist())
                    result = eval_with_features(
                        X_train_raw, y_train_raw, X_test_raw, y_test_raw,
                        features, task, args.downstream_epochs
                    )
                    metric_col = "accuracy" if task == "classification" else "r2"
                    result.update({
                        "model": model_name,
                        "dataset": dataset_name,
                        "task": task,
                        "sparse_weight": sw,
                        "threshold": round(float(threshold), 2),
                        "ambiguous_count": int(((probs > 0.3) & (probs < 0.7)).sum()),
                    })
                    all_results.append(result)
                    print(f"      threshold={threshold:.1f}  {metric_col}={result[metric_col]:.4f}  "
                          f"feat={result['n_features']}")

        # GSG-Softmax: no threshold needed (single point)
        from deepfs import GumbelSoftmaxGateModel
        for sw in sparse_weights:
            seed_all(args.seed)
            model = GumbelSoftmaxGateModel(data.feature_dim, 16, total_epochs=args.epochs,
                                           device=training_cfg["device"])
            if task == "classification":
                head = MLPClassifier(data.feature_dim, 128, data.cls_num, DEVICE)
            else:
                head = MLPRegressor(data.feature_dim, 128, 1, DEVICE)

            trainer = GateTrainer(model, head, task=task, sparse_loss_weight=sw,
                                  lr=training_cfg["lr"], device=training_cfg["device"], seed=args.seed)
            trainer.fit(data.train_loader, args.epochs, data.test_loader)
            final = model.get_selection_result()
            features = sorted(final.selected_indices.tolist())

            result = eval_with_features(
                X_train_raw, y_train_raw, X_test_raw, y_test_raw,
                features, task, args.downstream_epochs
            )
            metric_col = "accuracy" if task == "classification" else "r2"
            result.update({
                "model": "GSG-Softmax",
                "dataset": dataset_name,
                "task": task,
                "sparse_weight": sw,
                "threshold": "N/A (argmax)",
                "ambiguous_count": 0,
            })
            all_results.append(result)
            print(f"    GSG-Softmax sw={sw}: {metric_col}={result[metric_col]:.4f}  feat={result['n_features']}  (no threshold)")

    if all_results:
        df = pd.DataFrame(all_results)
        out_path = os.path.join(results_dir, "threshold_sensitivity.csv")
        df.to_csv(out_path, index=False)
        print(f"\nThreshold sensitivity saved to {out_path}")


if __name__ == "__main__":
    main()
