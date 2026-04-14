"""Unified experiment runner — YAML-driven, supports all 3 tasks.

Usage:
  python exp/run_experiment.py --config exp/configs/cls_digits.yaml
  python exp/run_experiment.py --config exp/configs/reg_sklearn.yaml --mode combined
"""
from __future__ import annotations

import os
import json
import time
import warnings
warnings.filterwarnings("ignore")

import yaml
import torch
import pandas as pd
import numpy as np

from deepfs import (
    ConcreteAutoencoderModel,
    GumbelSoftmaxGateConcreteModel,
    GumbelSoftmaxGateIndirectConcreteModel,
    GumbelSoftmaxGateModel,
    HardConcreteGateConcreteModel,
    HardConcreteGateIndirectConcreteModel,
    IndirectConcreteAutoencoderModel,
    StochasticGateModel,
    HardConcreteGateModel,
)
from exp.data import generate_train_test_loader
from exp.trainers import EncoderTrainer, GateTrainer, GateEncoderTrainer
from exp.utils import MLPClassifier, MLPRegressor, AutoencoderHead, seed_all

MODEL_REGISTRY = {
    "ConcreteAutoencoderModel": ConcreteAutoencoderModel,
    "IndirectConcreteAutoencoderModel": IndirectConcreteAutoencoderModel,
    "StochasticGateModel": StochasticGateModel,
    "GumbelSoftmaxGateModel": GumbelSoftmaxGateModel,
    "HardConcreteGateModel": HardConcreteGateModel,
    "GumbelSoftmaxGateConcreteModel": GumbelSoftmaxGateConcreteModel,
    "GumbelSoftmaxGateIndirectConcreteModel": GumbelSoftmaxGateIndirectConcreteModel,
    "HardConcreteGateConcreteModel": HardConcreteGateConcreteModel,
    "HardConcreteGateIndirectConcreteModel": HardConcreteGateIndirectConcreteModel,
}

HEAD_MAP = {
    "classification": MLPClassifier,
    "regression": MLPRegressor,
    "reconstruction": AutoencoderHead,
}


def _param_count(m):
    return sum(p.numel() for p in m.parameters())


def _build_model(model_cls, extra_params, training_epochs):
    import inspect
    sig = inspect.signature(model_cls)
    params = {}
    for k, v in extra_params.items():
        if k in sig.parameters:
            params[k] = v
    if "total_epochs" in sig.parameters and "total_epochs" not in params:
        params["total_epochs"] = training_epochs
    return model_cls(**params)


def _make_head(task, input_dim, hidden_dim, output_dim, device):
    return HEAD_MAP[task](input_dim, hidden_dim, output_dim, device)


def _save_group(results_dir, group_name, result_df, feature_df, meta):
    gdir = os.path.join(results_dir, "per_group")
    os.makedirs(gdir, exist_ok=True)
    result_df.to_csv(os.path.join(gdir, f"{group_name}_result.csv"), index=False)
    if feature_df is not None:
        feature_df.to_csv(os.path.join(gdir, f"{group_name}_features.csv"), index=False)
    with open(os.path.join(gdir, f"{group_name}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)


def _compute_convergence_epoch(df, metric_col):
    final_val = df[metric_col].iloc[-1]
    if abs(final_val) < 1e-8:
        return len(df)
    threshold = 0.95 * final_val if final_val > 0 else 1.05 * final_val
    for _, row in df.iterrows():
        if final_val > 0 and row[metric_col] >= threshold:
            return int(row["epoch"])
        elif final_val < 0 and row[metric_col] <= threshold:
            return int(row["epoch"])
    return len(df)


def run_encoder(config, dataset_name, seed, results_dir, task, training_cfg, head_cfg):
    data = generate_train_test_loader(
        name=dataset_name, batch_size=training_cfg["batch_size"],
        device=training_cfg["device"], random_state=seed,
    )
    all_results = []

    for model_cfg in config.get("encoder_models", []):
        model_cls = MODEL_REGISTRY[model_cfg["class"]]
        for k in model_cfg.get("k_values", [config.get("k_default", 20)]):
            seed_all(seed)
            t0 = time.time()
            params = {**model_cfg["params"], "input_dim": data.feature_dim, "output_dim": k,
                      "device": training_cfg["device"]}
            model = _build_model(model_cls, params, training_cfg["epochs"])

            if task == "classification":
                out_dim = data.cls_num
            elif task == "reconstruction":
                out_dim = data.feature_dim
            else:
                out_dim = 1

            head = _make_head(task, k, head_cfg["hidden_dim"], out_dim, training_cfg["device"])
            trainer_kwargs = {"input_dim": data.feature_dim} if task == "reconstruction" else {}
            trainer = EncoderTrainer(model, head, task=task, lr=training_cfg["lr"],
                                     device=training_cfg["device"], seed=seed, **trainer_kwargs)
            df = trainer.fit(data.train_loader, training_cfg["epochs"], data.test_loader)
            elapsed = time.time() - t0

            final = model.get_selection_result()
            metric_col = trainer.task.metric_name
            conv_epoch = _compute_convergence_epoch(df, metric_col)

            features = pd.DataFrame(
                [final.selected_indices],
                columns=[f"slot_{i}" for i in range(len(final.selected_indices))],
            )
            group_name = f"{model_cfg['name']}_{dataset_name}_seed{seed}_k{k}"
            meta = {
                "model": model_cfg["name"], "model_class": model_cfg["class"],
                "task": task, "dataset": dataset_name, "seed": seed, "k": k,
                "input_dim": data.feature_dim, "param_count": _param_count(model),
                "final_num_selected": int(final.num_selected),
                f"final_{metric_col}": float(df[metric_col].iloc[-1]),
                "convergence_epoch": conv_epoch,
                "training_time_sec": round(elapsed, 2),
            }
            df["model"] = model_cfg["name"]
            df["k"] = k
            df["seed"] = seed
            df["dataset"] = dataset_name
            df["task"] = task
            df["param_count"] = _param_count(model)
            df["convergence_epoch"] = conv_epoch
            df["training_time_sec"] = round(elapsed, 2)
            all_results.append(df)
            _save_group(results_dir, group_name, df, features, meta)

    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


def run_gate(config, dataset_name, seed, results_dir, task, training_cfg, head_cfg):
    data = generate_train_test_loader(
        name=dataset_name, batch_size=training_cfg["batch_size"],
        device=training_cfg["device"], random_state=seed,
    )
    all_results = []

    for model_cfg in config.get("gate_models", []):
        model_cls = MODEL_REGISTRY[model_cfg["class"]]
        for sw in model_cfg.get("sparse_loss_weights", [1.0]):
            seed_all(seed)
            t0 = time.time()
            params = {**model_cfg["params"], "input_dim": data.feature_dim,
                      "device": training_cfg["device"]}
            model = _build_model(model_cls, params, training_cfg["epochs"])

            if task == "classification":
                out_dim = data.cls_num
            elif task == "reconstruction":
                out_dim = data.feature_dim
            else:
                out_dim = 1

            head = _make_head(task, data.feature_dim, head_cfg["hidden_dim"], out_dim, training_cfg["device"])
            trainer_kwargs = {"input_dim": data.feature_dim} if task == "reconstruction" else {}
            trainer = GateTrainer(model, head, task=task, sparse_loss_weight=sw,
                                 lr=training_cfg["lr"], device=training_cfg["device"], seed=seed, **trainer_kwargs)
            df = trainer.fit(data.train_loader, training_cfg["epochs"], data.test_loader)
            elapsed = time.time() - t0

            final = model.get_selection_result()
            metric_col = trainer.task.metric_name
            conv_epoch = _compute_convergence_epoch(df, metric_col)

            features = pd.DataFrame(
                [final.selected_indices],
                columns=[f"slot_{i}" for i in range(len(final.selected_indices))],
            )
            group_name = f"{model_cfg['name']}_{dataset_name}_seed{seed}_sw{sw}"
            meta = {
                "model": model_cfg["name"], "model_class": model_cfg["class"],
                "task": task, "dataset": dataset_name, "seed": seed,
                "sparse_loss_weight": sw, "input_dim": data.feature_dim,
                "param_count": _param_count(model),
                "final_num_selected": int(final.num_selected),
                f"final_{metric_col}": float(df[metric_col].iloc[-1]),
                "convergence_epoch": conv_epoch,
                "training_time_sec": round(elapsed, 2),
            }
            df["model"] = model_cfg["name"]
            df["sparse_weight"] = sw
            df["seed"] = seed
            df["dataset"] = dataset_name
            df["task"] = task
            df["param_count"] = _param_count(model)
            df["convergence_epoch"] = conv_epoch
            df["training_time_sec"] = round(elapsed, 2)
            all_results.append(df)
            _save_group(results_dir, group_name, df, features, meta)

    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


def run_combined(config, dataset_name, seed, results_dir, task, training_cfg, head_cfg):
    data = generate_train_test_loader(
        name=dataset_name, batch_size=training_cfg["batch_size"],
        device=training_cfg["device"], random_state=seed,
    )
    all_results = []

    for model_cfg in config.get("combined_models", []):
        model_cls = MODEL_REGISTRY[model_cfg["class"]]
        for k in model_cfg.get("k_values", [config.get("k_default", 20)]):
            for sw in model_cfg.get("sparse_loss_weights", [1.0]):
                seed_all(seed)
                t0 = time.time()
                params = {**model_cfg["params"], "input_dim": data.feature_dim, "k": k,
                          "device": training_cfg["device"]}
                model = _build_model(model_cls, params, training_cfg["epochs"])

                if task == "classification":
                    out_dim = data.cls_num
                elif task == "reconstruction":
                    out_dim = data.feature_dim
                else:
                    out_dim = 1

                head = _make_head(task, k, head_cfg["hidden_dim"], out_dim, training_cfg["device"])
                trainer_kwargs = {"input_dim": data.feature_dim} if task == "reconstruction" else {}
                trainer = GateEncoderTrainer(model, head, task=task, sparse_loss_weight=sw,
                                             lr=training_cfg["lr"], device=training_cfg["device"],
                                             seed=seed, **trainer_kwargs)
                result_df, feature_df = trainer.fit(data.train_loader, training_cfg["epochs"], data.test_loader)
                elapsed = time.time() - t0

                metric_col = trainer.task.metric_name
                conv_epoch = _compute_convergence_epoch(result_df, metric_col)

                group_name = f"{model_cfg['name']}_{dataset_name}_seed{seed}_k{k}_sw{sw}"
                meta = {
                    "model": model_cfg["name"], "model_class": model_cfg["class"],
                    "task": task, "dataset": dataset_name, "seed": seed, "k": k,
                    "sparse_loss_weight": sw, "input_dim": data.feature_dim,
                    "param_count": _param_count(model),
                    "final_num_selected": int(result_df["num_selected"].iloc[-1]),
                    f"final_{metric_col}": float(result_df[metric_col].iloc[-1]),
                    "convergence_epoch": conv_epoch,
                    "training_time_sec": round(elapsed, 2),
                }
                result_df["model"] = model_cfg["name"]
                result_df["k"] = k
                result_df["sparse_weight"] = sw
                result_df["seed"] = seed
                result_df["dataset"] = dataset_name
                result_df["task"] = task
                result_df["param_count"] = _param_count(model)
                result_df["convergence_epoch"] = conv_epoch
                result_df["training_time_sec"] = round(elapsed, 2)
                all_results.append(result_df)
                _save_group(results_dir, group_name, result_df, feature_df, meta)

    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["encoder", "gate", "combined", "all"], default="all")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    task = config["task"]
    results_dir = config.get("results_dir", "exp/results/experiment")
    os.makedirs(results_dir, exist_ok=True)

    training_cfg = config["training"]
    training_cfg["lr"] = float(training_cfg["lr"])
    training_cfg["epochs"] = int(training_cfg["epochs"])
    training_cfg["batch_size"] = int(training_cfg["batch_size"])
    head_cfg = config.get("head", {"hidden_dim": 128})
    seeds = config.get("seeds", [0])

    all_results = []
    for dataset_name in config["datasets"]:
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"  Dataset: {dataset_name} | Task: {task} | Seed: {seed}")
            print(f"{'='*60}")
            if args.mode in ("encoder", "all"):
                df = run_encoder(config, dataset_name, seed, results_dir, task, training_cfg, head_cfg)
                if not df.empty:
                    all_results.append(df)
            if args.mode in ("gate", "all"):
                df = run_gate(config, dataset_name, seed, results_dir, task, training_cfg, head_cfg)
                if not df.empty:
                    all_results.append(df)
            if args.mode in ("combined", "all"):
                df = run_combined(config, dataset_name, seed, results_dir, task, training_cfg, head_cfg)
                if not df.empty:
                    all_results.append(df)

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(os.path.join(results_dir, "all_results.csv"), index=False)

        metric_col = {"classification": "accuracy", "regression": "r2", "reconstruction": "neg_mse"}[task]
        last_epoch = final_df["epoch"].max()
        summary = final_df[final_df["epoch"] == last_epoch].groupby(
            ["model", "dataset", "k"] if "k" in final_df.columns else ["model", "dataset"]
        ).agg(
            metric_mean=(metric_col, "mean"),
            metric_std=(metric_col, "std"),
            num_selected_mean=("num_selected", "mean"),
            convergence_epoch_mean=("convergence_epoch", "mean"),
            training_time_mean=("training_time_sec", "mean"),
        ).reset_index()
        summary.to_csv(os.path.join(results_dir, "summary.csv"), index=False)
        print(f"\nResults saved to {results_dir}/")


if __name__ == "__main__":
    main()
