from __future__ import annotations

import os
import json

import yaml
import torch
import pandas as pd
import numpy as np

from deepfs import (
    ConcreteAutoencoderModel,
    GumbelSigmoidGateModel,
    GumbelSoftmaxGateConcreteModel,
    GumbelSoftmaxGateModel,
    GumbelSoftmaxGateIndirectConcreteModel,
    HardConcreteGateConcreteModel,
    HardConcreteGateModel,
    HardConcreteGateIndirectConcreteModel,
    IndirectConcreteAutoencoderModel,
    StochasticGateConcreteModel,
    StochasticGateModel,
    StochasticGateIndirectConcreteModel,
)
from exp.data import generate_train_test_loader
from exp.trainers import EncoderTrainer, GateTrainer, GateEncoderTrainer
from exp.utils import MLPClassifier, seed_all

MODEL_REGISTRY = {
    "ConcreteAutoencoderModel": ConcreteAutoencoderModel,
    "IndirectConcreteAutoencoderModel": IndirectConcreteAutoencoderModel,
    "StochasticGateModel": StochasticGateModel,
    "GumbelSigmoidGateModel": GumbelSigmoidGateModel,
    "GumbelSoftmaxGateModel": GumbelSoftmaxGateModel,
    "HardConcreteGateModel": HardConcreteGateModel,
    "GumbelSoftmaxGateConcreteModel": GumbelSoftmaxGateConcreteModel,
    "GumbelSoftmaxGateIndirectConcreteModel": GumbelSoftmaxGateIndirectConcreteModel,
    "StochasticGateConcreteModel": StochasticGateConcreteModel,
    "StochasticGateIndirectConcreteModel": StochasticGateIndirectConcreteModel,
    "HardConcreteGateConcreteModel": HardConcreteGateConcreteModel,
    "HardConcreteGateIndirectConcreteModel": HardConcreteGateIndirectConcreteModel,
}


def _param_count(model):
    return sum(p.numel() for p in model.parameters())


def _save_group(results_dir, group_name, result_df, feature_df, meta):
    group_dir = os.path.join(results_dir, "per_group")
    os.makedirs(group_dir, exist_ok=True)
    result_df.to_csv(os.path.join(group_dir, f"{group_name}_result.csv"), index=False)
    if feature_df is not None:
        feature_df.to_csv(os.path.join(group_dir, f"{group_name}_features.csv"), index=False)
    with open(os.path.join(group_dir, f"{group_name}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)


def run_encoder_experiment(config, dataset_name, seed, results_dir):
    training_cfg = config["training"]
    classifier_cfg = config["classifier"]
    data = generate_train_test_loader(
        name=dataset_name, batch_size=training_cfg["batch_size"],
        device=training_cfg["device"], random_state=seed,
    )
    all_results = []
    for model_cfg in config["encoder_models"]:
        model_cls = MODEL_REGISTRY[model_cfg["class"]]
        for k in model_cfg["k_values"]:
            seed_all(seed)
            params = {**model_cfg["params"], "input_dim": data.feature_dim, "output_dim": k,
                      "total_epochs": training_cfg["epochs"], "device": training_cfg["device"]}
            model = model_cls(**params)
            head = MLPClassifier(k, classifier_cfg["hidden_dim"], data.cls_num, training_cfg["device"])
            trainer = EncoderTrainer(model, head, task="classification",
                                     lr=training_cfg["lr"], device=training_cfg["device"], seed=seed)
            df = trainer.fit(data.train_loader, training_cfg["epochs"], data.test_loader)
            final = model.get_selection_result()

            group_name = f"{model_cfg['name']}_{dataset_name}_seed{seed}_k{k}"
            features = pd.DataFrame(
                [final.selected_indices],
                columns=[f"slot_{i}" for i in range(len(final.selected_indices))],
            )
            meta = {
                "model": model_cfg["name"], "model_class": model_cfg["class"],
                "dataset": dataset_name, "seed": seed, "k": k,
                "input_dim": data.feature_dim, "param_count": _param_count(model),
                "final_num_selected": int(final.num_selected),
            }
            df["model"] = model_cfg["name"]
            df["k"] = k
            df["seed"] = seed
            df["dataset"] = dataset_name
            df["param_count"] = _param_count(model)
            all_results.append(df)
            _save_group(results_dir, group_name, df, features, meta)
    return pd.concat(all_results, ignore_index=True)


def run_gate_experiment(config, dataset_name, seed, results_dir):
    training_cfg = config["training"]
    classifier_cfg = config["classifier"]
    data = generate_train_test_loader(
        name=dataset_name, batch_size=training_cfg["batch_size"],
        device=training_cfg["device"], random_state=seed,
    )
    all_results = []
    for model_cfg in config["gate_models"]:
        model_cls = MODEL_REGISTRY[model_cfg["class"]]
        for sw in model_cfg["sparse_loss_weights"]:
            seed_all(seed)
            params = {**model_cfg["params"], "input_dim": data.feature_dim,
                      "total_epochs": training_cfg["epochs"], "device": training_cfg["device"]}
            model = model_cls(**params)
            head = MLPClassifier(data.feature_dim, classifier_cfg["hidden_dim"],
                                 data.cls_num, training_cfg["device"])
            trainer = GateTrainer(model, head, task="classification",
                                  sparse_loss_weight=sw, lr=training_cfg["lr"],
                                  device=training_cfg["device"], seed=seed)
            df = trainer.fit(data.train_loader, training_cfg["epochs"], data.test_loader)
            final = model.get_selection_result()

            group_name = f"{model_cfg['name']}_{dataset_name}_seed{seed}_sw{sw}"
            features = pd.DataFrame(
                [final.selected_indices],
                columns=[f"slot_{i}" for i in range(len(final.selected_indices))],
            )
            meta = {
                "model": model_cfg["name"], "model_class": model_cfg["class"],
                "dataset": dataset_name, "seed": seed,
                "sparse_loss_weight": sw, "input_dim": data.feature_dim,
                "param_count": _param_count(model),
                "final_num_selected": int(final.num_selected),
            }
            df["model"] = model_cfg["name"]
            df["sparse_weight"] = sw
            df["seed"] = seed
            df["dataset"] = dataset_name
            df["param_count"] = _param_count(model)
            all_results.append(df)
            _save_group(results_dir, group_name, df, features, meta)
    return pd.concat(all_results, ignore_index=True)


def run_combined_experiment(config, dataset_name, seed, results_dir):
    training_cfg = config["training"]
    classifier_cfg = config["classifier"]
    data = generate_train_test_loader(
        name=dataset_name, batch_size=training_cfg["batch_size"],
        device=training_cfg["device"], random_state=seed,
    )
    all_results = []
    for model_cfg in config["combined_models"]:
        model_cls = MODEL_REGISTRY[model_cfg["class"]]
        for k in model_cfg["k_values"]:
            for sw in model_cfg["sparse_loss_weights"]:
                seed_all(seed)
                params = {**model_cfg["params"], "input_dim": data.feature_dim, "k": k,
                          "total_epochs": training_cfg["epochs"], "device": training_cfg["device"]}
                model = model_cls(**params)
                head = MLPClassifier(k, classifier_cfg["hidden_dim"],
                                     data.cls_num, training_cfg["device"])
                trainer = GateEncoderTrainer(model, head, task="classification",
                                             sparse_loss_weight=sw, lr=training_cfg["lr"],
                                             device=training_cfg["device"], seed=seed)
                result_df, feature_df = trainer.fit(data.train_loader, training_cfg["epochs"], data.test_loader)

                group_name = f"{model_cfg['name']}_{dataset_name}_seed{seed}_k{k}_sw{sw}"
                meta = {
                    "model": model_cfg["name"], "model_class": model_cfg["class"],
                    "dataset": dataset_name, "seed": seed, "k": k,
                    "sparse_loss_weight": sw, "input_dim": data.feature_dim,
                    "param_count": _param_count(model),
                    "final_num_selected": int(result_df["num_selected"].iloc[-1]),
                    "final_accuracy": float(result_df["accuracy"].iloc[-1]),
                    "final_gate_open_ratio": float(result_df["gate_open_ratio"].iloc[-1]),
                }
                result_df["model"] = model_cfg["name"]
                result_df["k"] = k
                result_df["sparse_weight"] = sw
                result_df["seed"] = seed
                result_df["dataset"] = dataset_name
                result_df["param_count"] = _param_count(model)
                all_results.append(result_df)
                _save_group(results_dir, group_name, result_df, feature_df, meta)
    return pd.concat(all_results, ignore_index=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="exp/configs/contrast.yaml")
    parser.add_argument("--mode", choices=["encoder", "gate", "combined", "all"], default="all")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    results_dir = config.get("results_dir", "exp/results/contrast")
    os.makedirs(results_dir, exist_ok=True)

    all_results = []
    for dataset_name in config["datasets"]:
        for seed in config["seeds"]:
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset_name}, Seed: {seed}")
            print(f"{'='*60}")
            if args.mode in ("encoder", "all"):
                df = run_encoder_experiment(config, dataset_name, seed, results_dir)
                all_results.append(df)
            if args.mode in ("gate", "all"):
                df = run_gate_experiment(config, dataset_name, seed, results_dir)
                all_results.append(df)
            if args.mode in ("combined", "all"):
                df = run_combined_experiment(config, dataset_name, seed, results_dir)
                all_results.append(df)

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(os.path.join(results_dir, "contrast_results.csv"), index=False)

        summary = final_df[final_df["epoch"] == final_df["epoch"].max()].groupby(
            ["model", "dataset"]
        ).agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            num_selected_mean=("num_selected", "mean"),
        ).reset_index()
        summary.to_csv(os.path.join(results_dir, "contrast_summary.csv"), index=False)
        print(f"\nResults saved to {results_dir}/")
        print(f"  contrast_results.csv  — all epochs")
        print(f"  contrast_summary.csv  — final epoch summary (mean ± std across seeds)")
        print(f"  per_group/            — per-experiment details")


if __name__ == "__main__":
    main()
