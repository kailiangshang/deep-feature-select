from __future__ import annotations

import os
import json

import yaml
import pandas as pd

from deepfs import GumbelSoftmaxGateIndirectConcreteModel
from exp.data import generate_train_test_loader
from exp.trainers import GateEncoderTrainer
from exp.utils import MLPClassifier, seed_all


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


def run_hyperparameter(config):
    base_params = config["base_params"]
    training_cfg = config["training"]
    classifier_cfg = config["classifier"]
    results_dir = config.get("results_dir", "exp/results/hyperparameter")
    os.makedirs(results_dir, exist_ok=True)

    all_results = []
    for dataset_name in config["datasets"]:
        for seed in config["seeds"]:
            data = generate_train_test_loader(
                name=dataset_name,
                batch_size=training_cfg.get("batch_size", 512),
                device=training_cfg["device"],
                random_state=seed,
            )

            for hp_name, hp_cfg in config["hyperparameters"].items():
                print(f"\nHyperparam: {hp_name} | Dataset: {dataset_name} | Seed: {seed}")

                for val in hp_cfg["values"]:
                    seed_all(seed)
                    k = base_params["k"]
                    params = {**base_params, "input_dim": data.feature_dim, "k": k,
                              "total_epochs": training_cfg["epochs"], "device": training_cfg["device"]}
                    lr = training_cfg.get("lr", 1e-4)
                    if hp_cfg["param"] == "lr":
                        lr = val
                    else:
                        params[hp_cfg["param"]] = val

                    model = GumbelSoftmaxGateIndirectConcreteModel(**params)
                    head = MLPClassifier(k, classifier_cfg["hidden_dim"],
                                         data.cls_num, training_cfg["device"])
                    trainer = GateEncoderTrainer(
                        model, head, task="classification",
                        sparse_loss_weight=base_params.get("sparse_loss_weight", 1.0),
                        lr=lr, device=training_cfg["device"], seed=seed,
                    )
                    result_df, feature_df = trainer.fit(data.train_loader, training_cfg["epochs"], data.test_loader)
                    result_df["hyperparam"] = hp_name
                    result_df["param_value"] = val
                    result_df["dataset"] = dataset_name
                    result_df["seed"] = seed
                    result_df["k"] = k
                    result_df["param_count"] = _param_count(model)
                    all_results.append(result_df)

                    group_name = f"hp_{hp_name}_{dataset_name}_seed{seed}_{val}"
                    meta = {
                        "hyperparam": hp_name, "param_value": val,
                        "dataset": dataset_name, "seed": seed, "k": k, "lr": lr,
                        "param_count": _param_count(model),
                        "final_accuracy": float(result_df["accuracy"].iloc[-1]),
                        "final_num_selected": int(result_df["num_selected"].iloc[-1]),
                    }
                    _save_group(results_dir, group_name, result_df, feature_df, meta)

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(os.path.join(results_dir, "hyperparameter_results.csv"), index=False)

        summary = final_df[final_df["epoch"] == final_df["epoch"].max()].groupby(
            ["hyperparam", "param_value", "dataset"]
        ).agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            num_selected_mean=("num_selected", "mean"),
        ).reset_index()
        summary.to_csv(os.path.join(results_dir, "hyperparameter_summary.csv"), index=False)
        print(f"\nResults saved to {results_dir}/")
        print(f"  hyperparameter_results.csv  — all epochs")
        print(f"  hyperparameter_summary.csv  — final epoch summary")
        print(f"  per_group/                  — per-experiment details")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="exp/configs/hyperparameter.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_hyperparameter(config)


if __name__ == "__main__":
    main()
