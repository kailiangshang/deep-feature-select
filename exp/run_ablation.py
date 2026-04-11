from __future__ import annotations

import os
import json

import yaml
import torch
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


def _run_single(training_cfg, classifier_cfg, base_params, data, seed, results_dir,
                abl_name, param_value_str, trainer_overrides=None, **model_overrides):
    k = model_overrides.get("k", base_params.get("k", 20))
    params = {**base_params, "input_dim": data.feature_dim,
              "total_epochs": training_cfg["epochs"],
              "device": training_cfg["device"], **model_overrides}
    seed_all(seed)
    model = GumbelSoftmaxGateIndirectConcreteModel(**params)
    head = MLPClassifier(k, classifier_cfg["hidden_dim"], data.cls_num, training_cfg["device"])
    t_overrides = trainer_overrides or {}
    sw = t_overrides.get("sparse_loss_weight", training_cfg.get("sparse_loss_weight", 1.0))
    trainer = GateEncoderTrainer(model, head, task="classification",
                                 sparse_loss_weight=sw,
                                 lr=training_cfg["lr"], device=training_cfg["device"], seed=seed)
    result_df, feature_df = trainer.fit(data.train_loader, training_cfg["epochs"], data.test_loader)
    result_df["ablation"] = abl_name
    result_df["param_value"] = param_value_str
    result_df["k"] = k
    result_df["seed"] = seed
    result_df["dataset"] = data.name
    result_df["param_count"] = _param_count(model)

    group_name = f"ablation_{abl_name}_{data.name}_seed{seed}_{param_value_str}"
    meta = {
        "ablation": abl_name, "param_value": param_value_str,
        "dataset": data.name, "seed": seed, "k": k,
        "param_count": _param_count(model),
        "final_accuracy": float(result_df["accuracy"].iloc[-1]),
        "final_num_selected": int(result_df["num_selected"].iloc[-1]),
        "final_gate_open_ratio": float(result_df["gate_open_ratio"].iloc[-1]),
    }
    _save_group(results_dir, group_name, result_df, feature_df, meta)
    return result_df, feature_df


def run_ablation(config):
    training_cfg = config["training"]
    classifier_cfg = config["classifier"]
    base_params = config["base_params"]
    results_dir = config.get("results_dir", "exp/results/ablation")
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

            for abl_name, abl_cfg in config["ablations"].items():
                print(f"\nAblation: {abl_name} | Dataset: {dataset_name} | Seed: {seed}")

                if abl_name == "k_max":
                    for k in abl_cfg["k_values"]:
                        df, _ = _run_single(
                            training_cfg, classifier_cfg, base_params, data, seed,
                            results_dir, abl_name, str(k), k=k,
                        )
                        all_results.append(df)

                elif abl_name == "sparse_loss_weight":
                    k = abl_cfg["k"]
                    for sw in abl_cfg["values"]:
                        df, _ = _run_single(
                            training_cfg, classifier_cfg, base_params, data, seed,
                            results_dir, abl_name, str(sw), k=k,
                            trainer_overrides={"sparse_loss_weight": sw},
                        )
                        all_results.append(df)

                elif abl_name in ("embedding_dim_encoder", "embedding_dim_gate"):
                    k = abl_cfg["k"]
                    for val in abl_cfg["values"]:
                        df, _ = _run_single(
                            training_cfg, classifier_cfg, base_params, data, seed,
                            results_dir, abl_name, str(val), k=k, **{abl_name: val},
                        )
                        all_results.append(df)

                elif abl_name == "temperature_schedule":
                    k = abl_cfg["k"]
                    for temp_cfg in abl_cfg["configs"]:
                        pv = f"{temp_cfg['initial']}->{temp_cfg['final']}"
                        df, _ = _run_single(
                            training_cfg, classifier_cfg, base_params, data, seed,
                            results_dir, abl_name, pv, k=k,
                            initial_temperature=temp_cfg["initial"],
                            final_temperature=temp_cfg["final"],
                        )
                        all_results.append(df)

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(os.path.join(results_dir, "ablation_results.csv"), index=False)

        summary = final_df[final_df["epoch"] == final_df["epoch"].max()].groupby(
            ["ablation", "param_value", "dataset"]
        ).agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            num_selected_mean=("num_selected", "mean"),
        ).reset_index()
        summary.to_csv(os.path.join(results_dir, "ablation_summary.csv"), index=False)
        print(f"\nResults saved to {results_dir}/")
        print(f"  ablation_results.csv  — all epochs")
        print(f"  ablation_summary.csv  — final epoch summary")
        print(f"  per_group/            — per-experiment details")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="exp/configs/ablation.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_ablation(config)


if __name__ == "__main__":
    main()
