from __future__ import annotations

import os

import yaml
import torch
import pandas as pd

from deepfs import GumbelSoftmaxGateIndirectConcreteModel
from exp.data import generate_train_test_loader
from exp.trainers import GateEncoderTrainer
from exp.utils import MLPClassifier, seed_all


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
            ablations = config["ablations"]

            for abl_name, abl_cfg in ablations.items():
                print(f"\nAblation: {abl_name} | Dataset: {dataset_name} | Seed: {seed}")
                records = []

                if abl_name == "k_max":
                    for k in abl_cfg["k_values"]:
                        seed_all(seed)
                        params = {**base_params, "input_dim": data.feature_dim, "k": k,
                                  "total_epochs": training_cfg["epochs"], "device": training_cfg["device"]}
                        model = GumbelSoftmaxGateIndirectConcreteModel(**params)
                        classifier = MLPClassifier(k, classifier_cfg["hidden_dim"],
                                                   data.cls_num, training_cfg["device"])
                        trainer = GateEncoderTrainer(model, classifier,
                                                     sparse_loss_weight=training_cfg.get("sparse_loss_weight", 1.0),
                                                     lr=training_cfg["lr"], device=training_cfg["device"], seed=seed)
                        df, _ = trainer.fit(data.train_loader, training_cfg["epochs"], data.test_loader)
                        df["ablation"] = abl_name
                        df["param_value"] = k
                        df["k"] = k
                        records.append(df)

                elif abl_name == "sparse_loss_weight":
                    k = abl_cfg["k"]
                    for sw in abl_cfg["values"]:
                        seed_all(seed)
                        params = {**base_params, "input_dim": data.feature_dim, "k": k,
                                  "total_epochs": training_cfg["epochs"], "device": training_cfg["device"]}
                        model = GumbelSoftmaxGateIndirectConcreteModel(**params)
                        classifier = MLPClassifier(k, classifier_cfg["hidden_dim"],
                                                   data.cls_num, training_cfg["device"])
                        trainer = GateEncoderTrainer(model, classifier, sparse_loss_weight=sw,
                                                     lr=training_cfg["lr"], device=training_cfg["device"], seed=seed)
                        df, _ = trainer.fit(data.train_loader, training_cfg["epochs"], data.test_loader)
                        df["ablation"] = abl_name
                        df["param_value"] = sw
                        df["k"] = k
                        records.append(df)

                elif abl_name in ("embedding_dim_encoder", "embedding_dim_gate"):
                    k = abl_cfg["k"]
                    for val in abl_cfg["values"]:
                        seed_all(seed)
                        params = {**base_params, "input_dim": data.feature_dim, "k": k,
                                  abl_name: val, "total_epochs": training_cfg["epochs"],
                                  "device": training_cfg["device"]}
                        model = GumbelSoftmaxGateIndirectConcreteModel(**params)
                        classifier = MLPClassifier(k, classifier_cfg["hidden_dim"],
                                                   data.cls_num, training_cfg["device"])
                        trainer = GateEncoderTrainer(model, classifier,
                                                     sparse_loss_weight=training_cfg.get("sparse_loss_weight", 1.0),
                                                     lr=training_cfg["lr"], device=training_cfg["device"], seed=seed)
                        df, _ = trainer.fit(data.train_loader, training_cfg["epochs"], data.test_loader)
                        df["ablation"] = abl_name
                        df["param_value"] = val
                        df["k"] = k
                        records.append(df)

                elif abl_name == "temperature_schedule":
                    k = abl_cfg["k"]
                    for temp_cfg in abl_cfg["configs"]:
                        seed_all(seed)
                        params = {**base_params, "input_dim": data.feature_dim, "k": k,
                                  "initial_temperature": temp_cfg["initial"],
                                  "final_temperature": temp_cfg["final"],
                                  "total_epochs": training_cfg["epochs"],
                                  "device": training_cfg["device"]}
                        model = GumbelSoftmaxGateIndirectConcreteModel(**params)
                        classifier = MLPClassifier(k, classifier_cfg["hidden_dim"],
                                                   data.cls_num, training_cfg["device"])
                        trainer = GateEncoderTrainer(model, classifier,
                                                     sparse_loss_weight=training_cfg.get("sparse_loss_weight", 1.0),
                                                     lr=training_cfg["lr"], device=training_cfg["device"], seed=seed)
                        df, _ = trainer.fit(data.train_loader, training_cfg["epochs"], data.test_loader)
                        df["ablation"] = abl_name
                        df["param_value"] = f"{temp_cfg['initial']}->{temp_cfg['final']}"
                        df["k"] = k
                        records.append(df)

                if records:
                    ablation_df = pd.concat(records, ignore_index=True)
                    ablation_df["dataset"] = dataset_name
                    ablation_df["seed"] = seed
                    all_results.append(ablation_df)
                    ablation_df.to_csv(os.path.join(results_dir, f"{abl_name}_{dataset_name}_seed{seed}.csv"), index=False)

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(os.path.join(results_dir, "ablation_results.csv"), index=False)
        print(f"\nResults saved to {results_dir}/ablation_results.csv")


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
