from __future__ import annotations

import os

import yaml
import torch
import pandas as pd

from deepfs import GumbelSoftmaxGateIndirectConcreteModel
from exp.data import generate_train_test_loader
from exp.trainers import GateEncoderTrainer
from exp.utils import MLPClassifier, seed_all


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
                records = []

                for val in hp_cfg["values"]:
                    seed_all(seed)
                    k = base_params["k"]
                    params = dict(base_params)
                    params["input_dim"] = data.feature_dim
                    params["k"] = k
                    params["total_epochs"] = training_cfg["epochs"]
                    params["device"] = training_cfg["device"]

                    if hp_cfg["param"] == "lr":
                        lr = val
                    else:
                        lr = training_cfg["lr"]
                        params[hp_cfg["param"]] = val

                    model = GumbelSoftmaxGateIndirectConcreteModel(**params)
                    classifier = MLPClassifier(k, classifier_cfg["hidden_dim"],
                                               data.cls_num, training_cfg["device"])
                    trainer = GateEncoderTrainer(model, classifier,
                                                 sparse_loss_weight=base_params.get("sparse_loss_weight", 1.0),
                                                 lr=lr, device=training_cfg["device"], seed=seed)
                    df, _ = trainer.fit(data.train_loader, training_cfg["epochs"], data.test_loader)
                    df["hyperparam"] = hp_name
                    df["param_value"] = val
                    df["dataset"] = dataset_name
                    df["seed"] = seed
                    records.append(df)

                if records:
                    hp_df = pd.concat(records, ignore_index=True)
                    all_results.append(hp_df)
                    hp_df.to_csv(os.path.join(results_dir, f"{hp_name}_{dataset_name}_seed{seed}.csv"), index=False)

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(os.path.join(results_dir, "hyperparameter_results.csv"), index=False)
        print(f"\nResults saved to {results_dir}/hyperparameter_results.csv")


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
