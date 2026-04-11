from __future__ import annotations

import os
import importlib

import yaml
import torch
import pandas as pd

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

ENCODER_MODELS = {"ConcreteAutoencoderModel", "IndirectConcreteAutoencoderModel"}
GATE_ONLY_MODELS = {"StochasticGateModel", "GumbelSigmoidGateModel", "GumbelSoftmaxGateModel", "HardConcreteGateModel"}


def run_encoder_experiment(config, dataset_name, seed):
    training_cfg = config["training"]
    classifier_cfg = config["classifier"]
    data = generate_train_test_loader(
        name=dataset_name,
        batch_size=training_cfg["batch_size"],
        device=training_cfg["device"],
        random_state=seed,
    )
    results = []
    for model_cfg in config["encoder_models"]:
        model_cls = MODEL_REGISTRY[model_cfg["class"]]
        for k in model_cfg["k_values"]:
            seed_all(seed)
            params = {**model_cfg["params"], "input_dim": data.feature_dim, "output_dim": k,
                      "total_epochs": training_cfg["epochs"], "device": training_cfg["device"]}
            model = model_cls(**params)
            classifier = MLPClassifier(k, classifier_cfg["hidden_dim"], data.cls_num, training_cfg["device"])
            trainer = EncoderTrainer(model, classifier, lr=training_cfg["lr"],
                                     device=training_cfg["device"], seed=seed)
            df = trainer.fit(data.train_loader, training_cfg["epochs"], data.test_loader)
            result = model.get_selection_result()
            df["model"] = model_cfg["name"]
            df["k"] = k
            df["seed"] = seed
            df["dataset"] = dataset_name
            results.append(df)
    return pd.concat(results, ignore_index=True)


def run_gate_experiment(config, dataset_name, seed):
    training_cfg = config["training"]
    classifier_cfg = config["classifier"]
    data = generate_train_test_loader(
        name=dataset_name,
        batch_size=training_cfg["batch_size"],
        device=training_cfg["device"],
        random_state=seed,
    )
    results = []
    for model_cfg in config["gate_models"]:
        model_cls = MODEL_REGISTRY[model_cfg["class"]]
        for sw in model_cfg["sparse_loss_weights"]:
            seed_all(seed)
            params = {**model_cfg["params"], "input_dim": data.feature_dim,
                      "total_epochs": training_cfg["epochs"], "device": training_cfg["device"]}
            model = model_cls(**params)
            classifier = MLPClassifier(data.feature_dim, classifier_cfg["hidden_dim"],
                                       data.cls_num, training_cfg["device"])
            trainer = GateTrainer(model, classifier, sparse_loss_weight=sw,
                                  lr=training_cfg["lr"], device=training_cfg["device"], seed=seed)
            df = trainer.fit(data.train_loader, training_cfg["epochs"], data.test_loader)
            df["model"] = model_cfg["name"]
            df["sparse_weight"] = sw
            df["seed"] = seed
            df["dataset"] = dataset_name
            results.append(df)
    return pd.concat(results, ignore_index=True)


def run_combined_experiment(config, dataset_name, seed):
    training_cfg = config["training"]
    classifier_cfg = config["classifier"]
    data = generate_train_test_loader(
        name=dataset_name,
        batch_size=training_cfg["batch_size"],
        device=training_cfg["device"],
        random_state=seed,
    )
    results = []
    for model_cfg in config["combined_models"]:
        model_cls = MODEL_REGISTRY[model_cfg["class"]]
        for k in model_cfg["k_values"]:
            for sw in model_cfg["sparse_loss_weights"]:
                seed_all(seed)
                params = {**model_cfg["params"], "input_dim": data.feature_dim, "k": k,
                          "total_epochs": training_cfg["epochs"], "device": training_cfg["device"]}
                model = model_cls(**params)
                classifier = MLPClassifier(k, classifier_cfg["hidden_dim"],
                                           data.cls_num, training_cfg["device"])
                trainer = GateEncoderTrainer(model, classifier, sparse_loss_weight=sw,
                                             lr=training_cfg["lr"], device=training_cfg["device"], seed=seed)
                result_df, feature_df = trainer.fit(data.train_loader, training_cfg["epochs"], data.test_loader)
                result_df["model"] = model_cfg["name"]
                result_df["k"] = k
                result_df["sparse_weight"] = sw
                result_df["seed"] = seed
                result_df["dataset"] = dataset_name
                results.append(result_df)
    return pd.concat(results, ignore_index=True)


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
                df = run_encoder_experiment(config, dataset_name, seed)
                all_results.append(df)
            if args.mode in ("gate", "all"):
                df = run_gate_experiment(config, dataset_name, seed)
                all_results.append(df)
            if args.mode in ("combined", "all"):
                df = run_combined_experiment(config, dataset_name, seed)
                all_results.append(df)

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(os.path.join(results_dir, "contrast_results.csv"), index=False)
        print(f"\nResults saved to {results_dir}/contrast_results.csv")


if __name__ == "__main__":
    main()
