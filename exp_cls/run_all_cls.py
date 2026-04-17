"""One-click classification experiment runner.

Phases:
  1. Encoder grid search (CAE, IPCAE) — encoder_cls.yaml
  2. Gate-only experiments (STG, GSG-Sigmoid, GSG-Softmax, HCG) — gate_cls.yaml
  3. Combined experiments (6 models) — combined_cls.yaml
  4. Downstream evaluation on selected features

Usage:
  python -m exp_cls.run_all_cls --device cuda
  python -m exp_cls.run_all_cls --phase encoder --device cpu
  python -m exp_cls.run_all_cls --phase downstream --downstream-epochs 200
"""

from __future__ import annotations

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import yaml

from exp_cls.run_experiment import (
    run_encoder,
    run_gate,
    run_combined,
)
from exp_cls.run_downstream import run_downstream

PHASE_CONFIGS = {
    "encoder": os.path.join(os.path.dirname(__file__), "configs", "encoder_cls.yaml"),
    "gate": os.path.join(os.path.dirname(__file__), "configs", "gate_cls.yaml"),
    "combined": os.path.join(os.path.dirname(__file__), "configs", "combined_cls.yaml"),
}


def load_config(config_path, device=None):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if device:
        config["training"]["device"] = device
    config["training"]["lr"] = float(config["training"]["lr"])
    config["training"]["epochs"] = int(config["training"]["epochs"])
    config["training"]["batch_size"] = int(config["training"]["batch_size"])
    return config


def run_phase(phase, config, results_dir, device):
    import pandas as pd

    task = config["task"]
    training_cfg = config["training"]
    head_cfg = config.get("head", config.get("classifier", {"hidden_dim": 128}))
    seeds = config.get("seeds", [0])

    print(f"\n{'#' * 70}")
    print(f"  PHASE: {phase.upper()}")
    print(f"  Datasets: {config['datasets']} | Seeds: {seeds} | Device: {device}")
    print(f"{'#' * 70}")

    phase_results = []
    t0 = time.time()
    for dataset_name in config["datasets"]:
        for seed in seeds:
            print(f"\n{'=' * 60}")
            print(f"  {phase.upper()} | Dataset: {dataset_name} | Seed: {seed}")
            print(f"{'=' * 60}")
            if phase == "encoder":
                df = run_encoder(
                    config,
                    dataset_name,
                    seed,
                    results_dir,
                    task,
                    training_cfg,
                    head_cfg,
                    skip_completed=True,
                )
            elif phase == "gate":
                df = run_gate(
                    config,
                    dataset_name,
                    seed,
                    results_dir,
                    task,
                    training_cfg,
                    head_cfg,
                    skip_completed=True,
                )
            elif phase == "combined":
                df = run_combined(
                    config,
                    dataset_name,
                    seed,
                    results_dir,
                    task,
                    training_cfg,
                    head_cfg,
                    skip_completed=True,
                )
            if not df.empty:
                phase_results.append(df)

    elapsed = time.time() - t0
    if phase_results:
        phase_df = pd.concat(phase_results, ignore_index=True)
        phase_df.to_csv(
            os.path.join(results_dir, f"{phase}_all_results.csv"), index=False
        )

        metric_col = {
            "classification": "accuracy",
            "regression": "r2",
            "reconstruction": "neg_mse",
        }[task]
        last_epoch = phase_df["epoch"].max()
        summary = (
            phase_df[phase_df["epoch"] == last_epoch]
            .groupby(["model", "dataset"])
            .agg(
                metric_mean=(metric_col, "mean"),
                metric_std=(metric_col, "std"),
                num_selected_mean=("num_selected", "mean"),
                training_time_mean=("training_time_sec", "mean"),
            )
            .reset_index()
        )
        summary.to_csv(os.path.join(results_dir, f"{phase}_summary.csv"), index=False)
        print(f"\n  Phase {phase} completed in {elapsed:.1f}s")
        print(f"  Results: {results_dir}/{phase}_all_results.csv")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="One-click classification experiments")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--phase",
        default="all",
        choices=["encoder", "gate", "combined", "downstream", "all"],
    )
    parser.add_argument("--downstream-epochs", type=int, default=100)
    args = parser.parse_args()

    phases = (
        ["encoder", "gate", "combined", "downstream"]
        if args.phase == "all"
        else [args.phase]
    )

    results_dir = None
    for phase in phases:
        if phase == "downstream":
            if results_dir is None:
                results_dir = os.path.join(os.path.dirname(__file__), "results")
            run_downstream(results_dir, epochs=args.downstream_epochs)
            continue

        config = load_config(PHASE_CONFIGS[phase], device=args.device)
        results_dir = config.get(
            "results_dir", os.path.join(os.path.dirname(__file__), "results")
        )
        os.makedirs(results_dir, exist_ok=True)
        run_phase(phase, config, results_dir, args.device)

    print(f"\n{'#' * 70}")
    print(f"  ALL DONE. Results in: {results_dir}")
    print(f"{'#' * 70}")


if __name__ == "__main__":
    main()
