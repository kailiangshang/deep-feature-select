"""One-click classification experiment runner.

Phases:
  1. Encoder grid search (CAE, IPCAE)
  2. Gate-only experiments (STG, GSG-Sigmoid, GSG-Softmax, HCG)
  3. Combined experiments (6 models)
  4. Downstream evaluation on selected features

Usage:
  # Single GPU
  python -m exp_cls.run_all_cls --device cuda:0

  # Multi-GPU parallel (auto-detect all GPUs)
  python -m exp_cls.run_all_cls --gpus all

  # Specify GPUs
  python -m exp_cls.run_all_cls --gpus 0,1,2,3

  # Single phase
  python -m exp_cls.run_all_cls --phase encoder --gpus 0,1
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
import warnings
from functools import partial

warnings.filterwarnings("ignore")

import yaml

from exp_cls.run_experiment import run_encoder, run_gate, run_combined
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


def _worker_fn(phase, config, results_dir, task, training_cfg, head_cfg, device, ds_seed):
    dataset_name, seed = ds_seed
    pid = os.getpid()
    print(f"[GPU {device} | PID {pid}] {phase.upper()} | {dataset_name} | seed={seed}")
    training_cfg_dev = {**training_cfg, "device": device}
    if phase == "encoder":
        return run_encoder(config, dataset_name, seed, results_dir, task, training_cfg_dev, head_cfg, skip_completed=True)
    elif phase == "gate":
        return run_gate(config, dataset_name, seed, results_dir, task, training_cfg_dev, head_cfg, skip_completed=True)
    elif phase == "combined":
        return run_combined(config, dataset_name, seed, results_dir, task, training_cfg_dev, head_cfg, skip_completed=True)


def _save_phase_summary(phase, phase_results, results_dir, task):
    import pandas as pd
    if not phase_results:
        print(f"  Phase {phase}: no results")
        return
    phase_df = pd.concat(phase_results, ignore_index=True)
    phase_df.to_csv(os.path.join(results_dir, f"{phase}_all_results.csv"), index=False)
    metric_col = {"classification": "accuracy", "regression": "r2", "reconstruction": "neg_mse"}[task]
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


def run_phase_parallel(phase, config, results_dir, gpu_ids):
    task = config["task"]
    training_cfg = config["training"]
    head_cfg = config.get("head", config.get("classifier", {"hidden_dim": 128}))
    seeds = config.get("seeds", [0])
    datasets = config["datasets"]

    work_items = [(ds, s) for ds in datasets for s in seeds]
    n_gpus = len(gpu_ids)

    print(f"\n{'#' * 70}")
    print(f"  PHASE: {phase.upper()}")
    print(f"  Datasets: {datasets} | Seeds: {seeds}")
    print(f"  GPUs: {gpu_ids} | Work items: {len(work_items)} | Parallelism: {n_gpus}")
    print(f"{'#' * 70}")

    t0 = time.time()

    training_cfg_no_dev = {k: v for k, v in training_cfg.items() if k != "device"}
    worker = partial(
        _worker_fn, phase, config, results_dir, task, training_cfg_no_dev, head_cfg,
    )

    pool = mp.Pool(processes=n_gpus)
    async_results = []
    for i, item in enumerate(work_items):
        gpu = gpu_ids[i % n_gpus]
        ar = pool.apply_async(worker, args=(gpu, item))
        async_results.append(ar)
    pool.close()
    pool.join()

    phase_results = []
    for ar in async_results:
        df = ar.get()
        if not df.empty:
            phase_results.append(df)

    elapsed = time.time() - t0
    _save_phase_summary(phase, phase_results, results_dir, task)
    print(f"\n  Phase {phase} completed in {elapsed:.1f}s ({len(phase_results)} groups)")


def run_phase_serial(phase, config, results_dir, device):
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
                df = run_encoder(config, dataset_name, seed, results_dir, task, training_cfg, head_cfg, skip_completed=True)
            elif phase == "gate":
                df = run_gate(config, dataset_name, seed, results_dir, task, training_cfg, head_cfg, skip_completed=True)
            elif phase == "combined":
                df = run_combined(config, dataset_name, seed, results_dir, task, training_cfg, head_cfg, skip_completed=True)
            if not df.empty:
                phase_results.append(df)

    elapsed = time.time() - t0
    _save_phase_summary(phase, phase_results, results_dir, task)
    print(f"\n  Phase {phase} completed in {elapsed:.1f}s ({len(phase_results)} groups)")


def parse_gpus(gpus_arg):
    import torch
    if gpus_arg == "all":
        n = torch.cuda.device_count()
        if n == 0:
            raise RuntimeError("No GPUs found")
        return [f"cuda:{i}" for i in range(n)]
    ids = [int(x.strip()) for x in gpus_arg.split(",")]
    return [f"cuda:{i}" for i in ids]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="One-click classification experiments")
    parser.add_argument("--device", default=None, help="Single device (cpu, cuda:0, etc.)")
    parser.add_argument("--gpus", default=None, help="Multi-GPU: 'all' or comma-separated ids (e.g. '0,1,2,3')")
    parser.add_argument("--phase", default="all", choices=["encoder", "gate", "combined", "downstream", "all"])
    parser.add_argument("--downstream-epochs", type=int, default=100)
    args = parser.parse_args()

    if args.gpus:
        gpu_ids = parse_gpus(args.gpus)
        print(f"Multi-GPU mode: {gpu_ids}")
    elif args.device:
        gpu_ids = []
    else:
        args.device = "cpu"

    phases = ["encoder", "gate", "combined", "downstream"] if args.phase == "all" else [args.phase]

    results_dir = None
    for phase in phases:
        if phase == "downstream":
            if results_dir is None:
                results_dir = os.path.join(os.path.dirname(__file__), "results")
            run_downstream(results_dir, epochs=args.downstream_epochs)
            continue

        config = load_config(PHASE_CONFIGS[phase], device=args.device)
        results_dir = config.get("results_dir", os.path.join(os.path.dirname(__file__), "results"))
        os.makedirs(results_dir, exist_ok=True)

        if gpu_ids:
            mp.set_start_method("spawn", force=True)
            run_phase_parallel(phase, config, results_dir, gpu_ids)
        else:
            run_phase_serial(phase, config, results_dir, args.device)

    print(f"\n{'#' * 70}")
    print(f"  ALL DONE. Results in: {results_dir}")
    print(f"{'#' * 70}")


if __name__ == "__main__":
    main()
