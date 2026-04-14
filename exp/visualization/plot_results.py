from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_training_curves(
    df: pd.DataFrame,
    output_dir: str,
    group_by: str = "model",
    loss_col: str = "loss_task",
    metric_col: str = "accuracy",
):
    os.makedirs(output_dir, exist_ok=True)
    for name, group in df.groupby(group_by):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for seed_val, seed_df in group.groupby("seed") if "seed" in group.columns else [(None, group)]:
            label = f"seed={seed_val}" if seed_val is not None else ""
            if loss_col in seed_df.columns:
                axes[0].plot(seed_df["epoch"], seed_df[loss_col], label=label)
            if metric_col in seed_df.columns:
                axes[1].plot(seed_df["epoch"], seed_df[metric_col], label=label)
            if "loss_sparsity" in seed_df.columns:
                axes[2].plot(seed_df["epoch"], seed_df["loss_sparsity"], label=label)
        axes[0].set_title("Task Loss")
        axes[0].set_xlabel("Epoch")
        axes[1].set_title(metric_col.replace("_", " ").title())
        axes[1].set_xlabel("Epoch")
        axes[2].set_title("Sparsity Loss")
        axes[2].set_xlabel("Epoch")
        for ax in axes:
            if ax.get_legend_handles_labels()[1]:
                ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name}_training_curves.png"), dpi=150)
        plt.close()


def plot_metric_bar(df: pd.DataFrame, output_dir: str, metric: str = "accuracy"):
    os.makedirs(output_dir, exist_ok=True)
    final_epoch = df[df["epoch"] == df["epoch"].max()]
    group_cols = [c for c in ["dataset", "model"] if c in final_epoch.columns]
    if not group_cols:
        return
    grouped = final_epoch.groupby(group_cols)[metric].mean().reset_index()
    for dataset in grouped["dataset"].unique() if "dataset" in grouped.columns else [None]:
        subset = grouped[grouped["dataset"] == dataset] if dataset else grouped
        fig, ax = plt.subplots(figsize=(max(6, len(subset) * 0.8), 5))
        ax.bar(subset["model"], subset[metric])
        title = f"{metric} by model"
        if dataset:
            title += f" on {dataset}"
        ax.set_title(title)
        ax.set_ylabel(metric)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fname = f"{dataset}_{metric}_bar.png" if dataset else f"{metric}_bar.png"
        plt.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close()


def plot_feature_count(df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    if "num_selected" not in df.columns:
        return
    final_epoch = df[df["epoch"] == df["epoch"].max()]
    group_cols = [c for c in ["dataset", "model"] if c in final_epoch.columns]
    if not group_cols:
        return
    grouped = final_epoch.groupby(group_cols)["num_selected"].mean().reset_index()
    for dataset in grouped["dataset"].unique() if "dataset" in grouped.columns else [None]:
        subset = grouped[grouped["dataset"] == dataset] if dataset else grouped
        fig, ax = plt.subplots(figsize=(max(6, len(subset) * 0.8), 5))
        ax.bar(subset["model"], subset["num_selected"])
        title = "Selected features by model"
        if dataset:
            title += f" on {dataset}"
        ax.set_title(title)
        ax.set_ylabel("Number of features")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fname = f"{dataset}_features_bar.png" if dataset else "features_bar.png"
        plt.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close()


def plot_ablation(df: pd.DataFrame, output_dir: str, metric: str = "accuracy"):
    os.makedirs(output_dir, exist_ok=True)
    if "ablation" not in df.columns:
        return
    for ablation_name in df["ablation"].unique():
        subset = df[df["ablation"] == ablation_name]
        for dataset in subset["dataset"].unique() if "dataset" in subset.columns else [None]:
            ds_subset = subset[subset["dataset"] == dataset] if dataset else subset
            final = ds_subset[ds_subset["epoch"] == ds_subset["epoch"].max()]
            mean_by_param = final.groupby("param_value")[metric].agg(["mean", "std"]).reset_index()
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.errorbar(
                range(len(mean_by_param)),
                mean_by_param["mean"],
                yerr=mean_by_param["std"],
                marker="o",
                capsize=3,
            )
            ax.set_xticks(range(len(mean_by_param)))
            ax.set_xticklabels(mean_by_param["param_value"], rotation=45, ha="right")
            title = f"{ablation_name}"
            if dataset:
                title += f" on {dataset}"
            ax.set_title(title)
            ax.set_xlabel(ablation_name)
            ax.set_ylabel(metric)
            plt.tight_layout()
            fname = f"ablation_{ablation_name}_{dataset}.png" if dataset else f"ablation_{ablation_name}.png"
            plt.savefig(os.path.join(output_dir, fname), dpi=150)
            plt.close()


def generate_sample_plots(output_dir: str = "exp/results/sample_figures"):
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)
    epochs = np.arange(1, 101)
    models = ["CAE", "IPCAE", "STG", "GSG-Softmax", "GSG-Softmax+IPCAE"]
    records = []
    for model in models:
        for seed in [0, 1, 2]:
            base_acc = {"CAE": 0.7, "IPCAE": 0.72, "STG": 0.68, "GSG-Softmax": 0.71, "GSG-Softmax+IPCAE": 0.78}[model]
            acc_curve = base_acc - 0.3 * np.exp(-epochs / 15) + np.random.normal(0, 0.01, len(epochs))
            acc_curve = np.clip(acc_curve, 0, 1)
            loss_curve = 2.0 * np.exp(-epochs / 20) + np.random.normal(0, 0.02, len(epochs))
            sparse_curve = 0.5 * np.exp(-epochs / 30) + np.random.normal(0, 0.01, len(epochs))
            for i, epoch in enumerate(epochs):
                records.append({
                    "epoch": epoch,
                    "loss_task": loss_curve[i],
                    "accuracy": acc_curve[i],
                    "loss_sparsity": sparse_curve[i] if model in ["STG", "GSG-Softmax", "GSG-Softmax+IPCAE"] else 0,
                    "num_selected": np.random.randint(20, 60),
                    "model": model,
                    "seed": seed,
                    "dataset": "SampleData",
                })
    df = pd.DataFrame(records)
    plot_training_curves(df, output_dir, metric_col="accuracy")
    plot_metric_bar(df, output_dir, metric="accuracy")
    plot_feature_count(df, output_dir)
    print(f"Sample figures generated in {output_dir}/")
    print(f"  - *_training_curves.png (per model)")
    print(f"  - SampleData_accuracy_bar.png")
    print(f"  - SampleData_features_bar.png")
    return output_dir
