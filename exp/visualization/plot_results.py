from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_curves(df: pd.DataFrame, output_dir: str, group_by: str = "model"):
    os.makedirs(output_dir, exist_ok=True)
    for name, group in df.groupby(group_by):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for seed_df in group.groupby("seed"):
            seed_label = f"seed={seed_df[0]}"
            axes[0].plot(seed_df[1]["epoch"], seed_df[1]["loss_cls"], label=seed_label)
            axes[1].plot(seed_df[1]["epoch"], seed_df[1]["accuracy"], label=seed_label)
            if "loss_sparsity" in seed_df[1].columns:
                axes[2].plot(seed_df[1]["epoch"], seed_df[1]["loss_sparsity"], label=seed_label)
        axes[0].set_title("Classification Loss")
        axes[0].set_xlabel("Epoch")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[2].set_title("Sparsity Loss")
        axes[2].set_xlabel("Epoch")
        for ax in axes:
            ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name}_training_curves.png"), dpi=150)
        plt.close()


def plot_accuracy_bar(df: pd.DataFrame, output_dir: str, metric: str = "accuracy"):
    os.makedirs(output_dir, exist_ok=True)
    final_epoch = df[df["epoch"] == df["epoch"].max()]
    grouped = final_epoch.groupby(["dataset", "model"])[metric].mean().reset_index()
    for dataset in grouped["dataset"].unique():
        subset = grouped[grouped["dataset"] == dataset]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(subset["model"], subset[metric])
        ax.set_title(f"{metric} by model on {dataset}")
        ax.set_ylabel(metric)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{dataset}_{metric}_bar.png"), dpi=150)
        plt.close()


def plot_feature_count(df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    final_epoch = df[df["epoch"] == df["epoch"].max()]
    if "num_selected" not in final_epoch.columns:
        return
    grouped = final_epoch.groupby(["dataset", "model"])["num_selected"].mean().reset_index()
    for dataset in grouped["dataset"].unique():
        subset = grouped[grouped["dataset"] == dataset]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(subset["model"], subset["num_selected"])
        ax.set_title(f"Selected features by model on {dataset}")
        ax.set_ylabel("Number of features")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{dataset}_features_bar.png"), dpi=150)
        plt.close()


def plot_ablation(df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for ablation_name in df["ablation"].unique():
        subset = df[df["ablation"] == ablation_name]
        for dataset in subset["dataset"].unique():
            ds_subset = subset[subset["dataset"] == dataset]
            final = ds_subset[ds_subset["epoch"] == ds_subset["epoch"].max()]
            mean_by_param = final.groupby("param_value")["accuracy"].agg(["mean", "std"]).reset_index()
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.errorbar(mean_by_param["param_value"], mean_by_param["mean"],
                       yerr=mean_by_param["std"], marker="o", capsize=3)
            ax.set_title(f"{ablation_name} on {dataset}")
            ax.set_xlabel(ablation_name)
            ax.set_ylabel("Accuracy")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"ablation_{ablation_name}_{dataset}.png"), dpi=150)
            plt.close()
