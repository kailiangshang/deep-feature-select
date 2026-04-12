"""Generate all figures from experiment results.

Usage:
  python exp/run_visualization.py
"""
from __future__ import annotations

import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "axes.titleweight": "bold",
})


def plot_main_comparison(df, metric, output_dir, title_prefix=""):
    os.makedirs(output_dir, exist_ok=True)
    last = df[df["epoch"] == df["epoch"].max()]

    group_cols = ["model"]
    extra_cols = [c for c in ["k"] if c in last.columns]
    group_key = group_cols + extra_cols

    grouped = last.groupby(group_key).agg(
        metric_mean=(metric, "mean"),
        metric_std=(metric, "std"),
        feat_mean=("num_selected", "mean"),
    ).reset_index()

    grouped["label"] = grouped.apply(
        lambda r: f"{r['model']}\nk={int(r['k'])}" if "k" in grouped.columns else r["model"], axis=1
    )
    grouped = grouped.sort_values("metric_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(max(6, len(grouped) * 0.7), 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(grouped)))
    bars = ax.barh(range(len(grouped)), grouped["metric_mean"],
                   xerr=grouped["metric_std"], color=colors, capsize=3, edgecolor="black", linewidth=0.5)

    for i, (_, row) in enumerate(grouped.iterrows()):
        ax.text(row["metric_mean"] + 0.005, i, f'{row["metric_mean"]:.3f} ({int(row["feat_mean"])} feat)',
                va="center", fontsize=9)

    ax.set_yticks(range(len(grouped)))
    ax.set_yticklabels(grouped["label"])
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(f"{title_prefix}Model Comparison ({metric})")
    ax.set_xlim(grouped["metric_mean"].min() - 0.05, grouped["metric_mean"].max() + 0.08)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"main_comparison_{metric}.png"))
    plt.close()


def plot_downstream_comparison(deep_df, baseline_df, metric, output_dir, title_prefix=""):
    os.makedirs(output_dir, exist_ok=True)

    records = []
    for _, row in deep_df.iterrows():
        records.append({
            "method": row["model"],
            metric: row[metric],
            "n_features": row["n_features"],
            "type": "deep",
        })
    for _, row in baseline_df.iterrows():
        records.append({
            "method": f'{row["method"]}({int(row["n_features"])})',
            metric: row[metric],
            "n_features": row["n_features"],
            "type": "classic",
        })

    df = pd.DataFrame(records).sort_values(metric, ascending=True)

    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.6), 5))
    colors = ["#2196F3" if t == "deep" else "#FF9800" for t in df["type"]]
    ax.barh(range(len(df)), df[metric], color=colors, edgecolor="black", linewidth=0.5)

    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row[metric] + 0.003, i, f'{row[metric]:.3f} ({int(row["n_features"])} feat)',
                va="center", fontsize=9)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["method"])
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(f"{title_prefix}Downstream Evaluation (fixed MLP)")
    ax.set_xlim(0, df[metric].max() + 0.05)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#2196F3", label="Deep FS"),
                       Patch(facecolor="#FF9800", label="Classic")],
              loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"downstream_{metric}.png"))
    plt.close()


def plot_stability(stab_df, output_dir, title_prefix=""):
    os.makedirs(output_dir, exist_ok=True)
    df = stab_df.sort_values("jaccard_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.7), 5))
    colors = plt.cm.RdYlGn(df["jaccard_mean"].values)
    ax.barh(range(len(df)), df["jaccard_mean"], xerr=df["jaccard_std"],
            color=colors, capsize=3, edgecolor="black", linewidth=0.5)

    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["jaccard_mean"] + 0.01, i,
                f'{row["jaccard_mean"]:.3f} (u={int(row["union_features"])}, i={int(row["intersection_features"])})',
                va="center", fontsize=9)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["model"])
    ax.set_xlabel("Jaccard Index")
    ax.set_title(f"{title_prefix}Feature Stability (3 seeds)")
    ax.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stability_jaccard.png"))
    plt.close()


def plot_threshold_sensitivity(thresh_df, metric, output_dir, title_prefix=""):
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, g in thresh_df.groupby("model"):
        if model_name == "GSG-Softmax":
            ax.axhline(y=g[metric].mean(), linestyle="--", linewidth=2,
                       color="green", label=f"{model_name} (no threshold)")
        else:
            for sw, sw_g in g.groupby("sparse_weight"):
                ax.plot(sw_g["threshold"], sw_g[metric], marker="o",
                        label=f"{model_name} (sw={sw})", linewidth=1.5)

    ax.set_xlabel("Threshold")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{title_prefix}Threshold Sensitivity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "threshold_sensitivity.png"))
    plt.close()


def plot_convergence(df, metric, output_dir, title_prefix=""):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for (model, k), g in df.groupby(["model"] if "k" not in df.columns else ["model", "k"]):
        label = f"{model}" if "k" not in df.columns else f"{model} (k={int(k)})"
        for seed, sg in g.groupby("seed"):
            ax.plot(sg["epoch"], sg[metric], alpha=0.3, linewidth=0.5)
        mean_curve = g.groupby("epoch")[metric].mean()
        ax.plot(mean_curve.index, mean_curve.values, linewidth=2, label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{title_prefix}Training Curves")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"convergence_{metric}.png"))
    plt.close()


def main():
    experiments = [
        ("cls_digits", "accuracy", "Classification (Digits)"),
        ("reg_sklearn", "r2", "Regression (Sklearn)"),
    ]

    for exp_name, metric, title in experiments:
        rdir = f"exp/results/{exp_name}"
        fig_dir = f"{rdir}/figures"
        print(f"\n{'='*50}")
        print(f"  Generating figures: {title}")
        print(f"{'='*50}")

        # 1. Main comparison bar
        all_df = pd.read_csv(f"{rdir}/all_results.csv")
        plot_main_comparison(all_df, metric, fig_dir, title_prefix=f"{title} ")

        # 2. Convergence curves
        plot_convergence(all_df, metric, fig_dir, title_prefix=f"{title} ")

        # 3. Downstream + baselines
        ds_path = f"{rdir}/downstream_results.csv"
        bl_path = f"{rdir}/baseline_results.csv"
        if os.path.exists(ds_path) and os.path.exists(bl_path):
            ds_df = pd.read_csv(ds_path)
            bl_df = pd.read_csv(bl_path)
            plot_downstream_comparison(ds_df, bl_df, metric, fig_dir, title_prefix=f"{title} ")

        # 4. Stability
        stab_path = f"{rdir}/stability_results.csv"
        if os.path.exists(stab_path):
            stab_df = pd.read_csv(stab_path)
            plot_stability(stab_df, fig_dir, title_prefix=f"{title} ")

        # 5. Threshold sensitivity
        thresh_path = f"{rdir}/threshold_sensitivity.csv"
        if os.path.exists(thresh_path):
            thresh_df = pd.read_csv(thresh_path)
            plot_threshold_sensitivity(thresh_df, metric, fig_dir, title_prefix=f"{title} ")

        # List generated figures
        if os.path.exists(fig_dir):
            figs = [f for f in os.listdir(fig_dir) if f.endswith(".png")]
            for f in sorted(figs):
                print(f"  ✓ {fig_dir}/{f}")

    print(f"\nDone! All figures generated.")


if __name__ == "__main__":
    main()
