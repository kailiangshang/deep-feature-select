from __future__ import annotations

import os

import pandas as pd


def generate_comparison_table(df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    final_epoch = df[df["epoch"] == df["epoch"].max()]
    grouped = final_epoch.groupby(["dataset", "model"]).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        num_selected_mean=("num_selected", "mean"),
    ).reset_index()
    grouped["accuracy_str"] = grouped.apply(
        lambda r: f"{r['accuracy_mean']:.3f} ± {r['accuracy_std']:.3f}", axis=1
    )
    pivot = grouped.pivot(index="model", columns="dataset", values="accuracy_str")
    pivot = pivot.fillna("-")
    latex = pivot.to_latex(escape=False)
    with open(os.path.join(output_dir, "comparison_table.tex"), "w") as f:
        f.write(latex)
    pivot.to_csv(os.path.join(output_dir, "comparison_table.csv"))
    return pivot


def generate_ablation_table(df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    final_epoch = df[df["epoch"] == df["epoch"].max()]
    tables = {}
    for ablation_name in final_epoch["ablation"].unique():
        subset = final_epoch[final_epoch["ablation"] == ablation_name]
        grouped = subset.groupby(["dataset", "param_value"]).agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
        ).reset_index()
        grouped["accuracy_str"] = grouped.apply(
            lambda r: f"{r['accuracy_mean']:.3f} ± {r['accuracy_std']:.3f}", axis=1
        )
        pivot = grouped.pivot(index="param_value", columns="dataset", values="accuracy_str")
        pivot = pivot.fillna("-")
        latex = pivot.to_latex(escape=False)
        with open(os.path.join(output_dir, f"ablation_{ablation_name}.tex"), "w") as f:
            f.write(latex)
        tables[ablation_name] = pivot
    return tables
