"""Generate comprehensive comparison figures.

Main plot: CAE / IPCAE grid search curves (19 k values) with gate method points overlaid.
Subplots: convergence, threshold sensitivity, stability.

Usage:
  python exp/run_visualization.py
"""
from __future__ import annotations

import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.size": 10,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "axes.titleweight": "bold",
    "font.family": "sans-serif",
})

ENCODER_CFG = {
    "CAE":   {"color": "#E74C3C", "label": "CAE grid search"},
    "IPCAE": {"color": "#2ECC71", "label": "IPCAE grid search"},
}
GATE_CFG = {
    "GSG-Softmax+CAE":   {"marker": "P", "color": "#E74C3C", "ec": "darkred",   "label": "GSG-Softmax+CAE"},
    "GSG-Softmax+IPCAE": {"marker": "*", "color": "#2ECC71", "ec": "darkgreen",  "label": "GSG-Softmax+IPCAE (Ours)"},
    "HCG+CAE":           {"marker": "X", "color": "#E74C3C", "ec": "darkred",   "label": "HCG+CAE"},
    "HCG+IPCAE":         {"marker": "p", "color": "#2ECC71", "ec": "darkgreen",  "label": "HCG+IPCAE"},
}
BASELINE_CFG = {
    "LASSO": {"marker": "D", "color": "#8E44AD"},
    "RF":    {"marker": "d", "color": "#F39C12"},
}


def _enc(model):
    if "IPCAE" in model: return "IPCAE"
    if "CAE" in model: return "CAE"
    return None


def _plot_main(ax, grid_df, combined_df, metric, baseline_df, threshold_df):
    for enc in ["CAE", "IPCAE"]:
        cfg = ENCODER_CFG[enc]
        enc_df = grid_df[grid_df["model"] == enc]
        if len(enc_df) == 0:
            continue

        per_k = enc_df.groupby("k").agg(
            m_mean=(metric, "mean"), m_std=(metric, "std"),
            f_mean=("num_selected", "mean"), f_std=("num_selected", "std"),
        ).reset_index().sort_values("f_mean")

        for _, row in enc_df.iterrows():
            ax.scatter(row["num_selected"], row[metric],
                       color=cfg["color"], s=20, alpha=0.35, zorder=2)

        ax.fill_between(per_k["f_mean"],
                         per_k["m_mean"] - per_k["m_std"],
                         per_k["m_mean"] + per_k["m_std"],
                         alpha=0.10, color=cfg["color"], zorder=1)

        ax.plot(per_k["f_mean"], per_k["m_mean"],
                color=cfg["color"], linewidth=2.5, zorder=3,
                marker="o", markersize=6, markeredgecolor="black",
                markeredgewidth=0.8, label=cfg["label"])

    if combined_df is not None and len(combined_df) > 0:
        for gmodel, gcfg in GATE_CFG.items():
            gdf = combined_df[combined_df["model"] == gmodel]
            if len(gdf) == 0:
                continue
            for sw, sw_g in gdf.groupby("sparse_weight"):
                per_sw = sw_g.groupby("k").agg(
                    m_mean=(metric, "mean"), m_std=(metric, "std"),
                    f_mean=("num_selected", "mean"),
                ).reset_index()
                suffix = f" sw={sw}" if len(gdf["sparse_weight"].unique()) > 1 else ""
                for _, row in per_sw.iterrows():
                    ax.scatter(row["f_mean"], row["m_mean"],
                               marker=gcfg["marker"], s=200, color=gcfg["color"],
                               edgecolors=gcfg["ec"], linewidths=2, zorder=6,
                               label=gcfg["label"] + suffix)
                    ax.errorbar(row["f_mean"], row["m_mean"], yerr=row["m_std"],
                                fmt="none", ecolor=gcfg["ec"], capsize=3,
                                elinewidth=1.2, zorder=5)

    if baseline_df is not None and len(baseline_df) > 0:
        seen = set()
        for _, row in baseline_df.iterrows():
            method = row["method"]
            bcfg = BASELINE_CFG.get(method)
            if bcfg is None:
                continue
            lbl = f'{method}({int(row["n_features"])})' if method not in seen else None
            ax.scatter(row["n_features"], row[metric],
                       marker=bcfg["marker"], s=100, color=bcfg["color"],
                       edgecolors="black", linewidths=0.6, zorder=4, label=lbl)
            seen.add(method)

    if threshold_df is not None and len(threshold_df) > 0:
        for mname in ["STG", "HCG"]:
            sub = threshold_df[threshold_df["model"] == mname]
            if len(sub) > 1:
                sub_s = sub.sort_values("n_features")
                c = "#9C27B0" if mname == "STG" else "#E91E63"
                ax.plot(sub_s["n_features"], sub_s[metric],
                        color=c, linewidth=1, alpha=0.4, zorder=1, linestyle=":")
                best_i = sub_s[metric].idxmax()
                best_r = sub_s.loc[best_i]
                ax.annotate(mname, (best_r["n_features"], best_r[metric]),
                            textcoords="offset points", xytext=(4, 4),
                            fontsize=7, color=c, alpha=0.7)

    ax.set_xlabel("Number of Selected Features", fontsize=11)
    ylabel = metric.replace("_", " ").title()
    if metric == "r2":
        ylabel = "R\u00b2"
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.15)


def _plot_convergence(ax, all_df, metric, grid_df, top_n=4):
    last = all_df.sort_values("epoch").groupby(["model","k","seed"]).last().reset_index()
    summary = last.groupby(["model","k"]).agg(m=(metric,"mean")).reset_index()
    top = summary.nlargest(top_n, "m")

    for _, row in top.iterrows():
        model, k = row["model"], row["k"]
        mask = (all_df["model"]==model) & (all_df["k"]==k)
        g = all_df[mask]
        mean_c = g.groupby("epoch")[metric].mean()
        c = ENCODER_CFG.get(_enc(model), {}).get("color", "#888")
        ls = "-" if "GSG-Softmax" in model else ("--" if "HCG" in model else ":")
        ax.plot(mean_c.index, mean_c.values, linewidth=1.5,
                label=f"{model} k={int(k)}", color=c, linestyle=ls)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_"," ").title() if metric != "r2" else "R\u00b2")
    ax.set_title("(B) Convergence", fontsize=10)
    ax.legend(fontsize=6, loc="lower right")
    ax.grid(True, alpha=0.15)


def _plot_threshold(ax, threshold_df, metric):
    if threshold_df is None or len(threshold_df) == 0:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(C) Threshold Sensitivity", fontsize=10)
        return
    for mname, g in threshold_df.groupby("model"):
        if mname == "GSG-Softmax":
            ax.axhline(y=g[metric].mean(), color="#27AE60", ls="--",
                       linewidth=2, alpha=0.7, label="GSG-Softmax (no thresh)")
        else:
            for sw, sw_g in g.groupby("sparse_weight"):
                sw_s = sw_g.sort_values("threshold")
                c = "#9C27B0" if mname == "STG" else "#E91E63"
                ax.plot(sw_s["threshold"], sw_s[metric], marker="o", markersize=4,
                        linewidth=1.3, label=f"{mname} sw={sw}", color=c, alpha=0.8)
    ax.set_xlabel("Activation Threshold")
    ax.set_ylabel(metric.replace("_"," ").title() if metric != "r2" else "R\u00b2")
    ax.set_title("(C) Threshold Sensitivity\n(GSG-Softmax: no threshold)", fontsize=10)
    ax.legend(fontsize=6, loc="lower left")
    ax.grid(True, alpha=0.15)


def _plot_stability(ax, stability_df):
    if stability_df is None or len(stability_df) == 0:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(D) Feature Stability", fontsize=10)
        return
    stab = stability_df.copy()
    stab["label"] = stab["model"] + stab["k"].apply(
        lambda x: f" k={int(x)}" if pd.notna(x) else "")
    stab = stab.sort_values("jaccard_mean")
    colors = [ENCODER_CFG.get(_enc(m), {}).get("color", "#888") for m in stab["model"]]
    ax.barh(range(len(stab)), stab["jaccard_mean"],
            xerr=stab["jaccard_std"], color=colors,
            capsize=2, edgecolor="black", linewidth=0.3, alpha=0.8)
    for i, (_, row) in enumerate(stab.iterrows()):
        ax.text(row["jaccard_mean"]+0.01, i,
                f'{row["jaccard_mean"]:.2f} ({int(row["avg_features_per_seed"])}f)',
                va="center", fontsize=6)
    ax.set_yticks(range(len(stab)))
    ax.set_yticklabels(stab["label"], fontsize=6)
    ax.set_xlabel("Jaccard Index")
    ax.set_title("(D) Feature Stability (3 seeds)", fontsize=10)
    ax.set_xlim(0, 1.1)
    ax.grid(True, alpha=0.15, axis="x")


def make_figure(grid_name, grid_metric, combined_sources, title, 
                stability_df=None, threshold_df=None, baseline_df=None):
    grid_df = pd.read_csv(f"exp/results/{grid_name}/all_results.csv")
    grid_last = grid_df.sort_values("epoch").groupby(["model","k","seed"]).last().reset_index()

    _comb_frames = []
    for src in combined_sources:
        p = f"exp/results/{src}/all_results.csv"
        if os.path.exists(p):
            _tmp = pd.read_csv(p)
            _comb_frames.append(_tmp)
    if _comb_frames:
        _comb_full = pd.concat(_comb_frames, ignore_index=True)
        _comb_models = set(GATE_CFG.keys())
        _comb_sub = _comb_full[_comb_full["model"].isin(_comb_models)]
        if len(_comb_sub) > 0:
            combined_last = _comb_sub.sort_values("epoch").groupby(
                ["model", "k", "seed", "sparse_weight"]).last().reset_index()
        else:
            combined_last = None
    else:
        combined_last = None

    has_sub = any(x is not None and len(x) > 0 for x in [threshold_df, stability_df])

    if has_sub:
        fig = plt.figure(figsize=(16, 8))
        gs = gridspec.GridSpec(3, 2, width_ratios=[1.4, 1], hspace=0.45, wspace=0.30)
        ax_main = fig.add_subplot(gs[:, 0])
        ax_conv = fig.add_subplot(gs[0, 1])
        ax_thresh = fig.add_subplot(gs[1, 1])
        ax_stab = fig.add_subplot(gs[2, 1])
    else:
        fig, ax_main = plt.subplots(figsize=(10, 6))
        ax_conv = ax_thresh = ax_stab = None

    _plot_main(ax_main, grid_last, combined_last, grid_metric, baseline_df, threshold_df)
    ax_main.set_title("(A) Performance vs Feature Count", fontsize=11)

    handles, labels = ax_main.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_main.legend(by_label.values(), by_label.keys(), fontsize=7, loc="lower right")

    if ax_conv is not None:
        all_df = pd.read_csv(f"exp/results/{grid_name}/all_results.csv")
        if combined_last is not None:
            _conv_frames = [all_df]
            for src in combined_sources:
                p = f"exp/results/{src}/all_results.csv"
                if os.path.exists(p):
                    _conv_frames.append(pd.read_csv(p))
            all_df = pd.concat(_conv_frames, ignore_index=True)
        _plot_convergence(ax_conv, all_df, grid_metric, grid_last)

    if ax_thresh is not None:
        _plot_threshold(ax_thresh, threshold_df, grid_metric)

    if ax_stab is not None:
        _plot_stability(ax_stab, stability_df)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out = f"exp/results/{grid_name}/figures/comprehensive_comparison.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out}")
    return grid_last, combined_last


def print_conclusion(grid_last, combined_last, metric, threshold_df, title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

    grid_sum = grid_last.groupby(["model","k"]).agg(
        m=(metric,"mean"), s=(metric,"std"), f=("num_selected","mean"),
    ).reset_index()

    for enc in ["IPCAE", "CAE"]:
        sub = grid_sum[grid_sum["model"]==enc].sort_values("k")
        best = sub.loc[sub["m"].idxmax()]
        print(f"\n  {enc} grid search peak: k={int(best['k'])} "
              f"{metric}={best['m']:.4f} ({best['f']:.0f} features)")

    if combined_last is not None and len(combined_last) > 0:
        comb_sum = combined_last.groupby(["model","k","sparse_weight"]).agg(
            m=(metric,"mean"), s=(metric,"std"), f=("num_selected","mean"),
        ).reset_index()
        print()
        for _, row in comb_sum.sort_values("m", ascending=False).iterrows():
            print(f"  {row['model']:25s} k={int(row['k']):2d} sw={row['sparse_weight']:4}  "
                  f"{metric}={row['m']:.4f}+/-{row['s']:.4f}  feat={row['f']:.1f}")

    ours = combined_last[combined_last["model"]=="GSG-Softmax+IPCAE"] if combined_last is not None else None
    if ours is not None and len(ours) > 0:
        ours_per_k = ours.groupby(["k","sparse_weight"]).agg(
            m=(metric,"mean"), s=(metric,"std"), f=("num_selected","mean"),
        ).reset_index()
        best_ours = ours_per_k.loc[ours_per_k["m"].idxmax()]
        ipcae_peak = grid_sum[grid_sum["model"]=="IPCAE"].loc[grid_sum[grid_sum["model"]=="IPCAE"]["m"].idxmax()]
        print(f"\n  >>> GSG-Softmax+IPCAE (Ours) best: k={int(best_ours['k'])} sw={best_ours['sparse_weight']}")
        print(f"      {metric}={best_ours['m']:.4f}+/-{best_ours['s']:.4f}, "
              f"{best_ours['f']:.0f} features")
        print(f"      IPCAE grid peak: {metric}={ipcae_peak['m']:.4f}, "
              f"{ipcae_peak['f']:.0f} features (k={int(ipcae_peak['k'])})")
        ratio = best_ours['m']/ipcae_peak['m']*100 if ipcae_peak['m'] != 0 else 0
        print(f"      -> Ours auto-selects ~{best_ours['f']:.0f} features, "
              f"achieving {ratio:.1f}% of optimal grid search")

    if threshold_df is not None:
        for mname in ["STG", "HCG"]:
            sub = threshold_df[threshold_df["model"]==mname]
            if len(sub) > 1:
                rng = sub[metric].max() - sub[metric].min()
                print(f"  {mname} threshold impact: {rng:.4f} swing")


def main():
    cls_stab = pd.read_csv("exp/results/cls_grid/stability_results.csv") \
        if os.path.exists("exp/results/cls_grid/stability_results.csv") else None
    cls_thresh = pd.read_csv("exp/results/cls_grid/threshold_sensitivity.csv") \
        if os.path.exists("exp/results/cls_grid/threshold_sensitivity.csv") else None
    cls_bl = pd.read_csv("exp/results/cls_grid/baseline_results.csv") \
        if os.path.exists("exp/results/cls_grid/baseline_results.csv") else None
    reg_bl = pd.read_csv("exp/results/reg_grid/baseline_results.csv") \
        if os.path.exists("exp/results/reg_grid/baseline_results.csv") else None

    if cls_thresh is None and os.path.exists("exp/results/cls_digits/threshold_sensitivity.csv"):
        cls_thresh = pd.read_csv("exp/results/cls_digits/threshold_sensitivity.csv")
    if cls_stab is None and os.path.exists("exp/results/cls_digits/stability_results.csv"):
        cls_stab = pd.read_csv("exp/results/cls_digits/stability_results.csv")
    if cls_bl is None and os.path.exists("exp/results/cls_digits/baseline_results.csv"):
        cls_bl = pd.read_csv("exp/results/cls_digits/baseline_results.csv")
    if reg_bl is None and os.path.exists("exp/results/reg_sklearn/baseline_results.csv"):
        reg_bl = pd.read_csv("exp/results/reg_sklearn/baseline_results.csv")

    print("--- Classification ---")
    g, c = make_figure("cls_grid", "accuracy",
                        ["cls_digits", "cls_sw_ext"],
                        "Classification (Digits, 64 features, 10 classes)",
                        stability_df=cls_stab, threshold_df=cls_thresh, baseline_df=cls_bl)
    print_conclusion(g, c, "accuracy", cls_thresh,
                      "Classification (Digits)")

    print("\n--- Regression ---")
    g, c = make_figure("reg_grid", "r2",
                        ["reg_sklearn", "reg_sw_ext"],
                        "Regression (100 features, 30 informative)",
                        baseline_df=reg_bl)
    print_conclusion(g, c, "r2", None, "Regression")

    print("\nDone.")


if __name__ == "__main__":
    main()
