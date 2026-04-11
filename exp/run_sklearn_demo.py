from __future__ import annotations

import os
import json
import warnings

import numpy as np
import pandas as pd
import torch

from deepfs import (
    ConcreteAutoencoderModel,
    GumbelSoftmaxGateConcreteModel,
    GumbelSoftmaxGateIndirectConcreteModel,
    GumbelSoftmaxGateModel,
    HardConcreteGateConcreteModel,
    IndirectConcreteAutoencoderModel,
    StochasticGateModel,
)
from exp.data import generate_train_test_loader
from exp.trainers import EncoderTrainer, GateTrainer, GateEncoderTrainer
from exp.utils import MLPClassifier, MLPRegressor, AutoencoderHead, seed_all
from exp.visualization.plot_results import (
    plot_training_curves,
    plot_metric_bar,
    plot_feature_count,
    plot_ablation,
)
from exp.visualization.generate_tables import generate_comparison_table, generate_ablation_table

warnings.filterwarnings("ignore")

EPOCHS = 300
LR = 1e-3
DEVICE = "cpu"
HIDDEN = 64
RESULTS_DIR = "exp/results/sklearn_demo"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "per_group"), exist_ok=True)


def _param_count(m):
    return sum(p.numel() for p in m.parameters())


def _save_group(name, result_df, feature_df, meta):
    gdir = os.path.join(RESULTS_DIR, "per_group")
    result_df.to_csv(os.path.join(gdir, f"{name}_result.csv"), index=False)
    if feature_df is not None:
        feature_df.to_csv(os.path.join(gdir, f"{name}_features.csv"), index=False)
    with open(os.path.join(gdir, f"{name}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)


def _run_encoder(model, data, k, task, head, name, **tkwargs):
    trainer = EncoderTrainer(model, head, task=task, lr=LR, device=DEVICE, seed=0, **tkwargs)
    df = trainer.fit(data.train_loader, EPOCHS, data.test_loader)
    final = model.get_selection_result()
    feat_df = pd.DataFrame(
        [final.selected_indices],
        columns=[f"slot_{i}" for i in range(len(final.selected_indices))],
    )
    df["model"] = name
    df["seed"] = 0
    df["dataset"] = data.name
    df["task"] = task
    df["param_count"] = _param_count(model)
    metric_col = trainer.task.metric_name
    meta = {
        "model": name, "dataset": data.name, "task": task, "k": k,
        "param_count": _param_count(model),
        f"final_{metric_col}": float(df[metric_col].iloc[-1]),
        "final_num_selected": int(df["num_selected"].iloc[-1]),
    }
    _save_group(name, df, feat_df, meta)
    return df


def _run_gate(model, data, task, head, name, sw=1.0, **tkwargs):
    trainer = GateTrainer(model, head, task=task, sparse_loss_weight=sw,
                          lr=LR, device=DEVICE, seed=0, **tkwargs)
    df = trainer.fit(data.train_loader, EPOCHS, data.test_loader)
    final = model.get_selection_result()
    feat_df = pd.DataFrame(
        [final.selected_indices],
        columns=[f"slot_{i}" for i in range(len(final.selected_indices))],
    )
    df["model"] = name
    df["seed"] = 0
    df["dataset"] = data.name
    df["task"] = task
    df["sparse_weight"] = sw
    df["param_count"] = _param_count(model)
    metric_col = trainer.task.metric_name
    meta = {
        "model": name, "dataset": data.name, "task": task,
        "sparse_weight": sw, "param_count": _param_count(model),
        f"final_{metric_col}": float(df[metric_col].iloc[-1]),
        "final_num_selected": int(df["num_selected"].iloc[-1]),
    }
    _save_group(name, df, feat_df, meta)
    return df


def _run_combined(model, data, k, task, head, name, sw=1.0, **tkwargs):
    trainer = GateEncoderTrainer(model, head, task=task, sparse_loss_weight=sw,
                                 lr=LR, device=DEVICE, seed=0, **tkwargs)
    result_df, feature_df = trainer.fit(data.train_loader, EPOCHS, data.test_loader)
    metric_col = trainer.task.metric_name
    result_df["model"] = name
    result_df["seed"] = 0
    result_df["dataset"] = data.name
    result_df["task"] = task
    result_df["k"] = k
    result_df["sparse_weight"] = sw
    result_df["param_count"] = _param_count(model)
    meta = {
        "model": name, "dataset": data.name, "task": task, "k": k,
        "sparse_weight": sw, "param_count": _param_count(model),
        f"final_{metric_col}": float(result_df[metric_col].iloc[-1]),
        "final_num_selected": int(result_df["num_selected"].iloc[-1]),
        "final_gate_open_ratio": float(result_df["gate_open_ratio"].iloc[-1]),
    }
    _save_group(name, result_df, feature_df, meta)
    return result_df, feature_df


def main():
    all_cls = []
    all_reg = []
    all_rec = []
    all_ablation = []

    # ==================================================================
    # CLASSIFICATION (load_digits: 1797 × 64, 10 classes)
    # ==================================================================
    print("\n" + "#" * 60)
    print("# CLASSIFICATION (sklearn load_digits)")
    print("#" * 60)
    cls_data = generate_train_test_loader("SklearnDigitsCls", batch_size=128, device=DEVICE, random_state=0)
    k_cls = 20

    # Encoder: CAE
    m = ConcreteAutoencoderModel(cls_data.feature_dim, k_cls, total_epochs=EPOCHS, device=DEVICE)
    h = MLPClassifier(k_cls, HIDDEN, cls_data.cls_num, DEVICE)
    df = _run_encoder(m, cls_data, k_cls, "classification", h, "CAE")
    all_cls.append(df)

    # Encoder: IPCAE
    m = IndirectConcreteAutoencoderModel(cls_data.feature_dim, k_cls, 16, total_epochs=EPOCHS, device=DEVICE)
    h = MLPClassifier(k_cls, HIDDEN, cls_data.cls_num, DEVICE)
    df = _run_encoder(m, cls_data, k_cls, "classification", h, "IPCAE")
    all_cls.append(df)

    # Gate: STG
    m = StochasticGateModel(cls_data.feature_dim, sigma=0.5, device=DEVICE)
    h = MLPClassifier(cls_data.feature_dim, HIDDEN, cls_data.cls_num, DEVICE)
    df = _run_gate(m, cls_data, "classification", h, "STG", sw=1.0)
    all_cls.append(df)

    # Gate: GSG-Softmax
    m = GumbelSoftmaxGateModel(cls_data.feature_dim, 8, total_epochs=EPOCHS, device=DEVICE)
    h = MLPClassifier(cls_data.feature_dim, HIDDEN, cls_data.cls_num, DEVICE)
    df = _run_gate(m, cls_data, "classification", h, "GSG-Softmax", sw=1.0)
    all_cls.append(df)

    # Combined: GSG-Softmax+IPCAE (core)
    m = GumbelSoftmaxGateIndirectConcreteModel(
        cls_data.feature_dim, k_cls, 16, 8, total_epochs=EPOCHS, device=DEVICE)
    h = MLPClassifier(k_cls, HIDDEN, cls_data.cls_num, DEVICE)
    df, _ = _run_combined(m, cls_data, k_cls, "classification", h, "GSG-Softmax+IPCAE", sw=1.0)
    all_cls.append(df)

    # Combined: GSG-Softmax+CAE
    m = GumbelSoftmaxGateConcreteModel(
        cls_data.feature_dim, k_cls, 8, total_epochs=EPOCHS, device=DEVICE)
    h = MLPClassifier(k_cls, HIDDEN, cls_data.cls_num, DEVICE)
    df, _ = _run_combined(m, cls_data, k_cls, "classification", h, "GSG-Softmax+CAE", sw=1.0)
    all_cls.append(df)

    # Combined: HCG+CAE
    m = HardConcreteGateConcreteModel(
        cls_data.feature_dim, k_cls, total_epochs=EPOCHS, device=DEVICE)
    h = MLPClassifier(k_cls, HIDDEN, cls_data.cls_num, DEVICE)
    df, _ = _run_combined(m, cls_data, k_cls, "classification", h, "HCG+CAE", sw=1.0)
    all_cls.append(df)

    cls_df = pd.concat(all_cls, ignore_index=True)
    cls_df.to_csv(os.path.join(RESULTS_DIR, "classification_results.csv"), index=False)

    # ==================================================================
    # REGRESSION (make_regression: 1000 × 100, 30 informative)
    # ==================================================================
    print("\n" + "#" * 60)
    print("# REGRESSION (sklearn make_regression)")
    print("#" * 60)
    reg_data = generate_train_test_loader("SklearnReg", batch_size=64, device=DEVICE, random_state=0)
    k_reg = 20

    m = IndirectConcreteAutoencoderModel(reg_data.feature_dim, k_reg, 16, total_epochs=EPOCHS, device=DEVICE)
    h = MLPRegressor(k_reg, HIDDEN, 1, DEVICE)
    df = _run_encoder(m, reg_data, k_reg, "regression", h, "IPCAE")
    all_reg.append(df)

    m = ConcreteAutoencoderModel(reg_data.feature_dim, k_reg, total_epochs=EPOCHS, device=DEVICE)
    h = MLPRegressor(k_reg, HIDDEN, 1, DEVICE)
    df = _run_encoder(m, reg_data, k_reg, "regression", h, "CAE")
    all_reg.append(df)

    m = GumbelSoftmaxGateIndirectConcreteModel(
        reg_data.feature_dim, k_reg, 16, 8, total_epochs=EPOCHS, device=DEVICE)
    h = MLPRegressor(k_reg, HIDDEN, 1, DEVICE)
    df, _ = _run_combined(m, reg_data, k_reg, "regression", h, "GSG-Softmax+IPCAE", sw=0.1)
    all_reg.append(df)

    m = GumbelSoftmaxGateModel(reg_data.feature_dim, 8, total_epochs=EPOCHS, device=DEVICE)
    h = MLPRegressor(reg_data.feature_dim, HIDDEN, 1, DEVICE)
    df = _run_gate(m, reg_data, "regression", h, "GSG-Softmax", sw=0.1)
    all_reg.append(df)

    reg_df = pd.concat(all_reg, ignore_index=True)
    reg_df.to_csv(os.path.join(RESULTS_DIR, "regression_results.csv"), index=False)

    # ==================================================================
    # RECONSTRUCTION (load_digits: 1797 × 64, autoencoder)
    # ==================================================================
    print("\n" + "#" * 60)
    print("# RECONSTRUCTION (sklearn load_digits)")
    print("#" * 60)
    ae_data = generate_train_test_loader("SklearnDigitsAE", batch_size=128, device=DEVICE, random_state=0)
    k_ae = 20

    m = ConcreteAutoencoderModel(ae_data.feature_dim, k_ae, total_epochs=EPOCHS, device=DEVICE)
    h = AutoencoderHead(k_ae, HIDDEN, ae_data.feature_dim, DEVICE)
    df = _run_encoder(m, ae_data, k_ae, "reconstruction", h, "CAE", input_dim=ae_data.feature_dim)
    all_rec.append(df)

    m = IndirectConcreteAutoencoderModel(ae_data.feature_dim, k_ae, 16, total_epochs=EPOCHS, device=DEVICE)
    h = AutoencoderHead(k_ae, HIDDEN, ae_data.feature_dim, DEVICE)
    df = _run_encoder(m, ae_data, k_ae, "reconstruction", h, "IPCAE", input_dim=ae_data.feature_dim)
    all_rec.append(df)

    m = GumbelSoftmaxGateIndirectConcreteModel(
        ae_data.feature_dim, k_ae, 16, 8, total_epochs=EPOCHS, device=DEVICE)
    h = AutoencoderHead(k_ae, HIDDEN, ae_data.feature_dim, DEVICE)
    df, _ = _run_combined(m, ae_data, k_ae, "reconstruction", h, "GSG-Softmax+IPCAE",
                          sw=0.1, input_dim=ae_data.feature_dim)
    all_rec.append(df)

    rec_df = pd.concat(all_rec, ignore_index=True)
    rec_df.to_csv(os.path.join(RESULTS_DIR, "reconstruction_results.csv"), index=False)

    # ==================================================================
    # ABLATION (sparse_loss_weight sweep on classification)
    # ==================================================================
    print("\n" + "#" * 60)
    print("# ABLATION (sparse_loss_weight sweep)")
    print("#" * 60)
    k_abl = 20
    for sw in [0.01, 0.1, 1.0, 10.0, 100.0]:
        seed_all(0)
        m = GumbelSoftmaxGateIndirectConcreteModel(
            cls_data.feature_dim, k_abl, 16, 8, total_epochs=EPOCHS, device=DEVICE)
        h = MLPClassifier(k_abl, HIDDEN, cls_data.cls_num, DEVICE)
        trainer = GateEncoderTrainer(m, h, task="classification",
                                     sparse_loss_weight=sw, lr=LR, device=DEVICE, seed=0)
        result_df, feature_df = trainer.fit(cls_data.train_loader, EPOCHS, cls_data.test_loader)
        result_df["ablation"] = "sparse_loss_weight"
        result_df["param_value"] = sw
        result_df["dataset"] = "SklearnDigitsCls"
        result_df["seed"] = 0
        all_ablation.append(result_df)

        name = f"ablation_sparse_sw{sw}"
        meta = {"ablation": "sparse_loss_weight", "param_value": sw,
                "final_accuracy": float(result_df["accuracy"].iloc[-1]),
                "final_num_selected": int(result_df["num_selected"].iloc[-1])}
        _save_group(name, result_df, feature_df, meta)

    abl_df = pd.concat(all_ablation, ignore_index=True)
    abl_df.to_csv(os.path.join(RESULTS_DIR, "ablation_results.csv"), index=False)

    # ==================================================================
    # VISUALIZATION
    # ==================================================================
    print("\n" + "#" * 60)
    print("# GENERATING FIGURES")
    print("#" * 60)
    fig_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Classification figures
    plot_training_curves(cls_df, fig_dir, metric_col="accuracy")
    plot_metric_bar(cls_df, fig_dir, metric="accuracy")
    plot_feature_count(cls_df, fig_dir)

    # Regression figures
    reg_fig = os.path.join(fig_dir, "regression")
    plot_training_curves(reg_df, reg_fig, metric_col="r2")
    plot_metric_bar(reg_df, reg_fig, metric="r2")

    # Reconstruction figures
    rec_fig = os.path.join(fig_dir, "reconstruction")
    plot_training_curves(rec_df, rec_fig, metric_col="neg_mse")
    plot_metric_bar(rec_df, rec_fig, metric="neg_mse")

    # Ablation figures
    abl_fig = os.path.join(fig_dir, "ablation")
    plot_ablation(abl_df, abl_fig, metric="accuracy")

    # LaTeX tables
    tbl_dir = os.path.join(RESULTS_DIR, "tables")
    os.makedirs(tbl_dir, exist_ok=True)
    generate_comparison_table(cls_df, tbl_dir)
    generate_ablation_table(abl_df, tbl_dir)

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print("\n" + "=" * 60)
    print("DONE. All results saved to:")
    print(f"  {RESULTS_DIR}/")
    print(f"  ├── classification_results.csv")
    print(f"  ├── regression_results.csv")
    print(f"  ├── reconstruction_results.csv")
    print(f"  ├── ablation_results.csv")
    print(f"  ├── per_group/        ({len(os.listdir(os.path.join(RESULTS_DIR, 'per_group')))} files)")
    print(f"  ├── figures/")
    for d in ["", "regression/", "reconstruction/", "ablation/"]:
        p = os.path.join(fig_dir, d)
        if os.path.exists(p):
            pngs = [f for f in os.listdir(p) if f.endswith(".png")]
            print(f"  │   {d}{'  ' if d else ''}  {len(pngs)} figures")
    print(f"  └── tables/           ({len(os.listdir(tbl_dir))} files)")
    print("=" * 60)


if __name__ == "__main__":
    main()
