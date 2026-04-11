"""Fair comparison: all models with tuned hyperparameters, 300 epochs."""
from __future__ import annotations

import os
import warnings
warnings.filterwarnings("ignore")

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
from exp.visualization.plot_results import plot_training_curves, plot_metric_bar, plot_feature_count
from exp.visualization.generate_tables import generate_comparison_table

DEVICE = "cpu"
RESULTS_DIR = "exp/results/fair_comparison"
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_encoder(model, data, k, task, head, epochs, lr, name, **tkwargs):
    seed_all(0)
    trainer = EncoderTrainer(model, head, task=task, lr=lr, device=DEVICE, seed=0, **tkwargs)
    df = trainer.fit(data.train_loader, epochs, data.test_loader)
    last = df.iloc[-1]
    metric_col = trainer.task.metric_name
    print(f"  {name:40s}  {metric_col}={last[metric_col]:.4f}  features={int(last['num_selected'])}")
    df["model"] = name
    return df


def run_gate(model, data, task, head, epochs, lr, sw, name, **tkwargs):
    seed_all(0)
    trainer = GateTrainer(model, head, task=task, sparse_loss_weight=sw, lr=lr, device=DEVICE, seed=0, **tkwargs)
    df = trainer.fit(data.train_loader, epochs, data.test_loader)
    last = df.iloc[-1]
    metric_col = trainer.task.metric_name
    print(f"  {name:40s}  {metric_col}={last[metric_col]:.4f}  features={int(last['num_selected'])}")
    df["model"] = name
    df["sparse_weight"] = sw
    return df


def run_combined(model, data, k, task, head, epochs, lr, sw, name, **tkwargs):
    seed_all(0)
    trainer = GateEncoderTrainer(model, head, task=task, sparse_loss_weight=sw, lr=lr, device=DEVICE, seed=0, **tkwargs)
    result_df, feature_df = trainer.fit(data.train_loader, epochs, data.test_loader)
    last = result_df.iloc[-1]
    metric_col = trainer.task.metric_name
    print(f"  {name:40s}  {metric_col}={last[metric_col]:.4f}  features={int(last['num_selected'])}  sw={sw}")
    result_df["model"] = name
    result_df["sparse_weight"] = sw
    result_df["k"] = k
    return result_df


def main():
    all_results = []

    # ════════════════════════════════════════════════════════════════
    # CLASSIFICATION (load_digits: 1797 × 64, 10 classes)
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  CLASSIFICATION (load_digits, 300 epochs)")
    print("=" * 70)
    cls_data = generate_train_test_loader("SklearnDigitsCls", batch_size=128, device=DEVICE, random_state=0)
    EPOCHS = 300
    LR = 2e-3
    HIDDEN = 128
    dim = cls_data.feature_dim
    n_cls = cls_data.cls_num

    for sw_list, model_factory, head_factory, name_prefix, base_kwargs in [
        # Encoders
        ([], lambda: ConcreteAutoencoderModel(dim, 30, total_epochs=EPOCHS, device=DEVICE),
         lambda: MLPClassifier(30, HIDDEN, n_cls, DEVICE), "CAE", {"k": 30}),
        ([], lambda: IndirectConcreteAutoencoderModel(dim, 30, 32, total_epochs=EPOCHS, device=DEVICE),
         lambda: MLPClassifier(30, HIDDEN, n_cls, DEVICE), "IPCAE", {"k": 30}),
    ]:
        df = run_encoder(model_factory(), cls_data, 30, "classification", head_factory(),
                         EPOCHS, LR, name_prefix)
        df["task"] = "classification"
        all_results.append(df)

    # Gate-only models
    for sw in [0.01, 0.1, 1.0]:
        df = run_gate(
            StochasticGateModel(dim, sigma=0.5, device=DEVICE),
            cls_data, "classification", MLPClassifier(dim, HIDDEN, n_cls, DEVICE),
            EPOCHS, LR, sw, f"STG (sw={sw})")
        df["task"] = "classification"
        all_results.append(df)

    for sw in [0.01, 0.1]:
        df = run_gate(
            GumbelSoftmaxGateModel(dim, 16, total_epochs=EPOCHS, device=DEVICE),
            cls_data, "classification", MLPClassifier(dim, HIDDEN, n_cls, DEVICE),
            EPOCHS, LR, sw, f"GSG-Softmax (sw={sw})")
        df["task"] = "classification"
        all_results.append(df)

    # Combined models
    for k in [20, 30]:
        for sw in [0.01, 0.1]:
            df = run_combined(
                GumbelSoftmaxGateIndirectConcreteModel(dim, k, 64, 32, total_epochs=EPOCHS, device=DEVICE),
                cls_data, k, "classification", MLPClassifier(k, HIDDEN, n_cls, DEVICE),
                EPOCHS, LR, sw, f"GSG-Softmax+IPCAE (k={k},sw={sw})")
            df["task"] = "classification"
            all_results.append(df)

    for k in [20, 30]:
        for sw in [0.01, 0.1]:
            df = run_combined(
                GumbelSoftmaxGateConcreteModel(dim, k, 16, total_epochs=EPOCHS, device=DEVICE),
                cls_data, k, "classification", MLPClassifier(k, HIDDEN, n_cls, DEVICE),
                EPOCHS, LR, sw, f"GSG-Softmax+CAE (k={k},sw={sw})")
            df["task"] = "classification"
            all_results.append(df)

    for k in [20, 30]:
        for sw in [0.01, 0.1]:
            df = run_combined(
                HardConcreteGateConcreteModel(dim, k, total_epochs=EPOCHS, device=DEVICE),
                cls_data, k, "classification", MLPClassifier(k, HIDDEN, n_cls, DEVICE),
                EPOCHS, LR, sw, f"HCG+CAE (k={k},sw={sw})")
            df["task"] = "classification"
            all_results.append(df)

    # ════════════════════════════════════════════════════════════════
    # REGRESSION (make_regression: 1000 × 100, 30 informative)
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  REGRESSION (make_regression, 300 epochs)")
    print("=" * 70)
    reg_data = generate_train_test_loader("SklearnReg", batch_size=64, device=DEVICE, random_state=0)
    dim_r = reg_data.feature_dim

    for k, name in [(20, "CAE"), (20, "IPCAE")]:
        if name == "CAE":
            m = ConcreteAutoencoderModel(dim_r, k, total_epochs=EPOCHS, device=DEVICE)
        else:
            m = IndirectConcreteAutoencoderModel(dim_r, k, 32, total_epochs=EPOCHS, device=DEVICE)
        df = run_encoder(m, reg_data, k, "regression", MLPRegressor(k, HIDDEN, 1, DEVICE),
                         EPOCHS, LR, name)
        df["task"] = "regression"
        all_results.append(df)

    for sw in [0.01, 0.1, 1.0]:
        df = run_gate(
            StochasticGateModel(dim_r, sigma=0.5, device=DEVICE),
            reg_data, "regression", MLPRegressor(dim_r, HIDDEN, 1, DEVICE),
            EPOCHS, LR, sw, f"STG (sw={sw})")
        df["task"] = "regression"
        all_results.append(df)

    for sw in [0.01, 0.1]:
        df = run_gate(
            GumbelSoftmaxGateModel(dim_r, 16, total_epochs=EPOCHS, device=DEVICE),
            reg_data, "regression", MLPRegressor(dim_r, HIDDEN, 1, DEVICE),
            EPOCHS, LR, sw, f"GSG-Softmax (sw={sw})")
        df["task"] = "regression"
        all_results.append(df)

    for k in [20, 30]:
        for sw in [0.01, 0.1]:
            df = run_combined(
                GumbelSoftmaxGateIndirectConcreteModel(dim_r, k, 64, 32, total_epochs=EPOCHS, device=DEVICE),
                reg_data, k, "regression", MLPRegressor(k, HIDDEN, 1, DEVICE),
                EPOCHS, LR, sw, f"GSG-Softmax+IPCAE (k={k},sw={sw})")
            df["task"] = "regression"
            all_results.append(df)

    for k in [20, 30]:
        for sw in [0.01, 0.1]:
            df = run_combined(
                GumbelSoftmaxGateConcreteModel(dim_r, k, 16, total_epochs=EPOCHS, device=DEVICE),
                reg_data, k, "regression", MLPRegressor(k, HIDDEN, 1, DEVICE),
                EPOCHS, LR, sw, f"GSG-Softmax+CAE (k={k},sw={sw})")
            df["task"] = "regression"
            all_results.append(df)

    for k in [20, 30]:
        for sw in [0.01, 0.1]:
            df = run_combined(
                HardConcreteGateConcreteModel(dim_r, k, total_epochs=EPOCHS, device=DEVICE),
                reg_data, k, "regression", MLPRegressor(k, HIDDEN, 1, DEVICE),
                EPOCHS, LR, sw, f"HCG+CAE (k={k},sw={sw})")
            df["task"] = "regression"
            all_results.append(df)

    # ════════════════════════════════════════════════════════════════
    # RECONSTRUCTION (load_digits, autoencoder)
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  RECONSTRUCTION (load_digits, 300 epochs)")
    print("=" * 70)
    ae_data = generate_train_test_loader("SklearnDigitsAE", batch_size=128, device=DEVICE, random_state=0)
    dim_a = ae_data.feature_dim

    for k, name in [(30, "CAE"), (30, "IPCAE")]:
        if name == "CAE":
            m = ConcreteAutoencoderModel(dim_a, k, total_epochs=EPOCHS, device=DEVICE)
        else:
            m = IndirectConcreteAutoencoderModel(dim_a, k, 32, total_epochs=EPOCHS, device=DEVICE)
        df = run_encoder(m, ae_data, k, "reconstruction",
                         AutoencoderHead(k, HIDDEN, dim_a, DEVICE),
                         EPOCHS, LR, name, input_dim=dim_a)
        df["task"] = "reconstruction"
        all_results.append(df)

    for sw in [0.01, 0.1, 1.0]:
        df = run_gate(
            StochasticGateModel(dim_a, sigma=0.5, device=DEVICE),
            ae_data, "reconstruction", AutoencoderHead(dim_a, HIDDEN, dim_a, DEVICE),
            EPOCHS, LR, sw, f"STG (sw={sw})", input_dim=dim_a)
        df["task"] = "reconstruction"
        all_results.append(df)

    for sw in [0.01, 0.1]:
        df = run_gate(
            GumbelSoftmaxGateModel(dim_a, 16, total_epochs=EPOCHS, device=DEVICE),
            ae_data, "reconstruction", AutoencoderHead(dim_a, HIDDEN, dim_a, DEVICE),
            EPOCHS, LR, sw, f"GSG-Softmax (sw={sw})", input_dim=dim_a)
        df["task"] = "reconstruction"
        all_results.append(df)

    for k in [20, 30]:
        for sw in [0.01, 0.1]:
            df = run_combined(
                GumbelSoftmaxGateIndirectConcreteModel(dim_a, k, 64, 32, total_epochs=EPOCHS, device=DEVICE),
                ae_data, k, "reconstruction",
                AutoencoderHead(k, HIDDEN, dim_a, DEVICE),
                EPOCHS, LR, sw, f"GSG-Softmax+IPCAE (k={k},sw={sw})",
                input_dim=dim_a)
            df["task"] = "reconstruction"
            all_results.append(df)

    for k in [20, 30]:
        for sw in [0.01, 0.1]:
            df = run_combined(
                GumbelSoftmaxGateConcreteModel(dim_a, k, 16, total_epochs=EPOCHS, device=DEVICE),
                ae_data, k, "reconstruction",
                AutoencoderHead(k, HIDDEN, dim_a, DEVICE),
                EPOCHS, LR, sw, f"GSG-Softmax+CAE (k={k},sw={sw})",
                input_dim=dim_a)
            df["task"] = "reconstruction"
            all_results.append(df)

    for k in [20, 30]:
        for sw in [0.01, 0.1]:
            df = run_combined(
                HardConcreteGateConcreteModel(dim_a, k, total_epochs=EPOCHS, device=DEVICE),
                ae_data, k, "reconstruction",
                AutoencoderHead(k, HIDDEN, dim_a, DEVICE),
                EPOCHS, LR, sw, f"HCG+CAE (k={k},sw={sw})",
                input_dim=dim_a)
            df["task"] = "reconstruction"
            all_results.append(df)

    # ════════════════════════════════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════════════════════════════════
    all_df = pd.concat(all_results, ignore_index=True)
    all_df.to_csv(os.path.join(RESULTS_DIR, "all_results.csv"), index=False)

    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)

    for task, metric in [("classification", "accuracy"), ("regression", "r2"), ("reconstruction", "neg_mse")]:
        task_df = all_df[all_df["task"] == task]
        print(f"\n  --- {task.upper()} ({metric}) ---")
        rows = []
        for m, g in task_df.groupby("model"):
            last = g[g["epoch"] == g["epoch"].max()].iloc[0]
            rows.append((last[metric], int(last["num_selected"]), m))
        rows.sort(key=lambda x: -x[0])
        for val, feat, name in rows:
            print(f"    {val:8.4f}  feat={feat:3d}  {name}")

    # Figures
    fig_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    for task, metric in [("classification", "accuracy"), ("regression", "r2"), ("reconstruction", "neg_mse")]:
        task_df = all_df[all_df["task"] == task]
        if task == "classification":
            plot_training_curves(task_df, fig_dir, metric_col=metric)
            plot_metric_bar(task_df, fig_dir, metric=metric)
            plot_feature_count(task_df, fig_dir)
        else:
            sub = os.path.join(fig_dir, task)
            plot_training_curves(task_df, sub, metric_col=metric)
            plot_metric_bar(task_df, sub, metric=metric)


if __name__ == "__main__":
    main()
