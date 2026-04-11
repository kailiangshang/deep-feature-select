from __future__ import annotations

import numpy as np
import scanpy as sc
import os
from sklearn.datasets import load_digits, make_regression

out_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(out_dir, exist_ok=True)

# ── Classification: load_digits (1797 × 64, 10 classes) ──────────────
digits = load_digits()
X_cls = digits.data.astype(np.float32)
y_cls = digits.target.astype(int)
cell_type_cls = np.array([str(y) for y in y_cls])

adata_cls = sc.AnnData(X=X_cls)
adata_cls.obs["cell_type"] = cell_type_cls
adata_cls.write_h5ad(os.path.join(out_dir, "SklearnDigitsCls.h5ad"))
print(f"[CLS] load_digits: {X_cls.shape[0]} samples × {X_cls.shape[1]} features, 10 classes")

# ── Regression: make_regression (1000 × 100, 30 informative) ─────────
X_reg, y_reg = make_regression(
    n_samples=1000, n_features=100, n_informative=30,
    noise=5.0, random_state=42,
)
X_reg = X_reg.astype(np.float32)
y_reg = y_reg.astype(np.float32)
cell_type_reg = np.random.choice(["A", "B", "C"], size=len(y_reg))

adata_reg = sc.AnnData(X=X_reg)
adata_reg.obs["cell_type"] = cell_type_reg
adata_reg.obs["target"] = y_reg
adata_reg.write_h5ad(os.path.join(out_dir, "SklearnReg.h5ad"))
print(f"[REG] make_regression: {X_reg.shape[0]} samples × {X_reg.shape[1]} features, 30 informative")

# ── Reconstruction: load_digits (same data, autoencoder) ─────────────
adata_ae = sc.AnnData(X=X_cls)
adata_ae.obs["cell_type"] = cell_type_cls
adata_ae.write_h5ad(os.path.join(out_dir, "SklearnDigitsAE.h5ad"))
print(f"[AE]  load_digits: {X_cls.shape[0]} samples × {X_cls.shape[1]} features, reconstruction")
