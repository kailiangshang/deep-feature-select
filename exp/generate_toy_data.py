import numpy as np
import scanpy as sc
import os

np.random.seed(42)
n_cells = 200
n_genes = 500
n_informative = 50
out_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(out_dir, exist_ok=True)

class_ids = np.random.choice([0, 1, 2], size=n_cells)
cell_types = np.array([["A", "B", "C"][c] for c in class_ids])

X_noise = np.random.randn(n_cells, n_genes).astype(np.float32) * 0.3

X_cls = X_noise.copy()
for i in range(n_informative):
    for cls_id in range(3):
        mask = class_ids == cls_id
        signal = np.random.randn() * 2.0
        X_cls[mask, i] += signal
    shift = (class_ids / 2.0 - 1.0) * (1.5 + 0.3 * i / n_informative)
    X_cls[:, i] += shift

X_reg = X_noise.copy()
w_reg = np.random.randn(n_informative).astype(np.float32) * 1.5
target = X_reg[:, :n_informative] @ w_reg + np.random.randn(n_cells).astype(np.float32) * 0.1
target = target.astype(np.float32)

n_latent = 20
Z_ae = np.random.randn(n_cells, n_latent).astype(np.float32)
W_ae = np.random.randn(n_latent, n_informative).astype(np.float32) * 0.8
X_ae_signal = Z_ae @ W_ae
X_ae = X_noise.copy()
X_ae[:, :n_informative] += X_ae_signal

adata_cls = sc.AnnData(X=X_cls)
adata_cls.obs["cell_type"] = cell_types
adata_cls.write_h5ad(os.path.join(out_dir, "ToyCls.h5ad"))

adata_reg = sc.AnnData(X=X_reg)
adata_reg.obs["cell_type"] = cell_types
adata_reg.obs["target"] = target
adata_reg.write_h5ad(os.path.join(out_dir, "ToyReg.h5ad"))

adata_ae = sc.AnnData(X=X_ae)
adata_ae.obs["cell_type"] = cell_types
adata_ae.write_h5ad(os.path.join(out_dir, "ToyAE.h5ad"))

print(f"Generated 3 datasets: {n_cells} cells × {n_genes} genes")
print(f"  Informative features: first {n_informative} genes")
print(f"  ToyCls.h5ad — classification (3 classes, differential expression)")
print(f"  ToyReg.h5ad — regression (target = linear combo of first {n_informative} genes)")
print(f"  ToyAE.h5ad  — reconstruction ({n_latent} latent dims → {n_informative} signal genes)")
