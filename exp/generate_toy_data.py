import numpy as np
import scanpy as sc
import os

np.random.seed(42)
n_cells = 200
n_genes = 500
out_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(out_dir, exist_ok=True)

X = np.random.randn(n_cells, n_genes).astype(np.float32)
cell_types = np.random.choice(["A", "B", "C"], size=n_cells)
targets = np.random.randn(n_cells).astype(np.float32)

adata_cls = sc.AnnData(X=X)
adata_cls.obs["cell_type"] = cell_types
adata_cls.write_h5ad(os.path.join(out_dir, "ToyCls.h5ad"))

adata_reg = sc.AnnData(X=X)
adata_reg.obs["cell_type"] = cell_types
adata_reg.obs["target"] = targets
adata_reg.write_h5ad(os.path.join(out_dir, "ToyReg.h5ad"))

adata_ae = sc.AnnData(X=X)
adata_ae.obs["cell_type"] = cell_types
adata_ae.write_h5ad(os.path.join(out_dir, "ToyAE.h5ad"))

print(f"Generated 3 datasets: {n_cells} cells × {n_genes} genes")
print(f"  ToyCls.h5ad — classification (3 classes)")
print(f"  ToyReg.h5ad — regression (continuous target)")
print(f"  ToyAE.h5ad  — reconstruction (autoencoder)")
