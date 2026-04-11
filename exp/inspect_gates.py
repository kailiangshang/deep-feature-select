"""Inspect gate distributions for trained models."""
from __future__ import annotations

import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

from deepfs import (
    GumbelSoftmaxGateConcreteModel,
    GumbelSoftmaxGateIndirectConcreteModel,
    HardConcreteGateConcreteModel,
    StochasticGateModel,
)
from exp.data import generate_train_test_loader
from exp.trainers import GateTrainer, GateEncoderTrainer
from exp.utils import MLPClassifier, seed_all

DEVICE = "cpu"
EPOCHS = 100
LR = 1e-3


def inspect_gsg_combined(model, name):
    gate_logits = model._get_gate_logits()
    gate_probs = torch.softmax(gate_logits, dim=0)
    p_open = gate_probs[1, :].detach().numpy()
    open_slots = torch.argmax(gate_logits, dim=0).numpy()

    if hasattr(model, '_get_encoder_logits'):
        enc_logits = model._get_encoder_logits()
    else:
        enc_logits = model.logits_encoder
    enc_probs = torch.softmax(enc_logits, dim=0).detach().numpy()
    selected = torch.argmax(enc_logits, dim=0).numpy()

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  k={model.k}, temperature={model.temperature.item():.6f}")

    print(f"\n  Gate open prob per slot:")
    for i, p in enumerate(p_open):
        bar = "█" * int(p * 40)
        status = "OPEN " if open_slots[i] == 1 else "SHUT "
        print(f"    slot {i:2d}: {p:.4f} {bar}  [{status}]")

    selected_clean = selected.copy()
    selected_clean[open_slots == 0] = -1
    valid = selected_clean[selected_clean >= 0]

    print(f"\n  Encoder -> feature mapping:")
    for i, (s, o) in enumerate(zip(selected, open_slots)):
        status = "OPEN " if o == 1 else "SHUT "
        top3 = np.argsort(enc_probs[:, i])[::-1][:3]
        print(f"    slot {i:2d} -> feat {s:3d}  [{status}]  top3: {top3.tolist()}")

    if len(valid) > 0:
        unique, counts = np.unique(valid, return_counts=True)
        print(f"\n  Final selected features: {sorted(unique.tolist())}")
        print(f"  Unique: {len(unique)}, Open slots: {len(valid)}, Overlap: {len(valid) - len(unique)}")
    else:
        print(f"\n  ** ALL GATES CLOSED — no features selected! **")


def inspect_hcg(model, name, input_dim):
    with torch.no_grad():
        s = torch.sigmoid(model.gate_logits)
        gate_prob = torch.clamp(s * (model.zeta - model.gamma) + model.gamma, 0.0, 1.0).numpy()
        enc_probs = torch.softmax(model.logits_encoder, dim=0).numpy()
        selected = torch.argmax(model.logits_encoder, dim=0).numpy()

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  k={model.k}, temperature={model.temperature.item():.6f}")

    print(f"\n  Gate prob per slot (HardConcrete stretched sigmoid):")
    for i, p in enumerate(gate_prob):
        bar = "█" * int(p * 40)
        status = "OPEN " if p > 0.5 else "SHUT "
        print(f"    slot {i:2d}: {p:.4f} {bar}  [{status}]")

    open_mask = gate_prob > 0.5
    selected_clean = selected.copy()
    selected_clean[~open_mask] = -1
    valid = selected_clean[selected_clean >= 0]

    print(f"\n  Encoder -> feature mapping:")
    for i, s_val in enumerate(selected):
        status = "OPEN " if open_mask[i] else "SHUT "
        top3 = np.argsort(enc_probs[:, i])[::-1][:3]
        print(f"    slot {i:2d} -> feat {s_val:3d}  [{status}]  top3: {top3.tolist()}")

    if len(valid) > 0:
        unique, counts = np.unique(valid, return_counts=True)
        print(f"\n  Final selected features: {sorted(unique.tolist())}")
        print(f"  Unique: {len(unique)}, Open slots: {len(valid)}, Overlap: {len(valid) - len(unique)}")
    else:
        print(f"\n  ** ALL GATES CLOSED **")


def inspect_stg(model, name, input_dim):
    with torch.no_grad():
        probs = torch.sigmoid(model.mu_gate).numpy()
    selected = model.get_selection_result().selected_indices

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  input_dim={input_dim}")

    top_k = 20
    top_indices = np.argsort(probs)[::-1]
    print(f"\n  Top {top_k} gate probabilities:")
    for idx in top_indices[:top_k]:
        bar = "█" * int(probs[idx] * 40)
        selected_mark = " ★" if idx in selected else ""
        print(f"    feature {idx:3d}: p={probs[idx]:.4f} {bar}{selected_mark}")

    print(f"\n  Selected features ({len(selected)}): {sorted(selected.tolist())}")
    closed = np.where(probs < 0.5)[0]
    print(f"  Closed features: {len(closed)}/{input_dim}")


def main():
    cls_data = generate_train_test_loader("SklearnDigitsCls", batch_size=128, device=DEVICE, random_state=0)
    k = 20

    configs = [
        ("GSG-Softmax+CAE (sw=1.0)", GumbelSoftmaxGateConcreteModel, dict(k=k, embedding_dim_gate=8), 1.0),
        ("GSG-Softmax+CAE (sw=0.01)", GumbelSoftmaxGateConcreteModel, dict(k=k, embedding_dim_gate=8), 0.01),
        ("GSG-Softmax+IPCAE (sw=1.0)", GumbelSoftmaxGateIndirectConcreteModel,
         dict(k=k, embedding_dim_encoder=16, embedding_dim_gate=8), 1.0),
        ("GSG-Softmax+IPCAE (sw=0.01)", GumbelSoftmaxGateIndirectConcreteModel,
         dict(k=k, embedding_dim_encoder=16, embedding_dim_gate=8), 0.01),
        ("HCG+CAE (sw=1.0)", HardConcreteGateConcreteModel, dict(k=k), 1.0),
        ("HCG+CAE (sw=0.01)", HardConcreteGateConcreteModel, dict(k=k), 0.01),
    ]

    trained = []
    for name, cls, kwargs, sw in configs:
        seed_all(0)
        m = cls(cls_data.feature_dim, total_epochs=EPOCHS, device=DEVICE, **kwargs)
        h = MLPClassifier(k, 64, cls_data.cls_num, DEVICE)
        trainer = GateEncoderTrainer(m, h, task="classification", sparse_loss_weight=sw, lr=LR, device=DEVICE, seed=0)
        trainer.fit(cls_data.train_loader, EPOCHS, cls_data.test_loader)
        trained.append((name, m))

    # STG separately (uses full dim)
    seed_all(0)
    m_stg = StochasticGateModel(cls_data.feature_dim, sigma=0.5, device=DEVICE)
    h_stg = MLPClassifier(cls_data.feature_dim, 64, cls_data.cls_num, DEVICE)
    trainer = GateTrainer(m_stg, h_stg, task="classification", sparse_loss_weight=1.0, lr=LR, device=DEVICE, seed=0)
    trainer.fit(cls_data.train_loader, EPOCHS, cls_data.test_loader)

    # Inspect all
    for name, m in trained:
        if isinstance(m, HardConcreteGateConcreteModel):
            inspect_hcg(m, name, cls_data.feature_dim)
        else:
            inspect_gsg_combined(m, name)

    inspect_stg(m_stg, "STG (sw=1.0)", cls_data.feature_dim)


if __name__ == "__main__":
    main()
