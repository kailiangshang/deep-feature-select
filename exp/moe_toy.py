"""
MoE routing: Input-dependent GSG-Softmax + indirect parameterization

Key fix: gate logits depend on input x, not static.
- input_proj(x) -> embed_dim
- indirect param: logits = input_embed @ E_gate -> [batch, num_experts]
- 2-class Gumbel Softmax per expert: {NOT_SELECTED, SELECTED}
"""
from __future__ import annotations

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-3
EPOCHS = 30
BATCH_SIZE = 256
NUM_EXPERTS = 8
TOP_K = 2
EXPERT_HIDDEN = 128
EMBED_DIM = 16
TAU_INIT = 5.0
TAU_MIN = 0.1


class Expert(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_out),
        )

    def forward(self, x):
        return self.net(x)


def gumbel_noise(shape, device):
    return -torch.log(-torch.log(torch.rand(shape, device=device) + 1e-10) + 1e-10)


class StandardMoE(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_experts, top_k, use_aux_loss=True):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_aux_loss = use_aux_loss
        self.experts = nn.ModuleList([
            Expert(dim_in, dim_hidden, dim_out) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(dim_in, num_experts)
        self.aux_loss_weight = 0.01

    def forward(self, x):
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        topk_vals, topk_idx = torch.topk(probs, self.top_k, dim=-1)
        topk_vals = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-10)

        output = torch.zeros(x.size(0), self.experts[0].net[-1].out_features, device=x.device)
        for i in range(self.top_k):
            for j in range(self.num_experts):
                mask = (topk_idx[:, i] == j)
                if mask.any():
                    expert_out = self.experts[j](x[mask])
                    output[mask] += topk_vals[mask, i].unsqueeze(-1) * expert_out

        aux_loss = torch.tensor(0.0, device=x.device)
        if self.use_aux_loss:
            mean_probs = probs.mean(dim=0)
            ideal = 1.0 / self.num_experts
            aux_loss = torch.sum(mean_probs * torch.log(mean_probs / ideal + 1e-10))
        return output, aux_loss, probs


class GSGIPCAEMoE(nn.Module):
    """Input-dependent 2-class Gumbel Softmax gate with indirect parameterization."""
    def __init__(self, dim_in, dim_hidden, dim_out, num_experts, top_k,
                 embed_dim, total_epochs, tau_init=TAU_INIT, tau_min=TAU_MIN):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.total_epochs = total_epochs

        self.experts = nn.ModuleList([
            Expert(dim_in, dim_hidden, dim_out) for _ in range(num_experts)
        ])

        # Indirect parameterization:
        # input_proj: x -> embed_dim
        # expert_embedding: [embed_dim, num_experts]  (shared across experts)
        # logits_selected = input_proj(x) @ expert_embedding -> [batch, num_experts]
        # 2-class: logits = [0, logits_selected] per expert
        self.input_proj = nn.Linear(dim_in, embed_dim)
        self.expert_embedding = nn.Parameter(torch.randn(embed_dim, num_experts) * 0.1)

        self.temperature = tau_init
        self.tau_schedule = np.linspace(tau_init, tau_min, total_epochs)

    def forward(self, x):
        h = self.input_proj(x)  # [batch, embed_dim]
        selected_logits = h @ self.expert_embedding  # [batch, num_experts]

        # 2-class per expert: [0, selected_logit]
        not_selected = torch.zeros_like(selected_logits)
        gate_logits = torch.stack([not_selected, selected_logits], dim=1)  # [batch, 2, num_experts]

        if self.training:
            noise = gumbel_noise(gate_logits.shape, x.device)
            gate_soft = F.softmax((gate_logits + noise) / self.temperature, dim=1)
        else:
            gate_soft = F.softmax(gate_logits / max(self.temperature, 0.01), dim=1)

        p_open = gate_soft[:, 1, :]  # [batch, num_experts]

        # Select top-k experts
        topk_vals, topk_idx = torch.topk(p_open, self.top_k, dim=-1)
        topk_vals_norm = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-10)

        output = torch.zeros(x.size(0), self.experts[0].net[-1].out_features, device=x.device)
        for i in range(self.top_k):
            for j in range(self.num_experts):
                mask = (topk_idx[:, i] == j)
                if mask.any():
                    expert_out = self.experts[j](x[mask])
                    output[mask] += topk_vals_norm[mask, i].unsqueeze(-1) * expert_out

        return output, torch.tensor(0.0, device=x.device), p_open

    def set_epoch(self, epoch):
        idx = min(epoch, len(self.tau_schedule) - 1)
        self.temperature = self.tau_schedule[idx]


def get_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),
    ])
    train_ds = datasets.MNIST("exp/data/mnist", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("exp/data/mnist", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader


def train_model(model, train_loader, test_loader, name, epochs=EPOCHS):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    results = []

    for epoch in range(epochs):
        if hasattr(model, 'set_epoch'):
            model.set_epoch(epoch)

        model.train()
        total_loss = 0
        correct = 0
        total = 0
        expert_counts = np.zeros(model.num_experts)

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            output, aux_loss, gate_probs = model(x)
            loss = criterion(output, y)
            total_loss += loss.item() * x.size(0)

            if isinstance(model, StandardMoE) and model.use_aux_loss:
                loss = loss + model.aux_loss_weight * aux_loss

            pred = output.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += x.size(0)
            expert_counts += (gate_probs > 0.5).sum(dim=0).cpu().numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_acc = correct / total
        avg_loss = total_loss / total

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                output, _, _ = model(x)
                correct += (output.argmax(-1) == y).sum().item()
                total += x.size(0)
        test_acc = correct / total

        load_cv = np.std(expert_counts) / (np.mean(expert_counts) + 1e-10)
        tau_str = f"{model.temperature:.2f}" if hasattr(model, 'temperature') else "N/A"
        print(f"  [{name}] epoch {epoch+1:2d}/{epochs}  "
              f"loss={avg_loss:.4f}  train={train_acc:.4f}  "
              f"test={test_acc:.4f}  load_cv={load_cv:.3f}  tau={tau_str}")
        results.append({"epoch": epoch+1, "name": name, "test_acc": test_acc,
                         "load_cv": load_cv, "expert_counts": expert_counts.tolist()})
    return results


def print_load(results, name, num_experts):
    last = [r for r in results if r["name"] == name][-1]
    counts = np.array(last["expert_counts"])
    total = counts.sum()
    print(f"\n  Expert load:")
    for i, c in enumerate(counts):
        pct = c / total * 100 if total > 0 else 0
        print(f"    E{i}: {c:7.0f} ({pct:5.1f}%) {'█' * int(pct/2)}")
    print(f"    Ideal: {total/num_experts:7.0f} ({100/num_experts:5.1f}%)")


def main():
    train_loader, test_loader = get_loaders()
    dim_in = 28 * 28
    dim_out = 10
    all_results = []

    configs = [
        ("Softmax+Aux (baseline)", lambda: StandardMoE(dim_in, EXPERT_HIDDEN, dim_out, NUM_EXPERTS, TOP_K, True)),
        ("Softmax NoAux", lambda: StandardMoE(dim_in, EXPERT_HIDDEN, dim_out, NUM_EXPERTS, TOP_K, False)),
        ("GSG+IPCAE d=4", lambda: GSGIPCAEMoE(dim_in, EXPERT_HIDDEN, dim_out, NUM_EXPERTS, TOP_K, 4, EPOCHS)),
        ("GSG+IPCAE d=8", lambda: GSGIPCAEMoE(dim_in, EXPERT_HIDDEN, dim_out, NUM_EXPERTS, TOP_K, 8, EPOCHS)),
        ("GSG+IPCAE d=16", lambda: GSGIPCAEMoE(dim_in, EXPERT_HIDDEN, dim_out, NUM_EXPERTS, TOP_K, 16, EPOCHS)),
    ]

    for name, model_fn in configs:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        model = model_fn().to(DEVICE)
        results = train_model(model, train_loader, test_loader, name)
        all_results.extend(results)
        print_load(results, name, NUM_EXPERTS)

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for name in [c[0] for c in configs]:
        r = [r for r in all_results if r["name"] == name]
        last = r[-1]
        counts = np.array(last["expert_counts"])
        active = (counts > counts.sum() * 0.01).sum()
        print(f"  {name:30s}  test={last['test_acc']:.4f}  cv={last['load_cv']:.3f}  active_experts={active}/{NUM_EXPERTS}")


if __name__ == "__main__":
    main()
