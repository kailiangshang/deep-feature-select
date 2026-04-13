from __future__ import annotations

import torch
import torch.nn.functional as F


def generate_gumbel_noise(like_tensor: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    uniform = torch.rand_like(like_tensor, device=like_tensor.device)
    return -torch.log(-torch.log(uniform + eps) + eps)


def custom_one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    one_hot_enc = torch.zeros(indices.size(0), num_classes, device=indices.device)
    valid_mask = indices != -1
    if valid_mask.any():
        valid_indices = indices[valid_mask]
        one_hot_valid = F.one_hot(valid_indices, num_classes=num_classes).float()
        one_hot_enc[valid_mask] = one_hot_valid
    return one_hot_enc
