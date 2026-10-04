"""All-zero reference for sparse future purchase-value targets."""

from __future__ import annotations

import torch


def fit_predict(train, validation, *, feature_columns, target_column, device, seed):
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.zeros(len(validation), dtype=torch.float32, device=device)
