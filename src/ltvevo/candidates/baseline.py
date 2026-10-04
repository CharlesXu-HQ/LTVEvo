"""Calibrated historical-spend baseline for future purchase value."""

from __future__ import annotations

import numpy as np
import torch


def fit_predict(train, validation, *, feature_columns, target_column, device, seed):
    if "spend_90d" not in feature_columns:
        raise ValueError("historical baseline requires spend_90d")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    spend = np.asarray(train["spend_90d"], dtype=np.float64)
    target = np.asarray(train[target_column], dtype=np.float64)
    future_spend = np.asarray(validation["spend_90d"], dtype=np.float64)
    if not (np.isfinite(spend).all() and np.isfinite(target).all()
            and np.isfinite(future_spend).all()):
        raise ValueError("baseline inputs must be finite")
    if len(spend) == 0 or np.any(target < 0):
        raise ValueError("baseline requires nonempty, nonnegative purchase targets")

    x = torch.as_tensor(spend, dtype=torch.float64, device=device)
    y = torch.as_tensor(target, dtype=torch.float64, device=device)
    x_mean = x.mean()
    y_mean = y.mean()
    centered = x - x_mean
    variance = torch.sum(centered.square())
    if variance.item() == 0:
        slope = torch.zeros((), dtype=torch.float64, device=device)
    else:
        slope = torch.clamp(torch.sum(centered * (y - y_mean)) / variance, min=0)
    intercept = y_mean - slope * x_mean
    future = torch.as_tensor(future_spend, dtype=torch.float64, device=device)
    prediction = torch.clamp(intercept + slope * future, min=0)
    return prediction if device == "cuda" else prediction.numpy()
