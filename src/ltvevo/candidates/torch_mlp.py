"""Small PyTorch regressor for nonnegative future purchase value."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def fit_predict(train, validation, *, feature_columns, target_column, device, seed):
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if not feature_columns:
        raise ValueError("at least one feature is required")

    train_features = train.loc[:, feature_columns].to_numpy(dtype=np.float32)
    validation_features = validation.loc[:, feature_columns].to_numpy(dtype=np.float32)
    targets = train[target_column].to_numpy(dtype=np.float32)
    if len(targets) == 0 or np.any(targets < 0):
        raise ValueError("training targets must be nonempty and nonnegative")
    if not (np.isfinite(train_features).all()
            and np.isfinite(validation_features).all()
            and np.isfinite(targets).all()):
        raise ValueError("features and targets must be finite")

    mean = train_features.mean(axis=0)
    scale = np.maximum(train_features.std(axis=0), 1e-6)
    train_features = np.clip((train_features - mean) / scale, -10, 10)
    validation_features = np.clip((validation_features - mean) / scale, -10, 10)

    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    x = torch.as_tensor(train_features, dtype=torch.float32, device=device)
    y = torch.as_tensor(np.log1p(targets), dtype=torch.float32, device=device).unsqueeze(1)
    x_validation = torch.as_tensor(validation_features, dtype=torch.float32, device=device)

    model = nn.Sequential(
        nn.Linear(x.shape[1], 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 1), nn.Softplus(),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(20):
        order = torch.randperm(len(x), device=device)
        for indices in order.split(2048):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(x[indices]), y[indices])
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        log_prediction = model(x_validation).squeeze(1)
        prediction = torch.expm1(torch.clamp(log_prediction, max=20))
    return prediction if device == "cuda" else prediction.numpy()
