"""Container entry point: training labels and prediction features are the only data inputs."""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _checked_predictions(result, device: str, rows: int) -> np.ndarray:
    if device == "cuda" and (not isinstance(result, torch.Tensor)
                             or result.device.type != "cuda"):
        raise RuntimeError("CUDA candidate must return predictions as a CUDA torch.Tensor")
    if isinstance(result, torch.Tensor):
        result = result.detach().cpu().numpy()
    predictions = np.asarray(result, dtype=float).reshape(-1)
    if len(predictions) != rows or not np.isfinite(predictions).all():
        raise ValueError("candidate must return one finite prediction per row")
    return predictions


def main() -> None:
    input_dir, output_dir = Path(sys.argv[1]), Path(sys.argv[2])
    job = json.loads((input_dir / "job.json").read_text())
    device = job["device"]
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the candidate container")
        torch.cuda.reset_peak_memory_stats()
    elif device != "cpu":
        raise ValueError("device must be cpu or cuda")

    seed = int(job["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    train = pd.read_csv(input_dir / "train.csv")
    validation = pd.read_csv(input_dir / "predict.csv")
    spec = importlib.util.spec_from_file_location("candidate", input_dir / "candidate.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("candidate module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.fit_predict(
        train, validation, feature_columns=job["feature_columns"],
        target_column=job["target_column"], device=device, seed=seed)
    prediction_device = str(result.device) if isinstance(result, torch.Tensor) else "host_array"
    predictions = _checked_predictions(result, device, len(validation))
    peak = torch.cuda.max_memory_allocated() if device == "cuda" else 0
    if job["require_cuda_compute"] and peak == 0:
        raise RuntimeError("candidate did not allocate CUDA memory for model computation")
    pd.DataFrame({"prediction": predictions}).to_csv(output_dir / "predictions.csv", index=False)
    (output_dir / "response.json").write_text(json.dumps({"model_device": device,
                                                            "prediction_device": prediction_device,
                                                            "cuda_peak_bytes": peak}))


if __name__ == "__main__":
    main()
