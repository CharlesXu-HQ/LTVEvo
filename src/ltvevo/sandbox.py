"""Run Agent-written models with no network and no validation or test labels."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


class DockerSandbox:
    def __init__(self, image: str, *, timeout_seconds: int = 3600):
        if not image or timeout_seconds < 1:
            raise ValueError("Docker image and positive timeout are required")
        self.image = image
        self.timeout_seconds = timeout_seconds

    def predict(self, candidate: Path, train: pd.DataFrame, target: pd.DataFrame, *,
                feature_columns: list[str], target_column: str, seed: int, device: str,
                require_cuda_compute: bool = True) -> dict:
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if not feature_columns or target_column in feature_columns:
            raise ValueError("features must be nonempty and exclude the target")
        if any(column not in train or column not in target for column in feature_columns):
            raise ValueError("candidate input lacks a declared feature")
        if target_column not in train:
            raise ValueError("training input lacks the target")
        with tempfile.TemporaryDirectory(prefix="ltvevo-sandbox-") as scratch:
            root = Path(scratch)
            input_dir, output_dir = root / "input", root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            shutil.copyfile(candidate, input_dir / "candidate.py")
            train.loc[:, [*feature_columns, target_column]].to_csv(input_dir / "train.csv", index=False)
            target.loc[:, feature_columns].to_csv(input_dir / "predict.csv", index=False)
            (input_dir / "job.json").write_text(json.dumps({
                "feature_columns": feature_columns, "target_column": target_column,
                "seed": seed, "device": device,
                "require_cuda_compute": require_cuda_compute and device == "cuda"}))
            cidfile = root / "container.id"
            command = ["docker", "run", "--rm", "--cidfile", str(cidfile),
                       "--network", "none", "--read-only", "--cap-drop", "ALL",
                       "--security-opt", "no-new-privileges", "--pids-limit", "256",
                       "--memory", "16g", "--cpus", "8", "--user", f"{os.getuid()}:{os.getgid()}",
                       "--tmpfs", "/tmp:rw,nosuid,size=2g",
                       "--mount", f"type=bind,src={input_dir},dst=/input,readonly",
                       "--mount", f"type=bind,src={output_dir},dst=/output",
                       "--env", "PYTHONDONTWRITEBYTECODE=1"]
            if device == "cuda":
                command.extend(["--gpus", "all"])
            command.extend([self.image, "python", "/opt/ltvevo/sandbox_worker.py",
                            "/input", "/output"])
            try:
                run = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                                     text=True, timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired as error:
                if cidfile.exists():
                    subprocess.run(["docker", "rm", "-f", cidfile.read_text().strip()],
                                   stdin=subprocess.DEVNULL, capture_output=True, timeout=30)
                raise RuntimeError("candidate exceeded sandbox time limit") from error
            if run.returncode:
                raise RuntimeError(f"candidate container failed: {(run.stderr or run.stdout)[-2000:]}")
            response = json.loads((output_dir / "response.json").read_text())
            predictions = pd.read_csv(output_dir / "predictions.csv")["prediction"].to_numpy(float)
            if len(predictions) != len(target) or not np.isfinite(predictions).all():
                raise ValueError("candidate output length or values are invalid")
            return {"predictions": predictions, **response}
