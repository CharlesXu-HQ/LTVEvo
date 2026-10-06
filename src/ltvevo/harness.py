"""A pinned ModelEvoHarness research context for LTV prediction experiments."""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
from pathlib import Path

import pandas as pd

from .task import TaskSpec


_ROOT = Path(__file__).resolve().parents[2]
_SUBMODULE = _ROOT / "third_party" / "model-evo-harness"
_HOST_REFERENCES = ("baseline.py", "torch_mlp.py", "zero.py")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git(directory: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(directory), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("ModelEvoHarness pinned submodule cannot be verified") from exc


def _pinned_source(package: object) -> str:
    installed = Path(package.__file__).resolve()
    expected = (_SUBMODULE / "src" / "model_evo_harness" / "__init__.py").resolve()
    if installed != expected:
        raise RuntimeError("ModelEvoHarness must be imported from the pinned submodule checkout")
    source_commit = _git(_SUBMODULE, "rev-parse", "HEAD")
    link = _git(_ROOT, "ls-files", "--stage", "--", "third_party/model-evo-harness")
    if not link.startswith(f"160000 {source_commit} 0\t"):
        raise RuntimeError("ModelEvoHarness checkout differs from the pinned submodule commit")
    if _git(_SUBMODULE, "status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("ModelEvoHarness pinned submodule checkout has local changes")
    return source_commit


def _task_snapshot(task: TaskSpec, metadata: dict, identity: dict) -> dict:
    fields = metadata["feature_columns"]
    protocol = {
        "unit": "customer_as_of",
        "split": "chronological_purged_train_validation",
        "metric": "mae",
        "time_cutoff": "features strictly before each as_of date",
        "label_window": f"[as_of, as_of + {task.horizon_days} days)",
        "horizon_days": str(task.horizon_days),
        "train_observation_dates": f"{task.as_of_start}..{task.train_end}",
        "validation_observation_dates": f"{task.validation_start}..{task.validation_end}",
        "test_observation_dates": f"{task.test_start}..{task.test_end}",
        "holdout_policy": "test labels are sealed until the champion is selected",
        "evaluator_version": task.evaluator_version,
    }
    return {
        "task_id": f"ltv-prediction-{task.fingerprint[:12]}",
        "dataset_digest": _digest({
            "task_fingerprint": task.fingerprint,
            "raw_sha256": task.raw_sha256,
            "split_sha256": identity["split_sha256"],
            "evaluator_version": task.evaluator_version,
            "feature_columns": fields,
        }),
        "stage": "ltv_prediction",
        "framework": "pytorch",
        "fields": fields,
        "capabilities": ["tabular_features", "observed_outcome_labels"],
        "objective": {"name": "mae", "direction": "min"},
        "target_policy": task.target_policy,
        "raw_sha256": task.raw_sha256,
        "split_sha256": identity["split_sha256"],
        "evaluation_protocol": protocol,
    }


def _train_profile(snapshots: Path, fields: list[str]) -> dict:
    train = pd.read_csv(snapshots / "train.csv", usecols=["target", *fields])
    if train.empty:
        raise ValueError("training snapshot is empty")
    target = pd.to_numeric(train["target"], errors="raise")
    return {
        "source": "train.csv",
        "rows": len(train),
        "target_mean": float(target.mean()),
        "zero_target_fraction": float((target == 0).mean()),
        "target_p90": float(target.quantile(0.9)),
        "features": {name: {
            "missing_fraction": float(train[name].isna().mean()),
            "median": float(train[name].median()) if train[name].notna().any() else None,
        } for name in fields},
    }


def _host_references() -> dict:
    result = {}
    for name in _HOST_REFERENCES:
        source = (_ROOT / "src" / "ltvevo" / "candidates" / name).read_text(encoding="utf-8")
        result[name] = {"provenance": "host", "path": f"src/ltvevo/candidates/{name}",
                        "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                        "content": source}
    return result


def build_harness_context(task_path: Path, snapshots: Path, task_identity: dict) -> dict:
    """Return a versioned research context without reading validation or test rows."""
    try:
        package = importlib.import_module("model_evo_harness")
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install the pinned ModelEvoHarness submodule for --harness model-evo") from exc
    source_commit = _pinned_source(package)
    task_path, snapshots = Path(task_path), Path(snapshots)
    task = TaskSpec.load(task_path)
    metadata = json.loads((snapshots / "metadata.json").read_text(encoding="utf-8"))
    for name, expected in (
        ("task_fingerprint", task.fingerprint),
        ("raw_sha256", task.raw_sha256),
        ("target_policy", task.target_policy),
        ("evaluator_version", task.evaluator_version),
    ):
        if name in metadata and metadata[name] != expected:
            raise ValueError(f"snapshot {name} differs from frozen task")
        if task_identity.get(name) != expected:
            raise ValueError(f"task identity {name} differs from frozen task")
    fields = metadata.get("feature_columns")
    if (not isinstance(fields, list) or not fields or len(fields) != len(set(fields))
            or any(not isinstance(name, str) or not name or name in
                   {"target", "customer_id", "as_of"} for name in fields)
            or task_identity.get("feature_columns") != fields):
        raise ValueError("snapshot feature columns differ from task identity")
    splits = metadata.get("split_sha256")
    if (not isinstance(splits, dict) or set(splits) != {"train", "validation", "test"}
            or task_identity.get("split_sha256") != splits):
        raise ValueError("snapshot split hashes differ from task identity")
    for split in ("train", "validation"):
        if _sha256(snapshots / f"{split}.csv") != splits[split]:
            raise ValueError(f"{split} snapshot changed after preparation")
    if task_identity.get("task_json_sha256") != _sha256(task_path):
        raise ValueError("task JSON differs from task identity")

    family = {
        "id": "ltv_tabular_regression",
        "name": "Fixed-horizon customer value regression",
        "question": "Can a customer-history representation improve future positive-purchase value prediction?",
        "experiment": "Compare one PyTorch regression change on the same chronological validation rows using MAE.",
        "pitfalls": "Snapshot leakage, zero-heavy targets, outliers, customer overlap, and temporal drift.",
        "stages": ["ltv_prediction"],
        "requires": ["tabular_features", "observed_outcome_labels"],
        "source_repo": "LTVEvo",
        "source_url": "https://github.com/CharlesXu-HQ/LTVEvo",
    }
    catalog = package.load_catalog(extra_families=[family])
    snapshot = _task_snapshot(task, metadata, task_identity)
    references = _host_references()
    identity = {
        "source_commit": source_commit,
        "catalog_digest": package.catalog_digest(catalog),
        "implementation_digest": package.implementation_digest(),
        "adapter_sha256": _sha256(Path(__file__)),
        "host_reference_sha256": {name: item["sha256"] for name, item in references.items()},
        "task_snapshot_sha256": _digest(snapshot),
        **task_identity,
    }
    family_report = package.applicability(snapshot, catalog)
    decision_report = package.decision_applicability(snapshot, catalog)
    training_report = package.training_applicability(snapshot, catalog)
    ready_families = {item["family_id"] for item in family_report if item["status"] == "ready"}
    ready_decisions = {item["check_id"] for item in decision_report if item["status"] == "ready"}
    ready_training = {item["pattern_id"] for item in training_report if item["status"] == "ready"}
    prompt_context = {
        "applicability": family_report,
        "method_applicability": package.method_applicability(snapshot, catalog),
        "decision_applicability": decision_report,
        "training_applicability": training_report,
        "ready_families": [item for item in catalog["families"] if item["id"] in ready_families],
        "ready_decision_checks": [item for item in catalog["decision_checks"] if item["id"] in ready_decisions],
        "ready_training_patterns": [item for item in catalog["training_patterns"] if item["id"] in ready_training],
        "train_profile": _train_profile(snapshots, fields),
        "host_references": references,
    }
    return {"identity": identity, "task_snapshot": snapshot,
            "catalog": catalog, "prompt_context": prompt_context}
