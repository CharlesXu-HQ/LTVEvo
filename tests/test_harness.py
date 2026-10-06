"""ModelEvoHarness context stays bound to a frozen, prediction-only LTV task."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from ltvevo.data import FEATURE_COLUMNS
from ltvevo.harness import build_harness_context
from ltvevo.search import _identity
from ltvevo.task import TaskSpec
from model_evo_harness import applicability, implementation_digest


def _prepared_task(tmp_path: Path) -> tuple[Path, Path, dict]:
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps({
        "raw_path": "retail.csv", "raw_sha256": "a" * 64,
        "horizon_days": 30, "as_of_start": "2009-12-01", "as_of_end": "2010-05-01",
        "train_end": "2010-01-01", "validation_start": "2010-03-01",
        "validation_end": "2010-03-01", "test_start": "2010-05-01",
        "test_end": "2010-05-01", "seed": 7,
        "target_policy": "positive_purchase_amount", "primary_metric": "mae",
        "evaluator_version": "1",
    }), encoding="utf-8")
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    for split, targets in (("train", [0.0, 20.0]),
                           ("validation", [1.0, 2.0]),
                           ("test", [987654.0, 987654.0])):
        values = {name: [1.0, 2.0] for name in FEATURE_COLUMNS}
        pd.DataFrame({"customer_id": ["u1", "u2"],
                      "as_of": ["2010-01-01"] * 2,
                      "target": targets, **values}).to_csv(snapshots / f"{split}.csv", index=False)
    hashes = {split: hashlib.sha256((snapshots / f"{split}.csv").read_bytes()).hexdigest()
              for split in ("train", "validation", "test")}
    (snapshots / "metadata.json").write_text(json.dumps({
        "task_fingerprint": TaskSpec.load(task_path).fingerprint,
        "raw_sha256": "a" * 64,
        "feature_columns": FEATURE_COLUMNS,
        "target_policy": "positive_purchase_amount",
        "split_sha256": hashes,
    }), encoding="utf-8")
    task_identity, _, _ = _identity(task_path, snapshots, "cpu", "image")
    return task_path, snapshots, task_identity


def test_real_catalog_marks_ltv_ready_and_other_stages_inapplicable(tmp_path: Path) -> None:
    task_path, snapshots, task_identity = _prepared_task(tmp_path)
    (snapshots / "test.csv").unlink()  # Proposal context only needs the frozen holdout hash.
    context = build_harness_context(task_path, snapshots, task_identity)
    snapshot = context["task_snapshot"]

    assert set(context) == {"identity", "task_snapshot", "catalog", "prompt_context"}
    assert snapshot["stage"] == "ltv_prediction"
    assert snapshot["framework"] == "pytorch"
    assert snapshot["objective"] == {"name": "mae", "direction": "min"}
    assert snapshot["fields"] == FEATURE_COLUMNS
    assert snapshot["evaluation_protocol"]["split"] == "chronological_purged_train_validation"
    assert snapshot["evaluation_protocol"]["horizon_days"] == "30"
    statuses = {entry["family_id"]: entry["status"]
                for entry in applicability(snapshot, context["catalog"])}
    assert statuses["ltv_tabular_regression"] == "ready"
    assert statuses["slate_reranking"] == "other_stage"
    assert statuses["decision_mapping"] == "other_stage"
    prompt = context["prompt_context"]
    assert "ltv_tabular_regression" in {item["id"] for item in prompt["ready_families"]}
    assert "decision_mapping" not in {item["id"] for item in prompt["ready_families"]}
    assert "weight_decay" in {item["id"] for item in prompt["ready_training_patterns"]}
    assert "evaluation_protocol" in {item["id"] for item in prompt["ready_decision_checks"]}
    assert context["prompt_context"]["train_profile"]["target_mean"] == 10.0
    assert context["prompt_context"]["train_profile"]["zero_target_fraction"] == 0.5
    assert "987654" not in json.dumps(context, sort_keys=True)


def test_identity_records_pinned_package_and_host_sources(tmp_path: Path) -> None:
    task_path, snapshots, task_identity = _prepared_task(tmp_path)
    context = build_harness_context(task_path, snapshots, task_identity)
    identity = context["identity"]
    refs = context["prompt_context"]["host_references"]

    assert len(identity["source_commit"]) == 40
    assert identity["implementation_digest"] == implementation_digest()
    assert identity["catalog_digest"]
    assert identity["adapter_sha256"]
    assert identity["task_fingerprint"] == task_identity["task_fingerprint"]
    assert identity["raw_sha256"] == task_identity["raw_sha256"]
    assert identity["split_sha256"] == task_identity["split_sha256"]
    assert identity["evaluator_version"] == task_identity["evaluator_version"]
    assert set(refs) == {"baseline.py", "torch_mlp.py", "zero.py"}
    assert identity["host_reference_sha256"] == {
        name: reference["sha256"] for name, reference in refs.items()}
    assert all(reference["provenance"] == "host" for reference in refs.values())
    assert all(hashlib.sha256(reference["content"].encode()).hexdigest() == reference["sha256"]
               for reference in refs.values())
    assert "reference_reads" not in context["prompt_context"]


def test_identity_changes_with_task_and_implementation(tmp_path: Path, monkeypatch) -> None:
    import model_evo_harness

    task_path, snapshots, task_identity = _prepared_task(tmp_path)
    original = build_harness_context(task_path, snapshots, task_identity)["identity"]
    task_json = json.loads(task_path.read_text())
    task_json["raw_sha256"] = "b" * 64
    task_path.write_text(json.dumps(task_json))
    metadata_path = snapshots / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["raw_sha256"] = "b" * 64
    metadata["task_fingerprint"] = TaskSpec.load(task_path).fingerprint
    metadata_path.write_text(json.dumps(metadata))
    changed_task, _, _ = _identity(task_path, snapshots, "cpu", "image")
    changed_dataset = build_harness_context(task_path, snapshots, changed_task)["identity"]
    assert changed_dataset != original
    assert changed_dataset["task_fingerprint"] != original["task_fingerprint"]

    monkeypatch.setattr(model_evo_harness, "implementation_digest", lambda: "b" * 64)
    changed_implementation = build_harness_context(task_path, snapshots, changed_task)["identity"]
    assert changed_implementation != changed_dataset
    assert changed_implementation["implementation_digest"] == "b" * 64


def test_import_outside_pinned_submodule_fails_clearly(tmp_path: Path, monkeypatch) -> None:
    import model_evo_harness

    task_path, snapshots, task_identity = _prepared_task(tmp_path)
    monkeypatch.setattr(model_evo_harness, "__file__", str(tmp_path / "other" / "__init__.py"))
    with pytest.raises(RuntimeError, match="pinned submodule"):
        build_harness_context(task_path, snapshots, task_identity)
