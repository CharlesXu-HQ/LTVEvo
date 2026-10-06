"""The Agent sees validation evidence; final test labels stay sealed until finalization."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ltvevo.data import FEATURE_COLUMNS
from ltvevo.provider import ApiProvider, IncompleteResponseError, request_json
from ltvevo.sandbox import DockerSandbox
from ltvevo.sandbox_worker import _checked_predictions
from ltvevo.search import finalize_search, run_search
from ltvevo.task import TaskSpec


def _task_and_snapshots(tmp_path: Path) -> tuple[Path, Path]:
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps({
        "raw_path": "retail.xlsx", "raw_sha256": "a" * 64,
        "horizon_days": 30, "as_of_start": "2009-12-01", "as_of_end": "2010-05-01",
        "train_end": "2010-01-01", "validation_start": "2010-03-01",
        "validation_end": "2010-03-01", "test_start": "2010-05-01",
        "test_end": "2010-05-01", "seed": 7,
        "target_policy": "positive_purchase_amount", "primary_metric": "mae",
        "evaluator_version": "1",
    }))
    root = tmp_path / "snapshots"
    root.mkdir()
    for split, targets in (("train", [1.0, 3.0]),
                           ("validation", [2.0, 4.0]),
                           ("test", [9999.0, 9999.0])):
        rows = {name: [1.0, 2.0] for name in FEATURE_COLUMNS}
        frame = pd.DataFrame({"customer_id": ["u1", "u2"], "as_of": ["2010-01-01"] * 2,
                              "target": targets, **rows})
        frame.to_csv(root / f"{split}.csv", index=False)
    metadata = {"task_fingerprint": TaskSpec.load(task_path).fingerprint,
                "raw_sha256": "a" * 64, "feature_columns": FEATURE_COLUMNS,
                "target_policy": "positive_purchase_amount",
                "split_sha256": {split: hashlib.sha256((root / f"{split}.csv").read_bytes()).hexdigest()
                                 for split in ("train", "validation", "test")}}
    (root / "metadata.json").write_text(json.dumps(metadata))
    return task_path, root


def test_search_seals_test_labels_resumes_and_finalizes_once(tmp_path, monkeypatch):
    from ltvevo import search

    task_path, snapshots = _task_and_snapshots(tmp_path)
    journal_path = tmp_path / "runs" / "trial" / "journal.json"
    seen_inputs = []
    contexts = []
    reflections = []

    class FakeSandbox:
        def __init__(self, image):
            assert image == "fake-image"

        def predict(self, candidate, train, target, **kwargs):
            assert list(train.columns) == [*FEATURE_COLUMNS, "target"]
            assert list(target.columns) == FEATURE_COLUMNS
            assert "customer_id" not in target and "as_of" not in target
            seen_inputs.append((Path(candidate).name, len(target)))
            name = Path(candidate).name
            values = ([1.0, 1.0] if name == "baseline.py" else
                      [0.0, 0.0] if name == "zero.py" else
                          [1.0, 3.0] if name == "torch_mlp.py" else [2.0, 4.0])
            return {"predictions": np.asarray(values), "model_device": "cpu",
                    "cuda_peak_bytes": 0}

    monkeypatch.setattr(search, "DockerSandbox", FakeSandbox)

    def propose(_provider, context):
        contexts.append(json.dumps(context))
        return {"action": "experiment", "hypothesis": "A new model reduces MAE",
                "expected_result": "MAE below the seed model", "candidate_py":
                "import torch\n\n"
                "def fit_predict(train, validation, *, feature_columns, target_column, device, seed):\n"
                "    return torch.full((len(validation),), 2.0).numpy()\n", "feature_requests": []}

    monkeypatch.setattr(search, "propose_candidate", propose)
    def reflect(_provider, observation):
        reflections.append(observation)
        return {"verdict": "consistent", "evidence": "validation MAE fell",
                "lesson": "try similar scaling", "next_direction": "inspect calibration"}
    monkeypatch.setattr(search, "reflect_experiment", reflect)
    monkeypatch.setattr(search, "review_anomaly", lambda *_: {
        "decision": "allow", "evidence": "no leakage path", "next_direction": "finalize"})
    provider = ApiProvider("https://example.com", "test-model", "secret")

    journal = run_search(task_path, snapshots, journal_path, provider, steps=1,
                         device="cpu", docker_image="fake-image")
    assert len(journal["steps"]) == 1
    assert journal["version"] == 2
    assert journal["zero_reference"]["metrics"]["mae"] == 3.0
    assert journal["best_id"] == "step-001"
    assert journal["steps"][0]["selection"]["decision"] == "promote"
    assert journal["steps"][0]["paired_vs_reference"]["ci_lower"] > 0
    assert journal["steps"][0]["reference_id"] == "torch_seed"
    assert reflections[0]["reference_metrics"] == journal["torch_seed"]["metrics"]
    assert reflections[0]["paired_vs_reference"]["mae_improvement"] > 0
    assert set(journal["state_profile"]) == {"train", "validation"}
    assert journal["state_profile"]["validation"]["target_mean"] == 3.0
    assert "state_profile" in contexts[0]
    assert "period_metrics" in contexts[0]
    assert journal["baseline"]["period_metrics"][0]["period"] == "2010-01-01"
    assert contexts and all("9999" not in context for context in contexts)
    assert len(seen_inputs) == 4

    resumed = run_search(task_path, snapshots, journal_path, provider, steps=1,
                         device="cpu", docker_image="fake-image")
    assert resumed["best_id"] == journal["best_id"]
    assert len(seen_inputs) == 4

    finalized = finalize_search(journal_path, snapshots, device="cpu", docker_image="fake-image")
    assert finalized["final"]["champion_id"] == "step-001"
    assert "paired_mae_interval" in finalized["final"]
    assert "paired_vs_zero" in finalized["final"]
    assert finalized["final"]["zero_reference"]["metrics"]["mae"] > 0
    assert len(seen_inputs) == 7
    finalized_again = finalize_search(journal_path, snapshots, device="cpu", docker_image="fake-image")
    assert finalized_again["final"] == finalized["final"]
    assert len(seen_inputs) == 7

    test_prediction = journal_path.parent / finalized["final"]["champion"]["predictions"]
    original_prediction = test_prediction.read_bytes()
    test_prediction.write_bytes(original_prediction.replace(b"2.0", b"8.0"))
    with pytest.raises(ValueError, match="prediction artifact changed"):
        finalize_search(journal_path, snapshots, device="cpu", docker_image="fake-image")
    test_prediction.write_bytes(original_prediction)

    test_file = snapshots / "test.csv"
    original = test_file.read_bytes()
    test_file.write_bytes(original.replace(b"9999.0", b"8888.0"))
    with pytest.raises(ValueError, match="test snapshot changed"):
        finalize_search(journal_path, snapshots, device="cpu", docker_image="fake-image",
                        provider=provider)
    test_file.write_bytes(original)

    analysis_calls = []
    def analyze(_provider, evidence):
        analysis_calls.append(evidence)
        return {"summary": "test MAE improved", "evidence": "paired interval",
                "limitations": "purchase proxy", "next_experiment": "new frozen task"}
    monkeypatch.setattr(search, "analyze_final_report", analyze)
    with_analysis = finalize_search(journal_path, snapshots, device="cpu",
                                    docker_image="fake-image", provider=provider)
    assert with_analysis["final"]["analysis"]["summary"] == "test MAE improved"
    assert len(analysis_calls) == 1 and len(seen_inputs) == 7
    finalize_search(journal_path, snapshots, device="cpu", docker_image="fake-image",
                    provider=provider)
    assert len(analysis_calls) == 1 and len(seen_inputs) == 7


def test_harness_research_is_recorded_and_identity_freezes_resume(tmp_path, monkeypatch):
    from ltvevo import harness, search

    task_path, snapshots = _task_and_snapshots(tmp_path)
    journal_path = tmp_path / "runs" / "harness" / "journal.json"
    identity = {"digest": "first"}
    seen = {"test_calls": 0}

    def runtime(_task_path, _snapshots, _task_identity):
        return {"identity": dict(identity),
                "task_snapshot": {"stage": "ltv_prediction", "objective": {
                    "name": "mae", "direction": "min"}},
                "catalog": {}, "prompt_context": {}}

    class FakeSandbox:
        def __init__(self, _image):
            pass

        def predict(self, candidate, _train, target, **_kwargs):
            if len(target) == 2 and "9999" in str(target):
                seen["test_calls"] += 1
            value = 2.0 if Path(candidate).name != "zero.py" else 0.0
            return {"predictions": np.full(len(target), value), "model_device": "cpu"}

    research = {"direction": "Test log target", "mechanism": "Transform skewed values",
                "input_fields": [FEATURE_COLUMNS[0]]}
    monkeypatch.setattr(harness, "build_harness_context", runtime)
    monkeypatch.setattr(search, "DockerSandbox", FakeSandbox)

    def propose(_provider, context):
        assert context["harness_runtime"]["identity"] == identity
        assert "9999" not in json.dumps(context)
        return {"action": "experiment", "hypothesis": "Reduce skew error",
                "expected_result": "Lower validation MAE", "research": research,
                "reference_reads": [{"path": "models/pytorch/composition.py", "sha256": "a" * 64}],
                "candidate_py": "import torch\n"
                "def fit_predict(train, validation, *, feature_columns, target_column, device, seed):\n"
                "    return torch.full((len(validation),), 2.0)\n"}

    def reflect(_provider, observation):
        assert observation["research"] == research
        assert observation["harness_steps"][0]["research"] == research
        return {"verdict": "inconclusive", "evidence": "same validation set",
                "lesson": "No clear gain", "next_direction": "try another transform"}

    monkeypatch.setattr(search, "propose_candidate", propose)
    monkeypatch.setattr(search, "reflect_experiment", reflect)
    provider = ApiProvider("https://example.com", "test-model", "secret")
    journal = run_search(task_path, snapshots, journal_path, provider, steps=1,
                         device="cpu", docker_image="fake-image", harness="model-evo")
    assert journal["task"]["harness_identity"] == identity
    assert journal["steps"][0]["research"] == research
    assert journal["steps"][0]["reference_reads"][0]["sha256"] == "a" * 64
    with pytest.raises(ValueError, match="Harness mode changed"):
        run_search(task_path, snapshots, journal_path, provider, steps=1,
                   device="cpu", docker_image="fake-image", harness="none")
    identity["digest"] = "changed"
    with pytest.raises(ValueError, match="search task"):
        run_search(task_path, snapshots, journal_path, provider, steps=1,
                   device="cpu", docker_image="fake-image")
    with pytest.raises(ValueError, match="finalization task"):
        finalize_search(journal_path, snapshots, device="cpu", docker_image="fake-image")
    assert "final" not in json.loads(journal_path.read_text())


def test_zero_reference_can_win_validation_and_is_compared_on_final_test(tmp_path, monkeypatch):
    from ltvevo import search

    task_path, snapshots = _task_and_snapshots(tmp_path)
    validation = snapshots / "validation.csv"
    frame = pd.read_csv(validation)
    frame["target"] = 0.0
    frame.to_csv(validation, index=False)
    metadata_path = snapshots / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["split_sha256"]["validation"] = hashlib.sha256(validation.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata))
    journal_path = tmp_path / "runs" / "sparse" / "journal.json"
    calls = []

    class FakeSandbox:
        def __init__(self, _image):
            pass

        def predict(self, candidate, _train, target, **_kwargs):
            calls.append(Path(candidate).name)
            value = {"baseline.py": 5.0, "zero.py": 0.0, "torch_mlp.py": 2.0}[
                Path(candidate).name]
            return {"predictions": np.full(len(target), value), "model_device": "cpu"}

    monkeypatch.setattr(search, "DockerSandbox", FakeSandbox)
    journal = run_search(task_path, snapshots, journal_path, None, steps=0,
                         device="cpu", docker_image="fake-image")
    assert journal["best_id"] == "zero_reference"
    assert journal["zero_reference"]["metrics"]["mae"] == 0.0
    assert len(calls) == 3

    final = finalize_search(journal_path, snapshots, device="cpu", docker_image="fake-image")
    assert final["final"]["champion_id"] == "zero_reference"
    assert final["final"]["paired_vs_zero"]["mae_improvement"] == 0.0
    assert final["final"]["paired_mae_interval"]["mae_improvement"] < 0
    assert len(calls) == 5  # baseline and zero each evaluated once on test

    order = []
    def analyze(_provider, _evidence):
        order.append("high")
        if order.count("high") == 1:
            raise RuntimeError("temporary report failure")
        return {"summary": "test reversal", "evidence": "paired comparison",
                "limitations": "observational purchase proxy", "next_experiment": "new holdout"}

    def review(_provider, evidence):
        order.append("max")
        assert evidence["champion_id"] == "zero_reference"
        assert evidence["trigger_reasons"]
        if order.count("max") == 1:
            raise RuntimeError("temporary review failure")
        return {"finding": "validation gain reversed", "evidence": "held-out MAE",
                "limitations": "cannot attribute cause", "next_direction": "new frozen task"}

    monkeypatch.setattr(search, "analyze_final_report", analyze)
    monkeypatch.setattr(search, "review_final_anomaly", review)
    provider = ApiProvider("https://example.com", "test-model", "secret")
    with pytest.raises(RuntimeError, match="report failure"):
        finalize_search(journal_path, snapshots, device="cpu", docker_image="fake-image",
                        provider=provider)
    saved = json.loads(journal_path.read_text())
    assert "final" in saved and "analysis" not in saved["final"]
    assert len(calls) == 5

    with pytest.raises(RuntimeError, match="review failure"):
        finalize_search(journal_path, snapshots, device="cpu", docker_image="fake-image",
                        provider=provider)
    saved = json.loads(journal_path.read_text())
    assert "analysis" in saved["final"] and "anomaly_review" not in saved["final"]
    assert len(calls) == 5

    reviewed = finalize_search(journal_path, snapshots, device="cpu", docker_image="fake-image",
                               provider=provider)
    assert reviewed["best_id"] == "zero_reference"
    assert reviewed["final"]["champion_id"] == "zero_reference"
    assert reviewed["final"]["anomaly_review"]["effort"] == "max"
    assert any("reversed" in reason for reason in
               reviewed["final"]["anomaly_review"]["trigger_reasons"])
    assert order == ["high", "high", "max", "max"]
    finalize_search(journal_path, snapshots, device="cpu", docker_image="fake-image",
                    provider=provider)
    assert order == ["high", "high", "max", "max"] and len(calls) == 5


def test_zero_reference_respects_device_contract(monkeypatch):
    from ltvevo.candidates.zero import fit_predict

    validation = pd.DataFrame({"spend_90d": [1.0, 2.0]})
    result = fit_predict(pd.DataFrame(), validation, feature_columns=["spend_90d"],
                         target_column="target", device="cpu", seed=7)
    assert isinstance(result, torch.Tensor)
    assert result.device.type == "cpu" and result.tolist() == [0.0, 0.0]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        fit_predict(pd.DataFrame(), validation, feature_columns=["spend_90d"],
                    target_column="target", device="cuda", seed=7)


def test_old_journal_is_rejected_without_modification(tmp_path, monkeypatch):
    from ltvevo import search

    task_path, snapshots = _task_and_snapshots(tmp_path)
    journal_path = tmp_path / "runs" / "old" / "journal.json"

    class FakeSandbox:
        def __init__(self, _image):
            pass

        def predict(self, _candidate, _train, target, **_kwargs):
            return {"predictions": np.zeros(len(target)), "model_device": "cpu"}

    monkeypatch.setattr(search, "DockerSandbox", FakeSandbox)
    journal = run_search(task_path, snapshots, journal_path, None, steps=0,
                         device="cpu", docker_image="fake-image")
    journal["version"] = 1
    journal_path.write_text(json.dumps(journal))
    old_bytes = journal_path.read_bytes()
    with pytest.raises(ValueError, match="incompatible search journal version"):
        run_search(task_path, snapshots, journal_path, None, steps=0,
                   device="cpu", docker_image="fake-image")
    with pytest.raises(ValueError, match="incompatible search journal version"):
        finalize_search(journal_path, snapshots, device="cpu", docker_image="fake-image")
    assert journal_path.read_bytes() == old_bytes


def test_provider_enables_thinking_and_effort_without_disclosing_key(monkeypatch):
    from ltvevo import provider as module

    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return b'{"choices":[{"finish_reason":"stop","message":{"content":"{\\"ok\\":true}"}}]}'

    def fake_urlopen(request, timeout):
        captured.update(json.loads(request.data))
        return Response()

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    client = ApiProvider("https://api.deepseek.com", "deepseek-flash", "secret")
    assert "secret" not in repr(client)
    assert request_json(client, "max", [{"role": "user", "content": "Return JSON"}]) == {"ok": True}
    assert captured["thinking"] == {"type": "enabled"}
    assert captured["reasoning_effort"] == "max"
    assert ApiProvider("https://api.deepseek.com", "deepseek-flash", "secret",
                       thinking="disabled").thinking == "disabled"


def test_provider_error_keeps_reason_but_redacts_key(monkeypatch):
    from ltvevo import provider as module

    def reject(_request, timeout):
        raise urllib.error.HTTPError(
            "https://example.com/chat/completions", 400, "invalid", None,
            io.BytesIO(b'{"error":"reasoning_effort unsupported; secret leaked"}'))

    monkeypatch.setattr(module.urllib.request, "urlopen", reject)
    client = ApiProvider("https://example.com", "model", "secret")
    with pytest.raises(RuntimeError) as failure:
        request_json(client, "high", [{"role": "user", "content": "Return JSON"}])
    assert "reasoning_effort unsupported" in str(failure.value)
    assert "secret" not in str(failure.value)


def test_final_report_uses_high_then_anomaly_review_uses_max(monkeypatch):
    from ltvevo import agent

    efforts = []
    def fake_request(_provider, effort, _messages, *, max_tokens, timeout=None):
        efforts.append(effort)
        if effort == "high":
            return {"summary": "held-out loss", "evidence": "test MAE",
                    "limitations": "limited sample", "next_experiment": "new split"}
        return {"finding": "reversal", "evidence": "paired CI",
                "limitations": "cause unknown", "next_direction": "new frozen task"}

    monkeypatch.setattr(agent, "request_json", fake_request)
    provider = ApiProvider("https://example.com", "test-model", "secret")
    agent.analyze_final_report(provider, {"result": {}})
    agent.review_final_anomaly(provider, {"trigger_reasons": ["reversed"]})
    assert efforts == ["high", "max"]


def test_final_max_review_retries_length_once_with_larger_budget(monkeypatch):
    from ltvevo import agent

    attempts = []
    def fake_request(_provider, effort, messages, *, max_tokens, timeout=None):
        attempts.append((effort, max_tokens, messages[0]["content"]))
        if len(attempts) == 1:
            raise IncompleteResponseError("length")
        return {"finding": "test loss", "evidence": "negative paired interval",
                "limitations": "cause uncertain", "next_direction": "new frozen holdout"}

    monkeypatch.setattr(agent, "request_json", fake_request)
    provider = ApiProvider("https://example.com", "test-model", "secret")
    result = agent.review_final_anomaly(provider, {"trigger_reasons": ["test loss"]})
    assert result["finding"] == "test loss"
    assert [effort for effort, _, _ in attempts] == ["max", "max"]
    assert [budget for _, budget, _ in attempts] == [32768, 131072]
    assert "concise" in attempts[1][2].lower()


def test_final_max_review_length_retry_is_bounded(monkeypatch):
    from ltvevo import agent

    budgets = []
    def always_truncated(_provider, _effort, _messages, *, max_tokens, timeout=None):
        budgets.append(max_tokens)
        raise IncompleteResponseError("length")

    monkeypatch.setattr(agent, "request_json", always_truncated)
    provider = ApiProvider("https://example.com", "test-model", "secret")
    with pytest.raises(IncompleteResponseError, match="length"):
        agent.review_final_anomaly(provider, {"trigger_reasons": ["test loss"]})
    assert budgets == [32768, 131072]


def test_provider_exposes_length_finish_reason(monkeypatch):
    from ltvevo import provider as module

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return b'{"choices":[{"finish_reason":"length","message":{"content":""}}]}'

    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *_args, **_kwargs: Response())
    client = ApiProvider("https://example.com", "model", "secret")
    with pytest.raises(IncompleteResponseError) as failure:
        request_json(client, "max", [{"role": "user", "content": "Return JSON"}])
    assert failure.value.finish_reason == "length"


def test_sandbox_mounts_only_training_labels_and_prediction_features(monkeypatch, tmp_path):
    candidate = tmp_path / "candidate.py"
    candidate.write_text("def fit_predict(*args, **kwargs): pass\n")
    train = pd.DataFrame({"x": [1.0], "target": [2.0], "customer_id": ["private"]})
    predict = pd.DataFrame({"x": [3.0], "target": [9999.0], "customer_id": ["sealed"]})
    seen = {}

    def fake_run(command, **_kwargs):
        seen["command"] = command
        mounts = [command[index + 1] for index, item in enumerate(command[:-1])
                  if item == "--mount"]
        input_dir = Path(next(item for item in mounts if "dst=/input" in item).split(",")[1][4:])
        output_dir = Path(next(item for item in mounts if "dst=/output" in item).split(",")[1][4:])
        seen["train_columns"] = pd.read_csv(input_dir / "train.csv").columns.tolist()
        seen["predict_columns"] = pd.read_csv(input_dir / "predict.csv").columns.tolist()
        seen["predict_text"] = (input_dir / "predict.csv").read_text()
        pd.DataFrame({"prediction": [4.0]}).to_csv(output_dir / "predictions.csv", index=False)
        (output_dir / "response.json").write_text(
            json.dumps({"model_device": "cuda", "cuda_peak_bytes": 2048}))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("ltvevo.sandbox.subprocess.run", fake_run)
    result = DockerSandbox("test-image").predict(
        candidate, train, predict, feature_columns=["x"], target_column="target",
        seed=7, device="cuda")
    assert seen["train_columns"] == ["x", "target"]
    assert seen["predict_columns"] == ["x"]
    assert "9999" not in seen["predict_text"] and "sealed" not in seen["predict_text"]
    assert "--network" in seen["command"] and "none" in seen["command"]
    assert "--read-only" in seen["command"] and "--gpus" in seen["command"]
    assert result["cuda_peak_bytes"] == 2048


def test_experience_is_exact_task_bound_and_excludes_test_results(tmp_path):
    from ltvevo.search import _experience

    current = tmp_path / "runs" / "current" / "search.json"
    current.parent.mkdir(parents=True)
    identity = {"task_fingerprint": "task-a", "raw_sha256": "data-a",
                "split_sha256": {"train": "t", "validation": "v", "test": "h"},
                "evaluator_version": "1"}
    lesson = {"hypothesis": "log target helps", "status": "evaluated",
              "metrics": {"mae": 2.0},
              "reflection": {"verdict": "consistent", "lesson": "keep this transform"}}
    for name, task, version in (("same", identity, 2),
                                ("old_same", identity, 1),
                                ("different", {**identity, "raw_sha256": "data-b"}, 2)):
        path = tmp_path / "runs" / name / "search.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"version": version, "task": task, "steps": [lesson],
                                    "final": {"secret_test_result": 9999}}))
    selected = _experience(current, identity)
    assert len(selected) == 2
    assert {item["run"] for item in selected} == {"same", "old_same"}
    assert all(item["lesson"] == "keep this transform" for item in selected)
    assert all(item["implementation_reliability"] == "unverified" for item in selected)
    assert all(len(item["source"]["journal_sha256"]) == 64 for item in selected)
    assert "9999" not in json.dumps(selected)


def test_promotion_requires_paired_support_and_no_source_contradiction():
    from ltvevo import search

    best = {"id": "baseline", "metrics": {"mae": 10.0}}
    candidate = {"status": "evaluated", "metrics": {"mae": 9.0},
                 "paired_vs_reference": {"mae_improvement": 1.0, "ci_lower": -0.2},
                 "review": {"decision": "allow"},
                 "reflection": {"verdict": "consistent"},
                 "implementation_check": {"status": "unverified", "findings": []}}
    assert search._promotion_decision(candidate, best) == {
        "decision": "retain", "reason": "paired_gain_uncertain", "reference_id": "baseline",
        "policy": "paired_customer_bootstrap_95_v1"}
    candidate["paired_vs_reference"]["ci_lower"] = 0.2
    assert search._promotion_decision(candidate, best)["decision"] == "promote"
    candidate["implementation_check"] = {"status": "contradicted", "findings": ["silent fallback"]}
    assert search._promotion_decision(candidate, best)["reason"] == "implementation_contradicted"


def test_source_contradiction_is_recorded_without_running_candidate(tmp_path, monkeypatch):
    from ltvevo import search

    task_path, snapshots = _task_and_snapshots(tmp_path)
    journal_path = tmp_path / "runs" / "audited" / "journal.json"
    seen = []

    class FakeSandbox:
        def __init__(self, _image):
            pass

        def predict(self, candidate, _train, target, **_kwargs):
            seen.append(Path(candidate).name)
            return {"predictions": np.zeros(len(target)), "model_device": "cpu"}

    source = ("import torch\n"
              "def fit_predict(train, validation, *, feature_columns, target_column, device, seed):\n"
              "    prediction = torch.ones(len(validation))\n"
              "    if not torch.isfinite(prediction).all():\n"
              "        prediction = torch.zeros_like(prediction)\n"
              "    return prediction\n")
    monkeypatch.setattr(search, "DockerSandbox", FakeSandbox)
    monkeypatch.setattr(search, "propose_candidate", lambda *_: {
        "action": "experiment", "hypothesis": "model reduces MAE",
        "expected_result": "lower MAE", "candidate_py": source})
    monkeypatch.setattr(search, "reflect_experiment", lambda _provider, observation: {
        "verdict": "invalid", "evidence": observation["implementation_check"]["findings"][0],
        "lesson": "remove fallback", "next_direction": "retry"})
    provider = ApiProvider("https://example.com", "model", "secret")

    journal = run_search(task_path, snapshots, journal_path, provider, steps=1,
                         device="cpu", docker_image="fake-image")
    step = journal["steps"][0]
    assert step["status"] == "failed"
    assert step["implementation_check"]["status"] == "contradicted"
    assert step["selection"]["reason"] == "implementation_contradicted"
    assert seen == ["baseline.py", "zero.py", "torch_mlp.py"]

    for key in ("implementation_check", "reflection", "selection", "error"):
        step.pop(key)
    step["status"] = "pending"
    journal_path.write_text(json.dumps(journal))
    resumed = run_search(task_path, snapshots, journal_path, provider, steps=1,
                         device="cpu", docker_image="fake-image")
    assert resumed["steps"][0]["implementation_check"]["status"] == "contradicted"
    assert seen == ["baseline.py", "zero.py", "torch_mlp.py"]


def test_prediction_artifact_hash_detects_tampering(tmp_path):
    from ltvevo import search

    prediction = tmp_path / "predictions.csv"
    prediction.write_text("prediction\n1.0\n")
    entry = {"predictions": "predictions.csv", "predictions_sha256":
             hashlib.sha256(prediction.read_bytes()).hexdigest()}
    assert search._predictions(tmp_path, entry).tolist() == [1.0]
    prediction.write_text("prediction\n2.0\n")
    with pytest.raises(ValueError, match="prediction artifact changed"):
        search._predictions(tmp_path, entry)


def test_candidate_source_rejects_obvious_dynamic_imports():
    from ltvevo.search import _validate_source

    source = ("import torch\n"
              "def fit_predict(train, validation, *, feature_columns, target_column, device, seed):\n"
              "    module = __import__('os')\n"
              "    return torch.zeros(len(validation))\n")
    with pytest.raises(ValueError, match="dynamic execution"):
        _validate_source(source)
    indirect = source.replace("__import__('os')", "getattr(__builtins__, '__import__')('os')")
    with pytest.raises(ValueError, match="dynamic execution|builtins"):
        _validate_source(indirect)


def test_cuda_candidate_cannot_pass_with_cpu_predictions_after_dummy_gpu_work():
    # Even if a candidate touched CUDA, the output path must remain on CUDA.
    with pytest.raises(RuntimeError, match="CUDA torch.Tensor"):
        _checked_predictions(torch.tensor([1.0]), "cuda", 1)


def test_metric_tradeoff_triggers_max_review():
    from ltvevo.search import _anomaly_reasons

    reference = {"metrics": {"mae": 10.0, "rmse": 20.0, "top_decile_capture": 0.7}}
    candidate = {"metrics": {"mae": 9.0, "rmse": 40.0, "top_decile_capture": 0.3},
                 "negative_predictions": 1}
    reasons = _anomaly_reasons(candidate, reference)
    assert len(reasons) == 3


def test_final_clear_paired_loss_alone_triggers_max_review():
    from ltvevo.search import _final_anomaly_reasons

    journal = {"baseline": {"id": "baseline", "metrics": {"mae": 1.0}},
               "zero_reference": {"id": "zero_reference", "metrics": {"mae": 2.0}},
               "torch_seed": {"id": "torch_seed", "metrics": {"mae": 3.0}},
               "steps": [], "best_id": "baseline",
               "final": {"champion_id": "baseline",
                         "baseline": {"metrics": {"mae": 5.0}},
                         "champion": {"metrics": {"mae": 5.0}},
                         "paired_mae_interval": {"ci_upper": 0.0},
                         "paired_vs_zero": {"ci_upper": -1.0}}}
    reasons = _final_anomaly_reasons(journal)
    assert reasons == ["test paired 95% interval shows a loss versus zero reference"]


def test_test_snapshot_is_not_opened_until_finalization(tmp_path, monkeypatch):
    from ltvevo import search

    task_path, snapshots = _task_and_snapshots(tmp_path)
    journal_path = tmp_path / "runs" / "trial" / "search.json"

    class FakeSandbox:
        def __init__(self, _image):
            pass

        def predict(self, _candidate, _train, target, **_kwargs):
            return {"predictions": np.zeros(len(target)), "model_device": "cpu"}

    monkeypatch.setattr(search, "DockerSandbox", FakeSandbox)
    run_search(task_path, snapshots, journal_path, None, steps=0,
               device="cpu", docker_image="fake-image")
    test_file = snapshots / "test.csv"
    test_file.write_text(test_file.read_text().replace("9999.0", "8888.0"))
    run_search(task_path, snapshots, journal_path, None, steps=0,
               device="cpu", docker_image="fake-image")
    with pytest.raises(ValueError, match="test snapshot changed"):
        finalize_search(journal_path, snapshots, device="cpu", docker_image="fake-image")
