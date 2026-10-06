"""Dataset-bound lessons expose validation evidence with auditable provenance."""

import hashlib
import json

from ltvevo.experience import load_experience


IDENTITY = {"task_fingerprint": "task-a", "raw_sha256": "data-a",
            "split_sha256": {"train": "t", "validation": "v", "test": "h"},
            "evaluator_version": "1"}


def _journal(path, task, steps, *, version=2):
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"version": version, "task": task, "steps": steps,
                                "agent": {"api_key": "private-key"},
                                "final": {"test_mae": 9999}}))


def test_load_experience_keeps_exact_dataset_and_validation_only(tmp_path):
    current = tmp_path / "runs" / "current" / "journal.json"
    step = {"id": "step-001", "hypothesis": "log target helps", "status": "evaluated",
            "metrics": {"mae": 2.0, "test_mae": 9999},
            "reflection": {"verdict": "consistent", "lesson": "keep this transform",
                           "technical_experience": {"lesson": "try the transform",
                                                    "implementation_status": "verified"}}}
    same = tmp_path / "runs" / "same" / "journal.json"
    _journal(same, IDENTITY, [step])
    _journal(tmp_path / "runs" / "old_same" / "journal.json", IDENTITY, [step], version=1)
    _journal(tmp_path / "runs" / "different" / "journal.json",
             {**IDENTITY, "raw_sha256": "data-b"}, [step])

    lessons = load_experience(current, IDENTITY)

    assert [item["run"] for item in lessons] == ["same", "old_same"]
    assert lessons[0]["validation_mae"] == 2.0
    assert lessons[0]["source"]["journal_sha256"] == hashlib.sha256(same.read_bytes()).hexdigest()
    assert lessons[0]["source"]["step_id"] == "step-001"
    assert lessons[0]["implementation_reliability"] == "unverified"
    assert "private-key" not in json.dumps(lessons)
    assert "9999" not in json.dumps(lessons)


def test_load_experience_records_verified_host_check_and_harness_digest(tmp_path):
    current = tmp_path / "runs" / "current" / "journal.json"
    harness_identity = {"source_commit": "pinned"}
    task = {**IDENTITY, "harness_identity": harness_identity}
    _journal(tmp_path / "runs" / "verified" / "journal.json", task, [{
        "id": "step-002", "sha256": "b" * 64, "hypothesis": "new model",
        "status": "evaluated", "metrics": {"mae": 1.5},
        "implementation_check": {"status": "verified", "evidence": "host checked source"},
        "reflection": {"verdict": "consistent", "lesson": "source matches hypothesis"},
    }])

    lesson = load_experience(current, IDENTITY)[0]

    assert lesson["implementation_reliability"] == "verified"
    assert lesson["source"]["candidate_sha256"] == "b" * 64
    assert lesson["source"]["step_id"] == "step-002"
    assert lesson["source"]["harness_identity_sha256"] == hashlib.sha256(
        b'{"source_commit":"pinned"}').hexdigest()


def test_load_experience_preserves_host_contradiction(tmp_path):
    current = tmp_path / "runs" / "current" / "journal.json"
    _journal(tmp_path / "runs" / "failed" / "journal.json", IDENTITY, [{
        "id": "step-001", "hypothesis": "exact top-k", "status": "failed",
        "implementation_check": {"status": "contradicted",
                                 "findings": ["quantile ties exceed k"]},
        "reflection": {"verdict": "invalid", "lesson": "fix top-k",
                       "technical_experience": {"implementation_status": "contradicted"}},
    }])

    lesson = load_experience(current, IDENTITY)[0]
    assert lesson["implementation_reliability"] == "contradicted"
    assert lesson["technical_experience"]["implementation_status"] == "contradicted"


def test_load_experience_selects_latest_reflections_deterministically(tmp_path):
    current = tmp_path / "runs" / "current" / "journal.json"
    def step(identifier):
        return {"id": identifier, "hypothesis": identifier, "status": "evaluated",
                "metrics": {"mae": 2.0},
                "reflection": {"verdict": "inconclusive", "lesson": identifier}}

    _journal(tmp_path / "runs" / "z_new" / "journal.json", IDENTITY,
             [step("step-001"), step("step-002")])
    _journal(tmp_path / "runs" / "a_old" / "journal.json", IDENTITY,
             [step("step-003")])

    assert [item["source"]["step_id"] for item in load_experience(current, IDENTITY, limit=2)] == [
        "step-002", "step-001"]
    assert load_experience(current, IDENTITY, limit=0) == []
