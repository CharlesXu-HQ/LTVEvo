"""Validation lessons from finalized runs of the same frozen task and dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


_IDENTITY_KEYS = ("task_fingerprint", "raw_sha256", "split_sha256", "evaluator_version")
_TECHNICAL_FIELDS = ("lesson", "evidence", "uncertainty", "next_test",
                     "component_assessments")


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_experience(journal_path: Path, task: dict, limit: int = 8) -> list[dict]:
    """Return recent validation lessons, with source and implementation provenance."""
    if limit <= 0:
        return []
    lessons = []
    for path in sorted(journal_path.parent.parent.glob("*/*.json"), reverse=True):
        if path == journal_path:
            continue
        try:
            source = path.read_bytes()
            previous = json.loads(source)
            identity = previous["task"]
            if (previous.get("version") not in {1, 2} or "final" not in previous
                    or any(identity[key] != task[key] for key in _IDENTITY_KEYS)):
                continue
            journal_sha256 = hashlib.sha256(source).hexdigest()
            harness_identity = identity.get("harness_identity")
            for step in reversed(previous.get("steps", [])):
                if "reflection" not in step:
                    continue
                reflection = step["reflection"]
                check = step.get("implementation_check") or {}
                reliability = (check.get("status") if check.get("status") in
                               {"verified", "contradicted"} else "unverified")
                provenance = {"journal_sha256": journal_sha256}
                if isinstance(step.get("id"), str):
                    provenance["step_id"] = step["id"]
                if isinstance(step.get("sha256"), str):
                    provenance["candidate_sha256"] = step["sha256"]
                if isinstance(harness_identity, dict):
                    provenance["harness_identity_sha256"] = _digest(harness_identity)
                lesson = {"run": path.parent.name,
                          "hypothesis": step["hypothesis"][:500],
                          "validation_mae": step.get("metrics", {}).get("mae"),
                          "status": step["status"],
                          "verdict": reflection["verdict"],
                          "lesson": reflection["lesson"][:500],
                          "source": provenance,
                          "implementation_reliability": reliability}
                if "research" in step:
                    lesson["research"] = step["research"]
                if isinstance(reflection.get("technical_experience"), dict):
                    technical = reflection["technical_experience"]
                    lesson["technical_experience"] = {
                        key: technical[key] for key in _TECHNICAL_FIELDS if key in technical}
                    lesson["technical_experience"]["implementation_status"] = reliability
                    audit = step.get("change_audit") or {}
                    lesson["technical_experience"]["attribution"] = (
                        technical.get("attribution", "unverified")
                        if reliability == "verified" and audit.get("status") == "verified"
                        else "unverified")
                lessons.append(lesson)
                if len(lessons) >= limit:
                    return lessons
        except (KeyError, ValueError, OSError, TypeError, UnicodeError):
            continue
    return lessons
