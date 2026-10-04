"""Resumable Agent-led experiments on one frozen LTV prediction task."""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .agent import (analyze_final_report, diagnose_history, propose_candidate,
                    reflect_experiment, review_anomaly, review_final_anomaly)
from .evaluation import paired_mae_interval, summarize
from .provider import ApiProvider
from .sandbox import DockerSandbox
from .task import TaskSpec


_TARGET = "target"
_ID = "customer_id"
_JOURNAL_VERSION = 2
_ALLOWED_IMPORTS = {"__future__", "math", "numpy", "pandas", "torch"}
_DYNAMIC_BUILTINS = {"__import__", "eval", "exec", "compile", "open",
                     "getattr", "vars", "globals", "locals"}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _identity(task_path: Path, snapshots: Path, device: str, image: str) -> tuple[dict, list[str], int]:
    task = TaskSpec.load(task_path)
    metadata = json.loads((snapshots / "metadata.json").read_text())
    features = metadata.get("feature_columns")
    hashes = metadata.get("split_sha256")
    if (metadata.get("task_fingerprint") != task.fingerprint
            or metadata.get("raw_sha256") != task.raw_sha256
            or metadata.get("target_policy") != task.target_policy):
        raise ValueError("snapshot metadata does not match the frozen task")
    if (not isinstance(features, list) or not features or len(features) != len(set(features))
            or any(not isinstance(name, str) for name in features)
            or _TARGET in features or _ID in features or "as_of" in features):
        raise ValueError("snapshot feature_columns are invalid")
    if not isinstance(hashes, dict) or any(split not in hashes for split in
                                           ("train", "validation", "test")):
        raise ValueError("snapshot metadata must contain all split SHA-256 hashes")
    for split in ("train", "validation"):
        if _sha(snapshots / f"{split}.csv") != hashes[split]:
            raise ValueError(f"{split} snapshot changed after preparation")
    return ({"task_fingerprint": task.fingerprint, "task_json_sha256": _sha(task_path),
             "raw_sha256": task.raw_sha256, "split_sha256": hashes,
             "evaluator_version": task.evaluator_version, "feature_columns": features,
             "target_policy": task.target_policy, "device": device, "docker_image": image},
            features, task.seed)


def _agent_info(provider: ApiProvider | None) -> dict | None:
    if provider is None:
        return None
    return {"url": provider.url, "model": provider.model, "thinking": provider.thinking,
            "iteration_effort": provider.iteration_effort,
            "review_effort": provider.review_effort}


def _candidate_path(root: Path, entry: dict) -> Path:
    path = (root / entry["candidate"]).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file() or _sha(path) != entry["sha256"]:
        raise ValueError(f"candidate snapshot changed: {entry['id']}")
    return path


def _entries(journal: dict) -> list[dict]:
    return [journal["baseline"], journal["zero_reference"],
            journal["torch_seed"], *journal["steps"]]


def _find(journal: dict, identifier: str) -> dict:
    return next(entry for entry in _entries(journal) if entry["id"] == identifier)


def _validate_source(source: str) -> str:
    # This is a scope check, not the security boundary; Docker supplies the isolation.
    if not isinstance(source, str) or not source.strip():
        raise ValueError("candidate_py must contain Python source")
    source = source.strip()
    if source.startswith("```python"):
        source = source.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    tree = ast.parse(source)
    torch_imported = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in _ALLOWED_IMPORTS:
                    raise ValueError(f"candidate imports unsupported module: {root}")
                torch_imported |= root == "torch"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in _ALLOWED_IMPORTS:
                raise ValueError(f"candidate imports unsupported module: {root}")
            torch_imported |= root == "torch"
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id in _DYNAMIC_BUILTINS):
            raise ValueError(f"candidate uses unsupported dynamic execution: {node.func.id}")
        elif isinstance(node, ast.Name) and node.id == "__builtins__":
            raise ValueError("candidate accesses unsupported builtins")
    if not torch_imported:
        raise ValueError("candidate must use PyTorch")
    if not any(isinstance(node, ast.FunctionDef) and node.name == "fit_predict" for node in tree.body):
        raise ValueError("candidate must define fit_predict")
    compile(tree, "candidate.py", "exec")
    return source + "\n"


def _evaluate(root: Path, candidate: Path, split: str, snapshots: Path,
              sandbox: DockerSandbox, features: list[str], seed: int, device: str,
              *, identifier: str) -> dict:
    train = pd.read_csv(snapshots / "train.csv")
    target = pd.read_csv(snapshots / f"{split}.csv")
    required_train = [*features, _TARGET]
    required_target = [*features, _TARGET, _ID]
    if any(name not in train for name in required_train) or any(name not in target for name in required_target):
        raise ValueError("snapshot columns do not match metadata")
    result = sandbox.predict(candidate, train.loc[:, required_train],
                             target.loc[:, features], feature_columns=features,
                             target_column=_TARGET, seed=seed, device=device)
    predictions = np.asarray(result["predictions"], dtype=float)
    metrics = summarize(target[_TARGET], predictions, target[_ID])
    path = root / "predictions" / f"{split}-{identifier}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"prediction": predictions}).to_csv(path, index=False)
    return {"metrics": metrics, "predictions": str(path.relative_to(root)),
            "model_device": result.get("model_device"),
            "prediction_device": result.get("prediction_device"),
            "negative_predictions": int((predictions < 0).sum()),
            "cuda_peak_bytes": int(result.get("cuda_peak_bytes", 0))}


def _predictions(root: Path, entry: dict) -> np.ndarray:
    return pd.read_csv(root / entry["predictions"])["prediction"].to_numpy(float)


def _paired(root: Path, baseline: dict, candidate: dict, snapshots: Path,
            split: str, seed: int) -> dict:
    frame = pd.read_csv(snapshots / f"{split}.csv", usecols=[_ID, _TARGET])
    return paired_mae_interval(frame[_TARGET], _predictions(root, baseline),
                               _predictions(root, candidate), frame[_ID], seed=seed)


def _experience(journal_path: Path, task: dict, limit: int = 8) -> list[dict]:
    """Load validation lessons from finalized sibling runs with an identical dataset/task."""
    lessons = []
    for path in sorted(journal_path.parent.parent.glob("*/*.json"), reverse=True):
        if path == journal_path:
            continue
        try:
            previous = json.loads(path.read_text())
            current = previous["task"]
            keys = ("task_fingerprint", "raw_sha256", "split_sha256", "evaluator_version")
            if (previous.get("version") not in {1, _JOURNAL_VERSION} or "final" not in previous
                    or any(current[key] != task[key] for key in keys)):
                continue
            for step in reversed(previous.get("steps", [])):
                if "reflection" not in step:
                    continue
                lessons.append({"run": path.parent.name, "hypothesis": step["hypothesis"][:500],
                                "validation_mae": step.get("metrics", {}).get("mae"),
                                "status": step["status"],
                                "verdict": step["reflection"]["verdict"],
                                "lesson": step["reflection"]["lesson"][:500]})
                if len(lessons) >= limit:
                    return lessons
        except (KeyError, ValueError, OSError, TypeError):
            continue
    return lessons


def _context(journal: dict, root: Path) -> dict:
    chosen = {"baseline", "zero_reference", "torch_seed", journal["best_id"]}
    if journal["steps"]:
        chosen.add(journal["steps"][-1]["id"])
    available = {}
    for entry in _entries(journal):
        if entry["id"] in chosen:
            available[entry["id"]] = {"candidate_py": _candidate_path(root, entry).read_text(),
                                      "metrics": entry.get("metrics"), "status": entry["status"]}
    history = [{key: entry[key] for key in ("id", "status", "hypothesis", "expected_result",
                                           "reference_id", "metrics", "paired_mae_interval",
                                           "paired_vs_reference", "reflection", "error")
                if key in entry} for entry in _entries(journal)]
    return {"task_fingerprint": journal["task"]["task_fingerprint"],
            "raw_sha256": journal["task"]["raw_sha256"],
            "target_policy": journal["task"]["target_policy"],
            "feature_columns": journal["task"]["feature_columns"],
            "primary_metric": "mae", "best_id": journal["best_id"],
            "state_profile": journal["state_profile"],
            "history": history, "available": available,
            "experience": journal.get("experience", []),
            "diagnoses": journal.get("diagnoses", [])}


def _feature_requests(root: Path, journal: dict) -> None:
    lines = ["# Dataset feature requests", "",
             "These proposals require human review and a new frozen task/dataset before use.", ""]
    for step in journal["steps"]:
        for request in step.get("feature_requests", []):
            lines.extend([f"## {step['id']}: {request['field']}", "",
                          f"- Evidence: {request['evidence']}",
                          f"- Source: {request['source']}",
                          f"- Available at: {request['availability_time']}",
                          f"- Leakage risk: {request['leakage_risk']}",
                          f"- Validation plan: {request['validation_plan']}", ""])
    if len(lines) > 4:
        (root / "feature_requests.md").write_text("\n".join(lines))


def _anomaly_reasons(candidate: dict, reference: dict) -> list[str]:
    reasons = []
    current, previous = candidate["metrics"], reference["metrics"]
    if candidate["negative_predictions"]:
        reasons.append("negative predictions for a nonnegative target")
    if current["mae"] < 0.5 * previous["mae"]:
        reasons.append("validation MAE fell by more than half")
    if current["mae"] < previous["mae"] and current["rmse"] > 1.5 * previous["rmse"]:
        reasons.append("MAE improved while RMSE deteriorated sharply")
    current_capture, previous_capture = current["top_decile_capture"], previous["top_decile_capture"]
    if (current["mae"] < previous["mae"] and current_capture is not None
            and previous_capture is not None and current_capture < previous_capture - 0.2):
        reasons.append("MAE improved while top-decile capture deteriorated sharply")
    return reasons


def _state_profile(snapshots: Path, features: list[str]) -> dict:
    """Summarize current train/validation drift without opening test rows."""
    keys = [name for name in ("spend_90d", "recency_days", "history_spend") if name in features]
    profile = {}
    for split in ("train", "validation"):
        frame = pd.read_csv(snapshots / f"{split}.csv", usecols=[_TARGET, *keys])
        target = frame[_TARGET].to_numpy(float)
        profile[split] = {
            "rows": int(len(frame)),
            "target_zero_rate": float(np.mean(target == 0)),
            "target_mean": float(np.mean(target)),
            "target_p90": float(np.percentile(target, 90)),
            "features": {name: {"mean": float(frame[name].mean()),
                                "p90": float(frame[name].quantile(0.9))} for name in keys},
        }
    return profile


def _final_anomaly_reasons(journal: dict) -> list[str]:
    """Flag held-out reversals and paired losses without asking a model to decide."""
    final = journal["final"]
    baseline_validation = journal["baseline"]["metrics"]["mae"]
    champion_validation = _find(journal, final["champion_id"])["metrics"]["mae"]
    baseline_test = final["baseline"]["metrics"]["mae"]
    champion_test = final["champion"]["metrics"]["mae"]
    reasons = []
    if champion_validation < baseline_validation and champion_test > baseline_test:
        reasons.append("validation MAE gain versus historical baseline reversed on test")
    for key, name in (("paired_mae_interval", "historical baseline"),
                      ("paired_vs_zero", "zero reference")):
        interval = final.get(key)
        if interval is not None and interval["ci_upper"] < 0:
            reasons.append(f"test paired 95% interval shows a loss versus {name}")
    return reasons


def _complete_final_analysis(journal: dict, journal_path: Path,
                             provider: ApiProvider | None) -> dict:
    if provider is None:
        return journal
    final = journal["final"]
    if "analysis" not in final:
        final["analysis"] = analyze_final_report(provider, {
            "target_policy": journal["task"]["target_policy"],
            "primary_metric": "mae", "result": final})
        _save(journal_path, journal)
    reasons = _final_anomaly_reasons(journal)
    if reasons and "anomaly_review" not in final:
        review = review_final_anomaly(provider, {
            "trigger_reasons": reasons,
            "target_policy": journal["task"]["target_policy"],
            "champion_id": final["champion_id"],
            "validation": {"historical_baseline": journal["baseline"]["metrics"],
                           "champion": _find(journal, final["champion_id"])["metrics"]},
            "test": {"historical_baseline": final["baseline"]["metrics"],
                     "zero_reference": final.get("zero_reference", {}).get("metrics"),
                     "champion": final["champion"]["metrics"],
                     "paired_vs_historical": final["paired_mae_interval"],
                     "paired_vs_zero": final.get("paired_vs_zero")},
            "high_analysis": final["analysis"]})
        final["anomaly_review"] = {"trigger_reasons": reasons,
                                   "effort": provider.review_effort, **review}
        _save(journal_path, journal)
    return journal


def _initialize(task_path: Path, snapshots: Path, journal_path: Path,
                task: dict, features: list[str], seed: int,
                sandbox: DockerSandbox, device: str) -> dict:
    root = journal_path.parent
    if root.exists():
        raise ValueError("search directory already exists without a journal")
    try:
        (root / "candidates").mkdir(parents=True)
        source_dir = Path(__file__).with_name("candidates")
        entries = {}
        for identifier, filename in (("baseline", "baseline.py"),
                                     ("zero_reference", "zero.py"),
                                     ("torch_seed", "torch_mlp.py")):
            candidate = root / "candidates" / filename
            shutil.copyfile(source_dir / filename, candidate)
            evaluated = _evaluate(root, candidate, "validation", snapshots, sandbox,
                                  features, seed, device, identifier=identifier)
            entries[identifier] = {"id": identifier, "status": "evaluated",
                                   "candidate": str(candidate.relative_to(root)),
                                   "sha256": _sha(candidate), **evaluated}
        best = min(entries.values(), key=lambda entry: entry["metrics"]["mae"])["id"]
        journal = {"version": _JOURNAL_VERSION, "task_path": str(task_path),
                   "task": task, "agent": None,
                   "state_profile": _state_profile(snapshots, features),
                   "experience": _experience(journal_path, task),
                   "baseline": entries["baseline"],
                   "zero_reference": entries["zero_reference"],
                   "torch_seed": entries["torch_seed"],
                   "steps": [], "diagnoses": [], "best_id": best}
        _save(journal_path, journal)
        return journal
    except Exception:
        shutil.rmtree(root)
        raise


def run_search(task_path: str | Path, snapshot_dir: str | Path,
               journal_path: str | Path, provider: ApiProvider | None,
               steps: int, device: str = "cuda", docker_image: str = "ltvevo-sandbox") -> dict:
    """Run at most `steps` Agent experiments; test.csv is never read here."""
    if steps < 0 or device not in {"cpu", "cuda"}:
        raise ValueError("steps must be nonnegative and device must be cpu or cuda")
    if steps and provider is None:
        raise ValueError("an Agent provider is required for positive steps")
    task_path, snapshots, journal_path = (Path(path).resolve() for path in
                                          (task_path, snapshot_dir, journal_path))
    task, features, seed = _identity(task_path, snapshots, device, docker_image)
    sandbox = DockerSandbox(docker_image)
    if journal_path.exists():
        journal = json.loads(journal_path.read_text())
        if journal.get("version") != _JOURNAL_VERSION:
            raise ValueError("incompatible search journal version; start a new run")
        if journal["task"] != task or Path(journal["task_path"]) != task_path:
            raise ValueError("search task, snapshots, device, or sandbox image changed")
        for entry in _entries(journal):
            _candidate_path(journal_path.parent, entry)
        if "final" in journal and steps > len(journal["steps"]):
            raise ValueError("search was already finalized")
    else:
        journal = _initialize(task_path, snapshots, journal_path, task, features, seed,
                              sandbox, device)
    if steps == 0 or "final" in journal or "stop" in journal:
        return journal

    assert provider is not None
    info = _agent_info(provider)
    if journal["agent"] is None:
        journal["agent"] = info
        _save(journal_path, journal)
    elif journal["agent"] != info:
        raise ValueError("Agent provider configuration changed during search")
    root = journal_path.parent
    while True:
        pending = next((entry for entry in journal["steps"] if entry["status"] == "pending"), None)
        if pending is not None:
            candidate = _candidate_path(root, pending)
            try:
                evaluated = _evaluate(root, candidate, "validation", snapshots, sandbox,
                                      features, seed, device, identifier=pending["id"])
                pending.update(evaluated)
                pending["paired_mae_interval"] = _paired(
                    root, journal["baseline"], pending, snapshots, "validation", seed)
                reference = _find(journal, pending["reference_id"])
                pending["paired_vs_reference"] = (
                    pending["paired_mae_interval"] if reference["id"] == "baseline" else
                    _paired(root, reference, pending, snapshots, "validation", seed))
                pending["status"] = "evaluated"
            except Exception as error:
                pending["status"] = "failed"
                pending["error"] = str(error)[-2000:]
            _save(journal_path, journal)
            continue

        unreflected = next((entry for entry in journal["steps"]
                            if entry["status"] in {"evaluated", "failed"}
                            and "reflection" not in entry), None)
        if unreflected is not None:
            if unreflected["status"] == "evaluated" and "review" not in unreflected:
                reference = _find(journal, unreflected["reference_id"])
                reasons = _anomaly_reasons(unreflected, reference)
                if reasons:
                    unreflected["review"] = review_anomaly(provider, {
                        "anomaly_reasons": reasons,
                        "hypothesis": unreflected["hypothesis"],
                        "expected_result": unreflected["expected_result"],
                        "baseline_metrics": journal["baseline"]["metrics"],
                        "candidate_metrics": unreflected["metrics"],
                        "paired_mae_interval": unreflected["paired_mae_interval"],
                        "reference_metrics": reference["metrics"],
                        "paired_vs_reference": unreflected["paired_vs_reference"],
                        "candidate_py": _candidate_path(root, unreflected).read_text()})
                else:
                    unreflected["review"] = {"decision": "allow", "evidence": "no anomaly threshold",
                                             "next_direction": "reflect on validation"}
                _save(journal_path, journal)
            observation = {"status": unreflected["status"],
                           "hypothesis": unreflected["hypothesis"],
                           "expected_result": unreflected["expected_result"],
                           "baseline_metrics": journal["baseline"]["metrics"],
                           "reference_id": unreflected["reference_id"],
                           "reference_metrics": _find(journal, unreflected["reference_id"])["metrics"],
                           "candidate_metrics": unreflected.get("metrics"),
                           "paired_mae_interval": unreflected.get("paired_mae_interval"),
                           "paired_vs_reference": unreflected.get("paired_vs_reference"),
                           "review": unreflected.get("review"),
                           "error": unreflected.get("error")}
            unreflected["reflection"] = reflect_experiment(provider, observation)
            if (unreflected["status"] == "evaluated"
                    and unreflected["review"]["decision"] == "allow"
                    and unreflected["reflection"]["verdict"] != "invalid"
                    and unreflected["metrics"]["mae"] <
                    _find(journal, journal["best_id"])["metrics"]["mae"]):
                journal["best_id"] = unreflected["id"]
            _save(journal_path, journal)
            continue

        if len(journal["steps"]) >= steps:
            return journal
        context = _context(journal, root)
        proposal = propose_candidate(provider, context)
        if proposal["action"] == "stop":
            journal["stop"] = proposal["reason"]
            _save(journal_path, journal)
            return journal
        if proposal["action"] == "diagnose":
            if any(item["after_step"] == len(journal["steps"]) for item in journal["diagnoses"]):
                raise ValueError("only one diagnosis is allowed between experiments")
            finding = diagnose_history(provider, context, proposal["question"])
            journal["diagnoses"].append({"after_step": len(journal["steps"]),
                                         "question": proposal["question"], **finding})
            _save(journal_path, journal)
            continue
        source = _validate_source(proposal["candidate_py"])
        existing = {_sha(_candidate_path(root, entry)) for entry in _entries(journal)}
        digest = hashlib.sha256(source.encode()).hexdigest()
        if digest in existing:
            raise ValueError("Agent proposed an already evaluated candidate")
        identifier = f"step-{len(journal['steps']) + 1:03d}"
        candidate = root / "candidates" / f"{identifier}.py"
        candidate.write_text(source)
        step = {"id": identifier, "status": "pending", "candidate": str(candidate.relative_to(root)),
                "sha256": digest, "hypothesis": proposal["hypothesis"],
                "expected_result": proposal["expected_result"],
                "reference_id": journal["best_id"],
                "feature_requests": proposal.get("feature_requests", [])}
        journal["steps"].append(step)
        _feature_requests(root, journal)
        _save(journal_path, journal)


def finalize_search(journal_path: str | Path, snapshot_dir: str | Path, *,
                    device: str = "cuda", docker_image: str = "ltvevo-sandbox",
                    provider: ApiProvider | None = None) -> dict:
    """Evaluate the champion against zero and historical references on test once."""
    journal_path, snapshots = Path(journal_path).resolve(), Path(snapshot_dir).resolve()
    journal = json.loads(journal_path.read_text())
    if journal.get("version") != _JOURNAL_VERSION:
        raise ValueError("incompatible search journal version; start a new run")
    task, features, seed = _identity(Path(journal["task_path"]), snapshots, device, docker_image)
    if journal["task"] != task:
        raise ValueError("finalization task, snapshots, device, or sandbox image changed")
    if _sha(snapshots / "test.csv") != task["split_sha256"]["test"]:
        raise ValueError("test snapshot changed after preparation")
    if "final" in journal:
        return _complete_final_analysis(journal, journal_path, provider)
    root = journal_path.parent
    for entry in _entries(journal):
        _candidate_path(root, entry)
    sandbox = DockerSandbox(docker_image)
    baseline = _evaluate(root, _candidate_path(root, journal["baseline"]), "test",
                         snapshots, sandbox, features, seed, device, identifier="baseline")
    zero_reference = _evaluate(root, _candidate_path(root, journal["zero_reference"]),
                               "test", snapshots, sandbox, features, seed, device,
                               identifier="zero_reference")
    champion_entry = _find(journal, journal["best_id"])
    if champion_entry["id"] == "baseline":
        champion = baseline
    elif champion_entry["id"] == "zero_reference":
        champion = zero_reference
    else:
        champion = _evaluate(root, _candidate_path(root, champion_entry), "test",
                             snapshots, sandbox, features, seed, device,
                             identifier=champion_entry["id"])
    interval = _paired(root, baseline, champion, snapshots, "test", seed)
    interval_zero = _paired(root, zero_reference, champion, snapshots, "test", seed)
    journal["final"] = {"champion_id": champion_entry["id"],
                        "baseline": baseline, "zero_reference": zero_reference,
                        "champion": champion, "paired_mae_interval": interval,
                        "paired_vs_zero": interval_zero}
    _save(journal_path, journal)
    return _complete_final_analysis(journal, journal_path, provider)
