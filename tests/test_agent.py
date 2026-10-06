from unittest.mock import patch

import pytest

from model_evo_harness import validate_reflection, validate_research
from model_evo_harness.catalog import load_catalog

from ltvevo.agent import diagnose_history, propose_candidate, reflect_experiment
from ltvevo.provider import ApiProvider


def _provider():
    return ApiProvider("https://example.com", "model", "secret")


def _runtime():
    return {"identity": {"commit": "pinned"},
            "task_snapshot": {"stage": "ltv_prediction", "fields": ["recency_days"],
                              "framework": "pytorch", "objective": {"name": "mae",
                                                                   "direction": "min"}},
            "catalog": load_catalog(),
            "prompt_context": {"ltv_family": "ready", "excluded_stages": ["ranking", "policy"]}}


def _research(**changes):
    return {"direction": "shrink high predictions", "mechanism": "regularize the regression head",
            "why_now": "validation MAE exceeds the zero reference",
            "data_rationale": "recency_days is observed before as_of",
            "comparison": "same validation rows versus zero reference",
            "expected_result": "validation MAE falls below the zero reference",
            "falsification": "validation MAE does not fall below the zero reference",
            "input_fields": ["recency_days"],
            "alternatives": [{"direction": "keep zero reference", "mechanism": "predict zero",
                              "reason": "current MAE control"}], **changes}


def _proposal(**changes):
    return {"action": "experiment", "hypothesis": "regularization reduces error",
            "expected_result": "MAE falls", "candidate_py": "def fit_predict(*args, **kwargs): pass",
            "feature_requests": [], "research": _research(**changes)}


def test_harness_proposal_uses_research_validator_without_sending_catalog_to_provider():
    runtime = _runtime()
    with (patch("ltvevo.agent._call", return_value=_proposal()) as request,
          patch("model_evo_harness.validate_research", wraps=validate_research) as validator):
        result = propose_candidate(_provider(), {"best_id": "zero_reference",
                                                  "harness_runtime": runtime})
    assert result["research"]["input_fields"] == ["recency_days"]
    assert validator.call_count == 1
    sent = request.call_args.args[2]
    assert sent["best_id"] == "zero_reference"
    assert sent["harness"] == runtime["prompt_context"]
    assert sent["task"]["stage"] == "ltv_prediction"
    assert sent["task"]["objective"] == {"name": "mae", "direction": "min"}
    assert "harness_runtime" not in sent and "catalog" not in sent


def test_harness_rejects_wrong_stage_after_one_repair_attempt():
    invalid = _proposal(family_id="collaborative_retrieval")
    with patch("ltvevo.agent._call", side_effect=[invalid, invalid]) as request:
        with pytest.raises(ValueError, match="other_stage"):
            propose_candidate(_provider(), {"harness_runtime": _runtime()})
    assert request.call_count == 2


def test_harness_diagnosis_sends_compact_context_only():
    answer = {"finding": "zero reference is strongest", "evidence": "validation MAE",
              "next_direction": "test a sparse model"}
    with patch("ltvevo.agent._call", return_value=answer) as request:
        result = diagnose_history(_provider(), {"best_id": "zero_reference",
                                                "harness_runtime": _runtime()},
                                  "What explains the zero reference?")
    assert result == answer
    sent = request.call_args.args[2]["context"]
    assert sent["best_id"] == "zero_reference"
    assert sent["harness"]["ltv_family"] == "ready"
    assert "harness_runtime" not in sent and "catalog" not in sent


def test_harness_reference_ledger_survives_invalid_proposal_retry():
    responses = [{"action": "read_reference", "framework": "pytorch",
                  "method_ids": ["afm"]},
                 {"action": "experiment", "candidate_py": "pass"},
                 _proposal()]
    with patch("ltvevo.agent._call", side_effect=responses) as request:
        result = propose_candidate(_provider(), {"harness_runtime": _runtime()})
    assert request.call_count == 3
    assert "models/pytorch/interactions.py" in result["reference_reads"]
    assert len(result["reference_reads"]["models/pytorch/interactions.py"]) == 64
    assert "models/pytorch/interactions.py" in request.call_args.args[2]["reference_material"]


def test_harness_reference_limit_is_shared_across_proposal_repair():
    responses = [{"action": "read_reference", "framework": "pytorch",
                  "method_ids": ["afm"]},
                 {"action": "experiment", "candidate_py": "pass"},
                 {"action": "read_reference", "framework": "pytorch",
                  "method_ids": ["fm"]},
                 {"action": "read_reference", "framework": "pytorch",
                  "method_ids": ["esmm"]}]
    with patch("ltvevo.agent._call", side_effect=responses) as request:
        with pytest.raises(ValueError, match="round limit"):
            propose_candidate(_provider(), {"harness_runtime": _runtime()})
    assert request.call_count == 4


def test_harness_reflection_requires_non_observable_business_and_validates_technical():
    runtime = _runtime()
    answer = {"verdict": "inconclusive", "evidence": "paired interval crosses zero",
              "lesson": "regularization remains uncertain", "next_direction": "test calibration",
              "technical_experience": {"lesson": "regularization remains uncertain",
                                       "evidence": "paired interval crosses zero",
                                       "uncertainty": "validation only", "next_test": "test calibration"},
              "business_experience": {"status": "not_observable",
                                      "reason": "prediction labels contain no intervention outcome"}}
    observation = {"status": "evaluated", "research": _research(),
                   "harness_runtime": runtime,
                   "harness_steps": [{"id": "step-001", "status": "evaluated",
                                      "research": _research()}]}
    with (patch("ltvevo.agent._call", return_value=answer) as request,
          patch("model_evo_harness.validate_reflection", wraps=validate_reflection) as validator):
        result = reflect_experiment(_provider(), observation)
    assert result["business_experience"]["status"] == "not_observable"
    assert result["technical_experience"]["attribution"] == "unverified"
    assert validator.call_count == 1
    assert "harness_runtime" not in request.call_args.args[2]


def test_harness_reflection_rejects_business_claim_after_bounded_retry():
    answer = {"verdict": "consistent", "evidence": "MAE fell", "lesson": "uncertain",
              "next_direction": "new experiment", "technical_experience": {
                  "lesson": "uncertain", "evidence": "validation MAE fell",
                  "uncertainty": "no holdout", "next_test": "new experiment"},
              "business_experience": {"status": "observed", "insight": "coupon uplift",
                                      "limitations": "none", "observation_id": "invented"}}
    observation = {"status": "evaluated", "research": _research(),
                   "harness_runtime": _runtime(), "harness_steps": []}
    with patch("ltvevo.agent._call", return_value=answer) as request:
        with pytest.raises(ValueError, match="not_observable"):
            reflect_experiment(_provider(), observation)
    assert request.call_count == 2


def test_harness_reflection_retries_malformed_business_experience():
    answer = {"verdict": "inconclusive", "evidence": "MAE uncertain", "lesson": "unknown",
              "next_direction": "retest", "technical_experience": {
                  "lesson": "unknown", "evidence": "MAE uncertain", "uncertainty": "validation only",
                  "next_test": "retest"},
              "business_experience": {"status": "not_observable", "reason": "no contact data"}}
    invalid = {**answer, "business_experience": None}
    observation = {"status": "evaluated", "research": _research(),
                   "harness_runtime": _runtime(), "harness_steps": []}
    with patch("ltvevo.agent._call", side_effect=[invalid, answer]) as request:
        result = reflect_experiment(_provider(), observation)
    assert result["business_experience"]["status"] == "not_observable"
    assert request.call_count == 2


def test_reflection_retries_once_when_provider_omits_required_field():
    provider = ApiProvider("https://example.com", "model", "secret")
    invalid = {"verdict": "inconclusive", "evidence": "metric change is small", "lesson": "compare again"}
    valid = {**invalid, "next_direction": "try a calibrated model"}
    with patch("ltvevo.agent._call", side_effect=[invalid, valid]) as request:
        result = reflect_experiment(provider, {"status": "evaluated"})
    assert result == valid
    assert request.call_count == 2
