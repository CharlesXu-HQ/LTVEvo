"""Candidate audits flag concrete contradictions without claiming full verification."""

from ltvevo.candidate_audit import audit_candidate


def test_silent_nonfinite_to_zero_fallback_is_contradicted():
    source = """
import torch

def fit_predict(train, validation, *, feature_columns, target_column, device, seed):
    prediction = torch.ones(len(validation))
    if not bool(torch.isfinite(prediction).all()):
        prediction = torch.zeros_like(prediction)
    return prediction
"""
    result = audit_candidate(source, "Predict customer value", "Lower MAE")
    assert result["status"] == "contradicted"
    assert len(result["findings"]) == 1
    assert "non-finite" in result["findings"][0]


def test_nonfinite_guard_that_raises_does_not_claim_contradiction():
    source = """
import torch

def fit_predict(train, validation, *, feature_columns, target_column, device, seed):
    prediction = torch.ones(len(validation))
    if not torch.isfinite(prediction).all():
        raise ValueError("non-finite predictions")
    return prediction
"""
    assert audit_candidate(source, "Predict customer value", "Lower MAE") == {
        "status": "unverified", "findings": []}


def test_unconditional_zero_predictor_does_not_claim_contradiction():
    source = """
import torch

def fit_predict(train, validation, *, feature_columns, target_column, device, seed):
    prediction = torch.zeros(len(validation))
    return prediction
"""
    assert audit_candidate(source, "Try zero baseline", "Lower MAE") == {
        "status": "unverified", "findings": []}


def test_quantile_cutoff_used_for_claimed_top_k_is_contradicted():
    source = """
import torch

def fit_predict(train, validation, *, feature_columns, target_column, device, seed):
    score = torch.arange(len(validation))
    fraction = 0.2
    threshold = torch.quantile(score, 1.0 - fraction)
    selected = score >= threshold
    prediction = torch.where(selected, torch.ones_like(score), torch.zeros_like(score))
    return prediction
"""
    result = audit_candidate(source, "Select the exact top-k fraction", "Improve MAE")
    assert result["status"] == "contradicted"
    assert len(result["findings"]) == 1
    assert "ties" in result["findings"][0]


def test_quantile_cutoff_without_top_k_claim_does_not_claim_contradiction():
    source = """
import torch

def fit_predict(train, validation, *, feature_columns, target_column, device, seed):
    score = torch.arange(len(validation))
    threshold = torch.quantile(score, 0.8)
    selected = score >= threshold
    return torch.where(selected, torch.ones_like(score), torch.zeros_like(score))
"""
    assert audit_candidate(source, "Segment customers", "Lower MAE") == {
        "status": "unverified", "findings": []}


def test_quantile_cutoff_for_another_score_does_not_claim_contradiction():
    source = """
import torch

def fit_predict(train, validation, *, feature_columns, target_column, device, seed):
    score = torch.arange(len(validation))
    other = score + 1
    threshold = torch.quantile(other, 0.8)
    selected = score >= threshold
    return torch.where(selected, torch.ones_like(score), torch.zeros_like(score))
"""
    assert audit_candidate(source, "Select exact top-k", "Lower MAE") == {
        "status": "unverified", "findings": []}


def test_unused_quantile_mask_does_not_claim_contradiction():
    source = """
import torch

def fit_predict(train, validation, *, feature_columns, target_column, device, seed):
    score = torch.arange(len(validation))
    threshold = torch.quantile(score, 0.8)
    selected = score >= threshold
    return score
"""
    assert audit_candidate(source, "Select exact top-k", "Lower MAE") == {
        "status": "unverified", "findings": []}


def test_unrelated_helper_guards_do_not_claim_prediction_contradiction():
    source = """
import torch

def unused_helper(score):
    if not torch.isfinite(score).all():
        score = torch.zeros_like(score)
    threshold = torch.quantile(score, 0.8)
    selected = score >= threshold
    return torch.where(selected, score, torch.zeros_like(score))

def fit_predict(train, validation, *, feature_columns, target_column, device, seed):
    return torch.ones(len(validation))
"""
    assert audit_candidate(source, "Select exact top-k", "Lower MAE") == {
        "status": "unverified", "findings": []}


def test_replaced_quantile_mask_does_not_claim_contradiction():
    source = """
import torch

def fit_predict(train, validation, *, feature_columns, target_column, device, seed):
    score = torch.arange(len(validation))
    threshold = torch.quantile(score, 0.8)
    selected = score >= threshold
    selected = torch.zeros_like(score, dtype=torch.bool)
    return torch.where(selected, score, torch.zeros_like(score))
"""
    assert audit_candidate(source, "Select exact top-k", "Lower MAE") == {
        "status": "unverified", "findings": []}


def test_chained_output_helper_cannot_hide_nonfinite_fallback():
    source = """
import torch

def clean(prediction):
    if not torch.isfinite(prediction).all():
        prediction = torch.zeros_like(prediction)
    return prediction

def postprocess(prediction):
    return clean(prediction)

def fit_predict(train, validation, *, feature_columns, target_column, device, seed):
    return postprocess(torch.ones(len(validation)))
"""
    result = audit_candidate(source, "Predict value", "Lower MAE")
    assert result["status"] == "contradicted"
    assert "non-finite" in result["findings"][0]


def test_output_helper_cannot_hide_exact_top_k_quantile_mask():
    source = """
import torch

def topk_prediction(score):
    threshold = torch.quantile(score, 0.8)
    selected = score >= threshold
    return selected.float()

def fit_predict(train, validation, *, feature_columns, target_column, device, seed):
    return topk_prediction(torch.arange(len(validation)))
"""
    result = audit_candidate(source, "Select exact top-k fraction", "Lower MAE")
    assert result["status"] == "contradicted"
    assert "ties" in result["findings"][0]


def test_called_prediction_helper_zero_fallback_is_contradicted():
    source = """
import torch

def clean_prediction(prediction):
    if not torch.isfinite(prediction).all():
        prediction = torch.zeros_like(prediction)
    return prediction

def fit_predict(train, validation, *, feature_columns, target_column, device, seed):
    prediction = torch.ones(len(validation))
    return clean_prediction(prediction)
"""
    result = audit_candidate(source, "Predict value", "Lower MAE")
    assert result["status"] == "contradicted"
    assert "non-finite" in result["findings"][0]


def test_feature_cleaning_helper_does_not_claim_prediction_contradiction():
    source = """
import torch

def clean_features(features):
    if not torch.isfinite(features).all():
        features = torch.zeros_like(features)
    return features

def fit_predict(train, validation, *, feature_columns, target_column, device, seed):
    features = clean_features(torch.ones(len(validation)))
    return torch.ones(len(validation))
"""
    assert audit_candidate(source, "Predict value", "Lower MAE") == {
        "status": "unverified", "findings": []}


def test_nan_to_num_on_returned_prediction_is_contradicted():
    source = """
import torch

def fit_predict(train, validation, *, feature_columns, target_column, device, seed):
    prediction = torch.ones(len(validation))
    prediction = torch.nan_to_num(prediction)
    return prediction
"""
    result = audit_candidate(source, "Predict value", "Lower MAE")
    assert result["status"] == "contradicted"
    assert "sanitized" in result["findings"][0]


def test_nan_to_num_on_features_does_not_claim_prediction_contradiction():
    source = """
import torch

def fit_predict(train, validation, *, feature_columns, target_column, device, seed):
    features = torch.ones(len(validation))
    features = torch.nan_to_num(features)
    prediction = torch.ones(len(validation))
    return prediction
"""
    assert audit_candidate(source, "Predict value", "Lower MAE") == {
        "status": "unverified", "findings": []}


def test_returned_where_finite_zero_sanitizer_is_contradicted():
    source = """
import torch

def fit_predict(train, validation, *, feature_columns, target_column, device, seed):
    prediction = torch.ones(len(validation))
    return torch.where(torch.isfinite(prediction), prediction, torch.zeros_like(prediction))
"""
    result = audit_candidate(source, "Predict value", "Lower MAE")
    assert result["status"] == "contradicted"
    assert "non-finite" in result["findings"][0]


def test_top_k_mask_converted_to_float_is_contradicted():
    source = """
import torch

def fit_predict(train, validation, *, feature_columns, target_column, device, seed):
    score = torch.arange(len(validation))
    threshold = torch.quantile(score, 0.8)
    selected = score >= threshold
    prediction = selected.float()
    return prediction
"""
    result = audit_candidate(source, "Select exact top-k", "Lower MAE")
    assert result["status"] == "contradicted"
    assert "ties" in result["findings"][0]


def test_direct_top_k_mask_return_is_contradicted():
    source = """
import torch

def fit_predict(train, validation, *, feature_columns, target_column, device, seed):
    score = torch.arange(len(validation))
    threshold = torch.quantile(score, 0.8)
    return score >= threshold
"""
    result = audit_candidate(source, "Select exact top-k", "Lower MAE")
    assert result["status"] == "contradicted"
    assert "ties" in result["findings"][0]


def test_top_k_mask_multiplied_into_prediction_is_contradicted():
    source = """
import torch

def fit_predict(train, validation, *, feature_columns, target_column, device, seed):
    score = torch.arange(len(validation))
    threshold = torch.quantile(score, 0.8)
    selected = score >= threshold
    prediction = selected.float() * 3.0
    return prediction
"""
    result = audit_candidate(source, "Select exact top-k", "Lower MAE")
    assert result["status"] == "contradicted"
    assert "ties" in result["findings"][0]


def test_conditional_zero_fallback_does_not_hide_top_k_contradiction():
    source = """
import torch

def fit_predict(train, validation, *, feature_columns, target_column, device, seed):
    score = torch.arange(len(validation))
    threshold = torch.quantile(score, 0.8)
    selected = score >= threshold
    prediction = torch.where(selected, torch.ones_like(score), torch.zeros_like(score))
    if not torch.isfinite(prediction).all():
        prediction = torch.zeros_like(prediction)
    return prediction
"""
    result = audit_candidate(source, "Select exact top-k", "Lower MAE")
    assert result["status"] == "contradicted"
    assert any("ties" in finding for finding in result["findings"])


def test_unconditional_prediction_replacement_does_not_claim_top_k_contradiction():
    source = """
import torch

def fit_predict(train, validation, *, feature_columns, target_column, device, seed):
    score = torch.arange(len(validation))
    threshold = torch.quantile(score, 0.8)
    selected = score >= threshold
    prediction = selected.float()
    prediction = torch.zeros_like(score)
    return prediction
"""
    assert audit_candidate(source, "Select exact top-k", "Lower MAE") == {
        "status": "unverified", "findings": []}
