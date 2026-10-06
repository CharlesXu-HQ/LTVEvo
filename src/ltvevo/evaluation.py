"""Fixed metrics for future purchase-value prediction."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _checked_inputs(y_true, prediction, customer_ids):
    actual = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(prediction, dtype=np.float64)
    customers = pd.Series(customer_ids)
    if actual.ndim != 1 or predicted.ndim != 1 or len(actual) == 0:
        raise ValueError("actual and prediction must be nonempty one-dimensional arrays")
    if len(actual) != len(predicted) or len(actual) != len(customers):
        raise ValueError("actual, prediction, and customer IDs must have equal length")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("actual and prediction must be finite")
    if customers.isna().any():
        raise ValueError("customer IDs must be present")
    return actual, predicted, customers


def summarize(y_true, pred, customer_ids) -> dict:
    """Report errors, ranking quality, and calibration on the same rows."""
    actual, predicted, customers = _checked_inputs(y_true, pred, customer_ids)
    error = actual - predicted
    order = np.argsort(-predicted, kind="stable")
    tied_expected_actual = (
        pd.Series(actual).groupby(pd.Series(predicted), sort=False).transform("mean").to_numpy()
    )
    bins = np.array_split(order, min(10, len(order)))
    calibration = [
        {
            "decile": index,
            "count": int(len(rows)),
            "observed_mean": float(tied_expected_actual[rows].mean()),
            "predicted_mean": float(predicted[rows].mean()),
        }
        for index, rows in enumerate(bins, 1)
    ]

    gini = None
    capture = None
    total_value = float(actual.sum())
    if np.all(actual >= 0) and total_value > 0:
        count = max(1, math.ceil(0.1 * len(actual)))
        cutoff = predicted[order[count - 1]]
        above = predicted > cutoff
        tied = predicted == cutoff
        tied_share = (count - int(above.sum())) / int(tied.sum())
        capture = float((actual[above].sum() + tied_share * actual[tied].sum()) / total_value)
        center = (len(actual) + 1) / 2
        actual_ranks = pd.Series(actual).rank(method="average").to_numpy() - center
        pred_ranks = pd.Series(predicted).rank(method="average").to_numpy() - center
        ideal = float(np.dot(actual, actual_ranks))
        if ideal > 0:
            gini = float(np.dot(actual, pred_ranks) / ideal)

    return {
        "rows": int(len(actual)),
        "customers": int(customers.nunique()),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "normalized_gini": gini,
        "top_decile_capture": capture,
        "decile_calibration": calibration,
    }


def summarize_by_period(y_true, pred, customer_ids, periods) -> list[dict]:
    """Expose time-window error without using held-out test rows during search."""
    actual, predicted, customers = _checked_inputs(y_true, pred, customer_ids)
    dates = pd.Series(periods, dtype="string")
    if len(dates) != len(actual) or dates.isna().any():
        raise ValueError("periods must be present for every prediction")
    result = []
    for period in sorted(dates.unique()):
        rows = (dates == period).to_numpy()
        target = actual[rows]
        result.append({
            "period": str(period),
            "rows": int(rows.sum()),
            "customers": int(customers[rows].nunique()),
            "mae": float(np.abs(target - predicted[rows]).mean()),
            "target_mean": float(target.mean()),
            "target_zero_rate": float(np.mean(target == 0)),
        })
    return result


def paired_mae_interval(
    y_true, baseline_pred, candidate_pred, customer_ids, *, seed: int, reps: int = 1000
) -> dict:
    """Bootstrap MAE improvement by customer, keeping repeated snapshots together."""
    actual, baseline, customers = _checked_inputs(y_true, baseline_pred, customer_ids)
    _, candidate, _ = _checked_inputs(y_true, candidate_pred, customer_ids)
    if reps <= 0:
        raise ValueError("reps must be positive")

    improvement = np.abs(actual - baseline) - np.abs(actual - candidate)
    codes, unique_customers = pd.factorize(customers, sort=False)
    group_sums = np.bincount(codes, weights=improvement)
    group_counts = np.bincount(codes)
    generator = np.random.default_rng(seed)
    draws = np.empty(reps, dtype=np.float64)
    for index in range(reps):
        sampled = generator.integers(len(unique_customers), size=len(unique_customers))
        draws[index] = group_sums[sampled].sum() / group_counts[sampled].sum()
    lower, upper = np.percentile(draws, [2.5, 97.5])

    return {
        "mae_improvement": float(improvement.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence_level": 0.95,
        "bootstrap_reps": int(reps),
        "customer_clusters": int(len(unique_customers)),
    }
