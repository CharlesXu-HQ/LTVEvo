import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ltvevo.candidates.baseline import fit_predict as baseline_predict
from ltvevo.candidates.torch_mlp import fit_predict as torch_predict
from ltvevo.evaluation import paired_mae_interval, summarize


class EvaluationTests(unittest.TestCase):
    def test_perfect_ranking_captures_all_value_in_top_decile(self):
        actual = [10, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        result = summarize(actual, actual, list(range(10)))

        self.assertEqual(result["mae"], 0.0)
        self.assertEqual(result["rmse"], 0.0)
        self.assertAlmostEqual(result["normalized_gini"], 1.0)
        self.assertEqual(result["top_decile_capture"], 1.0)
        self.assertEqual(result["decile_calibration"][0], {
            "decile": 1,
            "count": 1,
            "observed_mean": 10.0,
            "predicted_mean": 10.0,
        })

    def test_error_and_reversed_ranking_have_hand_checked_values(self):
        result = summarize([0, 2], [1, 0], ["a", "b"])

        self.assertEqual(result["mae"], 1.5)
        self.assertAlmostEqual(result["rmse"], math.sqrt(2.5))
        self.assertAlmostEqual(result["normalized_gini"], -1.0)
        self.assertEqual(result["top_decile_capture"], 0.0)

    def test_ranking_is_undefined_for_signed_targets(self):
        result = summarize([-5, 10], [0, 0], ["a", "b"])

        self.assertIsNone(result["normalized_gini"])
        self.assertIsNone(result["top_decile_capture"])

    def test_tied_scores_do_not_make_capture_depend_on_row_order(self):
        values = [10, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        first = summarize(values, [0] * 10, list(range(10)))
        last = summarize(values[::-1], [0] * 10, list(range(10)))

        self.assertEqual(first["top_decile_capture"], 0.1)
        self.assertEqual(last["top_decile_capture"], 0.1)
        self.assertEqual(first["decile_calibration"][0]["observed_mean"], 1.0)
        self.assertEqual(last["decile_calibration"][0]["observed_mean"], 1.0)

    def test_cluster_bootstrap_keeps_same_customer_rows_together(self):
        interval = paired_mae_interval(
            [0, 0, 0], [5, 0, 0], [0, 5, 0], ["a", "a", "b"],
            seed=7, reps=100,
        )

        self.assertEqual(interval["mae_improvement"], 0.0)
        self.assertEqual(interval["ci_lower"], 0.0)
        self.assertEqual(interval["ci_upper"], 0.0)
        self.assertEqual(interval["customer_clusters"], 2)

    def test_cluster_bootstrap_reports_positive_paired_improvement(self):
        interval = paired_mae_interval(
            [0, 0], [2, 2], [0, 0], ["a", "b"], seed=7, reps=100,
        )

        self.assertEqual(interval["mae_improvement"], 2.0)
        self.assertEqual(interval["ci_lower"], 2.0)
        self.assertEqual(interval["ci_upper"], 2.0)


class CandidateTests(unittest.TestCase):
    def test_historical_spend_baseline_fits_future_amount(self):
        train = pd.DataFrame({"spend_90d": [0, 10, 20], "target": [0, 5, 10]})
        validation = pd.DataFrame({"spend_90d": [30]})

        predictions = baseline_predict(
            train, validation, feature_columns=["spend_90d"],
            target_column="target", device="cpu", seed=1,
        )

        np.testing.assert_allclose(predictions, [15.0])

    def test_torch_model_returns_finite_nonnegative_predictions(self):
        train = pd.DataFrame({
            "spend_90d": [0, 1, 2, 3],
            "transactions_90d": [0, 1, 1, 2],
            "target": [0, 2, 4, 6],
        })
        validation = pd.DataFrame({
            "spend_90d": [1, 4], "transactions_90d": [1, 2],
        })

        predictions = torch_predict(
            train, validation,
            feature_columns=["spend_90d", "transactions_90d"],
            target_column="target", device="cpu", seed=1,
        )

        self.assertEqual(len(predictions), 2)
        self.assertTrue(np.all(np.isfinite(predictions)))
        self.assertTrue(np.all(np.asarray(predictions) >= 0))

    def test_torch_model_refuses_cuda_fallback(self):
        train = pd.DataFrame({"spend_90d": [1], "target": [1]})
        validation = pd.DataFrame({"spend_90d": [1]})

        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                torch_predict(
                    train, validation, feature_columns=["spend_90d"],
                    target_column="target", device="cuda", seed=1,
                )


if __name__ == "__main__":
    unittest.main()
