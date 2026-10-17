"""Tests for the imperfect-review sensitivity analysis (R2-6)."""
import numpy as np

from deferral_frontier.experiments.run_imperfect_review import review_outcome


def test_review_outcome_total_errors_and_break_even_accuracy():
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    p = np.array([.9, .1, .9, .1, .9, .1, .4, .6, .4, .6])   # last 4 misclassified at 0.5
    keep = np.array([True] * 6 + [False] * 4)                  # defer the 4 errors
    r = review_outcome(y, p, keep, clinician_accuracy=0.9)
    assert r["model_errors_retained"] == 0 and r["model_errors_deferred"] == 4
    assert np.isclose(r["expected_errors_total"], 0.4)        # 4 deferred * (1 - 0.9)
    assert np.isclose(r["error_rate_no_deferral"], 0.4)
    assert np.isclose(r["break_even_accuracy"], 0.0)          # model wrong on all deferred
