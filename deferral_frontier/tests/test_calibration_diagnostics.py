"""Tests for calibration / class-conditional diagnostics (E-3, R1-6)."""
import numpy as np

from deferral_frontier.experiments.run_calibration_diagnostics import (
    calibration_bins, class_conditional, confusion_at,
)


def test_confusion_at_half():
    y = np.array([1, 1, 0, 0, 1])
    p = np.array([0.9, 0.2, 0.6, 0.1, 0.5])
    assert confusion_at(y, p, 0.5) == {"tp": 2, "fn": 1, "fp": 1, "tn": 1}


def test_class_conditional_brier_and_rates_within_retained_set():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.8, 0.4, 0.3, 0.9])
    keep = np.array([True, True, True, False])
    r = class_conditional(y, p, keep)
    assert r["n_pos"] == 2 and r["n_neg"] == 1
    assert np.isclose(r["brier_pos"], ((0.2 ** 2) + (0.6 ** 2)) / 2)
    assert np.isclose(r["brier_neg"], 0.3 ** 2)
    assert r["sensitivity"] == 0.5 and r["specificity"] == 1.0


def test_calibration_bins_quantile_equal_counts():
    rng = np.random.default_rng(0)
    p = rng.random(1000)
    y = (rng.random(1000) < p).astype(int)
    b = calibration_bins(y, p, n_bins=10)
    assert len(b) == 10 and b["n"].sum() == 1000 and (b["n"].between(95, 105)).all()
    assert np.allclose(b["mean_pred"], b["obs_rate"], atol=0.1)
