"""Tests for the extended subgroup audit (R1-7)."""
import numpy as np

from deferral_frontier.experiments.run_fairness_r1 import group_metrics


def test_group_metrics_error_capture_and_inappropriate_deferral():
    y = np.array([1, 0, 1, 0, 1, 0])
    p = np.array([0.2, 0.9, 0.8, 0.1, 0.7, 0.3])   # errors at 0.5: rows 0 and 1
    keep = np.array([False, True, False, True, True, True])  # defer rows 0 and 2
    m = group_metrics(y, p, keep)
    assert m["n"] == 6 and np.isclose(m["deferral_rate"], 2 / 6)
    assert np.isclose(m["error_capture"], 0.5)                  # 1 of 2 errors deferred
    assert np.isclose(m["inappropriate_deferral_rate"], 1 / 6)  # row 2 correct but deferred
    assert np.isclose(m["brier_retained"], np.mean((p[keep] - y[keep]) ** 2))


def test_race_group_harmonizes_mimic_and_eicu_labels():
    from deferral_frontier.experiments.extract_demographics_r1 import race_group
    assert race_group("WHITE - RUSSIAN") == "White" and race_group("Caucasian") == "White"
    assert race_group("BLACK/AFRICAN AMERICAN") == "Black" and race_group("African American") == "Black"
    assert race_group("HISPANIC OR LATINO") == "Hispanic" and race_group("Asian") == "Asian"
    assert race_group("UNABLE TO OBTAIN") == "Other/Unknown" and race_group(None) == "Other/Unknown"
    assert race_group("Native American") == "Other/Unknown"
