"""Tests for the post hoc extra analyses (plan/r1-extra-analyses-plan.md)."""
import numpy as np
import pandas as pd

from deferral_frontier.experiments.build_sepsis3_onset_task import onset_labels
from deferral_frontier.src.extra_analyses import (
    decision_loss, harmonized_drop, random_effects, t_centered_score)
from deferral_frontier.src.reliability_v2 import sufficiency_from


def test_onset_labels_exclude_onset_within_24h_and_label_later_onset():
    cohort = pd.DataFrame({"stay_id": [1, 2, 3], "intime": pd.to_datetime(["2100-01-01"] * 3)},
                          index=[10, 20, 30])
    onset = pd.Series({1: pd.Timestamp("2100-01-01 12:00"), 2: pd.Timestamp("2100-01-03")})
    lab = onset_labels(cohort, onset)
    assert np.isnan(lab[10]) and lab[20] == 1 and lab[30] == 0


def test_harmonized_drop_removes_counts_and_temperature_only():
    cols = ["vital_heart_rate_mean", "vital_heart_rate_count", "vital_temp_f_mean", "vital_temp_c_last", "lab_lactate_max"]
    assert harmonized_drop(cols) == ["vital_heart_rate_count", "vital_temp_f_mean", "vital_temp_c_last"]


def test_sufficiency_uses_supplied_counts_when_features_lack_them():
    X = np.zeros((2, 1))
    counts = np.array([[1, 0], [2, 3]])
    d2 = sufficiency_from(X, ["a_mean"], counts=counts)
    assert np.allclose(d2, [0.5, 1.0])
    X2 = np.array([[1.0, 0.0], [2.0, 3.0]])
    assert np.allclose(sufficiency_from(X2, ["a_count", "b_count"]), [0.5, 1.0])


def test_decision_loss_weights_errors_by_threshold():
    y = np.array([1, 0])
    p = np.array([0.05, 0.30])  # at t=0.1: FN for the first, FP for the second
    assert np.allclose(decision_loss(p, y, 0.1), [0.9, 0.1])


def test_t_centered_score_ranks_patients_nearest_threshold_as_least_reliable():
    p = np.array([0.02, 0.1, 0.5])
    s = t_centered_score(p, 0.1)
    assert s.argmin() == 1


def test_random_effects_pools_equal_estimates_without_heterogeneity():
    est, se = np.array([1.0, 1.0, 1.0]), np.array([0.1, 0.2, 0.1])
    r = random_effects(est, se)
    assert np.isclose(r["pooled"], 1.0) and r["i2"] == 0.0 and r["ci_lo"] < 1.0 < r["ci_hi"]


def test_subset_task_tables_keeps_only_label_rows_in_label_order():
    from deferral_frontier.experiments.build_sepsis3_onset_task import subset_tables
    feats = pd.DataFrame({"a": [1, 2, 3]}, index=pd.Index([10, 20, 30], name="hadm_id"))
    meta = pd.DataFrame({"m": [1, 2, 3]}, index=feats.index)
    ts = pd.DataFrame({"stay_id": [10, 20, 20, 30], "v": [1, 2, 3, 4]})
    labels = pd.Series([0, 1], index=pd.Index([30, 10], name="hadm_id"))
    f, m, t = subset_tables(feats, meta, ts, labels.index)
    assert f.index.equals(labels.index) and m.index.equals(labels.index)
    assert set(t.stay_id) == {10, 30}
