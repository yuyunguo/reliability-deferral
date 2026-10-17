"""Tests for Sepsis-3 relabelling (R1-5) and clinical-score baselines (R2-5)."""
import numpy as np
import pandas as pd

from deferral_frontier.experiments.build_sepsis3_task import sepsis3_labels
from deferral_frontier.experiments.run_clinical_score_baseline import score_metrics


def test_sepsis3_labels_flag_cohort_stays_by_stay_id():
    stays = pd.Series([101, 102, 103], index=pd.Index([1, 2, 3], name="hadm_id"))
    lab = sepsis3_labels(flagged_stay_ids={102, 999}, cohort_stays=stays)
    assert lab.tolist() == [0, 1, 0] and lab.name == "label" and lab.index.name == "hadm_id"


def test_score_metrics_reports_auroc_brier_and_symmetric_gate():
    rng = np.random.default_rng(0)
    p = rng.random(2000)
    y = (rng.random(2000) < p).astype(int)
    m = score_metrics(y, p)
    assert 0.7 < m["auroc"] < 0.9
    assert m["brier_symmetric_80"] < m["brier"]
    assert set(m) == {"n", "auroc", "brier", "brier_symmetric_80"}
