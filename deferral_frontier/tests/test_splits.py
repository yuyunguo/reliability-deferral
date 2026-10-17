"""Tests for patient-level train/val/test splitting (R1-1 / R2-1)."""
import numpy as np
import pytest

from deferral_frontier.src.splits import patient_group_split


def _synthetic_cohort(n_patients: int = 3000, seed: int = 0):
    """Patients with 1-4 admissions each and ~15% prevalence."""
    rng = np.random.default_rng(seed)
    n_adm = rng.integers(1, 5, size=n_patients)
    groups = np.repeat(np.arange(n_patients), n_adm)
    index = np.arange(len(groups)) + 10_000_000
    y = (rng.random(len(groups)) < 0.15).astype(int)
    return index, groups, y


def test_no_patient_in_two_splits():
    index, groups, y = _synthetic_cohort()
    tr, va, te = patient_group_split(index, groups, y)
    g = dict(zip(index, groups))
    gtr, gva, gte = ({g[i] for i in s} for s in (tr, va, te))
    assert not (gtr & gva) and not (gtr & gte) and not (gva & gte)


def test_every_row_assigned_exactly_once():
    index, groups, y = _synthetic_cohort()
    tr, va, te = patient_group_split(index, groups, y)
    allidx = np.concatenate([tr, va, te])
    assert len(allidx) == len(index)
    assert set(allidx) == set(index)


def test_sizes_within_2pct():
    index, groups, y = _synthetic_cohort()
    tr, va, te = patient_group_split(index, groups, y, sizes=(0.70, 0.15, 0.15))
    n = len(index)
    for part, target in ((tr, 0.70), (va, 0.15), (te, 0.15)):
        assert abs(len(part) / n - target) < 0.02


def test_prevalence_within_2pct():
    index, groups, y = _synthetic_cohort()
    tr, va, te = patient_group_split(index, groups, y)
    lab = dict(zip(index, y))
    overall = y.mean()
    for part in (tr, va, te):
        assert abs(np.mean([lab[i] for i in part]) - overall) < 0.02


def test_deterministic():
    index, groups, y = _synthetic_cohort()
    a = patient_group_split(index, groups, y, seed=42)
    b = patient_group_split(index, groups, y, seed=42)
    for x, z in zip(a, b):
        np.testing.assert_array_equal(x, z)


def test_rejects_missing_group():
    index, groups, y = _synthetic_cohort()
    groups = groups.astype(float)
    groups[5] = np.nan
    with pytest.raises(ValueError):
        patient_group_split(index, groups, y)


def test_attach_patient_ids_maps_all_rows():
    import pandas as pd
    from deferral_frontier.experiments.extract_patient_ids import attach_patient_ids
    rows = pd.Index([11, 12, 13], name="hadm_id")
    mapping = pd.Series([1, 1, 2], index=[11, 12, 13])
    out = attach_patient_ids(rows, mapping)
    assert out.index.equals(rows) and out["patient_id"].tolist() == [1, 1, 2]


def test_attach_patient_ids_fails_on_unmapped_row():
    import pandas as pd
    from deferral_frontier.experiments.extract_patient_ids import attach_patient_ids
    rows = pd.Index([11, 99], name="hadm_id")
    mapping = pd.Series([1], index=[11])
    with pytest.raises(ValueError, match="1 of 2"):
        attach_patient_ids(rows, mapping)


def test_patient_overlap_counts_test_patients_seen_in_development():
    import pandas as pd
    from deferral_frontier.experiments.r1_old_split_overlap import patient_overlap
    pid = pd.Series({1: "a", 2: "a", 3: "b", 4: "c", 5: "c", 6: "d"})
    res = patient_overlap(test_rows=pd.Index([2, 3, 5]), dev_rows=pd.Index([1, 4, 6]), pid=pid)
    assert res["test_rows"] == 3 and res["test_patients"] == 3
    assert res["test_patients_in_dev"] == 2 and res["test_rows_in_dev"] == 2
