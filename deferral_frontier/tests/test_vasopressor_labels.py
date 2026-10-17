"""Tests for the corrected MIMIC-IV vasopressor label (R1-2 / R2-2)."""
import pandas as pd
import pytest

from deferral_frontier.src.icu_labels import sepsis_labels, vasopressor_labels

T0 = pd.Timestamp("2150-01-01 00:00")


@pytest.fixture
def stays():
    # hadm_id -> stay window; prediction at +24 h
    return pd.DataFrame({
        "stay_id": [1, 2, 3, 4],
        "prediction_time": [T0 + pd.Timedelta(hours=24)] * 4,
        "outtime": [T0 + pd.Timedelta(days=5)] * 4,
    }, index=pd.Index([10, 20, 30, 40], name="hadm_id"))


def test_pre24h_vasopressor_excluded_late_start_positive_none_negative(stays):
    events = pd.DataFrame({
        "stay_id": [1, 2, 2, 4],
        "event_time": [T0 + pd.Timedelta(hours=3),     # stay 1: early -> excluded
                       T0 + pd.Timedelta(hours=30),    # stay 2: late -> positive
                       T0 + pd.Timedelta(hours=40),
                       T0 + pd.Timedelta(days=6)],     # stay 4: after discharge -> negative
    })
    lab = vasopressor_labels(events, stays)
    assert 10 not in lab.index
    assert lab.loc[20] == 1 and lab.loc[30] == 0 and lab.loc[40] == 0
    assert lab.index.name == "hadm_id" and lab.name == "label"


def test_event_exactly_at_prediction_time_counts_as_late(stays):
    events = pd.DataFrame({"stay_id": [3], "event_time": [T0 + pd.Timedelta(hours=24)]})
    lab = vasopressor_labels(events, stays)
    assert lab.loc[30] == 1


def test_sepsis_labels_use_same_severe_sepsis_codes_in_icd9_and_icd10():
    dx = pd.DataFrame({
        "hadm_id": [1, 2, 3, 4, 5],
        "icd_code": ["99592", "R6521", "A41.9", "0389", "A409"],  # last two: septicemia only
        "icd_version": [9, 10, 10, 9, 10],
    })
    lab = sepsis_labels(dx, pd.Index([1, 2, 3, 4, 5, 6], name="hadm_id"))
    assert lab.tolist() == [1, 1, 1, 0, 0, 0]
    assert lab.name == "label" and lab.index.name == "hadm_id"


def test_vasopressor_itemids_are_the_mimic_code_vasopressor_set():
    from deferral_frontier.src.icu_labels import VASO_IDS_CV, VASO_IDS_MV
    # norepinephrine, epinephrine, vasopressin, phenylephrine, dopamine
    assert set(VASO_IDS_MV) == {221906, 221289, 222315, 221749, 221662}
    assert set(VASO_IDS_CV) == {30047, 30120, 30044, 30119, 30309, 30051,
                                30127, 30128, 30043, 30307}


def test_vasopressor_labels_when_row_id_is_the_stay_id(stays):
    s = stays.set_index("stay_id", drop=False)      # eICU: row id == stay id
    s.index.name = "stay_id"
    events = pd.DataFrame({"stay_id": [2], "event_time": [T0 + pd.Timedelta(hours=30)]})
    lab = vasopressor_labels(events, s)
    assert lab.loc[2] == 1 and lab.loc[1] == 0
