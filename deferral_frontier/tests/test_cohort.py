"""Tests for the shared MIMIC ICU cohort rule (round-1 re-extraction)."""
import pandas as pd

from deferral_frontier.src.icu_labels import select_cohort

T0 = pd.Timestamp("2150-01-01")


def test_select_cohort_adult_los24h_first_stay_per_admission():
    stays = pd.DataFrame({
        "stay_id": [1, 2, 3, 4, 5],
        "hadm_id": [10, 10, 20, 30, 40],
        "intime": [T0 + pd.Timedelta(days=d) for d in (2, 1, 0, 0, 0)],
        "los": [3.0, 2.0, 0.5, 4.0, 2.0],
        "age": [60, 60, 50, 17, 300],          # 300 = MIMIC-III shifted >89
    })
    out = select_cohort(stays)
    assert list(out.index) == [10, 40]
    assert out.index.name == "hadm_id"
    assert out.loc[10, "stay_id"] == 2          # earliest eligible stay
    assert (out["prediction_time"] - out["admission_time"] == pd.Timedelta(hours=24)).all()
