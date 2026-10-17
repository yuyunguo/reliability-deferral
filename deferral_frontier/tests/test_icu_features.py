"""Tests for the shared first-24 h ICU feature aggregation (round-1 re-extraction)."""
import numpy as np
import pandas as pd

from deferral_frontier.src.icu_features import FEATURE_COLUMNS, aggregate_window

T0 = pd.Timestamp("2150-01-01 00:00")


def _h(hours: float) -> pd.Timestamp:
    return T0 + pd.Timedelta(hours=hours)


def _windows(ids):
    start = pd.Series([T0] * len(ids), index=ids)
    return start, start + pd.Timedelta(hours=24)


def test_feature_columns_are_23_signals_by_5_statistics():
    assert len(FEATURE_COLUMNS) == 115
    assert FEATURE_COLUMNS[:5] == ["vital_heart_rate_mean", "vital_heart_rate_min",
                                   "vital_heart_rate_max", "vital_heart_rate_last",
                                   "vital_heart_rate_count"]
    assert FEATURE_COLUMNS[-1] == "lab_lactate_count"


def test_observations_outside_window_are_dropped():
    ts = pd.DataFrame({"id": [1, 1, 1, 1],
                       "charttime": [_h(-1), _h(0), _h(23.9), _h(24)],
                       "feature_name": ["vital_heart_rate"] * 4,
                       "value": [500.0, 80.0, 90.0, 600.0]})
    start, end = _windows([1])
    out = aggregate_window(ts, "id", start, end)
    assert out.loc[1, "vital_heart_rate_count"] == 2
    assert out.loc[1, "vital_heart_rate_max"] == 90.0


def test_last_is_chronological_not_row_order():
    ts = pd.DataFrame({"id": [1, 1], "charttime": [_h(10), _h(2)],
                       "feature_name": ["lab_lactate"] * 2, "value": [4.0, 1.0]})
    start, end = _windows([1])
    out = aggregate_window(ts, "id", start, end)
    assert out.loc[1, "lab_lactate_last"] == 4.0


def test_output_has_all_columns_and_rows_with_zero_counts_for_missing_signals():
    ts = pd.DataFrame({"id": [1], "charttime": [_h(1)],
                       "feature_name": ["vital_heart_rate"], "value": [70.0]})
    start, end = _windows([1, 2])
    out = aggregate_window(ts, "id", start, end)
    assert list(out.columns) == FEATURE_COLUMNS
    assert list(out.index) == [1, 2]
    assert out.loc[1, "lab_lactate_count"] == 0 and np.isnan(out.loc[1, "lab_lactate_mean"])
    assert out.loc[2, "vital_heart_rate_count"] == 0
