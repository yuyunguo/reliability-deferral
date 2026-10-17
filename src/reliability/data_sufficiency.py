"""D2: Patient-baseline-adjusted data sufficiency."""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_patient_baseline_rate(
    timeseries: pd.DataFrame,
    stay_id: int,
    admission_time: pd.Timestamp,
    baseline_hours: int = 6,
) -> float:
    """Mean feature observation rate in the first N hours of admission.

    Args:
        timeseries: Long-format DataFrame with columns [stay_id, charttime, feature_name, value].
        stay_id: Patient stay identifier.
        admission_time: ICU admission timestamp.
        baseline_hours: Duration of baseline window.

    Returns:
        Proportion of (feature, time-slot) pairs observed. NaN if window empty.
    """
    end = admission_time + pd.Timedelta(hours=baseline_hours)
    patient_ts = timeseries[timeseries["stay_id"] == stay_id]
    window = patient_ts[
        (patient_ts["charttime"] >= admission_time)
        & (patient_ts["charttime"] < end)
    ]
    if window.empty:
        logger.warning("Empty baseline window for stay_id=%d.", stay_id)
        return float("nan")

    n_features = timeseries["feature_name"].nunique()
    n_slots = max(1, int(baseline_hours))
    observed = window["feature_name"].nunique()
    return float(observed / n_features)


def compute_data_sufficiency(
    timeseries: pd.DataFrame,
    stay_id: int,
    prediction_time: pd.Timestamp,
    patient_baseline_rate: float,
    lookback_hours: int = 24,
) -> float:
    """D2: observation rate in [T-lookback, T) normalized by patient baseline.

    Strictly prior to prediction_time — no future leakage.
    Clipped to [0, 1].
    """
    window_start = prediction_time - pd.Timedelta(hours=lookback_hours)
    patient_ts = timeseries[timeseries["stay_id"] == stay_id]
    window = patient_ts[
        (patient_ts["charttime"] >= window_start)
        & (patient_ts["charttime"] < prediction_time)
    ]

    if window.empty:
        return 0.0

    if np.isnan(patient_baseline_rate) or patient_baseline_rate == 0.0:
        return 0.0

    n_features = timeseries["feature_name"].nunique()
    observed = window["feature_name"].nunique()
    window_rate = observed / n_features
    return float(np.clip(window_rate / patient_baseline_rate, 0.0, 1.0))


def batch_compute_d2(
    timeseries: pd.DataFrame,
    stay_ids: pd.Index,
    admission_times: pd.Series,
    prediction_times: pd.Series,
    lookback_hours: int = 24,
    baseline_hours: int = 6,
) -> np.ndarray:
    """Compute D2 for all patients. Returns array aligned to stay_ids."""
    baseline_rates = {
        sid: compute_patient_baseline_rate(
            timeseries, sid, admission_times[sid], baseline_hours
        )
        for sid in stay_ids
    }
    return np.array([
        compute_data_sufficiency(
            timeseries, sid, prediction_times[sid],
            baseline_rates[sid], lookback_hours
        )
        for sid in stay_ids
    ])
