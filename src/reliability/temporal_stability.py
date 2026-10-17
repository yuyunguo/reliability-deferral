"""D4: Temporal stability — feature consistency in [T-24h, T)."""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_temporal_stability(
    timeseries: pd.DataFrame,
    stay_id: int,
    prediction_time: pd.Timestamp,
    lookback_hours: int = 24,
    min_observations: int = 2,
) -> float:
    """D4: 1 - mean coefficient of variation across features in lookback window.

    High D4 = stable clinical signal. Low D4 = high variability (deterioration
    or documentation inconsistency). Strictly prior to prediction_time.

    Args:
        timeseries: Long-format DataFrame [stay_id, charttime, feature_name, value].
        stay_id: Patient stay identifier.
        prediction_time: Upper bound (exclusive) of the window.
        lookback_hours: Window size in hours (default: 24, pre-committed).
        min_observations: Minimum observations per feature to include in CV.
                          Features with fewer observations are excluded.

    Returns:
        D4 ∈ [0, 1]. Returns 0.5 (neutral) if window has insufficient data.
    """
    window_start = prediction_time - pd.Timedelta(hours=lookback_hours)
    patient_ts = timeseries[timeseries["stay_id"] == stay_id]

    # Strictly prior to prediction_time — no future leakage
    window = patient_ts[
        (patient_ts["charttime"] >= window_start)
        & (patient_ts["charttime"] < prediction_time)
    ]

    if window.empty:
        logger.debug("Empty window for stay_id=%d at %s.", stay_id, prediction_time)
        return 0.5

    cv_values = []
    for feature, group in window.groupby("feature_name"):
        vals = group["value"].dropna().values
        if len(vals) < min_observations:
            continue
        mean = np.mean(vals)
        std = np.std(vals, ddof=1)
        if mean == 0:
            continue
        cv_values.append(std / abs(mean))

    if not cv_values:
        return 0.5

    mean_cv = float(np.mean(cv_values))
    return float(1.0 - np.clip(mean_cv, 0.0, 1.0))


def batch_compute_d4(
    timeseries: pd.DataFrame,
    stay_ids: pd.Index,
    prediction_times: pd.Series,
    lookback_hours: int = 24,
) -> np.ndarray:
    """Compute D4 for all patients."""
    return np.array([
        compute_temporal_stability(
            timeseries, sid, prediction_times[sid], lookback_hours
        )
        for sid in stay_ids
    ])
