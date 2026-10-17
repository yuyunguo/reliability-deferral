"""Optimized batch D2/D4 computation using groupby operations.

Avoids per-patient DataFrame filtering (the original per-patient loop
scans the full 8M-row timeseries once per patient = 84K full scans).
Instead uses merge + groupby aggregations for near-instant computation.
"""
import numpy as np
import pandas as pd


def compute_d2_fast(
    timeseries: pd.DataFrame,
    stay_ids: pd.Index,
    admission_times: pd.Series,
    prediction_times: pd.Series,
    lookback_hours: int = 24,
    baseline_hours: int = 6,
) -> np.ndarray:
    """D2: observation rate normalized by patient baseline.

    Uses a single merge + groupby instead of per-patient loops.
    ~1000x faster on MIMIC-IV (84K patients, 8M timeseries rows).
    """
    baseline_delta = pd.Timedelta(hours=baseline_hours)
    lookback_delta = pd.Timedelta(hours=lookback_hours)

    # Build meta lookup
    meta = pd.DataFrame({
        "admission_time": admission_times,
        "prediction_time": prediction_times,
        "baseline_end": admission_times + baseline_delta,
        "lookback_start": prediction_times - lookback_delta,
    }, index=stay_ids)

    # One merge instead of 84K filters
    ts = timeseries.merge(meta, left_on="stay_id", right_index=True, how="inner")
    n_features = timeseries["feature_name"].nunique()

    # Baseline: unique features in [admission, admission + baseline)
    baseline = ts[
        (ts["charttime"] >= ts["admission_time"])
        & (ts["charttime"] < ts["baseline_end"])
    ]
    baseline_counts = baseline.groupby("stay_id")["feature_name"].nunique()
    baseline_rate = baseline_counts / n_features

    # Lookback: unique features in [T - lookback, T)
    lookback = ts[
        (ts["charttime"] >= ts["lookback_start"])
        & (ts["charttime"] < ts["prediction_time"])
    ]
    lookback_counts = lookback.groupby("stay_id")["feature_name"].nunique()
    lookback_rate = lookback_counts / n_features

    # D2 = lookback_rate / baseline_rate, clipped to [0, 1]
    result = pd.Series(0.0, index=stay_ids)
    valid = baseline_rate.reindex(stay_ids, fill_value=0.0) > 0
    valid_lr = lookback_rate.reindex(stay_ids, fill_value=0.0)
    result[valid] = np.clip(valid_lr[valid] / baseline_rate[valid], 0.0, 1.0)
    return result.values


def compute_d4_fast(
    timeseries: pd.DataFrame,
    stay_ids: pd.Index,
    prediction_times: pd.Series,
    lookback_hours: int = 24,
    min_observations: int = 2,
) -> np.ndarray:
    """D4: 1 - mean coefficient of variation across features.

    Uses merge + groupby aggregations instead of per-patient loops.
    """
    lookback_delta = pd.Timedelta(hours=lookback_hours)

    meta = pd.DataFrame({
        "prediction_time": prediction_times,
        "lookback_start": prediction_times - lookback_delta,
    }, index=stay_ids)

    ts = timeseries.merge(meta, left_on="stay_id", right_index=True, how="inner")

    window = ts[
        (ts["charttime"] >= ts["lookback_start"])
        & (ts["charttime"] < ts["prediction_time"])
    ]

    if window.empty:
        return np.full(len(stay_ids), 0.5)

    # Per (stay_id, feature): mean, std, count
    stats = window.groupby(["stay_id", "feature_name"])["value"].agg(["mean", "std", "count"])
    stats = stats[stats["count"] >= min_observations]
    stats = stats[stats["mean"] != 0.0]
    if len(stats) == 0:
        return np.full(len(stay_ids), 0.5)

    stats["cv"] = stats["std"] / stats["mean"].abs()
    mean_cv = stats.groupby("stay_id")["cv"].mean()

    d4 = 1.0 - mean_cv.clip(0.0, 1.0)
    result = d4.reindex(stay_ids, fill_value=0.5)
    return result.values
