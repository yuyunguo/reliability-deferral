"""Composite v2: a cross-fitted squared-error model over reliability dimensions.

Implements plan/r1-composite-v2-preregistration.md exactly. Dimensions per patient:
D1' = p(1-p), D2' = fraction of signals observed, D3 = cohort similarity,
D4 = temporal stability, D5' = raw MC-dropout variance. The regressor predicts
(p - y)^2, is monotone increasing in D1', and is fit on cross-fitted validation
halves so the calibrator and the error target never share patients.
"""
import logging
from typing import Callable, Tuple

import numpy as np
import xgboost as xgb

from src.calibration.baselines import IsotonicCalibrator

logger = logging.getLogger(__name__)

XGB_V2 = dict(objective="reg:squarederror", max_depth=3, n_estimators=300,
              learning_rate=0.05, subsample=0.8, random_state=42,
              monotone_constraints="(1,0,0,0,0)")
DIM_NAMES_V2 = ("D1_p1mp", "D2_sufficiency", "D3_similarity", "D4_stability", "D5_mc_variance")


def sufficiency_fraction(counts: np.ndarray) -> np.ndarray:
    """D2': fraction of signals with at least one observation in the window."""
    return (np.asarray(counts) > 0).mean(axis=1)


def sufficiency_from(X: np.ndarray, feature_names, counts: np.ndarray = None) -> np.ndarray:
    """D2' from supplied count columns, else from the `_count` columns of X."""
    if counts is None:
        idx = [i for i, c in enumerate(feature_names) if c.endswith("_count")]
        counts = X[:, idx]
    return sufficiency_fraction(counts)


def mc_dropout_variance(dropout_fn: Callable[[np.ndarray], np.ndarray], X: np.ndarray,
                        n_passes: int = 50) -> np.ndarray:
    """D5': variance of predicted probability across MC-dropout passes (not normalized)."""
    return np.var(np.stack([dropout_fn(X) for _ in range(n_passes)], axis=0), axis=0)


class ErrorComposite:
    """Predict each patient's squared error; reliability = -predicted error."""

    def __init__(self) -> None:
        self._model: xgb.XGBRegressor = None

    @staticmethod
    def _halves(groups: np.ndarray, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Positions of two validation halves with no group (patient) in both."""
        uniq = np.unique(groups)
        rng = np.random.default_rng(seed)
        first = set(rng.permutation(uniq)[: len(uniq) // 2])
        in_a = np.array([g in first for g in groups])
        return np.flatnonzero(in_a), np.flatnonzero(~in_a)

    @staticmethod
    def _design(p: np.ndarray, other: np.ndarray) -> np.ndarray:
        return np.column_stack([p * (1 - p), other])

    def fit(self, raw_val: np.ndarray, y_val: np.ndarray, other_val: np.ndarray,
            groups_val: np.ndarray, seed: int = 42) -> "ErrorComposite":
        """Cross-fit: calibrate on one half, build the error target on the other, pool, fit.

        Args:
            raw_val: Uncalibrated validation scores.
            y_val: Validation labels.
            other_val: (n, 4) array of D2', D3, D4, D5' for validation patients.
            groups_val: Patient id per validation row.
        """
        a, b = self._halves(np.asarray(groups_val), seed)
        X_parts, t_parts = [], []
        for fit_idx, apply_idx in ((a, b), (b, a)):
            iso = IsotonicCalibrator().fit(raw_val[fit_idx], y_val[fit_idx])
            p = iso.predict_proba(raw_val[apply_idx])
            X_parts.append(self._design(p, other_val[apply_idx]))
            t_parts.append((p - y_val[apply_idx]) ** 2)
        self._model = xgb.XGBRegressor(**XGB_V2).fit(np.vstack(X_parts), np.concatenate(t_parts))
        gain = self._model.get_booster().get_score(importance_type="gain")
        logger.info("ErrorComposite gain: %s", {DIM_NAMES_V2[int(k[1:])]: round(v, 6) for k, v in gain.items()})
        return self

    def score(self, p: np.ndarray, other: np.ndarray) -> np.ndarray:
        """Reliability (higher = defer later) = -predicted squared error."""
        return -self._model.predict(self._design(p, other))

    def importance(self) -> dict:
        """Gain importance per dimension (0 for unused dimensions)."""
        gain = self._model.get_booster().get_score(importance_type="gain")
        return {name: gain.get(f"f{i}", 0.0) for i, name in enumerate(DIM_NAMES_V2)}


def patient_groups(data_dir, row_ids) -> np.ndarray:
    """Patient id for each row id, from data_dir/patient_ids.parquet."""
    import pandas as pd
    from pathlib import Path
    pid = pd.read_parquet(Path(data_dir) / "patient_ids.parquet")["patient_id"]
    return pid.reindex(row_ids).to_numpy()


def build_v2(raw_val, y_val, p_val, p_test, X_val, X_test, feature_names, d3_val, d3_test,
             d4_val, d4_test, dropout_fn, groups_val, counts_val=None, counts_test=None):
    """Compute v2 dimensions, fit the cross-fitted ErrorComposite, and score.

    counts_val/counts_test: optional observation-count matrices for D2' when the model
    features exclude the count columns (harmonized analysis); default reads them from X.

    Returns:
        (composite, dims_val, dims_test, r_val, r_test, other_test) where dims_* are
        DimensionMatrix(D1', D2', D3, D4, D5') and other_test holds D2'..D5' for test.
    """
    from src.reliability.composite import DimensionMatrix
    d2_val = sufficiency_from(X_val, feature_names, counts_val)
    d2_test = sufficiency_from(X_test, feature_names, counts_test)
    # D5' stored as negative MC variance so that higher = more reliable for every stored
    # dimension; the sign is immaterial to the (unconstrained-in-D5') tree model.
    d5_val, d5_test = -mc_dropout_variance(dropout_fn, X_val), -mc_dropout_variance(dropout_fn, X_test)
    other_val = np.column_stack([d2_val, d3_val, d4_val, d5_val])
    other_test = np.column_stack([d2_test, d3_test, d4_test, d5_test])
    comp = ErrorComposite().fit(raw_val, y_val, other_val, groups_val)
    r_val, r_test = comp.score(p_val, other_val), comp.score(p_test, other_test)
    dims_val = DimensionMatrix(p_val * (1 - p_val), d2_val, d3_val, d4_val, d5_val)
    dims_test = DimensionMatrix(p_test * (1 - p_test), d2_test, d3_test, d4_test, d5_test)
    return comp, dims_val, dims_test, r_val, r_test, other_test
