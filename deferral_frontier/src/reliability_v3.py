"""v3 reliability scores (plan/r1-reliability-v3-preregistration.md + amendment 1).

D3' : cohort similarity that sees missingness. The as-built D3 imputes before the
      Mahalanobis distance, so a patient whose labs are missing looks typical.
D5' : sensitivity of the NaN-trained primary model to hiding 10% of the observed
      entries. The as-built D5 multiplied imputed inputs by zero, which the model
      never saw during training.
Error models: cross-fitted on two patient-disjoint validation halves, exactly as
in v2 (calibrate on one half, build the (p - y)^2 target on the other).
"""
import logging
from typing import Any, Optional

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

from deferral_frontier.src.reliability_v2 import ErrorComposite
from src.calibration.baselines import IsotonicCalibrator

logger = logging.getLogger(__name__)

L_PARAMS = dict(objective="reg:squarederror", max_depth=2, n_estimators=200, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.5, min_child_weight=50, random_state=42)


class MissingAwareSimilarity:
    """D3' = exp(-d / median training d), d = Ledoit-Wolf Mahalanobis distance on
    [z-scored, train-median-imputed features, missingness indicators]."""

    def __init__(self) -> None:
        self.medians: Optional[np.ndarray] = None
        self.indicator_cols: Optional[np.ndarray] = None
        self._scaler: Optional[StandardScaler] = None
        self._cov: Optional[LedoitWolf] = None
        self._sigma: Optional[float] = None

    def _design(self, X_raw: np.ndarray) -> np.ndarray:
        miss = np.isnan(X_raw)
        X = np.where(miss, self.medians, X_raw)
        return np.column_stack([self._scaler.transform(X), miss[:, self.indicator_cols].astype(float)])

    def fit(self, X_raw_train: np.ndarray) -> "MissingAwareSimilarity":
        """Fit on NaN-preserving training features."""
        miss = np.isnan(X_raw_train)
        all_missing = miss.all(axis=0)
        med = np.zeros(X_raw_train.shape[1])
        med[~all_missing] = np.nanmedian(X_raw_train[:, ~all_missing], axis=0)
        self.medians = med
        self.indicator_cols = np.flatnonzero(miss.any(axis=0) & ~all_missing)
        self._scaler = StandardScaler().fit(np.where(miss, med, X_raw_train))
        Z = self._design(X_raw_train)
        self._cov = LedoitWolf().fit(Z)
        self._sigma = float(np.median(self._distance(Z)))
        logger.info("D3' fit: %d features + %d missingness indicators, sigma=%.3f",
                    X_raw_train.shape[1], len(self.indicator_cols), self._sigma)
        return self

    def _distance(self, Z: np.ndarray) -> np.ndarray:
        diff = Z - self._cov.location_
        return np.sqrt(np.einsum("ij,jk,ik->i", diff, self._cov.precision_, diff))

    def transform(self, X_raw: np.ndarray) -> np.ndarray:
        """D3' in (0, 1]; higher = closer to the training cohort."""
        return np.exp(-self._distance(self._design(X_raw)) / self._sigma)


def nan_mask_variance(model: Any, X_raw: np.ndarray, frac: float = 0.1, n_passes: int = 50,
                      seed: int = 42) -> np.ndarray:
    """Variance of P(y=1) when a random `frac` of each row's observed entries is set to NaN.

    D5' = -nan_mask_variance(...). Missing entries stay missing; the model sees only
    the representation it was trained on.
    """
    rng = np.random.default_rng(seed)
    observed = ~np.isnan(X_raw)
    preds = []
    for _ in range(n_passes):
        Xp = X_raw.copy()
        Xp[observed & (rng.random(X_raw.shape) < frac)] = np.nan
        preds.append(model.predict_proba(Xp)[:, 1])
    return np.var(np.stack(preds), axis=0)


def _design(p: np.ndarray, extra: np.ndarray) -> np.ndarray:
    return np.column_stack([p * (1 - p), extra])


def fit_error_model(model: Any, raw_val: np.ndarray, y_val: np.ndarray, extra_val: np.ndarray,
                    groups_val: np.ndarray, seed: int = 42) -> Any:
    """Cross-fit a regressor of (p - y)^2 on [p(1-p), extra] over validation halves.

    Args:
        model: Unfitted sklearn-style regressor (fit/predict).
        raw_val: Uncalibrated validation scores.
        y_val: Validation labels.
        extra_val: (n, m) extra inputs per validation row (may contain NaN for XGBoost).
        groups_val: Patient id per validation row; halves share no patient.
    """
    a, b = ErrorComposite._halves(np.asarray(groups_val), seed)
    X_parts, t_parts = [], []
    for fit_idx, apply_idx in ((a, b), (b, a)):
        p = IsotonicCalibrator().fit(raw_val[fit_idx], y_val[fit_idx]).predict_proba(raw_val[apply_idx])
        X_parts.append(_design(p, extra_val[apply_idx]))
        t_parts.append((p - y_val[apply_idx]) ** 2)
    return model.fit(np.vstack(X_parts), np.concatenate(t_parts))


def error_score(model: Any, p: np.ndarray, extra: np.ndarray) -> np.ndarray:
    """Reliability (higher = more reliable) = -predicted squared error."""
    return -model.predict(_design(p, extra))
