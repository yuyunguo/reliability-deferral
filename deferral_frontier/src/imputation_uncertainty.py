"""Data sufficiency as imputation uncertainty (plan/r1-d2-imputation-preregistration.md).

D2* = variance of the primary model's predicted probability over stochastic completions
of the patient's missing feature values. Completions are drawn by predictive mean matching
(PMM, the default in modern multiple imputation, van Buuren 2011): each missing value is
imputed with the actual observed value of a "donor" whose conditional mean is closest.

D2* answers "how much would the risk estimate move if we could measure the missing
signals". It is the continuous, value-of-information replacement for the *submitted* D2
(the 24h/baseline observation ratio, `compute_d2_fast`), which saturates at ~1.0 for
99.9% of patients under ICU anchoring. The deferral score reported downstream is
`-D2*` (higher = more reliable), matching the sign convention of every other score.

Complement to D5' (NaN-mask variance): D5' hides *observed* entries and is ~0 for a
patient with almost nothing recorded; D2* fills *missing* entries and is *large* for
exactly that patient. They measure different failure modes.
"""
import logging
from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class StochasticImputer:
    """Per-feature conditional imputer using predictive mean matching (single iteration).

    For each feature with any missing value in training, fit a Ridge regressor on the
    standardized, median-imputed complete feature vector (the fitted column excluded).
    A missing value is completed by drawing from the `n_donors` observed training values
    whose *value* is closest to the ridge prediction. Donor draws are always observed
    values, so completions never leave the feature's support.
    """

    def __init__(self, alpha: float = 1.0, n_donors: int = 5, seed: int = 42) -> None:
        self.alpha = alpha
        self.n_donors = n_donors
        self.seed = seed
        self.cols: List[int] = []
        self._models: Dict[int, Ridge] = {}
        self._donors: Dict[int, np.ndarray] = {}
        self._scaler: Optional[StandardScaler] = None

    def fit(self, X_imp: np.ndarray, X_raw: np.ndarray) -> "StochasticImputer":
        """Fit on the median-imputed complete matrix `X_imp` and NaN-preserving `X_raw`.

        `X_imp` supplies the conditioning set; `X_raw` marks missing entries and supplies
        the observed target values and donor pool. Both are (n, p), same column order.
        """
        X_imp = np.asarray(X_imp, dtype=float)
        X_raw = np.asarray(X_raw, dtype=float)
        if X_imp.shape != X_raw.shape:
            raise ValueError("X_imp and X_raw must have the same shape")
        n, p = X_raw.shape
        missing_any = np.isnan(X_raw).any(axis=0)
        self.cols = [int(j) for j in np.flatnonzero(missing_any)]
        self._scaler = StandardScaler().fit(X_imp)
        Z = self._scaler.transform(X_imp)
        for j in self.cols:
            obs = ~np.isnan(X_raw[:, j])
            if obs.sum() <= self.n_donors:
                logger.warning("feature %d has <= %d observed rows; skipping", j, self.n_donors)
                continue
            y = X_raw[obs, j]
            self._models[j] = Ridge(alpha=self.alpha).fit(np.delete(Z[obs], j, axis=1), y)
            self._donors[j] = np.sort(y)
        logger.info("StochasticImputer fit: %d of %d features imputed (PMM, d=%d)",
                    len(self._models), p, self.n_donors)
        return self

    def predict_mean_matrix(self, X_imp: np.ndarray, X_raw: np.ndarray) -> np.ndarray:
        """Deterministic completion: observed values kept, missing = ridge prediction."""
        X_imp = np.asarray(X_imp, dtype=float)
        X_raw = np.asarray(X_raw, dtype=float)
        if self._scaler is None:
            raise RuntimeError("imputer not fitted")
        mu = X_raw.copy()
        if not self._models:
            return mu
        Z = self._scaler.transform(X_imp)
        for j in self._models:
            missing = np.isnan(X_raw[:, j])
            if missing.any():
                mu[missing, j] = self._models[j].predict(np.delete(Z[missing], j, axis=1))
        return mu

    def sample_from_mean(self, mu: np.ndarray, X_raw: np.ndarray,
                         rng: np.random.Generator) -> np.ndarray:
        """One completion: `mu`'s missing entries replaced by PMM donor draws."""
        Xp = np.asarray(mu, dtype=float).copy()
        X_raw = np.asarray(X_raw, dtype=float)
        for j in self._models:
            missing = np.isnan(X_raw[:, j])
            if missing.any():
                Xp[missing, j] = self._draw_donors(j, mu[missing, j], rng)
        return Xp

    def sample(self, X_imp: np.ndarray, X_raw: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """One completion from `X_imp`/`X_raw` (convenience wrapper)."""
        return self.sample_from_mean(self.predict_mean_matrix(X_imp, X_raw), X_raw, rng)

    def _draw_donors(self, j: int, targets: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """For each ridge prediction, draw one of the `n_donors` nearest observed values."""
        donors = self._donors[j]
        d = self.n_donors
        n = len(donors)
        idx = np.searchsorted(donors, targets)
        out = np.empty_like(targets)
        for i, (t, ii) in enumerate(zip(targets, idx)):
            lo = ii - d // 2
            if lo < 0:
                lo = 0
            hi = lo + d
            if hi > n:
                hi = n
                lo = max(0, n - d)
            cand = donors[lo:hi]
            out[i] = cand[rng.integers(len(cand))]
        return out


def imputation_uncertainty(model, X_imp: np.ndarray, X_raw: np.ndarray,
                           imputer: StochasticImputer, n_passes: int = 50,
                           seed: int = 42) -> np.ndarray:
    """D2* = variance of P(y=1) across PMM completions of the missing entries.

    Args:
        model: Fitted classifier with `predict_proba` (the NaN-preserving primary model).
        X_imp: Median-imputed complete matrix (conditioning set), same order as X_raw.
        X_raw: NaN-preserving raw feature matrix.
        imputer: A fitted :class:`StochasticImputer`.
        n_passes: Number of completions (B).
        seed: RNG seed; draws are reproducible.

    Returns:
        Per-row D2* (>= 0). Rows with no missing values return 0 (all completions identical).
    """
    X_raw = np.asarray(X_raw, dtype=float)
    rng = np.random.default_rng(seed)
    mu = imputer.predict_mean_matrix(X_imp, X_raw)
    preds = []
    for _ in range(n_passes):
        Xp = imputer.sample_from_mean(mu, X_raw, rng)
        preds.append(model.predict_proba(Xp)[:, 1])
    return np.var(np.stack(preds, axis=0), axis=0)
