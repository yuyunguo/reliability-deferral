"""R-conditioned stratified isotonic regression recalibration.

Three tertile bins of R score; independent isotonic calibrator per bin.
Boundaries and calibrators frozen after validation fit.
"""
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


class RConditionedCalibrator:
    """Stratified isotonic regression conditioned on reliability score R.

    Pre-committed design:
    - 3 tertile bins (low / mid / high R)
    - Bin boundaries from validation set (frozen)
    - Independent IsotonicRegression per bin
    """

    def __init__(self, n_bins: int = 3) -> None:
        self.n_bins = n_bins
        self._boundaries: Optional[np.ndarray] = None
        self._calibrators: Optional[list] = None

    def fit(
        self,
        r_scores: np.ndarray,
        probs: np.ndarray,
        labels: np.ndarray,
    ) -> "RConditionedCalibrator":
        """Fit on validation set. Freeze tertile boundaries and isotonic fits.

        Args:
            r_scores: R composite scores ∈ [0,1], shape (n_val,).
            probs: Uncalibrated model probabilities, shape (n_val,).
            labels: Binary outcomes, shape (n_val,).
        """
        quantiles = np.linspace(0, 100, self.n_bins + 1)
        self._boundaries = np.percentile(r_scores, quantiles)
        self._boundaries[0] -= 1e-8
        self._boundaries[-1] += 1e-8

        self._calibrators = []
        for k in range(self.n_bins):
            lo, hi = self._boundaries[k], self._boundaries[k + 1]
            mask = (r_scores > lo) & (r_scores <= hi)
            if mask.sum() < 5:
                logger.warning(
                    "Bin %d has only %d samples — using global isotonic.", k, mask.sum()
                )
                iso = IsotonicRegression(out_of_bounds="clip").fit(probs, labels)
            else:
                iso = IsotonicRegression(out_of_bounds="clip").fit(
                    probs[mask], labels[mask]
                )
            self._calibrators.append(iso)
            logger.info(
                "Bin %d [R ∈ (%.3f, %.3f)]: n=%d, mean_label=%.3f",
                k, lo, hi, mask.sum(), labels[mask].mean() if mask.sum() > 0 else 0,
            )
        return self

    def predict_proba(
        self,
        r_scores: np.ndarray,
        probs: np.ndarray,
    ) -> np.ndarray:
        """Apply frozen calibrators. Each sample routed to its R bin.

        Args:
            r_scores: R scores for test samples.
            probs: Uncalibrated model probabilities for test samples.

        Returns:
            Calibrated probabilities, shape (n,).
        """
        if self._boundaries is None or self._calibrators is None:
            raise RuntimeError("Must call fit() first.")

        calibrated = np.empty(len(probs))
        for k in range(self.n_bins):
            lo, hi = self._boundaries[k], self._boundaries[k + 1]
            mask = (r_scores > lo) & (r_scores <= hi)
            if mask.sum() > 0:
                calibrated[mask] = self._calibrators[k].predict(probs[mask])

        return calibrated

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {"boundaries": self._boundaries, "calibrators": self._calibrators}, f
            )
        logger.info("Saved RConditionedCalibrator to %s", path)

    @classmethod
    def load(cls, path: Path) -> "RConditionedCalibrator":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls()
        obj._boundaries = state["boundaries"]
        obj._calibrators = state["calibrators"]
        logger.info("Loaded RConditionedCalibrator from %s", path)
        return obj
