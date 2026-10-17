"""D3: Mahalanobis-based cohort similarity. Frozen after training."""
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)


class CohortSimilarityEstimator:
    """Cohort similarity via regularized Mahalanobis distance.

    Must be fit on training data only. Fit freezes mean, precision matrix,
    and the median training distance (σ) used for normalization.
    """

    def __init__(self) -> None:
        self._cov: Optional[LedoitWolf] = None
        self._sigma: Optional[float] = None
        self._fitted: bool = False

    def fit(self, X_train: np.ndarray) -> "CohortSimilarityEstimator":
        """Fit on standardized training features (no NaN allowed).

        Args:
            X_train: Shape (n_samples, n_features).
        """
        if np.isnan(X_train).any():
            raise ValueError("X_train contains NaN — impute before fitting.")

        self._cov = LedoitWolf().fit(X_train)
        distances = self._mahalanobis(X_train)
        self._sigma = float(np.median(distances))

        if self._sigma < 1e-8:
            raise ValueError("Median Mahalanobis distance near zero — check features.")

        self._fitted = True
        logger.info(
            "CohortSimilarity fit: n=%d, p=%d, sigma=%.4f, "
            "D3_train mean=%.4f",
            X_train.shape[0], X_train.shape[1], self._sigma,
            np.exp(-distances / self._sigma).mean(),
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """D3 ∈ (0, 1] for each sample. 1 = at training centroid."""
        self._check_fitted()
        if np.isnan(X).any():
            raise ValueError("X contains NaN — impute before computing D3.")
        return np.exp(-self._mahalanobis(X) / self._sigma)

    def save(self, path: Path) -> None:
        self._check_fitted()
        with open(path, "wb") as f:
            pickle.dump({"cov": self._cov, "sigma": self._sigma}, f)
        logger.info("Saved CohortSimilarityEstimator to %s", path)

    @classmethod
    def load(cls, path: Path) -> "CohortSimilarityEstimator":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls()
        obj._cov = state["cov"]
        obj._sigma = state["sigma"]
        obj._fitted = True
        logger.info("Loaded CohortSimilarityEstimator from %s", path)
        return obj

    def _mahalanobis(self, X: np.ndarray) -> np.ndarray:
        mu = self._cov.location_
        VI = self._cov.precision_
        diff = X - mu
        return np.sqrt(np.einsum("ij,jk,ik->i", diff, VI, diff))

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Must call fit() before transform().")
