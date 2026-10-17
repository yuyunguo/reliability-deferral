"""R composite: logistic regression over five reliability dimensions."""
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

DIMENSION_NAMES = ["D1_confidence", "D2_sufficiency", "D3_similarity",
                   "D4_stability", "D5_uncertainty"]


@dataclass
class DimensionMatrix:
    """Five reliability dimensions aligned to a sample set."""
    D1: np.ndarray  # predictive confidence (calibrated)
    D2: np.ndarray  # data sufficiency
    D3: np.ndarray  # cohort similarity
    D4: np.ndarray  # temporal stability
    D5: np.ndarray  # uncertainty consistency

    def to_array(self) -> np.ndarray:
        """Stack into (n_samples, 5) matrix."""
        return np.column_stack([self.D1, self.D2, self.D3, self.D4, self.D5])


class ReliabilityComposite:
    """Learn and apply R = σ(wᵀd + b) on validation set.

    Binary label for logistic regression: high calibration error (M-ECE > median).
    Weights frozen after validation fit; applied unchanged to test set.
    """

    def __init__(self) -> None:
        self._scaler: Optional[StandardScaler] = None
        self._lr: Optional[LogisticRegression] = None
        self._fitted: bool = False

    def fit(
        self,
        dims: DimensionMatrix,
        calibration_error: np.ndarray,
    ) -> "ReliabilityComposite":
        """Fit on validation-set dimensions and calibration errors.

        Args:
            dims: DimensionMatrix for validation patients.
            calibration_error: Per-sample absolute calibration error |p̃ - y|,
                               shape (n_samples,). Binary label = error > median.
        """
        X = dims.to_array()
        y = (calibration_error > np.median(calibration_error)).astype(int)

        self._scaler = StandardScaler().fit(X)
        X_scaled = self._scaler.transform(X)

        self._lr = LogisticRegression(max_iter=1000, random_state=42).fit(X_scaled, y)
        self._fitted = True

        coef = dict(zip(DIMENSION_NAMES, self._lr.coef_[0]))
        logger.info("ReliabilityComposite fit. Coefficients: %s", coef)
        return self

    def transform(self, dims: DimensionMatrix) -> np.ndarray:
        """Compute R ∈ [0, 1] for each sample. Higher = more reliable."""
        self._check_fitted()
        X = dims.to_array()
        X_scaled = self._scaler.transform(X)
        # Probability of LOW calibration error = P(y=0)
        r_scores = self._lr.predict_proba(X_scaled)[:, 0]
        return r_scores

    def equal_weight_transform(self, dims: DimensionMatrix) -> np.ndarray:
        """Ablation: equal-weight composite (mean of standardized dims)."""
        X = dims.to_array()
        if self._scaler is not None:
            X = self._scaler.transform(X)
        return X.mean(axis=1)

    def single_dim_transform(
        self, dims: DimensionMatrix, dim_idx: int
    ) -> np.ndarray:
        """Ablation: single dimension reliability score."""
        return dims.to_array()[:, dim_idx]

    def save(self, path: Path) -> None:
        self._check_fitted()
        with open(path, "wb") as f:
            pickle.dump({"scaler": self._scaler, "lr": self._lr}, f)
        logger.info("Saved ReliabilityComposite to %s", path)

    @classmethod
    def load(cls, path: Path) -> "ReliabilityComposite":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls()
        obj._scaler = state["scaler"]
        obj._lr = state["lr"]
        obj._fitted = True
        logger.info("Loaded ReliabilityComposite from %s", path)
        return obj

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Must call fit() before transform().")
