"""Post-hoc calibration baselines: temperature scaling, Platt, isotonic regression."""
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class TemperatureScaling:
    """Guo et al. (2017) temperature scaling for post-hoc calibration."""

    def __init__(self) -> None:
        self._T: Optional[float] = None

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TemperatureScaling":
        """Find T minimizing NLL on validation logits."""
        def nll(T: float) -> float:
            scaled = expit(logits / T)
            return -np.mean(
                labels * np.log(scaled + 1e-12)
                + (1 - labels) * np.log(1 - scaled + 1e-12)
            )

        result = minimize_scalar(nll, bounds=(0.01, 10.0), method="bounded")
        self._T = float(result.x)
        logger.info("TemperatureScaling: T=%.4f", self._T)
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        if self._T is None:
            raise RuntimeError("Must call fit() first.")
        return expit(logits / self._T)


class PlattScaling:
    """Platt scaling: logistic regression on raw model scores."""

    def __init__(self) -> None:
        self._lr: Optional[LogisticRegression] = None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "PlattScaling":
        self._lr = LogisticRegression(max_iter=1000).fit(
            scores.reshape(-1, 1), labels
        )
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        if self._lr is None:
            raise RuntimeError("Must call fit() first.")
        return self._lr.predict_proba(scores.reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    """Non-parametric isotonic regression calibrator."""

    def __init__(self) -> None:
        self._iso: Optional[IsotonicRegression] = None

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        self._iso = IsotonicRegression(out_of_bounds="clip").fit(probs, labels)
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        if self._iso is None:
            raise RuntimeError("Must call fit() first.")
        return self._iso.predict(probs)


class UniformConfidence:
    """Trivial baseline: always predict base rate prevalence."""

    def __init__(self) -> None:
        self._rate: Optional[float] = None

    def fit(self, labels: np.ndarray) -> "UniformConfidence":
        self._rate = float(labels.mean())
        logger.info("UniformConfidence: base rate=%.4f", self._rate)
        return self

    def predict_proba(self, n: int) -> np.ndarray:
        if self._rate is None:
            raise RuntimeError("Must call fit() first.")
        return np.full(n, self._rate)


def save_calibrator(calibrator: object, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(calibrator, f)


def load_calibrator(path: Path) -> object:
    with open(path, "rb") as f:
        return pickle.load(f)
