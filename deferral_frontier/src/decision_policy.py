"""Deferral policies for reliability-gated selective prediction.

Each policy implements a gate that decides whether to predict or abstain
based on model output, reliability score, and a threshold.

Coverage target mode: given a desired coverage level, find the threshold
on the validation set that achieves that coverage, then apply to test set.
"""
from abc import ABC, abstractmethod
import numpy as np
from src.reliability.composite import DimensionMatrix


class DeferralPolicy(ABC):
    """Base class for deferral policies."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._threshold: float | None = None

    @abstractmethod
    def _score(self, proba: np.ndarray, r: np.ndarray,
               dims: DimensionMatrix | None = None) -> np.ndarray:
        """Compute per-sample gate score. Higher = more worth predicting."""

    def calibrate_threshold(self, score: np.ndarray,
                            target_coverage: float) -> float:
        """Find threshold achieving target coverage on a validation set."""
        n = len(score)
        n_predict = int(np.round(n * target_coverage))
        sorted_idx = np.argsort(-score)
        if n_predict >= n:
            return -np.inf
        threshold = score[sorted_idx[n_predict - 1]]
        return float(threshold)

    def decide(self, proba: np.ndarray, r: np.ndarray,
               dims: DimensionMatrix | None = None,
               threshold: float | None = None) -> np.ndarray:
        """Return boolean mask: True = predict, False = defer."""
        t = threshold if threshold is not None else self._threshold
        if t is None:
            raise ValueError("Threshold not set. Call calibrate_threshold first.")
        score = self._score(proba, r, dims)
        return score >= t

    def predict_or_abstain(self, proba: np.ndarray, r: np.ndarray,
                            dims: DimensionMatrix | None = None,
                            threshold: float | None = None
                            ) -> tuple[np.ndarray, np.ndarray]:
        """Return (predictions, deferral_mask). Deferred entries = nan."""
        mask = self.decide(proba, r, dims, threshold)
        out = np.where(mask, proba, np.nan)
        return out, ~mask


class ConfidenceGate(DeferralPolicy):
    """Defer when max predicted probability < threshold."""

    def __init__(self) -> None:
        super().__init__("confidence")

    def _score(self, proba: np.ndarray, r: np.ndarray,
               dims: DimensionMatrix | None = None) -> np.ndarray:
        return proba


class ReliabilityGate(DeferralPolicy):
    """Defer when R (multidimensional reliability) < threshold."""

    def __init__(self) -> None:
        super().__init__("reliability")

    def _score(self, proba: np.ndarray, r: np.ndarray,
               dims: DimensionMatrix | None = None) -> np.ndarray:
        return r


class DimensionGate(DeferralPolicy):
    """Defer when a specific reliability dimension < threshold."""

    def __init__(self, dim_idx: int, dim_name: str) -> None:
        super().__init__(f"D{dim_idx + 1}_{dim_name}")
        self.dim_idx = dim_idx

    def _score(self, proba: np.ndarray, r: np.ndarray,
               dims: DimensionMatrix | None = None) -> np.ndarray:
        if dims is None:
            raise ValueError("DimensionGate requires dims.")
        return dims.to_array()[:, self.dim_idx]
