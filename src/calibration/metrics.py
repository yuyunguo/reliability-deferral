"""Calibration metrics: M-ECE (adaptive equal-mass bins) and AUROC with CIs."""
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    mece: float
    mece_ci_low: float
    mece_ci_high: float
    auroc: float
    auroc_ci_low: float
    auroc_ci_high: float
    nll: float
    n_samples: int
    prevalence: float


def adaptive_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Adaptive ECE with equal-mass bins (M-ECE).

    Bins are defined by quantiles of predicted probability, ensuring equal
    sample count per bin regardless of probability distribution skew.
    This is robust to class imbalance — critical for clinical tasks.

    Args:
        probs: Predicted probabilities ∈ [0, 1], shape (n,).
        labels: Binary outcomes, shape (n,).
        n_bins: Number of equal-mass bins (pre-committed: 10).

    Returns:
        Scalar M-ECE ∈ [0, 1].
    """
    bin_edges = np.quantile(probs, np.linspace(0, 1, n_bins + 1))
    bin_edges[0] -= 1e-8
    bin_edges[-1] += 1e-8

    ece = 0.0
    n = len(probs)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs > lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        bin_probs = probs[mask]
        bin_labels = labels[mask]
        ece += (mask.sum() / n) * abs(bin_probs.mean() - bin_labels.mean())
    return float(ece)


def bootstrap_calibration(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 1000,
    n_bins: int = 10,
    alpha: float = 0.05,
    seed: int = 42,
) -> CalibrationResult:
    """Compute M-ECE and AUROC with 95% bootstrapped CIs.

    Args:
        probs: Predicted probabilities.
        labels: Binary outcomes.
        n_bootstrap: Bootstrap iterations (pre-committed: 1000).
        n_bins: ECE bins.
        alpha: CI level (0.05 → 95% CI).
        seed: Random seed for bootstrap.

    Returns:
        CalibrationResult with point estimates and CIs.
    """
    rng = np.random.default_rng(seed)
    n = len(probs)

    mece_samples = np.empty(n_bootstrap)
    auroc_samples = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        p_b, y_b = probs[idx], labels[idx]
        mece_samples[i] = adaptive_ece(p_b, y_b, n_bins)
        try:
            auroc_samples[i] = roc_auc_score(y_b, p_b)
        except ValueError:
            auroc_samples[i] = float("nan")

    mece_samples = mece_samples[~np.isnan(mece_samples)]
    auroc_samples = auroc_samples[~np.isnan(auroc_samples)]

    q_lo, q_hi = alpha / 2, 1 - alpha / 2
    nll = float(-np.mean(
        labels * np.log(probs + 1e-12) + (1 - labels) * np.log(1 - probs + 1e-12)
    ))

    return CalibrationResult(
        mece=adaptive_ece(probs, labels, n_bins),
        mece_ci_low=float(np.quantile(mece_samples, q_lo)),
        mece_ci_high=float(np.quantile(mece_samples, q_hi)),
        auroc=float(roc_auc_score(labels, probs)),
        auroc_ci_low=float(np.quantile(auroc_samples, q_lo)),
        auroc_ci_high=float(np.quantile(auroc_samples, q_hi)),
        nll=nll,
        n_samples=n,
        prevalence=float(labels.mean()),
    )


def bootstrap_delta_ece(
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 1000,
    n_bins: int = 10,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Paired bootstrap CI on ΔECE = ECE_a - ECE_b.

    A result is meaningful when:
    - |ΔECE| ≥ 0.01 (pre-committed minimum meaningful effect)
    - CI excludes zero
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    delta_samples = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b = labels[idx]
        delta_samples[i] = (
            adaptive_ece(probs_a[idx], y_b, n_bins)
            - adaptive_ece(probs_b[idx], y_b, n_bins)
        )

    q_lo, q_hi = alpha / 2, 1 - alpha / 2
    point_delta = adaptive_ece(probs_a, labels, n_bins) - adaptive_ece(probs_b, labels, n_bins)
    ci_lo = float(np.quantile(delta_samples, q_lo))
    ci_hi = float(np.quantile(delta_samples, q_hi))
    significant = (ci_lo > 0) or (ci_hi < 0)
    meaningful = abs(point_delta) >= 0.01

    return {
        "delta_ece": float(point_delta),
        "ci_low": ci_lo,
        "ci_high": ci_hi,
        "significant": significant,
        "meaningful": meaningful,
        "conclusion": "MEANINGFUL" if (significant and meaningful) else "NOT MEANINGFUL",
    }
