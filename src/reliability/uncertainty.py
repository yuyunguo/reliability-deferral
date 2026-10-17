"""D5: Uncertainty consistency via MC Dropout."""
import logging
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


def compute_uncertainty_consistency(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    calibrated_probs: np.ndarray,
    n_passes: int = 50,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """D5: 1 - normalized MC Dropout variance.

    Measures robustness of predictions under stochastic dropout perturbation.
    D5 near 1 = stable under perturbation. D5 near 0 = high sensitivity.

    Args:
        predict_fn: Callable that returns dropout-perturbed probabilities.
                    Must have dropout active (training mode or equivalent).
        X: Feature matrix, shape (n_samples, n_features).
        calibrated_probs: Calibrated confidence p̃ ∈ (0,1), shape (n_samples,).
                          Used as the Bernoulli variance normalizer.
        n_passes: Number of stochastic forward passes (pre-committed: 50).
        epsilon: Small constant to avoid division by zero.

    Returns:
        D5 ∈ [0, 1], shape (n_samples,).
    """
    passes = np.stack([predict_fn(X) for _ in range(n_passes)], axis=0)
    mc_var = np.var(passes, axis=0)

    # Bernoulli normalizer: p̃(1-p̃) is the maximum variance for a Bernoulli
    bernoulli_var = calibrated_probs * (1.0 - calibrated_probs) + epsilon
    normalized_var = mc_var / bernoulli_var

    d5 = 1.0 - np.clip(normalized_var, 0.0, 1.0)
    logger.debug(
        "D5 stats: mean=%.4f, std=%.4f, mc_var mean=%.6f",
        d5.mean(), d5.std(), mc_var.mean(),
    )
    return d5


def make_xgboost_dropout_fn(
    model,
    dropout_rate: float = 0.1,
    rng: np.random.Generator = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a dropout perturbation function for XGBoost.

    XGBoost has no native dropout. We approximate uncertainty consistency
    by randomly masking features (input dropout) at inference time.
    This tests sensitivity of predictions to input perturbations —
    consistent with the operational reliability framing.

    Args:
        model: Fitted XGBoost classifier.
        dropout_rate: Proportion of features to zero out per pass.
        rng: Random number generator for reproducibility.

    Returns:
        Callable that accepts X and returns perturbed probabilities.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    def _predict(X: np.ndarray) -> np.ndarray:
        mask = rng.random(X.shape) > dropout_rate
        X_perturbed = X * mask
        return model.predict_proba(X_perturbed)[:, 1]

    return _predict
