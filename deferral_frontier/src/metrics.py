"""Evaluation metrics for selective prediction.

All functions operate on a test set where some predictions may be deferred
(masked as NaN). We compute metrics only on the predicted subset.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.stats import bootstrap


def risk_coverage_curve(
    y_true: np.ndarray, y_prob: np.ndarray,
    gate_score: np.ndarray,
    coverage_targets: np.ndarray,
) -> dict:
    """Compute risk at each coverage target using a gate score.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities.
        gate_score: Gate score (higher = more confident).
        coverage_targets: Array of target coverages (e.g., np.arange(0.5, 1.05, 0.05)).

    Returns:
        dict with 'coverage', 'brier', 'auroc', 'ece', 'deferral_mask' per target.
    """
    results = []
    n = len(y_true)
    for target_cov in sorted(coverage_targets, reverse=True):
        n_predict = int(np.round(n * target_cov))
        if n_predict >= n:
            mask = np.zeros(n, dtype=bool)
        else:
            threshold = float(np.sort(-gate_score)[n_predict - 1])
            mask = gate_score < -threshold
        pred_idx = ~mask
        if pred_idx.sum() == 0:
            results.append({'coverage': 0.0, 'brier': np.nan,
                            'auroc': np.nan, 'ece': np.nan,
                            'n_predicted': 0})
            continue
        y_p = y_prob[pred_idx]
        y_t = y_true[pred_idx]
        brier = float(brier_score_loss(y_t, y_p))
        ece = _calc_ece(y_t, y_p, n_bins=10)
        try:
            auroc = float(roc_auc_score(y_t, y_p))
        except ValueError:
            auroc = np.nan
        results.append({'coverage': float(target_cov), 'brier': brier,
                        'auroc': auroc, 'ece': ece,
                        'n_predicted': int(pred_idx.sum())})
    return results


def effective_coverage(
    y_true: np.ndarray, y_prob: np.ndarray,
    gate_score: np.ndarray,
    max_acceptable_risk: float = 0.05,
) -> float:
    """Coverage at the point where risk first drops below max_acceptable_risk."""
    targets = np.linspace(0.1, 1.0, 91)
    curve = risk_coverage_curve(y_true, y_prob, gate_score, targets)
    for pt in curve:
        if pt['brier'] is not None and pt['brier'] <= max_acceptable_risk:
            return pt['coverage']
    return 0.0


def deferral_precision(y_true: np.ndarray, y_prob: np.ndarray,
                        deferral_mask: np.ndarray) -> float:
    """Of deferred cases, fraction that would have been errors.

    An error is defined as: predicted probability >= 0.5 but label != 1,
    or predicted probability < 0.5 but label == 1.
    """
    deferred_idx = deferral_mask
    if deferred_idx.sum() == 0:
        return np.nan
    preds = (y_prob[deferred_idx] >= 0.5).astype(int)
    errors = (preds != y_true[deferred_idx]).astype(float)
    return float(errors.mean())


def auroc_at_coverage(y_true: np.ndarray, y_prob: np.ndarray,
                       gate_score: np.ndarray,
                       target_coverage: float) -> float:
    """AUROC on the top-k predictions by gate score at target coverage."""
    n = len(y_true)
    n_predict = int(np.round(n * target_coverage))
    if n_predict >= n:
        return float(roc_auc_score(y_true, y_prob))
    if n_predict < 2:
        return np.nan
    idx = np.argsort(-gate_score)[:n_predict]
    try:
        return float(roc_auc_score(y_true[idx], y_prob[idx]))
    except ValueError:
        return np.nan


def _calc_ece(y_true: np.ndarray, y_prob: np.ndarray,
              n_bins: int = 10) -> float:
    """Expected calibration error with equal-mass bins."""
    idx = np.argsort(y_prob)
    y_true_s = y_true[idx]
    y_prob_s = y_prob[idx]
    bin_size = len(y_true_s) // n_bins
    ece = 0.0
    for i in range(n_bins):
        start = i * bin_size
        if i == n_bins - 1:
            end = len(y_true_s)
        else:
            end = (i + 1) * bin_size
        if start >= end:
            continue
        bin_acc = y_true_s[start:end].mean()
        bin_conf = y_prob_s[start:end].mean()
        bin_weight = (end - start) / len(y_true_s)
        ece += bin_weight * abs(bin_acc - bin_conf)
    return float(ece)


def bootstrap_ci_brier(
    y_true: np.ndarray, y_prob: np.ndarray,
    n_bootstrap: int = 1000, ci: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for Brier score.

    Returns (brier, ci_low, ci_high).
    """
    rng = np.random.default_rng(42)
    n = len(y_true)
    brier_obs = float(brier_score_loss(y_true, y_prob))
    boot_vals = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_vals[i] = brier_score_loss(y_true[idx], y_prob[idx])
    alpha = (1.0 - ci) / 2.0
    ci_low = float(np.percentile(boot_vals, alpha * 100))
    ci_high = float(np.percentile(boot_vals, (1.0 - alpha) * 100))
    return brier_obs, ci_low, ci_high


def net_benefit(
    y_true: np.ndarray, y_prob: np.ndarray,
    threshold: float,
) -> float:
    """Net benefit at a given clinical decision threshold.

    Net benefit = (TP / N) - (FP / N) * (threshold / (1 - threshold))
    """
    pred = (y_prob >= threshold).astype(int)
    tp = ((pred == 1) & (y_true == 1)).sum()
    fp = ((pred == 1) & (y_true == 0)).sum()
    n = len(y_true)
    nb = tp / n - (fp / n) * (threshold / (1 - threshold + 1e-12))
    return float(nb)


def decision_curve(
    y_true: np.ndarray, y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict:
    """Compute net benefit across a range of clinical thresholds.

    Also computes 'treat all' and 'treat none' baselines.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 50)
    prevalence = y_true.mean()
    nb_all = prevalence - (1 - prevalence) * thresholds / (1 - thresholds + 1e-12)
    rows = []
    for t in thresholds:
        nb = net_benefit(y_true, y_prob, t)
        rows.append({'threshold': t, 'net_benefit': nb,
                     'nb_treat_all': float(nb_all[thresholds == t][0]) if t in thresholds else 0.0})
    return {'curves': rows, 'prevalence': prevalence}


def number_needed_to_review(
    y_true: np.ndarray, y_prob: np.ndarray,
    deferral_mask: np.ndarray,
) -> tuple[float, float, float]:
    """Number of reviewed cases to prevent one error.

    Also returns total deferrals and error rate among deferred.
    """
    n_deferred = int(deferral_mask.sum())
    if n_deferred == 0:
        return float('inf'), 0, 0.0
    preds = (y_prob[deferral_mask] >= 0.5).astype(int)
    errors = (preds != y_true[deferral_mask]).sum()
    error_rate = errors / n_deferred
    nnr = n_deferred / max(errors, 1)
    return float(nnr), n_deferred, float(error_rate)


def false_positive_burden(
    y_true: np.ndarray, y_prob: np.ndarray,
    deferral_mask: np.ndarray,
) -> float:
    """False positive alerts per patient among non-deferred predictions."""
    pred_idx = ~deferral_mask
    if pred_idx.sum() == 0:
        return 0.0
    preds = (y_prob[pred_idx] >= 0.5).astype(int)
    fps = ((preds == 1) & (y_true[pred_idx] == 0)).sum()
    return float(fps / len(y_true))


def bootstrap_ci_auroc(
    y_true: np.ndarray, y_prob: np.ndarray,
    n_bootstrap: int = 1000, ci: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for AUROC."""
    rng = np.random.default_rng(42)
    n = len(y_true)
    try:
        auroc_obs = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return np.nan, np.nan, np.nan
    boot_vals = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            boot_vals[i] = roc_auc_score(y_true[idx], y_prob[idx])
        except ValueError:
            boot_vals[i] = np.nan
    valid = boot_vals[~np.isnan(boot_vals)]
    if len(valid) < n_bootstrap // 2:
        return auroc_obs, np.nan, np.nan
    alpha = (1.0 - ci) / 2.0
    ci_low = float(np.percentile(valid, alpha * 100))
    ci_high = float(np.percentile(valid, (1.0 - alpha) * 100))
    return auroc_obs, ci_low, ci_high
