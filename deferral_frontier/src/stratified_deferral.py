"""Risk-stratified deferral (plan/r1-stratified-deferral-preregistration.md).

Every score defers the same fraction within each calibrated-risk bin, so all
gates retain the same risk distribution. The effect of a score is the drop in
retained Brier relative to random deferral within the same bins.
"""
from typing import Dict, Tuple

import numpy as np

DEFER_FRAC = 0.2


def tie_ranks(n: int, seed: int) -> np.ndarray:
    """Fixed random rank per position, used to break ties reproducibly."""
    return np.random.default_rng(seed).permutation(n)


def risk_bins(p: np.ndarray, k: int, ties: np.ndarray) -> np.ndarray:
    """Equal-size bins of increasing p (label-free); ties broken by `ties`."""
    order = np.lexsort((ties, p))
    bins = np.empty(len(p), dtype=int)
    bins[order] = np.arange(len(p)) * k // len(p)
    return bins


def _n_defer(n_b: int, frac: float) -> int:
    return int(np.round(frac * n_b))


def stratified_keep(score: np.ndarray, bins: np.ndarray, frac: float, ties: np.ndarray) -> np.ndarray:
    """Retain mask: within each bin, defer the round(frac * n_b) lowest-scoring patients."""
    keep = np.ones(len(score), dtype=bool)
    for b in np.unique(bins):
        idx = np.flatnonzero(bins == b)
        order = idx[np.lexsort((ties[idx], score[idx]))]
        keep[order[: _n_defer(len(idx), frac)]] = False
    return keep


def null_brier(e: np.ndarray, bins: np.ndarray, frac: float) -> float:
    """Exact expected retained Brier under random deferral within bins."""
    num = den = 0.0
    for b in np.unique(bins):
        m = bins == b
        kept = m.sum() - _n_defer(m.sum(), frac)
        num += kept * e[m].mean()
        den += kept
    return num / den


def _delta(score: np.ndarray, p: np.ndarray, e: np.ndarray, k: int, ties: np.ndarray) -> float:
    bins = risk_bins(p, k, ties)
    keep = stratified_keep(score, bins, DEFER_FRAC, ties)
    return null_brier(e, bins, DEFER_FRAC) - e[keep].mean()


def bootstrap_delta(score: np.ndarray, p: np.ndarray, y: np.ndarray, k: int = 10,
                    n_boot: int = 1000, seed: int = 42) -> Dict[str, float]:
    """Delta = null Brier - retained Brier, with paired bootstrap CI and two-sided p.

    Bins and within-bin deferral are recomputed on every resample.
    """
    e = (p - y) ** 2
    n = len(p)
    ties = tie_ranks(n, seed)
    bins = risk_bins(p, k, ties)
    keep = stratified_keep(score, bins, DEFER_FRAC, ties)
    nb = null_brier(e, bins, DEFER_FRAC)
    delta = nb - e[keep].mean()
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = _delta(score[idx], p[idx], e[idx], k, ties[idx])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_two = max(2 * min((boots <= 0).mean(), (boots >= 0).mean()), 1 / n_boot)
    return {"delta": float(delta), "ci_lo": float(lo), "ci_hi": float(hi), "p": float(min(p_two, 1.0)),
            "brier_null": float(nb), "brier_retained": float(e[keep].mean())}


def bootstrap_deltas(scores: Dict[str, np.ndarray], p: np.ndarray, y: np.ndarray, k: int = 10,
                     n_boot: int = 1000, seed: int = 42) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Delta for several scores on the SAME resamples, so any contrast between them is paired.

    Resample indices depend only on (n, seed), so two settings with the same test
    patients in the same order (eICU and X-DB) also share resamples.

    Returns:
        (point delta per score, array of n_boot bootstrap deltas per score).
    """
    e = (p - y) ** 2
    n = len(p)
    ties = tie_ranks(n, seed)
    point = {name: _delta(s, p, e, k, ties) for name, s in scores.items()}
    boots = {name: np.empty(n_boot) for name in scores}
    rng = np.random.default_rng(seed)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        p_i, e_i, t_i = p[idx], e[idx], ties[idx]
        bins = risk_bins(p_i, k, t_i)
        nb = null_brier(e_i, bins, DEFER_FRAC)
        for name, s in scores.items():
            boots[name][i] = nb - e_i[stratified_keep(s[idx], bins, DEFER_FRAC, t_i)].mean()
    return point, boots


def summarize(point: float, boots: np.ndarray) -> Dict[str, float]:
    """Percentile 95% CI and two-sided bootstrap p (floored at 1/n_boot)."""
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_two = max(2 * min((boots <= 0).mean(), (boots >= 0).mean()), 1 / len(boots))
    return {"delta": float(point), "ci_lo": float(lo), "ci_hi": float(hi), "p": float(min(p_two, 1.0))}
