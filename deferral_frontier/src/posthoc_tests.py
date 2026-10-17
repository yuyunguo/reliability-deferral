"""Post hoc robustness tests for the risk-matched analysis (labeled post hoc in the manuscript).

- cluster_bootstrap_deltas: the risk-matched Delta with whole clusters (hospitals) resampled.
- lrt_beyond_risk: likelihood-ratio test of whether a score carries outcome information
  beyond a flexible function of predicted risk, using all patients (no binning).
- titrated_control: detection rate of injected error signals of decreasing strength,
  to estimate the smallest risk-matched gain the test can detect.
"""
import logging
from typing import Dict, Tuple

import numpy as np
import statsmodels.api as sm
from patsy import dmatrix

from deferral_frontier.src.stratified_deferral import (
    DEFER_FRAC, bootstrap_deltas, null_brier, risk_bins, stratified_keep, summarize, tie_ranks)

logger = logging.getLogger(__name__)


def cluster_bootstrap_deltas(scores: Dict[str, np.ndarray], p: np.ndarray, y: np.ndarray,
                             clusters: np.ndarray, k: int = 10, n_boot: int = 1000,
                             seed: int = 42) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Risk-matched Delta per score; bootstrap resamples whole clusters with replacement."""
    e = (p - y) ** 2
    n = len(p)
    ties = tie_ranks(n, seed)
    point, _ = bootstrap_deltas(scores, p, y, k=k, n_boot=1, seed=seed)
    uniq, inv = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inv == c) for c in range(len(uniq))]
    boots = {name: np.empty(n_boot) for name in scores}
    rng = np.random.default_rng(seed)
    for i in range(n_boot):
        idx = np.concatenate([members[c] for c in rng.integers(0, len(uniq), len(uniq))])
        p_i, e_i, t_i = p[idx], e[idx], ties[idx]
        bins = risk_bins(p_i, k, t_i)
        nb = null_brier(e_i, bins, DEFER_FRAC)
        for name, s in scores.items():
            boots[name][i] = nb - e_i[stratified_keep(s[idx], bins, DEFER_FRAC, t_i)].mean()
    return point, boots


def lrt_beyond_risk(y: np.ndarray, p: np.ndarray, s: np.ndarray, df: int = 5) -> Tuple[float, float]:
    """LR test: logit P(y) ~ cr(logit p, df) [+ standardized s]. Returns (statistic, p-value)."""
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    base = np.asarray(dmatrix(f"cr(x, df={df}) - 1", {"x": lp}, return_type="dataframe"))  # cr basis spans the constant
    z = (s - s.mean()) / (s.std() + 1e-12)
    m0 = sm.Logit(y, base).fit(disp=0, method="lbfgs", maxiter=500)
    m1 = sm.Logit(y, np.column_stack([base, z])).fit(disp=0, method="lbfgs", maxiter=500)
    stat = max(2.0 * (m1.llf - m0.llf), 0.0)
    from scipy.stats import chi2
    return float(stat), float(chi2.sf(stat, 1))


def titrated_control(p: np.ndarray, y: np.ndarray, k: float, n_rep: int = 20, n_boot: int = 200,
                     seed: int = 42) -> Dict[str, float]:
    """Inject -(e + N(0, k*SD(e))) n_rep times; report mean Delta_adj and the share detected (CI > 0)."""
    e = (p - y) ** 2
    sym = np.maximum(p, 1 - p)
    deltas, detected = [], []
    for r in range(n_rep):
        ctl = -(e + np.random.default_rng(seed + r).normal(0, k * e.std(), len(e)))
        point, boots = bootstrap_deltas({"symmetric": sym, "ctl": ctl}, p, y, k=10, n_boot=n_boot, seed=seed + r)
        sm_ = summarize(point["ctl"] - point["symmetric"], boots["ctl"] - boots["symmetric"])
        deltas.append(sm_["delta"])
        detected.append(sm_["ci_lo"] > 0)
    return {"k": k, "mean_delta_adj": float(np.mean(deltas)), "detect_rate": float(np.mean(detected))}
