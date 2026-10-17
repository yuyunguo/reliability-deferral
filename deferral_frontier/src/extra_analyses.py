"""Helpers for the post hoc extra analyses (plan/r1-extra-analyses-plan.md)."""
from typing import Dict, List, Sequence

import numpy as np


def harmonized_drop(columns: Sequence[str]) -> List[str]:
    """Features whose meaning differs across databases: observation counts and temperature."""
    return [c for c in columns if c.endswith("_count") or "_temp_" in c]


def decision_loss(p: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    """Per-patient loss of deciding at threshold t: FP costs t, FN costs 1 - t."""
    pred = p >= t
    return np.where(pred & (y == 0), t, 0.0) + np.where(~pred & (y == 1), 1.0 - t, 0.0)


def t_centered_score(p: np.ndarray, t: float) -> np.ndarray:
    """Reliability = -expected decision loss under calibration, -min(t(1-p), (1-t)p)."""
    return -np.minimum(t * (1.0 - p), (1.0 - t) * p)


def random_effects(est: np.ndarray, se: np.ndarray) -> Dict[str, float]:
    """DerSimonian-Laird random-effects pooling with I^2."""
    w = 1.0 / se ** 2
    fixed = np.sum(w * est) / np.sum(w)
    q = float(np.sum(w * (est - fixed) ** 2))
    df = len(est) - 1
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (q - df) / c) if df > 0 else 0.0
    wr = 1.0 / (se ** 2 + tau2)
    pooled = float(np.sum(wr * est) / np.sum(wr))
    se_p = float(np.sqrt(1.0 / np.sum(wr)))
    i2 = max(0.0, (q - df) / q) if q > 0 else 0.0
    return {"pooled": pooled, "ci_lo": pooled - 1.96 * se_p, "ci_hi": pooled + 1.96 * se_p,
            "tau2": float(tau2), "i2": float(i2), "q": q, "k": int(len(est))}
