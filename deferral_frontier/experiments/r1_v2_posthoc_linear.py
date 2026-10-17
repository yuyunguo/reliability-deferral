"""POST HOC (not pre-specified): low-variance error model vs. the symmetric gate.

Motivation: the pre-specified tree ErrorComposite loses to the symmetric gate even
in a positive control with a weak injected signal, so its failure may reflect
estimator variance rather than absence of signal. This refits the same
cross-fitted error target with ridge regression on standardized
[p(1-p), D2', D3, D4, D5'] (and on [p(1-p), D3] alone) and compares retained Brier
at 80% coverage with the symmetric gate (paired bootstrap 95% CI, unadjusted).

Output: deferral_frontier/results/r1_v2_posthoc_linear.csv

Usage:
    python -m deferral_frontier.experiments.r1_v2_posthoc_linear
"""
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from deferral_frontier.experiments.run_symmetric_confidence_gate import (
    COVERAGE, RESULTS_DIR, paired_brier_diff_ci, retained_mask)
from deferral_frontier.src.reliability_v2 import ErrorComposite, patient_groups
from deferral_frontier.src.train_pipeline import train_pipeline
from src.calibration.baselines import IsotonicCalibrator
from src.models.xgboost_model import ensemble_predict

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
SOURCES = {"MIMIC-IV": ("mimic4", "patient"), "MIMIC-III": ("mimic3", "patient"),
           "eICU": ("eicu", "patient"), "eICU-H": ("eicu", "hospital")}
VARIANTS = {"ridge_all": [0, 1, 2, 3], "ridge_D3": [1]}  # columns of [D2', D3, D4, D5'] used


def ridge_error_score(raw_val: np.ndarray, y_val: np.ndarray, other_val: np.ndarray,
                      groups: np.ndarray, p_test: np.ndarray, other_test: np.ndarray,
                      cols: Sequence[int]) -> np.ndarray:
    """Cross-fitted ridge model of (p - y)^2; returns reliability = -predicted error on test."""
    a, b = ErrorComposite._halves(np.asarray(groups))
    X_parts, t_parts = [], []
    for fit_idx, apply_idx in ((a, b), (b, a)):
        p = IsotonicCalibrator().fit(raw_val[fit_idx], y_val[fit_idx]).predict_proba(raw_val[apply_idx])
        X_parts.append(np.column_stack([p * (1 - p), other_val[apply_idx][:, list(cols)]]))
        t_parts.append((p - y_val[apply_idx]) ** 2)
    scaler = StandardScaler().fit(np.vstack(X_parts))
    model = Ridge(alpha=1.0).fit(scaler.transform(np.vstack(X_parts)), np.concatenate(t_parts))
    X_test = np.column_stack([p_test * (1 - p_test), other_test[:, list(cols)]])
    return -model.predict(scaler.transform(X_test))


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.INFO)
    rows = []
    for source, (db, split_by) in SOURCES.items():
        for task in ("sepsis", "mortality", "vasopressor"):
            art = train_pipeline(task, data_dir=ROOT / "data" / db, split_by=split_by)
            raw_val = ensemble_predict(art.ensemble, art.X_val_raw)
            groups = patient_groups(ROOT / "data" / db, art.idx_val)
            dv, dt = art.dims_val, art.dims_test
            other_val = np.column_stack([dv.D2, dv.D3, dv.D4, dv.D5])
            other_test = np.column_stack([dt.D2, dt.D3, dt.D4, dt.D5])
            y, p = art.y_test.astype(int), art.cal_test
            sq = (p - y) ** 2
            keep_sym = retained_mask(np.maximum(p, 1 - p), COVERAGE)
            for name, cols in VARIANTS.items():
                score = ridge_error_score(raw_val, art.y_val.astype(int), other_val, groups, p, other_test, cols)
                keep = retained_mask(score, COVERAGE)
                rng = np.random.default_rng(42)
                lo, hi = paired_brier_diff_ci(y, p, keep_sym, keep, rng)
                diff = sq[keep_sym].mean() - sq[keep].mean()
                rows.append({"source": source, "task": task, "variant": name,
                             "brier_symmetric": sq[keep_sym].mean(), "brier_variant": sq[keep].mean(),
                             "sym_minus_variant": diff, "ci_lo": lo, "ci_hi": hi})
                logger.info("%s %s %s: sym-var %+.4f [%+.4f, %+.4f]", source, task, name, diff, lo, hi)
            pd.DataFrame(rows).to_csv(RESULTS_DIR / "r1_v2_posthoc_linear.csv", index=False)
    out = pd.DataFrame(rows)
    out["excludes_zero"] = (out.ci_lo > 0) | (out.ci_hi < 0)
    out.to_csv(RESULTS_DIR / "r1_v2_posthoc_linear.csv", index=False)


if __name__ == "__main__":
    main()
