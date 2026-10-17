"""M2 robustness check: is the R-Brier relationship circular by construction?

A natural objection is that R is trained to
predict calibration error (ECE > median), and the headline result -- that
deferring low-R cases improves Brier score -- is close to definitionally
true, since Brier's reliability component IS calibration error.

Two independent checks, using already-saved val/test artifacts (no
retraining of the base XGBoost ensembles needed -- D1-D5 are already
computed):

  A. ORIGINAL: reproduce the paper's exact R (target = |D1-y| > median) as
     a sanity check that this script's reimplementation is faithful.
  B. OUTCOME-BASED: refit R against raw misclassification (round(p) != y)
     instead of calibration error -- a genuinely different, non-circular
     construct. If deferral benefit survives, D1-D5 carry real information
     about prediction quality, not just an artifact of the ECE target.
  C. PERMUTED-NULL: refit R on a randomly shuffled version of the ORIGINAL
     target. If deferral benefit collapses to ~0, the real R's benefit is
     not a mechanical artifact of the composite-then-defer pipeline.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from deferral_frontier.src.metrics import bootstrap_ci_brier, number_needed_to_review

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("artifacts")
RESULTS_DIR = Path("deferral_frontier/results")
DIM_COLS = ["D1", "D2", "D3", "D4", "D5"]
COVERAGE = 0.80


def fit_r(X_val, target_val, X_test):
    scaler = StandardScaler().fit(X_val)
    lr = LogisticRegression(max_iter=1000, random_state=42).fit(scaler.transform(X_val), target_val)
    # Probability of LOW target (class 0) = higher reliability, matching
    # ReliabilityComposite.transform's convention.
    r_test = lr.predict_proba(scaler.transform(X_test))[:, 0]
    return r_test


def evaluate_deferral(y_test, cal_test, r_test, coverage=COVERAGE):
    n = int(np.round(len(y_test) * coverage))
    idx = np.argsort(-r_test)[:n]
    y_sel, p_sel = y_test[idx], cal_test[idx]
    brier, lo, hi = bootstrap_ci_brier(y_sel, p_sel, n_bootstrap=1000)

    deferred = np.zeros(len(y_test), dtype=bool)
    deferred[np.argsort(-r_test)[n:]] = True
    nnr, _, _ = number_needed_to_review(y_test, cal_test, deferred)

    return brier, lo, hi, nnr


def run_task(task: str):
    val = pd.read_parquet(ARTIFACTS_DIR / f"{task}_val_artifacts.parquet")
    test = pd.read_parquet(ARTIFACTS_DIR / f"{task}_test_artifacts.parquet")

    X_val = val[DIM_COLS].values
    X_test = test[DIM_COLS].values
    y_val = val["y_true"].values
    y_test = test["y_true"].values
    cal_val = val["cal_proba"].values
    cal_test = test["cal_proba"].values

    rows = []

    # --- A. Original (reproduces paper's R) ---
    cal_err_val = np.abs(val["D1"].values - y_val)
    target_a = (cal_err_val > np.median(cal_err_val)).astype(int)
    r_test_a = fit_r(X_val, target_a, X_test)
    b, lo, hi, nnr = evaluate_deferral(y_test, cal_test, r_test_a)
    rows.append({"task": task, "variant": "A_original_ECE", "brier": b,
                 "brier_ci_lo": lo, "brier_ci_hi": hi, "nnr": nnr})

    # --- B. Outcome-based (raw misclassification target) ---
    target_b = (np.round(cal_val) != y_val).astype(int)
    if target_b.sum() == 0 or target_b.sum() == len(target_b):
        logger.warning("  %s: outcome-based target has no class variation, skipping", task)
    else:
        r_test_b = fit_r(X_val, target_b, X_test)
        b, lo, hi, nnr = evaluate_deferral(y_test, cal_test, r_test_b)
        rows.append({"task": task, "variant": "B_outcome_based", "brier": b,
                     "brier_ci_lo": lo, "brier_ci_hi": hi, "nnr": nnr})

    # --- C. Permuted null (shuffled original target) ---
    rng = np.random.RandomState(42)
    target_c = rng.permutation(target_a)
    r_test_c = fit_r(X_val, target_c, X_test)
    b, lo, hi, nnr = evaluate_deferral(y_test, cal_test, r_test_c)
    rows.append({"task": task, "variant": "C_permuted_null", "brier": b,
                 "brier_ci_lo": lo, "brier_ci_hi": hi, "nnr": nnr})

    # --- Reference: confidence gate (no R at all) ---
    b, lo, hi, nnr = evaluate_deferral(y_test, cal_test, cal_test)
    rows.append({"task": task, "variant": "Z_confidence_gate", "brier": b,
                 "brier_ci_lo": lo, "brier_ci_hi": hi, "nnr": nnr})

    return rows


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for task in ["sepsis", "mortality", "vasopressor"]:
        logger.info("Task: %s", task)
        rows = run_task(task)
        for r in rows:
            logger.info("  %-20s Brier=%.4f [%.4f,%.4f]  NNR=%.1f",
                        r["variant"], r["brier"], r["brier_ci_lo"], r["brier_ci_hi"], r["nnr"])
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_DIR / "m2_circularity_check.csv", index=False)
    print("\n" + "=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()
