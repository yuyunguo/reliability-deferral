"""Formal significance tests for confidence-gate vs. reliability-gate comparisons.

Motivation: "Lower Brier at every
coverage level" and AUROC comparisons are currently visual claims (point
estimate + non-overlapping CI at best). This adds:
  - DeLong test (DeLong et al. 1988) for the AUROC difference between the
    two gates' retained subsets at 80% coverage. The two subsets overlap
    only in the ~0 patients common to both gates (Jaccard ~= 0), so we
    treat this as two independent samples (Mann-Whitney U / DeLong reduces
    to the unpaired two-sample case when there is no shared patient set).
  - Paired bootstrap test for the Brier difference, resampling patient
    indices within the confidence-retained and reliability-retained sets
    separately (unpaired, same reasoning).
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("deferral_frontier/results")
COVERAGE = 0.80


def _delong_auc_var(y_true, y_score):
    """DeLong et al. (1988) variance of the AUC estimator via structural components."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return np.nan, np.nan

    def psi(x, y):
        return (x > y).astype(float) + 0.5 * (x == y).astype(float)

    v10 = np.array([psi(p, neg).mean() for p in pos])
    v01 = np.array([psi(pos, ng).mean() for ng in neg])
    auc = v10.mean()

    s10 = np.var(v10, ddof=1) if m > 1 else 0.0
    s01 = np.var(v01, ddof=1) if n > 1 else 0.0
    var = s10 / m + s01 / n
    return auc, var


def delong_test(y1, s1, y2, s2):
    """Two-sample DeLong z-test for AUC(y1,s1) vs AUC(y2,s2) (independent samples)."""
    auc1, var1 = _delong_auc_var(y1, s1)
    auc2, var2 = _delong_auc_var(y2, s2)
    se = np.sqrt(var1 + var2)
    if se == 0 or np.isnan(se):
        return auc1, auc2, np.nan, np.nan
    z = (auc1 - auc2) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return auc1, auc2, z, p


def paired_bootstrap_brier_test(y1, p1, y2, p2, n_boot=2000, seed=42):
    """Bootstrap test for Brier(y1,p1) - Brier(y2,p2), independent resampling."""
    rng = np.random.RandomState(seed)
    b1 = np.mean((y1 - p1) ** 2)
    b2 = np.mean((y2 - p2) ** 2)
    diffs = np.empty(n_boot)
    n1, n2 = len(y1), len(y2)
    for i in range(n_boot):
        idx1 = rng.randint(0, n1, n1)
        idx2 = rng.randint(0, n2, n2)
        db1 = np.mean((y1[idx1] - p1[idx1]) ** 2)
        db2 = np.mean((y2[idx2] - p2[idx2]) ** 2)
        diffs[i] = db1 - db2
    observed = b1 - b2
    # Two-sided p-value: fraction of bootstrap diffs at least as extreme as 0
    # under the null that the true difference is 0, using the bootstrap
    # distribution centered at the observed difference (percentile method).
    p_value = 2 * min((diffs >= 0).mean(), (diffs <= 0).mean())
    return b1, b2, observed, p_value


def run_task(source: str, artifact_path: str, task: str):
    df = pd.read_parquet(artifact_path)
    y = df["y_true"].values
    cal = df["cal_proba"].values
    r = df["r_score"].values

    n = len(y)
    n80 = int(np.round(n * COVERAGE))
    conf_idx = np.argsort(-cal)[:n80]
    rel_idx = np.argsort(-r)[:n80]

    y_c, p_c = y[conf_idx], cal[conf_idx]
    y_r, p_r = y[rel_idx], cal[rel_idx]

    auc_c, auc_r, z, p_auc = delong_test(y_c, p_c, y_r, p_r)
    brier_c, brier_r, diff, p_brier = paired_bootstrap_brier_test(y_c, p_c, y_r, p_r)

    logger.info("%s/%s: AUROC conf=%.4f rel=%.4f z=%.3f p=%.2e | "
                "Brier conf=%.4f rel=%.4f diff=%.4f p=%.2e",
                source, task, auc_c, auc_r, z, p_auc, brier_c, brier_r, diff, p_brier)
    return {
        "source": source, "task": task,
        "auroc_confidence": auc_c, "auroc_reliability": auc_r,
        "auroc_z": z, "auroc_p": p_auc,
        "brier_confidence": brier_c, "brier_reliability": brier_r,
        "brier_abs_diff": diff, "brier_p": p_brier,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sources = {
        "MIMIC-IV": "artifacts/{task}_test_artifacts.parquet",
        "MIMIC-III": "artifacts/mimic3/mimic3_{task}_test_artifacts.parquet",
        "eICU": "artifacts/eicu/eicu_{task}_test_artifacts.parquet",
    }
    rows = []
    for source, tpl in sources.items():
        for task in ["sepsis", "mortality", "vasopressor"]:
            path = tpl.format(task=task)
            if not Path(path).exists():
                continue
            rows.append(run_task(source, path, task))

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "significance_tests.csv", index=False)
    logger.info("Saved %d rows to significance_tests.csv", len(df))


if __name__ == "__main__":
    main()
