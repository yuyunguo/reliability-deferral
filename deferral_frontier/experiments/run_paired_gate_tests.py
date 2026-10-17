"""Paired patient-level bootstrap tests for gate comparisons at 80% coverage.

Replaces run_significance_tests.py, which treated the two gates' retained sets
as independent samples (two-sample DeLong test and unpaired bootstrap). At 80%
coverage the retained sets share most of their patients (the *deferred* sets
are disjoint, the *retained* sets are not), so independence does not hold.

Here each gate's retained mask is fixed on the full test set, exactly as in
run_symmetric_confidence_gate.py. Each bootstrap replicate resamples test
patients with replacement and recomputes retained-set Brier score and AUROC for
both gates on the same resampled patients, which preserves their correlation.
Two-sided p-values come from the bootstrap distribution of the difference, and
Holm-Bonferroni correction is applied across the nine primary dataset-task pairs
(the eICU held-out-hospital analysis, eICU-H, is a separate family)
separately for each comparison and metric.

Output:
    results/paired_gate_tests.csv
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ARTIFACT_DIR = Path("artifacts")
RESULTS_DIR = Path("deferral_frontier/results")
COVERAGE = 0.80
N_BOOT = 2000
SEED = 42
TASKS = ("sepsis", "mortality", "vasopressor")
SOURCES = {
    "MIMIC-IV": "{task}_test_artifacts.parquet",
    "MIMIC-III": "mimic3/mimic3_{task}_test_artifacts.parquet",
    "eICU": "eicu/eicu_{task}_test_artifacts.parquet",
    "eICU-H": "eicu_hospital/eicu_hospital_{task}_test_artifacts.parquet",  # held-out hospitals (round 1)
}
COMPARISONS = (("one_sided_confidence", "reliability"), ("symmetric_confidence", "reliability"))


def retained_mask(score: np.ndarray, coverage: float) -> np.ndarray:
    """Retain the top `coverage` fraction by score (stable order breaks ties)."""
    keep = np.zeros(len(score), dtype=bool)
    keep[np.argsort(-score, kind="stable")[: int(np.round(len(score) * coverage))]] = True
    return keep


def _metrics(y: np.ndarray, p: np.ndarray, keep: np.ndarray) -> tuple[float, float]:
    """Retained-set Brier score and AUROC."""
    yk, pk = y[keep], p[keep]
    return float(np.mean((pk - yk) ** 2)), float(roc_auc_score(yk, pk))


def _bootstrap_p(diffs: np.ndarray) -> float:
    """Two-sided bootstrap p-value for H0: difference = 0 (floored at 1/N_BOOT)."""
    p = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return float(min(1.0, max(p, 1.0 / len(diffs))))


def paired_test(y: np.ndarray, p: np.ndarray, keep_a: np.ndarray, keep_b: np.ndarray,
                rng: np.random.Generator) -> dict:
    """Paired bootstrap of Brier and AUROC differences (A minus B)."""
    n = len(y)
    brier_a, auc_a = _metrics(y, p, keep_a)
    brier_b, auc_b = _metrics(y, p, keep_b)
    d_brier = np.empty(N_BOOT)
    d_auc = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yi, pi = y[idx], p[idx]
        ba, aa = _metrics(yi, pi, keep_a[idx])
        bb, ab = _metrics(yi, pi, keep_b[idx])
        d_brier[i], d_auc[i] = ba - bb, aa - ab
    return {
        "brier_a": brier_a, "brier_b": brier_b, "brier_diff": brier_a - brier_b,
        "brier_ci_lo": float(np.percentile(d_brier, 2.5)),
        "brier_ci_hi": float(np.percentile(d_brier, 97.5)),
        "brier_p": _bootstrap_p(d_brier),
        "auroc_a": auc_a, "auroc_b": auc_b, "auroc_diff": auc_a - auc_b,
        "auroc_ci_lo": float(np.percentile(d_auc, 2.5)),
        "auroc_ci_hi": float(np.percentile(d_auc, 97.5)),
        "auroc_p": _bootstrap_p(d_auc),
    }


def holm(pvals: pd.Series) -> pd.Series:
    """Holm-Bonferroni adjusted p-values."""
    order = np.argsort(pvals.to_numpy())
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, j in enumerate(order):
        running = max(running, (m - rank) * pvals.iloc[j])
        adj[j] = min(1.0, running)
    return pd.Series(adj, index=pvals.index)


def main() -> None:
    """Run both paired comparisons on every available artifact file."""
    rng = np.random.default_rng(SEED)
    rows = []
    for source, template in SOURCES.items():
        for task in TASKS:
            path = ARTIFACT_DIR / template.format(task=task)
            try:
                df = pd.read_parquet(path)
            except FileNotFoundError:
                logger.warning("Missing artifact: %s", path)
                continue
            y = df["y_true"].to_numpy().astype(int)
            p = df["cal_proba"].to_numpy()
            scores = {
                "one_sided_confidence": p,
                "symmetric_confidence": np.maximum(p, 1 - p),
                "reliability": df["r_score"].to_numpy(),
            }
            keep = {g: retained_mask(s, COVERAGE) for g, s in scores.items()}
            for gate_a, gate_b in COMPARISONS:
                res = paired_test(y, p, keep[gate_a], keep[gate_b], rng)
                rows.append({"source": source, "task": task,
                             "gate_a": gate_a, "gate_b": gate_b, **res})
                logger.info("%s %s %s vs %s: dAUROC=%.4f p=%.4f | dBrier=%.4f p=%.4f",
                            source, task, gate_a, gate_b, res["auroc_diff"], res["auroc_p"],
                            res["brier_diff"], res["brier_p"])
    out = pd.DataFrame(rows)
    for metric in ("brier", "auroc"):
        out[f"{metric}_p_holm"] = out.groupby(["gate_a", out["source"].eq("eICU-H")])[f"{metric}_p"].transform(holm)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS_DIR / "paired_gate_tests.csv", index=False)
    logger.info("Saved %d rows to paired_gate_tests.csv", len(out))


if __name__ == "__main__":
    main()
