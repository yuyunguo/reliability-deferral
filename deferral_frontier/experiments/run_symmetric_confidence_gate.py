"""Symmetric confidence gate vs. one-sided confidence gate vs. reliability gate.

The confidence gate used throughout the main analysis ranks patients by the
calibrated positive-class probability p (defer when p < t), i.e. it defers the
lowest-risk patients. The textbook selective-prediction rule instead ranks by
max(p, 1 - p). For a calibrated predictor E[(p - y)^2 | p] = p(1 - p), so the
symmetric rule is also the rule that minimises expected retained-set Brier score
under calibration, while the one-sided rule removes the patients with the
smallest expected Brier contribution.

This script evaluates all three gates at matched 80% coverage (stable top-k, so
isotonic ties cannot change the number retained) on the saved per-patient test
artifacts for MIMIC-IV, MIMIC-III, and eICU. No model is retrained.

Outputs:
    results/symmetric_confidence_gate.csv
    results/symmetric_gate_net_benefit.csv
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ARTIFACT_DIR = Path("artifacts")
RESULTS_DIR = Path("deferral_frontier/results")
COVERAGE = 0.80
N_BOOT = 1000
SEED = 42
TASKS = ("sepsis", "mortality", "vasopressor")
SOURCES = {
    "MIMIC-IV": "{task}_test_artifacts.parquet",
    "MIMIC-III": "mimic3/mimic3_{task}_test_artifacts.parquet",
    "eICU": "eicu/eicu_{task}_test_artifacts.parquet",
    "eICU-H": "eicu_hospital/eicu_hospital_{task}_test_artifacts.parquet",  # held-out hospitals (round 1)
}
NB_THRESHOLDS = np.round(np.arange(0.02, 0.51, 0.02), 2)


def retained_mask(score: np.ndarray, coverage: float) -> np.ndarray:
    """Retain the top `coverage` fraction by score (stable order breaks ties)."""
    n = len(score)
    keep = np.zeros(n, dtype=bool)
    keep[np.argsort(-score, kind="stable")[: int(np.round(n * coverage))]] = True
    return keep


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard index between two boolean masks."""
    union = np.sum(a | b)
    return float(np.sum(a & b) / union) if union else float("nan")


def nnr(y: np.ndarray, p: np.ndarray, deferred: np.ndarray) -> tuple[float, float]:
    """Number needed to review and deferred error rate (error = misclassified at 0.5)."""
    errors = int(np.sum((p[deferred] >= 0.5).astype(int) != y[deferred]))
    n_def = int(deferred.sum())
    return n_def / max(errors, 1), errors / n_def


def net_benefit(y: np.ndarray, p: np.ndarray, pt: float) -> float:
    """Net benefit of acting when p >= pt, normalised by the retained population."""
    act = p >= pt
    n = len(y)
    return float(np.sum(act & (y == 1)) / n - np.sum(act & (y == 0)) / n * pt / (1 - pt))


def paired_brier_diff_ci(y: np.ndarray, p: np.ndarray, keep_a: np.ndarray,
                         keep_b: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """95% bootstrap CI for Brier(A) - Brier(B), resampling test patients."""
    n = len(y)
    sq = (p - y) ** 2
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        diffs[i] = sq[idx][keep_a[idx]].mean() - sq[idx][keep_b[idx]].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def evaluate(source: str, task: str, df: pd.DataFrame,
             rng: np.random.Generator) -> tuple[list[dict], list[dict]]:
    """Evaluate the three gates for one dataset-task pair."""
    y = df["y_true"].to_numpy().astype(int)
    p = df["cal_proba"].to_numpy()
    gates = {
        "one_sided_confidence": p,
        "symmetric_confidence": np.maximum(p, 1 - p),
        "reliability": df["r_score"].to_numpy(),
    }
    keep = {g: retained_mask(s, COVERAGE) for g, s in gates.items()}
    rows, nb_rows = [], []
    for gate, k in keep.items():
        nnr_val, err = nnr(y, p, ~k)
        rows.append({
            "source": source, "task": task, "gate": gate,
            "brier": brier_score_loss(y[k], p[k]),
            "auroc": roc_auc_score(y[k], p[k]),
            "retained_prevalence": y[k].mean(),
            "retained_mean_risk": p[k].mean(),
            "deferred_prevalence": y[~k].mean(),
            "deferred_error_rate": err, "nnr": nnr_val,
            "jaccard_deferred_vs_reliability": jaccard(~k, ~keep["reliability"]),
            "jaccard_deferred_vs_one_sided": jaccard(~k, ~keep["one_sided_confidence"]),
        })
        for pt in NB_THRESHOLDS:
            nb_rows.append({"source": source, "task": task, "gate": gate,
                            "threshold": pt, "net_benefit": net_benefit(y[k], p[k], pt)})
    lo, hi = paired_brier_diff_ci(y, p, keep["symmetric_confidence"], keep["reliability"], rng)
    for r in rows:
        r["brier_sym_minus_rel_ci_lo"], r["brier_sym_minus_rel_ci_hi"] = lo, hi
    for pt in NB_THRESHOLDS:
        nb_rows.append({"source": source, "task": task, "gate": "no_deferral",
                        "threshold": pt, "net_benefit": net_benefit(y, p, pt)})
    return rows, nb_rows


def main() -> None:
    """Run the symmetric-gate comparison on every available artifact file."""
    rng = np.random.default_rng(SEED)
    rows, nb_rows = [], []
    for source, template in SOURCES.items():
        for task in TASKS:
            path = ARTIFACT_DIR / template.format(task=task)
            try:
                df = pd.read_parquet(path)
            except FileNotFoundError:
                logger.warning("Missing artifact: %s", path)
                continue
            r, nb = evaluate(source, task, df, rng)
            rows += r
            nb_rows += nb
            logger.info("%s %s done (n=%d)", source, task, len(df))
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "symmetric_confidence_gate.csv", index=False)
    pd.DataFrame(nb_rows).to_csv(RESULTS_DIR / "symmetric_gate_net_benefit.csv", index=False)
    logger.info("Saved %d gate rows and %d net-benefit rows", len(rows), len(nb_rows))


if __name__ == "__main__":
    main()
