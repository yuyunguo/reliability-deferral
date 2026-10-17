"""Imperfect clinician review of deferred cases (reviewer R2-6).

For each gate at 80% coverage, deferred patients are decided by a clinician who
is correct with probability `a`. Expected total errors =
model errors among retained + (1 - a) x number deferred.
Break-even accuracy = 1 - model error rate among deferred: below it, deferral
adds errors rather than removing them.

Output: deferral_frontier/results/imperfect_review.csv

Usage:
    python -m deferral_frontier.experiments.run_imperfect_review
"""
import logging
from typing import Dict

import numpy as np
import pandas as pd

from deferral_frontier.experiments.run_symmetric_confidence_gate import (
    ARTIFACT_DIR, COVERAGE, RESULTS_DIR, SOURCES, TASKS, retained_mask,
)

logger = logging.getLogger(__name__)
ACCURACIES = (1.0, 0.9, 0.8, 0.7)


def review_outcome(y: np.ndarray, p: np.ndarray, keep: np.ndarray,
                   clinician_accuracy: float) -> Dict[str, float]:
    """Expected errors when deferred cases are reviewed with the given accuracy."""
    wrong = (p >= 0.5).astype(int) != y
    n_def = int((~keep).sum())
    err_def = int(wrong[~keep].sum())
    expected = int(wrong[keep].sum()) + (1 - clinician_accuracy) * n_def
    return {
        "n": len(y), "n_deferred": n_def,
        "model_errors_retained": int(wrong[keep].sum()),
        "model_errors_deferred": err_def,
        "expected_errors_total": float(expected),
        "error_rate_no_deferral": float(wrong.mean()),
        "error_rate_with_review": float(expected / len(y)),
        "break_even_accuracy": 1 - err_def / n_def if n_def else np.nan,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rows = []
    for source, tmpl in SOURCES.items():
        for task in TASKS:
            df = pd.read_parquet(ARTIFACT_DIR / tmpl.format(task=task))
            y, p = df["y_true"].to_numpy().astype(int), df["cal_proba"].to_numpy()
            gates = {"one_sided_confidence": p, "symmetric_confidence": np.maximum(p, 1 - p),
                     "reliability": df["r_score"].to_numpy()}
            for gate, score in gates.items():
                keep = retained_mask(score, COVERAGE)
                for a in ACCURACIES:
                    rows.append({"source": source, "task": task, "gate": gate,
                                 "clinician_accuracy": a, **review_outcome(y, p, keep, a)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "imperfect_review.csv", index=False)
    logger.info("\n%s", out[out["clinician_accuracy"] == 1.0][
        ["source", "task", "gate", "break_even_accuracy"]].to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
