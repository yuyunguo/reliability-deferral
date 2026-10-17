"""Calibration and class-conditional diagnostics (editor E-3, reviewer R1-6).

Outputs (deferral_frontier/results/):
    calibration_bins.csv          10 quantile bins of calibrated risk, all 9 pairs
    confusion_matrix_05.csv       full-test confusion matrix at threshold 0.5
    class_conditional_gates.csv   per gate at 80% coverage: Brier and counts by class,
                                  sensitivity and specificity at 0.5

Usage:
    python -m deferral_frontier.experiments.run_calibration_diagnostics
"""
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from deferral_frontier.experiments.run_symmetric_confidence_gate import (
    ARTIFACT_DIR, COVERAGE, RESULTS_DIR, SOURCES, TASKS, retained_mask,
)

logger = logging.getLogger(__name__)
THRESHOLD = 0.5


def confusion_at(y: np.ndarray, p: np.ndarray, t: float) -> Dict[str, int]:
    """Confusion counts when predicting positive for p >= t."""
    pred = p >= t
    return {"tp": int(np.sum(pred & (y == 1))), "fn": int(np.sum(~pred & (y == 1))),
            "fp": int(np.sum(pred & (y == 0))), "tn": int(np.sum(~pred & (y == 0)))}


def class_conditional(y: np.ndarray, p: np.ndarray, keep: np.ndarray) -> Dict[str, float]:
    """Brier score and 0.5-threshold rates by outcome class within the retained set."""
    yk, pk = y[keep], p[keep]
    pos, neg = yk == 1, yk == 0
    c = confusion_at(yk, pk, THRESHOLD)
    return {
        "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
        "brier_pos": float(np.mean((pk[pos] - 1) ** 2)) if pos.any() else np.nan,
        "brier_neg": float(np.mean(pk[neg] ** 2)) if neg.any() else np.nan,
        "sensitivity": c["tp"] / max(c["tp"] + c["fn"], 1),
        "specificity": c["tn"] / max(c["tn"] + c["fp"], 1),
    }


def calibration_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Equal-count bins of predicted risk with mean prediction and observed rate."""
    order = np.argsort(p, kind="stable")
    rows = []
    for b, idx in enumerate(np.array_split(order, n_bins)):
        rows.append({"bin": b, "n": len(idx), "mean_pred": float(p[idx].mean()),
                     "obs_rate": float(y[idx].mean())})
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    bins, conf, cc = [], [], []
    for source, tmpl in SOURCES.items():
        for task in TASKS:
            df = pd.read_parquet(ARTIFACT_DIR / tmpl.format(task=task))
            y = df["y_true"].to_numpy().astype(int)
            p = df["cal_proba"].to_numpy()
            key = {"source": source, "task": task}
            bins.append(calibration_bins(y, p).assign(**key))
            conf.append({**key, **confusion_at(y, p, THRESHOLD)})
            gates = {"one_sided_confidence": p, "symmetric_confidence": np.maximum(p, 1 - p),
                     "reliability": df["r_score"].to_numpy()}
            cc.append({**key, "gate": "no_deferral", **class_conditional(y, p, np.ones_like(y, bool))})
            for gate, score in gates.items():
                cc.append({**key, "gate": gate, **class_conditional(y, p, retained_mask(score, COVERAGE))})
    pd.concat(bins).to_csv(RESULTS_DIR / "calibration_bins.csv", index=False)
    pd.DataFrame(conf).to_csv(RESULTS_DIR / "confusion_matrix_05.csv", index=False)
    pd.DataFrame(cc).to_csv(RESULTS_DIR / "class_conditional_gates.csv", index=False)
    logger.info("Wrote calibration_bins.csv, confusion_matrix_05.csv, class_conditional_gates.csv")


if __name__ == "__main__":
    main()
