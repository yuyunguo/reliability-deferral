"""Subgroup audit beyond deferral rates (reviewer R1-7).

For every dataset-task pair, gate (80% coverage), and subgroup (sex, age band,
race/ethnicity): deferral rate, error capture (share of the group's model
errors at 0.5 that are deferred), retained Brier score, and inappropriate
deferral rate (correctly classified patients deferred, per patient in the group).

Requires data/{db}/demographics.parquet (extract_demographics_r1.py).
Output: deferral_frontier/results/fairness_r1.csv

Usage:
    python -m deferral_frontier.experiments.run_fairness_r1
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
ROOT = Path(__file__).resolve().parents[2]
DB_OF = {"MIMIC-IV": ("mimic4", "hadm_id"), "MIMIC-III": ("mimic3", "hadm_id"), "eICU": ("eicu", "stay_id"),
         "eICU-H": ("eicu", "stay_id")}
AGE_BINS, AGE_LABELS = [18, 40, 55, 65, 75, 200], ["18-39", "40-54", "55-64", "65-74", "75+"]
MIN_GROUP = 30


def group_metrics(y: np.ndarray, p: np.ndarray, keep: np.ndarray) -> Dict[str, float]:
    """Deferral, error-capture, retained-Brier, and inappropriate-deferral metrics."""
    wrong = (p >= 0.5).astype(int) != y
    deferred = ~keep
    n_err = int(wrong.sum())
    return {
        "n": len(y), "prevalence": float(y.mean()),
        "deferral_rate": float(deferred.mean()),
        "error_capture": float((wrong & deferred).sum() / n_err) if n_err else np.nan,
        "inappropriate_deferral_rate": float((~wrong & deferred).mean()),
        "brier_retained": float(np.mean((p[keep] - y[keep]) ** 2)) if keep.any() else np.nan,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rows = []
    for source, tmpl in SOURCES.items():
        db, id_col = DB_OF[source]
        demo = pd.read_parquet(ROOT / "data" / db / "demographics.parquet")
        demo["age_band"] = pd.cut(demo["age"], AGE_BINS, labels=AGE_LABELS, right=False).astype(str)
        for task in TASKS:
            df = pd.read_parquet(ARTIFACT_DIR / tmpl.format(task=task)).set_index(id_col)
            df = df.join(demo[["sex", "age_band", "race_group"]], how="left")
            y, p = df["y_true"].to_numpy().astype(int), df["cal_proba"].to_numpy()
            gates = {"one_sided_confidence": p, "symmetric_confidence": np.maximum(p, 1 - p),
                     "reliability": df["r_score"].to_numpy()}
            for gate, score in gates.items():
                keep = retained_mask(score, COVERAGE)
                for strat in ("sex", "age_band", "race_group"):
                    for group, pos in df.reset_index().groupby(strat).indices.items():
                        if len(pos) < MIN_GROUP:
                            continue
                        rows.append({"source": source, "task": task, "gate": gate,
                                     "stratification": strat, "group": group,
                                     **group_metrics(y[pos], p[pos], keep[pos])})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "fairness_r1.csv", index=False)
    logger.info("Wrote %d subgroup rows to fairness_r1.csv", len(out))


if __name__ == "__main__":
    main()
