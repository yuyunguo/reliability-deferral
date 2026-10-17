"""Sepsis-3 sensitivity (reviewer R1-5): the three-gate comparison on MIMIC-IV
with Sepsis-3 labels, side by side with the ICD-label task.

Output: deferral_frontier/results/sepsis3_sensitivity.csv

Usage:
    python -m deferral_frontier.experiments.r1_sepsis3_summary
"""
import logging

import numpy as np
import pandas as pd

from deferral_frontier.experiments.run_symmetric_confidence_gate import (
    ARTIFACT_DIR, RESULTS_DIR, SEED, evaluate,
)

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rows = []
    for label, task in (("ICD (primary)", "sepsis"), ("Sepsis-3", "sepsis3")):
        df = pd.read_parquet(ARTIFACT_DIR / f"{task}_test_artifacts.parquet")
        r, _ = evaluate("MIMIC-IV", task, df, np.random.default_rng(SEED))
        for row in r:
            row.update(label_definition=label, prevalence=df["y_true"].mean())
        rows += r
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "sepsis3_sensitivity.csv", index=False)
    logger.info("\n%s", out[["label_definition", "gate", "brier", "auroc", "jaccard_deferred_vs_reliability",
                             "brier_sym_minus_rel_ci_lo", "brier_sym_minus_rel_ci_hi"]]
                .to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
