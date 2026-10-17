"""Editor E-1 table: early-stopping round and train/validation AUROC and log loss,
mean (min-max) over the five ensemble members, for the nine primary models.

Output: deferral_frontier/results/base_model_diagnostics.csv

Usage:
    python -m deferral_frontier.experiments.r1_diagnostics_table
"""
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
RESULTS = Path(__file__).resolve().parents[1] / "results"
ORDER = [(d, t) for d in ("mimic4", "mimic3", "eicu") for t in ("sepsis", "mortality", "vasopressor")]
METRICS = ("best_iteration", "train_auroc", "val_auroc", "train_logloss", "val_logloss")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rows = []
    for db, task in ORDER:
        d = pd.read_csv(RESULTS / f"base_diagnostics_{db}_{task}.csv")
        row = {"dataset": db, "task": task}
        for m in METRICS:
            row[f"{m}_mean"], row[f"{m}_min"], row[f"{m}_max"] = d[m].mean(), d[m].min(), d[m].max()
        row["auroc_gap_mean"] = (d["train_auroc"] - d["val_auroc"]).mean()
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "base_model_diagnostics.csv", index=False, float_format="%.4f")
    logger.info("\n%s", out[["dataset", "task", "best_iteration_mean", "train_auroc_mean", "val_auroc_mean",
                             "auroc_gap_mean", "train_logloss_mean", "val_logloss_mean"]]
                .to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
