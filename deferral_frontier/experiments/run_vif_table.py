"""Corrected VIF for D1-D5 across all datasets and tasks (supplementary table).

Motivation: "include a brief
supplementary table showing the corrected VIF values for all five
dimensions across all three tasks and all three datasets, not just the
summary statement '1.0-1.4'."

Uses sm.add_constant() before variance_inflation_factor() -- the omission
of the intercept was the root cause of the original, incorrect VIF=19.9
claim (regression through the origin inflates VIF via uncentered R-squared).
"""
import logging
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("deferral_frontier/results")
DIM_COLS = ["D1", "D2", "D3", "D4", "D5"]

SOURCES = {
    "MIMIC-IV": "artifacts/{task}_test_artifacts.parquet",
    "MIMIC-III": "artifacts/mimic3/mimic3_{task}_test_artifacts.parquet",
    "eICU": "artifacts/eicu/eicu_{task}_test_artifacts.parquet",
}
TASKS = ["sepsis", "mortality", "vasopressor"]


def compute_vif(df: pd.DataFrame) -> dict:
    X = sm.add_constant(df[DIM_COLS].values)
    vifs = {}
    for i, name in enumerate(DIM_COLS, start=1):  # column 0 is the constant
        vifs[name] = float(variance_inflation_factor(X, i))
    return vifs


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for source, tpl in SOURCES.items():
        for task in TASKS:
            path = Path(tpl.format(task=task))
            if not path.exists():
                logger.warning("Missing %s, skipping", path)
                continue
            df = pd.read_parquet(path)
            vifs = compute_vif(df)
            row = {"source": source, "task": task, **vifs}
            rows.append(row)
            logger.info("%s/%s: %s", source, task,
                        ", ".join(f"{k}={v:.2f}" for k, v in vifs.items()))

    out = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "vif_table.csv"
    out.to_csv(out_path, index=False)
    logger.info("Saved %d rows to %s", len(out), out_path)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
