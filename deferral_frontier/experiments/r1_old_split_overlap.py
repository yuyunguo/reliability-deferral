"""Quantify patient overlap in the ORIGINAL (round-0) splits (R2-1, round 1).

Development set = every cohort row not in the archived test set (train + val;
both shaped the model through fitting, early stopping and calibration).

Usage:
    python -m deferral_frontier.experiments.r1_old_split_overlap
"""
import logging
from pathlib import Path
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
SNAP = ROOT / "artifacts_pre_r1_2026-10-16"
OUT = ROOT / "deferral_frontier" / "results" / "r1_old_split_patient_overlap.csv"
TASKS = ("sepsis", "mortality", "vasopressor")
DBS = {  # db -> (archived test-artifact path template, row-id column)
    "mimic4": ("{t}_test_artifacts.parquet", "hadm_id"),
    "mimic3": ("mimic3/mimic3_{t}_test_artifacts.parquet", "hadm_id"),
    "eicu": ("eicu/eicu_{t}_test_artifacts.parquet", "stay_id"),
}


def patient_overlap(test_rows: pd.Index, dev_rows: pd.Index, pid: pd.Series) -> Dict[str, float]:
    """Count test rows/patients whose patient also has a row in the development set."""
    test_pid = pid.loc[test_rows]
    dev_patients = set(pid.loc[dev_rows])
    in_dev = test_pid.isin(dev_patients)
    return {
        "test_rows": len(test_rows),
        "test_patients": test_pid.nunique(),
        "test_rows_in_dev": int(in_dev.sum()),
        "test_patients_in_dev": test_pid[in_dev].nunique(),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rows = []
    for db, (tmpl, col) in DBS.items():
        pid = pd.read_parquet(ROOT / "data" / db / "patient_ids.parquet")["patient_id"]
        for t in TASKS:
            cohort = pd.read_parquet(ROOT / "data" / db / f"{t}_labels.parquet").index
            test = pd.Index(pd.read_parquet(SNAP / tmpl.format(t=t))[col])
            res = patient_overlap(test, cohort.difference(test), pid)
            res.update(db=db, task=t,
                       pct_test_rows_in_dev=100 * res["test_rows_in_dev"] / res["test_rows"],
                       pct_test_patients_in_dev=100 * res["test_patients_in_dev"] / res["test_patients"])
            rows.append(res)
    df = pd.DataFrame(rows)[["db", "task", "test_rows", "test_patients", "test_rows_in_dev",
                             "pct_test_rows_in_dev", "test_patients_in_dev", "pct_test_patients_in_dev"]]
    df.to_csv(OUT, index=False, float_format="%.1f")
    logger.info("\n%s", df.to_string(index=False, float_format="%.1f"))


if __name__ == "__main__":
    main()
