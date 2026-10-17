"""Build a MIMIC-IV 'sepsis3' task with Sepsis-3 labels (reviewer R1-5 sensitivity).

Label = mimic_derived.sepsis3.sepsis3 is true for the cohort's ICU stay.
Features, metadata and timeseries are copied from the 'sepsis' task, so only
the label differs. Writes data/mimic4/sepsis3_*.parquet.

Usage:
    python -m deferral_frontier.experiments.build_sepsis3_task
"""
import logging
import shutil
from pathlib import Path
from typing import Set

import pandas as pd

from deferral_frontier.experiments.extract_mimic_features import SQL, _q
from deferral_frontier.experiments.extract_patient_ids import _connect, _load_env
from deferral_frontier.src.icu_labels import select_cohort

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "mimic4"


def sepsis3_labels(flagged_stay_ids: Set[int], cohort_stays: pd.Series) -> pd.Series:
    """0/1 Sepsis-3 label per admission, given the admission's ICU stay_id."""
    lab = cohort_stays.isin(flagged_stay_ids).astype(int).rename("label")
    lab.index.name = "hadm_id"
    return lab


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rows = pd.read_parquet(DATA / "sepsis_labels.parquet").index
    with _connect(_load_env(ROOT / ".env"), "MIMICIV_DB") as con, con.cursor() as cur:
        stays = _q(cur, SQL["mimic4"]["stays"], {}, ["stay_id", "hadm_id", "subject_id", "intime",
                                                     "outtime", "los", "age", "hospital_expire_flag"])
        cur.execute("SELECT stay_id FROM mimic_derived.sepsis3 WHERE sepsis3")
        flagged = {int(r[0]) for r in cur.fetchall()}
    stays[["age", "los"]] = stays[["age", "los"]].astype(float)
    cohort = select_cohort(stays).loc[rows]
    labels = sepsis3_labels(flagged, cohort["stay_id"])
    labels.to_frame().to_parquet(DATA / "sepsis3_labels.parquet")
    for part in ("features", "meta", "timeseries"):
        shutil.copyfile(DATA / f"sepsis_{part}.parquet", DATA / f"sepsis3_{part}.parquet")
    icd = pd.read_parquet(DATA / "sepsis_labels.parquet")["label"]
    agree = (icd == labels).mean()
    logger.info("Sepsis-3 prevalence %.3f vs ICD %.3f; agreement %.3f; n=%d",
                labels.mean(), icd.mean(), agree, len(labels))


if __name__ == "__main__":
    main()
