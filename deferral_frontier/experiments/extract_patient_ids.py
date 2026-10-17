"""Extract patient identifiers for every cohort row (R1-1 / R2-1, round 1).

Writes data/{mimic4,mimic3,eicu}/patient_ids.parquet with index = row id
(hadm_id or stay_id) and one column, patient_id. Fails if any row is unmapped.

Usage:
    python -m deferral_frontier.experiments.extract_patient_ids
"""
import logging
import os
from pathlib import Path
from typing import Dict

import pandas as pd
import psycopg2

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
TASKS = ("sepsis", "mortality", "vasopressor")

# db name -> (env prefix, mapping query)
SOURCES: Dict[str, tuple] = {
    "mimic4": ("MIMICIV_DB", "SELECT hadm_id AS row_id, subject_id AS patient_id FROM mimic_core.admissions"),
    "mimic3": ("MIMIC3", "SELECT hadm_id AS row_id, subject_id AS patient_id FROM mimiciii.admissions"),
    "eicu": ("EICU_DB", "SELECT patientunitstayid AS row_id, uniquepid AS patient_id FROM eicu_crd.patient"),
}


def attach_patient_ids(rows: pd.Index, mapping: pd.Series) -> pd.DataFrame:
    """Return a frame indexed by `rows` with a patient_id column; raise if any row is unmapped."""
    missing = ~rows.isin(mapping.index)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} of {len(rows)} rows have no patient id")
    return pd.DataFrame({"patient_id": mapping.loc[rows].to_numpy()}, index=rows)


def _load_env(path: Path) -> Dict[str, str]:
    env = {}
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def _connect(env: Dict[str, str], prefix: str):
    get = lambda k, d=None: env.get(f"{prefix}_{k}", d)  # noqa: E731
    if prefix == "MIMIC3":
        return psycopg2.connect(host=get("HOST"), port=get("PORT", "5432"), dbname=get("DB"),
                                user=get("USER"), password=get("PASSWORD"), connect_timeout=10)
    return psycopg2.connect(host=get("HOST"), port=get("PORT", "5432"), dbname=get("NAME"),
                            user=get("USER"), password=get("PASS"), connect_timeout=10)


def _cohort_rows(data_dir: Path) -> pd.Index:
    idx = [pd.read_parquet(data_dir / f"{t}_labels.parquet").index for t in TASKS]
    out = idx[0]
    for i in idx[1:]:
        out = out.union(i)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    env = {**_load_env(ROOT / ".env"), **os.environ}
    for db, (prefix, query) in SOURCES.items():
        data_dir = ROOT / "data" / db
        rows = _cohort_rows(data_dir)
        with _connect(env, prefix) as con, con.cursor() as cur:
            cur.execute(query)
            m = pd.DataFrame(cur.fetchall(), columns=["row_id", "patient_id"])
        mapping = m.drop_duplicates("row_id").set_index("row_id")["patient_id"]
        out = attach_patient_ids(rows, mapping)
        out.to_parquet(data_dir / "patient_ids.parquet")
        n_pat = out["patient_id"].nunique()
        logger.info("%s: %d rows -> %d patients (100%% mapped)", db, len(out), n_pat)


if __name__ == "__main__":
    main()
