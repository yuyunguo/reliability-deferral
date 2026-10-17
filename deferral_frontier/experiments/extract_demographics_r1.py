"""Demographics for every cohort row in all three databases (reviewer R1-7).

Writes data/{mimic4,mimic3,eicu}/demographics.parquet with columns sex, age,
race_group (White, Black, Hispanic, Asian, Other/Unknown).

Usage:
    python -m deferral_frontier.experiments.extract_demographics_r1
"""
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from deferral_frontier.experiments.extract_patient_ids import TASKS, _connect, _load_env

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]

SQL = {
    "mimic4": ("MIMICIV_DB", """
        SELECT a.hadm_id, p.gender, p.anchor_age + (EXTRACT(YEAR FROM a.admittime) - p.anchor_year), a.race
        FROM mimic_core.admissions a JOIN mimic_core.patients p ON p.subject_id = a.subject_id
        WHERE a.hadm_id = ANY(%(ids)s)"""),
    "mimic3": ("MIMIC3", """
        SELECT a.hadm_id, p.gender, LEAST(EXTRACT(EPOCH FROM a.admittime - p.dob) / 31557600.0, 90), a.ethnicity
        FROM mimiciii.admissions a JOIN mimiciii.patients p ON p.subject_id = a.subject_id
        WHERE a.hadm_id = ANY(%(ids)s)"""),
    "eicu": ("EICU_DB", """
        SELECT patientunitstayid, gender,
               CASE WHEN age = '> 89' THEN 90 WHEN age ~ '^[0-9]+$' THEN age::int END, ethnicity
        FROM eicu_crd.patient WHERE patientunitstayid = ANY(%(ids)s)"""),
}


def race_group(raw: Optional[str]) -> str:
    """Map database-specific race/ethnicity strings to five shared groups."""
    s = (raw or "").upper()
    if "WHITE" in s or "CAUCASIAN" in s:
        return "White"
    if "BLACK" in s or "AFRICAN" in s:
        return "Black"
    if "HISPANIC" in s or "LATINO" in s:
        return "Hispanic"
    if "ASIAN" in s:
        return "Asian"
    return "Other/Unknown"


def _sex(raw: Optional[str]) -> str:
    s = (raw or "").upper()
    return "Female" if s in ("F", "FEMALE") else "Male" if s in ("M", "MALE") else "Unknown"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    env = _load_env(ROOT / ".env")
    for db, (prefix, sql) in SQL.items():
        data_dir = ROOT / "data" / db
        ids = sorted(set().union(*(pd.read_parquet(data_dir / f"{t}_labels.parquet").index for t in TASKS)))
        with _connect(env, prefix) as con, con.cursor() as cur:
            cur.execute(sql, {"ids": [int(i) for i in ids]})
            df = pd.DataFrame(cur.fetchall(), columns=["row_id", "sex", "age", "race"])
        df = df.drop_duplicates("row_id").set_index("row_id")
        out = pd.DataFrame({"sex": df["sex"].map(_sex), "age": df["age"].astype(float),
                            "race_group": df["race"].map(race_group)})
        out.index.name = "hadm_id" if db != "eicu" else "stay_id"
        missing = len(set(ids) - set(out.index))
        if missing:
            raise ValueError(f"{db}: {missing} cohort rows have no demographics")
        out.to_parquet(data_dir / "demographics.parquet")
        logger.info("%s: %d rows; %s", db, len(out), out["race_group"].value_counts().to_dict())


if __name__ == "__main__":
    main()
