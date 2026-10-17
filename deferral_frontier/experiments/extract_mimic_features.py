"""Re-extract MIMIC-III / MIMIC-IV cohorts with one shared definition (round 1).

Cohort: adult ICU stays >= 24 h, earliest per admission; prediction at ICU
intime + 24 h; features from [intime, intime + 24 h). Outputs go to
data/{db}_r1/ in the layout MIMICLoader expects; swap directories after review.

Usage:
    python -m deferral_frontier.experiments.extract_mimic_features mimic4
    python -m deferral_frontier.experiments.extract_mimic_features mimic3
"""
import argparse
import logging
from pathlib import Path
from typing import Dict

import pandas as pd

from deferral_frontier.experiments.extract_patient_ids import _connect, _load_env
from deferral_frontier.src.icu_features import ITEM_TO_SIGNAL, LAB_MAP, VITAL_MAP, aggregate_window
from deferral_frontier.src.icu_labels import (
    VASO_IDS_CV, VASO_IDS_MV, select_cohort, sepsis_labels, vasopressor_labels,
)

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
BATCH = 2000

SQL: Dict[str, Dict[str, str]] = {
    "mimic4": {
        "env": "MIMICIV_DB",
        "stays": """SELECT i.stay_id, i.hadm_id, i.subject_id, i.intime, i.outtime, i.los,
                     p.anchor_age + (EXTRACT(YEAR FROM i.intime) - p.anchor_year) AS age,
                     a.hospital_expire_flag
                   FROM mimic_icu.icustays i
                   JOIN mimic_core.admissions a ON a.hadm_id = i.hadm_id
                   JOIN mimic_core.patients p ON p.subject_id = i.subject_id""",
        "vitals": """SELECT w.hadm_id, c.charttime, c.itemid, c.valuenum
                    FROM mimic_icu.chartevents c
                    JOIN unnest(%(stay)s::int[], %(hadm)s::int[], %(s)s::timestamp[], %(e)s::timestamp[])
                         AS w(stay_id, hadm_id, s, e) ON c.stay_id = w.stay_id
                    WHERE c.itemid = ANY(%(ids)s) AND c.valuenum > 0
                      AND c.charttime >= w.s AND c.charttime < w.e""",
        "labs": """SELECT w.hadm_id, l.charttime, l.itemid, l.valuenum
                  FROM mimic_hosp.labevents l
                  JOIN unnest(%(hadm)s::int[], %(s)s::timestamp[], %(e)s::timestamp[])
                       AS w(hadm_id, s, e) ON l.hadm_id = w.hadm_id
                  WHERE l.itemid = ANY(%(ids)s) AND l.valuenum > 0
                    AND l.charttime >= w.s AND l.charttime < w.e""",
        "dx": "SELECT hadm_id, icd_code, icd_version FROM mimic_hosp.diagnoses_icd WHERE hadm_id = ANY(%(hadm)s)",
        "vaso": """SELECT stay_id, starttime AS event_time FROM mimic_icu.inputevents
                  WHERE itemid = ANY(%(mv)s) AND stay_id = ANY(%(stay)s)""",
    },
    "mimic3": {
        "env": "MIMIC3",
        "stays": """SELECT i.icustay_id AS stay_id, i.hadm_id, i.subject_id, i.intime, i.outtime, i.los,
                     EXTRACT(EPOCH FROM i.intime - p.dob) / 31557600.0 AS age,
                     a.hospital_expire_flag
                   FROM mimiciii.icustays i
                   JOIN mimiciii.admissions a ON a.hadm_id = i.hadm_id
                   JOIN mimiciii.patients p ON p.subject_id = i.subject_id""",
        "vitals": """SELECT w.hadm_id, c.charttime, c.itemid, c.valuenum
                    FROM mimiciii.chartevents c
                    JOIN unnest(%(stay)s::int[], %(hadm)s::int[], %(s)s::timestamp[], %(e)s::timestamp[])
                         AS w(stay_id, hadm_id, s, e) ON c.icustay_id = w.stay_id
                    WHERE c.itemid = ANY(%(ids)s) AND c.valuenum > 0 AND c.error IS DISTINCT FROM 1
                      AND c.charttime >= w.s AND c.charttime < w.e""",
        "labs": """SELECT w.hadm_id, l.charttime, l.itemid, l.valuenum
                  FROM mimiciii.labevents l
                  JOIN unnest(%(hadm)s::int[], %(s)s::timestamp[], %(e)s::timestamp[])
                       AS w(hadm_id, s, e) ON l.hadm_id = w.hadm_id
                  WHERE l.itemid = ANY(%(ids)s) AND l.valuenum > 0
                    AND l.charttime >= w.s AND l.charttime < w.e""",
        "dx": """SELECT hadm_id, icd9_code AS icd_code, 9 AS icd_version
                FROM mimiciii.diagnoses_icd WHERE hadm_id = ANY(%(hadm)s)""",
        "vaso": """SELECT icustay_id AS stay_id, starttime AS event_time FROM mimiciii.inputevents_mv
                  WHERE itemid = ANY(%(mv)s) AND icustay_id = ANY(%(stay)s)
                  UNION ALL
                  SELECT icustay_id, charttime FROM mimiciii.inputevents_cv
                  WHERE itemid = ANY(%(cv)s) AND icustay_id = ANY(%(stay)s)""",
    },
}


def _q(cur, sql: str, params: dict, cols: list) -> pd.DataFrame:
    cur.execute(sql, params)
    return pd.DataFrame(cur.fetchall(), columns=cols)


def _batched(cur, sql: str, cohort: pd.DataFrame, ids: list, with_stay: bool) -> pd.DataFrame:
    parts = []
    for i in range(0, len(cohort), BATCH):
        b = cohort.iloc[i:i + BATCH]
        p = {"hadm": [int(h) for h in b.index], "s": list(b["admission_time"]),
             "e": list(b["prediction_time"]), "ids": ids}
        if with_stay:
            p["stay"] = [int(s) for s in b["stay_id"]]
        parts.append(_q(cur, sql, p, ["hadm_id", "charttime", "itemid", "value"]))
        logger.info("  batch %d/%d", i // BATCH + 1, (len(cohort) - 1) // BATCH + 1)
    df = pd.concat(parts, ignore_index=True)
    df["feature_name"] = df["itemid"].map(ITEM_TO_SIGNAL)
    return df.dropna(subset=["feature_name"])[["hadm_id", "charttime", "feature_name", "value"]]


def extract(db: str) -> None:
    q, out_dir = SQL[db], ROOT / "data" / f"{db}_r1"
    out_dir.mkdir(parents=True, exist_ok=True)
    with _connect(_load_env(ROOT / ".env"), q["env"]) as con, con.cursor() as cur:
        stays = _q(cur, q["stays"], {}, ["stay_id", "hadm_id", "subject_id", "intime",
                                         "outtime", "los", "age", "hospital_expire_flag"])
        stays["age"] = stays["age"].astype(float)
        stays["los"] = stays["los"].astype(float)
        cohort = select_cohort(stays)
        logger.info("%s: %d stays -> %d admissions after cohort rule", db, len(stays), len(cohort))

        vit_ids = [i for ids in VITAL_MAP.values() for i in ids]
        lab_ids = [i for ids in LAB_MAP.values() for i in ids]
        ts = pd.concat([_batched(cur, q["vitals"], cohort, vit_ids, True),
                        _batched(cur, q["labs"], cohort, lab_ids, False)], ignore_index=True)
        dx = _q(cur, q["dx"], {"hadm": [int(h) for h in cohort.index]},
                ["hadm_id", "icd_code", "icd_version"])
        vaso = _q(cur, q["vaso"], {"mv": list(VASO_IDS_MV), "cv": list(VASO_IDS_CV),
                                   "stay": [int(s) for s in cohort["stay_id"]]},
                  ["stay_id", "event_time"])

    feats = aggregate_window(ts, "hadm_id", cohort["admission_time"], cohort["prediction_time"])
    has_obs = feats.filter(like="_count").sum(axis=1) > 0
    cohort, feats = cohort[has_obs], feats[has_obs]
    logger.info("%s: %d admissions with >= 1 observation in the window", db, len(cohort))

    ts = ts[ts["hadm_id"].isin(cohort.index)].rename(columns={"hadm_id": "stay_id"})
    meta = cohort[["admission_time", "prediction_time"]].assign(admit_year=cohort["intime"].dt.year)
    labels = {
        "sepsis": sepsis_labels(dx, cohort.index),
        "mortality": cohort["hospital_expire_flag"].astype(int).rename("label"),
        "vasopressor": vasopressor_labels(vaso, cohort[["stay_id", "prediction_time", "outtime"]]),
    }
    for task, lab in labels.items():
        lab.index.name = "hadm_id"
        lab.to_frame("label").to_parquet(out_dir / f"{task}_labels.parquet")
        feats.loc[lab.index].to_parquet(out_dir / f"{task}_features.parquet")
        meta.loc[lab.index].to_parquet(out_dir / f"{task}_meta.parquet")
        ts[ts["stay_id"].isin(lab.index)].to_parquet(out_dir / f"{task}_timeseries.parquet")
        logger.info("  %s: n=%d prevalence=%.3f", task, len(lab), lab.mean())
    cohort[["subject_id"]].rename(columns={"subject_id": "patient_id"}).to_parquet(out_dir / "patient_ids.parquet")


def rebuild_vasopressor(db: str, data_dir: Path) -> None:
    """Recompute only the vasopressor task in an extracted directory (same cohort and windows)."""
    q = SQL[db]
    rows = pd.read_parquet(data_dir / "mortality_labels.parquet").index
    with _connect(_load_env(ROOT / ".env"), q["env"]) as con, con.cursor() as cur:
        stays = _q(cur, q["stays"], {}, ["stay_id", "hadm_id", "subject_id", "intime",
                                         "outtime", "los", "age", "hospital_expire_flag"])
        stays[["age", "los"]] = stays[["age", "los"]].astype(float)
        cohort = select_cohort(stays).loc[rows]
        vaso = _q(cur, q["vaso"], {"mv": list(VASO_IDS_MV), "cv": list(VASO_IDS_CV),
                                   "stay": [int(s) for s in cohort["stay_id"]]},
                  ["stay_id", "event_time"])
    lab = vasopressor_labels(vaso, cohort[["stay_id", "prediction_time", "outtime"]])
    for part in ("features", "meta"):
        pd.read_parquet(data_dir / f"mortality_{part}.parquet").loc[lab.index].to_parquet(
            data_dir / f"vasopressor_{part}.parquet")
    ts = pd.read_parquet(data_dir / "mortality_timeseries.parquet")
    ts[ts["stay_id"].isin(lab.index)].to_parquet(data_dir / "vasopressor_timeseries.parquet")
    lab.to_frame("label").to_parquet(data_dir / "vasopressor_labels.parquet")
    logger.info("%s vasopressor rebuilt: n=%d prevalence=%.3f", db, len(lab), lab.mean())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("db", choices=sorted(SQL))
    ap.add_argument("--vasopressor-only", type=Path, default=None,
                    help="rebuild only the vasopressor task inside this data directory")
    a = ap.parse_args()
    if a.vasopressor_only:
        rebuild_vasopressor(a.db, a.vasopressor_only)
    else:
        extract(a.db)


if __name__ == "__main__":
    main()
