"""Extract MIMIC-III features mirroring MIMIC-IV pipeline (vectorized).

Extracts ICU cohorts with 23 raw features (12 vitals + 11 labs),
aggregated identically to MIMIC-IV (mean, min, max, last, count within 24h).
"""
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_SCHEMA = "mimiciii"

OUTPUT_DIR = Path("data/mimic3")

VITAL_MAP = {
    "vital_heart_rate": [211, 220045],
    "vital_resp_rate": [618, 615, 220210, 224690],
    "vital_abp_sys": [51, 220050, 220179],
    "vital_abp_dia": [8368, 220051, 220180],
    "vital_abp_mean": [52, 220052, 220181],
    "vital_nibp_sys": [442, 455, 220179],
    "vital_nibp_dia": [8440, 8441, 220180],
    "vital_nibp_mean": [456, 220181],
    "vital_temp_f": [678, 223761],
    "vital_temp_c": [676, 223762],
}

LAB_MAP = {
    "lab_creatinine": [50912],
    "lab_hemoglobin": [50811],
    "lab_wbc": [51301],
    "lab_platelet": [51265],
    "lab_sodium": [50983],
    "lab_potassium": [50971],
    "lab_glucose": [50931],
    "lab_bicarbonate": [50882],
    "lab_bun": [51006],
    "lab_alt": [50861],
    "lab_ast": [50878],
    "lab_bilirubin": [50885],
    "lab_lactate": [50813],
}

ALL_VITAL_IDS = [i for ids in VITAL_MAP.values() for i in ids]
ALL_LAB_IDS = [i for ids in LAB_MAP.values() for i in ids]

SEPSIS_ICD9 = ("99591", "99592", "78552")

VASO_IDS_CV = [30047, 30051, 30053, 30054, 30055, 30056, 30057, 30058, 30059,
               30119, 30120, 30125, 30128, 30140, 30141]
VASO_IDS_MV = [221906, 221289, 222315]


def get_conn():
    import psycopg2
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASS,
    )


def extract_cohort(conn):
    logger.info("Extracting ICU cohort...")
    q = f"""
    SELECT
        i.icustay_id AS stay_id,
        i.hadm_id,
        i.subject_id,
        i.intime AS admission_time,
        i.intime + interval '24 hours' AS prediction_time,
        i.outtime,
        EXTRACT(YEAR FROM i.intime)::int AS admit_year,
        a.hospital_expire_flag
    FROM {DB_SCHEMA}.icustays i
    JOIN {DB_SCHEMA}.admissions a ON i.hadm_id = a.hadm_id
    WHERE i.los >= 1.0
    """
    df = pd.read_sql(q, conn)
    logger.info("  %d ICU stays (LOS >= 1 day)", len(df))
    return df


def _int_list(arr):
    return [int(x) for x in arr]


def extract_vitals(conn, stay_ids):
    logger.info("Extracting vitals for %d stays...", len(stay_ids))
    chunks = []
    stay_list = _int_list(stay_ids)
    vital_ids = _int_list(ALL_VITAL_IDS)
    batch = 2000
    for i in range(0, len(stay_list), batch):
        q = f"""
        SELECT icustay_id AS stay_id, charttime, itemid, valuenum
        FROM {DB_SCHEMA}.chartevents
        WHERE icustay_id = ANY(%(s)s)
          AND itemid = ANY(%(ids)s)
          AND valuenum IS NOT NULL AND valuenum > 0
          AND error IS DISTINCT FROM 1
        """
        chunks.append(pd.read_sql(q, conn, params={"s": stay_list[i:i+batch], "ids": vital_ids}))
        logger.info("    batch %d/%d", i//batch+1, (len(stay_list)-1)//batch+1)
    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    logger.info("  %d vital rows", len(df))

    id_to_feat = {}
    for feat, ids in VITAL_MAP.items():
        for iid in ids:
            id_to_feat[iid] = feat
    df["feature_name"] = df["itemid"].map(id_to_feat)
    df = df.dropna(subset=["feature_name"])
    return df[["stay_id", "charttime", "feature_name", "valuenum"]].rename(columns={"valuenum": "value"})


def extract_labs(conn, hadm_ids):
    logger.info("Extracting labs for %d admissions...", len(hadm_ids))
    chunks = []
    hlist = _int_list(hadm_ids)
    lab_ids = _int_list(ALL_LAB_IDS)
    batch = 2000
    for i in range(0, len(hlist), batch):
        q = f"""
        SELECT hadm_id, charttime, itemid, valuenum
        FROM {DB_SCHEMA}.labevents
        WHERE hadm_id = ANY(%(h)s)
          AND itemid = ANY(%(ids)s)
          AND valuenum IS NOT NULL AND valuenum > 0
        """
        chunks.append(pd.read_sql(q, conn, params={"h": hlist[i:i+batch], "ids": lab_ids}))
        logger.info("    batch %d/%d", i//batch+1, (len(hlist)-1)//batch+1)
    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    logger.info("  %d lab rows", len(df))

    id_to_feat = {}
    for feat, ids in LAB_MAP.items():
        for iid in ids:
            id_to_feat[iid] = feat
    df["feature_name"] = df["itemid"].map(id_to_feat)
    df = df.dropna(subset=["feature_name"])
    return df[["hadm_id", "charttime", "feature_name", "valuenum"]].rename(columns={"valuenum": "value"})


def vectorized_aggregate(ts_df, id_col, cohort):
    logger.info("Vectorized aggregation on %d rows...", len(ts_df))
    pred_map = cohort.drop_duplicates(id_col).set_index(id_col)["prediction_time"]
    ts_df = ts_df.copy()
    ts_df["pred_time"] = ts_df[id_col].map(pred_map)
    ts_df = ts_df.dropna(subset=["pred_time"])
    ts_df = ts_df[ts_df["charttime"] <= ts_df["pred_time"]]

    agg = (ts_df.groupby([id_col, "feature_name"])["value"]
           .agg(mean="mean", mi="min", ma="max", last="last", count="count")
           .reset_index())

    rows = []
    for _, row in agg.iterrows():
        fid = row[id_col]
        feat = row["feature_name"]
        rows.append({
            id_col: fid,
            f"{feat}_mean": row["mean"],
            f"{feat}_min": row["mi"],
            f"{feat}_max": row["ma"],
            f"{feat}_last": row["last"],
            f"{feat}_count": row["count"],
        })
    long = pd.DataFrame(rows)
    wide = long.groupby(id_col).first()
    for col in long.columns:
        if col != id_col:
            if col not in wide.columns:
                wide[col] = np.nan
            else:
                wide[col] = long.groupby(id_col)[col].first()
    return wide


def get_sepsis_labels(conn, hadm_ids):
    q = f"""
    SELECT DISTINCT hadm_id FROM {DB_SCHEMA}.diagnoses_icd
    WHERE hadm_id = ANY(%(h)s) AND icd9_code IN %(codes)s
    """
    pos = pd.read_sql(q, conn, params={"h": _int_list(hadm_ids), "codes": SEPSIS_ICD9})["hadm_id"].values
    labels = pd.Series(0, index=hadm_ids, name="label", dtype=int)
    labels[pos] = 1
    labels.index.name = "hadm_id"
    logger.info("  Sepsis: %.3f (%d/%d)", labels.mean(), labels.sum(), len(labels))
    return labels


def get_mortality_labels(cohort, hadm_ids):
    mort = cohort.drop_duplicates("hadm_id").set_index("hadm_id")["hospital_expire_flag"]
    labels = mort.reindex(hadm_ids).fillna(0).astype(int)
    labels.name = "label"
    labels.index.name = "hadm_id"
    logger.info("  Mortality: %.3f (%d/%d)", labels.mean(), labels.sum(), len(labels))
    return labels


def get_vasopressor_labels(conn, hadm_ids, cohort):
    """Stay-level, time-windowed vasopressor task (matches Deferral Frontier's
    MIMIC-IV definition exactly): new vasopressor initiation after 24h through
    ICU discharge, among patients who had NOT yet received vasopressors by 24h.

    The prior version of this function checked "any vasopressor ever" at the
    subject level with no time windowing at all -- identified as the likely cause of the inflated
    62.7% MIMIC-III vasopressor prevalence and the elevated Jaccard (0.142)
    on that task relative to sepsis/mortality (<0.001).
    """
    q = f"""
    SELECT icustay_id, charttime AS event_time FROM {DB_SCHEMA}.inputevents_cv
    WHERE itemid = ANY(%(cv)s) AND icustay_id IS NOT NULL
    UNION ALL
    SELECT icustay_id, starttime AS event_time FROM {DB_SCHEMA}.inputevents_mv
    WHERE itemid = ANY(%(mv)s) AND icustay_id IS NOT NULL
    """
    events = pd.read_sql(q, conn, params={"cv": VASO_IDS_CV, "mv": VASO_IDS_MV})

    stay_meta = cohort.drop_duplicates("hadm_id").set_index("hadm_id")[
        ["stay_id", "admission_time", "prediction_time", "outtime"]
    ].reindex(hadm_ids)

    ev = events.merge(
        stay_meta.reset_index(), left_on="icustay_id", right_on="stay_id", how="inner"
    )
    early = ev[ev["event_time"] < ev["prediction_time"]]["hadm_id"].unique()
    late = ev[
        (ev["event_time"] >= ev["prediction_time"]) & (ev["event_time"] <= ev["outtime"])
    ]["hadm_id"].unique()

    eligible = stay_meta.index.difference(pd.Index(early))
    labels = pd.Series(0, index=eligible, name="label", dtype=int)
    labels.loc[labels.index.isin(late)] = 1
    labels.index.name = "hadm_id"
    logger.info(
        "  Vasopressor: %.3f (%d/%d eligible; %d/%d excluded for pre-24h vasopressor use)",
        labels.mean(), labels.sum(), len(labels), len(early), len(hadm_ids),
    )
    return labels


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = get_conn()

    cohort = extract_cohort(conn)

    # Keep first ICU stay per admission (consistent with MIMIC-IV hadm_id index)
    cohort = cohort.sort_values("admission_time").drop_duplicates("hadm_id")
    hadm_ids = cohort["hadm_id"].unique()
    stay_ids = cohort["stay_id"].unique()
    logger.info("After dedup: %d admissions", len(cohort))

    vitals_ts = extract_vitals(conn, stay_ids)
    labs_ts = extract_labs(conn, hadm_ids)

    logger.info("Aggregating vitals...")
    vital_feat = vectorized_aggregate(vitals_ts, "stay_id", cohort)
    logger.info("  Vitals features: %s", vital_feat.shape)

    stay_to_hadm = cohort.set_index("stay_id")["hadm_id"]
    vital_feat["hadm_id"] = vital_feat.index.map(stay_to_hadm)

    logger.info("Aggregating labs...")
    lab_feat = vectorized_aggregate(labs_ts, "hadm_id", cohort)

    hadm_to_stay = cohort.drop_duplicates("hadm_id").set_index("hadm_id")["stay_id"]
    lab_feat["stay_id"] = lab_feat.index.map(hadm_to_stay)

    merged = pd.merge(vital_feat, lab_feat, on="hadm_id", how="outer", suffixes=("", "_lab"))
    merged = merged.set_index("hadm_id")
    count_cols = [c for c in merged.columns if c.endswith("_count")]
    merged[count_cols] = merged[count_cols].fillna(0)

    hadm_common = merged.index.unique()

    # Build a single combined timeseries keyed by hadm_id (NOT icustay_id) in
    # the "stay_id" column. compute_d2_fast/compute_d4_fast
    # (deferral_frontier/src/fast_reliability.py) merge this "stay_id" column
    # against the hadm_id-indexed features/meta, so the column must actually
    # contain hadm_id values despite its name. MIMIC-III's hadm_id and
    # icustay_id occupy disjoint numeric ranges (verified: 100001-199999 vs
    # 200001-299999 in this deployment); using raw icustay_id here silently
    # produces a zero-row merge for every patient, so compute_d2_fast/
    # compute_d4_fast fall back to their constant defaults (D2=0.0, D4=0.5)
    # for the entire cohort -- a bug that predates this fix and was present
    # in previously reported MIMIC-III D2/D4 results.
    vitals_ts_h = vitals_ts.assign(hadm_id=vitals_ts["stay_id"].map(stay_to_hadm)).drop(
        columns=["stay_id"]
    )
    # labs_ts is already hadm_id-keyed (no icustay_id column to drop).
    ts_out = pd.concat([vitals_ts_h, labs_ts], ignore_index=True)
    ts_out = ts_out.rename(columns={"hadm_id": "stay_id"})[
        ["stay_id", "charttime", "feature_name", "value"]
    ]
    ts_out.to_parquet(OUTPUT_DIR / "timeseries.parquet")
    logger.info("Saved combined timeseries: %d rows", len(ts_out))

    for task, label_fn in [
        ("sepsis", lambda: get_sepsis_labels(conn, hadm_common)),
        ("mortality", lambda: get_mortality_labels(cohort, hadm_common)),
        ("vasopressor", lambda: get_vasopressor_labels(conn, hadm_common, cohort)),
    ]:
        logger.info("Task: %s", task)
        labels = label_fn()
        feat_cols = [c for c in merged.columns if not c.endswith("_count_lab")
                     and c not in ("stay_id", "stay_id_lab")]
        feat = merged[feat_cols].reindex(labels.index)

        meta_cols = cohort.drop_duplicates("hadm_id").set_index("hadm_id")
        meta = meta_cols.reindex(labels.index)[["admission_time", "prediction_time", "admit_year"]]

        feat.to_parquet(OUTPUT_DIR / f"{task}_features.parquet")
        labels.to_frame().to_parquet(OUTPUT_DIR / f"{task}_labels.parquet")
        meta.to_parquet(OUTPUT_DIR / f"{task}_meta.parquet")

        task_ts = ts_out[ts_out["stay_id"].isin(labels.index)]
        task_ts.to_parquet(OUTPUT_DIR / f"{task}_timeseries.parquet")

        logger.info("  Saved %s: %d patients, %d features, %d timeseries rows",
                     task, len(feat), feat.shape[1], len(task_ts))

    conn.close()
    logger.info("DONE.")


if __name__ == "__main__":
    main()
