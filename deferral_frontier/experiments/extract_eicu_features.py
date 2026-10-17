"""Extract eICU features mirroring the MIMIC-III/MIMIC-IV pipeline (vectorized).

Extracts ICU cohorts with the same 23 raw features (10 vitals + 13 labs) as
extract_mimic3_features.py, aggregated identically (mean, min, max, last, count
within 24h), for genuine cross-institutional external validation (eICU spans
~200 US hospitals, distinct from BIDMC where MIMIC-III/IV originate).

Schema notes (eICU-CRD, confirmed against this deployment via
Paper_7th_math/src/data/eicu_feature_extractor.py):
  - Primary key: patientunitstayid (not subject_id/hadm_id)
  - Time offsets are in MINUTES from unit admission, not timestamps
  - Vitals: vitalperiodic (heartrate, systemicsystolic/diastolic, respiration,
    temperature, sao2) + vitalaperiodic (noninvasivesystolic/diastolic) for
    patients without an arterial line
  - Labs: `lab` table, free-text `labname` (not itemid); confirmed values for
    this deployment include 'creatinine', 'potassium', 'sodium', 'BUN', 'pH',
    'chloride', 'glucose', 'Hgb'; others (lactate, platelets, ALT, AST,
    bicarbonate) use ILIKE pattern matching since exact strings are
    unconfirmed for this deployment -- see --validate-schema.
  - Diagnoses: free-text `diagnosisstring` (no ICD codes) -- sepsis identified
    via ILIKE '%sepsis%' OR '%septic%', consistent with the ILIKE approach
    already used for other conditions in eicu_feature_extractor.py.
  - Vasopressors: free-text `drugname` ILIKE match on norepinephrine,
    dopamine, vasopressin, epinephrine, phenylephrine (same drug list as
    eicu_feature_extractor.py's vasopressor_orders feature).
  - Age: '> 89' for nonagenarians+, else numeric string.

Run `python extract_eicu_features.py --validate-schema` FIRST after the
workstation/DB is available -- this prints row counts and sample matches for
every ILIKE pattern without running the full extraction, so unconfirmed lab
name / diagnosis / drug patterns can be checked before committing compute.
"""
import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DB_HOST = os.getenv("EICU_DB_HOST", "localhost")
DB_NAME = os.getenv("EICU_DB_NAME", "eicu")
DB_USER = os.getenv("EICU_DB_USER", "postgres")
DB_PASS = os.getenv("EICU_DB_PASSWORD", "")
DB_PORT = os.getenv("EICU_DB_PORT", "5432")
DB_SCHEMA = os.getenv("EICU_DB_SCHEMA", "eicu_crd")

OUTPUT_DIR = Path("data/eicu")

# 24h window in minutes (eICU offsets are minutes from unit admission)
WINDOW_MIN = 24 * 60
BASELINE_MIN = 6 * 60

MIN_LOS_MIN = 24 * 60  # require >=24h ICU stay so a 24h prediction point exists
MIN_AGE, MAX_AGE = 18, 100

# Vasopressor drug-name patterns (matches eicu_feature_extractor.py's vasopressor_orders)
VASO_PATTERNS = ["norepinephrine", "dopamine", "vasopressin", "epinephrine", "phenylephrine"]

# Lab name matches. CONFIRMED = exact match verified against this deployment.
# UNCONFIRMED = ILIKE pattern, standard eICU-CRD naming convention, needs
# --validate-schema check before full extraction.
LAB_MATCH = {
    "lab_creatinine":  ("exact", "creatinine"),
    "lab_sodium":      ("exact", "sodium"),
    "lab_potassium":   ("exact", "potassium"),
    "lab_bun":         ("exact", "BUN"),
    "lab_glucose":     ("exact", "glucose"),
    "lab_hemoglobin":  ("exact", "Hgb"),
    "lab_bilirubin":   ("exact", "total bilirubin"),
    "lab_wbc":         ("exact", "WBC x 1000"),
    "lab_lactate":     ("exact", "lactate"),
    "lab_platelet":    ("exact", "platelets x 1000"),
    "lab_alt":         ("exact", "ALT (SGPT)"),
    "lab_ast":         ("exact", "AST (SGOT)"),
    "lab_bicarbonate": ("exact", "bicarbonate"),
}
UNCONFIRMED_LABS = [k for k, (kind, _) in LAB_MATCH.items() if kind == "ilike"]


def get_conn():
    import psycopg2
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASS,
    )


def validate_schema(conn):
    """Print distinct labname matches / diagnosis+drug sample hits without extracting.

    Run this before the full extraction to confirm every UNCONFIRMED_LABS
    pattern actually matches sensible lab names in this deployment, and that
    sepsis / vasopressor text patterns aren't matching zero rows.
    """
    cur = conn.cursor()

    print("=" * 60)
    print("LAB NAME VALIDATION")
    print("=" * 60)
    for feat, (kind, pattern) in LAB_MATCH.items():
        if kind == "exact":
            cur.execute(f"SELECT COUNT(*) FROM {DB_SCHEMA}.lab WHERE labname = %s", (pattern,))
        else:
            cur.execute(f"SELECT COUNT(*) FROM {DB_SCHEMA}.lab WHERE labname ILIKE %s", (pattern,))
        n = cur.fetchone()[0]
        flag = "  [UNCONFIRMED PATTERN]" if kind == "ilike" else ""
        print(f"  {feat:20s} {kind:6s} {pattern!r:20s} -> {n:>10,} rows{flag}")
        if kind == "ilike":
            cur.execute(
                f"SELECT DISTINCT labname FROM {DB_SCHEMA}.lab WHERE labname ILIKE %s LIMIT 10",
                (pattern,),
            )
            for (name,) in cur.fetchall():
                print(f"      matched labname: {name!r}")

    print("\n" + "=" * 60)
    print("SEPSIS DIAGNOSIS VALIDATION")
    print("=" * 60)
    cur.execute(
        f"SELECT COUNT(DISTINCT patientunitstayid) FROM {DB_SCHEMA}.diagnosis "
        f"WHERE diagnosisstring ILIKE '%sepsis%' OR diagnosisstring ILIKE '%septic%'"
    )
    print(f"  Distinct stays with sepsis/septic diagnosis: {cur.fetchone()[0]:,}")
    cur.execute(
        f"SELECT DISTINCT diagnosisstring FROM {DB_SCHEMA}.diagnosis "
        f"WHERE diagnosisstring ILIKE '%sepsis%' OR diagnosisstring ILIKE '%septic%' LIMIT 15"
    )
    for (s,) in cur.fetchall():
        print(f"      {s!r}")

    print("\n" + "=" * 60)
    print("VASOPRESSOR DRUG NAME VALIDATION")
    print("=" * 60)
    for p in VASO_PATTERNS:
        cur.execute(
            f"SELECT COUNT(DISTINCT patientunitstayid) FROM {DB_SCHEMA}.medication "
            f"WHERE drugname ILIKE %s",
            (f"%{p}%",),
        )
        print(f"  {p:20s} -> {cur.fetchone()[0]:>10,} distinct stays")

    print("\n" + "=" * 60)
    print("COHORT SIZE ESTIMATE")
    print("=" * 60)
    cur.execute(f"""
        SELECT COUNT(*) FROM {DB_SCHEMA}.patient
        WHERE unitdischargeoffset > {MIN_LOS_MIN}
          AND (age ~ '^[0-9]+$' OR age = '> 89')
          AND (CASE WHEN age = '> 89' THEN 90 WHEN age ~ '^[0-9]+$' THEN age::numeric ELSE NULL END)
              BETWEEN {MIN_AGE} AND {MAX_AGE}
    """)
    print(f"  Eligible ICU stays (LOS > 24h, age {MIN_AGE}-{MAX_AGE}): {cur.fetchone()[0]:,}")
    cur.close()


def extract_cohort(conn, limit: int = None, seed: int = 42):
    logger.info("Extracting eICU cohort...")
    limit_clause = f"ORDER BY RANDOM() LIMIT {limit}" if limit else ""
    q = f"""
    SELECT
        patientunitstayid AS stay_id,
        patientunitstayid AS hadm_id,
        uniquepid AS subject_id,
        0 AS admission_offset,
        {WINDOW_MIN} AS prediction_offset,
        hospitaldischargeyear::int AS admit_year,
        CASE WHEN hospitaldischargestatus = 'Expired' THEN 1 ELSE 0 END AS hospital_expire_flag
    FROM {DB_SCHEMA}.patient
    WHERE unitdischargeoffset > {MIN_LOS_MIN}
      AND (age ~ '^[0-9]+$' OR age = '> 89')
      AND (CASE WHEN age = '> 89' THEN 90 WHEN age ~ '^[0-9]+$' THEN age::numeric ELSE NULL END)
          BETWEEN {MIN_AGE} AND {MAX_AGE}
    {limit_clause}
    """
    df = pd.read_sql(q, conn)
    logger.info("  %d ICU stays (LOS > 24h)", len(df))
    return df


def _int_list(arr):
    return [int(x) for x in arr]


def extract_vitals(conn, stay_ids):
    """Long-format vitals: hr, resp_rate, sbp/dbp/map (invasive+noninvasive), temp.

    Confirmed against information_schema for this deployment:
    vitalperiodic.systemicmean and vitalaperiodic.noninvasivemean both exist,
    so abp_mean/nibp_mean are included (9 of MIMIC's 10 VITAL_MAP signals).
    temp_f has no eICU equivalent -- vitalperiodic.temperature is the only
    temperature column (Celsius), so vital_temp_f is intentionally omitted
    rather than guessed; this is disclosed as a methodological difference,
    not silently dropped.
    """
    logger.info("Extracting vitals for %d stays...", len(stay_ids))
    stay_list = _int_list(stay_ids)
    batch = 2000
    chunks = []
    for i in range(0, len(stay_list), batch):
        ids = stay_list[i:i + batch]
        q = f"""
        SELECT patientunitstayid AS stay_id, observationoffset AS offset_min,
               'vital_heart_rate' AS feature_name, heartrate AS value
        FROM {DB_SCHEMA}.vitalperiodic
        WHERE patientunitstayid = ANY(%(s)s) AND observationoffset BETWEEN 0 AND {WINDOW_MIN}
          AND heartrate > 0
        UNION ALL
        SELECT patientunitstayid, observationoffset, 'vital_resp_rate', respiration
        FROM {DB_SCHEMA}.vitalperiodic
        WHERE patientunitstayid = ANY(%(s)s) AND observationoffset BETWEEN 0 AND {WINDOW_MIN}
          AND respiration > 0
        UNION ALL
        SELECT patientunitstayid, observationoffset, 'vital_abp_sys', systemicsystolic
        FROM {DB_SCHEMA}.vitalperiodic
        WHERE patientunitstayid = ANY(%(s)s) AND observationoffset BETWEEN 0 AND {WINDOW_MIN}
          AND systemicsystolic > 0
        UNION ALL
        SELECT patientunitstayid, observationoffset, 'vital_abp_dia', systemicdiastolic
        FROM {DB_SCHEMA}.vitalperiodic
        WHERE patientunitstayid = ANY(%(s)s) AND observationoffset BETWEEN 0 AND {WINDOW_MIN}
          AND systemicdiastolic > 0
        UNION ALL
        SELECT patientunitstayid, observationoffset, 'vital_abp_mean', systemicmean
        FROM {DB_SCHEMA}.vitalperiodic
        WHERE patientunitstayid = ANY(%(s)s) AND observationoffset BETWEEN 0 AND {WINDOW_MIN}
          AND systemicmean > 0
        UNION ALL
        SELECT patientunitstayid, observationoffset, 'vital_temp_c', temperature
        FROM {DB_SCHEMA}.vitalperiodic
        WHERE patientunitstayid = ANY(%(s)s) AND observationoffset BETWEEN 0 AND {WINDOW_MIN}
          AND temperature BETWEEN 20 AND 45
        UNION ALL
        SELECT patientunitstayid, observationoffset, 'vital_nibp_sys', noninvasivesystolic
        FROM {DB_SCHEMA}.vitalaperiodic
        WHERE patientunitstayid = ANY(%(s)s) AND observationoffset BETWEEN 0 AND {WINDOW_MIN}
          AND noninvasivesystolic > 0
        UNION ALL
        SELECT patientunitstayid, observationoffset, 'vital_nibp_dia', noninvasivediastolic
        FROM {DB_SCHEMA}.vitalaperiodic
        WHERE patientunitstayid = ANY(%(s)s) AND observationoffset BETWEEN 0 AND {WINDOW_MIN}
          AND noninvasivediastolic > 0
        UNION ALL
        SELECT patientunitstayid, observationoffset, 'vital_nibp_mean', noninvasivemean
        FROM {DB_SCHEMA}.vitalaperiodic
        WHERE patientunitstayid = ANY(%(s)s) AND observationoffset BETWEEN 0 AND {WINDOW_MIN}
          AND noninvasivemean > 0
        """
        chunks.append(pd.read_sql(q, conn, params={"s": ids}))
        logger.info("    batch %d/%d", i // batch + 1, (len(stay_list) - 1) // batch + 1)
    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(
        columns=["stay_id", "offset_min", "feature_name", "value"])
    logger.info("  %d vital rows", len(df))
    return df.dropna(subset=["value"])


def extract_labs(conn, stay_ids):
    logger.info("Extracting labs for %d stays...", len(stay_ids))
    stay_list = _int_list(stay_ids)
    batch = 2000
    chunks = []
    for feat, (kind, pattern) in LAB_MATCH.items():
        op = "=" if kind == "exact" else "ILIKE"
        for i in range(0, len(stay_list), batch):
            ids = stay_list[i:i + batch]
            q = f"""
            SELECT patientunitstayid AS stay_id, labresultoffset AS offset_min,
                   %(feat)s AS feature_name, labresult AS value
            FROM {DB_SCHEMA}.lab
            WHERE patientunitstayid = ANY(%(s)s)
              AND labresultoffset BETWEEN 0 AND {WINDOW_MIN}
              AND labresult IS NOT NULL AND labresult > 0
              AND labname {op} %(pattern)s
            """
            chunks.append(pd.read_sql(q, conn, params={"s": ids, "feat": feat, "pattern": pattern}))
        logger.info("  extracted %s (%s %r)", feat, kind, pattern)
    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(
        columns=["stay_id", "offset_min", "feature_name", "value"])
    logger.info("  %d lab rows total", len(df))
    return df


def vectorized_aggregate(ts_df):
    """mean/min/max/last/count within [0, WINDOW_MIN], keyed by stay_id."""
    logger.info("Vectorized aggregation on %d rows...", len(ts_df))
    if ts_df.empty:
        return pd.DataFrame()
    ts_df = ts_df.sort_values("offset_min")
    agg = (ts_df.groupby(["stay_id", "feature_name"])["value"]
           .agg(mean="mean", mi="min", ma="max", last="last", count="count")
           .reset_index())
    rows = []
    for _, row in agg.iterrows():
        rows.append({
            "stay_id": row["stay_id"],
            f"{row['feature_name']}_mean": row["mean"],
            f"{row['feature_name']}_min": row["mi"],
            f"{row['feature_name']}_max": row["ma"],
            f"{row['feature_name']}_last": row["last"],
            f"{row['feature_name']}_count": row["count"],
        })
    long = pd.DataFrame(rows)
    wide = long.groupby("stay_id").first()
    for col in long.columns:
        if col != "stay_id":
            wide[col] = long.groupby("stay_id")[col].first()
    return wide


def get_sepsis_labels(conn, stay_ids):
    q = f"""
    SELECT DISTINCT patientunitstayid FROM {DB_SCHEMA}.diagnosis
    WHERE patientunitstayid = ANY(%(s)s)
      AND (diagnosisstring ILIKE '%%sepsis%%' OR diagnosisstring ILIKE '%%septic%%')
    """
    pos = pd.read_sql(q, conn, params={"s": _int_list(stay_ids)})["patientunitstayid"].values
    labels = pd.Series(0, index=stay_ids, name="label", dtype=int)
    labels.loc[labels.index.isin(pos)] = 1
    labels.index.name = "stay_id"
    logger.info("  Sepsis: %.3f (%d/%d)", labels.mean(), labels.sum(), len(labels))
    return labels


def get_mortality_labels(cohort, stay_ids):
    mort = cohort.drop_duplicates("stay_id").set_index("stay_id")["hospital_expire_flag"]
    labels = mort.reindex(stay_ids).fillna(0).astype(int)
    labels.name = "label"
    labels.index.name = "stay_id"
    logger.info("  Mortality: %.3f (%d/%d)", labels.mean(), labels.sum(), len(labels))
    return labels


def get_vasopressor_labels(conn, stay_ids):
    # Literal '%' in the ILIKE patterns must be escaped as '%%' -- psycopg2
    # otherwise misreads them as stray parameter-style markers alongside the
    # %(s)s named placeholder ("argument formats can't be mixed").
    pattern = " OR ".join(f"drugname ILIKE '%%{p}%%'" for p in VASO_PATTERNS)
    q = f"""
    SELECT DISTINCT patientunitstayid FROM {DB_SCHEMA}.medication
    WHERE patientunitstayid = ANY(%(s)s) AND ({pattern})
    """
    pos = pd.read_sql(q, conn, params={"s": _int_list(stay_ids)})["patientunitstayid"].values
    labels = pd.Series(0, index=stay_ids, name="label", dtype=int)
    labels.loc[labels.index.isin(pos)] = 1
    labels.index.name = "stay_id"
    logger.info("  Vasopressor: %.3f (%d/%d)", labels.mean(), labels.sum(), len(labels))
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-schema", action="store_true",
                         help="Print schema/pattern validation and exit (no extraction).")
    parser.add_argument("--limit", type=int, default=None,
                         help="Optional cohort size cap (default: all eligible stays).")
    args = parser.parse_args()

    conn = get_conn()

    if args.validate_schema:
        if UNCONFIRMED_LABS:
            logger.warning("Unconfirmed lab patterns before validation: %s", UNCONFIRMED_LABS)
        validate_schema(conn)
        conn.close()
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cohort = extract_cohort(conn, limit=args.limit)
    stay_ids = cohort["stay_id"].unique()

    vitals_ts = extract_vitals(conn, stay_ids)
    labs_ts = extract_labs(conn, stay_ids)

    logger.info("Aggregating vitals + labs...")
    ts_all = pd.concat([vitals_ts, labs_ts], ignore_index=True)
    feat = vectorized_aggregate(ts_all)
    count_cols = [c for c in feat.columns if c.endswith("_count")]
    feat[count_cols] = feat[count_cols].fillna(0)

    stay_common = feat.index.unique()

    for task, label_fn in [
        ("sepsis", lambda: get_sepsis_labels(conn, stay_common)),
        ("mortality", lambda: get_mortality_labels(cohort, stay_common)),
        ("vasopressor", lambda: get_vasopressor_labels(conn, stay_common)),
    ]:
        logger.info("Task: %s", task)
        labels = label_fn()
        f = feat.reindex(labels.index)

        meta_cols = cohort.drop_duplicates("stay_id").set_index("stay_id")
        meta = meta_cols.reindex(labels.index)[["admit_year"]].copy()
        # eICU has no wall-clock timestamps; offsets are relative to unit
        # admission (0 = admission, WINDOW_MIN = 24h prediction point).
        # compute_d2_fast/compute_d4_fast (deferral_frontier/src/fast_reliability.py)
        # add pd.Timedelta directly to admission/prediction times and compare
        # against timeseries charttime, so all three must be real Timestamps,
        # not raw integers -- anchor everyone to a shared arbitrary epoch.
        # This is semantically equivalent to using the integer offsets
        # directly since eICU offsets are already relative-to-admission.
        EPOCH = pd.Timestamp("2000-01-01")
        meta["admission_time"] = EPOCH
        meta["prediction_time"] = EPOCH + pd.Timedelta(minutes=WINDOW_MIN)

        f.to_parquet(OUTPUT_DIR / f"{task}_features.parquet")
        labels.to_frame().to_parquet(OUTPUT_DIR / f"{task}_labels.parquet")
        meta.to_parquet(OUTPUT_DIR / f"{task}_meta.parquet")
        logger.info("  Saved %s: %d patients, %d features", task, len(f), f.shape[1])

        ts_task = ts_all[ts_all["stay_id"].isin(labels.index)].rename(
            columns={"offset_min": "charttime"})
        ts_task["charttime"] = EPOCH + pd.to_timedelta(ts_task["charttime"], unit="min")
        ts_task[["stay_id", "charttime", "feature_name", "value"]].to_parquet(
            OUTPUT_DIR / f"{task}_timeseries.parquet")

    conn.close()
    logger.info("DONE. Remember: %d lab features used ILIKE pattern matching "
                "(%s) -- re-run --validate-schema to sanity-check these before "
                "trusting downstream results.", len(UNCONFIRMED_LABS), UNCONFIRMED_LABS)


if __name__ == "__main__":
    main()
