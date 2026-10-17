"""Independent verification of the round-1 re-extraction and reanalysis.

Each check recomputes a quantity by a route that does not reuse the pipeline
code being checked (plain SQL in the database, or direct numpy on artifacts),
then compares it with the pipeline output.

Output: plan/r1-verification.md (markdown table of checks)

Usage:
    python -m deferral_frontier.experiments.r1_verify_pipeline
"""
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from deferral_frontier.experiments.extract_patient_ids import _connect, _load_env

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "deferral_frontier" / "results"
ART = ROOT / "artifacts"
Check = Tuple[str, str, str, bool]  # (area, check, detail, passed)

COHORT_SQL = {
    "mimic4": ("MIMICIV_DB", """
        WITH s AS (
          SELECT i.hadm_id, i.stay_id, i.intime, i.outtime, a.hospital_expire_flag,
                 ROW_NUMBER() OVER (PARTITION BY i.hadm_id ORDER BY i.intime) AS rn
          FROM mimic_icu.icustays i
          JOIN mimic_core.admissions a ON a.hadm_id = i.hadm_id
          JOIN mimic_core.patients p ON p.subject_id = i.subject_id
          WHERE i.los >= 1 AND p.anchor_age + (EXTRACT(YEAR FROM i.intime) - p.anchor_year) >= 18)
        SELECT hadm_id, stay_id, intime, outtime, hospital_expire_flag FROM s WHERE rn = 1"""),
    "mimic3": ("MIMIC3", """
        WITH s AS (
          SELECT i.hadm_id, i.icustay_id AS stay_id, i.intime, i.outtime, a.hospital_expire_flag,
                 ROW_NUMBER() OVER (PARTITION BY i.hadm_id ORDER BY i.intime) AS rn
          FROM mimiciii.icustays i
          JOIN mimiciii.admissions a ON a.hadm_id = i.hadm_id
          JOIN mimiciii.patients p ON p.subject_id = i.subject_id
          WHERE i.los >= 1 AND EXTRACT(EPOCH FROM i.intime - p.dob) / 31557600.0 >= 18)
        SELECT hadm_id, stay_id, intime, outtime, hospital_expire_flag FROM s WHERE rn = 1"""),
}
SEPSIS_SQL = {
    "mimic4": """SELECT DISTINCT hadm_id FROM mimic_hosp.diagnoses_icd WHERE hadm_id = ANY(%(h)s) AND
                 ((icd_version = 9 AND icd_code IN ('99591','99592','78552')) OR
                  (icd_version = 10 AND icd_code IN ('A419','R6520','R6521')))""",
    "mimic3": """SELECT DISTINCT hadm_id FROM mimiciii.diagnoses_icd WHERE hadm_id = ANY(%(h)s)
                 AND icd9_code IN ('99591','99592','78552')""",
}
VASO_SQL = {  # per stay: first vasopressor time (all five drugs), via SQL only
    "mimic4": """SELECT stay_id, min(starttime) FROM mimic_icu.inputevents
                 WHERE stay_id = ANY(%(s)s) AND itemid IN (221906,221289,222315,221749,221662)
                 GROUP BY stay_id""",
    "mimic3": """SELECT icustay_id, min(t) FROM (
                   SELECT icustay_id, starttime t FROM mimiciii.inputevents_mv WHERE icustay_id = ANY(%(s)s)
                     AND itemid IN (221906,221289,222315,221749,221662)
                   UNION ALL
                   SELECT icustay_id, charttime FROM mimiciii.inputevents_cv WHERE icustay_id = ANY(%(s)s)
                     AND itemid IN (30047,30120,30044,30119,30309,30051,30127,30128,30043,30307)) x
                 GROUP BY icustay_id""",
}
HR_SQL = {  # heart rate mean/count in [intime, intime+24h) for a sample of stays
    "mimic4": """SELECT c.stay_id, avg(valuenum), count(*) FROM mimic_icu.chartevents c
                 JOIN mimic_icu.icustays i ON i.stay_id = c.stay_id
                 WHERE c.stay_id = ANY(%(s)s) AND itemid = 220045 AND valuenum > 0
                   AND charttime >= i.intime AND charttime < i.intime + interval '24 hours'
                 GROUP BY c.stay_id""",
    "mimic3": """SELECT c.icustay_id, avg(valuenum), count(*) FROM mimiciii.chartevents c
                 JOIN mimiciii.icustays i ON i.icustay_id = c.icustay_id
                 WHERE c.icustay_id = ANY(%(s)s) AND itemid IN (211, 220045) AND valuenum > 0
                   AND c.error IS DISTINCT FROM 1
                   AND charttime >= i.intime AND charttime < i.intime + interval '24 hours'
                 GROUP BY c.icustay_id""",
}


def _fetch(cur, sql: str, params: dict, cols: List[str]) -> pd.DataFrame:
    cur.execute(sql, params)
    return pd.DataFrame(cur.fetchall(), columns=cols)


def check_database(db: str, env: dict) -> List[Check]:
    """Cohort, labels, features, and time windows recomputed in plain SQL."""
    out: List[Check] = []
    prefix, csql = COHORT_SQL[db]
    data = ROOT / "data" / db
    lab = {t: pd.read_parquet(data / f"{t}_labels.parquet")["label"] for t in ("sepsis", "mortality", "vasopressor")}
    feats = pd.read_parquet(data / "mortality_features.parquet")
    meta = pd.read_parquet(data / "mortality_meta.parquet")
    with _connect(env, prefix) as con, con.cursor() as cur:
        co = _fetch(cur, csql, {}, ["hadm_id", "stay_id", "intime", "outtime", "mort"]).set_index("hadm_id")
        rows = lab["mortality"].index
        out.append((db, "cohort ⊆ SQL cohort (adult, LOS≥24h, first stay)",
                    f"SQL {len(co):,}; pipeline {len(rows):,} (after ≥1 observation filter)",
                    rows.isin(co.index).all() and len(co) - len(rows) < 0.01 * len(co)))
        co = co.loc[rows]
        out.append((db, "prediction time = ICU intime + 24 h",
                    "meta.prediction_time vs SQL intime",
                    bool((meta.loc[rows, "prediction_time"] == co["intime"] + pd.Timedelta(hours=24)).all())))
        mism = int((co["mort"].astype(int) != lab["mortality"]).sum())
        out.append((db, "mortality label = hospital_expire_flag", f"{mism} mismatches", mism == 0))
        pos = set(_fetch(cur, SEPSIS_SQL[db], {"h": [int(h) for h in rows]}, ["h"])["h"])
        mism = int((lab["sepsis"] != lab["sepsis"].index.isin(pos).astype(int)).sum())
        out.append((db, "sepsis label = SQL ICD query", f"{mism} mismatches; prevalence {lab['sepsis'].mean():.3f}", mism == 0))
        first = _fetch(cur, VASO_SQL[db], {"s": [int(s) for s in co["stay_id"]]}, ["stay_id", "t"]).set_index("stay_id")["t"]
        ft = co["stay_id"].map(first)
        pred = co["intime"] + pd.Timedelta(hours=24)
        eligible = ft.isna() | (ft >= pred)
        y_sql = (ft.notna() & (ft >= pred) & (ft <= co["outtime"])).astype(int)[eligible]
        v = lab["vasopressor"]
        same_rows = set(v.index) == set(y_sql.index)
        mism = int((v != y_sql.reindex(v.index)).sum()) if same_rows else -1
        out.append((db, "vasopressor cohort and label = SQL (first dose ≥ 24 h)",
                    f"eligible SQL {len(y_sql):,} vs pipeline {len(v):,}; {mism} label mismatches; prevalence {v.mean():.3f}",
                    same_rows and mism == 0))
        sample = co.sample(300, random_state=0)
        hr = _fetch(cur, HR_SQL[db], {"s": [int(s) for s in sample["stay_id"]]}, ["stay_id", "mean", "n"]).set_index("stay_id")
        f = feats.loc[sample.index]
        sql_mean = sample["stay_id"].map(hr["mean"]).astype(float)
        sql_n = sample["stay_id"].map(hr["n"]).fillna(0).astype(int)
        ok_n = (sql_n.values == f["vital_heart_rate_count"].values).mean()
        ok_m = np.isclose(sql_mean.values, f["vital_heart_rate_mean"].values, equal_nan=True).mean()
        out.append((db, "heart-rate features = SQL (300 random admissions)",
                    f"count match {ok_n:.1%}, mean match {ok_m:.1%}", ok_n > 0.99 and ok_m > 0.99))
    ts = pd.read_parquet(data / "mortality_timeseries.parquet", columns=["stay_id", "charttime"])
    m = ts.join(meta, on="stay_id")
    inside = ((m["charttime"] >= m["admission_time"]) & (m["charttime"] < m["prediction_time"])).mean()
    out.append((db, "no observation after prediction time", f"{inside:.4%} of {len(ts):,} rows inside window", inside == 1.0))
    return out


def check_eicu(env: dict) -> List[Check]:
    """eICU cohort, labels, and held-out-hospital split recomputed in plain SQL."""
    out: List[Check] = []
    data = ROOT / "data" / "eicu"
    lab = {t: pd.read_parquet(data / f"{t}_labels.parquet")["label"] for t in ("sepsis", "mortality", "vasopressor")}
    rows = lab["mortality"].index
    vaso = "(norepinephrine|levophed|dopamine|vasopressin|epinephrine|phenylephrine|neo-?synephrine)"
    with _connect(env, "EICU_DB") as con, con.cursor() as cur:
        co = _fetch(cur, """
            WITH s AS (SELECT patientunitstayid, hospitalid, unitdischargeoffset, hospitaldischargestatus,
                              ROW_NUMBER() OVER (PARTITION BY patienthealthsystemstayid ORDER BY unitvisitnumber) rn
                       FROM eicu_crd.patient
                       WHERE unitdischargeoffset >= 1440 AND hospitaldischargestatus IN ('Alive','Expired')
                         AND (age = '> 89' OR (age ~ '^[0-9]+$' AND age::int >= 18)))
            SELECT patientunitstayid, hospitalid, unitdischargeoffset, hospitaldischargestatus FROM s WHERE rn = 1""",
            {}, ["stay_id", "hospital", "disch", "status"]).set_index("stay_id")
        out.append(("eicu", "cohort ⊆ SQL cohort (first eligible unit stay per hospitalization)",
                    f"SQL {len(co):,}; pipeline {len(rows):,}", rows.isin(co.index).all() and len(co) - len(rows) < 0.01 * len(co)))
        co = co.loc[rows]
        mism = int(((co["status"] == "Expired").astype(int) != lab["mortality"]).sum())
        out.append(("eicu", "mortality label = discharge status", f"{mism} mismatches", mism == 0))
        pos = set(_fetch(cur, r"SELECT DISTINCT patientunitstayid FROM eicu_crd.diagnosis WHERE patientunitstayid = ANY(%(s)s) "
                              r"AND diagnosisstring ~* '\|(sepsis|septic shock)(\||$)'",
                         {"s": [int(i) for i in rows]}, ["s"])["s"])
        mism = int((lab["sepsis"] != lab["sepsis"].index.isin(pos).astype(int)).sum())
        out.append(("eicu", "sepsis label = SQL diagnosis-string query", f"{mism} mismatches; prevalence {lab['sepsis'].mean():.3f}", mism == 0))
        cur.execute(f"SELECT DISTINCT p.hospitalid FROM eicu_crd.infusiondrug i JOIN eicu_crd.patient p USING (patientunitstayid) WHERE i.drugname ~* '{vaso}'")
        charting = {r[0] for r in cur.fetchall()}
        vc = co[co["hospital"].isin(charting)]
        ids = [int(i) for i in vc.index]
        first_any = _fetch(cur, f"""SELECT patientunitstayid, min(o) FROM (
              SELECT patientunitstayid, infusionoffset o FROM eicu_crd.infusiondrug WHERE patientunitstayid = ANY(%(s)s) AND drugname ~* '{vaso}'
              UNION ALL SELECT patientunitstayid, drugstartoffset FROM eicu_crd.medication WHERE patientunitstayid = ANY(%(s)s)
                AND drugname ~* '{vaso}' AND drugstartoffset < 1440) x GROUP BY 1""", {"s": ids}, ["s", "m"]).set_index("s")["m"]
        early = set(first_any[first_any < 1440].index)
        late = _fetch(cur, f"""SELECT DISTINCT i.patientunitstayid FROM eicu_crd.infusiondrug i JOIN eicu_crd.patient p USING (patientunitstayid)
              WHERE i.patientunitstayid = ANY(%(s)s) AND i.drugname ~* '{vaso}' AND i.infusionoffset >= 1440
                AND i.infusionoffset <= p.unitdischargeoffset""", {"s": ids}, ["s"])["s"]
        elig = vc.index.difference(pd.Index(list(early)))
        y_sql = pd.Series(elig.isin(set(late)).astype(int), index=elig)
        v = lab["vasopressor"]
        same = set(v.index) == set(y_sql.index)
        mism = int((v != y_sql.reindex(v.index)).sum()) if same else -1
        out.append(("eicu", "vasopressor cohort and label = SQL (charting hospitals, no pre-24 h use)",
                    f"eligible SQL {len(y_sql):,} vs pipeline {len(v):,}; {mism} mismatches; prevalence {v.mean():.3f}", same and mism == 0))
    from deferral_frontier.src.splits import load_patient_split
    hosp = pd.read_parquet(data / "hospital_ids.parquet")["hospital_id"]
    pid = pd.read_parquet(data / "patient_ids.parquet")["patient_id"]
    for t in ("sepsis", "mortality", "vasopressor"):
        tr, va, te = load_patient_split(data, lab[t], ids_file="hospital_ids.parquet", column="hospital_id")
        art = set(pd.read_parquet(ART / "eicu_hospital" / f"eicu_hospital_{t}_test_artifacts.parquet")["stay_id"])
        disjoint = not (set(hosp.loc[tr]) & set(hosp.loc[te])) and not (set(hosp.loc[va]) & set(hosp.loc[te]))
        shared_pat = len(set(pid.loc[te]) & set(pid.loc[np.concatenate([tr, va])]))
        out.append(("eicu-H", f"{t}: held-out hospitals disjoint; artifact = split",
                    f"test {len(te):,} stays in {hosp.loc[te].nunique()} hospitals; patients also in training hospitals: {shared_pat}",
                    disjoint and art == set(te)))
    return out


def check_splits() -> List[Check]:
    """Patient disjointness of the splits actually used, via artifact ids."""
    out: List[Check] = []
    from deferral_frontier.src.splits import load_patient_split
    spec = {"mimic4": ("{t}_test_artifacts.parquet", "hadm_id"),
            "mimic3": ("mimic3/mimic3_{t}_test_artifacts.parquet", "hadm_id"),
            "eicu": ("eicu/eicu_{t}_test_artifacts.parquet", "stay_id")}
    for db, (tmpl, col) in spec.items():
        pid = pd.read_parquet(ROOT / "data" / db / "patient_ids.parquet")["patient_id"]
        for t in ("sepsis", "mortality", "vasopressor"):
            y = pd.read_parquet(ROOT / "data" / db / f"{t}_labels.parquet")["label"]
            tr, va, te = load_patient_split(ROOT / "data" / db, y)
            art = set(pd.read_parquet(ART / tmpl.format(t=t))[col])
            disjoint = not (set(pid.loc[tr]) & set(pid.loc[te])) and not (set(pid.loc[va]) & set(pid.loc[te]))
            out.append((db, f"{t}: artifact test set = patient split; no shared patients",
                        f"test {len(te):,}; artifact {len(art):,}", art == set(te) and disjoint))
    return out


def check_headline() -> List[Check]:
    """Retained Brier at 80% for each gate, recomputed with plain numpy."""
    out: List[Check] = []
    sg = pd.read_csv(RES / "symmetric_confidence_gate.csv").set_index(["source", "task", "gate"])
    spec = {"MIMIC-IV": "{t}_test_artifacts.parquet", "MIMIC-III": "mimic3/mimic3_{t}_test_artifacts.parquet",
            "eICU": "eicu/eicu_{t}_test_artifacts.parquet",
            "eICU-H": "eicu_hospital/eicu_hospital_{t}_test_artifacts.parquet"}
    worst = 0.0
    for src, tmpl in spec.items():
        for t in ("sepsis", "mortality", "vasopressor"):
            a = pd.read_parquet(ART / tmpl.format(t=t))
            y, p = a["y_true"].to_numpy(float), a["cal_proba"].to_numpy()
            n_keep = int(round(0.8 * len(y)))
            for gate, s in (("one_sided_confidence", p), ("symmetric_confidence", np.abs(p - 0.5)),
                            ("reliability", a["r_score"].to_numpy())):
                keep = np.argsort(-s, kind="stable")[:n_keep]
                b = float(np.mean((p[keep] - y[keep]) ** 2))
                worst = max(worst, abs(b - sg.loc[(src, t, gate), "brier"]))
    out.append(("all", "retained Brier (3 gates × 12 pairs incl. eICU-H) recomputed with numpy",
                f"max abs difference {worst:.2e}", worst < 1e-3))
    return out


def check_calibration() -> List[Check]:
    out: List[Check] = []
    for path in sorted(ART.glob("**/*_test_artifacts.parquet")):
        a = pd.read_parquet(path)
        y, p = a["y_true"].to_numpy(float), a["cal_proba"].to_numpy()
        bins = np.array_split(np.argsort(p), 10)
        ece = sum(len(b) * abs(p[b].mean() - y[b].mean()) for b in bins) / len(y)
        out.append(("calibration", f"{path.name}: test ECE (10 quantile bins)",
                    f"ECE {ece:.3f}; mean p {p.mean():.3f} vs prevalence {y.mean():.3f}", ece < 0.03))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    env = _load_env(ROOT / ".env")
    checks = check_database("mimic4", env) + check_database("mimic3", env) + check_eicu(env) + check_splits() \
        + check_headline() + check_calibration()
    lines = ["# Round-1 independent verification", "", "| Area | Check | Detail | Pass |", "|---|---|---|---|"]
    lines += [f"| {a} | {c} | {d} | {'✅' if ok else '❌'} |" for a, c, d, ok in checks]
    n_fail = sum(not ok for *_, ok in checks)
    lines += ["", f"**{len(checks) - n_fail}/{len(checks)} checks passed.**"]
    (ROOT / "plan" / "r1-verification.md").write_text("\n".join(lines) + "\n")
    logger.info("\n".join(lines))


if __name__ == "__main__":
    main()
