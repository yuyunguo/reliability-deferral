"""Redesigned eICU extraction (round 1; see plan/r1-eicu-design-audit.md).

Cohort: first eligible ICU (unit) stay per hospitalization; age >= 18; unit stay
>= 24 h; known hospital discharge status. Features: [0, 24 h) from unit
admission, shared aggregation. Labels:
  - mortality: hospitaldischargestatus == 'Expired'
  - sepsis: diagnosis strings in the sepsis / septic shock category
  - vasopressor: first vasopressor infusion in [24 h, unit discharge], among stays
    with no vasopressor infusion or IV order before 24 h; restricted to hospitals
    that chart vasopressor infusions at all.
eICU has no clock times, so offsets are anchored to a shared epoch.
Writes data/eicu_r1/ (swap with data/eicu after review), including
patient_ids.parquet and hospital_ids.parquet.

Usage:
    python -m deferral_frontier.experiments.extract_eicu_r1
"""
import logging
from pathlib import Path

import pandas as pd

from deferral_frontier.experiments.extract_patient_ids import _connect, _load_env
from deferral_frontier.src.icu_features import aggregate_window
from deferral_frontier.src.icu_labels import MIN_AGE, vasopressor_labels

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "eicu_r1"
EPOCH = pd.Timestamp("2000-01-01")
WINDOW_MIN = 24 * 60
BATCH = 2000
SEPSIS_RE = r"\|(?:sepsis|septic shock)(?:\||$)"
VASO_RE = "(norepinephrine|levophed|dopamine|vasopressin|epinephrine|phenylephrine|neo-?synephrine)"
LABS = {"lab_creatinine": "creatinine", "lab_sodium": "sodium", "lab_potassium": "potassium",
        "lab_bun": "BUN", "lab_glucose": "glucose", "lab_hemoglobin": "Hgb",
        "lab_bilirubin": "total bilirubin", "lab_wbc": "WBC x 1000", "lab_lactate": "lactate",
        "lab_platelet": "platelets x 1000", "lab_alt": "ALT (SGPT)", "lab_ast": "AST (SGOT)",
        "lab_bicarbonate": "bicarbonate"}
VITALS = [("vitalperiodic", "heartrate", "vital_heart_rate", "> 0"),
          ("vitalperiodic", "respiration", "vital_resp_rate", "> 0"),
          ("vitalperiodic", "systemicsystolic", "vital_abp_sys", "> 0"),
          ("vitalperiodic", "systemicdiastolic", "vital_abp_dia", "> 0"),
          ("vitalperiodic", "systemicmean", "vital_abp_mean", "> 0"),
          ("vitalperiodic", "temperature", "vital_temp_c", "BETWEEN 20 AND 45"),
          ("vitalaperiodic", "noninvasivesystolic", "vital_nibp_sys", "> 0"),
          ("vitalaperiodic", "noninvasivediastolic", "vital_nibp_dia", "> 0"),
          ("vitalaperiodic", "noninvasivemean", "vital_nibp_mean", "> 0")]


def eicu_sepsis_flag(diagnosisstring: pd.Series) -> pd.Series:
    """True for diagnosis strings in the sepsis / septic shock category."""
    return diagnosisstring.str.contains(SEPSIS_RE, case=False, regex=True)


def select_eicu_cohort(stays: pd.DataFrame) -> pd.DataFrame:
    """First eligible unit stay per hospitalization; indexed by stay_id, with mortality."""
    ok = ((stays["unitdischargeoffset"] >= WINDOW_MIN) & (stays["age"] >= MIN_AGE)
          & stays["hospitaldischargestatus"].isin(["Alive", "Expired"]))
    first = stays[ok].sort_values(["hosp_stay_id", "unitvisitnumber"]).drop_duplicates("hosp_stay_id")
    out = first.set_index("stay_id")
    out["mortality"] = (out["hospitaldischargestatus"] == "Expired").astype(int)
    return out


def _q(cur, sql: str, params: dict, cols: list) -> pd.DataFrame:
    cur.execute(sql, params)
    return pd.DataFrame(cur.fetchall(), columns=cols)


def _timeseries(cur, ids: list) -> pd.DataFrame:
    parts = []
    for i in range(0, len(ids), BATCH):
        b = ids[i:i + BATCH]
        vit = " UNION ALL ".join(
            f"SELECT patientunitstayid, observationoffset, '{name}', {col}::float FROM eicu_crd.{tab} "
            f"WHERE patientunitstayid = ANY(%(s)s) AND observationoffset >= 0 "
            f"AND observationoffset < {WINDOW_MIN} AND {col} {cond}" for tab, col, name, cond in VITALS)
        parts.append(_q(cur, vit, {"s": b}, ["stay_id", "offset", "feature_name", "value"]))
        lab = _q(cur, f"SELECT patientunitstayid, labresultoffset, labname, labresult FROM eicu_crd.lab "
                      f"WHERE patientunitstayid = ANY(%(s)s) AND labresultoffset >= 0 "
                      f"AND labresultoffset < {WINDOW_MIN} AND labresult > 0 AND labname = ANY(%(n)s)",
                 {"s": b, "n": list(LABS.values())}, ["stay_id", "offset", "labname", "value"])
        lab["feature_name"] = lab["labname"].map({v: k for k, v in LABS.items()})
        parts.append(lab[["stay_id", "offset", "feature_name", "value"]])
        logger.info("  batch %d/%d", i // BATCH + 1, (len(ids) - 1) // BATCH + 1)
    ts = pd.concat(parts, ignore_index=True)
    ts["value"] = ts["value"].astype(float)
    ts["charttime"] = EPOCH + pd.to_timedelta(ts["offset"], unit="min")
    return ts[["stay_id", "charttime", "feature_name", "value"]]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    with _connect(_load_env(ROOT / ".env"), "EICU_DB") as con, con.cursor() as cur:
        stays = _q(cur, """SELECT patientunitstayid, patienthealthsystemstayid, uniquepid, hospitalid,
                                  unitvisitnumber, unitdischargeoffset,
                                  CASE WHEN age = '> 89' THEN 90 WHEN age ~ '^[0-9]+$' THEN age::int END,
                                  hospitaldischargestatus
                           FROM eicu_crd.patient""", {},
                   ["stay_id", "hosp_stay_id", "patient_id", "hospital_id", "unitvisitnumber",
                    "unitdischargeoffset", "age", "hospitaldischargestatus"])
        stays["age"] = stays["age"].astype(float)
        cohort = select_eicu_cohort(stays)
        ids = [int(s) for s in cohort.index]
        logger.info("cohort: %d first eligible unit stays", len(cohort))
        ts = _timeseries(cur, ids)
        dx = _q(cur, "SELECT DISTINCT patientunitstayid, diagnosisstring FROM eicu_crd.diagnosis "
                     "WHERE patientunitstayid = ANY(%(s)s)", {"s": ids}, ["stay_id", "dx"])
        inf = _q(cur, f"SELECT patientunitstayid, infusionoffset FROM eicu_crd.infusiondrug "
                      f"WHERE patientunitstayid = ANY(%(s)s) AND drugname ~* '{VASO_RE}'",
                 {"s": ids}, ["stay_id", "offset"])
        med = _q(cur, f"SELECT patientunitstayid, drugstartoffset FROM eicu_crd.medication "
                      f"WHERE patientunitstayid = ANY(%(s)s) AND drugname ~* '{VASO_RE}' "
                      f"AND drugstartoffset < {WINDOW_MIN}", {"s": ids}, ["stay_id", "offset"])
        cur.execute(f"SELECT DISTINCT p.hospitalid FROM eicu_crd.infusiondrug i JOIN eicu_crd.patient p "
                    f"USING (patientunitstayid) WHERE i.drugname ~* '{VASO_RE}'")
        charting = {r[0] for r in cur.fetchall()}

    start = pd.Series(EPOCH, index=cohort.index)
    feats = aggregate_window(ts, "stay_id", start, start + pd.Timedelta(minutes=WINDOW_MIN))
    has_obs = feats.filter(like="_count").sum(axis=1) > 0
    cohort, feats = cohort[has_obs], feats[has_obs]
    logger.info("with >= 1 observation: %d", len(cohort))

    sepsis_ids = set(dx.loc[eicu_sepsis_flag(dx["dx"]), "stay_id"])
    vstays = cohort[cohort["hospital_id"].isin(charting)]
    vstays = pd.DataFrame({"stay_id": vstays.index, "prediction_time": EPOCH + pd.Timedelta(minutes=WINDOW_MIN),
                           "outtime": EPOCH + pd.to_timedelta(vstays["unitdischargeoffset"].to_numpy(), unit="min")},
                          index=vstays.index)
    events = pd.concat([inf, med], ignore_index=True)
    events["event_time"] = EPOCH + pd.to_timedelta(events["offset"], unit="min")
    labels = {
        "sepsis": pd.Series(cohort.index.isin(sepsis_ids).astype(int), index=cohort.index),
        "mortality": cohort["mortality"],
        "vasopressor": vasopressor_labels(events[["stay_id", "event_time"]], vstays),
    }
    meta = pd.DataFrame({"admission_time": EPOCH, "prediction_time": EPOCH + pd.Timedelta(minutes=WINDOW_MIN),
                         "admit_year": 0}, index=cohort.index)
    for task, lab in labels.items():
        lab = lab.rename("label")
        lab.index.name = "stay_id"
        lab.to_frame().to_parquet(OUT / f"{task}_labels.parquet")
        feats.loc[lab.index].to_parquet(OUT / f"{task}_features.parquet")
        meta.loc[lab.index].to_parquet(OUT / f"{task}_meta.parquet")
        ts[ts["stay_id"].isin(lab.index)].to_parquet(OUT / f"{task}_timeseries.parquet")
        logger.info("  %s: n=%d prevalence=%.3f", task, len(lab), lab.mean())
    cohort[["patient_id"]].to_parquet(OUT / "patient_ids.parquet")
    cohort[["hospital_id"]].to_parquet(OUT / "hospital_ids.parquet")
    logger.info("hospitals: %d (vasopressor task: %d charting infusions)",
                cohort["hospital_id"].nunique(), vstays.shape[0] and cohort.loc[vstays.index, "hospital_id"].nunique())


if __name__ == "__main__":
    main()
