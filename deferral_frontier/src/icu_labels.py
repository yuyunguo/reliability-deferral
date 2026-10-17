"""Shared outcome definitions for MIMIC-III and MIMIC-IV (round-1 re-extraction).

Sepsis: ICD codes for sepsis, severe sepsis, or septic shock. ICD-10 codes
are the General Equivalence Mappings of the ICD-9 set (995.91 -> A41.9,
995.92 -> R65.20, 785.52 -> R65.21). Septicemia codes (038.x, A40/A41.x other
than A41.9) are not included.

Vasopressor: new vasopressor start between prediction time (ICU intime + 24 h)
and ICU discharge, among stays with no vasopressor before prediction time.
"""
import pandas as pd

SEPSIS_ICD9 = frozenset({"99591", "99592", "78552"})
SEPSIS_ICD10 = frozenset({"A419", "R6520", "R6521"})
# Vasopressor item IDs from the MIMIC Code Repository (norepinephrine, epinephrine,
# vasopressin, phenylephrine, dopamine). Round 1 replaced a CareVue list that
# included non-vasopressor items (PO intake, GT flush, free-water bolus, Ativan).
VASO_IDS_MV = (221906, 221289, 222315, 221749, 221662)
VASO_IDS_CV = (30047, 30120, 30044, 30119, 30309, 30051, 30127, 30128, 30043, 30307)


def sepsis_labels(dx: pd.DataFrame, hadm_ids: pd.Index) -> pd.Series:
    """0/1 sepsis label per admission from diagnoses (hadm_id, icd_code, icd_version)."""
    code = dx["icd_code"].astype(str).str.replace(".", "", regex=False).str.upper().str.strip()
    is9 = (dx["icd_version"].astype(int) == 9) & code.isin(SEPSIS_ICD9)
    is10 = (dx["icd_version"].astype(int) == 10) & code.isin(SEPSIS_ICD10)
    pos = set(dx.loc[is9 | is10, "hadm_id"])
    labels = pd.Series([int(h in pos) for h in hadm_ids], index=hadm_ids, name="label")
    labels.index.name = "hadm_id"
    return labels


def vasopressor_labels(events: pd.DataFrame, stays: pd.DataFrame) -> pd.Series:
    """Label eligible admissions.

    Args:
        events: columns stay_id, event_time (vasopressor administrations).
        stays: index hadm_id; columns stay_id, prediction_time, outtime.

    Returns:
        0/1 Series named "label", indexed by hadm_id, excluding pre-24 h users.
    """
    ev = events.merge(stays.rename_axis("hadm_id").reset_index(), on="stay_id", how="inner")
    early = ev.loc[ev["event_time"] < ev["prediction_time"], "hadm_id"].unique()
    late_mask = (ev["event_time"] >= ev["prediction_time"]) & (ev["event_time"] <= ev["outtime"])
    late = ev.loc[late_mask, "hadm_id"].unique()
    eligible = stays.index.difference(pd.Index(early))
    labels = pd.Series(0, index=eligible, name="label", dtype=int)
    labels.loc[labels.index.isin(late)] = 1
    labels.index.name = "hadm_id"
    return labels


MIN_LOS_DAYS = 1.0
MIN_AGE = 18


def select_cohort(stays: pd.DataFrame) -> pd.DataFrame:
    """Apply the shared cohort rule to ICU stays.

    Keeps stays with ICU length of stay >= 24 h in patients aged >= 18, then the
    earliest such stay per admission. Prediction time is ICU intime + 24 h.

    Args:
        stays: columns stay_id, hadm_id, intime, los (days), age (years), plus any extras.

    Returns:
        Frame indexed by hadm_id with admission_time (= intime) and prediction_time added.
    """
    keep = stays[(stays["los"] >= MIN_LOS_DAYS) & (stays["age"] >= MIN_AGE)]
    first = keep.sort_values(["hadm_id", "intime"]).drop_duplicates("hadm_id")
    out = first.set_index("hadm_id")
    out["admission_time"] = out["intime"]
    out["prediction_time"] = out["intime"] + pd.Timedelta(hours=24)
    return out
