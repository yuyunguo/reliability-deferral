"""Shared first-24 h ICU feature definitions for MIMIC-III and MIMIC-IV.

One definition for both MIMIC versions (round-1 re-extraction): 23 signals
(10 vitals, 13 labs) observed in [ICU intime, ICU intime + 24 h), each
summarized by mean, min, max, chronologically last value, and count.

Item IDs follow the MIMIC Code Repository conventions. MetaVision vitals
(2200xx) and lab item IDs are shared by MIMIC-III and MIMIC-IV; CareVue IDs
(< 1000) only occur in MIMIC-III.
"""
import logging
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

VITAL_MAP: Dict[str, List[int]] = {
    "vital_heart_rate": [211, 220045],
    "vital_resp_rate": [615, 618, 220210, 224690],
    "vital_abp_sys": [51, 220050],
    "vital_abp_dia": [8368, 220051],
    "vital_abp_mean": [52, 220052],
    "vital_nibp_sys": [442, 455, 220179],
    "vital_nibp_dia": [8440, 8441, 220180],
    "vital_nibp_mean": [456, 220181],
    "vital_temp_f": [678, 223761],
    "vital_temp_c": [676, 223762],
}

LAB_MAP: Dict[str, List[int]] = {
    "lab_creatinine": [50912],
    "lab_hemoglobin": [51222, 50811],
    "lab_wbc": [51301, 51300],
    "lab_platelet": [51265],
    "lab_sodium": [50983, 50824],
    "lab_potassium": [50971, 50822],
    "lab_glucose": [50931, 50809],
    "lab_bicarbonate": [50882],
    "lab_bun": [51006],
    "lab_alt": [50861],
    "lab_ast": [50878],
    "lab_bilirubin": [50885],
    "lab_lactate": [50813],
}

STATS = ("mean", "min", "max", "last", "count")
SIGNALS = list(VITAL_MAP) + list(LAB_MAP)
FEATURE_COLUMNS = [f"{s}_{stat}" for s in SIGNALS for stat in STATS]
ITEM_TO_SIGNAL = {i: s for m in (VITAL_MAP, LAB_MAP) for s, ids in m.items() for i in ids}


def aggregate_window(
    ts: pd.DataFrame, id_col: str, start: pd.Series, end: pd.Series
) -> pd.DataFrame:
    """Aggregate long-format observations within [start, end) per row id.

    Args:
        ts: Long format with columns id_col, charttime, feature_name, value.
        id_col: Name of the row-id column.
        start: Window start per row id.
        end: Window end (exclusive) per row id; same index as `start`.

    Returns:
        Frame indexed by start.index with FEATURE_COLUMNS; counts are 0 where unobserved.
    """
    ts = ts[ts["feature_name"].isin(SIGNALS)]
    win = ts.assign(_s=ts[id_col].map(start), _e=ts[id_col].map(end))
    win = win[(win["charttime"] >= win["_s"]) & (win["charttime"] < win["_e"])]
    win = win.sort_values([id_col, "charttime"], kind="stable")
    agg = win.groupby([id_col, "feature_name"])["value"].agg(list(STATS))
    wide = agg.unstack("feature_name")
    wide.columns = [f"{sig}_{stat}" for stat, sig in wide.columns]
    out = wide.reindex(index=start.index, columns=FEATURE_COLUMNS)
    count_cols = [c for c in FEATURE_COLUMNS if c.endswith("_count")]
    out[count_cols] = out[count_cols].fillna(0)
    out.index.name = start.index.name
    return out
