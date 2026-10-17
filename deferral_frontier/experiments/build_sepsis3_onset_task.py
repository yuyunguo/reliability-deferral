"""Build a MIMIC-IV 'sepsis3onset' task: Sepsis-3 with onset after the 24 h prediction time.

Post hoc analysis D (plan/r1-extra-analyses-plan.md). Onset = earlier of
suspected_infection_time and sofa_time in mimic_derived.sepsis3. Stays with
Sepsis-3 onset at or before ICU intime + 24 h are excluded; label 1 if onset is
later than that, else 0. Features, meta, and timeseries are restricted copies of the
'sepsis' task, so only the label (and the excluded stays) differ.
Credentials are read from the project .env only.

Usage:
    python -m deferral_frontier.experiments.build_sepsis3_onset_task
"""
import logging

import numpy as np
import pandas as pd

from deferral_frontier.experiments.build_sepsis3_task import DATA, ROOT, SQL, _connect, _load_env, _q, select_cohort

logger = logging.getLogger(__name__)
PREDICTION_HOURS = 24


def onset_labels(cohort: pd.DataFrame, onset: pd.Series) -> pd.Series:
    """0/1 label per admission; NaN (to drop) when Sepsis-3 onset <= intime + 24 h.

    Args:
        cohort: rows indexed by hadm_id with 'stay_id' and 'intime'.
        onset: Sepsis-3 onset time per stay_id (only Sepsis-3-positive stays).
    """
    t = cohort["stay_id"].map(onset)
    cutoff = pd.to_datetime(cohort["intime"]) + pd.Timedelta(hours=PREDICTION_HOURS)
    lab = pd.Series(0.0, index=cohort.index, name="label")
    lab[t.notna() & (pd.to_datetime(t) > cutoff)] = 1.0
    lab[t.notna() & (pd.to_datetime(t) <= cutoff)] = np.nan
    return lab


def subset_tables(features: pd.DataFrame, meta: pd.DataFrame, timeseries: pd.DataFrame,
                  rows: pd.Index):
    """Restrict the sepsis task tables to the label rows (in label order)."""
    return features.loc[rows], meta.loc[rows], timeseries[timeseries["stay_id"].isin(rows)]


def write_subset(rows: pd.Index) -> None:
    """Write sepsis3onset features/meta/timeseries restricted to `rows`."""
    tabs = [pd.read_parquet(DATA / f"sepsis_{p}.parquet") for p in ("features", "meta", "timeseries")]
    for part, tab in zip(("features", "meta", "timeseries"), subset_tables(*tabs, rows)):
        tab.to_parquet(DATA / f"sepsis3onset_{part}.parquet")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rows = pd.read_parquet(DATA / "sepsis_labels.parquet").index
    with _connect(_load_env(ROOT / ".env"), "MIMICIV_DB") as con, con.cursor() as cur:
        stays = _q(cur, SQL["mimic4"]["stays"], {}, ["stay_id", "hadm_id", "subject_id", "intime",
                                                     "outtime", "los", "age", "hospital_expire_flag"])
        cur.execute("SELECT stay_id, LEAST(suspected_infection_time, sofa_time) FROM mimic_derived.sepsis3 "
                    "WHERE sepsis3")
        onset = pd.Series({int(r[0]): r[1] for r in cur.fetchall()})
    stays[["age", "los"]] = stays[["age", "los"]].astype(float)
    cohort = select_cohort(stays).loc[rows]
    lab = onset_labels(cohort, onset)
    n_excl = int(lab.isna().sum())
    lab = lab.dropna().astype(int)
    lab.index.name = "hadm_id"
    lab.to_frame().to_parquet(DATA / "sepsis3onset_labels.parquet")
    write_subset(lab.index)
    logger.info("sepsis3onset: n=%d, prevalence %.3f, excluded (onset <= 24 h) %d", len(lab), lab.mean(), n_excl)


if __name__ == "__main__":
    main()
