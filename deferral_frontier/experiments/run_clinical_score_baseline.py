"""Clinical-score baselines for in-hospital mortality (reviewer R2-5).

MIMIC-IV: first-day SOFA (mimic_derived.sofa, sofa_24hours at hour 23),
recalibrated to a probability by logistic regression on the validation set.
eICU: APACHE IVa predicted hospital mortality (already a probability).
Each score is evaluated on the same test patients as the XGBoost model:
AUROC, Brier score, and retained Brier under the symmetric gate at 80% coverage.

Output: deferral_frontier/results/clinical_score_baseline.csv

Usage:
    python -m deferral_frontier.experiments.run_clinical_score_baseline
"""
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from deferral_frontier.experiments.build_sepsis3_task import DATA
from deferral_frontier.experiments.extract_mimic_features import SQL, _q
from deferral_frontier.experiments.extract_patient_ids import _connect, _load_env
from deferral_frontier.experiments.run_symmetric_confidence_gate import (
    ARTIFACT_DIR, COVERAGE, RESULTS_DIR, retained_mask,
)
from deferral_frontier.src.icu_labels import select_cohort

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]


def score_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    """AUROC, Brier, and symmetric-gate retained Brier at 80% coverage."""
    keep = retained_mask(np.maximum(p, 1 - p), COVERAGE)
    return {"n": len(y), "auroc": float(roc_auc_score(y, p)),
            "brier": float(np.mean((p - y) ** 2)),
            "brier_symmetric_80": float(np.mean((p[keep] - y[keep]) ** 2))}


def _xgb_row(source: str, df: pd.DataFrame) -> dict:
    y, p = df["y_true"].to_numpy().astype(int), df["cal_proba"].to_numpy()
    return {"source": source, "model": "XGBoost (calibrated)", **score_metrics(y, p)}


def _mimic4_sofa(env: dict) -> list:
    test = pd.read_parquet(ARTIFACT_DIR / "mortality_test_artifacts.parquet").set_index("hadm_id")
    val = pd.read_parquet(ARTIFACT_DIR / "mortality_val_artifacts.parquet").set_index("hadm_id")
    with _connect(env, "MIMICIV_DB") as con, con.cursor() as cur:
        stays = _q(cur, SQL["mimic4"]["stays"], {}, ["stay_id", "hadm_id", "subject_id", "intime",
                                                     "outtime", "los", "age", "hospital_expire_flag"])
        stays[["age", "los"]] = stays[["age", "los"]].astype(float)
        cohort = select_cohort(stays)
        ids = [int(s) for s in cohort.loc[test.index.union(val.index), "stay_id"]]
        sofa = _q(cur, "SELECT stay_id, max(sofa_24hours) FROM mimic_derived.sofa "
                       "WHERE hr BETWEEN 0 AND 23 AND stay_id = ANY(%(s)s) GROUP BY stay_id",
                  {"s": ids}, ["stay_id", "sofa"]).set_index("stay_id")["sofa"]
    s_of = lambda idx: cohort.loc[idx, "stay_id"].map(sofa).fillna(0).to_numpy(float)  # noqa: E731
    lr = LogisticRegression().fit(s_of(val.index)[:, None], val["y_true"].astype(int))
    p = lr.predict_proba(s_of(test.index)[:, None])[:, 1]
    y = test["y_true"].to_numpy().astype(int)
    return [{"source": "MIMIC-IV", "model": "SOFA (first 24 h, logistic)", **score_metrics(y, p)},
            _xgb_row("MIMIC-IV", test.reset_index())]


def _eicu_apache(env: dict) -> list:
    test = pd.read_parquet(ARTIFACT_DIR / "eicu" / "eicu_mortality_test_artifacts.parquet")
    with _connect(env, "EICU_DB") as con, con.cursor() as cur:
        ap = _q(cur, "SELECT patientunitstayid, predictedhospitalmortality::float "
                     "FROM eicu_crd.apachepatientresult WHERE apacheversion = 'IVa' "
                     "AND patientunitstayid = ANY(%(s)s)",
                {"s": [int(s) for s in test["stay_id"]]}, ["stay_id", "p"]).set_index("stay_id")["p"]
    ap = ap[ap >= 0]
    sub = test[test["stay_id"].isin(ap.index)]
    logger.info("APACHE IVa available for %d of %d eICU test stays", len(sub), len(test))
    y = sub["y_true"].to_numpy().astype(int)
    return [{"source": "eICU", "model": "APACHE IVa", **score_metrics(y, sub["stay_id"].map(ap).to_numpy())},
            _xgb_row("eICU", sub)]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    env = _load_env(ROOT / ".env")
    out = pd.DataFrame(_mimic4_sofa(env) + _eicu_apache(env))
    out.to_csv(RESULTS_DIR / "clinical_score_baseline.csv", index=False)
    logger.info("\n%s", out.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
