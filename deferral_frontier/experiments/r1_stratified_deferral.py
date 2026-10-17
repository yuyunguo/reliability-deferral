"""Pre-specified risk-stratified deferral test (plan/r1-stratified-deferral-preregistration.md).

Outputs:
    deferral_frontier/results/r1_stratified_deferral.csv
    deferral_frontier/results/r1_whole_cohort_panel.csv

Usage:
    python -m deferral_frontier.experiments.r1_stratified_deferral
"""
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

from deferral_frontier.experiments.r1_v2_posthoc_linear import ridge_error_score
from deferral_frontier.experiments.run_symmetric_confidence_gate import RESULTS_DIR, retained_mask
from deferral_frontier.src.reliability_v2 import patient_groups
from deferral_frontier.src.stratified_deferral import bootstrap_delta
from deferral_frontier.src.train_pipeline import train_pipeline
from src.models.xgboost_model import ensemble_predict

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
SOURCES = {"MIMIC-IV": ("mimic4", "patient"), "MIMIC-III": ("mimic3", "patient"),
           "eICU": ("eicu", "patient"), "eICU-H": ("eicu", "hospital")}
PRIMARY = ("D2", "D3", "D4", "D5", "neg_ensemble_sd", "v2", "ridge")
KS = (10, 20)
N_BOOT = 1000
SEED = 42


def scores_for(art, db: str) -> Dict[str, np.ndarray]:
    """All scores of the pre-registration (higher = more reliable)."""
    p, y = art.cal_test, art.y_test.astype(int)
    e = (p - y) ** 2
    dt, dv = art.dims_test, art.dims_val
    other_val = np.column_stack([dv.D2, dv.D3, dv.D4, dv.D5])
    other_test = np.column_stack([dt.D2, dt.D3, dt.D4, dt.D5])
    raw_val = ensemble_predict(art.ensemble, art.X_val_raw)
    groups = patient_groups(ROOT / "data" / db, art.idx_val)
    s = {"D2": dt.D2, "D3": dt.D3, "D4": dt.D4, "D5": dt.D5, "neg_ensemble_sd": -art.ensemble_sd_test,
         "v2": art.r_test,
         "ridge": ridge_error_score(raw_val, art.y_val.astype(int), other_val, groups, p, other_test, [0, 1, 2, 3]),
         "symmetric": np.maximum(p, 1 - p), "oracle": -e}
    rng = np.random.default_rng(SEED)
    for k in (1, 2, 4):
        s[f"positive_control_k{k}"] = -(e + rng.normal(0, k * e.std(), len(e)))
    return {name: np.asarray(v, dtype=float) for name, v in s.items()}


def whole_cohort(scores: Dict[str, np.ndarray], p: np.ndarray, y: np.ndarray) -> list:
    """Side panel: unstratified gates at 80% coverage, plus BSS(sym) - BSS(one-sided) CI."""
    e = (p - y) ** 2
    gates = {"one_sided": p, "symmetric": scores["symmetric"], "defer_top20_p": -p,
             "v2": scores["v2"], "ridge": scores["ridge"],
             "random": np.random.default_rng(SEED).random(len(p))}

    def bss(keep: np.ndarray) -> float:
        pi = y[keep].mean()
        return 1 - e[keep].mean() / (pi * (1 - pi))

    rows = []
    masks = {g: retained_mask(s, 0.8) for g, s in gates.items()}
    for g, k in masks.items():
        rows.append({"gate": g, "brier": e[k].mean(), "prevalence": y[k].mean(), "bss": bss(k),
                     "auroc": roc_auc_score(y[k], p[k])})
    rng = np.random.default_rng(SEED)
    boots = np.empty(N_BOOT)
    ks, ko = masks["symmetric"], masks["one_sided"]
    for i in range(N_BOOT):
        idx = rng.integers(0, len(p), len(p))
        e_i, y_i = e[idx], y[idx]
        def b(m): pi = y_i[m].mean(); return 1 - e_i[m].mean() / (pi * (1 - pi))
        boots[i] = b(ks[idx]) - b(ko[idx])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    for r in rows:
        r.update({"bss_sym_minus_one_sided": bss(ks) - bss(ko), "ci_lo": lo, "ci_hi": hi})
    return rows


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.INFO)
    strat, panel = [], []
    for source, (db, split_by) in SOURCES.items():
        for task in ("sepsis", "mortality", "vasopressor"):
            art = train_pipeline(task, data_dir=ROOT / "data" / db, split_by=split_by)
            p, y = art.cal_test, art.y_test.astype(int)
            sc = scores_for(art, db)
            for k in KS:
                oracle = None
                for name, s in sc.items():
                    r = bootstrap_delta(s, p, y, k=k, n_boot=N_BOOT, seed=SEED)
                    oracle = r["delta"] if name == "oracle" else oracle
                    strat.append({"source": source, "task": task, "K": k, "score": name,
                                  "family": "primary" if name in PRIMARY else "reference", **r})
                for row in strat[-len(sc):]:
                    row["share_of_oracle"] = row["delta"] / oracle
            for row in whole_cohort(sc, p, y):
                panel.append({"source": source, "task": task, **row})
            logger.info("%s %s done", source, task)
            pd.DataFrame(strat).to_csv(RESULTS_DIR / "r1_stratified_deferral.csv", index=False)
            pd.DataFrame(panel).to_csv(RESULTS_DIR / "r1_whole_cohort_panel.csv", index=False)
    out = pd.DataFrame(strat)
    out["p_holm"] = np.nan
    for (src, k), g in out[out.family == "primary"].groupby(["source", "K"]):
        out.loc[g.index, "p_holm"] = multipletests(g.p.values, method="holm")[1]
    out.to_csv(RESULTS_DIR / "r1_stratified_deferral.csv", index=False)
    pd.DataFrame(panel).to_csv(RESULTS_DIR / "r1_whole_cohort_panel.csv", index=False)


if __name__ == "__main__":
    main()
