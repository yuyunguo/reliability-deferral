"""POST HOC extra analyses C and D (plan/r1-extra-analyses-plan.md, sha256 d48b8bb1...).

C. Harmonized MIMIC-IV->eICU transfer: counts and temperature removed from all model inputs.
D. MIMIC-IV Sepsis-3 with onset after 24 h ('sepsis3onset' task).

Outputs (deferral_frontier/results/):
    r1_extra_harmonized.csv          risk-matched rows (as r1_v3_stratified) + Holm
    r1_extra_harmonized_desc.csv     M-ECE, LRT, recalibration
    r1_extra_sepsis_onset.csv        risk-matched rows + Holm
    r1_extra_sepsis_onset_gates.csv  whole-cohort gates at 80%

Usage:
    python -m deferral_frontier.experiments.r1_extra_cd
"""
import logging
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from deferral_frontier.experiments.r1_v3 import KS, ROOT, TASKS, build_scores, evaluate
from deferral_frontier.experiments.r1_v3_posthoc import PRIMARY, unstrat_ci
from deferral_frontier.experiments.run_symmetric_confidence_gate import RESULTS_DIR, retained_mask
from deferral_frontier.src.extra_analyses import harmonized_drop
from deferral_frontier.src.posthoc_tests import lrt_beyond_risk
from deferral_frontier.src.train_pipeline import train_pipeline
from src.calibration.baselines import IsotonicCalibrator
from src.calibration.metrics import adaptive_ece
from src.models.xgboost_model import ensemble_predict

logger = logging.getLogger(__name__)


def holm(df: pd.DataFrame) -> pd.DataFrame:
    """Holm within setting over the primary family at K = 10."""
    df = df.copy()
    df["p_holm"] = np.nan
    fam = df[(df.family == "primary") & (df.K == 10)]
    for _, g in fam.groupby("setting"):
        df.loc[g.index, "p_holm"] = multipletests(g.p.values, method="holm")[1]
    return df


def gates(p: np.ndarray, y: np.ndarray, r: np.ndarray) -> dict:
    e = (p - y) ** 2
    k = {"one_sided": retained_mask(p, 0.8), "symmetric": retained_mask(np.maximum(p, 1 - p), 0.8),
         "composite": retained_mask(r, 0.8)}
    out = {f"brier80_{g}": e[m].mean() for g, m in k.items()}
    out.update({f"prev80_{g}": y[m].mean() for g, m in k.items()})
    out.update({f"sym_minus_comp_{a}": b for a, b in unstrat_ci(e, k["symmetric"], k["composite"]).items()})
    return out


def main(parts: str = "CD") -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    logger.setLevel(logging.INFO)
    rows_c, desc_c, boots = [], [], {}
    if "C" in parts:
        run_c(rows_c, desc_c, boots)
    if "D" in parts:
        run_d(boots)


def run_c(rows_c: list, desc_c: list, boots: dict) -> None:
    """C: harmonized transfer."""
    for task in TASKS:
        cols = list(pd.read_parquet(ROOT / "data" / "mimic4" / f"{task}_features.parquet").columns)
        drop = harmonized_drop(cols)
        m4 = train_pipeline(task, data_dir=ROOT / "data" / "mimic4", drop_features=drop)
        ei = train_pipeline(task, data_dir=ROOT / "data" / "eicu", drop_features=drop)
        for setting, (p, y, sc) in (("eICU-harm", build_scores(ei, "eicu")),
                                    ("X-DB-harm", build_scores(m4, "mimic4", target=ei))):
            r, _ = evaluate(setting, task, p, y, sc, boots)
            rows_c.extend(r)
            d = {"setting": setting, "task": task, "n_features": len(m4.feature_names), "mece": adaptive_ece(p, y)}
            for n in PRIMARY:
                stat, pv = lrt_beyond_risk(y, p, sc[n])
                d[f"lrt_{n}"], d[f"lrt_p_{n}"] = stat, pv
            if setting == "X-DB-harm":
                c = [ei.feature_names.index(f) for f in m4.feature_names]
                p_rec = IsotonicCalibrator().fit(ensemble_predict(m4.ensemble, ei.X_val_raw[:, c]),
                                                 ei.y_val.astype(int)).predict_proba(
                    ensemble_predict(m4.ensemble, ei.X_test_raw[:, c]))
                e = (p - y) ** 2
                er = (p_rec - y) ** 2
                d.update({"mece_recalibrated": adaptive_ece(p_rec, y),
                          "brier80_sym": e[retained_mask(np.maximum(p, 1 - p), 0.8)].mean(),
                          "brier80_Cm": e[retained_mask(sc["Cp"], 0.8)].mean(),
                          "brier80_sym_recal": er[retained_mask(np.maximum(p_rec, 1 - p_rec), 0.8)].mean()})
            desc_c.append(d)
            logger.info("%s %s done", setting, task)
        holm(pd.DataFrame(rows_c)).to_csv(RESULTS_DIR / "r1_extra_harmonized.csv", index=False)
        pd.DataFrame(desc_c).to_csv(RESULTS_DIR / "r1_extra_harmonized_desc.csv", index=False)
        del m4, ei


def run_d(boots: dict) -> None:
    """D: MIMIC-IV Sepsis-3 with onset after 24 h."""
    art = train_pipeline("sepsis3onset", data_dir=ROOT / "data" / "mimic4")
    p, y, sc = build_scores(art, "mimic4")
    r, _ = evaluate("MIMIC-IV", "sepsis3onset", p, y, sc, boots)
    holm(pd.DataFrame(r)).to_csv(RESULTS_DIR / "r1_extra_sepsis_onset.csv", index=False)
    g = {"n": len(y), "prevalence": y.mean(), "mece": adaptive_ece(p, y), **gates(p, y, art.r_test)}
    pd.DataFrame([g]).to_csv(RESULTS_DIR / "r1_extra_sepsis_onset_gates.csv", index=False)
    logger.info("sepsis3onset done")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "CD")
