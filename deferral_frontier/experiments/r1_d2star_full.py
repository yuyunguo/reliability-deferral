"""D2* descriptive full run (plan/r1-d2-imputation-addendum-1.md, hash ac06c37e…).

Computes D2* (imputation uncertainty) as a reference score across the same 15 settings
and endpoint as r1-reliability-v3-preregistration.md (amendment 1). Descriptive only —
no Holm family, no confirmatory claim.

Outputs (deferral_frontier/results/):
    r1_d2star_full.csv               gate comparison (Delta_adj vs symmetric primary)
    r1_d2star_full_discrimination.csv  saturation + redundancy summary

Usage:
    python -m deferral_frontier.experiments.r1_d2star_full
"""
import logging
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from deferral_frontier.experiments.run_symmetric_confidence_gate import RESULTS_DIR
from deferral_frontier.src.fast_reliability import compute_d2_fast
from deferral_frontier.src.imputation_uncertainty import StochasticImputer, imputation_uncertainty
from deferral_frontier.src.stratified_deferral import bootstrap_deltas, summarize
from deferral_frontier.src.train_pipeline import train_pipeline
from src.models.xgboost_model import ensemble_predict

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
TASKS = ("sepsis", "mortality", "vasopressor")
N_BOOT = 1000
N_PASSES = 50
SEED = 42
K = 10


def _impute(X_raw: np.ndarray, medians: pd.Series) -> np.ndarray:
    med = np.nan_to_num(medians.to_numpy(dtype=float), nan=0.0)
    return np.where(np.isnan(X_raw), med, X_raw)


def d2star_scores(art, target=None):
    """(p, y, scores, d2star_unc, d2_obs) for art's fitted objects on the target test set."""
    tgt = target if target is not None else art
    cols = [tgt.feature_names.index(c) for c in art.feature_names]
    X_t = tgt.X_test_raw[:, cols]
    y = tgt.y_test.astype(int)
    p = art.iso_cal.predict_proba(ensemble_predict(art.ensemble, X_t))
    # Conditioning set: X-DB uses the source (MIMIC-IV) column medians, as v3 does for D3'.
    X_imp = tgt.X_test if target is None else _impute(X_t, art.col_medians)
    imputer = StochasticImputer().fit(art.X_train, art.X_train_raw)
    d2star_unc = imputation_uncertainty(art.primary_model, X_imp, X_t, imputer,
                                        n_passes=N_PASSES, seed=SEED)
    d2_obs = compute_d2_fast(tgt.dataset.timeseries, pd.Index(tgt.idx_test),
                             tgt.dataset.admission_times, tgt.dataset.prediction_times,
                             lookback_hours=24, baseline_hours=6)
    d2_countfrac = tgt.dims_test.D2
    e = (p - y) ** 2
    scores = {"D2star": -d2star_unc, "D2_obsrate": d2_obs, "D2_countfrac": d2_countfrac,
              "symmetric": np.maximum(p, 1 - p), "oracle": -e}
    return p, y, scores, d2star_unc, d2_obs


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.setLevel(logging.INFO)
    gate_rows, disc_rows = [], []

    def run(setting, task, p, y, scores, d2star_unc, d2_obs):
        point, boots = bootstrap_deltas(scores, p, y, k=K, n_boot=N_BOOT, seed=SEED)
        for name in scores:
            rnd = summarize(point[name], boots[name])
            adj = summarize(point[name] - point["symmetric"], boots[name] - boots["symmetric"])
            gate_rows.append({"setting": setting, "task": task, "score": name,
                              "delta_vs_random": rnd["delta"], "ci_lo_vs_random": rnd["ci_lo"],
                              "ci_hi_vs_random": rnd["ci_hi"], "p_vs_random": rnd["p"],
                              "delta_adj": adj["delta"], "ci_lo_adj": adj["ci_lo"],
                              "ci_hi_adj": adj["ci_hi"], "p_adj": adj["p"],
                              "share_of_oracle": point[name] / point["oracle"]})
        disc_rows.append({
            "setting": setting, "task": task, "n": len(y), "prevalence": float(y.mean()),
            "d2_obsrate_saturated_frac": float((d2_obs >= 1.0 - 1e-9).mean()),
            "d2star_zero_frac": float((d2star_unc <= 1e-12).mean()),
            "d2star_cv": float(d2star_unc.std() / max(d2star_unc.mean(), 1e-12)),
            "spearman_d2star_p": float(spearmanr(d2star_unc, p).statistic),
            "spearman_d2star_p1mp": float(spearmanr(d2star_unc, p * (1 - p)).statistic),
        })
        logger.info("%s %s: D2_obs sat %.3f | D2* zero %.3f CV %.2f", setting, task,
                    (d2_obs >= 1.0 - 1e-9).mean(), (d2star_unc <= 1e-12).mean(),
                    d2star_unc.std() / max(d2star_unc.mean(), 1e-12))
        pd.DataFrame(gate_rows).to_csv(RESULTS_DIR / "r1_d2star_full.csv", index=False)
        pd.DataFrame(disc_rows).to_csv(RESULTS_DIR / "r1_d2star_full_discrimination.csv", index=False)

    for task in TASKS:
        m4 = train_pipeline(task, data_dir=ROOT / "data" / "mimic4")
        run("MIMIC-IV", task, *d2star_scores(m4))
        ei = train_pipeline(task, data_dir=ROOT / "data" / "eicu")
        run("eICU", task, *d2star_scores(ei))
        run("X-DB", task, *d2star_scores(m4, target=ei))
        del m4, ei
        run("MIMIC-III", task, *d2star_scores(train_pipeline(task, data_dir=ROOT / "data" / "mimic3")))
        run("eICU-H", task, *d2star_scores(
            train_pipeline(task, data_dir=ROOT / "data" / "eicu", split_by="hospital")))

    logger.info("wrote r1_d2star_full.csv and r1_d2star_full_discrimination.csv")


if __name__ == "__main__":
    main()
