"""POST HOC robustness analyses for the risk-matched test (labeled post hoc).

1. Titrated positive controls (k = 4, 8, 16, 32; 10 injections each): smallest detectable gain.
2. Hospital-cluster bootstrap of Delta_adj for eICU-H and MIMIC-IV->eICU (clusters = eICU hospitals).
3. Likelihood-ratio test of each score beyond a spline of logit(p) (all patients, no bins).
4. MIMIC-IV->eICU recalibration: isotonic fitted on eICU validation outputs of the MIMIC-IV model;
   unstratified retained Brier at 80% for symmetric (as transferred), symmetric (recalibrated), and C^m.
5. Planted-signal recovery (MIMIC-IV): a feature z = e + N(0, 2 SD(e)) appended to L's inputs.

Outputs: deferral_frontier/results/r1_v3_posthoc_{titration,cluster,lrt,recal,planted}.csv

Usage:
    python -m deferral_frontier.experiments.r1_v3_posthoc
"""
import logging
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import xgboost as xgb

from deferral_frontier.experiments.r1_v3 import ROOT, SEED, TASKS, build_scores
from deferral_frontier.experiments.run_symmetric_confidence_gate import RESULTS_DIR, retained_mask
from deferral_frontier.src.posthoc_tests import cluster_bootstrap_deltas, lrt_beyond_risk, titrated_control
from deferral_frontier.src.reliability_v2 import patient_groups
from deferral_frontier.src.reliability_v3 import L_PARAMS, error_score, fit_error_model
from deferral_frontier.src.stratified_deferral import bootstrap_deltas, summarize
from deferral_frontier.src.train_pipeline import train_pipeline
from src.calibration.baselines import IsotonicCalibrator
from src.calibration.metrics import adaptive_ece
from src.models.xgboost_model import ensemble_predict

logger = logging.getLogger(__name__)
PRIMARY = ("D3p", "D5p", "Cp", "L", "LD")
KS_CTL = (4, 8, 16, 32)


def hospitals(art) -> np.ndarray:
    hid = pd.read_parquet(ROOT / "data" / "eicu" / "hospital_ids.parquet")["hospital_id"]
    return hid.reindex(art.idx_test).to_numpy()


def unstrat_ci(e: np.ndarray, keep_a: np.ndarray, keep_b: np.ndarray, n_boot: int = 1000) -> dict:
    """Paired bootstrap of retained Brier difference (a - b) at fixed retained sets."""
    rng = np.random.default_rng(SEED)
    d = e[keep_a].mean() - e[keep_b].mean()
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(e), len(e))
        boots[i] = e[idx][keep_a[idx]].mean() - e[idx][keep_b[idx]].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff": d, "ci_lo": lo, "ci_hi": hi}


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    logger.setLevel(logging.INFO)
    tit, clu, lrt, rec, pla = [], [], [], [], []

    def per_setting(setting, task, p, y, sc):
        for k in KS_CTL:
            tit.append({"setting": setting, "task": task, **titrated_control(p, y, k, n_rep=10, n_boot=100, seed=SEED)})
        for name in PRIMARY:
            stat, pv = lrt_beyond_risk(y, p, sc[name])
            lrt.append({"setting": setting, "task": task, "score": name, "lr_stat": stat, "p": pv})
        logger.info("%s %s titration+lrt done", setting, task)

    for task in TASKS:
        m4 = train_pipeline(task, data_dir=ROOT / "data" / "mimic4")
        p, y, sc = build_scores(m4, "mimic4")
        per_setting("MIMIC-IV", task, p, y, sc)
        # 5. planted-signal recovery
        e_val_proxy = None
        rng = np.random.default_rng(SEED)
        raw_val = ensemble_predict(m4.ensemble, m4.X_val_raw)
        p_val = m4.iso_cal.predict_proba(raw_val)
        e_val = (p_val - m4.y_val.astype(int)) ** 2
        e_t = (p - y) ** 2
        z_val = e_val + rng.normal(0, 2 * e_val.std(), len(e_val))
        z_t = e_t + rng.normal(0, 2 * e_t.std(), len(e_t))
        groups = patient_groups(ROOT / "data" / "mimic4", m4.idx_val)
        Lz = fit_error_model(xgb.XGBRegressor(**L_PARAMS), raw_val, m4.y_val.astype(int),
                             np.column_stack([m4.X_val_raw, z_val]), groups)
        s_lz = error_score(Lz, p, np.column_stack([m4.X_test_raw, z_t]))
        point, boots = bootstrap_deltas({"symmetric": np.maximum(p, 1 - p), "Lz": s_lz, "z": -z_t},
                                        p, y, k=10, n_boot=1000, seed=SEED)
        for nm in ("Lz", "z"):
            pla.append({"task": task, "score": nm,
                        **summarize(point[nm] - point["symmetric"], boots[nm] - boots["symmetric"])})
        ei = train_pipeline(task, data_dir=ROOT / "data" / "eicu")
        p_e, y_e, sc_e = build_scores(ei, "eicu")
        per_setting("eICU", task, p_e, y_e, sc_e)
        px, yx, scx = build_scores(m4, "mimic4", target=ei)
        per_setting("X-DB", task, px, yx, scx)
        # 2. hospital-cluster bootstrap, X-DB
        hx = hospitals(ei)
        sub = {"symmetric": scx["symmetric"], **{n: scx[n] for n in PRIMARY}}
        point, boots = cluster_bootstrap_deltas(sub, px, yx, hx, k=10, n_boot=1000, seed=SEED)
        for n in PRIMARY:
            clu.append({"setting": "X-DB", "task": task, "score": n,
                        **summarize(point[n] - point["symmetric"], boots[n] - boots["symmetric"])})
        # 4. recalibration at the target site
        cols = [ei.feature_names.index(c) for c in m4.feature_names]
        raw_ev = ensemble_predict(m4.ensemble, ei.X_val_raw[:, cols])
        raw_et = ensemble_predict(m4.ensemble, ei.X_test_raw[:, cols])
        p_rec = IsotonicCalibrator().fit(raw_ev, ei.y_val.astype(int)).predict_proba(raw_et)
        e_tr = (px - yx) ** 2
        e_rc = (p_rec - yx) ** 2
        k_sym = retained_mask(np.maximum(px, 1 - px), 0.8)
        k_cm = retained_mask(scx["Cp"], 0.8)
        k_rec = retained_mask(np.maximum(p_rec, 1 - p_rec), 0.8)
        rec.append({"task": task, "mece_transferred": adaptive_ece(px, yx), "mece_recalibrated": adaptive_ece(p_rec, yx),
                    "brier80_sym_transferred": e_tr[k_sym].mean(), "brier80_Cm": e_tr[k_cm].mean(),
                    "brier80_sym_recalibrated": e_rc[k_rec].mean(),
                    **{f"cm_minus_sym_{k}": v for k, v in unstrat_ci(e_tr, k_cm, k_sym).items()}})
        del m4, ei
        m3 = train_pipeline(task, data_dir=ROOT / "data" / "mimic3")
        per_setting("MIMIC-III", task, *build_scores(m3, "mimic3"))
        del m3
        eh = train_pipeline(task, data_dir=ROOT / "data" / "eicu", split_by="hospital")
        ph, yh, sch = build_scores(eh, "eicu")
        per_setting("eICU-H", task, ph, yh, sch)
        hh = hospitals(eh)
        sub = {"symmetric": sch["symmetric"], **{n: sch[n] for n in PRIMARY}}
        point, boots = cluster_bootstrap_deltas(sub, ph, yh, hh, k=10, n_boot=1000, seed=SEED)
        for n in PRIMARY:
            clu.append({"setting": "eICU-H", "task": task, "score": n,
                        **summarize(point[n] - point["symmetric"], boots[n] - boots["symmetric"])})
        del eh
        for name, rows in (("titration", tit), ("cluster", clu), ("lrt", lrt), ("recal", rec), ("planted", pla)):
            pd.DataFrame(rows).to_csv(RESULTS_DIR / f"r1_v3_posthoc_{name}.csv", index=False)
        logger.info("task %s complete", task)


if __name__ == "__main__":
    main()
