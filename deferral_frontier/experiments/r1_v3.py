"""Corrected dimensions, learned reliability, and shift
(plan/r1-reliability-v3-preregistration.md, hash 5fffa60c...; amendment 1, hash dce86dfe...).

Primary effect (amendment 1): Delta_adj = Delta(score) - Delta(symmetric), the gain beyond
confidence at matched risk, on the same paired bootstrap resamples.

Outputs (deferral_frontier/results/):
    r1_v3_stratified.csv, r1_v3_increment.csv, r1_v3_shift_contrast.csv,
    r1_v3_descriptive.csv, r1_v3_boots.npz

Usage:
    python -m deferral_frontier.experiments.r1_v3
"""
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

# Single OpenMP thread: torch/xgboost OpenMP clash on macOS (see run_neural_baselines.py).
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from deferral_frontier.experiments.run_symmetric_confidence_gate import RESULTS_DIR, retained_mask
from deferral_frontier.src.reliability_v2 import mc_dropout_variance, patient_groups
from deferral_frontier.src.reliability_v3 import (
    L_PARAMS, MissingAwareSimilarity, error_score, fit_error_model, nan_mask_variance)
from deferral_frontier.src.stratified_deferral import bootstrap_deltas, summarize
from deferral_frontier.src.train_pipeline import PipelineArtifacts, train_pipeline
from src.calibration.metrics import adaptive_ece
from src.models.xgboost_model import ensemble_predict, ensemble_predict_all_seeds

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
TASKS = ("sepsis", "mortality", "vasopressor")
KS = (10, 20)
N_BOOT = 1000
SEED = 42
PRIMARY = ("D3p", "D5p", "Cp", "L", "LD")
IN_DIST = ("MIMIC-IV", "MIMIC-III", "eICU")


def confirmatory(setting: str, score: str) -> bool:
    """Amendment 1, change 2."""
    if score in ("L", "LD"):
        return setting not in IN_DIST
    if score == "D5p":
        return setting != "MIMIC-IV"
    return score in PRIMARY


def _impute(X_raw: np.ndarray, medians: pd.Series) -> np.ndarray:
    med = np.nan_to_num(medians.to_numpy(dtype=float), nan=0.0)
    return np.where(np.isnan(X_raw), med, X_raw)


def build_scores(art: PipelineArtifacts, db: str, target: Optional[PipelineArtifacts] = None
                 ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """All v3 scores (higher = more reliable) on the target test set.

    `art` supplies every fitted object (models, calibrator, D3', error models). With
    `target`, they are applied unchanged to target's test patients (X-DB); otherwise
    to art's own test set.
    """
    tgt = target if target is not None else art
    cols = [tgt.feature_names.index(c) for c in art.feature_names]
    X_t = tgt.X_test_raw[:, cols]
    y = tgt.y_test.astype(int)
    p = art.iso_cal.predict_proba(ensemble_predict(art.ensemble, X_t))
    if target is None and not np.allclose(p, art.cal_test):
        raise RuntimeError("recomputed p differs from pipeline p")

    raw_val = ensemble_predict(art.ensemble, art.X_val_raw)
    y_val = art.y_val.astype(int)
    groups = patient_groups(ROOT / "data" / db, art.idx_val)
    sim = MissingAwareSimilarity().fit(art.X_train_raw)

    def dims(X_raw: np.ndarray, d2: np.ndarray, d4: np.ndarray) -> np.ndarray:
        return np.column_stack([d2, sim.transform(X_raw), d4,
                                -nan_mask_variance(art.primary_model, X_raw, seed=SEED),
                                -ensemble_predict_all_seeds(art.ensemble, X_raw)[1]])

    D_val = dims(art.X_val_raw, art.dims_val.D2, art.dims_val.D4)
    D_t = dims(X_t, tgt.dims_test.D2, tgt.dims_test.D4)

    ridge = fit_error_model(make_pipeline(StandardScaler(), Ridge(alpha=1.0)), raw_val, y_val, D_val, groups)
    L = fit_error_model(xgb.XGBRegressor(**L_PARAMS), raw_val, y_val, art.X_val_raw, groups)
    LD = fit_error_model(xgb.XGBRegressor(**L_PARAMS), raw_val, y_val,
                         np.column_stack([art.X_val_raw, D_val]), groups)

    if target is None:
        d3_built, d5_built = art.dims_test.D3, art.dims_test.D5
    else:
        X_imp = _impute(X_t, art.col_medians)
        d3_built = art.cohort_sim.transform(art.scaler.transform(X_imp))
        d5_built = -mc_dropout_variance(art.dropout_fn, X_imp)

    e = (p - y) ** 2
    s = {"D3p": D_t[:, 1], "D5p": D_t[:, 3], "Cp": error_score(ridge, p, D_t),
         "L": error_score(L, p, X_t), "LD": error_score(LD, p, np.column_stack([X_t, D_t])),
         "D2": D_t[:, 0], "D4": D_t[:, 2], "neg_ensemble_sd": D_t[:, 4],
         "D3_as_built": d3_built, "D5_as_built": d5_built,
         "symmetric": np.maximum(p, 1 - p), "oracle": -e}
    rng = np.random.default_rng(SEED)
    for k in (1, 2, 4):
        s[f"positive_control_k{k}"] = -(e + rng.normal(0, k * e.std(), len(e)))
    return p, y, {name: np.asarray(v, dtype=float) for name, v in s.items()}


def evaluate(setting: str, task: str, p: np.ndarray, y: np.ndarray, scores: Dict[str, np.ndarray],
             boots_store: Dict[str, np.ndarray]) -> Tuple[list, dict]:
    """Stratified rows (Delta_adj primary, Delta vs random descriptive) and descriptive row."""
    rows = []
    for k in KS:
        point, boots = bootstrap_deltas(scores, p, y, k=k, n_boot=N_BOOT, seed=SEED)
        for name in scores:
            boots_store[f"{setting}|{task}|{k}|{name}"] = boots[name]
        for name in scores:
            adj = summarize(point[name] - point["symmetric"], boots[name] - boots["symmetric"])
            rnd = summarize(point[name], boots[name])
            rows.append({"setting": setting, "task": task, "K": k, "score": name,
                         "family": "primary" if name in PRIMARY else "reference",
                         "confirmatory": confirmatory(setting, name), **adj,
                         "delta_vs_random": rnd["delta"], "ci_lo_vs_random": rnd["ci_lo"],
                         "ci_hi_vs_random": rnd["ci_hi"],
                         "share_of_oracle": point[name] / point["oracle"],
                         "sym_share_of_oracle": point["symmetric"] / point["oracle"]})
    e = (p - y) ** 2
    desc = {"setting": setting, "task": task, "n": len(y), "prevalence": y.mean(),
            "mece": adaptive_ece(p, y, n_bins=10)}
    for g in ("symmetric", "Cp", "L"):
        desc[f"brier80_{g}"] = e[retained_mask(scores[g], 0.8)].mean()
    return rows, desc


def holm(df: pd.DataFrame, by: list) -> pd.Series:
    """Holm-adjusted p-values within each group of `by`."""
    out = pd.Series(np.nan, index=df.index)
    for _, g in df.groupby(by):
        out.loc[g.index] = multipletests(g.p.values, method="holm")[1]
    return out


def finalize(strat: pd.DataFrame, boots: Dict[str, np.ndarray]) -> None:
    """Holm on the primary family, increment and shift-contrast families; write all outputs."""
    # Bin check (pre-registered rule 4, carried into v3): K = 20 primary if symmetric > 25% of oracle at K = 10.
    k10_sym = strat[(strat.K == 10) & (strat.score == "symmetric")].set_index(["setting", "task"])
    primary_k = {key: (20 if r.sym_share_of_oracle > 0.25 else 10) for key, r in k10_sym.iterrows()}
    strat["primary_K"] = [primary_k[(s, t)] for s, t in zip(strat.setting, strat.task)]
    fam = strat[(strat.family == "primary") & (strat.K == strat.primary_K)]
    strat["p_holm"] = np.nan
    strat.loc[fam.index, "p_holm"] = holm(fam, ["setting"])
    strat.to_csv(RESULTS_DIR / "r1_v3_stratified.csv", index=False)

    inc, shift = [], []
    for (setting, task), k in primary_k.items():
        pt = strat[(strat.setting == setting) & (strat.task == task) & (strat.K == k)].set_index("score")
        inc.append({"setting": setting, "task": task, "K": k,
                    **summarize(pt.delta_vs_random["LD"] - pt.delta_vs_random["L"], boots[f"{setting}|{task}|{k}|LD"] - boots[f"{setting}|{task}|{k}|L"])})
    inc = pd.DataFrame(inc)
    inc["p_holm"] = holm(inc, ["setting"])
    inc.to_csv(RESULTS_DIR / "r1_v3_increment.csv", index=False)

    for task in TASKS:
        if ("X-DB", task) not in primary_k:
            continue
        for k in KS:
            def adj(setting):
                pt = strat[(strat.setting == setting) & (strat.task == task) & (strat.K == k)].set_index("score")
                return (pt.delta["L"],
                        boots[f"{setting}|{task}|{k}|L"] - boots[f"{setting}|{task}|{k}|symmetric"])
            (px, bx), (pe, be) = adj("X-DB"), adj("eICU")
            shift.append({"task": task, "K": k, **summarize(px - pe, bx - be)})
    shift = pd.DataFrame(shift)
    shift["p_holm"] = holm(shift, ["K"])
    shift.to_csv(RESULTS_DIR / "r1_v3_shift_contrast.csv", index=False)
    np.savez_compressed(RESULTS_DIR / "r1_v3_boots.npz", **boots)


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.setLevel(logging.INFO)
    strat, desc, boots = [], [], {}

    def run(setting: str, task: str, p, y, scores) -> None:
        rows, d = evaluate(setting, task, p, y, scores, boots)
        strat.extend(rows)
        desc.append(d)
        pd.DataFrame(strat).to_csv(RESULTS_DIR / "r1_v3_stratified.csv", index=False)
        pd.DataFrame(desc).to_csv(RESULTS_DIR / "r1_v3_descriptive.csv", index=False)
        logger.info("%s %s done", setting, task)

    for task in TASKS:
        m4 = train_pipeline(task, data_dir=ROOT / "data" / "mimic4")
        run("MIMIC-IV", task, *build_scores(m4, "mimic4"))
        ei = train_pipeline(task, data_dir=ROOT / "data" / "eicu")
        run("eICU", task, *build_scores(ei, "eicu"))
        p, y, sc = build_scores(m4, "mimic4", target=ei)
        if not np.array_equal(y, ei.y_test.astype(int)):
            raise RuntimeError("X-DB labels differ from eICU test labels")
        run("X-DB", task, p, y, sc)
        del m4, ei
        run("MIMIC-III", task, *build_scores(train_pipeline(task, data_dir=ROOT / "data" / "mimic3"), "mimic3"))
        run("eICU-H", task, *build_scores(
            train_pipeline(task, data_dir=ROOT / "data" / "eicu", split_by="hospital"), "eicu"))
    finalize(pd.DataFrame(strat), boots)


if __name__ == "__main__":
    main()
