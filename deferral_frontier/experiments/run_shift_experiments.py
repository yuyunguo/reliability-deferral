"""Real distribution-shift experiments: missingness vs. correlated covariate shift.

Motivation: "Section 2.8 randomly
zeros features at test time. Real distribution shift in clinical settings
involves correlated changes (e.g., a new lab assay replacing an old one...).
Random feature dropout simulates missingness, not shift... ideally include
a correlated-shift experiment (e.g., shift the mean of a subset of
features)."

Integrity note: the existing deferral_frontier/src/shift_experiment.py
function does not actually push shifted features through the trained
model -- it fabricates synthetic noise proportional to the drop rate
(`shifted_probs = cal_test + noise`). This script replaces that with real
re-inference for both experiments, using the frozen (not retrained)
pipeline artifacts -- ensemble, isotonic calibrator, cohort-similarity
estimator, dropout function, and composite -- to simulate genuine
deployment drift after training is complete.

Experiment 1 (missingness, corrected): randomly zero a fraction of
feature columns, re-run inference end to end.
Experiment 2 (correlated shift, new): shift a clinically coherent lab
panel's mean by k standard deviations (renal: creatinine+BUN; liver:
ALT/AST/bilirubin), re-run inference end to end. D2/D4 are computed from
raw timeseries, not the static feature matrix, so they do not respond to
either perturbation; they are held at their unshifted test values and
this is stated explicitly rather than silently assumed.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.models.xgboost_model import ensemble_predict
from src.reliability.composite import DimensionMatrix
from src.reliability.uncertainty import compute_uncertainty_consistency

from deferral_frontier.src.train_pipeline import train_pipeline
from deferral_frontier.src.reliability_v2 import mc_dropout_variance

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("deferral_frontier/results")
COVERAGE = 0.80
DROP_RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
SHIFT_MAGNITUDES = [0.0, 0.5, 1.0, 1.5, 2.0]
N_REPLICATES = 5


def _panel_columns(feature_names, prefixes):
    return [i for i, c in enumerate(feature_names)
            if any(c.startswith(p) for p in prefixes) and not c.endswith("_count")]


def _score_gates(y_test, cal_shifted, r_shifted, coverage=COVERAGE):
    n_pred = int(np.round(len(y_test) * coverage))
    out = {}
    # "symmetric" added in round 1 (R1-3 / R2-4).
    for policy, score in [("confidence", cal_shifted),
                          ("symmetric", np.maximum(cal_shifted, 1 - cal_shifted)),
                          ("reliability", r_shifted)]:
        idx = np.argsort(-score)[:n_pred]
        y_sel, p_sel = y_test[idx], cal_shifted[idx]
        brier = float(brier_score_loss(y_sel, p_sel))
        try:
            auroc = float(roc_auc_score(y_sel, p_sel))
        except ValueError:
            auroc = float("nan")
        out[policy] = {"brier": brier, "auroc": auroc}
    return out


def _reinfer(artifacts, X_shifted_raw, X_shifted_imputed):
    """Push shifted features through the frozen pipeline; D2/D4 held at test values.

    D1 (confidence) must be recomputed on the NaN-preserving representation the
    ensemble was actually trained/predicted on (XGBoost's native missing-value
    routing); D3/D5 require the complete (imputed, scaled) matrix, matching
    train_pipeline.py's own phase order exactly.
    """
    raw = ensemble_predict(artifacts.ensemble, X_shifted_raw)
    cal = artifacts.iso_cal.predict_proba(raw)
    d1 = cal
    X_shifted_sc = artifacts.scaler.transform(X_shifted_imputed)
    d3 = artifacts.cohort_sim.transform(X_shifted_sc)
    # Composite v2: D1' and D3, D5' recomputed under shift; D2', D4 held at test values.
    d5 = -mc_dropout_variance(artifacts.dropout_fn, X_shifted_imputed)
    other = np.column_stack([artifacts.dims_test.D2, d3, artifacts.dims_test.D4, d5])
    r = artifacts.composite.score(d1, other)
    return cal, r


def run_missingness(artifacts, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_features = artifacts.X_test.shape[1]
    rows = []
    for drop_rate in DROP_RATES:
        n_reps = 1 if drop_rate == 0.0 else N_REPLICATES
        for rep in range(n_reps):
            X_shifted_raw = artifacts.X_test_raw.copy()
            X_shifted_imp = artifacts.X_test.copy()
            if drop_rate > 0.0:
                drop_cols = rng.choice(n_features, size=int(n_features * drop_rate), replace=False)
                X_shifted_raw[:, drop_cols] = 0.0
                X_shifted_imp[:, drop_cols] = 0.0
            cal_shifted, r_shifted = _reinfer(artifacts, X_shifted_raw, X_shifted_imp)
            scores = _score_gates(artifacts.y_test, cal_shifted, r_shifted)
            for policy, m in scores.items():
                rows.append({"task": artifacts.task, "experiment": "missingness",
                             "magnitude": drop_rate, "replicate": rep, "policy": policy, **m})
        logger.info("[%s] missingness drop_rate=%.1f: conf brier=%.4f sym brier=%.4f rel brier=%.4f",
                    artifacts.task, drop_rate, rows[-3]["brier"], rows[-2]["brier"], rows[-1]["brier"])
    return pd.DataFrame(rows)


def run_correlated_shift(artifacts, panel_name, prefixes, seed=42) -> pd.DataFrame:
    col_idx = _panel_columns(artifacts.feature_names, prefixes)
    if not col_idx:
        logger.warning("No columns found for panel %s (prefixes=%s)", panel_name, prefixes)
        return pd.DataFrame()
    train_std = artifacts.X_train[:, col_idx].std(axis=0)
    rng = np.random.default_rng(seed)
    rows = []
    for mag in SHIFT_MAGNITUDES:
        n_reps = 1 if mag == 0.0 else N_REPLICATES
        for rep in range(n_reps):
            X_shifted_raw = artifacts.X_test_raw.copy()
            X_shifted_imp = artifacts.X_test.copy()
            direction = rng.choice([-1.0, 1.0]) if mag > 0.0 else 1.0
            offset = direction * mag * train_std
            X_shifted_raw[:, col_idx] = X_shifted_raw[:, col_idx] + offset
            X_shifted_imp[:, col_idx] = X_shifted_imp[:, col_idx] + offset
            cal_shifted, r_shifted = _reinfer(artifacts, X_shifted_raw, X_shifted_imp)
            scores = _score_gates(artifacts.y_test, cal_shifted, r_shifted)
            for policy, m in scores.items():
                rows.append({"task": artifacts.task, "experiment": f"correlated_{panel_name}",
                             "magnitude": mag, "replicate": rep, "policy": policy, **m})
        logger.info("[%s] correlated %s shift=%.1fSD: conf brier=%.4f sym brier=%.4f rel brier=%.4f",
                    artifacts.task, panel_name, mag, rows[-3]["brier"], rows[-2]["brier"], rows[-1]["brier"])
    return pd.DataFrame(rows)


def run_task(task: str) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("Real shift experiments: %s", task)
    logger.info("=" * 60)
    artifacts = train_pipeline(task)

    dfs = [run_missingness(artifacts)]
    dfs.append(run_correlated_shift(artifacts, "renal", ["lab_creatinine_", "lab_bun_"]))
    dfs.append(run_correlated_shift(artifacts, "liver", ["lab_alt_", "lab_ast_", "lab_bilirubin_"]))
    return pd.concat(dfs, ignore_index=True)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_dfs = []
    for task in ["sepsis", "mortality", "vasopressor"]:
        all_dfs.append(run_task(task))
    df = pd.concat(all_dfs, ignore_index=True)
    out_path = RESULTS_DIR / "shift_experiments_real.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved %d rows to %s", len(df), out_path)

    summary = df.groupby(["task", "experiment", "magnitude", "policy"]).agg(
        brier_mean=("brier", "mean"), auroc_mean=("auroc", "mean")).reset_index()
    summary.to_csv(RESULTS_DIR / "shift_experiments_real_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
