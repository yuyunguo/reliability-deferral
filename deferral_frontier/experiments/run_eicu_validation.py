"""Run deferral pipeline on eICU for genuine cross-institutional external validation.

Reuses the exact same training/evaluation logic as run_mimic3_validation.py.
Unlike MIMIC-III (same hospital, BIDMC, as MIMIC-IV), eICU spans ~200 US
hospitals, so this is real external-institution validation, not temporal
replication within one institution.

eICU has no wall-clock timestamps (offsets are minutes from unit admission),
so admit_year is used only for reporting/stratification, not for a temporal
train/val/test split -- a random stratified split is used instead (the
MIMIC-III script's own fallback path for when a temporal split isn't
viable/available).
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler

from src.calibration.baselines import IsotonicCalibrator
from src.data.loader import MIMICLoader
from src.models.xgboost_model import (
    ensemble_predict,
    ensemble_predict_all_seeds,
    make_dropout_predict_fn,
    train_ensemble,
)
from src.reliability.cohort_similarity import CohortSimilarityEstimator
from src.reliability.composite import DimensionMatrix, ReliabilityComposite
from src.reliability.uncertainty import compute_uncertainty_consistency

from deferral_frontier.src.fast_reliability import compute_d2_fast, compute_d4_fast
from deferral_frontier.src.splits import load_patient_split
from deferral_frontier.src.reliability_v2 import build_v2, patient_groups
from deferral_frontier.src.diagnostics import training_diagnostics
from deferral_frontier.src.decision_policy import ConfidenceGate, ReliabilityGate
from deferral_frontier.src.metrics import (
    risk_coverage_curve,
    auroc_at_coverage,
    bootstrap_ci_brier,
    bootstrap_ci_auroc,
    number_needed_to_review,
    false_positive_burden,
)

logger = logging.getLogger(__name__)

XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "eval_metric": "logloss",
    "early_stopping_rounds": 20,
    "use_label_encoder": False,
    "random_state": 42,
    "n_jobs": -1,
}

DATA_DIR = Path("data/eicu")
RESULTS_DIR = Path("deferral_frontier/results")
ARTIFACTS_DIR = Path("artifacts/eicu")
COVERAGE_TARGETS = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
SEEDS = [0, 1, 2, 3, 4]


def _quick_auroc(y_true, y_prob):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def run_eicu_task(task: str, split: str = "patient"):
    """Run one eICU task; split is "patient" (default) or "hospital" (held-out hospitals)."""
    tag = "eicu" if split == "patient" else "eicu_hospital"
    art_dir = ARTIFACTS_DIR if split == "patient" else Path("artifacts/eicu_hospital")
    logger.info("=" * 60)
    logger.info("eICU External Validation (cross-institutional) — %s", task.upper())
    logger.info("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    loader = MIMICLoader(DATA_DIR)
    dataset = loader.load(task)

    # Patient-level stratified 70/15/15 split (round-1 R1-1/R2-1).
    if split == "hospital":
        idx_train, idx_val, idx_test = load_patient_split(
            DATA_DIR, dataset.labels, ids_file="hospital_ids.parquet", column="hospital_id")
    else:
        idx_train, idx_val, idx_test = load_patient_split(DATA_DIR, dataset.labels)

    X_train = dataset.features.loc[idx_train].values.copy()
    X_val = dataset.features.loc[idx_val].values.copy()
    X_test = dataset.features.loc[idx_test].values.copy()
    y_train = dataset.labels.loc[idx_train].values
    y_val = dataset.labels.loc[idx_val].values
    y_test = dataset.labels.loc[idx_test].values
    logger.info("  Train: %d | Val: %d | Test: %d | Pos: %.3f",
                len(X_train), len(X_val), len(X_test), y_train.mean())

    logger.info("[2/7] Training XGBoost ensemble")
    ensemble = train_ensemble(X_train, y_train, X_val, y_val, DictConfig(XGB_PARAMS), seeds=SEEDS)
    primary_model = ensemble[0].model
    training_diagnostics(ensemble, X_train, y_train, X_val, y_val).assign(
        dataset=tag, task=task).to_csv(
        RESULTS_DIR / f"base_diagnostics_{tag}_{task}.csv", index=False)

    raw_val = ensemble_predict(ensemble, X_val)
    raw_test = ensemble_predict(ensemble, X_test)
    _, ensemble_sd_val = ensemble_predict_all_seeds(ensemble, X_val)
    _, ensemble_sd_test = ensemble_predict_all_seeds(ensemble, X_test)
    logger.info("  Raw AUROC val: %.4f | test: %.4f",
                _quick_auroc(y_val, raw_val), _quick_auroc(y_test, raw_test))

    logger.info("[3/7] Imputing NaN")
    train_df = dataset.features.loc[idx_train]
    col_medians = train_df.median()
    for i in range(X_train.shape[1]):
        med = col_medians.iloc[i]
        if pd.isna(med):
            med = 0.0
        X_train[:, i] = np.where(np.isnan(X_train[:, i]), med, X_train[:, i])
        X_val[:, i] = np.where(np.isnan(X_val[:, i]), med, X_val[:, i])
        X_test[:, i] = np.where(np.isnan(X_test[:, i]), med, X_test[:, i])
    scaler = StandardScaler().fit(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    cohort_sim = CohortSimilarityEstimator().fit(scaler.transform(X_train))

    logger.info("[4/7] Computing reliability dimensions")
    iso_for_d1 = IsotonicCalibrator().fit(raw_val, y_val)
    d1_val = iso_for_d1.predict_proba(raw_val)
    d1_test = iso_for_d1.predict_proba(raw_test)

    ts = dataset.timeseries
    d2_val = compute_d2_fast(ts, pd.Index(idx_val), dataset.admission_times,
                              dataset.prediction_times, lookback_hours=24, baseline_hours=6)
    d2_test = compute_d2_fast(ts, pd.Index(idx_test), dataset.admission_times,
                               dataset.prediction_times, lookback_hours=24, baseline_hours=6)

    d3_val = cohort_sim.transform(X_val_sc)
    d3_test = cohort_sim.transform(X_test_sc)

    d4_val = compute_d4_fast(ts, pd.Index(idx_val), dataset.prediction_times, lookback_hours=24)
    d4_test = compute_d4_fast(ts, pd.Index(idx_test), dataset.prediction_times, lookback_hours=24)

    dropout_fn = make_dropout_predict_fn(primary_model, dropout_rate=0.1, seed=42)
    d5_val = compute_uncertainty_consistency(dropout_fn, X_val, d1_val, n_passes=50)
    d5_test = compute_uncertainty_consistency(dropout_fn, X_test, d1_test, n_passes=50)

    # Composite v2 (pre-registered: plan/r1-composite-v2-preregistration.md)
    val_cal_err = np.abs(d1_val - y_val)
    composite, dims_val, dims_test, r_val, r_test, other_test = build_v2(
        raw_val, y_val, d1_val, d1_test, X_val, X_test, list(dataset.features.columns),
        d3_val, d3_test, d4_val, d4_test, dropout_fn, patient_groups(DATA_DIR, idx_val))
    pd.Series(composite.importance()).to_csv(
        RESULTS_DIR / f"composite_v2_importance_{tag}_{task}.csv")

    # D1+D5 minimal composite
    from sklearn.linear_model import LogisticRegression
    X_d1d5_val = np.column_stack([d1_val, d5_val])
    X_d1d5_test = np.column_stack([d1_test, d5_test])
    y_binary_val = (val_cal_err > np.median(val_cal_err)).astype(int)
    sc_d1d5 = StandardScaler().fit(X_d1d5_val)
    lr_d1d5 = LogisticRegression(max_iter=1000, random_state=42).fit(
        sc_d1d5.transform(X_d1d5_val), y_binary_val)
    r_d1d5_test = lr_d1d5.predict_proba(sc_d1d5.transform(X_d1d5_test))[:, 0]

    logger.info("[6/7] Calibrating")
    iso_cal = IsotonicCalibrator().fit(raw_val, y_val)
    cal_val = iso_cal.predict_proba(raw_val)
    cal_test = iso_cal.predict_proba(raw_test)

    artifact = pd.DataFrame({
        "stay_id": idx_test,
        "y_true": y_test,
        "cal_proba": cal_test,
        "raw_proba": raw_test,
        "ensemble_sd": ensemble_sd_test,
        "r_score": r_test,
        "D1": dims_test.D1, "D2": dims_test.D2, "D3": dims_test.D3,
        "D4": dims_test.D4, "D5": dims_test.D5,
    })
    artifact.to_parquet(art_dir / f"{tag}_{task}_test_artifacts.parquet", index=False)

    logger.info("[7/7] Evaluating deferral policies")
    policies = {
        "confidence": cal_test,
        "reliability": r_test,
        "D1D5_minimal": r_d1d5_test,
        "conformal_variance": -ensemble_sd_test,
    }

    ci_rows, all_rows = [], []
    for pname, pscore in policies.items():
        for cov in COVERAGE_TARGETS:
            n = int(np.round(len(y_test) * cov))
            if n < 2:
                continue
            idx = np.argsort(-pscore)[:n]
            b = np.mean((y_test[idx] - cal_test[idx]) ** 2)
            a = _quick_auroc(y_test[idx], cal_test[idx])
            all_rows.append({"task": task, "policy": pname, "coverage": cov,
                              "brier": b, "auroc": a, "n_predicted": n})

        n80 = int(np.round(len(y_test) * 0.80))
        idx80 = np.argsort(-pscore)[:n80]
        y80, p80 = y_test[idx80], cal_test[idx80]
        brier, b_lo, b_hi = bootstrap_ci_brier(y80, p80, n_bootstrap=1000)
        auroc, a_lo, a_hi = bootstrap_ci_auroc(y80, p80, n_bootstrap=1000)
        ci_rows.append({"task": task, "policy": pname, "brier": brier,
                         "brier_ci_lo": b_lo, "brier_ci_hi": b_hi,
                         "auroc": auroc, "auroc_ci_lo": a_lo, "auroc_ci_hi": a_hi})
        logger.info("  %s: Brier=%.4f [%.4f,%.4f] AUROC=%.4f [%.4f,%.4f]",
                    pname, brier, b_lo, b_hi, auroc, a_lo, a_hi)

    pd.DataFrame(ci_rows).to_csv(RESULTS_DIR / f"{tag}_{task}_deferral_ci.csv", index=False)
    pd.DataFrame(all_rows).to_csv(RESULTS_DIR / f"{tag}_{task}_deferral_results.csv", index=False)

    n80 = int(np.round(len(y_test) * 0.80))
    conf_def = set(np.argsort(-cal_test)[n80:])
    r_def = set(np.argsort(-r_test)[n80:])
    jaccard = len(conf_def & r_def) / len(conf_def | r_def) if conf_def | r_def else 0
    logger.info("  Jaccard(conf, R) = %.4f", jaccard)

    dm_conf = np.zeros(len(y_test), dtype=bool)
    dm_conf[np.argsort(-cal_test)[n80:]] = True
    nnr_c, _, _ = number_needed_to_review(y_test, cal_test, dm_conf)
    dm_r = np.zeros(len(y_test), dtype=bool)
    dm_r[np.argsort(-r_test)[n80:]] = True
    nnr_r, _, _ = number_needed_to_review(y_test, cal_test, dm_r)
    logger.info("  NNR: confidence=%.1f, reliability=%.1f", nnr_c, nnr_r)

    print(f"\n{'='*60}\n  eICU (cross-institutional) — {task.upper()}\n{'='*60}")
    for _, row in pd.DataFrame(ci_rows).iterrows():
        print(f"  {row['policy']:20s} | Brier={row['brier']:.4f} [{row['brier_ci_lo']:.4f},{row['brier_ci_hi']:.4f}]"
              f" | AUROC={row['auroc']:.4f}")
    print(f"  Jaccard = {jaccard:.4f}\n{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", nargs="?", default="sepsis")
    parser.add_argument("--split", choices=["patient", "hospital"], default="patient")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_eicu_task(args.task, args.split)
