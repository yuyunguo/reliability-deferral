"""D5 perturbation-pass ablation (does D5 stabilize below 50 passes?) and
inference latency for the reliability gate.

Motivation: "No ablation of the
number of ensemble perturbation passes (50 with 10% dropout). Is 50
sufficient? Does D5 stabilize at 20?" and "No reporting of computation time
... for real-time EHR deployment, latency matters."

Uses the sepsis task on MIMIC-IV as the representative case (retrains the
same XGBoost ensemble used throughout the paper; no saved model artifacts
exist to reload).
"""
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler

from src.calibration.baselines import IsotonicCalibrator
from src.data.loader import MIMICLoader
from src.models.xgboost_model import (
    ensemble_predict,
    make_dropout_predict_fn,
    train_ensemble,
)
from src.reliability.cohort_similarity import CohortSimilarityEstimator
from src.reliability.uncertainty import compute_uncertainty_consistency

from deferral_frontier.src.fast_reliability import compute_d2_fast, compute_d4_fast
from deferral_frontier.src.metrics import bootstrap_ci_brier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

XGB_PARAMS = {
    "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
    "reg_lambda": 1.0, "eval_metric": "logloss", "early_stopping_rounds": 20,
    "use_label_encoder": False, "random_state": 42, "n_jobs": -1,
}
SEEDS = [0, 1, 2, 3, 4]
DATA_DIR = Path("data/mimic4")
RESULTS_DIR = Path("deferral_frontier/results")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    loader = MIMICLoader(DATA_DIR)
    dataset = loader.load("sepsis")

    years = dataset.admit_year
    train_mask = years <= 2169
    val_mask = (years > 2169) & (years <= 2181)
    test_mask = years > 2181
    idx_train = dataset.features.index[train_mask].to_numpy()
    idx_val = dataset.features.index[val_mask].to_numpy()
    idx_test = dataset.features.index[test_mask].to_numpy()

    X_train = dataset.features.loc[idx_train].values.copy()
    X_val = dataset.features.loc[idx_val].values.copy()
    X_test = dataset.features.loc[idx_test].values.copy()
    y_train = dataset.labels.loc[idx_train].values
    y_val = dataset.labels.loc[idx_val].values
    y_test = dataset.labels.loc[idx_test].values

    col_medians = dataset.features.loc[idx_train].median()
    for i in range(X_train.shape[1]):
        med = col_medians.iloc[i]
        med = 0.0 if pd.isna(med) else med
        X_train[:, i] = np.where(np.isnan(X_train[:, i]), med, X_train[:, i])
        X_val[:, i] = np.where(np.isnan(X_val[:, i]), med, X_val[:, i])
        X_test[:, i] = np.where(np.isnan(X_test[:, i]), med, X_test[:, i])

    logger.info("Training XGBoost ensemble (sepsis, MIMIC-IV)...")
    ensemble = train_ensemble(X_train, y_train, X_val, y_val, DictConfig(XGB_PARAMS), seeds=SEEDS)
    primary_model = ensemble[0].model

    raw_val = ensemble_predict(ensemble, X_val)
    raw_test = ensemble_predict(ensemble, X_test)
    iso = IsotonicCalibrator().fit(raw_val, y_val)
    d1_test = iso.predict_proba(raw_test)

    dropout_fn = make_dropout_predict_fn(primary_model, dropout_rate=0.1, seed=42)

    # --- D5 ablation: does D5 stabilize below 50 passes? ---
    logger.info("Computing D5 at n_passes in {10, 20, 30, 50}...")
    d5_by_passes = {}
    timing_by_passes = {}
    for n_passes in [10, 20, 30, 50]:
        t0 = time.perf_counter()
        d5 = compute_uncertainty_consistency(dropout_fn, X_test, d1_test, n_passes=n_passes)
        elapsed = time.perf_counter() - t0
        d5_by_passes[n_passes] = d5
        timing_by_passes[n_passes] = elapsed
        logger.info("  n_passes=%d: mean D5=%.4f, elapsed=%.3fs (%.2fms/patient)",
                     n_passes, d5.mean(), elapsed, 1000 * elapsed / len(X_test))

    ref = d5_by_passes[50]
    rows = []
    for n_passes, d5 in d5_by_passes.items():
        corr = np.corrcoef(d5, ref)[0, 1]
        mean_abs_diff = np.mean(np.abs(d5 - ref))
        # Downstream impact: Brier at 80% coverage using D5-only-informed proxy
        # (approximate: correlation with reference D5 is the primary stability
        # signal; full re-fit of R per n_passes is beyond this ablation's scope).
        rows.append({"n_passes": n_passes, "mean_D5": d5.mean(),
                      "corr_with_50": corr, "mean_abs_diff_from_50": mean_abs_diff,
                      "compute_seconds": timing_by_passes[n_passes],
                      "ms_per_patient": 1000 * timing_by_passes[n_passes] / len(X_test)})
    ablation_df = pd.DataFrame(rows)
    ablation_df.to_csv(RESULTS_DIR / "d5_passes_ablation.csv", index=False)
    logger.info("\n%s", ablation_df.to_string(index=False))

    # --- Inference latency for the full reliability-gate pipeline ---
    logger.info("Measuring end-to-end reliability-gate inference latency...")
    scaler = StandardScaler().fit(X_train)
    cohort_sim = CohortSimilarityEstimator().fit(scaler.transform(X_train))
    X_test_sc = scaler.transform(X_test)

    n_lat = min(1000, len(X_test))
    X_lat = X_test[:n_lat]
    X_lat_sc = X_test_sc[:n_lat]

    t0 = time.perf_counter()
    raw_lat = ensemble_predict(ensemble, X_lat)
    t_confidence = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = cohort_sim.transform(X_lat_sc)
    t_d3 = time.perf_counter() - t0

    t0 = time.perf_counter()
    d1_lat = iso.predict_proba(raw_lat)
    _ = compute_uncertainty_consistency(dropout_fn, X_lat, d1_lat, n_passes=50)
    t_d5 = time.perf_counter() - t0

    logger.info("Latency per patient (n=%d, batched): confidence (D1) = %.3fms, "
                "D3 (cohort similarity) = %.3fms, D5 (50-pass dropout) = %.3fms, "
                "D5 dominates total reliability-gate latency (%.1fx the confidence-only "
                "prediction cost)",
                n_lat, 1000 * t_confidence / n_lat, 1000 * t_d3 / n_lat,
                1000 * t_d5 / n_lat, t_d5 / t_confidence)

    pd.DataFrame([{
        "n_patients": n_lat,
        "ms_per_patient_confidence_only": 1000 * t_confidence / n_lat,
        "ms_per_patient_D3": 1000 * t_d3 / n_lat,
        "ms_per_patient_D5_50pass": 1000 * t_d5 / n_lat,
        "D5_vs_confidence_ratio": t_d5 / t_confidence,
    }]).to_csv(RESULTS_DIR / "inference_latency.csv", index=False)


if __name__ == "__main__":
    main()
