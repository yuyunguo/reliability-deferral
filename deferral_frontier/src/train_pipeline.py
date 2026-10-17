"""Reusable retraining pipeline shared by baseline-comparison and shift-experiment scripts.

Reproduces run_deferral.py's Phases 1-6 (data split, XGBoost ensemble, isotonic
calibration, cohort similarity, MC-dropout function, R composite fit) as an
importable function, so new experiments do not have to duplicate or diverge
from the validated main pipeline. run_deferral.py itself is left untouched.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler

from src.calibration.baselines import IsotonicCalibrator
from src.data.loader import MIMICLoader
from src.models.xgboost_model import (
    XGBResult,
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

logger = logging.getLogger(__name__)

XGB_PARAMS = {
    "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
    "reg_lambda": 1.0, "eval_metric": "logloss", "early_stopping_rounds": 20,
    "use_label_encoder": False, "random_state": 42, "n_jobs": -1,
}
SEEDS = [0, 1, 2, 3, 4]
DATA_DIR = Path("data/mimic4")


@dataclass
class PipelineArtifacts:
    task: str
    feature_names: List[str]
    ensemble: List[XGBResult]
    primary_model: object
    iso_cal: IsotonicCalibrator
    cohort_sim: CohortSimilarityEstimator
    scaler: StandardScaler
    dropout_fn: Callable[[np.ndarray], np.ndarray]
    composite: ReliabilityComposite
    col_medians: pd.Series
    dataset: object
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    X_train_raw: np.ndarray
    X_val_raw: np.ndarray
    X_test_raw: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    dims_val: DimensionMatrix
    dims_test: DimensionMatrix
    cal_val: np.ndarray
    cal_test: np.ndarray
    r_val: np.ndarray
    r_test: np.ndarray
    ensemble_sd_test: np.ndarray


def _impute(X_train, X_val, X_test, col_medians):
    for i in range(X_train.shape[1]):
        med = col_medians.iloc[i]
        med = 0.0 if pd.isna(med) else med
        X_train[:, i] = np.where(np.isnan(X_train[:, i]), med, X_train[:, i])
        X_val[:, i] = np.where(np.isnan(X_val[:, i]), med, X_val[:, i])
        X_test[:, i] = np.where(np.isnan(X_test[:, i]), med, X_test[:, i])


def train_pipeline(task: str, data_dir: Path = DATA_DIR, seeds: List[int] = None,
                   split_seed: int = 42, split_by: str = "patient",
                   drop_features: List[str] = None) -> PipelineArtifacts:
    """Retrain the full reliability pipeline for one task in `data_dir`.

    Mirrors run_deferral.py's patient-level split (seed `split_seed`) and Phases 2-6.
    drop_features: columns removed from every model's inputs (harmonized analysis);
    D2' is still computed from the observation counts, which are kept aside.
    """
    seeds = seeds if seeds is not None else SEEDS
    loader = MIMICLoader(data_dir)
    dataset = loader.load(task)
    counts_full = None
    if drop_features:
        counts_full = dataset.features[[c for c in dataset.features.columns if c.endswith("_count")]]
        dataset.features = dataset.features.drop(columns=list(drop_features))

    # Patient-level stratified 70/15/15 split (round-1 R1-1/R2-1).
    ids_file, column = (("hospital_ids.parquet", "hospital_id") if split_by == "hospital"
                        else ("patient_ids.parquet", "patient_id"))
    idx_train, idx_val, idx_test = load_patient_split(data_dir, dataset.labels, seed=split_seed,
                                                      ids_file=ids_file, column=column)

    X_train = dataset.features.loc[idx_train].values.copy()
    X_val = dataset.features.loc[idx_val].values.copy()
    X_test = dataset.features.loc[idx_test].values.copy()
    y_train = dataset.labels.loc[idx_train].values
    y_val = dataset.labels.loc[idx_val].values
    y_test = dataset.labels.loc[idx_test].values

    logger.info("[%s] Train: %d | Val: %d | Test: %d | Pos rate: %.3f",
                task, len(X_train), len(X_val), len(X_test), y_train.mean())

    logger.info("[%s] Training XGBoost ensemble (%d seeds)", task, len(seeds))
    ensemble = train_ensemble(X_train, y_train, X_val, y_val, DictConfig(XGB_PARAMS), seeds=seeds)
    primary_model = ensemble[0].model

    raw_val = ensemble_predict(ensemble, X_val)
    raw_test = ensemble_predict(ensemble, X_test)
    _, ensemble_sd_test = ensemble_predict_all_seeds(ensemble, X_test)

    # Ensemble is trained/predicted on NaN-preserving features (XGBoost's native
    # missing-value routing); D1/raw predictions must always be recomputed on this
    # pre-imputation representation. D3/D5 (and StandardScaler) require a complete
    # matrix, so imputation happens only after raw_val/raw_test are captured, and
    # the pre-imputation arrays are kept alongside the imputed ones for any
    # downstream re-inference (e.g. shift experiments) that needs to recompute D1.
    X_train_raw, X_val_raw, X_test_raw = X_train.copy(), X_val.copy(), X_test.copy()
    col_medians = dataset.features.loc[idx_train].median()
    _impute(X_train, X_val, X_test, col_medians)
    scaler = StandardScaler().fit(X_train)
    X_train_sc, X_val_sc, X_test_sc = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)

    cohort_sim = CohortSimilarityEstimator().fit(X_train_sc)

    iso_cal = IsotonicCalibrator().fit(raw_val, y_val)
    d1_val = iso_cal.predict_proba(raw_val)
    d1_test = iso_cal.predict_proba(raw_test)

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
        d3_val, d3_test, d4_val, d4_test, dropout_fn, patient_groups(data_dir, idx_val),
        counts_val=None if counts_full is None else counts_full.loc[idx_val].to_numpy(),
        counts_test=None if counts_full is None else counts_full.loc[idx_test].to_numpy())

    cal_val = iso_cal.predict_proba(raw_val)
    cal_test = iso_cal.predict_proba(raw_test)

    return PipelineArtifacts(
        task=task, feature_names=list(dataset.features.columns),
        ensemble=ensemble, primary_model=primary_model, iso_cal=iso_cal,
        cohort_sim=cohort_sim, scaler=scaler, dropout_fn=dropout_fn, composite=composite,
        col_medians=col_medians, dataset=dataset,
        idx_train=idx_train, idx_val=idx_val, idx_test=idx_test,
        X_train=X_train, X_val=X_val, X_test=X_test,
        X_train_raw=X_train_raw, X_val_raw=X_val_raw, X_test_raw=X_test_raw,
        y_train=y_train, y_val=y_val, y_test=y_test,
        dims_val=dims_val, dims_test=dims_test,
        cal_val=cal_val, cal_test=cal_test, r_val=r_val, r_test=r_test,
        ensemble_sd_test=ensemble_sd_test,
    )
