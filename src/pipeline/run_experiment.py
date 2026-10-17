"""End-to-end experiment pipeline.

Usage:
    python -m src.pipeline.run_experiment
    python -m src.pipeline.run_experiment experiment.task=mortality
"""
import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler

from src.calibration.baselines import (
    IsotonicCalibrator,
    PlattScaling,
    TemperatureScaling,
    UniformConfidence,
    save_calibrator,
)
from src.calibration.metrics import bootstrap_calibration, bootstrap_delta_ece
from src.calibration.r_conditioned import RConditionedCalibrator
from src.data.loader import MIMICLoader
from src.data.splits import temporal_split
from src.models.xgboost_model import (
    ensemble_predict,
    make_dropout_predict_fn,
    train_ensemble,
)
from src.reliability.cohort_similarity import CohortSimilarityEstimator
from src.reliability.composite import DimensionMatrix, ReliabilityComposite
from src.reliability.data_sufficiency import batch_compute_d2
from src.reliability.temporal_stability import batch_compute_d4
from src.reliability.uncertainty import compute_uncertainty_consistency

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    artifacts = Path(cfg.paths.artifacts_dir)
    results_dir = Path(cfg.paths.results_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Data ────────────────────────────────────────────────────────
    logger.info("=== Phase 1: Loading data ===")
    loader = MIMICLoader(Path(cfg.paths.data_dir))
    dataset = loader.load(cfg.experiment.task)
    splits = temporal_split(
        dataset,
        train_end=cfg.experiment.split.train_end,
        val_end=cfg.experiment.split.val_end,
        test_end=cfg.experiment.split.test_end,
    )

    X_train = splits.train.features.values
    y_train = splits.train.labels.values
    X_val = splits.val.features.values
    y_val = splits.val.labels.values
    X_test = splits.test.features.values
    y_test = splits.test.labels.values

    # ── Phase 2: Base model ───────────────────────────────────────────────────
    logger.info("=== Phase 2: Training base model (5 seeds) ===")
    ensemble = train_ensemble(
        X_train, y_train, X_val, y_val,
        cfg.model, seeds=list(cfg.seeds)
    )
    primary_model = ensemble[0].model

    raw_val = ensemble_predict(ensemble, X_val)
    raw_test = ensemble_predict(ensemble, X_test)

    # ── Phase 3: Freeze training statistics ───────────────────────────────────
    logger.info("=== Phase 3: Fitting & freezing training-set statistics ===")
    scaler = StandardScaler().fit(X_train)
    X_train_sc = scaler.transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    cohort_sim = CohortSimilarityEstimator().fit(X_train_sc)
    cohort_sim.save(artifacts / "cohort_similarity.pkl")

    # ── Phase 4: R dimensions ─────────────────────────────────────────────────
    logger.info("=== Phase 4: Computing reliability dimensions ===")

    # D1: calibrated confidence (fitted from val — use isotonic as D1 proxy)
    iso_for_d1 = IsotonicCalibrator().fit(raw_val, y_val)
    d1_val = iso_for_d1.predict_proba(raw_val)
    d1_test = iso_for_d1.predict_proba(raw_test)

    # D2: data sufficiency
    d2_val = batch_compute_d2(
        splits.val.timeseries, splits.val.features.index,
        splits.val.admission_times, splits.val.prediction_times,
        lookback_hours=cfg.experiment.reliability.lookback_hours,
        baseline_hours=cfg.experiment.reliability.baseline_hours,
    )
    d2_test = batch_compute_d2(
        splits.test.timeseries, splits.test.features.index,
        splits.test.admission_times, splits.test.prediction_times,
        lookback_hours=cfg.experiment.reliability.lookback_hours,
        baseline_hours=cfg.experiment.reliability.baseline_hours,
    )

    # D3: cohort similarity (frozen training statistics)
    d3_val = cohort_sim.transform(X_val_sc)
    d3_test = cohort_sim.transform(X_test_sc)

    # D4: temporal stability
    d4_val = batch_compute_d4(
        splits.val.timeseries, splits.val.features.index,
        splits.val.prediction_times,
        lookback_hours=cfg.experiment.reliability.lookback_hours,
    )
    d4_test = batch_compute_d4(
        splits.test.timeseries, splits.test.features.index,
        splits.test.prediction_times,
        lookback_hours=cfg.experiment.reliability.lookback_hours,
    )

    # D5: uncertainty consistency (MC Dropout approximation)
    dropout_fn = make_dropout_predict_fn(
        primary_model,
        dropout_rate=cfg.experiment.reliability.dropout_rate,
        seed=cfg.seed,
    )
    d5_val = compute_uncertainty_consistency(
        dropout_fn, X_val, d1_val,
        n_passes=cfg.experiment.reliability.mc_dropout_passes,
    )
    d5_test = compute_uncertainty_consistency(
        dropout_fn, X_test, d1_test,
        n_passes=cfg.experiment.reliability.mc_dropout_passes,
    )

    dims_val = DimensionMatrix(d1_val, d2_val, d3_val, d4_val, d5_val)
    dims_test = DimensionMatrix(d1_test, d2_test, d3_test, d4_test, d5_test)

    # ── Phase 5: R composite (validation set only) ────────────────────────────
    logger.info("=== Phase 5: Fitting R composite ===")
    val_cal_err = np.abs(d1_val - y_val)
    composite = ReliabilityComposite().fit(dims_val, val_cal_err)
    composite.save(artifacts / "reliability_composite.pkl")

    r_val = composite.transform(dims_val)
    r_test = composite.transform(dims_test)

    # ── Phase 6: Calibration baselines (all fitted on val only) ───────────────
    logger.info("=== Phase 6: Fitting calibration baselines ===")
    val_logits = np.log(raw_val / (1 - raw_val + 1e-12))

    temp_cal = TemperatureScaling().fit(val_logits, y_val)
    platt_cal = PlattScaling().fit(raw_val, y_val)
    iso_cal = IsotonicCalibrator().fit(raw_val, y_val)
    unif_cal = UniformConfidence().fit(y_val)
    r_cal = RConditionedCalibrator(
        n_bins=cfg.experiment.reliability.n_tertile_bins
    ).fit(r_val, raw_val, y_val)

    test_logits = np.log(raw_test / (1 - raw_test + 1e-12))
    ensemble_test = ensemble_predict(ensemble, X_test)

    predictions = {
        "uncalibrated": raw_test,
        "uniform": unif_cal.predict_proba(len(y_test)),
        "temperature": temp_cal.predict_proba(test_logits),
        "platt": platt_cal.predict_proba(raw_test),
        "isotonic": iso_cal.predict_proba(raw_test),
        "ensemble": ensemble_test,
        "r_conditioned": r_cal.predict_proba(r_test, raw_test),
    }

    # ── Phase 7: Evaluation (test set — one pass) ─────────────────────────────
    logger.info("=== Phase 7: Test set evaluation ===")
    rows = []
    for method, probs in predictions.items():
        result = bootstrap_calibration(
            probs, y_test,
            n_bootstrap=cfg.experiment.eval.n_bootstrap,
            n_bins=cfg.experiment.eval.ece_n_bins,
        )
        delta = bootstrap_delta_ece(
            iso_cal.predict_proba(raw_test), probs, y_test,
            n_bootstrap=cfg.experiment.eval.n_bootstrap,
            n_bins=cfg.experiment.eval.ece_n_bins,
        )
        rows.append({
            "method": method,
            "mece": result.mece,
            "mece_ci": f"[{result.mece_ci_low:.4f}, {result.mece_ci_high:.4f}]",
            "auroc": result.auroc,
            "nll": result.nll,
            "delta_ece_vs_isotonic": delta["delta_ece"],
            "delta_ci": f"[{delta['ci_low']:.4f}, {delta['ci_high']:.4f}]",
            "conclusion": delta["conclusion"],
        })

    results_df = pd.DataFrame(rows)
    logger.info("\n%s", results_df.to_string(index=False))
    results_df.to_csv(results_dir / f"{cfg.experiment.task}_results.csv", index=False)

    # ── Phase 8: Ablations ────────────────────────────────────────────────────
    logger.info("=== Phase 8: Ablations ===")
    ablation_rows = []
    dim_names = ["D1_confidence", "D2_sufficiency", "D3_similarity",
                 "D4_stability", "D5_uncertainty"]

    for i, name in enumerate(dim_names):
        r_single_val = composite.single_dim_transform(dims_val, i)
        r_single_test = composite.single_dim_transform(dims_test, i)
        r_cal_single = RConditionedCalibrator(
            n_bins=cfg.experiment.reliability.n_tertile_bins
        ).fit(r_single_val, raw_val, y_val)
        probs = r_cal_single.predict_proba(r_single_test, raw_test)
        result = bootstrap_calibration(probs, y_test,
                                       n_bootstrap=cfg.experiment.eval.n_bootstrap)
        ablation_rows.append({"model": f"R({name})", "mece": result.mece, "auroc": result.auroc})

    r_eq_val = composite.equal_weight_transform(dims_val)
    r_eq_test = composite.equal_weight_transform(dims_test)
    r_cal_eq = RConditionedCalibrator(
        n_bins=cfg.experiment.reliability.n_tertile_bins
    ).fit(r_eq_val, raw_val, y_val)
    probs_eq = r_cal_eq.predict_proba(r_eq_test, raw_test)
    result_eq = bootstrap_calibration(probs_eq, y_test,
                                      n_bootstrap=cfg.experiment.eval.n_bootstrap)
    ablation_rows.append({"model": "R(equal_weight)", "mece": result_eq.mece,
                          "auroc": result_eq.auroc})
    ablation_rows.append({"model": "R(full_learned)", "mece": results_df[
        results_df.method == "r_conditioned"]["mece"].values[0],
        "auroc": results_df[results_df.method == "r_conditioned"]["auroc"].values[0]})

    ablation_df = pd.DataFrame(ablation_rows)
    logger.info("\nAblations:\n%s", ablation_df.to_string(index=False))
    ablation_df.to_csv(results_dir / f"{cfg.experiment.task}_ablations.csv", index=False)

    logger.info("=== Experiment complete. Results saved to %s ===", results_dir)


if __name__ == "__main__":
    main()
