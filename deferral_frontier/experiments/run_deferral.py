"""Main experiment: evaluate deferral policies on MIMIC-IV.

Usage:
    python -m deferral_frontier.experiments.run_deferral sepsis
    python -m deferral_frontier.experiments.run_deferral mortality
    python -m deferral_frontier.experiments.run_deferral vasopressor
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
from deferral_frontier.src.decision_policy import (
    ConfidenceGate,
    ReliabilityGate,
    DimensionGate,
)
from deferral_frontier.src.shift_experiment import run_synthetic_shift_all_tasks
from deferral_frontier.src.splits import load_patient_split
from deferral_frontier.src.reliability_v2 import build_v2, patient_groups
from deferral_frontier.src.diagnostics import training_diagnostics
from deferral_frontier.src.metrics import (
    risk_coverage_curve,
    auroc_at_coverage,
    deferral_precision,
    bootstrap_ci_brier,
    bootstrap_ci_auroc,
    decision_curve,
    number_needed_to_review,
    false_positive_burden,
)

logger = logging.getLogger(__name__)

# Model hyperparameters (reuse from missingness paper)
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

DATA_DIR = Path("data/mimic4")
ARTIFACTS_DIR = Path("artifacts")
RESULTS_DIR = Path("deferral_frontier/results")
COVERAGE_TARGETS = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
SEEDS = [0, 1, 2, 3, 4]


def run_task(task: str) -> None:
    logger.info("=" * 60)
    logger.info("Task: %s", task)
    logger.info("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Data ────────────────────────────────────────────────────────
    logger.info("[1/7] Loading data")
    loader = MIMICLoader(DATA_DIR)
    dataset = loader.load(task)

    # Patient-level stratified 70/15/15 split (round-1 R1-1/R2-1): every patient
    # contributes to exactly one partition.
    idx_train, idx_val, idx_test = load_patient_split(DATA_DIR, dataset.labels)

    X_train = dataset.features.loc[idx_train].values.copy()
    X_val   = dataset.features.loc[idx_val].values.copy()
    X_test  = dataset.features.loc[idx_test].values.copy()
    y_train = dataset.labels.loc[idx_train].values
    y_val   = dataset.labels.loc[idx_val].values
    y_test  = dataset.labels.loc[idx_test].values
    logger.info("  Train: %d | Val: %d | Test: %d | Pos rate: %.3f",
                len(X_train), len(X_val), len(X_test), y_train.mean())

    # ── Phase 2: Base model ───────────────────────────────────────────────────
    logger.info("[2/7] Training XGBoost ensemble (%d seeds)", len(SEEDS))
    model_cfg = DictConfig(XGB_PARAMS)
    ensemble = train_ensemble(
        X_train, y_train, X_val, y_val,
        model_cfg, seeds=SEEDS,
    )
    primary_model = ensemble[0].model
    training_diagnostics(ensemble, X_train, y_train, X_val, y_val).assign(
        dataset="mimic4", task=task).to_csv(
        RESULTS_DIR / f"base_diagnostics_mimic4_{task}.csv", index=False)

    raw_val = ensemble_predict(ensemble, X_val)
    raw_test = ensemble_predict(ensemble, X_test)
    _, ensemble_sd_val = ensemble_predict_all_seeds(ensemble, X_val)
    _, ensemble_sd_test = ensemble_predict_all_seeds(ensemble, X_test)
    logger.info("  Raw AUROC val: %.4f | test: %.4f",
                _quick_auroc(y_val, raw_val), _quick_auroc(y_test, raw_test))

    # Impute NaN and freeze training statistics
    logger.info("[3/7] Imputing NaN and freezing training statistics")
    train_df = dataset.features.loc[idx_train]
    col_medians = train_df.median()
    for i in range(X_train.shape[1]):
        med = col_medians.iloc[i]
        if pd.isna(med):
            med = 0.0
        X_train[:, i] = np.where(np.isnan(X_train[:, i]), med, X_train[:, i])
        X_val[:, i]   = np.where(np.isnan(X_val[:, i]), med, X_val[:, i])
        X_test[:, i]  = np.where(np.isnan(X_test[:, i]), med, X_test[:, i])
    scaler = StandardScaler().fit(X_train)
    X_train_sc = scaler.transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    cohort_sim = CohortSimilarityEstimator().fit(X_train_sc)

    # ── Phase 4: R dimensions ─────────────────────────────────────────────────
    logger.info("[4/7] Computing reliability dimensions")

    # D1
    iso_for_d1 = IsotonicCalibrator().fit(raw_val, y_val)
    d1_val = iso_for_d1.predict_proba(raw_val)
    d1_test = iso_for_d1.predict_proba(raw_test)

    # D2: data sufficiency (fast groupby-based computation)
    ts = dataset.timeseries
    d2_val = compute_d2_fast(ts, pd.Index(idx_val), dataset.admission_times,
                              dataset.prediction_times, lookback_hours=24, baseline_hours=6)
    d2_test = compute_d2_fast(ts, pd.Index(idx_test), dataset.admission_times,
                               dataset.prediction_times, lookback_hours=24, baseline_hours=6)
    logger.info('  D2 mean: val=%.4f test=%.4f', d2_val.mean(), d2_test.mean())

    # D3: cohort similarity
    d3_val = cohort_sim.transform(X_val_sc)
    d3_test = cohort_sim.transform(X_test_sc)

    # D4: temporal stability (fast groupby-based computation)
    d4_val = compute_d4_fast(ts, pd.Index(idx_val), dataset.prediction_times, lookback_hours=24)
    d4_test = compute_d4_fast(ts, pd.Index(idx_test), dataset.prediction_times, lookback_hours=24)
    logger.info('  D4 mean: val=%.4f test=%.4f', d4_val.mean(), d4_test.mean())

    # D5
    dropout_fn = make_dropout_predict_fn(primary_model, dropout_rate=0.1, seed=42)
    d5_val = compute_uncertainty_consistency(dropout_fn, X_val, d1_val, n_passes=50)
    d5_test = compute_uncertainty_consistency(dropout_fn, X_test, d1_test, n_passes=50)

    # Composite v2 (pre-registered: plan/r1-composite-v2-preregistration.md)
    val_cal_err = np.abs(d1_val - y_val)
    composite, dims_val, dims_test, r_val, r_test, other_test = build_v2(
        raw_val, y_val, d1_val, d1_test, X_val, X_test, list(dataset.features.columns),
        d3_val, d3_test, d4_val, d4_test, dropout_fn, patient_groups(DATA_DIR, idx_val))
    pd.Series(composite.importance()).to_csv(
        RESULTS_DIR / f"composite_v2_importance_mimic4_{task}.csv")
    logger.info("  R score range: [%.3f, %.3f] (val) [%.3f, %.3f] (test)",
                r_val.min(), r_val.max(), r_test.min(), r_test.max())

    # ── Phase 6: Calibration ─────────────────────────────────────────────────
    logger.info("[6/7] Calibrating predictions")
    iso_cal = IsotonicCalibrator().fit(raw_val, y_val)
    cal_val = iso_cal.predict_proba(raw_val)
    cal_test = iso_cal.predict_proba(raw_test)

    # Save per-patient artifacts for post-hoc analysis (conformal, fairness)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = pd.DataFrame({
        "hadm_id": idx_test,
        "y_true": y_test,
        "cal_proba": cal_test,
        "raw_proba": raw_test,
        "ensemble_sd": ensemble_sd_test,
        "r_score": r_test,
        "D1": dims_test.D1, "D2": dims_test.D2, "D3": dims_test.D3,
        "D4": dims_test.D4, "D5": dims_test.D5,
    })
    artifact.to_parquet(ARTIFACTS_DIR / f"{task}_test_artifacts.parquet", index=False)
    logger.info("  Saved test artifacts (%d rows)", len(artifact))

    val_artifact = pd.DataFrame({
        "hadm_id": idx_val,
        "y_true": y_val,
        "cal_proba": cal_val,
        "raw_proba": raw_val,
        "ensemble_sd": ensemble_sd_val,
        "r_score": r_val,
        "D1": dims_val.D1, "D2": dims_val.D2, "D3": dims_val.D3,
        "D4": dims_val.D4, "D5": dims_val.D5,
    })
    val_artifact.to_parquet(ARTIFACTS_DIR / f"{task}_val_artifacts.parquet", index=False)

    # Coefficient stability: bootstrap R composite
    logger.info("  Bootstrapping R composite coefficients")
    # (logistic-composite coefficient analysis retired with composite v2)

    # ── Phase 7: Deferral evaluation ─────────────────────────────────────────
    logger.info("[7/7] Evaluating deferral policies")

    # (equal-weight ablation retired with composite v2; see run_d1d5_ablation)

    all_policies = [
        ("confidence", ConfidenceGate()),
        ("reliability", ReliabilityGate()),
        ("conformal_variance", type("_", (), {"_score": lambda s, p, r, d: -ensemble_sd_test})()),
        ("D2_sufficiency", DimensionGate(1, "sufficiency")),
        ("D3_similarity", DimensionGate(2, "similarity")),
        ("D4_stability", DimensionGate(3, "stability")),
        ("D5_uncertainty", DimensionGate(4, "uncertainty")),
    ]
    class _ConformalVarianceGate:
        def _score(self, proba, r, dims):
            return -ensemble_sd_test if len(ensemble_sd_test) == len(proba) else -r
    all_policies[2] = ("conformal_variance", _ConformalVarianceGate())

    all_rows = []
    for policy_name, policy in all_policies:
        val_score = policy._score(cal_val, r_val, dims_val)
        test_score = policy._score(cal_test, r_test, dims_test)

        curve = risk_coverage_curve(y_test, cal_test, test_score, COVERAGE_TARGETS)

        for pt in curve:
            cov = pt["coverage"]
            auroc = auroc_at_coverage(y_test, cal_test, test_score, cov)
            all_rows.append({
                "task": task,
                "policy": policy_name,
                "coverage": pt["coverage"],
                "brier": pt["brier"],
                "auroc": auroc,
                "ece": pt["ece"],
                "n_predicted": pt["n_predicted"],
            })

        # Deferral precision at 80% coverage
        n80 = int(np.round(len(y_test) * 0.80))
        if n80 < len(y_test):
            thresh = float(np.sort(-test_score)[n80 - 1])
            dm = test_score < -thresh
            dp = deferral_precision(y_test, cal_test, dm)
            all_rows.append({
                "task": task, "policy": policy_name,
                "coverage": 0.80, "deferral_precision": dp,
                "brier": np.nan, "auroc": np.nan, "ece": np.nan,
                "n_predicted": n80,
            })

        logger.info("  %s: AUROC@80%%=%.4f, brier@80%%=%.4f",
                    policy_name,
                    auroc_at_coverage(y_test, cal_test, test_score, 0.80),
                    next((p["brier"] for p in curve if p["coverage"] == 0.80), np.nan))

    results_df = pd.DataFrame(all_rows)
    out_path = RESULTS_DIR / f"{task}_deferral_results.csv"
    results_df.to_csv(out_path, index=False)
    logger.info("Saved results to %s", out_path)

    # Bootstrap CIs at 80% coverage
    logger.info("[8/8] Computing bootstrap CIs at 80% coverage")
    ci_rows = []
    for policy_name, policy in all_policies:
        test_score = policy._score(cal_test, r_test, dims_test)
        n80 = int(np.round(len(y_test) * 0.80))
        if n80 < 2:
            continue
        idx80 = np.argsort(-test_score)[:n80]
        y80 = y_test[idx80]
        p80 = cal_test[idx80]
        brier, b_lo, b_hi = bootstrap_ci_brier(y80, p80, n_bootstrap=1000)
        auroc, a_lo, a_hi = bootstrap_ci_auroc(y80, p80, n_bootstrap=1000)
        ci_rows.append({
            "task": task, "policy": policy_name,
            "brier": brier, "brier_ci_lo": b_lo, "brier_ci_hi": b_hi,
            "auroc": auroc, "auroc_ci_lo": a_lo, "auroc_ci_hi": a_hi,
        })
    ci_df = pd.DataFrame(ci_rows)
    ci_path = RESULTS_DIR / f"{task}_deferral_ci.csv"
    ci_df.to_csv(ci_path, index=False)
    logger.info("Saved bootstrap CIs to %s", ci_path)

    # Deferred-case overlap analysis
    logger.info("  Deferred-case overlap analysis")
    _deferred_overlap_analysis(y_test, cal_test, r_test, dims_val, dims_test, task)

    # Decision curve / clinical utility analysis
    logger.info("  Clinical utility analysis")
    _clinical_utility_analysis(y_test, cal_test, r_test, dims_val, dims_test, task)

    # Dimension collinearity analysis
    logger.info("  Dimension collinearity analysis")
    # (logistic-composite coefficient analysis retired with composite v2)

    # Synthetic shift experiment (missingness amplification)
    logger.info("  Synthetic shift experiment")
    try:
        run_synthetic_shift_all_tasks(RESULTS_DIR, task, X_test, y_test, cal_test, r_test)
    except Exception as e:
        logger.warning("  Shift experiment failed: %s", e)

    # Phase 10: Case studies --------------------------------------------------------------------
    _extract_case_studies(y_test, cal_test, r_test, dims_test, task)

    # Print summary
    _print_summary(results_df, task)


def _bootstrap_r_coefficients(
    dims_val: DimensionMatrix,
    val_cal_err: np.ndarray,
    results_dir: Path,
    task: str,
    n_bootstrap: int = 100,
) -> None:
    """Bootstrap the logistic regression to estimate coefficient stability."""
    from src.reliability.composite import ReliabilityComposite, DIMENSION_NAMES
    rng = np.random.default_rng(42)
    n = len(val_cal_err)
    coefs = {name: [] for name in DIMENSION_NAMES}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        dims_boot = DimensionMatrix(
            dims_val.D1[idx], dims_val.D2[idx], dims_val.D3[idx],
            dims_val.D4[idx], dims_val.D5[idx],
        )
        err_boot = val_cal_err[idx]
        try:
            comp = ReliabilityComposite().fit(dims_boot, err_boot)
            for name, c in zip(DIMENSION_NAMES, comp._lr.coef_[0]):
                coefs[name].append(float(c))
        except Exception:
            continue
    summary = []
    for name in DIMENSION_NAMES:
        vals = coefs[name]
        if vals:
            summary.append({
                "dimension": name,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "ci_lo": float(np.percentile(vals, 2.5)),
                "ci_hi": float(np.percentile(vals, 97.5)),
            })
    coef_df = pd.DataFrame(summary)
    coef_path = results_dir / f"{task}_coef_stability.csv"
    coef_df.to_csv(coef_path, index=False)
    logger.info("  Coefficient stability (mean ± SD):")
    for _, r in coef_df.iterrows():
        logger.info("    %s: %.2f ± %.2f [%.2f, %.2f]",
                    r["dimension"], r["mean"], r["std"], r["ci_lo"], r["ci_hi"])


def _deferred_overlap_analysis(
    y_test: np.ndarray, cal_test: np.ndarray, r_test: np.ndarray,
    dims_val: DimensionMatrix, dims_test: DimensionMatrix, task: str,
) -> None:
    """Show that R-deferred cases are clinically distinct from confidence-deferred."""
    # Defer bottom 20% by each gate
    n80 = int(np.round(len(y_test) * 0.80))
    conf_deferred = np.argsort(-cal_test)[n80:]
    r_deferred = np.argsort(-r_test)[n80:]

    conf_set = set(conf_deferred)
    r_set = set(r_deferred)
    intersection = conf_set & r_set
    union = conf_set | r_set
    jaccard = len(intersection) / len(union) if len(union) > 0 else 0.0
    only_conf = conf_set - r_set
    only_r = r_set - conf_set

    logger.info("    Jaccard(conf_deferred, R_deferred)=%.3f", jaccard)
    logger.info("    Deferred by confidence only: %d", len(only_conf))
    logger.info("    Deferred by R only: %d", len(only_r))
    logger.info("    Both: %d", len(intersection))

    # Show R-only deferred cases have lower data sufficiency
    dim_array = dims_test.to_array()
    if len(only_r) > 0:
        r_only_mean = dim_array[list(only_r)].mean(axis=0)
        conf_only_mean = dim_array[list(only_conf)].mean(axis=0) if only_conf else np.zeros(5)
        logger.info("    R-only deferred: D2(suff)=%.4f, D3(sim)=%.4f",
                    r_only_mean[1], r_only_mean[2])
        logger.info("    Conf-only deferred: D2(suff)=%.4f, D3(sim)=%.4f",
                    conf_only_mean[1], conf_only_mean[2])

    overlap = pd.DataFrame([{
        "task": task,
        "jaccard": jaccard,
        "n_conf_only": len(only_conf),
        "n_r_only": len(only_r),
        "n_both": len(intersection),
        "r_only_D2_mean": float(dim_array[list(only_r), 1].mean()) if len(only_r) > 0 else 0,
        "r_only_D3_mean": float(dim_array[list(only_r), 2].mean()) if len(only_r) > 0 else 0,
        "conf_only_D2_mean": float(dim_array[list(only_conf), 1].mean()) if len(only_conf) > 0 else 0,
        "conf_only_D3_mean": float(dim_array[list(only_conf), 2].mean()) if len(only_conf) > 0 else 0,
    }])
    overlap.to_csv(RESULTS_DIR / f"{task}_deferred_overlap.csv", index=False)


def _extract_case_studies(
    y_test: np.ndarray, cal_test: np.ndarray, r_test: np.ndarray,
    dims_test: DimensionMatrix, task: str,
) -> None:
    dim_array = dims_test.to_array()
    cases = []

    # High-confidence / Low-R
    mask_hclr = (cal_test >= 0.8) & (r_test <= 0.3)
    hclr_idx = np.where(mask_hclr)[0]
    for idx in hclr_idx[:2]:
        cases.append({
            "task": task, "category": "high_conf_low_r",
            "confidence": float(cal_test[idx]),
            "reliability": float(r_test[idx]),
            "true_label": int(y_test[idx]),
            **{f"dim_D{i+1}": float(dim_array[idx, i]) for i in range(5)},
        })

    # Low-confidence / High-R
    mask_lchr = (cal_test < 0.3) & (r_test >= 0.7)
    lchr_idx = np.where(mask_lchr)[0]
    for idx in lchr_idx[:2]:
        cases.append({
            "task": task, "category": "low_conf_high_r",
            "confidence": float(cal_test[idx]),
            "reliability": float(r_test[idx]),
            "true_label": int(y_test[idx]),
            **{f"dim_D{i+1}": float(dim_array[idx, i]) for i in range(5)},
        })

    case_df = pd.DataFrame(cases)
    if len(case_df) > 0:
        case_path = RESULTS_DIR / f"{task}_case_studies.csv"
        case_df.to_csv(case_path, index=False)
        logger.info("Case studies: %d exemplars saved", len(case_df))


def _quick_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def _print_summary(df: pd.DataFrame, task: str) -> None:
    print(f"\n{'='*60}")
    print(f"  DEFERRAL FRONTIER — {task.upper()}")
    print(f"{'='*60}")
    for policy in df["policy"].unique():
        sub = df[(df["policy"] == policy) & (df["coverage"] == 0.80)]
        if len(sub) == 0:
            continue
        s = sub.iloc[0]
        print(f"  {policy:20s} | AUROC={s['auroc']:.4f} | Brier={s['brier']:.4f}")
    print(f"{'='*60}\n")


def _clinical_utility_analysis(
    y_test: np.ndarray, cal_test: np.ndarray, r_test: np.ndarray,
    dims_val: DimensionMatrix, dims_test: DimensionMatrix, task: str,
) -> None:
    """Decision curve analysis, NNR, false-positive burden."""
    from deferral_frontier.src.metrics import decision_curve, number_needed_to_review, false_positive_burden
    rows = []
    for policy_name, test_score in [("confidence", cal_test), ("reliability", r_test)]:
        n80 = int(np.round(len(y_test) * 0.80))
        dm = np.zeros(len(y_test), dtype=bool)
        if n80 < len(y_test):
            thresh = float(np.sort(-test_score)[n80 - 1])
            dm = test_score < -thresh
        pred_idx = ~dm
        y_sel = y_test[pred_idx]
        p_sel = cal_test[pred_idx]
        nnr, n_def, err_rate = number_needed_to_review(y_test, cal_test, dm)
        fpb = false_positive_burden(y_test, cal_test, dm)
        dc = decision_curve(y_sel, p_sel, thresholds=np.array([0.05, 0.1, 0.2]))
        nb_01 = dc['curves'][1]['net_benefit'] if len(dc['curves']) > 1 else 0.0
        rows.append({
            "task": task, "policy": policy_name,
            "n_deferred": n_def, "error_rate_deferred": err_rate,
            "nnr": nnr, "false_positive_burden": fpb,
            "net_benefit_at_01": nb_01,
        })
        logger.info("  %s: NNR=%.1f, FP_burden=%.4f, NB@0.1=%.4f",
                    policy_name, nnr, fpb, nb_01)
    util_df = pd.DataFrame(rows)
    util_path = RESULTS_DIR / f"{task}_clinical_utility.csv"
    util_df.to_csv(util_path, index=False)


def _collinearity_analysis(
    dims_val: DimensionMatrix, dims_test: DimensionMatrix, task: str,
) -> None:
    """VIF and correlation matrix for reliability dimensions."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    try:
        X = dims_test.to_array()
        corr = np.corrcoef(X.T)
        dim_names = ["D1_conf", "D2_suff", "D3_sim", "D4_stab", "D5_unc"]
        vif = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        logger.info("  Correlation matrix:")
        for i, name in enumerate(dim_names):
            corr_str = " ".join(f"{corr[i,j]:.3f}" for j in range(len(dim_names)))
            logger.info("    %s: %s", name, corr_str)
        logger.info("  VIF: %s", " ".join(f"{v:.2f}" for v in vif))
        coll_df = pd.DataFrame({"dimension": dim_names, "VIF": vif})
        coll_df.to_csv(RESULTS_DIR / f"{task}_collinearity.csv", index=False)
    except ImportError:
        logger.warning("  statsmodels not available; skipping VIF")
    except Exception as e:
        logger.warning("  Collinearity analysis failed: %s", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, nargs="?",
                        default="sepsis",
                        help="Task: sepsis, mortality, or vasopressor")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    run_task(args.task)
