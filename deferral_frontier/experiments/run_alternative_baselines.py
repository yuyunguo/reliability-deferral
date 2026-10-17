"""Alternative uncertainty-quantification baselines for deferral comparison.

Motivation: "No Comparison to
Alternative Uncertainty Methods." The conformal-variance baseline
(run_conformal_baseline.py) already exists; this adds three further,
genuinely distinct UQ paradigms, all evaluated at matched coverage against
the confidence and reliability gates:

  1. MC-dropout-variance gate: raw predictive variance across n_passes
     stochastic forward passes (distinct from D5, which normalizes variance
     by the Bernoulli bound and combines it with D1 inside the composite).
  2. Temperature-scaling + entropy gate: Guo et al. (2017) temperature
     scaling, deferral by predictive (binary) entropy.
  3. Bayesian logistic regression gate: Laplace-approximated posterior over
     a logistic-regression base classifier fit on the same features;
     deferral by posterior-predictive logit variance (epistemic
     uncertainty), an entirely different UQ mechanism from ensembling or
     dropout.

All three retrain from the shared pipeline (train_pipeline.py) -- no reuse
of precomputed artifacts that would not reflect an independent UQ method.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from deferral_frontier.src.train_pipeline import train_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("deferral_frontier/results")
COVERAGE = 0.80
N_DROPOUT_PASSES = 50


def mc_dropout_variance_gate(artifacts) -> np.ndarray:
    """Raw predictive SD across stochastic forward passes (deferral score = -SD)."""
    passes = np.stack([artifacts.dropout_fn(artifacts.X_test) for _ in range(N_DROPOUT_PASSES)], axis=0)
    sd = passes.std(axis=0)
    return -sd


def temperature_entropy_gate(artifacts) -> tuple[np.ndarray, float]:
    """Temperature-scaled probabilities; deferral score = -binary entropy."""
    from src.calibration.baselines import TemperatureScaling
    from src.models.xgboost_model import ensemble_predict

    # Ensemble was trained/predicted on NaN-preserving features (XGBoost's native
    # missing-value routing); use the pre-imputation arrays here too, matching D1.
    raw_val = ensemble_predict(artifacts.ensemble, artifacts.X_val_raw)
    raw_test = ensemble_predict(artifacts.ensemble, artifacts.X_test_raw)
    eps = 1e-6
    logit_val = logit(np.clip(raw_val, eps, 1 - eps))
    logit_test = logit(np.clip(raw_test, eps, 1 - eps))

    temp = TemperatureScaling().fit(logit_val, artifacts.y_val)
    p_temp = temp.predict_proba(logit_test)
    p_c = np.clip(p_temp, eps, 1 - eps)
    entropy = -(p_c * np.log(p_c) + (1 - p_c) * np.log(1 - p_c))
    return -entropy, temp._T


def bayesian_logreg_gate(artifacts, prior_var: float = 1.0) -> np.ndarray:
    """Laplace-approximated Bayesian logistic regression; deferral score = -posterior logit variance.

    MAP fit via sklearn (L2 penalty C=1/prior_var), then Gaussian-Laplace
    posterior covariance Sigma = (X^T diag(p(1-p)) X + I/prior_var)^{-1}
    evaluated at the MAP weights, following Bishop (2006) Sec. 4.5.
    """
    X_train_sc = artifacts.scaler.transform(artifacts.X_train)
    X_test_sc = artifacts.scaler.transform(artifacts.X_test)

    lr = LogisticRegression(C=prior_var, max_iter=2000, random_state=42)
    lr.fit(X_train_sc, artifacts.y_train)
    w = lr.coef_[0]
    b = lr.intercept_[0]

    X_design = np.column_stack([X_train_sc, np.ones(len(X_train_sc))])
    w_full = np.concatenate([w, [b]])
    p_train = expit(X_design @ w_full)
    S = p_train * (1 - p_train)
    H = (X_design * S[:, None]).T @ X_design + np.eye(X_design.shape[1]) / prior_var
    Sigma = np.linalg.inv(H)

    X_test_design = np.column_stack([X_test_sc, np.ones(len(X_test_sc))])
    logit_var = np.einsum("ij,jk,ik->i", X_test_design, Sigma, X_test_design)
    return -logit_var


def evaluate_gate(y_test, cal_test, score, coverage=COVERAGE) -> dict:
    n_pred = int(np.round(len(y_test) * coverage))
    idx = np.argsort(-score)[:n_pred]
    y_sel, p_sel = y_test[idx], cal_test[idx]
    brier = float(brier_score_loss(y_sel, p_sel))
    try:
        auroc = float(roc_auc_score(y_sel, p_sel))
    except ValueError:
        auroc = float("nan")
    return {"brier": brier, "auroc": auroc, "n_predicted": n_pred}


def run_task(task: str) -> list[dict]:
    logger.info("=" * 60)
    logger.info("Alternative UQ baselines: %s", task)
    logger.info("=" * 60)
    artifacts = train_pipeline(task)

    gates = {
        "confidence": artifacts.cal_test,
        "reliability": artifacts.r_test,
        "conformal_variance": -artifacts.ensemble_sd_test,
    }

    logger.info("[%s] MC-dropout-variance gate (%d passes)...", task, N_DROPOUT_PASSES)
    gates["mc_dropout_variance"] = mc_dropout_variance_gate(artifacts)

    logger.info("[%s] Temperature-scaling + entropy gate...", task)
    ent_score, temp_T = temperature_entropy_gate(artifacts)
    gates["temperature_entropy"] = ent_score

    logger.info("[%s] Bayesian logistic regression gate (Laplace approx)...", task)
    gates["bayesian_logreg"] = bayesian_logreg_gate(artifacts)

    rows = []
    for gate_name, score in gates.items():
        res = evaluate_gate(artifacts.y_test, artifacts.cal_test, score)
        res.update({"task": task, "policy": gate_name, "coverage": COVERAGE})
        rows.append(res)
        logger.info("  %-22s brier=%.4f auroc=%.4f", gate_name, res["brier"], res["auroc"])

    return rows


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for task in ["sepsis", "mortality", "vasopressor"]:
        all_rows.extend(run_task(task))
    df = pd.DataFrame(all_rows)
    out_path = RESULTS_DIR / "alternative_uq_baselines.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved %d rows to %s", len(df), out_path)
    print(df.pivot(index="policy", columns="task", values="brier").to_string())


if __name__ == "__main__":
    main()
