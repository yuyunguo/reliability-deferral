"""Sensitivity of D3 (cohort similarity) to the sigma normalization choice.

Motivation: "the choice of
sigma (median training distance) is arbitrary. A sensitivity analysis with
sigma at the 25th and 75th percentiles would strengthen confidence."

D3 = exp(-Mahalanobis_distance / sigma); sigma only rescales D3, it does not
change the underlying distances, so this recomputes D3 with sigma set to the
25th/50th/75th percentile of TRAINING-set Mahalanobis distances (fit's mean
and precision matrix are unchanged), refits the composite on each variant,
and compares Brier at 80% coverage and D3's learned weight.

Uses sepsis on MIMIC-IV as the representative task (consistent with other
single-task ablations in this paper, e.g. run_d5_ablation_latency.py).
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from src.reliability.composite import DimensionMatrix, ReliabilityComposite

from deferral_frontier.src.train_pipeline import train_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("deferral_frontier/results")
COVERAGE = 0.80
PERCENTILES = [25, 50, 75]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = train_pipeline("sepsis")

    X_train_sc = artifacts.scaler.transform(artifacts.X_train)
    X_val_sc = artifacts.scaler.transform(artifacts.X_val)
    X_test_sc = artifacts.scaler.transform(artifacts.X_test)

    train_dist = artifacts.cohort_sim._mahalanobis(X_train_sc)
    val_dist = artifacts.cohort_sim._mahalanobis(X_val_sc)
    test_dist = artifacts.cohort_sim._mahalanobis(X_test_sc)

    rows = []
    for pct in PERCENTILES:
        sigma = float(np.percentile(train_dist, pct))
        d3_val = np.exp(-val_dist / sigma)
        d3_test = np.exp(-test_dist / sigma)

        dims_val = DimensionMatrix(
            artifacts.dims_val.D1, artifacts.dims_val.D2, d3_val,
            artifacts.dims_val.D4, artifacts.dims_val.D5)
        dims_test = DimensionMatrix(
            artifacts.dims_test.D1, artifacts.dims_test.D2, d3_test,
            artifacts.dims_test.D4, artifacts.dims_test.D5)

        val_cal_err = np.abs(artifacts.dims_val.D1 - artifacts.y_val)
        composite = ReliabilityComposite().fit(dims_val, val_cal_err)
        r_test = composite.transform(dims_test)
        d3_weight = float(composite._lr.coef_[0][2])  # D3 is index 2

        n_pred = int(np.round(len(artifacts.y_test) * COVERAGE))
        idx = np.argsort(-r_test)[:n_pred]
        brier = float(brier_score_loss(artifacts.y_test[idx], artifacts.cal_test[idx]))

        rows.append({
            "sigma_percentile": pct, "sigma": sigma,
            "d3_weight": d3_weight, "brier_at_80pct": brier,
            "d3_test_mean": float(d3_test.mean()),
        })
        logger.info("sigma=P%d (%.4f): D3 weight=%.4f, Brier@80%%=%.4f, D3 mean=%.4f",
                     pct, sigma, d3_weight, brier, d3_test.mean())

    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "d3_sigma_sensitivity.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved %d rows to %s", len(df), out_path)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
