"""D2* feasibility test (plan/r1-d2-imputation-preregistration.md).

MIMIC-IV, three tasks, in-distribution, exploratory. Answers Reviewer 1, comment 4:
(1) does D2* discriminate where the submitted D2 saturates, and (2) does D2* carry
within-risk error signal beyond confidence?

The submitted D2 (24h-observed / first-6h-observed, `compute_d2_fast`) saturates at ~1.0;
D2* is its value-of-information replacement. Outputs (deferral_frontier/results/):
    r1_d2star.csv               gate comparison (Delta vs random, Delta_adj vs symmetric)
    r1_d2star_discrimination.csv  saturation + redundancy summary

Usage:
    python -m deferral_frontier.experiments.r1_d2star
"""
import logging
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from deferral_frontier.experiments.run_symmetric_confidence_gate import RESULTS_DIR
from deferral_frontier.src.fast_reliability import compute_d2_fast
from deferral_frontier.src.imputation_uncertainty import StochasticImputer, imputation_uncertainty
from deferral_frontier.src.stratified_deferral import bootstrap_deltas, summarize
from deferral_frontier.src.train_pipeline import train_pipeline

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
TASKS = ("sepsis", "mortality", "vasopressor")
N_BOOT = 1000
N_PASSES = 50
SEED = 42
K = 10


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.setLevel(logging.INFO)
    gate_rows, disc_rows = [], []

    for task in TASKS:
        art = train_pipeline(task, data_dir=ROOT / "data" / "mimic4")
        imputer = StochasticImputer().fit(art.X_train, art.X_train_raw)
        d2star_unc = imputation_uncertainty(art.primary_model, art.X_test, art.X_test_raw,
                                            imputer, n_passes=N_PASSES, seed=SEED)

        p = art.cal_test
        y = art.y_test.astype(int)
        e = (p - y) ** 2
        # The submitted D2 (observation ratio) — the measure that saturates.
        d2_obs = compute_d2_fast(art.dataset.timeseries, pd.Index(art.idx_test),
                                 art.dataset.admission_times, art.dataset.prediction_times,
                                 lookback_hours=24, baseline_hours=6)
        d2_countfrac = art.dims_test.D2  # the v2/v3 count-fraction replacement (reference)

        scores = {
            "D2star": -d2star_unc,          # reliability = -imputation uncertainty
            "D2_obsrate": d2_obs,           # submitted D2 (saturated; reference)
            "D2_countfrac": d2_countfrac,   # v2/v3 count-fraction (coarse; reference)
            "symmetric": np.maximum(p, 1 - p),
            "oracle": -e,
        }
        point, boots = bootstrap_deltas(scores, p, y, k=K, n_boot=N_BOOT, seed=SEED)
        for name in scores:
            rnd = summarize(point[name], boots[name])
            adj = summarize(point[name] - point["symmetric"], boots[name] - boots["symmetric"])
            gate_rows.append({"task": task, "score": name,
                              "delta_vs_random": rnd["delta"], "ci_lo_vs_random": rnd["ci_lo"],
                              "ci_hi_vs_random": rnd["ci_hi"], "p_vs_random": rnd["p"],
                              "delta_adj": adj["delta"], "ci_lo_adj": adj["ci_lo"],
                              "ci_hi_adj": adj["ci_hi"], "p_adj": adj["p"],
                              "share_of_oracle": point[name] / point["oracle"]})

        disc_rows.append({
            "task": task, "n": len(y), "prevalence": float(y.mean()),
            "d2_obsrate_saturated_frac": float((d2_obs >= 1.0 - 1e-9).mean()),
            "d2_obsrate_mean": float(d2_obs.mean()),
            "d2_countfrac_at_max": float((d2_countfrac >= 1.0 - 1e-9).mean()),
            "d2_countfrac_nunique": int(len(np.unique(d2_countfrac.round(9)))),
            "d2star_zero_frac": float((d2star_unc <= 1e-12).mean()),
            "d2star_cv": float(d2star_unc.std() / max(d2star_unc.mean(), 1e-12)),
            "d2star_q50": float(np.quantile(d2star_unc, 0.5)),
            "d2star_q90": float(np.quantile(d2star_unc, 0.9)),
            "d2star_q99": float(np.quantile(d2star_unc, 0.99)),
            "d2star_max": float(d2star_unc.max()),
            "spearman_d2star_p": float(spearmanr(d2star_unc, p).statistic),
            "spearman_d2star_p1mp": float(spearmanr(d2star_unc, p * (1 - p)).statistic),
        })
        logger.info("%s: D2_obs sat %.3f | D2* zero %.3f CV %.2f | rho(D2*,p)=%.3f rho(D2*,p(1-p))=%.3f",
                    task, (d2_obs >= 1.0 - 1e-9).mean(), (d2star_unc <= 1e-12).mean(),
                    d2star_unc.std() / max(d2star_unc.mean(), 1e-12),
                    spearmanr(d2star_unc, p).statistic, spearmanr(d2star_unc, p * (1 - p)).statistic)

    pd.DataFrame(gate_rows).to_csv(RESULTS_DIR / "r1_d2star.csv", index=False)
    pd.DataFrame(disc_rows).to_csv(RESULTS_DIR / "r1_d2star_discrimination.csv", index=False)
    logger.info("wrote r1_d2star.csv and r1_d2star_discrimination.csv")


if __name__ == "__main__":
    main()
