"""Decision-curve (net benefit) analysis: reliability vs. confidence gate.

Motivation:
  - circularity check requires an evaluation criterion orthogonal to
    calibration -- net benefit at a clinical decision threshold is exactly
    that (it depends on discrimination at the threshold, not on Brier/ECE).
  - decision-curve analysis is needed to
    assess whether the AUROC degradation under reliability gating matters
    for threshold-based binary clinical decisions.

Net benefit at threshold p_t, standard formulation (Vickers & Elkin 2006):
    NB(p_t) = TP/n - FP/n * (p_t / (1 - p_t))
computed on the RETAINED (non-deferred) subset under each gate at 80%
coverage, using the same calibrated probabilities and labels already saved
in the test artifacts. No retraining required.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("deferral_frontier/results")
COVERAGE = 0.80
THRESHOLDS = np.round(np.arange(0.02, 0.51, 0.02), 3)


def net_benefit_curve(y_true, y_prob, thresholds):
    n = len(y_true)
    nb = []
    for pt in thresholds:
        pred_pos = y_prob >= pt
        tp = np.sum(pred_pos & (y_true == 1))
        fp = np.sum(pred_pos & (y_true == 0))
        nb.append(tp / n - fp / n * (pt / (1 - pt)))
    return np.array(nb)


def run_task(source: str, artifact_path: str, task: str):
    df = pd.read_parquet(artifact_path)
    y = df["y_true"].values
    cal = df["cal_proba"].values
    r = df["r_score"].values

    n = len(y)
    n80 = int(np.round(n * COVERAGE))

    conf_idx = np.argsort(-cal)[:n80]
    rel_idx = np.argsort(-r)[:n80]

    nb_conf = net_benefit_curve(y[conf_idx], cal[conf_idx], THRESHOLDS)
    nb_rel = net_benefit_curve(y[rel_idx], cal[rel_idx], THRESHOLDS)
    nb_all = net_benefit_curve(y, cal, THRESHOLDS)  # treat-all reference (100% coverage)

    rows = []
    for pt, c, r_, a in zip(THRESHOLDS, nb_conf, nb_rel, nb_all):
        rows.append({"source": source, "task": task, "threshold": pt,
                      "nb_confidence": c, "nb_reliability": r_, "nb_treat_all": a})
    df_out = pd.DataFrame(rows)

    # Summary: at clinically plausible thresholds (0.1, 0.2, 0.3), which gate wins?
    for pt_check in [0.1, 0.2, 0.3]:
        idx = np.argmin(np.abs(THRESHOLDS - pt_check))
        logger.info("  %s/%s @ p_t=%.2f: NB(confidence)=%.4f  NB(reliability)=%.4f  delta=%+.4f",
                     source, task, THRESHOLDS[idx], nb_conf[idx], nb_rel[idx],
                     nb_rel[idx] - nb_conf[idx])
    return df_out


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sources = {
        "MIMIC-IV": "artifacts/{task}_test_artifacts.parquet",
        "MIMIC-III": "artifacts/mimic3/mimic3_{task}_test_artifacts.parquet",
        "eICU": "artifacts/eicu/eicu_{task}_test_artifacts.parquet",
    }
    all_rows = []
    for source, tpl in sources.items():
        for task in ["sepsis", "mortality", "vasopressor"]:
            path = tpl.format(task=task)
            if not Path(path).exists():
                continue
            df_out = run_task(source, path, task)
            all_rows.append(df_out)

    result = pd.concat(all_rows, ignore_index=True)
    result.to_csv(RESULTS_DIR / "decision_curve_net_benefit.csv", index=False)
    logger.info("Saved %d rows to decision_curve_net_benefit.csv", len(result))


if __name__ == "__main__":
    main()
