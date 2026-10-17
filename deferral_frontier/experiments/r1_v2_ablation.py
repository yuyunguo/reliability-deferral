"""Pre-registered descriptive analyses for composite v2.

For every dataset-task pair, refit ErrorComposite with dimensions blanked
(set to a constant) and report, on the test set:
  - retained Brier at 80% coverage (full v2, drop-one for D2'-D5', D1'-only);
  - held-out squared-error prediction: mean squared error of the predicted
    (p - y)^2 against the observed (p - y)^2, and the reduction relative to
    the D1'-only model.

Output: deferral_frontier/results/composite_v2_ablation.csv

Usage:
    python -m deferral_frontier.experiments.r1_v2_ablation
"""
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from deferral_frontier.experiments.run_symmetric_confidence_gate import COVERAGE, RESULTS_DIR, retained_mask
from deferral_frontier.src.reliability_v2 import ErrorComposite, patient_groups
from deferral_frontier.src.train_pipeline import train_pipeline
from src.models.xgboost_model import ensemble_predict

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
DBS = {"MIMIC-IV": "mimic4", "MIMIC-III": "mimic3", "eICU": "eicu"}
OTHER = ("D2_sufficiency", "D3_similarity", "D4_stability", "D5_mc_variance")


def blank(other: np.ndarray, cols: Sequence[int]) -> np.ndarray:
    """Copy of `other` with the given columns set to 0 (dimension removed)."""
    out = np.array(other, dtype=float, copy=True)
    out[:, list(cols)] = 0.0
    return out


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.INFO)
    rows = []
    for source, db in DBS.items():
        for task in ("sepsis", "mortality", "vasopressor"):
            a = train_pipeline(task, data_dir=ROOT / "data" / db)
            raw_val = ensemble_predict(a.ensemble, a.X_val_raw)
            groups = patient_groups(ROOT / "data" / db, a.idx_val)
            dv, dt = a.dims_val, a.dims_test
            other_val = np.column_stack([dv.D2, dv.D3, dv.D4, dv.D5])
            other_test = np.column_stack([dt.D2, dt.D3, dt.D4, dt.D5])
            y, p = a.y_test.astype(int), a.cal_test
            sq = (p - y) ** 2
            variants = {"full": [], "D1_only": [0, 1, 2, 3]}
            variants.update({f"drop_{n}": [k] for k, n in enumerate(OTHER)})
            mse = {}
            for name, cols in variants.items():
                comp = ErrorComposite().fit(raw_val, a.y_val.astype(int), blank(other_val, cols), groups)
                pred_err = -comp.score(p, blank(other_test, cols))
                keep = retained_mask(-pred_err, COVERAGE)
                mse[name] = float(np.mean((pred_err - sq) ** 2))
                rows.append({"source": source, "task": task, "variant": name,
                             "brier_retained": float(sq[keep].mean()), "error_pred_mse": mse[name]})
            for r in rows[-len(variants):]:
                r["error_pred_mse_reduction_vs_D1_only"] = 1 - r["error_pred_mse"] / mse["D1_only"]
            logger.info("%s %s: full Brier %.4f, D1-only %.4f, error-MSE reduction %.2f%%", source, task,
                        rows[-len(variants)]["brier_retained"], rows[-len(variants) + 1]["brier_retained"],
                        100 * rows[-len(variants)]["error_pred_mse_reduction_vs_D1_only"])
            pd.DataFrame(rows).to_csv(RESULTS_DIR / "composite_v2_ablation.csv", index=False)


if __name__ == "__main__":
    main()
