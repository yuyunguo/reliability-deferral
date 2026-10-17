"""Is the symmetric-vs-composite difference stable across patient-level splits?

Retrains the full pipeline (train_pipeline) for every dataset-task pair under
five split seeds and records the retained-Brier difference (symmetric minus
composite) at 80% coverage with a paired-bootstrap 95% CI.

Output: deferral_frontier/results/r1_split_seed_sensitivity.csv

Usage:
    python -m deferral_frontier.experiments.r1_split_seed_sensitivity [--sources eICU eICU-H]
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from deferral_frontier.experiments.run_symmetric_confidence_gate import (
    COVERAGE, RESULTS_DIR, paired_brier_diff_ci, retained_mask,
)
from deferral_frontier.src.train_pipeline import train_pipeline

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
SPLIT_SEEDS = (42, 1, 2, 3, 4)
DBS = {"MIMIC-IV": ("mimic4", "patient"), "MIMIC-III": ("mimic3", "patient"),
       "eICU": ("eicu", "patient"), "eICU-H": ("eicu", "hospital")}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=list(DBS), choices=list(DBS))
    sources = ap.parse_args().sources
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.INFO)
    out_path = RESULTS_DIR / "r1_split_seed_sensitivity.csv"
    keep = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame(columns=["source"])
    keep = keep[~keep["source"].isin(sources)]
    rows = []
    for source in sources:
        db, split_by = DBS[source]
        for task in ("sepsis", "mortality", "vasopressor"):
            for seed in SPLIT_SEEDS:
                a = train_pipeline(task, data_dir=ROOT / "data" / db, split_seed=seed, split_by=split_by)
                y, p = a.y_test.astype(int), a.cal_test
                k_sym = retained_mask(np.maximum(p, 1 - p), COVERAGE)
                k_rel = retained_mask(a.r_test, COVERAGE)
                sq = (p - y) ** 2
                lo, hi = paired_brier_diff_ci(y, p, k_sym, k_rel, np.random.default_rng(42))
                rows.append({"source": source, "task": task, "split_seed": seed,
                             "brier_symmetric": sq[k_sym].mean(), "brier_composite": sq[k_rel].mean(),
                             "sym_minus_comp": sq[k_sym].mean() - sq[k_rel].mean(),
                             "ci_lo": lo, "ci_hi": hi,
                             "jaccard_deferred": np.sum(~k_sym & ~k_rel) / np.sum(~k_sym | ~k_rel)})
                logger.info("%s %s seed %d: sym-comp %.4f [%.4f, %.4f]", source, task, seed,
                            rows[-1]["sym_minus_comp"], lo, hi)
                pd.concat([keep, pd.DataFrame(rows)]).to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
