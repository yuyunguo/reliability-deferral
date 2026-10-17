"""Decompose the entropy-vs-composite Brier gap into scope vs. ranking-quality.

Motivation: "what fraction of the
entropy-vs-composite Brier gap is explained by [entropy's two-sided]
scope difference versus genuinely different uncertainty ranking?"

Since both gates retain the same NUMBER of patients (matched 80% coverage),
the overlap between the two retained sets contributes identically to both
Brier scores and cancels exactly:

    Brier_composite - Brier_entropy = mean_sq_err(only_composite) -
                                       mean_sq_err(only_entropy)

So the entire Brier gap is attributable to the patients each gate uniquely
retains. We report that decomposition directly, plus what fraction of each
gate's uniquely-retained set falls below the confidence gate's own 80%
threshold (i.e., patients confidence itself would have deferred) -- the
operational definition of the "scope" (one-sided vs. two-sided selection)
effect described in the manuscript.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from deferral_frontier.experiments.run_alternative_baselines import temperature_entropy_gate
from deferral_frontier.src.train_pipeline import train_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("deferral_frontier/results")
COVERAGE = 0.80


def run_task(task: str) -> dict:
    artifacts = train_pipeline(task)
    y = artifacts.y_test
    cal = artifacts.cal_test
    r = artifacts.r_test
    entropy_score, _ = temperature_entropy_gate(artifacts)

    n = len(y)
    n_pred = int(np.round(n * COVERAGE))
    conf_idx = set(np.argsort(-cal)[:n_pred].tolist())
    rel_idx = set(np.argsort(-r)[:n_pred].tolist())
    ent_idx = set(np.argsort(-entropy_score)[:n_pred].tolist())

    sq_err = (y - cal) ** 2

    only_rel = np.array(sorted(rel_idx - ent_idx))
    only_ent = np.array(sorted(ent_idx - rel_idx))
    overlap = rel_idx & ent_idx

    brier_rel = sq_err[list(rel_idx)].mean()
    brier_ent = sq_err[list(ent_idx)].mean()
    mse_only_rel = sq_err[only_rel].mean() if len(only_rel) else float("nan")
    mse_only_ent = sq_err[only_ent].mean() if len(only_ent) else float("nan")

    # Reconstructed gap: should equal brier_rel - brier_ent exactly. Both retained
    # sets have size n_pred (matched coverage), so the shared overlap term cancels
    # and the denominator here must be n_pred (the retained-set size Brier is
    # averaged over), not n (the full test set).
    reconstructed_gap = (len(only_rel) * mse_only_rel - len(only_ent) * mse_only_ent) / n_pred
    actual_gap = brier_rel - brier_ent

    frac_only_ent_below_conf_threshold = float(np.mean(
        [i not in conf_idx for i in only_ent]
    )) if len(only_ent) else float("nan")
    frac_only_rel_below_conf_threshold = float(np.mean(
        [i not in conf_idx for i in only_rel]
    )) if len(only_rel) else float("nan")

    jaccard_rel_ent = len(overlap) / len(rel_idx | ent_idx)

    result = {
        "task": task,
        "jaccard_reliability_entropy": jaccard_rel_ent,
        "n_only_reliability": len(only_rel),
        "n_only_entropy": len(only_ent),
        "n_overlap": len(overlap),
        "brier_reliability_full": brier_rel,
        "brier_entropy_full": brier_ent,
        "actual_brier_gap": actual_gap,
        "reconstructed_brier_gap": reconstructed_gap,
        "mse_only_reliability_retains": mse_only_rel,
        "mse_only_entropy_retains": mse_only_ent,
        "frac_only_entropy_would_be_deferred_by_confidence": frac_only_ent_below_conf_threshold,
        "frac_only_reliability_would_be_deferred_by_confidence": frac_only_rel_below_conf_threshold,
    }
    logger.info(
        "[%s] Jaccard(rel,ent)=%.3f | actual gap=%.5f reconstructed=%.5f | "
        "only-entropy patients confidence-would-defer=%.1f%% | "
        "only-reliability patients confidence-would-defer=%.1f%%",
        task, jaccard_rel_ent, actual_gap, reconstructed_gap,
        100 * frac_only_ent_below_conf_threshold, 100 * frac_only_rel_below_conf_threshold,
    )
    return result


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [run_task(task) for task in ["sepsis", "mortality", "vasopressor"]]
    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "entropy_decomposition.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved %d rows to %s", len(df), out_path)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
