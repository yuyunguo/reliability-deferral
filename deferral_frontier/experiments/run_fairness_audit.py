"""Fairness audit: deferral disparity across demographic groups.

Computes deferral rates and error rates stratified by race, gender, and age
at the 80% coverage threshold using the full R composite (not the D2 proxy
used in the manuscript).

REQUIRES:
  1. artifacts/{task}_test_artifacts.parquet (from run_deferral.py)
  2. data/mimic4/{task}_demographics.parquet (see extract_demographics below)

To extract demographics from MIMIC-IV PostgreSQL:

    SELECT a.hadm_id, p.gender,
           p.anchor_age,
           CASE
             WHEN a.race LIKE '%White%' THEN 'White'
             WHEN a.race LIKE '%Black%' OR a.race LIKE '%African American%' THEN 'Black'
             WHEN a.race LIKE '%Hispanic%' OR a.race LIKE '%Latino%' THEN 'Hispanic'
             WHEN a.race LIKE '%Asian%' THEN 'Asian'
             ELSE 'Other'
           END AS race_group
    FROM mimiciv.admissions a
    JOIN mimiciv.patients p ON a.subject_id = p.subject_id
    WHERE a.hadm_id = ANY(ARRAY[{hadm_ids}]);

Save the result as data/mimic4/{task}_demographics.parquet with hadm_id as index.

Usage:
    python -m deferral_frontier.experiments.run_fairness_audit sepsis
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

ARTIFACTS_DIR = Path("artifacts")
DATA_DIR = Path("data/mimic4")
RESULTS_DIR = Path("deferral_frontier/results")
COVERAGE = 0.80

logger = logging.getLogger(__name__)


def run_fairness_audit(task: str) -> None:
    logger.info("=" * 60)
    logger.info("Fairness audit: %s", task)
    logger.info("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    test_path = ARTIFACTS_DIR / f"{task}_test_artifacts.parquet"
    demo_path = DATA_DIR / f"{task}_demographics.parquet"

    if not test_path.exists():
        logger.error("Run run_deferral.py first to generate artifacts.")
        return
    if not demo_path.exists():
        logger.error(
            "Demographics file not found: %s\n"
            "Extract from MIMIC-IV using the SQL query in this file's docstring.",
            demo_path,
        )
        return

    test = pd.read_parquet(test_path)
    demo = pd.read_parquet(demo_path)

    merged = test.merge(demo, on="hadm_id", how="left")
    n_total = len(merged)
    n_missing = merged["race_group"].isna().sum()
    logger.info("Merged %d test patients; %d missing demographics", n_total, n_missing)

    n80 = int(np.round(n_total * COVERAGE))
    r_threshold = float(np.sort(-merged["r_score"].values)[n80 - 1])
    merged["deferred"] = merged["r_score"].values < (-r_threshold)

    rows = []
    for col in ["race_group", "gender"]:
        if col not in merged.columns:
            logger.warning("Column '%s' not found; skipping", col)
            continue
        for group, grp_df in merged.groupby(col):
            if pd.isna(group):
                continue
            n = len(grp_df)
            n_deferred = grp_df["deferred"].sum()
            deferral_rate = n_deferred / n

            pred_mask = ~grp_df["deferred"]
            if pred_mask.sum() > 1:
                brier = float(brier_score_loss(
                    grp_df.loc[pred_mask, "y_true"],
                    grp_df.loc[pred_mask, "cal_proba"],
                ))
            else:
                brier = np.nan

            error_mask = grp_df["deferred"]
            if error_mask.sum() > 0:
                preds = (grp_df.loc[error_mask, "cal_proba"] >= 0.5).astype(int)
                error_rate = float((preds != grp_df.loc[error_mask, "y_true"]).mean())
            else:
                error_rate = np.nan

            rows.append({
                "task": task,
                "stratification": col,
                "group": group,
                "n_total": n,
                "n_deferred": int(n_deferred),
                "deferral_rate": deferral_rate,
                "brier_predicted": brier,
                "error_rate_deferred": error_rate,
            })

    if "anchor_age" in merged.columns:
        age_bins = [0, 40, 55, 65, 75, 100]
        age_labels = ["<40", "40-54", "55-64", "65-74", "75+"]
        merged["age_group"] = pd.cut(merged["anchor_age"], bins=age_bins, labels=age_labels)
        for group, grp_df in merged.groupby("age_group", observed=True):
            if pd.isna(group):
                continue
            n = len(grp_df)
            n_deferred = grp_df["deferred"].sum()
            deferral_rate = n_deferred / n
            rows.append({
                "task": task,
                "stratification": "age_group",
                "group": str(group),
                "n_total": n,
                "n_deferred": int(n_deferred),
                "deferral_rate": deferral_rate,
                "brier_predicted": np.nan,
                "error_rate_deferred": np.nan,
            })

    results_df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / f"{task}_fairness_audit.csv"
    results_df.to_csv(out_path, index=False)
    logger.info("Saved fairness audit to %s", out_path)

    print(f"\n{'='*80}")
    print(f"  FAIRNESS AUDIT — {task.upper()} (R composite @ {COVERAGE:.0%} coverage)")
    print(f"{'='*80}")
    for strat in results_df["stratification"].unique():
        sub = results_df[results_df["stratification"] == strat]
        print(f"\n  [{strat}]")
        print(f"  {'Group':20s} | {'N':>6s} | {'DefRate':>8s} | {'Brier':>7s} | {'ErrRate':>8s}")
        print(f"  {'-'*20}-+-{'-'*6}-+-{'-'*8}-+-{'-'*7}-+-{'-'*8}")
        for _, r in sub.iterrows():
            print(f"  {r['group']:20s} | {r['n_total']:6d} | {r['deferral_rate']:8.3f} | "
                  f"{r['brier_predicted']:7.4f} | {r['error_rate_deferred']:8.3f}")

    overall_rate = merged["deferred"].mean()
    print(f"\n  Overall deferral rate: {overall_rate:.3f}")
    print(f"{'='*80}\n")

    max_disparity = results_df.groupby("stratification")["deferral_rate"].agg(
        lambda x: x.max() - x.min()
    )
    print("  Max deferral disparity by group:")
    for strat, disp in max_disparity.items():
        print(f"    {strat}: {disp:.3f}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, nargs="?", default="sepsis")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_fairness_audit(args.task)
