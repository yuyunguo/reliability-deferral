"""D1+D5 minimal composite ablation study.

Trains D1+D5 logistic composite on the validation set, evaluates on the
test set — identical methodology to the full R composite.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

ARTIFACTS_DIR = Path("artifacts")
RESULTS_DIR = Path("deferral_frontier/results")
TASKS = ["sepsis", "mortality", "vasopressor"]
COVERAGE = 0.80


def _quick_auroc(y, p):
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")


def run_d1d5_ablation():
    all_ci_rows = []
    for task in TASKS:
        print(f"\n{'='*50}")
        print(f"  D1+D5 Ablation (val-trained) — {task.upper()}")
        print(f"{'='*50}")

        val = pd.read_parquet(ARTIFACTS_DIR / f"{task}_val_artifacts.parquet")
        test = pd.read_parquet(ARTIFACTS_DIR / f"{task}_test_artifacts.parquet")

        y_val = val["y_true"].values
        p_val = val["cal_proba"].values
        d1_val = val["D1"].values
        d5_val = val["D5"].values

        y_test = test["y_true"].values
        p_test = test["cal_proba"].values
        d1_test = test["D1"].values
        d5_test = test["D5"].values
        r_full = test["r_score"].values
        sd = test["ensemble_sd"].values

        cal_err_val = np.abs(p_val - y_val)
        y_binary_val = (cal_err_val > np.median(cal_err_val)).astype(int)

        X_val = np.column_stack([d1_val, d5_val])
        X_test = np.column_stack([d1_test, d5_test])

        sc = StandardScaler().fit(X_val)
        lr = LogisticRegression(max_iter=1000, random_state=42).fit(
            sc.transform(X_val), y_binary_val)
        r_d1d5 = lr.predict_proba(sc.transform(X_test))[:, 0]

        print(f"  D1+D5 weights: D1={lr.coef_[0][0]:.2f}, D5={lr.coef_[0][1]:.2f}")
        print(f"  D1+D5 R range: [{r_d1d5.min():.3f}, {r_d1d5.max():.3f}]")

        for name, score in [
            ("confidence", p_test),
            ("reliability", r_full),
            ("D1D5_minimal", r_d1d5),
            ("conformal_variance", -sd),
        ]:
            n80 = int(np.round(len(y_test) * COVERAGE))
            idx80 = np.argsort(-score)[:n80]
            y80 = y_test[idx80]
            p80 = p_test[idx80]
            brier = brier_score_loss(y80, p80)
            auroc = _quick_auroc(y80, p80)
            print(f"  {name:20s} | Brier={brier:.4f} | AUROC={auroc:.4f}")

            rng = np.random.default_rng(42)
            briers = []
            aurocs = []
            for _ in range(1000):
                boot = rng.choice(len(y_test), size=len(y_test), replace=True)
                s_b = score[boot]
                y_b = y_test[boot]
                p_b = p_test[boot]
                n_b = int(np.round(len(y_b) * COVERAGE))
                ix = np.argsort(-s_b)[:n_b]
                briers.append(brier_score_loss(y_b[ix], p_b[ix]))
                aurocs.append(_quick_auroc(y_b[ix], p_b[ix]))

            all_ci_rows.append({
                "task": task,
                "policy": name,
                "brier": brier,
                "brier_ci_lo": float(np.percentile(briers, 2.5)),
                "brier_ci_hi": float(np.percentile(briers, 97.5)),
                "auroc": auroc,
                "auroc_ci_lo": float(np.percentile(aurocs, 2.5)),
                "auroc_ci_hi": float(np.percentile(aurocs, 97.5)),
            })

    ci_df = pd.DataFrame(all_ci_rows)
    out = RESULTS_DIR / "d1d5_ablation_ci.csv"
    ci_df.to_csv(out, index=False)
    print(f"\nSaved to {out}")
    print(ci_df.to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_d1d5_ablation()
