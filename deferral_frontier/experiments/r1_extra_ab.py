"""POST HOC extra analyses A and B (plan/r1-extra-analyses-plan.md, sha256 d48b8bb1...).

A. Per-hospital heterogeneity in eICU-H (saved artifacts + hospital ids).
B. Threshold-centered decision loss, all 12 pairs (saved artifacts).

Outputs: deferral_frontier/results/r1_extra_hospital{,_pooled}.csv, r1_extra_threshold.csv

Usage:
    python -m deferral_frontier.experiments.r1_extra_ab
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from deferral_frontier.experiments.run_symmetric_confidence_gate import RESULTS_DIR, retained_mask
from deferral_frontier.src.extra_analyses import decision_loss, random_effects, t_centered_score

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"
TASKS = ("sepsis", "mortality", "vasopressor")
ARTS = {"MIMIC-IV": "{t}_test_artifacts.parquet", "MIMIC-III": "mimic3/mimic3_{t}_test_artifacts.parquet",
        "eICU": "eicu/eicu_{t}_test_artifacts.parquet",
        "eICU-H": "eicu_hospital/eicu_hospital_{t}_test_artifacts.parquet"}
THRESHOLDS = (0.05, 0.10, 0.20, 0.30, 0.50)
SEED = 42


def gate_diff(p: np.ndarray, y: np.ndarray, r: np.ndarray) -> float:
    e = (p - y) ** 2
    return e[retained_mask(np.maximum(p, 1 - p), 0.8)].mean() - e[retained_mask(r, 0.8)].mean()


def hospital_analysis() -> None:
    hid = pd.read_parquet(ROOT / "data" / "eicu" / "hospital_ids.parquet")["hospital_id"]
    rows, pooled = [], []
    rng = np.random.default_rng(SEED)
    for t in TASKS:
        a = pd.read_parquet(ART / ARTS["eICU-H"].format(t=t))
        a["hospital"] = hid.reindex(a.stay_id.to_numpy()).to_numpy()  # eICU artifacts key rows by stay_id
        for h, g in a.groupby("hospital"):
            p, y, r = g.cal_proba.to_numpy(), g.y_true.to_numpy().astype(int), g.r_score.to_numpy()
            if len(g) < 200 or y.sum() < 10:
                continue
            d = gate_diff(p, y, r)
            bs = [gate_diff(p[i], y[i], r[i]) for i in (rng.integers(0, len(p), len(p)) for _ in range(500))]
            rows.append({"task": t, "hospital": h, "n": len(g), "events": int(y.sum()),
                         "citl": y.mean() - p.mean(), "sym_minus_comp": d, "se": float(np.std(bs, ddof=1))})
        x = pd.DataFrame([r_ for r_ in rows if r_["task"] == t])
        re = random_effects(x.sym_minus_comp.to_numpy(), x.se.to_numpy())
        pooled.append({"task": t, **re, "n_favor_symmetric": int((x.sym_minus_comp < 0).sum()),
                       "n_favor_composite": int((x.sym_minus_comp > 0).sum()),
                       "citl_min": x.citl.min(), "citl_max": x.citl.max(), "citl_sd": x.citl.std()})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "r1_extra_hospital.csv", index=False)
    pd.DataFrame(pooled).to_csv(RESULTS_DIR / "r1_extra_hospital_pooled.csv", index=False)


def threshold_analysis() -> None:
    rows = []
    for s, tmpl in ARTS.items():
        for task in TASKS:
            a = pd.read_parquet(ART / tmpl.format(t=task))
            p, y, r = a.cal_proba.to_numpy(), a.y_true.to_numpy().astype(int), a.r_score.to_numpy()
            for t in THRESHOLDS:
                loss = decision_loss(p, y, t)
                keep = {"one_sided": retained_mask(p, 0.8), "symmetric": retained_mask(np.maximum(p, 1 - p), 0.8),
                        "composite": retained_mask(r, 0.8), "t_centered": retained_mask(t_centered_score(p, t), 0.8)}
                rng = np.random.default_rng(SEED)
                d = loss[keep["composite"]].mean() - loss[keep["t_centered"]].mean()
                bs = np.empty(1000)
                for i in range(1000):
                    idx = rng.integers(0, len(p), len(p))
                    bs[i] = loss[idx][keep["composite"][idx]].mean() - loss[idx][keep["t_centered"][idx]].mean()
                lo, hi = np.percentile(bs, [2.5, 97.5])
                rows.append({"setting": s, "task": task, "t": t, "no_deferral": loss.mean(),
                             **{f"loss_{g}": loss[m].mean() for g, m in keep.items()},
                             "comp_minus_tcent": d, "ci_lo": lo, "ci_hi": hi})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "r1_extra_threshold.csv", index=False)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    hospital_analysis()
    threshold_analysis()
    logger.info("A and B written")


if __name__ == "__main__":
    main()
