"""Round-1 stop gate: headline gate comparison before vs after re-extraction/re-split.

Reads paired_gate_tests.csv and symmetric_confidence_gate.csv from the
pre-round-1 snapshot and from the current results directory.

Usage:
    python -m deferral_frontier.experiments.r1_before_after
"""
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "deferral_frontier" / "results_pre_r1_2026-10-16"
NEW = ROOT / "deferral_frontier" / "results"
ALPHA = 0.05


def verdict(diff: float, p_holm: float) -> str:
    """Classify symmetric-minus-composite Brier difference (negative = symmetric better)."""
    if p_holm >= ALPHA:
        return "n.s."
    return "favors symmetric" if diff < 0 else "favors composite"


def _summary(res_dir: Path) -> pd.DataFrame:
    pt = pd.read_csv(res_dir / "paired_gate_tests.csv")
    one = pt[pt["gate_a"] == "one_sided_confidence"].set_index(["source", "task"])
    sym = pt[pt["gate_a"] == "symmetric_confidence"].set_index(["source", "task"])
    sg = pd.read_csv(res_dir / "symmetric_confidence_gate.csv")
    jac = sg[sg["gate"] == "symmetric_confidence"].set_index(["source", "task"])
    out = pd.DataFrame({
        "brier_one_sided": one["brier_a"],
        "brier_symmetric": sym["brier_a"],
        "brier_composite": sym["brier_b"],
        "ratio_one_sided_over_composite": one["brier_a"] / one["brier_b"],
        "sym_minus_comp": sym["brier_diff"],
        "ci_lo": sym["brier_ci_lo"], "ci_hi": sym["brier_ci_hi"],
        "p_holm": sym["brier_p_holm"],
        "jaccard_sym_vs_comp": jac["jaccard_deferred_vs_reliability"],
    })
    out["verdict"] = [verdict(d, p) for d, p in zip(out["sym_minus_comp"], out["p_holm"])]
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    both = _summary(OLD).join(_summary(NEW), lsuffix="_old", rsuffix="_new", how="outer")
    both["verdict_changed"] = both["verdict_old"] != both["verdict_new"]
    both.to_csv(NEW / "r1_before_after.csv", float_format="%.5f")
    cols = ["ratio_one_sided_over_composite_old", "ratio_one_sided_over_composite_new",
            "verdict_old", "verdict_new", "jaccard_sym_vs_comp_old", "jaccard_sym_vs_comp_new"]
    logger.info("\n%s", both[cols].to_string(float_format="%.2f"))
    logger.info("Verdict changes: %d of %d", both["verdict_changed"].sum(), len(both))


if __name__ == "__main__":
    main()
