"""Figure: risk-matched gain over the symmetric gate (Delta_adj), all settings and tasks.

Left three panels: Delta_adj (x1e-3) with 95% paired-bootstrap CIs for the five
pre-specified scores (D3', D5', C', L, L+D) in each setting, K = 10 risk bins.
Right panel: the k = 2 positive control (an error signal of known, weak strength)
on its own scale, to show the test's sensitivity.

Source: deferral_frontier/results/r1_v3_stratified.csv.

Usage:
    python deferral_frontier/experiments/plot_risk_matched_figure.py OUT_DIR
"""
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
RESULTS = Path("deferral_frontier/results/r1_v3_stratified.csv")
TASKS = ("sepsis", "mortality", "vasopressor")
SETTINGS = ("MIMIC-IV", "MIMIC-III", "eICU", "eICU-H", "X-DB")
SETTING_LABELS = {"MIMIC-IV": "MIMIC-IV", "MIMIC-III": "MIMIC-III", "eICU": "eICU",
                  "eICU-H": "eICU, held-out\nhospitals", "X-DB": "MIMIC-IV → eICU"}
SCORES = {"D3p": ("D3ᵐ similarity", "#8172B3", "o"), "D5p": ("D5ᵐ perturbation", "#937860", "s"),
          "Cp": ("Cᵐ ridge composite", "#DD8452", "D"), "L": ("L learned (record)", "#4C72B0", "^"),
          "LD": ("L+D learned + dims", "#55A868", "v")}


def _forest(ax, d: pd.DataFrame, task: str) -> None:
    offs = np.linspace(-0.3, 0.3, len(SCORES))
    for i, s in enumerate(SETTINGS):
        y0 = len(SETTINGS) - 1 - i
        for o, (sc, (lab, col, mk)) in zip(offs, SCORES.items()):
            r = d[(d.setting == s) & (d.task == task) & (d.score == sc)].iloc[0]
            ax.errorbar(r.delta * 1e3, y0 + o, xerr=[[(r.delta - r.ci_lo) * 1e3], [(r.ci_hi - r.delta) * 1e3]],
                        fmt=mk, color=col, ms=4, lw=1, capsize=0, label=lab if i == 0 else None)
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.set_yticks(range(len(SETTINGS)))
    ax.set_yticklabels([SETTING_LABELS[s] for s in reversed(SETTINGS)], fontsize=8)
    ax.set_title(task.capitalize(), fontsize=10)
    ax.set_xlabel("Gain over symmetric gate (×10⁻³)", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)


def main(out_dir: Path) -> None:
    d = pd.read_csv(RESULTS)
    d = d[d.K == 10]
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.2), gridspec_kw={"width_ratios": [3, 3, 3, 1.6]})
    for ax, t in zip(axes[:3], TASKS):
        _forest(ax, d, t)
    prim = d[d.score.isin(list(SCORES))]
    lim = (prim.ci_lo.min() * 1e3 - 0.5, prim.ci_hi.max() * 1e3 + 0.5)
    for ax in axes[:3]:
        ax.set_xlim(*lim)
    for ax in axes[1:3]:
        ax.set_yticklabels([])
    pc = axes[3]
    marks = {"sepsis": "o", "mortality": "s", "vasopressor": "^"}
    for i, s in enumerate(SETTINGS):
        y0 = len(SETTINGS) - 1 - i
        for o, t in zip((-0.2, 0, 0.2), TASKS):
            r = d[(d.setting == s) & (d.task == t) & (d.score == "positive_control_k2")].iloc[0]
            pc.errorbar(r.delta * 1e3, y0 + o, xerr=[[(r.delta - r.ci_lo) * 1e3], [(r.ci_hi - r.delta) * 1e3]],
                        fmt=marks[t], color="black", ms=3.5, lw=1, label=t.capitalize() if i == 0 else None)
    pc.axvline(0, color="grey", lw=0.8, ls="--")
    pc.set_yticks(range(len(SETTINGS)))
    pc.set_yticklabels([])
    pc.set_xlim(left=-2)
    pc.set_title("Positive control", fontsize=10)
    pc.set_xlabel("(×10⁻³)", fontsize=8)
    pc.tick_params(axis="x", labelsize=8)
    pc.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False,
              handletextpad=0.2, columnspacing=0.6)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.45, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = out_dir / "fig_risk_matched.pdf"
    fig.savefig(out, bbox_inches="tight")
    logger.info("wrote %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(Path(sys.argv[1]))
