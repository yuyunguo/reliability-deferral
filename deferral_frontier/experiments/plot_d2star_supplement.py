"""Figure S2 for the R1-4 response: D2* (imputation-uncertainty sufficiency) vs the submitted D2.

Panel (a): the submitted D2 (observation ratio) is at its ceiling in ~99% of patients in every
setting, while D2* has coefficient of variation 1.8-4.5 (discriminative).
Panel (b): D2*'s risk-matched gain over the symmetric confidence gate (Delta_adj, x1e-3) with
95% paired-bootstrap CIs across all 15 settings. D2* never significantly exceeds the symmetric
gate (all CIs <= 0 or spanning 0).

Sources: deferral_frontier/results/r1_d2star_full.csv and r1_d2star_full_discrimination.csv.
"""
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "deferral_frontier" / "results"
FIG_DIR = ROOT / "deferral_frontier" / "manuscript" / "bioengineering" / "figures"

SETTING_ORDER = ["MIMIC-IV", "MIMIC-III", "eICU", "eICU-H", "X-DB"]
DISPLAY = {"X-DB": "MIMIC-IV\u2192eICU"}  # label used in the manuscript
SETTING_COLORS = {"MIMIC-IV": "#0C7BDC", "MIMIC-III": "#E66100", "eICU": "#648FFF",
                  "eICU-H": "#D41159", "X-DB": "#117733"}
TASKS = ["sepsis", "mortality", "vasopressor"]
TASK_LABELS = {"sepsis": "Sepsis", "mortality": "Mortality", "vasopressor": "Vasopressor"}


def main() -> None:
    disc = pd.read_csv(RESULTS / "r1_d2star_full_discrimination.csv")
    gate = pd.read_csv(RESULTS / "r1_d2star_full.csv")
    d2 = gate[gate.score == "D2star"].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.0))

    # Panel (a): saturation contrast.
    for s in SETTING_ORDER:
        m = disc[disc.setting == s]
        ax1.scatter(m.d2_obsrate_saturated_frac * 100, m.d2star_cv,
                    color=SETTING_COLORS[s], s=70, label=DISPLAY.get(s, s), zorder=3)
    ax1.axvline(99.0, color="0.3", ls=":", lw=1)
    ax1.text(99.05, 0.3, "ceiling", rotation=90, va="bottom", fontsize=8, color="0.3")
    ax1.set_xlabel("Submitted D2: fraction of patients at ceiling (%)")
    ax1.set_ylabel("D2*: coefficient of variation")
    ax1.set_title("(a) D2 saturates; D2* discriminates")
    ax1.set_xlim(97.5, 100.2)
    ax1.set_ylim(0, 5.5)
    ax1.legend(fontsize=8, frameon=False, loc="upper left")

    # Panel (b): forest plot of Delta_adj (x1e-3) vs symmetric.
    yticks, ylabels = [], []
    y = 0
    for task in TASKS:
        for s in SETTING_ORDER:
            r = d2[(d2.setting == s) & (d2.task == task)].iloc[0]
            color = SETTING_COLORS[s]
            ax2.errorbar(r.delta_adj * 1e3, y, xerr=[[(r.delta_adj - r.ci_lo_adj) * 1e3],
                                                     [(r.ci_hi_adj - r.delta_adj) * 1e3]],
                         fmt="o", color=color, ms=6, capsize=2, lw=1.4, zorder=3)
            yticks.append(y)
            ylabels.append(f"{DISPLAY.get(s, s)} · {TASK_LABELS[task]}")
            y -= 1
    ax2.axvline(0, color="0.3", lw=1)
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(ylabels, fontsize=8)
    ax2.set_xlabel(r"D2* $\Delta_{\mathrm{adj}}$ vs symmetric gate ($\times 10^{-3}$)")
    ax2.set_title("(b) D2* gain over the symmetric gate")
    ax2.set_xlim(-10, 4)
    for s in SETTING_ORDER:
        ax2.plot([], [], "o", color=SETTING_COLORS[s], label=DISPLAY.get(s, s))
    ax2.legend(fontsize=8, frameon=False, loc="lower right")

    fig.tight_layout()
    out = FIG_DIR / "fig_s2_d2star.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
