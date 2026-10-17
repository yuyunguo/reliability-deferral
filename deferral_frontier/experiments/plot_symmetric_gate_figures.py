"""Figures for the one-sided vs. symmetric confidence vs. reliability comparison.

Reads the saved MIMIC-IV per-patient test artifacts and the net-benefit table
written by run_symmetric_confidence_gate.py, and writes three PDFs:

    fig_deferral_geometry.pdf   predicted risk of patients deferred by each gate
    fig_risk_coverage_3gates.pdf retained-set Brier score and prevalence vs coverage
    fig_decision_curve_3gates.pdf net benefit vs threshold at 80% coverage

Usage:
    python deferral_frontier/experiments/plot_symmetric_gate_figures.py OUT_DIR
"""
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ARTIFACT_DIR = Path("artifacts")
RESULTS_DIR = Path("deferral_frontier/results")
TASKS = ("sepsis", "mortality", "vasopressor")
TITLES = {"sepsis": "Sepsis", "mortality": "Mortality", "vasopressor": "Vasopressor"}
GATES = {
    "one_sided_confidence": ("One-sided confidence", "#4C72B0", "o"),
    "symmetric_confidence": ("Symmetric confidence", "#55A868", "s"),
    "reliability": ("Reliability composite", "#DD8452", "^"),
}
COVERAGES = np.round(np.arange(0.50, 1.001, 0.05), 2)


def gate_scores(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per-patient gate scores (higher = retain)."""
    p = df["cal_proba"].to_numpy()
    return {"one_sided_confidence": p,
            "symmetric_confidence": np.maximum(p, 1 - p),
            "reliability": df["r_score"].to_numpy()}


def retained(score: np.ndarray, coverage: float) -> np.ndarray:
    """Boolean mask of the top `coverage` fraction by score (stable ties)."""
    keep = np.zeros(len(score), dtype=bool)
    keep[np.argsort(-score, kind="stable")[: int(np.round(len(score) * coverage))]] = True
    return keep


def plot_geometry(frames: dict[str, pd.DataFrame], out: Path) -> None:
    """Box plots of calibrated risk among deferred patients (20% deferral)."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, task in zip(axes, TASKS):
        df = frames[task]
        p = df["cal_proba"].to_numpy()
        data, labels, colors = [p], ["All patients"], ["#999999"]
        for g, s in gate_scores(df).items():
            data.append(p[~retained(s, 0.80)])
            labels.append(GATES[g][0].replace(" ", "\n", 1))
            colors.append(GATES[g][1])
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        for med in bp["medians"]:
            med.set_color("black")
        ax.set_xticks(range(1, len(labels) + 1), labels, fontsize=7)
        ax.set_title(TITLES[task], fontsize=10)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Calibrated predicted risk $\\tilde{p}$")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_risk_coverage(frames: dict[str, pd.DataFrame], out: Path) -> None:
    """Retained-set Brier score (top) and outcome prevalence (bottom) vs coverage."""
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), sharex=True)
    for j, task in enumerate(TASKS):
        df = frames[task]
        y = df["y_true"].to_numpy()
        p = df["cal_proba"].to_numpy()
        for g, s in gate_scores(df).items():
            masks = [retained(s, c) for c in COVERAGES]
            label, color, marker = GATES[g]
            axes[0, j].plot(COVERAGES, [np.mean((p[m] - y[m]) ** 2) for m in masks],
                            color=color, marker=marker, ms=3, lw=1.5, label=label)
            axes[1, j].plot(COVERAGES, [y[m].mean() for m in masks],
                            color=color, marker=marker, ms=3, lw=1.5)
        axes[0, j].set_title(TITLES[task], fontsize=10)
        axes[1, j].set_xlabel("Coverage")
        for i in range(2):
            axes[i, j].grid(alpha=0.3)
    axes[0, 0].set_ylabel("Brier score (retained)")
    axes[1, 0].set_ylabel("Outcome prevalence\n(retained)")
    axes[0, 2].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_decision_curve(nb: pd.DataFrame, out: Path) -> None:
    """Net benefit vs threshold on MIMIC-IV at 80% coverage, plus no deferral."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    sub = nb[nb["source"] == "MIMIC-IV"]
    for ax, task in zip(axes, TASKS):
        t = sub[sub["task"] == task]
        for g, (label, color, marker) in GATES.items():
            r = t[t["gate"] == g]
            ax.plot(r["threshold"], r["net_benefit"], color=color, marker=marker,
                    ms=2.5, lw=1.5, label=label)
        r = t[t["gate"] == "no_deferral"]
        ax.plot(r["threshold"], r["net_benefit"], color="gray", ls="--", lw=1.2,
                label="No deferral (100% coverage)")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(TITLES[task], fontsize=10)
        ax.set_xlabel("Decision threshold $p_t$")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Net benefit")
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main(out_dir: Path) -> None:
    """Load artifacts and write all three figures to `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = {t: pd.read_parquet(ARTIFACT_DIR / f"{t}_test_artifacts.parquet") for t in TASKS}
    nb = pd.read_csv(RESULTS_DIR / "symmetric_gate_net_benefit.csv")
    plot_geometry(frames, out_dir / "fig_deferral_geometry.pdf")
    plot_risk_coverage(frames, out_dir / "fig_risk_coverage_3gates.pdf")
    plot_decision_curve(nb, out_dir / "fig_decision_curve_3gates.pdf")
    logger.info("Figures written to %s", out_dir)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("deferral_frontier/results/figures"))
