"""Publication-quality figures for the Deferral Frontier paper."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.family": "Helvetica",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

COLORS = {
    "confidence": "#4C72B0",
    "reliability": "#DD8452",
    "R_equal_weight": "#937860",
    "treat_all": "#888888",
    "D1_confidence": "#4C72B0",
    "D2_sufficiency": "#55A868",
    "D3_similarity": "#C44E52",
    "D4_stability": "#8172B3",
    "D5_uncertainty": "#CCB974",
}

TASKS = ["sepsis", "mortality", "vasopressor"]
TASK_LABELS = {"sepsis": "Sepsis", "mortality": "Mortality", "vasopressor": "Vasopressor"}
POLICIES = ["confidence", "reliability"]
MAIN_POLICIES = ["confidence", "reliability"]  # headline comparison; equal-weight is an ablation
POLICY_LABELS = {
    "confidence": "Confidence Gate",
    "reliability": "Reliability Gate",
    "R_equal_weight": "R (Equal Wt)",
}
DATASETS = ["MIMIC-IV", "MIMIC-III", "eICU"]
DATASET_COLORS = {"MIMIC-IV": "#4C72B0", "MIMIC-III": "#55A868", "eICU": "#C44E52"}

FIGURE_DIR = Path(__file__).resolve().parents[2] / "deferral_frontier" / "manuscript" / "figures"


def _ensure_figure_dir():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def _load_all_tasks(results_dir: Path) -> pd.DataFrame:
    frames = []
    for t in TASKS:
        p = results_dir / f"{t}_deferral_results.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not df.empty:
        # Each (task, policy) has a spurious duplicate row at coverage=0.80
        # with NaN brier/auroc (upstream artifact of merging the headline
        # 80%-coverage CI computation into the full sweep). Left in, this
        # breaks the plotted line into two disconnected segments at that
        # coverage. Drop rows with no metric to plot.
        df = df.dropna(subset=["brier", "auroc"])
    return df


def fig1_risk_coverage(results_csv: str, task: str = "sepsis") -> str:
    _ensure_figure_dir()
    results_dir = Path(results_csv).parent
    all_df = _load_all_tasks(results_dir)
    df = all_df[all_df.policy.isin(MAIN_POLICIES)].copy()

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=False)
    for ax, t in zip(axes, TASKS):
        sub = df[df.task == t]
        for policy in MAIN_POLICIES:
            psub = sub[sub.policy == policy].sort_values("coverage")
            if psub.empty:
                continue
            ax.plot(psub["coverage"], psub["brier"],
                    label=POLICY_LABELS[policy],
                    color=COLORS.get(policy, "#999"),
                    linewidth=1.6, marker="o", markersize=2.5)
            # No CI overlay: bootstrap CIs are only computed at the headline
            # 80%-coverage operating point (see Table "Performance at 80%
            # coverage"), not across this sweep. A single-point whisker on an
            # otherwise CI-less curve reads as an anomaly rather than as
            # "uncertainty is only known here" -- exact CIs belong in the
            # table, not as a partial visual overlay on this figure.
        ax.set_xlabel("Coverage")
        ax.set_title(TASK_LABELS[t])
        ax.set_xlim(0.45, 1.05)

    axes[0].set_ylabel("Brier Score")
    axes[-1].legend(frameon=True, loc="upper left")
    fig.tight_layout()
    path = FIGURE_DIR / "fig1_risk_coverage.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def fig2_auroc_vs_coverage(results_csv: str, task: str = "sepsis") -> str:
    _ensure_figure_dir()
    results_dir = Path(results_csv).parent
    all_df = _load_all_tasks(results_dir)
    df = all_df[all_df.policy.isin(MAIN_POLICIES)].copy()

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=False)
    for ax, t in zip(axes, TASKS):
        sub = df[df.task == t]
        for policy in MAIN_POLICIES:
            psub = sub[sub.policy == policy].sort_values("coverage")
            if psub.empty:
                continue
            ax.plot(psub["coverage"], psub["auroc"],
                    label=POLICY_LABELS[policy],
                    color=COLORS.get(policy, "#999"),
                    linewidth=1.6, marker="s", markersize=2.5)
            # No CI overlay -- see note in fig1_risk_coverage above.
        ax.set_xlabel("Coverage")
        ax.set_title(TASK_LABELS[t])
        ax.set_xlim(0.45, 1.05)

    axes[0].set_ylabel("AUROC")
    axes[-1].legend(frameon=True, loc="lower right")
    fig.tight_layout()
    path = FIGURE_DIR / "fig2_auroc_vs_coverage.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def fig3_deferral_frontier(results_csv: str, task: str = "sepsis") -> str:
    _ensure_figure_dir()
    results_dir = Path(results_csv).parent
    all_df = _load_all_tasks(results_dir)
    df = all_df[all_df.policy.isin(POLICIES)].copy()
    df["deferral_rate"] = 1.0 - df["coverage"]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=False)
    for ax, t in zip(axes, TASKS):
        sub = df[df.task == t]
        for policy in POLICIES:
            psub = sub[sub.policy == policy].sort_values("deferral_rate")
            if psub.empty:
                continue
            ax.plot(psub["deferral_rate"], psub["brier"],
                    label=POLICY_LABELS[policy],
                    color=COLORS.get(policy, "#999"),
                    linewidth=1.6)
            # No CI overlay -- see note in fig1_risk_coverage above.
        ax.set_xlabel("Deferral Rate")
        ax.set_title(TASK_LABELS[t])

    axes[0].set_ylabel("Brier Score")
    axes[-1].legend(frameon=True, loc="upper right")
    fig.tight_layout()
    path = FIGURE_DIR / "fig3_deferral_frontier.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def fig5_case_studies(case_csv: str, task: str = "sepsis") -> str:
    _ensure_figure_dir()
    results_dir = Path(case_csv).parent
    all_frames = []
    for t in TASKS:
        p = results_dir / f"{t}_case_studies.csv"
        if p.exists():
            all_frames.append(pd.read_csv(p))
    if not all_frames:
        return ""
    df = pd.concat(all_frames, ignore_index=True)

    categories = [
        ("High-Confidence, Low-Reliability", "high_conf_low_r"),
        ("Low-Confidence, High-Reliability", "low_conf_high_r"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.5))
    for row_idx, (row_label, cat) in enumerate(categories):
        for col_idx, t in enumerate(TASKS):
            ax = axes[row_idx, col_idx]
            subset = df[(df["category"] == cat) & (df["task"] == t)]
            if len(subset) > 0:
                case = subset.iloc[0]
                _plot_single_case(ax, case, row_idx == 0 and col_idx == 0)
            else:
                ax.axis("off")
            if row_idx == 0:
                ax.set_title(TASK_LABELS[t], fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=7, fontweight="bold")

    fig.tight_layout()
    path = FIGURE_DIR / "fig5_case_studies.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _plot_single_case(ax, case: pd.Series, show_dim_labels: bool = True) -> None:
    dim_names = ["Confidence", "Data Suff.", "Cohort Sim.", "Temp. Stab.", "Uncert. Cons."]
    dim_cols = ["D1", "D2", "D3", "D4", "D5"]
    values = [case.get(f"dim_{d}", 0) for d in dim_cols]
    dim_keys = ["D1_confidence", "D2_sufficiency", "D3_similarity", "D4_stability", "D5_uncertainty"]
    colors_bar = [COLORS.get(k, "#999") for k in dim_keys]

    bars = ax.barh(dim_names, values, color=colors_bar, edgecolor="white", height=0.6)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Dimension Value", fontsize=7)
    ax.tick_params(labelsize=6)
    if not show_dim_labels:
        ax.set_yticklabels([])

    true_label = int(case.get("true_label", 0))
    conf_val = case.get("confidence", 0)
    rel_val = case.get("reliability", 0)
    ax.text(0.98, 0.95,
            f"y={true_label}  conf={conf_val:.2f}  R={rel_val:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=6,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    for bar, val in zip(bars, values):
        if val > 0.05:
            ax.text(min(bar.get_width() - 0.03, 0.95),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", ha="right", va="center", fontsize=6, color="white")


def fig4_auroc_vs_weight(results_csv: str, coef_csv: str, task: str = "sepsis") -> str:
    """Standalone AUROC (top row) vs. |composite weight| (bottom row) per dimension.

    Two stacked single-axis panel rows rather than a twin (dual-axis) plot:
    dual y-axes on the same panel invite misreading which bar belongs to
    which scale, so AUROC and weight are given their own proper axes and
    aligned by dimension (same x position, same column) instead.
    """
    _ensure_figure_dir()
    results_dir = Path(results_csv).parent
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.0), sharex=True)
    # D1 (confidence) is included alongside D2-D5: it is the dimension
    # central to the paper's argument (high learned weight for sepsis/
    # mortality despite AUROC that is often lower than D2-D4's), and its
    # omission removed the most argument-relevant point from this figure.
    dim_res_policies = ["confidence", "D2_sufficiency", "D3_similarity",
                         "D4_stability", "D5_uncertainty"]
    dim_coef_prefixes = ["D1", "D2", "D3", "D4", "D5"]
    dim_short = ["D1", "D2", "D3", "D4", "D5"]
    dim_colors = [COLORS["D1_confidence"], COLORS["D2_sufficiency"],
                  COLORS["D3_similarity"], COLORS["D4_stability"],
                  COLORS["D5_uncertainty"]]

    for col_idx, t in enumerate(TASKS):
        ax_auroc, ax_weight = axes[0, col_idx], axes[1, col_idx]
        res_path = results_dir / f"{t}_deferral_results.csv"
        coef_path = results_dir / f"{t}_coef_stability.csv"
        if not res_path.exists() or not coef_path.exists():
            ax_auroc.axis("off")
            ax_weight.axis("off")
            continue

        res_df = pd.read_csv(res_path)
        coef_df = pd.read_csv(coef_path)

        aurocs, abs_weights = [], []
        for rp, cp in zip(dim_res_policies, dim_coef_prefixes):
            row = res_df[(res_df.policy == rp) & (res_df.coverage == 0.8)]
            aurocs.append(row["auroc"].values[0] if len(row) > 0 else 0)
            crow = coef_df[coef_df.dimension.str.startswith(cp)]
            abs_weights.append(abs(crow["mean"].values[0]) if len(crow) > 0 else 0)

        x = np.arange(len(dim_short))
        ax_auroc.bar(x, aurocs, color=dim_colors, edgecolor="white", alpha=0.85)
        ax_auroc.set_ylim(0.78, 0.93)
        ax_auroc.set_title(TASK_LABELS[t])

        ax_weight.bar(x, abs_weights, color=dim_colors, edgecolor="white", alpha=0.85)
        ax_weight.set_xticks(x)
        ax_weight.set_xticklabels(dim_short, fontsize=8)
        ax_weight.set_xlabel("Dimension")

        if col_idx == 0:
            ax_auroc.set_ylabel("AUROC\n(standalone gate)")
            ax_weight.set_ylabel("|Coefficient|\nin R composite")

    fig.tight_layout()
    path = FIGURE_DIR / "fig4_auroc_vs_weight.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def fig6_decision_curve(dca_csv: str, source: str = "MIMIC-IV") -> str:
    """Net benefit vs. decision threshold: confidence, reliability, treat-all.

    Addresses the gap flagged in review: decision-curve analysis
    (Section "Decision-Curve Analysis") was reported only as a table.
    Net benefit is conventionally shown as a curve across thresholds, not
    a handful of point estimates.
    """
    _ensure_figure_dir()
    df = pd.read_csv(dca_csv)
    df = df[df.source == source]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=False)
    for ax, t in zip(axes, TASKS):
        sub = df[df.task == t].sort_values("threshold")
        if sub.empty:
            ax.axis("off")
            continue
        ax.plot(sub["threshold"], sub["nb_confidence"], label="Confidence Gate",
                 color=COLORS["confidence"], linewidth=1.6, marker="o", markersize=2.5)
        ax.plot(sub["threshold"], sub["nb_reliability"], label="Reliability Gate",
                 color=COLORS["reliability"], linewidth=1.6, marker="s", markersize=2.5)
        ax.plot(sub["threshold"], sub["nb_treat_all"], label="Treat-all",
                 color=COLORS["treat_all"], linewidth=1.2, linestyle="--")
        ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
        ax.set_xlabel(r"Decision threshold $p_t$")
        ax.set_title(TASK_LABELS[t])
        ax.set_xlim(0.0, 0.52)

    axes[0].set_ylabel("Net benefit")
    axes[-1].legend(frameon=True, loc="upper right")
    fig.tight_layout()
    path = FIGURE_DIR / "fig6_decision_curve.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def fig7_shift_robustness(shift_summary_csv: str) -> str:
    """Brier score vs. shift magnitude: missingness and correlated (renal) shift.

    Companion figure to Supplementary Table S2 -- the full magnitude range
    makes the non-monotonic missingness degradation (discussed in
    Section "Robustness Under Distribution Shift") visible at a glance,
    which a condensed 3-row table cannot show.
    """
    _ensure_figure_dir()
    df = pd.read_csv(shift_summary_csv)
    experiments = [("missingness", "Missingness (fraction dropped)"),
                   ("correlated_renal", "Renal-panel shift (training-set SD)")]

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.0), sharex=False)
    for row_idx, (exp, xlabel) in enumerate(experiments):
        for col_idx, t in enumerate(TASKS):
            ax = axes[row_idx, col_idx]
            sub = df[(df.experiment == exp) & (df.task == t)].sort_values("magnitude")
            if sub.empty:
                ax.axis("off")
                continue
            for policy, label, marker in [("confidence", "Confidence Gate", "o"),
                                           ("reliability", "Reliability Gate", "s")]:
                psub = sub[sub.policy == policy]
                ax.plot(psub["magnitude"], psub["brier_mean"], label=label,
                         color=COLORS[policy], linewidth=1.6, marker=marker, markersize=3)
            if row_idx == 0:
                ax.set_title(TASK_LABELS[t])
            if row_idx == 1:
                ax.set_xlabel(xlabel)
            if col_idx == 0:
                ax.set_ylabel(f"Brier score\n({'Missingness' if row_idx == 0 else 'Renal shift'})")

    axes[0, -1].legend(frameon=True, loc="upper left")
    fig.tight_layout()
    path = FIGURE_DIR / "fig7_shift_robustness.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def fig8_cross_institutional(summary_csv: str) -> str:
    """Brier-reduction ratio (confidence / reliability) at 80% coverage,
    grouped by task, one bar per dataset -- visualizes the cross-institutional
    generalization claim that is otherwise only stated in text and tables.
    """
    _ensure_figure_dir()
    df = pd.read_csv(summary_csv)

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.4))
    x = np.arange(len(TASKS))
    w = 0.25
    for i, source in enumerate(DATASETS):
        sub = df[df.source == source].set_index("task").reindex(TASKS)
        ax.bar(x + (i - 1) * w, sub["ratio"], w, label=source,
               color=DATASET_COLORS[source], edgecolor="white")

    ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[t] for t in TASKS])
    ax.set_ylabel("Brier reduction ratio\n(confidence / reliability)")
    ax.legend(frameon=True, loc="upper right", fontsize=7)
    fig.tight_layout()
    path = FIGURE_DIR / "fig8_cross_institutional.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


if __name__ == "__main__":
    results_dir = Path(__file__).resolve().parents[1] / "results"
    csv_path = str(results_dir / "sepsis_deferral_results.csv")
    print(fig1_risk_coverage(csv_path))
    print(fig2_auroc_vs_coverage(csv_path))
    print(fig3_deferral_frontier(csv_path))
    print(fig4_auroc_vs_weight(csv_path, str(results_dir / "sepsis_coef_stability.csv")))
    print(fig5_case_studies(str(results_dir / "sepsis_case_studies.csv")))
