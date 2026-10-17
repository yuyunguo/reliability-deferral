"""Generate the R1 supplement tables and figures from saved results (no hand transcription).

Writes, under deferral_frontier/manuscript/bioengineering/:
    supp_generated_s1_s4.tex   Tables S1-S4, Figure S1
    supp_generated_s6_s10.tex  Tables S6-S10, Figure S3
    figures/fig_s1_perturbation.pdf, figures/fig_s3_calibration.pdf
Table S5 / Figure S2 (D2*) stay hand-written in supplementary_bioengineering.tex.
Numbering follows plan/r1-supplement-map.md.

Usage:
    python -m deferral_frontier.experiments.build_supplement_r1
"""
import logging
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "deferral_frontier" / "results"
OUT = ROOT / "deferral_frontier" / "manuscript" / "bioengineering"
SOURCES = ("MIMIC-IV", "MIMIC-III", "eICU", "eICU-H")
TASKS = ("sepsis", "mortality", "vasopressor")
SETTINGS = ("MIMIC-IV", "MIMIC-III", "eICU", "eICU-H", "X-DB")
SET_LAB = {"MIMIC-IV": "MIMIC-IV", "MIMIC-III": "MIMIC-III", "eICU": "eICU", "eICU-H": "eICU-H",
           "X-DB": "MIMIC-IV$\\rightarrow$eICU"}
GATE_LAB = {"one_sided_confidence": "One-sided", "symmetric_confidence": "Symmetric", "reliability": "Composite",
            "no_deferral": "No deferral"}
V3_LAB = {"D3p": "D3$^{\\mathrm{m}}$", "D5p": "D5$^{\\mathrm{m}}$", "Cp": "C$^{\\mathrm{m}}$", "L": "L",
          "LD": "L+D", "D3_as_built": "D3", "D5_as_built": "D5", "D2": "D2", "D4": "D4",
          "neg_ensemble_sd": "Ensemble SD", "positive_control_k1": "Positive control, $k=1$",
          "positive_control_k2": "Positive control, $k=2$", "positive_control_k4": "Positive control, $k=4$",
          "oracle": "Oracle", "symmetric": "Symmetric"}
PRIMARY_V3 = ("D3p", "D5p", "Cp", "L", "LD")
DB_KEY = {"mimic4": "MIMIC-IV", "mimic3": "MIMIC-III", "eicu": "eICU", "eicu_hospital": "eICU-H"}


def pval(p: float, floor: float) -> str:
    return f"$\\leq${floor:g}" if p <= floor + 1e-12 else f"{p:.3f}"


def ci(d: float, lo: float, hi: float, scale: float = 1.0, k: int = 2) -> str:
    return f"${d * scale:+.{k}f}$ [${lo * scale:+.{k}f}$, ${hi * scale:+.{k}f}$]"


def longtable(label: str, cols: str, header: str, rows: List[str], caption: str, size: str = "\\scriptsize",
              sep: int = 3) -> str:
    return "\n".join([
        "{" + size, f"\\setlength{{\\tabcolsep}}{{{sep}pt}}", f"\\begin{{longtable}}{{{cols}}}",
        f"\\caption*{{\\textbf{{Table {label}.}} {caption}}}\\\\", "\\toprule", header + " \\\\", "\\midrule", "\\endfirsthead",
        "\\toprule", header + " \\\\", "\\midrule", "\\endhead", "\\bottomrule", "\\endlastfoot",
        *rows, "\\end{longtable}", "}"])


def table(label: str, cols: str, header: str, rows: List[str], caption: str, size: str = "\\small", sep: int = 4) -> str:
    return "\n".join(["\\begin{table}[H]", "\\centering", size, f"\\setlength{{\\tabcolsep}}{{{sep}pt}}",
                      f"\\caption*{{\\textbf{{Table {label}.}} {caption}}}",
                      f"\\begin{{tabular}}{{{cols}}}", "\\toprule", header + " \\\\", "\\midrule", *rows,
                      "\\bottomrule", "\\end{tabular}", "\\end{table}"])


# ---------------------------------------------------------------- S1
def s1() -> str:
    frames = [pd.read_csv(f) for f in sorted(RES.glob("base_diagnostics_*.csv"))]
    d = pd.concat(frames)
    d = d[d.task.isin(TASKS)]
    rows = []
    for db in ("mimic4", "mimic3", "eicu", "eicu_hospital"):
        for t in TASKS:
            x = d[(d.dataset == db) & (d.task == t)]
            f = lambda c, k=3: f"{x[c].mean():.{k}f} ({x[c].min():.{k}f}--{x[c].max():.{k}f})"
            rows.append(f"{DB_KEY[db]} & {t.capitalize()} & {x.best_iteration.mean():.0f} "
                        f"({x.best_iteration.min()}--{x.best_iteration.max()}) & {f('train_auroc')} & "
                        f"{f('val_auroc')} & {x.train_logloss.mean():.3f} & {x.val_logloss.mean():.3f} \\\\")
    hdr = ("\\textbf{Database} & \\textbf{Task} & \\textbf{Stopping Iteration} & \\textbf{Train AUROC} & "
           "\\textbf{Validation AUROC} & \\textbf{Train LL} & \\textbf{Val.\\ LL}")
    body = table("S1", "llccccc", hdr, rows, "Training diagnostics of the five-seed XGBoost ensembles: mean "
                 "(minimum--maximum) over ensemble members. Early stopping after 20 rounds without "
                 "improvement in validation log loss (LL). eICU-H: hospital-level split.", "\\scriptsize", 3)
    return ("\\section*{Table S1: Training Diagnostics}\n\nEach ensemble member was trained with the "
            "settings in Section~2.3. No class weighting was used.\n\n" + body + "\n")


# ---------------------------------------------------------------- S2
def s2() -> str:
    pt = pd.read_csv(RES / "paired_gate_tests.csv")
    out = []
    for ga, title in (("symmetric_confidence", "symmetric"), ("one_sided_confidence", "one-sided")):
        rows = []
        for s in SOURCES:
            for t in TASKS:
                r = pt[(pt.source == s) & (pt.task == t) & (pt.gate_a == ga)].iloc[0]
                floor = 0.0015 if s == "eICU-H" else 0.0045
                rows.append(f"{s} & {t.capitalize()} & {ci(r.brier_diff, r.brier_ci_lo, r.brier_ci_hi, 1, 4)} & "
                            f"{pval(r.brier_p_holm, floor)} & {ci(r.auroc_diff, r.auroc_ci_lo, r.auroc_ci_hi, 1, 3)} & "
                            f"{pval(r.auroc_p_holm, floor)} \\\\")
        hdr = ("\\textbf{Database} & \\textbf{Task} & \\textbf{Brier Difference [95\\% CI]} & \\textbf{Adj.\\ $p$} & "
               "\\textbf{AUROC Difference [95\\% CI]} & \\textbf{Adj.\\ $p$}")
        out.append(table("S2a" if ga == "symmetric_confidence" else "S2b", "llcccc", hdr, rows,
                         f"Retained-set Brier score and AUROC at 80\\% coverage, {title} gate minus composite. "
                         "Negative Brier differences favor the first gate.", "\\footnotesize", 3))
    return ("\\section*{Table S2: Paired Gate Comparisons, All Pairs}\n\nPaired bootstrap with 2,000 "
            "resamples (seed 42). $p$-values are Holm-adjusted across the nine primary database--task pairs "
            "and, separately, across the three eICU-H pairs (smallest attainable values 0.0045 and 0.0015).\n\n"
            + "\n\n".join(out) + "\n")


# ---------------------------------------------------------------- S3 + Fig S1
EXP_LAB = {"missingness": "Columns zeroed", "correlated_renal": "Renal panel", "correlated_liver": "Liver panel"}
POL = {"confidence": ("One-sided", "#4C72B0", "o"), "symmetric": ("Symmetric", "#55A868", "s"),
       "reliability": ("Composite", "#DD8452", "^")}


def s3() -> str:
    s = pd.read_csv(RES / "shift_experiments_real_summary.csv")
    rows = []
    for e in EXP_LAB:
        mags = sorted(s[s.experiment == e].magnitude.unique())
        for i, m in enumerate(mags):
            mag = f"{100 * m:.0f}\\%" if e == "missingness" else f"{m:.1f} SD"
            cells = []
            for t in TASKS:
                x = s[(s.experiment == e) & (s.magnitude == m) & (s.task == t)].set_index("policy").brier_mean
                cells += [f"{x['confidence']:.3f}", f"{x['symmetric']:.3f}", f"{x['reliability']:.3f}"]
            lab = f"\\multirow{{{len(mags)}}}{{*}}{{{EXP_LAB[e]}}}" if i == 0 else ""
            rows.append(f"{lab} & {mag} & " + " & ".join(cells) + " \\\\")
        rows.append("\\midrule")
    rows = rows[:-1]
    hdr = ("& & \\multicolumn{3}{c}{\\textbf{Sepsis}} & \\multicolumn{3}{c}{\\textbf{Mortality}} & "
           "\\multicolumn{3}{c}{\\textbf{Vasopressor}} \\\\\n\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-11}\n"
           "\\textbf{Perturbation} & \\textbf{Size} & OS & Sym & Comp & OS & Sym & Comp & OS & Sym & Comp")
    tab = table("S3", "llccccccccc", hdr, rows, "Retained-set Brier score at 80\\% coverage under test-time "
                "perturbation (MIMIC-IV, mean of five replicates, frozen pipelines). OS: one-sided; "
                "Sym: symmetric; Comp: composite. Unperturbed rows recompute D5 with a fresh perturbation draw, "
                "so they can differ from Table 3 in the third decimal.", "\\footnotesize")
    fig, axes = plt.subplots(3, 3, figsize=(10, 8), sharex="row")
    for i, e in enumerate(EXP_LAB):
        for j, t in enumerate(TASKS):
            ax = axes[i, j]
            for pkey, (lab, col, mk) in POL.items():
                x = s[(s.experiment == e) & (s.task == t) & (s.policy == pkey)].sort_values("magnitude")
                ax.plot(x.magnitude * (100 if e == "missingness" else 1), x.brier_mean, marker=mk, color=col,
                        label=lab, ms=4)
            if i == 0:
                ax.set_title(t.capitalize(), fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{EXP_LAB[e]}\nretained Brier", fontsize=8)
            ax.set_xlabel("% of columns zeroed" if e == "missingness" else "Shift (training SD)", fontsize=8)
            ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig_s1_perturbation.pdf", bbox_inches="tight")
    plt.close(fig)
    figtex = ("\\section*{Figure S1: Perturbation Curves}\n\n\\begin{figure}[H]\n\\centering\n"
              "\\includegraphics[width=\\textwidth]{figures/fig_s1_perturbation.pdf}\n"
              "\\caption*{\\textbf{Figure S1.} Retained-set Brier score at 80\\% coverage versus perturbation size for the three gates "
              "(MIMIC-IV). Zeroing columns injects out-of-range values and is a stress test, not a model of "
              "missingness.}\n\\end{figure}\n")
    return ("\\section*{Table S3: Test-Time Perturbation}\n\nFrozen MIMIC-IV pipelines were applied to perturbed "
            "test features. D2 and D4 are computed from raw time series and were held at their unperturbed "
            "values.\n\n" + tab + "\n\n" + figtex)


# ---------------------------------------------------------------- S4
def s4() -> str:
    f = pd.read_csv(RES / "fairness_r1.csv")
    order = ["Female", "Male", "18-39", "40-54", "55-64", "65-74", "75+", "White", "Black", "Hispanic", "Asian",
             "Other/Unknown"]
    rows = []
    m = f[f.source == "MIMIC-IV"]
    for t in TASKS:
        first = True
        for g in order:
            cells = []
            for gate in ("symmetric_confidence", "reliability"):
                r = m[(m.task == t) & (m.group == g) & (m.gate == gate)]
                if r.empty:
                    cells += ["--"] * 4
                    continue
                r = r.iloc[0]
                cells += [f"{100 * r.deferral_rate:.1f}", f"{100 * r.error_capture:.1f}", f"{r.brier_retained:.3f}",
                          f"{100 * r.inappropriate_deferral_rate:.1f}"]
            n = m[(m.task == t) & (m.group == g)].n.iloc[0]
            lab = f"\\multirow{{12}}{{*}}{{{t.capitalize()}}}" if first else ""
            first = False
            rows.append(f"{lab} & {g.replace('-', '--')} & {n:,} & " + " & ".join(cells) + " \\\\")
        rows.append("\\midrule")
    rows = rows[:-1]
    hdr = ("& & & \\multicolumn{4}{c}{\\textbf{Symmetric}} & \\multicolumn{4}{c}{\\textbf{Composite}} \\\\\n"
           "\\cmidrule(lr){4-7}\\cmidrule(lr){8-11}\n\\textbf{Task} & \\textbf{Group} & \\textbf{$n$} & Def.\\ & Capt.\\ & "
           "Brier & Inapp.\\ & Def.\\ & Capt.\\ & Brier & Inapp.")
    t1 = table("S4a", "llrcccccccc", hdr, rows, "Subgroup audit, MIMIC-IV, 80\\% coverage. Def.: deferral rate (\\%). "
               "Capt.: share of the group's misclassifications at 0.5 that were deferred (\\%). Brier: retained-set "
               "Brier score. Inapp.: correctly classified patients deferred, per patient in the group (\\%).",
               "\\scriptsize")
    sp = f.groupby(["source", "task", "gate", "stratification"]).deferral_rate.agg(lambda v: 100 * (v.max() - v.min()))
    rows2 = []
    for s in SOURCES:
        for t in TASKS:
            cells = []
            for strat in ("sex", "age_band", "race_group"):
                cells += [f"{sp[(s, t, g, strat)]:.1f}" for g in ("one_sided_confidence", "symmetric_confidence",
                                                                  "reliability")]
            rows2.append(f"{s} & {t.capitalize()} & " + " & ".join(cells) + " \\\\")
    hdr2 = ("& & \\multicolumn{3}{c}{\\textbf{Sex}} & \\multicolumn{3}{c}{\\textbf{Age Band}} & "
            "\\multicolumn{3}{c}{\\textbf{Race/Ethnicity}} \\\\\n\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-11}\n"
            "\\textbf{Database} & \\textbf{Task} & OS & Sym & Comp & OS & Sym & Comp & OS & Sym & Comp")
    t2 = table("S4b", "llccccccccc", hdr2, rows2, "Largest minus smallest deferral rate across groups (percentage points), "
               "all database--task pairs. OS: one-sided; Sym: symmetric; Comp: composite.", "\\footnotesize")
    return ("\\section*{Table S4: Subgroup Audit}\n\nAge bands: 18--39, 40--54, 55--64, 65--74, and $\\geq$75 "
            "years. Results for every database are in the code release.\n\n" + t1 + "\n\n" + t2 + "\n")


# ---------------------------------------------------------------- S6
def s6() -> str:
    v = pd.read_csv(RES / "r1_v3_stratified.csv")
    show = ["D3p", "D5p", "Cp", "L", "LD", "D3_as_built", "D5_as_built", "neg_ensemble_sd", "positive_control_k1",
            "positive_control_k2", "positive_control_k4", "oracle", "symmetric"]
    rows = []
    k10 = v[v.K == 10]
    for s in SETTINGS:
        for t in TASKS:
            for sc in show:
                r = k10[(k10.setting == s) & (k10.task == t) & (k10.score == sc)].iloc[0]
                adj = "--" if sc in ("oracle", "symmetric") else ci(r.delta, r.ci_lo, r.ci_hi, 1e3, 1)
                ph = f"{r.p_holm:.3f}" if pd.notna(r.p_holm) else "--"
                rnd = ci(r.delta_vs_random, r.ci_lo_vs_random, r.ci_hi_vs_random, 1e3, 1)
                rows.append(f"{SET_LAB[s]} & {t.capitalize()} & {V3_LAB[sc]} & {adj} & {ph} & {rnd} & "
                            f"{100 * r.share_of_oracle:.1f}\\% \\\\")
            rows.append("\\midrule")
    rows = rows[:-1]
    hdr = ("\\textbf{Setting} & \\textbf{Task} & \\textbf{Score} & \\textbf{$\\Delta_{\\mathrm{adj}}$ [95\\% CI]} & "
           "\\textbf{Holm $p$} & \\textbf{$\\Delta$ vs.\\ Random [95\\% CI]} & \\textbf{Share}")
    lt = longtable("S6a", "lllcccc", hdr, rows, "Risk-matched test, $K=10$ bins ($\\times10^{-3}$). $\\Delta_{\\mathrm{adj}}$: "
                   "gain over the symmetric gate at matched risk (primary). $\\Delta$ vs.\\ random: gain over random "
                   "deferral within bins. Share: $\\Delta$ vs.\\ random as a percentage of the oracle's. Holm $p$ is given for the five primary scores (D3$^{\\mathrm{m}}$, "
                   "D5$^{\\mathrm{m}}$, C$^{\\mathrm{m}}$, L, L+D), adjusted within each setting.", "\\scriptsize", 1.5)
    k20 = v[(v.K == 20) & (v.family == "primary")]
    sens = (f"With $K = 20$ bins, none of the 75 primary tests had a CI above zero ({int((k20.delta > 0).sum())} "
            f"positive point estimates).")
    inc = pd.read_csv(RES / "r1_v3_increment.csv")
    rows_i = [f"{SET_LAB[r.setting]} & {r.task.capitalize()} & {ci(r.delta, r.ci_lo, r.ci_hi, 1e3, 1)} & "
              f"{r.p_holm:.3f} \\\\" for r in inc.itertuples()]
    ti = table("S6b", "llcc", "\\textbf{Setting} & \\textbf{Task} & \\textbf{$\\Delta$(L+D) $-$ $\\Delta$(L) [95\\% CI]} & "
               "\\textbf{Holm $p$}", rows_i, "Added value of the dimensions to the learned error model "
               "($\\times10^{-3}$; Holm within setting across tasks).", "\\footnotesize")
    sh = pd.read_csv(RES / "r1_v3_shift_contrast.csv")
    rows_s = [f"{r.task.capitalize()} & {r.K} & {ci(r.delta, r.ci_lo, r.ci_hi, 1e3, 2)} & {r.p_holm:.3f} \\\\"
              for r in sh.itertuples()]
    ts = table("S6c", "lccc", "\\textbf{Task} & \\textbf{$K$} & \\textbf{Difference [95\\% CI]} & \\textbf{Holm $p$}", rows_s,
               "Shift contrast: $\\Delta_{\\mathrm{adj}}$ of L in MIMIC-IV$\\rightarrow$eICU minus that in eICU, on "
               "the same test patients ($\\times10^{-3}$; paired bootstrap; Holm across tasks within $K$). "
               "$K=10$ is primary.", "\\footnotesize")
    a = pd.read_csv(RES / "r1_stratified_deferral.csv")
    a = a[(a.K == 10) & (a.family == "primary")]
    lab_a = {"D2": "D2", "D3": "D3", "D4": "D4", "D5": "D5", "neg_ensemble_sd": "Ensemble SD", "v2": "Composite",
             "ridge": "Ridge composite"}
    rows_a = []
    for sc, lab in lab_a.items():
        x = a[a.score == sc]
        rows_a.append(f"{lab} & {int(((x.delta > 0) & (x.p_holm < .05)).sum())}/12 & "
                      f"{int(((x.delta < 0) & (x.p_holm < .05)).sum())}/12 & "
                      f"${x.delta.min() * 1e3:+.1f}$ to ${x.delta.max() * 1e3:+.1f}$ \\\\")
    sym = pd.read_csv(RES / "r1_stratified_deferral.csv")
    sym = sym[(sym.K == 10) & (sym.score == "symmetric")]
    rows_a.append(f"Symmetric (reference) & -- & -- & ${sym.delta.min() * 1e3:+.1f}$ to ${sym.delta.max() * 1e3:+.1f}$ \\\\")
    ta = table("S6d", "lccc", "\\textbf{Score} & \\textbf{Above Random} & \\textbf{Below Random} & \\textbf{Range of $\\Delta$}",
               rows_a, "First, random-referenced analysis (12 database--task pairs, as-built scores, $K=10$, "
               "$\\times10^{-3}$). Counts are Holm-adjusted $p<0.05$ within each database over seven scores and "
               "three tasks.", "\\footnotesize")
    dsc = pd.read_csv(RES / "r1_v3_descriptive.csv")
    rows_e = []
    for s_ in SETTINGS:
        for t in TASKS:
            r = dsc[(dsc.setting == s_) & (dsc.task == t)].iloc[0]
            rows_e.append(f"{SET_LAB[s_]} & {t.capitalize()} & {100 * r.mece:.1f}\\% & {r.brier80_symmetric:.4f} & "
                          f"{r.brier80_Cp:.4f} & {r.brier80_L:.4f} \\\\")
    te = table("S6e", "llcccc", "\\textbf{Setting} & \\textbf{Task} & \\textbf{M-ECE} & \\textbf{Symmetric} & "
               "\\textbf{C$^{\\mathrm{m}}$} & \\textbf{L}", rows_e,
               "Descriptive: calibration of $\\tilde p$ (M-ECE) and retained-set Brier score at 80\\% coverage "
               "without risk matching, by setting.", "\\footnotesize")
    return ("\\section*{Table S6: Risk-Matched Deferral Test, Full Results}\n\n" + lt + "\n\n" + sens + "\n\n"
            + ti + "\n\n" + ts + "\n\n" + ta + "\n\n" + te + "\n")


# ---------------------------------------------------------------- S7 + Fig S3
def s7() -> str:
    c = pd.read_csv(RES / "class_conditional_gates.csv")
    rows = []
    for s in SOURCES:
        for t in TASKS:
            for i, g in enumerate(("no_deferral", "one_sided_confidence", "symmetric_confidence", "reliability")):
                r = c[(c.source == s) & (c.task == t) & (c.gate == g)].iloc[0]
                lab = f"\\multirow{{4}}{{*}}{{{s} {t}}}" if i == 0 else ""
                rows.append(f"{lab} & {GATE_LAB[g]} & {r.n_pos:,} & {r.n_neg:,} & {r.brier_pos:.3f} & "
                            f"{r.brier_neg:.3f} & {r.sensitivity:.3f} & {r.specificity:.3f} \\\\")
            rows.append("\\midrule")
    rows = rows[:-1]
    hdr = ("\\textbf{Pair} & \\textbf{Gate} & \\textbf{$n^+$} & \\textbf{$n^-$} & \\textbf{Brier$^+$} & "
           "\\textbf{Brier$^-$} & \\textbf{Sens.} & \\textbf{Spec.}")
    lt = longtable("S7a", "llrrcccc", hdr, rows, "Class-conditional performance of retained predictions at 80\\% "
                   "coverage. Brier$^+$/Brier$^-$: Brier score among retained patients with/without the outcome. "
                   "Sensitivity and specificity at a threshold of 0.5.")
    cm = pd.read_csv(RES / "confusion_matrix_05.csv")
    rows_c = [f"{r.source} & {r.task.capitalize()} & {r.tp:,} & {r.fn:,} & {r.fp:,} & {r.tn:,} \\\\"
              for r in cm.itertuples()]
    tc = table("S7b", "llrrrr", "\\textbf{Database} & \\textbf{Task} & \\textbf{TP} & \\textbf{FN} & \\textbf{FP} & \\textbf{TN}",
               rows_c, "Confusion matrices at 0.5, full test sets (calibrated ensemble, no deferral).", "\\footnotesize")
    b = pd.read_csv(RES / "calibration_bins.csv")
    fig, axes = plt.subplots(3, 4, figsize=(11, 8.2), sharex=True, sharey=True)
    for j, s in enumerate(SOURCES):
        for i, t in enumerate(TASKS):
            ax = axes[i, j]
            x = b[(b.source == s) & (b.task == t)].sort_values("bin")
            ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls="--")
            ax.plot(x.mean_pred, x.obs_rate, marker="o", ms=3.5, color="#4C72B0")
            if i == 0:
                ax.set_title(s, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{t.capitalize()}\nobserved rate", fontsize=8)
            if i == 2:
                ax.set_xlabel("Mean predicted risk", fontsize=8)
            ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig_s3_calibration.pdf", bbox_inches="tight")
    plt.close(fig)
    figtex = ("\\section*{Figure S3: Calibration}\n\n\\begin{figure}[H]\n\\centering\n"
              "\\includegraphics[width=\\textwidth]{figures/fig_s3_calibration.pdf}\n"
              "\\caption*{\\textbf{Figure S3.} Calibration of the isotonic-calibrated ensemble on the test sets: observed outcome rate "
              "versus mean predicted risk in 10 equal-mass bins. Dashed line: perfect calibration.}\n\\end{figure}\n")
    return ("\\section*{Table S7: Class-Conditional Performance and Confusion Matrices}\n\n" + lt + "\n\n" + tc
            + "\n\n" + figtex)


# ---------------------------------------------------------------- S8
def s8() -> str:
    s3 = pd.read_csv(RES / "sepsis3_sensitivity.csv")
    rows = []
    for lab_def, sub in s3.groupby("label_definition", sort=False):
        for i, g in enumerate(("one_sided_confidence", "symmetric_confidence", "reliability")):
            r = sub[sub.gate == g].iloc[0]
            lab = f"\\multirow{{3}}{{*}}{{{lab_def} ({100 * r.prevalence:.1f}\\%)}}" if i == 0 else ""
            rows.append(f"{lab} & {GATE_LAB[g]} & {r.brier:.3f} & {r.auroc:.3f} & {100 * r.retained_prevalence:.1f}\\% "
                        f"& {r.nnr:.1f} \\\\")
        rows.append("\\midrule")
    rows = rows[:-1]
    r0 = s3[s3.label_definition == "Sepsis-3"].iloc[0]
    t1 = table("S8a", "llcccc", "\\textbf{Label (Prevalence)} & \\textbf{Gate} & \\textbf{Brier} & \\textbf{AUROC} & "
               "\\textbf{Retained Prev.} & \\textbf{NNR}", rows,
               "MIMIC-IV sepsis with ICD-based and Sepsis-3 labels, 80\\% coverage. Sepsis-3: symmetric minus "
               f"composite Brier difference 95\\% CI {r0.brier_sym_minus_rel_ci_lo:+.4f} to "
               f"{r0.brier_sym_minus_rel_ci_hi:+.4f}.", "\\footnotesize")
    cs = pd.read_csv(RES / "clinical_score_baseline.csv")
    rows2 = [f"{r.source} & {r.model} & {r.n:,} & {r.auroc:.3f} & {r.brier:.3f} & {r.brier_symmetric_80:.3f} \\\\"
             for r in cs.itertuples()]
    t2 = table("S8b", "llrccc", "\\textbf{Database} & \\textbf{Model} & \\textbf{$n$} & \\textbf{AUROC} & \\textbf{Brier} & "
               "\\textbf{Symmetric, 80\\%}", rows2,
               "In-hospital mortality: clinical scores versus the XGBoost model on the same test patients. Last "
               "column: retained Brier score under the symmetric gate at 80\\% coverage. eICU rows are restricted "
               "to stays with an APACHE~IVa prediction.", "\\footnotesize")
    return ("\\section*{Table S8: Sepsis-3 Labels and Clinical Scores}\n\nSepsis-3 labels were taken from "
            "\\texttt{mimic\\_derived.sepsis3} for the cohort's ICU stay; features and splits were unchanged. "
            "First-day SOFA was recalibrated to a probability by logistic regression on the validation set.\n\n"
            + t1 + "\n\n" + t2 + "\n")


# ---------------------------------------------------------------- S9
def s9() -> str:
    d = pd.read_csv(RES / "imperfect_review.csv")
    rows = []
    for s in SOURCES:
        for t in TASKS:
            for i, g in enumerate(("one_sided_confidence", "symmetric_confidence", "reliability")):
                x = d[(d.source == s) & (d.task == t) & (d.gate == g)].set_index("clinician_accuracy")
                r = x.loc[1.0]
                lab = f"\\multirow{{3}}{{*}}{{{s} {t}}}" if i == 0 else ""
                rows.append(f"{lab} & {GATE_LAB[g]} & {r.n_deferred:,} & {r.model_errors_deferred:,} & "
                            f"{100 * r.break_even_accuracy:.1f} & {100 * r.error_rate_no_deferral:.1f} & "
                            + " & ".join(f"{100 * x.loc[a].error_rate_with_review:.1f}" for a in (1.0, 0.9, 0.8, 0.7))
                            + " \\\\")
            rows.append("\\midrule")
    rows = rows[:-1]
    hdr = ("\\textbf{Pair} & \\textbf{Gate} & \\textbf{Deferred} & \\textbf{Errors Def.} & \\textbf{Break-Even (\\%)} & "
           "\\textbf{No Def.} & \\textbf{100\\%} & \\textbf{90\\%} & \\textbf{80\\%} & \\textbf{70\\%}")
    lt = longtable("S9", "llrrcccccc", hdr, rows, "Imperfect clinician review at 80\\% coverage. Errors Def.: model "
                   "misclassifications at 0.5 among deferred patients. Break-even: reviewer accuracy below which "
                   "deferral adds errors. Last five columns: overall error rate (\\%) without deferral and with "
                   "review at the stated reviewer accuracy.")
    return ("\\section*{Table S9: Imperfect Clinician Review}\n\nDeferred patients were assumed to be decided by a "
            "reviewer who is correct with the stated probability; expected total errors are model errors among "
            "retained patients plus reviewer errors among deferred patients.\n\n" + lt + "\n")


# ---------------------------------------------------------------- S10
def s10() -> str:
    p = pd.read_csv(RES / "r1_whole_cohort_panel.csv")
    gates = (("one_sided", "One-sided"), ("symmetric", "Symmetric"), ("v2", "Composite"), ("random", "Random"))
    rows = []
    for s in SOURCES:
        for t in TASKS:
            x = p[(p.source == s) & (p.task == t)].set_index("gate")
            cells = [f"{x.loc[g, 'bss']:.3f} ({100 * x.loc[g, 'prevalence']:.1f}\\%)" for g, _ in gates]
            r = x.iloc[0]
            rows.append(f"{s} & {t.capitalize()} & " + " & ".join(cells) + " & "
                        + ci(r.bss_sym_minus_one_sided, r.ci_lo, r.ci_hi, 1, 3) + " \\\\")
    hdr = ("\\textbf{Database} & \\textbf{Task} & " + " & ".join(f"\\textbf{{{lab}}}" for _, lab in gates)
           + " & \\textbf{Sym.\\ $-$ One-Sided [95\\% CI]}")
    t1 = table("S10", "llccccc", hdr, rows, "Retained-set Brier skill score, $1-\\mathrm{Brier}/[\\pi(1-\\pi)]$, at 80\\% "
               "coverage (retained prevalence $\\pi$ in parentheses). Last column: paired bootstrap (1,000 "
               "resamples) for the symmetric minus one-sided difference.", "\\scriptsize")
    return ("\\section*{Table S10: Retained-Set Brier Skill Score}\n\nThe skill score scales the Brier score by "
            "the outcome variance of the retained population, so gates that retain populations of different "
            "prevalence can rank differently than on the Brier score (main text, Section~3.7).\n\n" + t1 + "\n")



# ---------------------------------------------------------------- S12
def s12() -> str:
    rows = []
    ss = pd.read_csv(RES / "r1_split_seed_sensitivity.csv")
    for (src, t), g in ss.groupby(["source", "task"], sort=False):
        rows.append(f"Split seeds & {src} {t} & {len(g)} seeds; composite better (CI$>$0) {int((g.ci_lo > 0).sum())}, "
                    f"symmetric better (CI$<$0) {int((g.ci_hi < 0).sum())}; difference "
                    f"{g.sym_minus_comp.min():+.4f} to {g.sym_minus_comp.max():+.4f} \\\\")
    from scipy.stats import spearmanr
    arts = {"MIMIC-IV": "{t}_test_artifacts.parquet", "MIMIC-III": "mimic3/mimic3_{t}_test_artifacts.parquet",
            "eICU": "eicu/eicu_{t}_test_artifacts.parquet",
            "eICU-H": "eicu_hospital/eicu_hospital_{t}_test_artifacts.parquet"}
    for s_, tmpl in arts.items():
        db = {"MIMIC-IV": "mimic4", "MIMIC-III": "mimic3", "eICU": "eicu", "eICU-H": "eicu_hospital"}[s_]
        for t in TASKS:
            a = pd.read_parquet(ROOT / "artifacts" / tmpl.format(t=t))
            rho = spearmanr(a.r_score, a.cal_proba * (1 - a.cal_proba))[0]
            imp = pd.read_csv(RES / f"composite_v2_importance_{db}_{t}.csv", index_col=0).iloc[:, 0]
            sh = imp / imp.sum()
            rows.append(f"Composite & {s_} {t} & Spearman($R$, $\\tilde p(1-\\tilde p)$) {rho:.2f}; gain share D1 "
                        f"{sh.iloc[0]:.2f}, D2 {sh.iloc[1]:.2f}, D3 {sh.iloc[2]:.2f}, D4 {sh.iloc[3]:.2f}, D5 {sh.iloc[4]:.2f} \\\\")
    ab = pd.read_csv(RES / "composite_v2_ablation.csv")
    for r in ab[ab.variant == "full"].itertuples():
        rows.append(f"Held-out error prediction & {r.source} {r.task} & full model mean squared error vs.\\ D1 alone: "
                    f"{100 * -r.error_pred_mse_reduction_vs_D1_only:+.1f}\\% \\\\")
    nb = pd.read_csv(RES / "neural_baselines.csv")
    for r in nb.drop_duplicates("task").itertuples():
        rows.append(f"MLP & MIMIC-IV {r.task} & symmetric $-$ composite Brier 95\\% CI {r.brier_sym_minus_rel_ci_lo:+.4f} "
                    f"to {r.brier_sym_minus_rel_ci_hi:+.4f} \\\\")
    g = pd.read_csv(RES / "symmetric_confidence_gate.csv")
    j = g[g.gate == "symmetric_confidence"].jaccard_deferred_vs_one_sided
    rows.append(f"Overlap & all 12 pairs & Jaccard, symmetric vs.\\ one-sided deferred sets: maximum {j.max():.3f} \\\\")
    nbt = pd.read_csv(RES / "symmetric_gate_net_benefit.csv")
    x = nbt[(nbt.source == "MIMIC-IV") & (nbt.threshold.round(2) == 0.10)]
    for t in TASKS:
        v = x[x.task == t].set_index("gate").net_benefit
        rows.append(f"Net benefit, $p_t=0.10$ & MIMIC-IV {t} & no deferral {v['no_deferral']:.3f}; one-sided "
                    f"{v['one_sided_confidence']:.3f}; symmetric {v['symmetric_confidence']:.3f}; composite "
                    f"{v['reliability']:.3f} \\\\")
    for db, lab in (("mimic4", "MIMIC-IV"), ("mimic3", "MIMIC-III"), ("eicu", "eICU")):
        f = pd.read_parquet(ROOT / "data" / db / "mortality_features.parquet")
        rows.append(f"Documentation & {lab} (mortality cohort) & median heart-rate observations "
                    f"{f.vital_heart_rate_count.median():.0f}, respiratory rate {f.vital_resp_rate_count.median():.0f}; "
                    f"temperature (F) observed {100 * f.vital_temp_f_mean.notna().mean():.1f}\\%, (C) "
                    f"{100 * f.vital_temp_c_mean.notna().mean():.1f}\\%; $\\geq$1 lactate {100 * (f.lab_lactate_count > 0).mean():.1f}\\% \\\\")
    lt = longtable("S12", "lp{3.2cm}p{9.4cm}", "\\textbf{Analysis} & \\textbf{Setting} & \\textbf{Result}", rows,
                   "Secondary results cited in the main text without a dedicated display.", "\\scriptsize", 3)
    return "\\section*{Table S12: Secondary Results Cited in the Text}\n\n" + lt + "\n"


# ---------------------------------------------------------------- S13
S13_ROWS = [
    ("Splits", "Seven stratified group folds by patient (one test, one validation), seed 42, separately per task; eICU-H uses hospitals as groups."),
    ("Base model", "Five XGBoost classifiers (seeds 0--4): up to 500 trees, depth 6, learning rate 0.05, subsample and column subsample 0.8, minimum child weight 5, L2 penalty 1; early stopping after 20 rounds on validation log loss; native missing-value handling; no class weighting."),
    ("Calibration", "Isotonic regression (out-of-range inputs clipped) of the ensemble mean on the full validation set gives $\\tilde p$ at test time."),
    ("Imputation for D3/D5", "Training medians (0 for a feature missing in every training row, e.g., Fahrenheit temperature in eICU), then standardization with training mean and SD."),
    ("Composite", "XGBoost regressor of $(\\tilde p-y)^2$ (depth 3, 300 trees, learning rate 0.05, subsample 0.8, seed 42), monotone in D1; cross-fitted on two patient-disjoint validation halves (seed 42), isotonic calibration fitted on one half and the target computed on the other; at test time D1 uses the full-validation $\\tilde p$."),
    ("D5 (composite input)", "Variance of the primary model's prediction over 50 passes with 10\\% of median-imputed inputs set to zero (seed 42)."),
    ("D3$^{\\mathrm{m}}$", "Ledoit--Wolf Mahalanobis distance on standardized, median-imputed features plus binary indicators for features with any training missingness; $\\exp(-d/\\text{median training } d)$."),
    ("D5$^{\\mathrm{m}}$", "Primary model (ensemble member 1) on unimputed features; 50 passes with 10\\% of observed entries set to missing (seed 42); score $=-$variance."),
    ("C$^{\\mathrm{m}}$", "Ridge regression ($\\alpha=1$, standardized inputs) of $(\\tilde p-y)^2$ on $\\tilde p(1-\\tilde p)$, D2, D3$^{\\mathrm{m}}$, D4, D5$^{\\mathrm{m}}$, $-$ensemble SD; same cross-fitting as the composite."),
    ("L, L+D", "XGBoost regressor (depth 2, 200 trees, learning rate 0.03, subsample 0.8, column subsample 0.5, minimum child weight 50, seed 42) on $\\tilde p(1-\\tilde p)$ and the 115 unimputed features (L+D: plus the six dimension inputs of C$^{\\mathrm{m}}$); same cross-fitting."),
    ("D2*", "For each feature with training missingness, a ridge model ($\\alpha=1$) on the other median-imputed features predicts the value; a donor is drawn from the 5 nearest observed training values (predictive mean matching); 50 completions; variance of the primary model's prediction (seed 42)."),
    ("Risk-matched test", "10 (sensitivity: 20) equal-size bins of $\\tilde p$, ties broken by a fixed permutation (seed 42); $\\mathrm{round}(0.2\\,n_b)$ lowest-scoring patients deferred per bin; 1,000 paired bootstrap resamples (seed 42) with bins recomputed; two-sided $p$ floored at 0.001; positive-control noise SD $=k\\,\\mathrm{SD}(e)$."),
    ("Whole-cohort gates", "Rank by score with a stable sort (ties keep row order); retain 80\\%; paired bootstrap with 2,000 resamples (seed 42)."),
    ("MLP", "Hidden layers of 128 and 64 units (ReLU, dropout 0.1); Adam (learning rate $10^{-3}$, weight decay $10^{-4}$); up to 60 epochs with early stopping (patience 8) on validation loss; missing values set to 0 and features standardized; five seeds; isotonic calibration."),
    ("SelectiveNet", "Same trunk with a selection head; target coverage 0.8, coverage penalty $\\lambda=32$; Adam (learning rate $10^{-3}$); up to 80 epochs, patience 10."),
    ("Other scores", "Ensemble variance: SD across the five members. Monte Carlo dropout: SD over 50 D5-type passes. Temperature scaling: one temperature fitted by validation log loss (bounded search 0.01--10), score $=-$entropy. Bayesian logistic regression: Laplace approximation with Gaussian prior variance 1 on standardized features, score $=-$posterior logit variance."),
]


def s13() -> str:
    rows = [f"{a} & {b} \\\\" for a, b in S13_ROWS]
    t = table("S13", "p{3.0cm}p{11.0cm}", "\\textbf{Component} & \\textbf{Specification}", rows,
              "Implementation details.", "\\footnotesize", 3)
    return "\\section*{Table S13: Implementation Details}\n\n" + t + "\n"


# ---------------------------------------------------------------- S14 (post hoc)
def s14() -> str:
    pre = RES / "r1_v3_posthoc_"
    t = pd.read_csv(str(pre) + "titration.csv")
    rows = []
    for (st, tk), g in t.groupby(["setting", "task"], sort=False):
        g = g.set_index("k")
        rows.append(f"{SET_LAB[st]} & {tk.capitalize()} & " + " & ".join(
            f"{1e3 * g.loc[k, 'mean_delta_adj']:+.1f} ({100 * g.loc[k, 'detect_rate']:.0f}\\%)" for k in (4, 8, 16, 32)) + " \\\\")
    ta = table("S14a", "llcccc", "\\textbf{Setting} & \\textbf{Task} & $k=4$ & $k=8$ & $k=16$ & $k=32$", rows,
               "Titrated positive controls: mean gain over the symmetric gate ($\\times10^{-3}$) and the share of 10 "
               "injections detected (95\\% CI above zero; 100 bootstrap resamples each).", "\\footnotesize")
    c = pd.read_csv(str(pre) + "cluster.csv")
    rows = [f"{SET_LAB[r.setting]} & {r.task.capitalize()} & {V3_LAB[r.score]} & {ci(r.delta, r.ci_lo, r.ci_hi, 1e3, 1)} \\\\"
            for r in c.itertuples()]
    tb = longtable("S14b", "lllc", "\\textbf{Setting} & \\textbf{Task} & \\textbf{Score} & "
                   "\\textbf{$\\Delta_{\\mathrm{adj}}$ [95\\% CI]}", rows,
                   "Hospital-cluster bootstrap (1,000 resamples of eICU hospitals) of the risk-matched gain over the "
                   "symmetric gate ($\\times10^{-3}$).")
    l = pd.read_csv(str(pre) + "lrt.csv")
    rows = [f"{SET_LAB[r.setting]} & {r.task.capitalize()} & {V3_LAB[r.score]} & {r.lr_stat:.1f} & "
            f"{('$<$0.001' if r.p_holm < 0.001 else f'{r.p_holm:.3f}')} \\\\" for r in l.itertuples()]
    tc = longtable("S14c", "lllcc", "\\textbf{Setting} & \\textbf{Task} & \\textbf{Score} & \\textbf{Likelihood-Ratio $\\chi^2_1$} & "
                   "\\textbf{Holm $p$}", rows,
                   "Likelihood-ratio test of whether each score adds outcome information to a natural cubic spline "
                   "(5 df) of logit $\\tilde p$, all test patients; Holm correction within setting (15 tests).")
    r = pd.read_csv(str(pre) + "recal.csv")
    rows = [f"{x.task.capitalize()} & {100 * x.mece_transferred:.1f}\\% & {100 * x.mece_recalibrated:.1f}\\% & "
            f"{x.brier80_sym_transferred:.4f} & {x.brier80_Cm:.4f} & {x.brier80_sym_recalibrated:.4f} & "
            f"{ci(x.cm_minus_sym_diff, x.cm_minus_sym_ci_lo, x.cm_minus_sym_ci_hi, 1, 4)} \\\\" for x in r.itertuples()]
    td = table("S14d", "lcccccc", "\\textbf{Task} & \\textbf{M-ECE} & \\textbf{M-ECE, Recal.} & \\textbf{Sym.} & "
               "\\textbf{C$^{\\mathrm{m}}$} & \\textbf{Sym., Recal.} & \\textbf{C$^{\\mathrm{m}}$ $-$ Sym.\\ [95\\% CI]}", rows,
               "MIMIC-IV$\\rightarrow$eICU: calibration and retained Brier score at 80\\% coverage, without risk matching, "
               "for the transferred model and after isotonic recalibration on eICU validation patients (target-site "
               "labels).", "\\footnotesize")
    pl = pd.read_csv(str(pre) + "planted.csv")
    rows = [f"{x.task.capitalize()} & {'L with planted feature' if x.score == 'Lz' else 'Planted feature alone'} & "
            f"{ci(x.delta, x.ci_lo, x.ci_hi, 1e3, 1)} \\\\" for x in pl.itertuples()]
    te = table("S14e", "llc", "\\textbf{Task} & \\textbf{Score} & \\textbf{$\\Delta_{\\mathrm{adj}}$ [95\\% CI]}", rows,
               "Planted-signal recovery, MIMIC-IV: a feature $z = e + \\varepsilon$ (noise SD $2\\,\\mathrm{SD}(e)$) was "
               "appended to L's inputs; gain over the symmetric gate ($\\times10^{-3}$).", "\\footnotesize")
    return ("\\section*{Table S14: Post Hoc Robustness Analyses}\n\nThese analyses were added after the planned "
            "results were known and are reported as post hoc.\n\n" + "\n\n".join([ta, tb, tc, td, te]) + "\n")


# ---------------------------------------------------------------- S15 (post hoc extra analyses)
def s15() -> str:
    hp = pd.read_csv(RES / "r1_extra_hospital_pooled.csv")
    rows = [f"{r.task.capitalize()} & {r.k} & {1e3 * r.pooled:+.1f} [{1e3 * r.ci_lo:+.1f}, {1e3 * r.ci_hi:+.1f}] & "
            f"{100 * r.i2:.0f}\\% & {r.n_favor_symmetric}/{r.n_favor_symmetric + r.n_favor_composite} & "
            f"{r.citl_min:+.3f} to {r.citl_max:+.3f} \\\\" for r in hp.itertuples()]
    ta = table("S15a", "lccccc", "\\textbf{Task} & \\textbf{Hospitals} & \\textbf{Pooled Sym.\\ $-$ Comp.\\ ($\\times10^{-3}$)} & "
               "\\textbf{$I^2$} & \\textbf{Favor Sym.} & \\textbf{Calibration-in-the-Large}", rows,
               "Per-hospital analysis, eICU held-out hospitals (hospitals with $\\geq$200 test patients and "
               "$\\geq$10 events): random-effects pooled retained-Brier difference, symmetric minus composite, at "
               "80\\% coverage within hospital (DerSimonian--Laird; within-hospital bootstrap variances).", "\\footnotesize")
    th = pd.read_csv(RES / "r1_extra_threshold.csv")
    rows = []
    for t, g in th.groupby("t"):
        rows.append(f"{t:.2f} & {int((g.ci_lo > 0).sum())}/12 & {int((g.ci_hi < 0).sum())}/12 & "
                    f"{1e3 * g.comp_minus_tcent.min():+.1f} to {1e3 * g.comp_minus_tcent.max():+.1f} \\\\")
    tb = table("S15b", "lccc", "\\textbf{Threshold $t$} & \\textbf{Composite Worse} & \\textbf{Composite Better} & "
               "\\textbf{Composite $-$ $t$-Centered ($\\times10^{-3}$)}", rows,
               "Decision loss at threshold $t$ (false positive costs $t$, false negative $1-t$) among retained patients "
               "at 80\\% coverage: composite versus a gate that defers the highest expected loss at $t$; counts of the "
               "12 pairs whose 95\\% paired-bootstrap CI excludes zero.", "\\footnotesize")
    hr = pd.read_csv(RES / "r1_extra_harmonized.csv")
    hk = hr[(hr.K == 10)]
    hd = pd.read_csv(RES / "r1_extra_harmonized_desc.csv")
    rows = []
    for (st, tk), g in hk.groupby(["setting", "task"], sort=False):
        g = g.set_index("score")
        d = hd[(hd.setting == st) & (hd.task == tk)].iloc[0]
        lab = "eICU" if st == "eICU-harm" else "MIMIC-IV$\\rightarrow$eICU"
        rows.append(f"{lab} & {tk.capitalize()} & {100 * d.mece:.1f}\\% & "
                    + " & ".join(f"{1e3 * g.loc[n, 'delta']:+.1f}" for n in ("D3p", "D5p", "Cp", "L", "LD"))
                    + f" & {int(((g.loc[list(PRIMARY_V3), 'delta'] > 0) & (g.loc[list(PRIMARY_V3), 'p_holm'] < .05)).sum())} \\\\")
    tc = table("S15c", "llccccccc", "\\textbf{Setting} & \\textbf{Task} & \\textbf{M-ECE} & D3$^{\\mathrm{m}}$ & "
               "D5$^{\\mathrm{m}}$ & C$^{\\mathrm{m}}$ & L & L+D & \\textbf{Holm Gain}", rows,
               "Harmonized features (observation counts and temperature removed from all model inputs; 84 features): "
               "risk-matched gain over the symmetric gate ($\\Delta_{\\mathrm{adj}}$, $\\times10^{-3}$, $K=10$) and the "
               "number of primary scores with a gain at Holm-adjusted $p<0.05$.", "\\footnotesize")
    so = pd.read_csv(RES / "r1_extra_sepsis_onset.csv")
    sk = so[(so.K == 10) & (so.family == "primary")]
    gt = pd.read_csv(RES / "r1_extra_sepsis_onset_gates.csv").iloc[0]
    rows = [f"{V3_LAB[r.score]} & {ci(r.delta, r.ci_lo, r.ci_hi, 1e3, 1)} & {r.p_holm:.2f} \\\\" for r in sk.itertuples()]
    td = table("S15d", "lcc", "\\textbf{Score} & \\textbf{$\\Delta_{\\mathrm{adj}}$ [95\\% CI]} & \\textbf{Holm $p$}", rows,
               f"MIMIC-IV sepsis with Sepsis-3 onset after 24~h (stays with onset at or before 24~h excluded; test "
               f"$n={int(gt.n):,}$, prevalence {100 * gt.prevalence:.1f}\\%): risk-matched gain ($\\times10^{{-3}}$). "
               f"Whole-cohort retained Brier at 80\\%: one-sided {gt.brier80_one_sided:.4f}, symmetric "
               f"{gt.brier80_symmetric:.4f}, composite {gt.brier80_composite:.4f} (symmetric $-$ composite "
               f"{gt.sym_minus_comp_diff:+.4f}, 95\\% CI {gt.sym_minus_comp_ci_lo:+.4f} to {gt.sym_minus_comp_ci_hi:+.4f}).",
               "\\footnotesize")
    return ("\\section*{Table S15: Additional Post Hoc Analyses}\n\nPer-hospital heterogeneity, threshold-centered "
            "decision loss, harmonized transfer, and sepsis with onset after the prediction time; all post hoc.\n\n"
            + "\n\n".join([ta, tb, tc, td]) + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    (OUT / "supp_generated_s1_s4.tex").write_text(
        "% GENERATED by deferral_frontier/experiments/build_supplement_r1.py -- do not edit by hand\n\n"
        + "\n".join([s1(), s2(), s3(), s4()]))
    (OUT / "supp_generated_s6_s10.tex").write_text(
        "% GENERATED by deferral_frontier/experiments/build_supplement_r1.py -- do not edit by hand\n\n"
        + "\n".join([s6(), s7(), s8(), s9(), s10()]))
    (OUT / "supp_generated_s12_s13.tex").write_text(
        "% GENERATED by deferral_frontier/experiments/build_supplement_r1.py -- do not edit by hand\n\n"
        + "\n".join([s12(), s13(), s14(), s15()]))
    logger.info("wrote supplement fragments and figures to %s", OUT)


if __name__ == "__main__":
    main()
