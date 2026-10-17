"""Collect every number cited in the R1 Methods/Results draft, with its source file.

Writes plan/r1-draft-facts.md. Reads only saved results/artifacts (no retraining),
except the cohort section, which reads labels and patient/hospital ids.

Usage:
    python -m deferral_frontier.experiments.r1_draft_facts
"""
import io
import logging
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from deferral_frontier.experiments.run_symmetric_confidence_gate import retained_mask
from deferral_frontier.src.splits import load_patient_split
from src.calibration.metrics import adaptive_ece

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "deferral_frontier" / "results"
ART = ROOT / "artifacts"
TASKS = ("sepsis", "mortality", "vasopressor")
DBS = {"MIMIC-IV": "mimic4", "MIMIC-III": "mimic3", "eICU": "eicu"}
ARTS = {"MIMIC-IV": "{t}_test_artifacts.parquet", "MIMIC-III": "mimic3/mimic3_{t}_test_artifacts.parquet",
        "eICU": "eicu/eicu_{t}_test_artifacts.parquet",
        "eICU-H": "eicu_hospital/eicu_hospital_{t}_test_artifacts.parquet"}
ORDER = ["MIMIC-IV", "MIMIC-III", "eICU", "eICU-H"]


def r(x, k=3):
    return np.round(x, k)


def cohorts() -> None:
    print("## F1 Cohorts (data/{db}/*_labels.parquet, patient_ids.parquet, load_patient_split seed 42)\n")
    for name, db in DBS.items():
        d = ROOT / "data" / db
        pid = pd.read_parquet(d / "patient_ids.parquet")["patient_id"]
        for t in TASKS:
            y = pd.read_parquet(d / f"{t}_labels.parquet").iloc[:, 0]
            tr, va, te = load_patient_split(d, y, seed=42)
            line = (f"- {name} {t}: N={len(y):,} rows, patients={pid.reindex(y.index).nunique():,}, "
                    f"prev={y.mean():.3f}; split {len(tr):,}/{len(va):,}/{len(te):,}")
            if db == "eicu":
                hid = pd.read_parquet(d / "hospital_ids.parquet")["hospital_id"].reindex(y.index)
                line += f"; hospitals={hid.nunique()}"
                htr, hva, hte = load_patient_split(d, y, seed=42, ids_file="hospital_ids.parquet",
                                                   column="hospital_id")
                line += (f"; eICU-H split {len(htr):,}/{len(hva):,}/{len(hte):,} "
                         f"(test hospitals={hid.loc[hte].nunique()})")
            print(line)
    print()


def test_sets() -> None:
    print("## F2 Test sets (artifacts: cal_proba, y_true)\n")
    print("| Setting | Task | n | prev | AUROC | Brier | M-ECE |\n|---|---|---|---|---|---|---|")
    for s in ORDER:
        for t in TASKS:
            a = pd.read_parquet(ART / ARTS[s].format(t=t))
            p, y = a.cal_proba.to_numpy(), a.y_true.to_numpy().astype(int)
            print(f"| {s} | {t} | {len(y):,} | {y.mean():.3f} | {roc_auc_score(y, p):.3f} | "
                  f"{np.mean((p - y) ** 2):.4f} | {adaptive_ece(p, y):.4f} |")
    print()


def main_table() -> None:
    print("## F3 Gates at 80% coverage (symmetric_confidence_gate.csv)\n")
    g = pd.read_csv(RES / "symmetric_confidence_gate.csv")
    cols = ["source", "task", "gate", "brier", "auroc", "retained_prevalence", "deferred_prevalence", "nnr",
            "jaccard_deferred_vs_reliability", "jaccard_deferred_vs_one_sided"]
    print(g[cols].round(4).to_string(index=False))
    piv = g.pivot_table(index=["source", "task"], columns="gate", values="brier")
    ratio = piv.one_sided_confidence / piv.reliability
    print(f"\nratio one-sided/composite: {ratio.min():.2f}–{ratio.max():.2f}")
    print(ratio.round(2).to_string())
    js = g[g.gate == "symmetric_confidence"].jaccard_deferred_vs_reliability
    jo = g[g.gate == "one_sided_confidence"].jaccard_deferred_vs_reliability
    jso = g[g.gate == "symmetric_confidence"].jaccard_deferred_vs_one_sided
    print(f"Jaccard sym vs comp: median {js.median():.2f}, range {js.min():.2f}–{js.max():.2f}")
    print(f"Jaccard one-sided vs comp: max {jo.max():.4f}; sym vs one-sided: max {jso.max():.4f}")
    for gate in ("one_sided_confidence", "symmetric_confidence", "reliability"):
        n = g[g.gate == gate].nnr
        print(f"NNR {gate}: {n.min():.1f}–{n.max():.1f}")
    print()
    print("### Paired tests (paired_gate_tests.csv; Holm over 12 pairs per comparison/metric)\n")
    pt = pd.read_csv(RES / "paired_gate_tests.csv")
    pt["brier_sig"] = pt.brier_p_holm < 0.05
    pt["auroc_sig"] = pt.auroc_p_holm < 0.05
    print(pt[["source", "task", "gate_a", "brier_diff", "brier_ci_lo", "brier_ci_hi", "brier_p_holm",
              "auroc_diff", "auroc_ci_lo", "auroc_ci_hi", "auroc_p_holm"]].round(4).to_string(index=False))
    for ga, sub in pt.groupby("gate_a"):
        print(f"{ga} vs composite: Brier Holm<.05 in {int(sub.brier_sig.sum())}/{len(sub)} "
              f"(diff<0 i.e. gate_a better: {int(((sub.brier_diff < 0) & sub.brier_sig).sum())}); "
              f"AUROC Holm<.05 in {int(sub.auroc_sig.sum())}/{len(sub)} "
              f"(gate_a higher: {int(((sub.auroc_diff > 0) & sub.auroc_sig).sum())}, "
              f"lower: {int(((sub.auroc_diff < 0) & sub.auroc_sig).sum())})")
    print("min attainable Holm p (from file):", r(pt.brier_p_holm.min(), 4), "\n")


def geometry() -> None:
    print("## F4 Geometry, MIMIC-IV (artifacts; 20% deferral)\n")
    for t in TASKS:
        a = pd.read_parquet(ART / ARTS["MIMIC-IV"].format(t=t))
        p, y, rs = a.cal_proba.to_numpy(), a.y_true.to_numpy(), a.r_score.to_numpy()
        for gate, s in (("one_sided", p), ("symmetric", np.maximum(p, 1 - p)), ("composite", rs)):
            d = ~retained_mask(s, 0.8)
            print(f"- {t} {gate}: deferred median p {np.median(p[d]):.3f}, deferred prev {y[d].mean():.3f}")
        e = (p - y) ** 2
        for cov in (1.0, 0.5):
            k = retained_mask(p, cov)
            print(f"  one-sided @cov {cov}: Brier {e[k].mean():.3f}, prev {y[k].mean():.3f}")
        print(f"  Spearman(composite R, p(1-p)) = {spearmanr(rs, p * (1 - p))[0]:.2f}; "
              f"Spearman(p, p(1-p)) = {spearmanr(p, p * (1 - p))[0]:.2f}")
    print()
    print("Spearman(R, p(1-p)) all settings:")
    for s in ORDER:
        vals = []
        for t in TASKS:
            a = pd.read_parquet(ART / ARTS[s].format(t=t))
            vals.append(spearmanr(a.r_score, a.cal_proba * (1 - a.cal_proba))[0])
        print(f"- {s}: " + ", ".join(f"{v:.2f}" for v in vals))
    print()


def composite_learns() -> None:
    print("## F5 Composite v2 importance and ablation\n")
    for s, db in (("MIMIC-IV", "mimic4"), ("MIMIC-III", "mimic3"), ("eICU", "eicu"), ("eICU-H", "eicu_hospital")):
        for t in TASKS:
            imp = pd.read_csv(RES / f"composite_v2_importance_{db}_{t}.csv", index_col=0).iloc[:, 0]
            share = imp / imp.sum()
            print(f"- {s} {t}: gain share " + ", ".join(f"{k.split('_')[0]} {v:.2f}" for k, v in share.items()))
    ab = pd.read_csv(RES / "composite_v2_ablation.csv")
    print()
    print(ab.pivot_table(index=["source", "task"], columns="variant", values="brier_retained").round(4).to_string())
    print()


def alternatives() -> None:
    print("## F6 Alternatives (alternative_uq_baselines.csv, neural_baselines.csv), MIMIC-IV\n")
    a = pd.read_csv(RES / "alternative_uq_baselines.csv")
    print(a.pivot_table(index="policy", columns="task", values="brier").round(4).to_string())
    n = pd.read_csv(RES / "neural_baselines.csv")
    print(n.pivot_table(index="policy", columns="task", values="brier").round(4).to_string())
    print(n.drop_duplicates("task")[["task", "brier_sym_minus_rel_ci_lo", "brier_sym_minus_rel_ci_hi"]]
          .round(4).to_string(index=False))
    print(n[n.policy == "symmetric_mlp"][["task", "jaccard_deferred_vs_reliability"]].round(2).to_string(index=False))
    print()


def net_benefit() -> None:
    print("## F7 Net benefit at p_t = 0.10 (symmetric_gate_net_benefit.csv), MIMIC-IV\n")
    nb = pd.read_csv(RES / "symmetric_gate_net_benefit.csv")
    x = nb[(nb.source == "MIMIC-IV") & (nb.threshold.round(2) == 0.10)]
    print(x.pivot_table(index="gate", columns="task", values="net_benefit").round(3).to_string())
    print()


def shift_stress() -> None:
    print("## F8 Test-time perturbation (shift_experiments_real_summary.csv), MIMIC-IV\n")
    s = pd.read_csv(RES / "shift_experiments_real_summary.csv")
    print(s.pivot_table(index=["experiment", "magnitude"], columns=["task", "policy"], values="brier_mean")
          .round(3).to_string())
    print()


def fairness() -> None:
    print("## F9 Fairness (fairness_r1.csv): max-min deferral-rate spread (pp) by gate\n")
    f = pd.read_csv(RES / "fairness_r1.csv")
    sp = f.groupby(["source", "task", "gate", "stratification"]).deferral_rate.agg(lambda v: (v.max() - v.min()) * 100)
    print(sp.unstack("stratification").round(1).to_string())
    print()


def risk_matched() -> None:
    print("## F10 Risk-matched (r1_stratified_deferral.csv: as-built; r1_v3_stratified.csv; r1_d2star_full.csv)\n")
    s1 = pd.read_csv(RES / "r1_stratified_deferral.csv")
    k = s1[s1.K == 10]
    prim = k[k.family == "primary"]
    print("As-built (vs random, K=10): Holm<.05 & delta>0 per score:")
    print(prim.assign(pos=(prim.p_holm < .05) & (prim.delta > 0), neg=(prim.p_holm < .05) & (prim.delta < 0))
          .groupby("score")[["pos", "neg"]].sum().to_string())
    sym = k[k.score == "symmetric"]
    print(f"symmetric share of oracle (K=10): {sym.share_of_oracle.min():.3f}–{sym.share_of_oracle.max():.3f}")
    v3 = pd.read_csv(RES / "r1_v3_stratified.csv")
    v = v3[(v3.K == v3.primary_K)]
    f = v[v.family == "primary"]
    print(f"v3 primary: {len(f)} tests; Δ_adj>0 & Holm<.05: {int(((f.delta > 0) & (f.p_holm < .05)).sum())}; "
          f"Δ_adj<0 & Holm<.05: {int(((f.delta < 0) & (f.p_holm < .05)).sum())}; point>0: {int((f.delta > 0).sum())}")
    print(f"v3 Δ_adj range (x1e-3): {f.delta.min() * 1e3:.1f} to {f.delta.max() * 1e3:.1f}")
    pc = v[v.score == "positive_control_k2"]
    print(f"positive control k2 Δ_adj (x1e-3): {pc.delta.min() * 1e3:.1f}–{pc.delta.max() * 1e3:.1f}; CI>0 in "
          f"{int((pc.ci_lo > 0).sum())}/{len(pc)}")
    for kk in (1, 4):
        q = v[v.score == f"positive_control_k{kk}"]
        print(f"positive control k{kk} Δ_adj CI>0 in {int((q.ci_lo > 0).sum())}/{len(q)}; "
              f"min {q.delta.min() * 1e3:.1f}")
    o = v[v.score == "oracle"]
    print(f"oracle Δ vs random (x1e-3): {o.delta_vs_random.min() * 1e3:.1f}–{o.delta_vs_random.max() * 1e3:.1f}")
    sy = v[v.score == "symmetric"]
    print(f"symmetric Δ vs random (x1e-3): {sy.delta_vs_random.min() * 1e3:.1f}–{sy.delta_vs_random.max() * 1e3:.1f}")
    for sc in ("L", "LD", "Cp", "D3p", "D5p"):
        q = f[f.score == sc]
        print(f"{sc}: Δ_adj (x1e-3) {q.delta.min() * 1e3:.1f} to {q.delta.max() * 1e3:.1f}; "
              f"CIs spanning 0: {int(((q.ci_lo <= 0) & (q.ci_hi >= 0)).sum())}/{len(q)}")
    inc = pd.read_csv(RES / "r1_v3_increment.csv")
    print(f"increment L+D − L: min Holm p {inc.p_holm.min():.2f}; Δ range (x1e-3) "
          f"{inc.delta.min() * 1e3:.1f} to {inc.delta.max() * 1e3:.1f}")
    sh = pd.read_csv(RES / "r1_v3_shift_contrast.csv")
    print("shift contrast:\n" + sh.assign(**{c: sh[c] * 1e3 for c in ("delta", "ci_lo", "ci_hi")}).round(2)
          .to_string(index=False))
    d2 = pd.read_csv(RES / "r1_d2star_full.csv")
    d2 = d2[d2.score == "D2star"]
    print(f"D2*: Δ_adj CI>0 in {int((d2.ci_lo_adj > 0).sum())}/15; point>0 {int((d2.delta_adj > 0).sum())}; "
          f"range (x1e-3) {d2.delta_adj.min() * 1e3:.1f} to {d2.delta_adj.max() * 1e3:.1f}")
    print()


def shift_v3() -> None:
    print("## F11 v3 descriptive (r1_v3_descriptive.csv)\n")
    d = pd.read_csv(RES / "r1_v3_descriptive.csv")
    print(d.round(4).to_string(index=False))
    print()


def bss() -> None:
    print("## F12 Whole-cohort BSS (r1_whole_cohort_panel.csv)\n")
    p = pd.read_csv(RES / "r1_whole_cohort_panel.csv")
    x = p.drop_duplicates(["source", "task"])
    neg = int((x.ci_hi < 0).sum())
    pos = int((x.ci_lo > 0).sum())
    print(f"BSS(sym) − BSS(one-sided): CI<0 in {neg}/{len(x)}; CI>0 in {pos}/{len(x)}; "
          f"range {x.bss_sym_minus_one_sided.min():.3f} to {x.bss_sym_minus_one_sided.max():.3f}")
    print(p.pivot_table(index=["source", "task"], columns="gate", values="bss").round(3).to_string())
    print()


def secondary() -> None:
    print("## F13 Secondary (Sepsis-3, SOFA/APACHE, class-conditional, imperfect review, old-split overlap)\n")
    for f in ("sepsis3_sensitivity", "clinical_score_baseline", "r1_old_split_patient_overlap"):
        print(f"### {f}\n" + pd.read_csv(RES / f"{f}.csv").round(4).to_string(index=False) + "\n")
    c = pd.read_csv(RES / "class_conditional_gates.csv")
    print(c[c.source == "MIMIC-IV"].round(3).to_string(index=False))
    ir = pd.read_csv(RES / "imperfect_review.csv")
    print(ir[(ir.source == "MIMIC-IV")].drop_duplicates(["task", "gate"])[["task", "gate", "n_deferred",
          "model_errors_deferred", "break_even_accuracy"]].round(3).to_string(index=False))
    print()


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    buf = io.StringIO()
    with redirect_stdout(buf):
        print("# R1 draft facts (generated by deferral_frontier/experiments/r1_draft_facts.py)\n")
        for fn in (cohorts, test_sets, main_table, geometry, composite_learns, alternatives, net_benefit,
                   shift_stress, fairness, risk_matched, shift_v3, bss, secondary):
            fn()
    (ROOT / "plan" / "r1-draft-facts.md").write_text(buf.getvalue())


if __name__ == "__main__":
    main()
