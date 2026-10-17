# Addendum 1 to `r1-d2-imputation-preregistration.md` (hash 447e076d…) — descriptive full run

**Fixed 2026-10-16, before the full-run results are opened.**

## Feasibility result (already opened; drives this addendum)

The MIMIC-IV feasibility test answered its own two questions:

1. **Does D2* discriminate?** Yes. Coefficient of variation 2.67–3.51 across tasks; the submitted
   D2 saturates at 99.9% of patients while D2* is zero for only 14–30% of patients.
2. **Does D2* carry within-risk error beyond confidence?** No. Δ vs. random is null in all three
   tasks; Δ_adj vs. the symmetric gate is ≤ 0 (−0.0026 to −0.0046), significant for mortality
   and vasopressor.

**Per the pre-registered escalation rule, no confirmatory claim is made for D2* in-distribution.**
D2* is not added to any Holm family and does not change any v3 confirmatory result.

## What this addendum runs

D2* is computed and reported **descriptively** as a reference score across the same 15 settings
and the same endpoint as `r1-reliability-v3-preregistration.md` (amendment 1):

- Settings: MIMIC-IV, MIMIC-III, eICU (in-distribution); eICU-H (held-out hospitals); X-DB
  (MIMIC-IV models applied to the eICU patient-split test set, nothing refit), × 3 tasks.
- Endpoint: risk-stratified deferral, K = 10 bins, 20% deferred per bin, primary effect
  `Δ_adj = Δ(score) − Δ(symmetric)`, paired bootstrap 1,000 (seed 42), two-sided p floored at 0.001.
- For X-DB the imputer is fit on MIMIC-IV training data only (the "nothing refit" principle),
  and the eICU features are reordered to MIMIC-IV column order and median-imputed with
  MIMIC-IV column medians before scoring, exactly as D3′ is handled in v3.

## Two descriptive purposes (no hypothesis tests)

1. **Generality of the discrimination result** beyond MIMIC-IV: is D2* unsaturated (and the
   submitted D2 saturated) in every setting?
2. **Shift question (descriptive, hypothesis-generating only):** does D2*'s within-risk signal
   `Δ_adj` vs. symmetric become positive under cross-database shift (X-DB), where calibration
   degrades and the paper's own decomposition predicts a sufficiency metric *could* help? A
   positive value here would be flagged as hypothesis-generating, not confirmatory.

## Reference scores reported alongside

`symmetric = max(p, 1-p)`; the submitted `D2` (observation ratio, `compute_d2_fast`); the v2/v3
count-fraction `D2′` (`sufficiency_fraction`); the oracle `-(p-y)^2`.

## Commitments

- Everything is reported descriptively, whatever its direction. No Holm family is formed.
- Outputs: `deferral_frontier/results/r1_d2star_full.csv` (gate comparison, one row per
  setting/task/score) and `deferral_frontier/results/r1_d2star_full_discrimination.csv`.
