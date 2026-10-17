# Risk-stratified deferral: pre-specified analysis (fixed 2026-10-16, before any stratified result was seen)

**Author decision:** run this before any framing or co-author communication (`plan/r1-homework.md`).

**Already seen, and the reason for this test** (`plan/r1-v2-sanity-check.md`): at fixed coverage, retained Brier depends on case mix. The symmetric gate ≈ "defer the top 20% of risk." Unadjusted quartile contrasts suggested D3 carries some signal and D2 has a reversed sign. Those contrasts were exploratory and had no CIs. This analysis is the confirmatory test.

## Question

Among patients with the same calibrated predicted risk, does a reliability score select for deferral the patients whose predictions have larger squared error, more than random deferral at the same risk does?

## Data and models

- **Pairs:** 12 = 3 sources × 3 tasks, plus eICU held-out hospitals (eICU-H).
- **Models:** the current corrected pipelines (`train_pipeline`; patient split seed 42; `split_by="hospital"` for eICU-H). No change to models, calibration, or v2.
- **All quantities are on the test set.**
- **Notation:** p = calibrated probability (`cal_test`); y = label; e = (p − y)², the per-patient squared error.

## Risk strata (label-free)

- Sort test patients by p. Ties are broken by a fixed random permutation (seed 42).
- Cut into **K = 10 equal-size bins**.
- **Pre-specified sensitivity:** K = 20.

## Deferral rule (identical for every score)

- Each score s is a reliability score: higher means more reliable, which means retained.
- **Within each bin b** of size n_b, defer the round(0.2·n_b) patients with the lowest s. Ties in s are broken by the same fixed permutation.
- Overall coverage is therefore ≈ 80%, and every gate retains the same risk distribution.

## Null and effect

- **Null:** random deferral within bins. Its expected retained Brier is computed exactly as Σ_b k_b·ē_b / Σ_b k_b, where k_b = n_b − round(0.2·n_b) and ē_b is the mean e in bin b.
- **Effect:** Δ(s) = Brier_null − Brier_retained(s). Δ > 0 means the score defers more error than chance at the same risk.
- Also reported: Δ / Δ(oracle), the share of the achievable within-stratum gain.

## Scores evaluated

**Primary family** (7 scores; direction as stored, higher = more reliable):

| Score | Definition |
|---|---|
| D2′ | Fraction of signals observed |
| D3 | Cohort similarity |
| D4 | Temporal stability |
| D5′ | −MC-dropout variance |
| −ensemble SD | Spread of the 5-seed XGBoost ensemble |
| v2 | The pre-specified tree ErrorComposite (`r_test`) |
| ridge | Cross-fitted ridge error model on standardized [p(1−p), D2′, D3, D4, D5′] (as in `r1_v2_posthoc_linear.py`, α = 1); reliability = −predicted error |

**Reference scores** (not in the Holm family):

| Score | Role |
|---|---|
| Symmetric gate, max(p, 1−p) | Built-in check: within narrow bins it should be close to the null |
| Oracle, −e | Ceiling |
| Positive controls: −(e + ε), ε ~ N(0, k·SD(e)), k ∈ {1, 2, 4}, seed 42 | Detectable effect size |

## Inference

- **Paired bootstrap**, 1,000 resamples of test patients (seed 42). In each resample, bins and within-bin deferral are recomputed on the resampled set.
- **95% percentile CI for Δ.** Two-sided p = 2·min(P(Δ* ≤ 0), P(Δ* ≥ 0)), floored at 1/1000.
- **Holm correction** within each source family: 7 scores × 3 tasks = 21 tests per family, with 4 families (MIMIC-IV, MIMIC-III, eICU, eICU-H).

## Decision rules (fixed now)

1. **Reliability signal:** a score with Δ > 0 and Holm p < 0.05 for the same task in at least 2 of the 3 primary sources (MIMIC-IV, MIMIC-III, eICU) counts as a reproducible within-risk reliability signal for that task. eICU-H is reported as secondary support.
2. **Credible negative:** a pair where no primary score meets rule 1 but the k = 2 positive control has a CI excluding 0 supports "no detectable within-risk signal at that effect size".
3. **Harm:** Δ < 0 with Holm p < 0.05 is reported as a score that defers *less* error than chance (for example, a reversed direction).
4. **Bin check:** if the symmetric gate's Δ exceeds 25% of the oracle Δ at K = 10 in a pair, risk is not adequately held fixed for that pair. The K = 20 result becomes primary for that pair.

## Side panel: whole-cohort gates (descriptive plus one contrast)

- **Gates:** one-sided p, symmetric, defer-top-20%-p, v2, ridge, and random. Each is evaluated at 80% coverage on the whole cohort, without stratification.
- **Metrics:**
  - retained Brier;
  - retained prevalence π;
  - retained Brier skill BSS = 1 − Brier/[π(1−π)];
  - retained AUROC.
- **Pre-specified contrast:** BSS(symmetric) − BSS(one-sided), with a paired bootstrap 95% CI (1,000 resamples) per pair. It answers whether the submitted one-sided vs. symmetric conclusion survives a prevalence-adjusted metric.

## Commitments

- No change to bins, scores, coverage, or rules after results are seen. Any extra analysis is labeled post hoc.
- Every result is reported per pair, whatever its direction.
- Outputs:
  - `deferral_frontier/results/r1_stratified_deferral.csv` (with a K column);
  - `deferral_frontier/results/r1_whole_cohort_panel.csv`.
