# Corrected dimensions and learned reliability: pre-specified analysis (fixed 2026-10-16)

**Status when fixed:**
- The previous pre-registered stratified run (`r1-stratified-deferral-preregistration.md`, hash 84f06205…) had finished, but **its results had not been opened**.
- Already seen, all exploratory and MIMIC-IV only (`r1-v2-sanity-check.md` §5):
  - the dimension construction bugs;
  - a corrected D5 (Δ ≈ 0);
  - a full-x error model with two hyperparameter settings (Δ +0.002 to +0.004).
- **Because the learned-score hyperparameters were chosen after seeing MIMIC-IV, MIMIC-IV is exploratory for the learned scores.** MIMIC-III, eICU, eICU-H, and the cross-database shift are confirmatory.

## Question

1. Is there within-risk error signal in the patient data that a reliability score can capture?
2. Do correctly built reliability dimensions capture it?
3. Do the dimensions add anything beyond the raw data?
4. Is the signal larger under distribution shift?

## Endpoint (unchanged from the previous pre-registration)

- Risk-stratified deferral: 10 equal-size bins of calibrated p (label-free), with ties broken by a fixed permutation (seed 42).
- Defer round(0.2·n_b) of the lowest-scoring patients within each bin.
- Effect Δ = (exact random-within-bin retained Brier) − (retained Brier).
- Paired bootstrap, 1,000 resamples (seed 42), with bins recomputed on each resample.
- Two-sided p = 2·min(P(Δ* ≤ 0), P(Δ* ≥ 0)), floored at 0.001.
- K = 20 is the sensitivity analysis. The same bin-check rule applies: if the symmetric gate's Δ exceeds 25% of the oracle Δ, K = 20 becomes primary for that pair.
- Implementation: `deferral_frontier/src/stratified_deferral.py` (unchanged).

## Settings

| Setting | Train / validation | Test | Status |
|---|---|---|---|
| MIMIC-IV | MIMIC-IV | MIMIC-IV | In-distribution; exploratory for L, L+D |
| MIMIC-III | MIMIC-III | MIMIC-III | In-distribution; confirmatory |
| eICU | eICU | eICU | In-distribution; confirmatory |
| eICU-H | eICU (training hospitals) | eICU held-out hospitals | Site shift; confirmatory |
| **X-DB** | MIMIC-IV (models, calibrator, scaler, D3′, error models; nothing refit) | **eICU patient-split test set** (the same patients as the eICU row) | Cross-database shift; confirmatory |

- The models are the current corrected pipelines (patient split seed 42).
- The 115 features are identical by name across databases. eICU columns are reordered to MIMIC-IV order.
- In X-DB, the label definitions differ between databases (sepsis especially), so the shift includes concept shift. This is reported as such.

## Scores (higher = more reliable)

**Corrected dimensions**
- **D3′ (cohort similarity, missingness-aware):**
  - inputs: z-scored, train-median-imputed features, plus a binary missingness indicator for every feature with any missing value in training;
  - Ledoit–Wolf Mahalanobis distance fit on training only;
  - D3′ = exp(−d / median training d).
- **D5′ (input-perturbation sensitivity):**
  - the primary model (ensemble seed 0) on **raw** X, with missing values kept as NaN;
  - in each of 50 passes, a random 10% of the *observed* entries is set to NaN (seed 42);
  - D5′ = −variance of the predicted probability.
- **D2 and D4:** unchanged, since they are correct as defined. **−ensemble SD:** the 5-seed XGBoost spread, also computed for validation.

**Composites.** All are cross-fitted on two patient-disjoint validation halves, exactly as in v2. Isotonic calibration is fit on one half, and the target (p̃ − y)² is computed on the other.

| Score | Model | Inputs |
|---|---|---|
| **C′**, corrected-dimension composite | Ridge (α = 1, standardized) | [p(1−p), D2, D3′, D4, D5′, −ens SD] |
| **L**, learned reliability | XGBRegressor (squared error, max_depth 2, 200 trees, learning rate 0.03, subsample 0.8, colsample_bytree 0.5, min_child_weight 50, random_state 42) | [p(1−p), raw X with NaN] |
| **L+D** | The same XGBRegressor as L | [p(1−p), raw X, D2, D3′, D4, D5′, −ens SD] |

- For every composite, reliability = −predicted error.
- In X-DB, all composites are those fit on MIMIC-IV validation data.

**Reference scores (outside the Holm family):** symmetric max(p, 1−p); oracle −(p−y)²; positive controls −(e + N(0, k·SD(e))) for k ∈ {1, 2, 4}; and the as-built D3 and D5 for continuity.

## Hypothesis families and Holm correction

- **Primary family per setting:** {D3′, D5′, C′, L, L+D} × 3 tasks = 15 tests, Holm within each setting.
- **Increment family per setting:** Δ(L+D) − Δ(L), paired bootstrap on the same resamples, × 3 tasks = 3 tests, Holm.
- **Shift contrast (X-DB vs. eICU, same test patients):** Δ_L(X-DB) − Δ_L(eICU), using a paired bootstrap over shared patient resamples, × 3 tasks, Holm.

## Decision rules (fixed now)

- **R1, learned within-risk reliability exists (in-distribution).** L has Δ > 0 with Holm p < 0.05 in **both** MIMIC-III and eICU for a task. MIMIC-IV is reported but cannot satisfy R1.
- **R2, reliability under shift.**
  - L has Δ > 0 with Holm p < 0.05 in X-DB, and separately in eICU-H, for a task.
  - "Larger under shift" requires the shift contrast to have Δ > 0 with Holm p < 0.05.
- **R3, corrected dimensions work.** D3′, D5′, or C′ has Δ > 0 with Holm p < 0.05 for the same task in ≥ 2 of {MIMIC-IV, MIMIC-III, eICU}, or in X-DB.
- **R4, the dimensions add beyond the raw data.** The increment has Δ > 0 with Holm p < 0.05 in ≥ 2 settings for a task.
- **R5, credible negative.** No score meets the rules for a pair, but the k = 2 positive control has a CI excluding 0. Then the pair supports "no within-risk signal at that effect size".
- Harms (Δ < 0, Holm p < 0.05) are reported as such.

## Descriptive (no tests)

- Expected calibration error of p in each setting (to document the size of the shift).
- Δ / Δ(oracle) for every score.
- Unstratified retained Brier at 80% coverage for L, C′, and the symmetric gate.

## Commitments

- No changes after results are seen. Anything extra is labeled post hoc.
- Every pair is reported, whatever its direction.
- Outputs:
  - `deferral_frontier/results/r1_v3_stratified.csv`;
  - `r1_v3_increment.csv`;
  - `r1_v3_shift_contrast.csv`;
  - `r1_v3_descriptive.csv`.
