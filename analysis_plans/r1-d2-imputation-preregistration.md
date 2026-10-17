# D2* — data sufficiency as imputation uncertainty: pre-specified feasibility test (2026-10-16)

**Purpose.** Address Reviewer 1, comment 4 ("the D2 data-sufficiency measure saturates in
approximately 80% of patients; a more discriminative and clinically meaningful sufficiency
metric should be developed") with a solution rather than a decline.

**Status at freeze.** This document is fixed *before* the D2* result is opened.

## Which D2 saturates, and what already changed

- **Submitted D2** (`deferral_frontier/src/fast_reliability.py: compute_d2_fast`): the
  fraction of signals observed in the 24 h window before prediction divided by the fraction
  observed in the first 6 h after admission, clipped to [0, 1] (Methods, Sec. 2.2). Because
  the 24 h look-back window contains the 6 h baseline window, this ratio is at its ceiling of
  1.0 for ~99.9% of patients under correct ICU anchoring (80% in the submitted, hospital-
  anchored cohort). This is the measure the reviewer flagged.
- **v2/v3 D2′** (`reliability_v2.py: sufficiency_fraction`): the fraction of `_count` features
  with at least one observation. This does *not* saturate (about 1% at 1.0) but is coarse
  (a 23-level histogram) and, as the v2 sanity check showed, is confounded with case mix
  (healthier patients have fewer labs).
- **D2* (this document)** replaces both with a continuous, value-of-information measure.

## Question

1. Does D2* discriminate patients (i.e., is it *not* saturated, unlike the submitted D2)?
2. Does D2* carry information about within-risk error beyond confidence (the symmetric gate)?

## Score definition (higher = more reliable)

D2* is the **imputation-driven prediction uncertainty**: the variance of the model's predicted
probability over stochastic completions of the patient's missing feature values.

- `D2*(x) = Var_b[ f_p(x^(b)) ]`, `b = 1..B`, `B = 50`.
- `x^(b)` is a completion of the patient's raw feature vector in which every missing entry is
  drawn from a conditional distribution estimated on the training set; observed entries are
  unchanged.
- `f_p` is the **primary model** (XGBoost ensemble seed 0, NaN-preserving training), the same
  model used for D5' in `r1-reliability-v3-preregistration.md`.
- The deferral/reliability score reported downstream is **`-D2*`** (higher = more reliable),
  matching the sign convention of every other reliability score.

### The conditional imputer (predictive mean matching, single iteration)

Per feature `j` with any missing value in training, fit a Ridge regressor (`alpha=1.0`,
standardized predictors, column `j` excluded) mapping the **median-imputed complete** feature
vector to the observed value of `j`, on training rows where `j` is observed. A missing value
at row `i` is then completed by **predictive mean matching**: draw from the `d = 5` observed
training values of `j` whose value is closest to the ridge prediction.

- PMM is the default in modern multiple imputation (van Buuren 2011) and is chosen over
  Gaussian residual noise because imputed values are always *real observed values*: completions
  never leave the feature's support (no negative laboratory values), and it is robust to
  non-normality and skew.
- Conditioning set: the full 115-feature median-imputed matrix (`art.X_train` / `art.X_test`),
  the same representation the pipeline already computes (`src/train_pipeline.py: _impute`).
- Only **value features** (the 92 non-`_count` columns) are imputed; `_count` columns are
  complete by construction (count = 0 is recorded, not missing) and carry the same signal.
- Column `j` is excluded from its own imputation model so the model cannot read back the
  observed value (which would collapse the donor spread to zero).

Interpretation: `D2*` is the value of information of the missing data — "how much would the
risk estimate move if we could measure the missing signals". It is high for a patient missing
features that (a) drive the prediction and (b) cannot be inferred from the other features.
This is exactly the residual-error term `(p − η(x))²` in the paper's decomposition
`E[(p−y)²|x] = p(1−p) + (p−η(x))²`.

**Complement to D5'.** D5' (NaN-mask variance) hides *observed* entries and measures
sensitivity to their removal; it is ~0 for a patient with almost nothing recorded. D2* fills
*missing* entries and measures uncertainty from what is already absent; it is *large* for
exactly that patient. They address different failure modes and are not interchangeable.

## Endpoint (unchanged from `r1-reliability-v3-preregistration.md` amendment 1)

- Risk-stratified deferral: `K = 10` equal-size bins of calibrated `p`, 20% deferred per bin.
- Primary effect `Delta_adj = Delta(score) - Delta(symmetric)`, the gain *beyond confidence*.
- `Delta = (random-within-bin retained Brier) - (retained Brier)`, paired bootstrap `1,000`
  resamples (seed 42), bins recomputed per resample, two-sided p floored at `0.001`.
- Reference scores reported alongside: `symmetric = max(p, 1-p)`, the submitted `D2` (the
  saturated observation ratio), the v2/v3 count-fraction `D2′`, and the oracle `-(p-y)^2`.

## Scope of this run ("small test")

- **MIMIC-IV only, all three tasks (sepsis, mortality, vasopressor), in-distribution.**
- **Exploratory.** This is a feasibility/discrimination test to answer R1-4. D2* is **not**
  added to any Holm family and does not change any confirmatory result in v3. If D2* (a)
  discriminates and (b) shows credible within-risk signal, a separate, full pre-registered
  confirmatory run (all settings + Holm family) will be written before any inference is drawn.

## Descriptive checks (reported whatever the outcome)

1. **Saturation contrast:** fraction of test patients with the submitted D2 at its ceiling
   vs. fraction with `D2* == 0`, the coefficient of variation, and quantiles of `D2*`.
2. **Redundancy with confidence:** Spearman correlation of `D2*` with `p` and with `p(1-p)`.
3. **Gate comparison:** `Delta` vs. random and `Delta_adj` vs. symmetric, with 95% CI.

## Commitments

- No changes after results are seen; anything extra is labeled post hoc.
- Every task is reported whatever its direction.
- Outputs: `deferral_frontier/results/r1_d2star.csv` (gate comparison) and
  `deferral_frontier/results/r1_d2star_discrimination.csv` (saturation/redundancy summary).

## Known limitations (documented now, before results)

- Single-iteration PMM (no chained feedback between features); multi-iteration MICE would be
  more faithful but is out of scope for a feasibility test.
- Homoscedastic donor spread within the `d` nearest neighbours; no per-patient heteroscedasticity.
- `B = 50` completions gives a variance estimate with ~20% relative standard error, sufficient
  for a *ranking* (deferral) metric, matching D5'.
- Uses the seed-0 model (not the 5-seed ensemble), matching D5'; a cross-model average is a
  possible refinement.
