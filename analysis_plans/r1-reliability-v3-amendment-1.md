# Amendment 1 to `r1-reliability-v3-preregistration.md` (hash 5fffa60c…), fixed 2026-10-16 before any v3 run

## Why

After v3 was frozen, two things were seen:
- the results of the stratified-deferral pre-registration (hash 84f06205…);
- an exploratory paired check: full-x model L, ridge, and v2 vs. the symmetric gate, within bins, in MIMIC-IV, MIMIC-III, and eICU.

They showed that **within K = 10 or 20 risk bins, the symmetric gate itself gains Δ ≈ +2 to +5 × 10⁻³ over random deferral.** Residual within-bin variation in p(1−p) explains gains of that size. Composites that include p(1−p) gained about the same, and paired against the symmetric gate none was better (0/18 comparisons with CI above 0).

**A null of "random within bin" therefore credits confidence to the reliability scores.**

## Changes

1. **Primary effect** for every score is now Δ_adj = Δ(score) − Δ(symmetric), the gain *beyond confidence* at matched risk.
   - Same paired bootstrap: 1,000 resamples, seed 42, bins recomputed, symmetric and score evaluated on the same resample.
   - Δ against random is still reported, as a descriptive measure.
2. **Confirmatory status, revised:**
   - **L and L+D are exploratory in all in-distribution settings** (MIMIC-IV, MIMIC-III, eICU), because they were seen in the paired check.
   - They remain **confirmatory in eICU-H and X-DB**, which have not been seen.
   - D3′, D5′, and C′ remain confirmatory everywhere except D5′ in MIMIC-IV (exploratory).
3. **Decision rules** are unchanged in form but use Δ_adj in place of Δ:
   - R1 becomes descriptive (exploratory).
   - **R2 (L under shift: eICU-H and X-DB) is the primary confirmatory question.**
   - R3 and R4 as written.
   - For R5, the positive control is also evaluated as Δ_adj.
4. **Shift contrast:** Δ_adj,L(X-DB) − Δ_adj,L(eICU), on the same test patients.

Everything else in v3 is unchanged.
