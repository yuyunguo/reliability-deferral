# Composite v2: pre-specified analysis (fixed 2026-10-16, before any v2 result was seen)

**Author decisions:** replace the original composite with v2 everywhere. Primary criterion: a Holm-significant gain in retained Brier over the symmetric gate.

## Question

Do the non-confidence reliability dimensions predict a patient's squared error beyond p̃(1 − p̃)? That is, do they capture the term (p̃ − η(x))², and does deferring on that prediction lower retained Brier relative to the symmetric gate?

## Dimensions (computed per patient at prediction time)

| | Definition | Change from the original composite |
|---|---|---|
| D1′ confidence | p̃(1 − p̃), using the calibrated probability | Was raw p̃ |
| D2′ sufficiency | Fraction of the model's raw signals with ≥1 observation in the 24 h window (count features > 0) | Was a ratio that is constant under ICU anchoring |
| D3 similarity | Unchanged: Mahalanobis similarity to training (Ledoit–Wolf), fit on training data only | — |
| D4 stability | Unchanged: within-window variability measure | — |
| D5′ uncertainty | Raw variance of 50 MC-dropout passes of the primary model | No longer divided by p̃(1 − p̃) |

## Fitting (no test data used)

1. Split the validation set into two halves **by patient** (seed 42).
2. For each half *h*:
   - fit isotonic calibration on the other half's raw scores;
   - compute p̃_h, and D1′ and D5′ from p̃_h;
   - compute the target (p̃_h − y_h)².
3. Pool the two cross-fitted halves and fit one regressor:
   - XGBRegressor, objective squared error;
   - max_depth 3, n_estimators 300, learning_rate 0.05, subsample 0.8, random_state 42;
   - **monotone increasing in D1′**, unconstrained in D2′–D5′.
4. **Test time:**
   - p̃ is the pipeline's existing isotonic calibration, fit on the full validation set (unchanged);
   - reliability = −(predicted squared error).
5. **Hyperparameters are fixed here and are not tuned.**

## Endpoints

- **Primary:** retained Brier at 80% coverage, v2 vs. the symmetric gate, in the 9 primary pairs.
  - Paired bootstrap, 1,000 resamples.
  - Holm across the 9 pairs.
  - "Dimensions help" in a pair = v2 better with Holm-adjusted p < 0.05. Results are reported per pair, whatever their direction.
- **Secondary:**
  - eICU held-out hospitals (separate Holm family);
  - shift experiments (missingness, correlated lab shift);
  - MLP base model;
  - split-seed sensitivity (5 splits).
- **Descriptive:** held-out improvement in predicting squared error over D1′ alone; feature importance; drop-one-dimension ablation.

## Commitments

- No changes to the v2 specification after results are seen. Any later change is reported as post hoc.
- The original composite's result (≈ "defer the highest risk") goes in the Discussion and the response letter as the reason for the redesign.
