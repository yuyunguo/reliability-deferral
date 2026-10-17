# The Deferral Frontier: Reliability-Gated Selective Prediction for Clinical AI

Code accompanying the manuscript submitted to *AI in Medicine* (MDPI). This
repository is private during peer review and will be made public on
acceptance.

## Contents

- `src/` — shared library: data loading (`src/data`), base model training
  (`src/models`), post-hoc calibration (`src/calibration`), and the five
  reliability dimensions plus composite (`src/reliability`).
- `deferral_frontier/src/` — deferral policies, fast reliability-dimension
  computation, shift experiments, significance tests, and the shared
  retraining pipeline used by the analyses below.
- `deferral_frontier/experiments/` — entry-point scripts for each analysis
  reported in the manuscript (main deferral evaluation, cross-institutional
  validation on MIMIC-III/eICU, decision-curve analysis, alternative
  uncertainty-quantification baselines, non-tree/SelectiveNet comparison,
  distribution-shift experiments, fairness audits).
- `conf/`, `deferral_frontier/conf/` — Hydra configuration.

## Data access

This repository contains **no patient data**. All experiments use MIMIC-IV,
MIMIC-III, and the eICU Collaborative Research Database, each of which
requires independent PhysioNet credentialing and a completed data use
agreement:

- MIMIC-IV: https://physionet.org/content/mimiciv/
- MIMIC-III: https://physionet.org/content/mimiciii/
- eICU-CRD: https://physionet.org/content/eicu-crd/

Feature extraction scripts (`deferral_frontier/experiments/extract_*.py`)
expect a PostgreSQL instance loaded with the corresponding database schema.
Connection parameters are read from environment variables (see
`.env.example`); no credentials are hardcoded.

## Setup

```bash
pip install -e .
cp .env.example .env   # fill in your own DB connection details
```

## Reproducing the main results

```bash
python -m deferral_frontier.experiments.run_deferral sepsis
python -m deferral_frontier.experiments.run_deferral mortality
python -m deferral_frontier.experiments.run_deferral vasopressor
```

Downstream analyses (conformal/alternative-baseline comparisons, decision
curves, shift experiments, external validation) are separate scripts under
`deferral_frontier/experiments/` and read the artifacts produced above.

## Citation

Citation details will be added on acceptance.

## Revision analyses (Bioengineering, round 1)

Data access: MIMIC-IV, MIMIC-III, and eICU-CRD require PhysioNet credentialing. No data
are included here. Put your own database connection settings in `.env` (see `.env.example`).

| Analysis | Entry point |
|---|---|
| Patient- and hospital-level splits | `src/data/splits.py`, `deferral_frontier/src/splits.py` |
| Cohort re-extraction (shared definitions) | `deferral_frontier/experiments/extract_eicu_r1.py`, `deferral_frontier/src/icu_features.py`, `deferral_frontier/src/icu_labels.py` |
| Reliability composite (cross-fitted error model) | `deferral_frontier/src/reliability_v2.py` |
| Whole-cohort gate comparisons | `deferral_frontier/experiments/run_symmetric_confidence_gate.py`, `run_paired_gate_tests.py` |
| Risk-matched deferral test | `deferral_frontier/src/stratified_deferral.py`, `experiments/r1_stratified_deferral.py`, `experiments/r1_v3.py` |
| Rebuilt and learned scores | `deferral_frontier/src/reliability_v3.py` |
| Data-sufficiency measure D2* | `deferral_frontier/src/imputation_uncertainty.py`, `experiments/r1_d2star_full.py` |
| Post hoc robustness analyses | `deferral_frontier/src/posthoc_tests.py`, `experiments/r1_v3_posthoc.py`, `experiments/r1_extra_ab.py`, `experiments/r1_extra_cd.py` |
| Supplementary tables | `deferral_frontier/experiments/build_supplement_r1.py` |

Dated analysis plans are in `analysis_plans/`. Run the tests with `pytest deferral_frontier/tests`.
