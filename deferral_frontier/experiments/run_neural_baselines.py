"""Non-tree base model validation and SelectiveNet comparison.

Motivation: "The paper claims the
framework is 'model-agnostic' but validates only XGBoost. At minimum, one
non-tree-based model should be tested" and "Other selective prediction
frameworks (SelectiveNet, learned rejection)."

Runs on all three MIMIC-IV tasks (sepsis, mortality, vasopressor) --
originally scoped to sepsis only; extended because limiting this to one task weakens the comparison, particularly
for SelectiveNet as the only end-to-end alternative framework tested.

Part A: retrains the base classifier as a 5-seed MLP ensemble (PyTorch),
recomputes D1 (isotonic on ensemble mean), D5 (true MC-dropout via active
dropout layers at inference -- more faithful than XGBoost's input-dropout
approximation), and reuses the data-only D2/D3/D4 dimensions, to test
whether the reliability-gate-over-confidence-gate advantage replicates
with a non-tree base model.

Part B: trains a SelectiveNet (Geifman & El-Yaniv, ICML 2019) on the same
features -- a jointly-trained prediction+selection head that is itself an
alternative selective-prediction framework -- and compares its learned
rejection at matched coverage to the confidence and reliability gates.
"""
import logging
import os
from pathlib import Path

# Single OpenMP thread: torch and xgboost load different OpenMP runtimes on macOS and
# segfault (exit 139) during MLP training otherwise. Must be set before either import.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import xgboost  # noqa: F401  -- must be imported before torch: OpenMP runtime clash segfaults on macOS
import torch
import torch.nn as nn
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.calibration.baselines import IsotonicCalibrator
from src.data.loader import MIMICLoader
from src.reliability.cohort_similarity import CohortSimilarityEstimator
from src.reliability.composite import DimensionMatrix, ReliabilityComposite

from deferral_frontier.src.fast_reliability import compute_d2_fast, compute_d4_fast
from deferral_frontier.src.splits import load_patient_split
from deferral_frontier.src.reliability_v2 import ErrorComposite, patient_groups, sufficiency_fraction
from deferral_frontier.experiments.run_symmetric_confidence_gate import (
    jaccard, paired_brier_diff_ci, retained_mask,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("deferral_frontier/results")
DATA_DIR = Path("data/mimic4")
TASKS = ["sepsis", "mortality", "vasopressor"]
SEEDS = [0, 1, 2, 3, 4]
COVERAGE = 0.80
N_DROPOUT_PASSES = 50
DEVICE = "cpu"


class MLPClassifier(nn.Module):
    def __init__(self, n_features: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SelectiveNet(nn.Module):
    """Geifman & El-Yaniv (2019): joint prediction (f) + selection (g) heads."""

    def __init__(self, n_features: int, dropout: float = 0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.f_head = nn.Linear(64, 1)
        self.g_head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x):
        h = self.trunk(x)
        return self.f_head(h).squeeze(-1), self.g_head(h).squeeze(-1)


def _load_splits(task: str):
    loader = MIMICLoader(DATA_DIR)
    dataset = loader.load(task)
    # Patient-level split, identical to the XGBoost pipeline (round-1 R1-1/R2-1).
    idx_train, idx_val, idx_test = load_patient_split(DATA_DIR, dataset.labels)

    X_train = dataset.features.loc[idx_train].values.copy()
    X_val = dataset.features.loc[idx_val].values.copy()
    X_test = dataset.features.loc[idx_test].values.copy()
    y_train = dataset.labels.loc[idx_train].values.astype(np.float32)
    y_val = dataset.labels.loc[idx_val].values.astype(np.float32)
    y_test = dataset.labels.loc[idx_test].values.astype(np.float32)

    col_medians = dataset.features.loc[idx_train].median()
    for i in range(X_train.shape[1]):
        med = col_medians.iloc[i]
        med = 0.0 if pd.isna(med) else med
        X_train[:, i] = np.where(np.isnan(X_train[:, i]), med, X_train[:, i])
        X_val[:, i] = np.where(np.isnan(X_val[:, i]), med, X_val[:, i])
        X_test[:, i] = np.where(np.isnan(X_test[:, i]), med, X_test[:, i])

    scaler = StandardScaler().fit(X_train)
    X_train_sc = scaler.transform(X_train).astype(np.float32)
    X_val_sc = scaler.transform(X_val).astype(np.float32)
    X_test_sc = scaler.transform(X_test).astype(np.float32)

    return (dataset, idx_train, idx_val, idx_test,
            X_train_sc, X_val_sc, X_test_sc, y_train, y_val, y_test)


def _train_mlp(X_train, y_train, X_val, y_val, seed, n_epochs=60, patience=8):
    torch.manual_seed(seed)
    model = MLPClassifier(X_train.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    Xt, yt = torch.tensor(X_train), torch.tensor(y_train)
    Xv, yv = torch.tensor(X_val), torch.tensor(y_val)

    best_val, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 512):
            idx = perm[i:i + 512]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()
        if val_loss < best_val - 1e-4:
            best_val, best_state, bad_epochs = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    model.load_state_dict(best_state)
    return model


def _mlp_predict(models, X, train_mode=False):
    X_t = torch.tensor(X)
    preds = []
    for m in models:
        m.train(train_mode)
        with torch.no_grad():
            preds.append(torch.sigmoid(m(X_t)).numpy())
    return np.stack(preds, axis=0)


def part_a_reliability_on_mlp(task, dataset, idx_train, idx_val, idx_test,
                               X_train_sc, X_val_sc, X_test_sc, y_train, y_val, y_test) -> pd.DataFrame:
    logger.info("[Part A] [%s] Training 5-seed MLP ensemble (non-tree base model)...", task)
    models = [_train_mlp(X_train_sc, y_train, X_val_sc, y_val, seed) for seed in SEEDS]

    raw_val = _mlp_predict(models, X_val_sc, train_mode=False).mean(axis=0)
    raw_test = _mlp_predict(models, X_test_sc, train_mode=False).mean(axis=0)
    logger.info("  MLP ensemble val AUROC=%.4f test AUROC=%.4f",
                roc_auc_score(y_val, raw_val), roc_auc_score(y_test, raw_test))

    iso = IsotonicCalibrator().fit(raw_val, y_val)
    d1_val, d1_test = iso.predict_proba(raw_val), iso.predict_proba(raw_test)

    ts = dataset.timeseries
    d2_val = compute_d2_fast(ts, pd.Index(idx_val), dataset.admission_times,
                              dataset.prediction_times, lookback_hours=24, baseline_hours=6)
    d2_test = compute_d2_fast(ts, pd.Index(idx_test), dataset.admission_times,
                               dataset.prediction_times, lookback_hours=24, baseline_hours=6)
    d4_val = compute_d4_fast(ts, pd.Index(idx_val), dataset.prediction_times, lookback_hours=24)
    d4_test = compute_d4_fast(ts, pd.Index(idx_test), dataset.prediction_times, lookback_hours=24)

    cohort_sim = CohortSimilarityEstimator().fit(X_train_sc)
    d3_val, d3_test = cohort_sim.transform(X_val_sc), cohort_sim.transform(X_test_sc)

    # D5: true MC-dropout (active dropout layers), not XGBoost's input-dropout proxy
    passes_val = np.stack([_mlp_predict([models[0]], X_val_sc, train_mode=True)[0]
                            for _ in range(N_DROPOUT_PASSES)], axis=0)
    passes_test = np.stack([_mlp_predict([models[0]], X_test_sc, train_mode=True)[0]
                             for _ in range(N_DROPOUT_PASSES)], axis=0)
    bernoulli_val = d1_val * (1 - d1_val) + 1e-6
    bernoulli_test = d1_test * (1 - d1_test) + 1e-6
    d5_val = 1.0 - np.clip(passes_val.var(axis=0) / bernoulli_val, 0.0, 1.0)
    d5_test = 1.0 - np.clip(passes_test.var(axis=0) / bernoulli_test, 0.0, 1.0)

    # Composite v2 on the MLP: D5' = true MC-dropout variance of the MLP.
    counts = [c for c in dataset.features.columns]
    Xc_val = dataset.features.loc[idx_val].fillna(0).to_numpy()
    Xc_test = dataset.features.loc[idx_test].fillna(0).to_numpy()
    count_idx = [k for k, c in enumerate(counts) if c.endswith('_count')]
    other_val = np.column_stack([sufficiency_fraction(Xc_val[:, count_idx]), d3_val, d4_val,
                                 -passes_val.var(axis=0)])
    other_test = np.column_stack([sufficiency_fraction(Xc_test[:, count_idx]), d3_test, d4_test,
                                  -passes_test.var(axis=0)])
    composite = ErrorComposite().fit(raw_val, y_val.astype(int), other_val,
                                     patient_groups(DATA_DIR, idx_val))
    r_test = composite.score(d1_test, other_test)

    # Symmetric gate added in round 1 (R1-3 / R2-4): the comparison the paper's claim rests on.
    gates = {"confidence_mlp": d1_test,
             "symmetric_mlp": np.maximum(d1_test, 1 - d1_test),
             "reliability_mlp": r_test}
    keep = {g: retained_mask(s, COVERAGE) for g, s in gates.items()}
    y_int = y_test.astype(int)
    ci_lo, ci_hi = paired_brier_diff_ci(y_int, d1_test, keep["symmetric_mlp"],
                                        keep["reliability_mlp"], np.random.default_rng(42))
    rows = []
    for policy_name, k in keep.items():
        rows.append({
            "task": task, "base_model": "MLP", "policy": policy_name, "coverage": COVERAGE,
            "brier": float(brier_score_loss(y_int[k], d1_test[k])),
            "auroc": float(roc_auc_score(y_int[k], d1_test[k])),
            "n_predicted": int(k.sum()),
            "jaccard_deferred_vs_reliability": jaccard(~k, ~keep["reliability_mlp"]),
            "brier_sym_minus_rel_ci_lo": ci_lo, "brier_sym_minus_rel_ci_hi": ci_hi,
        })
        logger.info("  [MLP] %s: brier=%.4f auroc=%.4f", policy_name, rows[-1]["brier"], rows[-1]["auroc"])
    return pd.DataFrame(rows)


def part_b_selectivenet(task, X_train_sc, X_val_sc, X_test_sc, y_train, y_val, y_test,
                         target_coverage=COVERAGE,
                         lamb=32.0, n_epochs=80, patience=10) -> pd.DataFrame:
    logger.info("[Part B] [%s] Training SelectiveNet (target coverage=%.2f)...", task, target_coverage)
    torch.manual_seed(0)
    model = SelectiveNet(X_train_sc.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    Xt, yt = torch.tensor(X_train_sc), torch.tensor(y_train)
    Xv, yv = torch.tensor(X_val_sc), torch.tensor(y_val)

    def selective_loss(f_logits, g, y):
        per_sample = bce(f_logits, y)
        coverage = g.mean().clamp(min=1e-6)
        sel_risk = (g * per_sample).mean() / coverage
        cov_penalty = torch.clamp(target_coverage - coverage, min=0) ** 2
        return sel_risk + lamb * cov_penalty, coverage.item()

    best_val, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 512):
            idx = perm[i:i + 512]
            opt.zero_grad()
            f_logits, g = model(Xt[idx])
            loss, _ = selective_loss(f_logits, g, yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            f_logits, g = model(Xv)
            val_loss, val_cov = selective_loss(f_logits, g, yv)
        if val_loss.item() < best_val - 1e-4:
            best_val, best_state, bad_epochs = val_loss.item(), {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    model.load_state_dict(best_state)
    logger.info("  SelectiveNet trained. Final val coverage=%.3f", val_cov)

    model.eval()
    with torch.no_grad():
        f_logits_test, g_test = model(torch.tensor(X_test_sc))
    p_test = torch.sigmoid(f_logits_test).numpy()
    g_test = g_test.numpy()

    n_pred = int(np.round(len(y_test) * target_coverage))
    idx = np.argsort(-g_test)[:n_pred]
    y_sel, p_sel = y_test[idx], p_test[idx]
    row = {
        "task": task, "base_model": "SelectiveNet", "policy": "selectivenet_learned_rejection",
        "coverage": target_coverage,
        "brier": float(brier_score_loss(y_sel, p_sel)),
        "auroc": float(roc_auc_score(y_sel, p_sel)),
        "n_predicted": n_pred,
    }
    logger.info("  SelectiveNet: brier=%.4f auroc=%.4f", row["brier"], row["auroc"])

    # Reference rows: XGBoost confidence/reliability gates on the SAME test split,
    # evaluated against SelectiveNet's own calibrated probabilities is not comparable
    # (different base model); report separately with XGBoost's own probabilities.
    return pd.DataFrame([row])


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for task in TASKS:
        (dataset, idx_train, idx_val, idx_test,
         X_train_sc, X_val_sc, X_test_sc, y_train, y_val, y_test) = _load_splits(task)

        df_a = part_a_reliability_on_mlp(task, dataset, idx_train, idx_val, idx_test,
                                          X_train_sc, X_val_sc, X_test_sc, y_train, y_val, y_test)
        df_b = part_b_selectivenet(task, X_train_sc, X_val_sc, X_test_sc, y_train, y_val, y_test)
        all_rows.append(pd.concat([df_a, df_b], ignore_index=True))

    df = pd.concat(all_rows, ignore_index=True)
    out_path = RESULTS_DIR / "neural_baselines.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved %d rows to %s", len(df), out_path)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
