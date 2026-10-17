"""Tests for base-model training diagnostics (editor comment E-1)."""
import numpy as np
import pandas as pd
import xgboost as xgb

from deferral_frontier.src.diagnostics import training_diagnostics
from deferral_frontier.src.splits import load_patient_split
from src.models.xgboost_model import XGBResult


def _toy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + rng.normal(scale=1.0, size=n) > 0).astype(int)
    return X, y


def test_training_diagnostics_one_row_per_member_with_best_iteration():
    X, y = _toy()
    members = []
    for s in (0, 1):
        m = xgb.XGBClassifier(n_estimators=200, early_stopping_rounds=5, eval_metric="logloss",
                              random_state=s, max_depth=2)
        m.fit(X[:400], y[:400], eval_set=[(X[400:], y[400:])], verbose=False)
        members.append(XGBResult(model=m, seed=s, val_auroc=0.0))
    d = training_diagnostics(members, X[:400], y[:400], X[400:], y[400:])
    assert list(d.columns) == ["seed", "best_iteration", "train_auroc", "train_logloss",
                               "val_auroc", "val_logloss"]
    assert len(d) == 2 and (d["best_iteration"] < 199).all()
    assert (d["train_auroc"] >= d["val_auroc"] - 0.05).all()
    assert d["train_logloss"].between(0, 1).all()


def test_load_patient_split_reads_ids_and_keeps_patients_disjoint(tmp_path):
    idx = pd.Index(np.arange(1000) + 100, name="hadm_id")
    pid = pd.DataFrame({"patient_id": np.arange(1000) // 2}, index=idx)
    pid.to_parquet(tmp_path / "patient_ids.parquet")
    y = pd.Series(np.arange(1000) % 5 == 0, index=idx).astype(int)
    tr, va, te = load_patient_split(tmp_path, y)
    p = pid["patient_id"]
    assert not set(p.loc[tr]) & set(p.loc[te]) and not set(p.loc[va]) & set(p.loc[te])
    assert len(tr) + len(va) + len(te) == 1000


def test_load_patient_split_can_group_by_hospital(tmp_path):
    idx = pd.Index(np.arange(2000) + 100, name="stay_id")
    pd.DataFrame({"hospital_id": np.arange(2000) % 40}, index=idx).to_parquet(tmp_path / "hospital_ids.parquet")
    y = pd.Series(np.arange(2000) % 7 == 0, index=idx).astype(int)
    tr, va, te = load_patient_split(tmp_path, y, ids_file="hospital_ids.parquet", column="hospital_id")
    h = pd.read_parquet(tmp_path / "hospital_ids.parquet")["hospital_id"]
    assert not set(h.loc[tr]) & set(h.loc[te]) and not set(h.loc[va]) & set(h.loc[te])
