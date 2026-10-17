"""Base-model training diagnostics (editor comment E-1, round 1)."""
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from src.models.xgboost_model import XGBResult


def training_diagnostics(
    ensemble: List[XGBResult],
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
) -> pd.DataFrame:
    """Early-stopping round and train/validation AUROC and log loss for each ensemble member.

    Predictions use the early-stopped model (XGBoost predicts with best_iteration).
    """
    rows = []
    for m in ensemble:
        p_tr = m.model.predict_proba(X_train)[:, 1]
        p_va = m.model.predict_proba(X_val)[:, 1]
        rows.append({
            "seed": m.seed,
            "best_iteration": int(m.model.best_iteration),
            "train_auroc": roc_auc_score(y_train, p_tr),
            "train_logloss": log_loss(y_train, p_tr),
            "val_auroc": roc_auc_score(y_val, p_va),
            "val_logloss": log_loss(y_val, p_va),
        })
    return pd.DataFrame(rows)
