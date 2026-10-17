"""XGBoost base model with multi-seed training and MC Dropout approximation."""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import numpy as np
import xgboost as xgb
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@dataclass
class XGBResult:
    model: xgb.XGBClassifier
    seed: int
    val_auroc: float


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: DictConfig,
    seed: int,
) -> XGBResult:
    """Train a single XGBoost model with the given seed.

    Args:
        X_train, y_train: Training features and labels.
        X_val, y_val: Validation features and labels (for early stopping).
        cfg: Model config from conf/model/xgboost.yaml.
        seed: Random seed (one of the pre-committed 5 seeds).

    Returns:
        XGBResult with fitted model and validation AUROC.
    """
    from sklearn.metrics import roc_auc_score

    model = xgb.XGBClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        min_child_weight=cfg.min_child_weight,
        eval_metric=cfg.eval_metric,
        early_stopping_rounds=cfg.early_stopping_rounds,
        random_state=seed,
        use_label_encoder=False,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    val_probs = model.predict_proba(X_val)[:, 1]
    val_auroc = float(roc_auc_score(y_val, val_probs))
    logger.info("Seed %d: val AUROC=%.4f, best_iter=%d", seed, val_auroc, model.best_iteration)
    return XGBResult(model=model, seed=seed, val_auroc=val_auroc)


def train_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: DictConfig,
    seeds: List[int],
) -> List[XGBResult]:
    """Train N XGBoost models with different seeds (multi-seed + deep ensemble base)."""
    results = []
    for seed in seeds:
        logger.info("Training seed %d / %s", seed, seeds)
        results.append(train_xgboost(X_train, y_train, X_val, y_val, cfg, seed))
    return results


def ensemble_predict(results: List[XGBResult], X: np.ndarray) -> np.ndarray:
    """Average predicted probabilities across ensemble members."""
    preds = np.stack([r.model.predict_proba(X)[:, 1] for r in results], axis=0)
    return preds.mean(axis=0)


def ensemble_predict_all_seeds(
    results: List[XGBResult], X: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-seed predictions for mean ± SD reporting."""
    preds = np.stack([r.model.predict_proba(X)[:, 1] for r in results], axis=0)
    return preds.mean(axis=0), preds.std(axis=0)


def make_dropout_predict_fn(
    model: xgb.XGBClassifier,
    dropout_rate: float,
    seed: int = 42,
) -> Callable[[np.ndarray], np.ndarray]:
    """Input dropout perturbation function for MC Dropout approximation (D5)."""
    rng = np.random.default_rng(seed)

    def _predict(X: np.ndarray) -> np.ndarray:
        mask = rng.random(X.shape) > dropout_rate
        return model.predict_proba(X * mask)[:, 1]

    return _predict
