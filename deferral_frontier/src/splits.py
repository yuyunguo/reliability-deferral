"""Patient-level train/validation/test splitting.

Admissions (MIMIC) or ICU stays (eICU) from the same patient are kept in a
single partition, so no patient contributes to both training and evaluation
(reviewer comments R1-1 / R2-1, round 1).
"""
import logging
from typing import Tuple

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

logger = logging.getLogger(__name__)


def _take_group_fold(
    idx: np.ndarray, groups: np.ndarray, y: np.ndarray, frac: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Split positions into (rest, held) with ~frac held out, grouped and stratified."""
    n_splits = max(2, int(round(1.0 / frac)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest_pos, held_pos = next(sgkf.split(idx, y, groups))
    return idx[rest_pos], idx[held_pos]


def patient_group_split(
    index: np.ndarray,
    groups: np.ndarray,
    y: np.ndarray,
    sizes: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split rows into train/val/test with every patient in exactly one partition.

    Args:
        index: Row identifiers (hadm_id or patientunitstayid).
        groups: Patient identifier for each row (subject_id or uniquepid).
        y: Binary outcome for each row, used for stratification.
        sizes: Target (train, val, test) fractions; must sum to 1.
        seed: Random seed.

    Returns:
        Sorted arrays of row identifiers for (train, val, test).

    Raises:
        ValueError: If inputs differ in length, any group is missing, or sizes do not sum to 1.
    """
    index, groups, y = np.asarray(index), np.asarray(groups), np.asarray(y)
    if not (len(index) == len(groups) == len(y)):
        raise ValueError("index, groups and y must have equal length")
    if any(g is None or (isinstance(g, float) and np.isnan(g)) for g in groups):
        raise ValueError("every row needs a patient identifier")
    if abs(sum(sizes) - 1.0) > 1e-6:
        raise ValueError(f"sizes must sum to 1, got {sizes}")

    _, val_frac, test_frac = sizes
    pos = np.arange(len(index))
    rest, test = _take_group_fold(pos, groups, y, test_frac, seed)
    train, val = _take_group_fold(
        rest, groups[rest], y[rest], val_frac / (1.0 - test_frac), seed
    )
    logger.info("Patient split: train=%d val=%d test=%d", len(train), len(val), len(test))
    return tuple(np.sort(index[p]) for p in (train, val, test))


def load_patient_split(
    data_dir, y, sizes: Tuple[float, float, float] = (0.70, 0.15, 0.15), seed: int = 42,
    ids_file: str = "patient_ids.parquet", column: str = "patient_id",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grouped split for the rows of `y`; groups are patients by default.

    Args:
        data_dir: Directory containing `ids_file` (index = row id).
        y: Outcome Series indexed by row id.
        sizes: Target (train, val, test) fractions.
        seed: Random seed.
        ids_file: Parquet file holding the group column (hospital_ids.parquet for a
            held-out-hospital split).
        column: Group column name.

    Returns:
        Row ids for (train, val, test).
    """
    import pandas as pd
    from pathlib import Path

    pid = pd.read_parquet(Path(data_dir) / ids_file)[column]
    groups = pid.reindex(y.index).to_numpy()
    return patient_group_split(y.index.to_numpy(), groups, y.to_numpy(), sizes, seed)
