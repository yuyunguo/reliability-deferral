"""Tests for the v3 reliability scores (plan/r1-reliability-v3-preregistration.md)."""
import numpy as np
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from deferral_frontier.src.reliability_v3 import (
    L_PARAMS, MissingAwareSimilarity, error_score, fit_error_model, nan_mask_variance)
from deferral_frontier.src.stratified_deferral import bootstrap_delta, bootstrap_deltas, summarize


def _train_with_missing(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    X[rng.random(n) < 0.05, 0] = np.nan  # feature 0 missing in 5% of training rows
    return X


def test_similarity_penalizes_rare_missingness_that_imputation_hides():
    X = _train_with_missing()
    sim = MissingAwareSimilarity().fit(X)
    med = np.nanmedian(X[:, 0])
    observed_at_median = np.array([[med, 0.0, 0.0]])
    missing = np.array([[np.nan, 0.0, 0.0]])  # imputes to the same vector
    assert sim.transform(missing)[0] < sim.transform(observed_at_median)[0]


def test_similarity_in_unit_interval_and_ignores_never_missing_columns():
    X = _train_with_missing()
    sim = MissingAwareSimilarity().fit(X)
    s = sim.transform(X)
    assert np.all((s > 0) & (s <= 1))
    assert sim.indicator_cols.tolist() == [0]


def test_similarity_uses_train_medians_on_new_data():
    X = _train_with_missing()
    sim = MissingAwareSimilarity().fit(X)
    far = np.array([[np.nan, 50.0, 50.0]])
    assert np.isfinite(sim.transform(far)).all()


class _CountNaN:
    """Fake classifier: P(y=1) = fraction of NaN entries in the row."""

    def predict_proba(self, X):
        f = np.isnan(X).mean(axis=1)
        return np.column_stack([1 - f, f])


def test_nan_mask_variance_masks_only_observed_entries():
    X = np.array([[np.nan] * 20, list(range(20))], dtype=float)
    v = nan_mask_variance(_CountNaN(), X, frac=0.1, n_passes=50, seed=42)
    assert v[0] == 0.0  # nothing observed, nothing to mask
    assert v[1] > 0.0


def test_nan_mask_variance_is_reproducible_and_leaves_input_unchanged():
    X = np.arange(40, dtype=float).reshape(2, 20)
    X0 = X.copy()
    a = nan_mask_variance(_CountNaN(), X, seed=42)
    b = nan_mask_variance(_CountNaN(), X, seed=42)
    assert np.array_equal(a, b) and np.array_equal(X, X0)


def _miscalibrated_subgroup(n=6000, seed=1):
    """Raw score is right for g=0 but badly wrong for g=1."""
    rng = np.random.default_rng(seed)
    g = (rng.random(n) < 0.3).astype(float)
    raw = rng.uniform(0.1, 0.4, n)
    true_p = np.where(g == 1, 1 - raw, raw)
    y = (rng.random(n) < true_p).astype(int)
    groups = np.arange(n) // 2
    return raw, y, g[:, None], groups


def test_learned_error_model_ranks_miscalibrated_subgroup_as_less_reliable():
    raw, y, extra, groups = _miscalibrated_subgroup()
    model = fit_error_model(xgb.XGBRegressor(**L_PARAMS), raw, y, extra, groups)
    p = np.full(2, 0.3)
    s = error_score(model, p, np.array([[0.0], [1.0]]))
    assert s[1] < s[0]


def test_ridge_error_model_ranks_miscalibrated_subgroup_as_less_reliable():
    raw, y, extra, groups = _miscalibrated_subgroup()
    model = fit_error_model(make_pipeline(StandardScaler(), Ridge(alpha=1.0)), raw, y, extra, groups)
    s = error_score(model, np.full(2, 0.3), np.array([[0.0], [1.0]]))
    assert s[1] < s[0]


def _sim(n=3000, seed=2):
    rng = np.random.default_rng(seed)
    p = rng.random(n)
    y = (rng.random(n) < p).astype(int)
    return p, y


def test_bootstrap_deltas_matches_single_score_bootstrap():
    p, y = _sim()
    s = np.random.default_rng(3).random(len(p))
    single = bootstrap_delta(s, p, y, k=10, n_boot=100, seed=42)
    point, boots = bootstrap_deltas({"s": s}, p, y, k=10, n_boot=100, seed=42)
    summ = summarize(point["s"], boots["s"])
    for key in ("delta", "ci_lo", "ci_hi", "p"):
        assert np.isclose(summ[key], single[key])


def test_adjusted_effect_of_symmetric_against_itself_is_zero():
    p, y = _sim()
    sym = np.maximum(p, 1 - p)
    point, boots = bootstrap_deltas({"symmetric": sym, "same": sym.copy()}, p, y, k=10, n_boot=50, seed=42)
    assert point["same"] - point["symmetric"] == 0
    assert np.all(boots["same"] - boots["symmetric"] == 0)
