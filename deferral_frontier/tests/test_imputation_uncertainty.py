"""Tests for D2* (imputation uncertainty; plan/r1-d2-imputation-preregistration.md)."""
import numpy as np

from deferral_frontier.src.imputation_uncertainty import StochasticImputer, imputation_uncertainty


class _LinearModel:
    """Classifier: P(y=1) = clip(0, 1, w @ x); sensitive to the feature w makes large."""

    def __init__(self, w):
        self.w = np.asarray(w, dtype=float)

    def predict_proba(self, X):
        p = np.clip(X @ self.w, 0.0, 1.0)
        return np.column_stack([1 - p, p])


def _make_data(n=600, p=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    imp = X.copy()
    X[-n // 3:, 2] = np.nan            # feature 2 missing for last third
    X[rng.random(n) < 0.05, 1] = np.nan  # feature 1 occasionally
    med = np.nanmedian(X, axis=0)
    for j in range(p):
        imp[np.isnan(X[:, j]), j] = med[j]
    return X, imp


def test_imputer_only_fills_missing_and_preserves_observed():
    X, imp = _make_data()
    impu = StochasticImputer().fit(imp, X)
    rng = np.random.default_rng(0)
    Xp = impu.sample(imp, X, rng)
    observed = ~np.isnan(X)
    assert np.allclose(Xp[observed], X[observed])
    assert not np.isnan(Xp).any()


def test_imputer_reproducible():
    X, imp = _make_data()
    impu = StochasticImputer().fit(imp, X)
    a = imputation_uncertainty(_LinearModel(np.ones(4)), imp, X, impu, seed=7)
    b = imputation_uncertainty(_LinearModel(np.ones(4)), imp, X, impu, seed=7)
    assert np.array_equal(a, b)


def test_uncertainty_zero_for_fully_observed_rows():
    X, imp = _make_data()
    impu = StochasticImputer().fit(imp, X)
    d2 = imputation_uncertainty(_LinearModel(np.ones(4)), imp, X, impu, seed=42)
    fully_observed = ~np.isnan(X).any(axis=1)
    assert fully_observed.sum() > 0
    assert np.allclose(d2[fully_observed], 0.0)


def test_uncertainty_positive_for_missing_informative_feature():
    X, imp = _make_data()
    impu = StochasticImputer().fit(imp, X)
    w = np.array([0.0, 0.0, 5.0, 0.0])  # only feature 2 matters
    d2 = imputation_uncertainty(_LinearModel(w), imp, X, impu, n_passes=50, seed=42)
    missing_feat2 = np.isnan(X[:, 2])
    assert d2[missing_feat2].mean() > 0
    only_feat1 = np.isnan(X[:, 1]) & ~np.isnan(X[:, 2])
    assert d2[only_feat1].mean() < d2[missing_feat2].mean()


def test_imputed_values_are_observed_donor_values():
    X, imp = _make_data()
    impu = StochasticImputer().fit(imp, X)
    rng = np.random.default_rng(0)
    Xp = impu.sample(imp, X, rng)
    for j in impu._models:
        missing = np.isnan(X[:, j])
        if not missing.any():
            continue
        observed_vals = set(np.round(X[~missing, j], 12))
        assert set(np.round(Xp[missing, j], 12)).issubset(observed_vals)


def test_imputer_excludes_fitted_column_from_predictors():
    X, imp = _make_data()
    p = X.shape[1]
    impu = StochasticImputer().fit(imp, X)
    for j in impu._models:
        assert impu._models[j].coef_.shape[0] == p - 1
