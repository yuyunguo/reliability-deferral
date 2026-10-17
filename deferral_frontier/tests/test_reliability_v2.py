"""Tests for the pre-registered composite v2 (plan/r1-composite-v2-preregistration.md)."""
import numpy as np

from deferral_frontier.src.reliability_v2 import ErrorComposite, sufficiency_fraction


def _synthetic(n=6000, seed=0):
    """Raw scores are well ranked; an extra 'quality' feature adds label noise where it is low."""
    rng = np.random.default_rng(seed)
    eta = rng.beta(1.5, 6, n)
    quality = rng.random(n)
    noisy = quality < 0.3                         # low-quality patients: model's p is off
    y = (rng.random(n) < np.where(noisy, 0.5, eta)).astype(int)
    raw = eta                                     # model sees eta, not the noise mechanism
    other = np.column_stack([quality, rng.random(n), rng.random(n), rng.random(n)])
    groups = np.arange(n) // 2
    return raw, y, other, groups


def test_sufficiency_fraction_counts_observed_signals():
    counts = np.array([[0, 3, 1], [0, 0, 0], [2, 2, 2]])
    assert np.allclose(sufficiency_fraction(counts), [2 / 3, 0, 1])


def test_reliability_is_non_increasing_in_p_times_one_minus_p():
    raw, y, other, groups = _synthetic()
    comp = ErrorComposite().fit(raw, y, other, groups)
    p = np.linspace(0.01, 0.5, 50)
    fixed = np.tile(other[:1], (50, 1))
    r = comp.score(p, fixed)
    assert np.all(np.diff(r) <= 1e-9)


def test_cross_fit_halves_share_no_group():
    raw, y, other, groups = _synthetic()
    a, b = ErrorComposite()._halves(groups, seed=42)
    assert not set(groups[a]) & set(groups[b]) and len(a) + len(b) == len(groups)


def test_uses_extra_dimension_when_it_predicts_error():
    raw, y, other, groups = _synthetic(seed=1)
    comp = ErrorComposite().fit(raw, y, other, groups)
    rt, yt, ot, _ = _synthetic(seed=2)
    p = rt
    keep_v2 = np.argsort(-comp.score(p, ot), kind="stable")[: int(0.8 * len(p))]
    keep_sym = np.argsort(-np.maximum(p, 1 - p), kind="stable")[: int(0.8 * len(p))]
    brier = lambda k: np.mean((p[k] - yt[k]) ** 2)  # noqa: E731
    assert brier(keep_v2) < brier(keep_sym)


def test_build_v2_shapes_and_d1_is_p_times_one_minus_p():
    from deferral_frontier.src.reliability_v2 import build_v2
    raw, y, other, groups = _synthetic(n=2000)
    rng = np.random.default_rng(3)
    X = np.column_stack([rng.integers(0, 3, 2000), rng.random(2000)]).astype(float)
    names = ["vital_heart_rate_count", "vital_heart_rate_mean"]
    dropout_fn = lambda Z: np.clip(Z[:, 1] + rng.normal(0, 0.01, len(Z)), 0, 1)  # noqa: E731
    comp, dv, dt, rv, rt, ot = build_v2(raw, y, raw, raw, X, X, names, other[:, 1], other[:, 1],
                                        other[:, 2], other[:, 2], dropout_fn, groups)
    assert np.allclose(dt.D1, raw * (1 - raw))
    assert np.allclose(dt.D2, (X[:, 0] > 0).astype(float))
    assert rt.shape == (2000,) and ot.shape == (2000, 4)


def test_blank_dimension_sets_one_column_constant_and_keeps_shape():
    from deferral_frontier.experiments.r1_v2_ablation import blank
    other = np.arange(12, dtype=float).reshape(3, 4)
    out = blank(other, [1, 3])
    assert out.shape == (3, 4) and (out[:, [1, 3]] == 0).all() and (out[:, [0, 2]] == other[:, [0, 2]]).all()
    assert (other[:, 1] != 0).all()   # input not mutated
