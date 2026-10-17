"""Tests for risk-stratified deferral (plan/r1-stratified-deferral-preregistration.md)."""
import numpy as np

from deferral_frontier.src.stratified_deferral import (
    bootstrap_delta, null_brier, risk_bins, stratified_keep, tie_ranks)


def test_risk_bins_are_equal_size_and_ordered_by_risk():
    rng = np.random.default_rng(0)
    p = rng.random(1003)
    bins = risk_bins(p, 10, tie_ranks(len(p), 42))
    sizes = np.bincount(bins)
    assert len(sizes) == 10 and sizes.max() - sizes.min() <= 1
    assert all(p[bins == b].max() <= p[bins == b + 1].min() for b in range(9))


def test_risk_bins_split_tied_probabilities_by_fixed_permutation():
    p = np.repeat([0.1, 0.5], 50)
    bins = risk_bins(p, 4, tie_ranks(100, 42))
    assert np.array_equal(np.bincount(bins), [25, 25, 25, 25])
    assert np.array_equal(bins, risk_bins(p, 4, tie_ranks(100, 42)))


def test_stratified_keep_defers_lowest_scores_within_each_bin():
    bins = np.repeat([0, 1], 10)
    score = np.tile(np.arange(10), 2).astype(float)
    keep = stratified_keep(score, bins, 0.2, tie_ranks(20, 42))
    assert (~keep).sum() == 4
    assert set(np.flatnonzero(~keep)) == {0, 1, 10, 11}


def test_null_brier_is_exact_random_within_bin_expectation():
    bins = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    e = np.array([0, 0, 0, 0, 1.0, 1, 1, 1, 1, 1])
    # each bin keeps 4 of 5; expected retained mean = (4*0.2 + 4*1.0) / 8
    assert np.isclose(null_brier(e, bins, 0.2), (4 * 0.2 + 4 * 1.0) / 8)


def test_oracle_score_beats_null_and_constant_score_does_not():
    rng = np.random.default_rng(1)
    n = 4000
    p = rng.random(n)
    y = (rng.random(n) < p).astype(int)
    e = (p - y) ** 2
    oracle = bootstrap_delta(-e, p, y, k=10, n_boot=200, seed=42)
    assert oracle["delta"] > 0 and oracle["ci_lo"] > 0
    noise = bootstrap_delta(rng.random(n), p, y, k=10, n_boot=200, seed=42)
    assert noise["ci_lo"] < 0 < noise["ci_hi"]
