"""Tests for post hoc robustness tests (cluster bootstrap, likelihood-ratio test, control titration)."""
import numpy as np

from deferral_frontier.src.posthoc_tests import (
    cluster_bootstrap_deltas, lrt_beyond_risk, titrated_control)


def _sim(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.02, 0.6, n)
    y = (rng.random(n) < p).astype(int)
    return p, y, rng


def test_cluster_bootstrap_returns_paired_resamples_for_every_score():
    p, y, rng = _sim()
    clusters = rng.integers(0, 40, len(p))
    scores = {"symmetric": np.maximum(p, 1 - p), "noise": rng.random(len(p))}
    point, boots = cluster_bootstrap_deltas(scores, p, y, clusters, k=10, n_boot=50, seed=1)
    assert set(point) == set(scores) and all(len(b) == 50 for b in boots.values())


def test_cluster_bootstrap_is_wider_when_errors_share_a_cluster_effect():
    rng = np.random.default_rng(3)
    n, n_cl = 6000, 30
    clusters = rng.integers(0, n_cl, n)
    shift = rng.normal(0, 0.15, n_cl)[clusters]
    p = rng.uniform(0.05, 0.5, n)
    y = (rng.random(n) < np.clip(p + shift, 0, 1)).astype(int)
    s = np.clip(p + shift, 0, 1)  # score aligned with the cluster effect
    scores = {"symmetric": np.maximum(p, 1 - p), "s": -s}
    _, b_cl = cluster_bootstrap_deltas(scores, p, y, clusters, k=10, n_boot=200, seed=1)
    _, b_pt = cluster_bootstrap_deltas(scores, p, y, np.arange(n), k=10, n_boot=200, seed=1)
    assert np.std(b_cl["s"]) > np.std(b_pt["s"])


def test_lrt_detects_score_that_carries_outcome_information_beyond_risk():
    p, y, rng = _sim(n=6000, seed=4)
    eta = np.clip(p + rng.normal(0, 0.1, len(p)), 0.001, 0.999)
    y = (rng.random(len(p)) < eta).astype(int)
    informative = eta - p
    stat, pval = lrt_beyond_risk(y, p, informative)
    assert pval < 1e-6 and stat > 0


def test_lrt_does_not_flag_pure_noise():
    p, y, rng = _sim(n=6000, seed=5)
    _, pval = lrt_beyond_risk(y, p, rng.random(len(p)))
    assert pval > 0.001


def test_titrated_control_weakens_as_noise_grows():
    p, y, _ = _sim(n=5000, seed=6)
    strong = titrated_control(p, y, k=1, n_rep=5, n_boot=50, seed=1)
    weak = titrated_control(p, y, k=32, n_rep=5, n_boot=50, seed=1)
    assert strong["mean_delta_adj"] > weak["mean_delta_adj"]
    assert 0.0 <= weak["detect_rate"] <= strong["detect_rate"] <= 1.0
