"""Tests for the round-1 before/after stop-gate summary."""
from deferral_frontier.experiments.r1_before_after import verdict


def test_verdict_uses_holm_p_and_sign_of_symmetric_minus_composite():
    assert verdict(diff=-0.002, p_holm=0.01) == "favors symmetric"
    assert verdict(diff=0.002, p_holm=0.01) == "favors composite"
    assert verdict(diff=-0.002, p_holm=0.20) == "n.s."
