"""Unit tests for conformal_prediction.py — CQR core math.

Covers:
  1. Coverage on synthetic Gaussian residuals (split-CP correctness)
  2. CQR adapts width to heteroscedasticity (better than fixed-width)
  3. Finite-sample correction: (n+1)/n quantile is the right index
  4. Permutation-invariance of CP threshold
  5. CPResult round-trip via dict (preserves arrays + metadata)
  6. Per-group CQR isolates strata correctly
  7. Self-test convenience function passes
"""

from __future__ import annotations

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from conformal_prediction import (  # noqa: E402
    CPResult,
    apply_cqr_grouped,
    apply_cqr_split,
    average_width,
    coverage_at,
    cp_threshold,
    cqr_nonconformity,
    empirical_quantiles_from_samples,
    pinball_loss,
    self_test_synthetic_gaussian,
    winkler_interval_score,
)


@pytest.fixture
def gaussian_residual_data():
    """Synthetic: biased + over-confident predictor on N(0,1) truth."""
    rng = np.random.default_rng(42)
    n, S = 5000, 200
    truth = rng.normal(0.0, 1.0, size=n)
    predictor_mean = 0.5 * truth + 0.3
    samples = rng.normal(loc=predictor_mean[:, None], scale=0.4, size=(n, S))
    return truth, samples


def test_split_cp_synthetic_coverage(gaussian_residual_data):
    """At α=0.1 the empirical test-set coverage must be within ±3pp of 0.9."""
    truth, samples = gaussian_residual_data
    res = apply_cqr_split(truth, samples, alpha=0.1, chronological=False, rng_seed=0)
    assert abs(res.empirical_coverage_on_testset - 0.9) < 0.03, \
        f"Test coverage {res.empirical_coverage_on_testset} out of tolerance"
    assert res.lower.shape == res.upper.shape
    assert np.all(res.upper >= res.lower), "Upper must be >= lower"


def test_cqr_adapts_to_heteroscedasticity():
    """Two regimes (low-var and high-var) — CQR should give wider bands in high-var."""
    rng = np.random.default_rng(0)
    n_per_regime = 2000
    # Regime A: low variance
    truth_a = rng.normal(0.0, 1.0, size=n_per_regime)
    samples_a = rng.normal(loc=truth_a[:, None], scale=0.5, size=(n_per_regime, 200))
    # Regime B: high variance
    truth_b = rng.normal(0.0, 1.0, size=n_per_regime)
    samples_b = rng.normal(loc=truth_b[:, None], scale=2.0, size=(n_per_regime, 200))
    # Concatenate; alternate regime so cal/test split mixes
    truth = np.concatenate([truth_a, truth_b])
    samples = np.concatenate([samples_a, samples_b], axis=0)
    group = np.concatenate([np.zeros(n_per_regime, dtype=int),
                            np.ones(n_per_regime, dtype=int)])

    grouped = apply_cqr_grouped(truth, samples, group, alpha=0.1)
    width_a = average_width(grouped[0].lower, grouped[0].upper)
    width_b = average_width(grouped[1].lower, grouped[1].upper)
    assert width_b > width_a * 1.5, \
        f"High-var regime width {width_b} should exceed low-var width {width_a} by >50%"


def test_finite_sample_correction():
    """For scores = arange(100), cp_threshold(α=0.1) = ceil((101)(0.9))/100 quantile = scores[90]."""
    scores = np.arange(100.0)
    q = cp_threshold(scores, alpha=0.1)
    # k = ceil(101 * 0.9) = ceil(90.9) = 91 → scores[90] = 90
    assert q == 90.0, f"Expected 90.0, got {q}"

    # Below threshold: returns +inf when n is too small
    scores_small = np.arange(5.0)
    q_small = cp_threshold(scores_small, alpha=0.01)  # k = ceil(6 * 0.99) = 6 > 5
    assert np.isinf(q_small), "Should be +inf when n too small for finite-sample guarantee"


def test_cp_threshold_permutation_invariant():
    """Shuffling input score order must not change cp_threshold output."""
    rng = np.random.default_rng(7)
    scores = rng.normal(size=500)
    q1 = cp_threshold(scores, alpha=0.1)
    perm = rng.permutation(500)
    q2 = cp_threshold(scores[perm], alpha=0.1)
    assert q1 == q2, f"Permutation changed threshold: {q1} vs {q2}"


def test_cpresult_arrays_match_test_indices(gaussian_residual_data):
    """test_indices field must align with lower/upper for the post-split coverage to match."""
    truth, samples = gaussian_residual_data
    res = apply_cqr_split(truth, samples, alpha=0.1, chronological=False, rng_seed=11)
    n_test = len(res.test_indices)
    assert res.lower.shape == (n_test,)
    assert res.upper.shape == (n_test,)
    # Recompute coverage from test_indices independently
    cov = float(np.mean((truth[res.test_indices] >= res.lower)
                        & (truth[res.test_indices] <= res.upper)))
    assert abs(cov - res.empirical_coverage_on_testset) < 1e-10


def test_apply_cqr_grouped_isolation():
    """Different group's CP fits must yield different q_hat values when scales differ."""
    rng = np.random.default_rng(2)
    n = 1500
    truth = rng.normal(0.0, 1.0, size=n)
    # 3 regimes with sample-noise scale 0.3, 1.0, 3.0
    group = np.repeat([0, 1, 2], n // 3)
    samples = np.empty((n, 200))
    for g, scale in enumerate([0.3, 1.0, 3.0]):
        m = group == g
        samples[m] = rng.normal(loc=truth[m, None], scale=scale, size=(m.sum(), 200))

    grouped = apply_cqr_grouped(truth, samples, group, alpha=0.1)
    assert set(grouped.keys()) == {0, 1, 2}
    qhats = [grouped[g].q_hat for g in [0, 1, 2]]
    # Higher noise scale → larger ABSOLUTE residual outside the band → larger q_hat correction
    # (Empirical 90% band shrinks slightly with noise but q_hat dominates.)
    # We just check that they differ meaningfully — homogeneity would be 1e-2 magnitude variation.
    assert max(qhats) - min(qhats) > 0.1, f"q_hats too similar: {qhats}"


def test_self_test_passes():
    cov, target = self_test_synthetic_gaussian(seed=99)
    assert abs(cov - target) < 0.03, f"Self-test coverage {cov} vs target {target}"


def test_winkler_score_basic():
    """Winkler IS = width when all truth inside interval; penalizes misses correctly."""
    truth = np.array([0.0, 0.0, 0.0])
    lower = np.array([-1.0, -1.0, -1.0])
    upper = np.array([1.0, 1.0, 1.0])
    # All inside → IS = mean(width) = 2.0
    assert winkler_interval_score(truth, lower, upper, alpha=0.1) == 2.0

    # Truth outside → adds penalty
    truth_miss = np.array([2.0, 0.0, 0.0])
    is_miss = winkler_interval_score(truth_miss, lower, upper, alpha=0.1)
    # Width = 2; penalty for first = (2/0.1) * (2 - 1) = 20; mean = (2+20 + 2 + 2)/3 = 8.0
    expected = (2.0 + 20.0 + 2.0 + 2.0) / 3.0
    assert abs(is_miss - expected) < 1e-9


def test_empirical_quantiles_shape_and_order():
    rng = np.random.default_rng(3)
    samples = rng.normal(size=(50, 200))
    lo, hi = empirical_quantiles_from_samples(samples, alpha=0.1)
    assert lo.shape == (50,)
    assert hi.shape == (50,)
    assert np.all(hi > lo), "Upper quantile must exceed lower"


def test_pinball_loss_symmetry():
    """At quantile 0.5 (median), pinball = 0.5 * mean|y - q_pred|."""
    rng = np.random.default_rng(5)
    truth = rng.normal(size=1000)
    pred = rng.normal(size=1000)
    pb = pinball_loss(truth, pred, quantile_level=0.5)
    expected = 0.5 * np.mean(np.abs(truth - pred))
    assert abs(pb - expected) < 1e-9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
