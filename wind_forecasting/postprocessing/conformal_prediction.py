"""Conformalized Quantile Regression (CQR) for TACTiS-2 sample-based forecasts.

Implements the Romano-Patterson-Candès 2019 algorithm with finite-sample
correction. Given per-(time, lead, turbine, component) empirical quantiles
derived from N=200 raw model samples, computes a calibrated additive offset
so that test-time intervals have coverage >= the nominal level.

Reused upstream: probabilistic_metrics.prediction_interval_from_samples.

This module is loader-agnostic: callers pass numpy arrays. The orchestration
script (apply_cp_to_phase0i_b.py) handles the parquet I/O + reshaping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CPResult:
    """Calibrated CP outputs for a single stratum × confidence level.

    For each test-set forecast row, `lower` and `upper` are the calibrated
    interval edges (raw m/s). `q_hat` is the nonconformity quantile that
    was added on each side. `n_calibration` is the cal-set size used.
    `test_indices` is the integer index array into the original (input)
    truth/samples arrays for the test split, so callers can reconstruct
    matched (truth, lower, upper) triples without re-running the split.
    """

    lower: np.ndarray
    upper: np.ndarray
    q_hat: float
    nominal_alpha: float
    n_calibration: int
    empirical_coverage_on_calset: float
    test_indices: np.ndarray
    empirical_coverage_on_testset: float


def empirical_quantiles_from_samples(
    samples: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row α/2 and 1−α/2 empirical quantiles of `samples`.

    Args:
        samples: shape (n_rows, n_samples) — raw model samples per forecast row.
        alpha:   miscoverage level, e.g. 0.1 for a 90% interval.

    Returns:
        (q_lo, q_hi) each shape (n_rows,).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    lo_p = (alpha / 2.0) * 100.0
    hi_p = (1.0 - alpha / 2.0) * 100.0
    q_lo = np.percentile(samples, lo_p, axis=1)
    q_hi = np.percentile(samples, hi_p, axis=1)
    return q_lo, q_hi


def cqr_nonconformity(
    truth: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
) -> np.ndarray:
    """CQR score: max(q_lo − y, y − q_hi).

    Positive when y is outside the interval; negative when it is inside.
    Same shape as inputs (n,).
    """
    return np.maximum(q_lo - truth, truth - q_hi)


def cp_threshold(
    scores: np.ndarray,
    alpha: float,
) -> float:
    """Finite-sample-corrected (1−α) quantile of nonconformity scores.

    Vovk/Romano correction: ceil((n+1)(1−α))/n quantile of scores. Returns
    +inf if n is too small to guarantee finite-sample coverage at α.
    """
    n = len(scores)
    if n == 0:
        raise ValueError("Empty calibration scores")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    return float(np.sort(scores)[k - 1])


def apply_cqr_split(
    truth: np.ndarray,
    samples: np.ndarray,
    alpha: float,
    calibration_fraction: float = 0.8,
    chronological: bool = True,
    rng_seed: int | None = 42,
) -> CPResult:
    """Run a single split-CQR pass on one stratum.

    Args:
        truth:        (n,)        observed values
        samples:      (n, S)      model samples per row, S = num samples
        alpha:        miscoverage level
        calibration_fraction:    cal/total split (e.g. 0.8 → 80% cal, 20% test)
        chronological: if True, take the first `calibration_fraction*n`
                       rows as calibration. If False, shuffle with rng_seed.
        rng_seed:     used only if chronological=False
    """
    n = len(truth)
    if n < 4:
        raise ValueError(f"Need at least 4 rows for split-CQR, got {n}")

    idx = np.arange(n)
    if not chronological:
        rng = np.random.default_rng(rng_seed)
        rng.shuffle(idx)
    n_cal = int(round(calibration_fraction * n))
    cal_idx = idx[:n_cal]
    test_idx = idx[n_cal:]

    q_lo_all, q_hi_all = empirical_quantiles_from_samples(samples, alpha)
    scores_cal = cqr_nonconformity(truth[cal_idx], q_lo_all[cal_idx], q_hi_all[cal_idx])
    q_hat = cp_threshold(scores_cal, alpha)

    lo_test = q_lo_all[test_idx] - q_hat
    hi_test = q_hi_all[test_idx] + q_hat
    cov_cal = float(
        np.mean(
            (truth[cal_idx] >= q_lo_all[cal_idx] - q_hat)
            & (truth[cal_idx] <= q_hi_all[cal_idx] + q_hat)
        )
    )
    cov_test = float(
        np.mean((truth[test_idx] >= lo_test) & (truth[test_idx] <= hi_test))
    )

    return CPResult(
        lower=lo_test,
        upper=hi_test,
        q_hat=q_hat,
        nominal_alpha=alpha,
        n_calibration=len(cal_idx),
        empirical_coverage_on_calset=cov_cal,
        test_indices=test_idx,
        empirical_coverage_on_testset=cov_test,
    )


def apply_cqr_grouped(
    truth: np.ndarray,
    samples: np.ndarray,
    group_keys: np.ndarray,
    alpha: float,
    calibration_fraction: float = 0.8,
    chronological: bool = True,
    min_group_size: int = 20,
) -> dict[int, CPResult]:
    """Run split-CQR independently per group (e.g. per lead-time).

    Args:
        truth:        (n,)
        samples:      (n, S)
        group_keys:   (n,) integer or string keys; one CPResult per unique key
        alpha:        miscoverage level
        min_group_size: groups smaller than this fall through with empty result

    Returns:
        dict mapping group_key → CPResult. Groups too small to calibrate are
        omitted (caller may handle them via a global fallback).
    """
    out: dict = {}
    keys = np.asarray(group_keys)
    for key in np.unique(keys):
        mask = keys == key
        if int(mask.sum()) < min_group_size:
            continue
        out[key.item() if hasattr(key, "item") else key] = apply_cqr_split(
            truth=truth[mask],
            samples=samples[mask],
            alpha=alpha,
            calibration_fraction=calibration_fraction,
            chronological=chronological,
        )
    return out


def coverage_at(
    truth: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Empirical interval coverage."""
    return float(np.mean((truth >= lower) & (truth <= upper)))


def average_width(
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Mean width of intervals."""
    return float(np.mean(upper - lower))


def winkler_interval_score(
    truth: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> float:
    """Winkler/Gneiting interval score at level (1−α). Lower is better.

    IS = (u−l) + (2/α)(l−y) 1{y<l} + (2/α)(y−u) 1{y>u}
    """
    width = upper - lower
    below = (truth < lower) * (2.0 / alpha) * (lower - truth)
    above = (truth > upper) * (2.0 / alpha) * (truth - upper)
    return float(np.mean(width + below + above))


def pinball_loss(
    truth: np.ndarray,
    quantile_pred: np.ndarray,
    quantile_level: float,
) -> float:
    """Mean pinball loss at quantile level `q`."""
    diff = truth - quantile_pred
    return float(np.mean(np.maximum(quantile_level * diff, (quantile_level - 1.0) * diff)))


def stratify_alpha_levels(
    alphas: Iterable[float],
) -> list[float]:
    """Validate + sort a list of miscoverage levels.

    Common usage: alphas = [0.5, 0.2, 0.1, 0.05, 0.01] for {50%, 80%, 90%, 95%, 99%}
    confidence levels respectively.
    """
    alphas = sorted(set(alphas))
    for a in alphas:
        if not (0.0 < a < 1.0):
            raise ValueError(f"alpha must be in (0,1), got {a}")
    return alphas


def self_test_synthetic_gaussian(
    n: int = 2000,
    n_samples: int = 200,
    alpha: float = 0.1,
    seed: int = 0,
) -> tuple[float, float]:
    """Sanity self-test: biased + over-confident predictor on Gaussian truth.

    Setup: truth ~ N(0, 1). Predictor outputs samples ~ N(0.5*truth + 0.3, σ=0.4),
    i.e. biased AND too narrow. Without CQR, raw 90% interval would miscover.
    With CQR, post-calibration test-set coverage should hit ~90% ± 3%.
    """
    rng = np.random.default_rng(seed)
    truth = rng.normal(0.0, 1.0, size=n)
    predictor_mean = 0.5 * truth + 0.3
    samples = rng.normal(loc=predictor_mean[:, None], scale=0.4, size=(n, n_samples))
    res = apply_cqr_split(truth, samples, alpha=alpha, chronological=False, rng_seed=seed)
    return res.empirical_coverage_on_testset, 1.0 - alpha


if __name__ == "__main__":
    cov, tgt = self_test_synthetic_gaussian()
    print(f"Self-test: empirical coverage = {cov:.4f}, target = {tgt:.4f}")
    print(f"  PASS" if abs(cov - tgt) < 0.03 else "  FAIL")
