"""Apply CQR to a TACTiS-2 sample-based forecast parquet.

End-to-end pipeline:
  1. Load forecast parquet (long format with 200 samples per (time, test_idx))
  2. Load matched truth parquet (target_N columns + item_id 'SPLIT{N}')
  3. Reshape to per-(turbine, component, lead) sample tensors and matched truth
  4. Run split-CQR per stratum × confidence level
  5. Persist calibrated_intervals.parquet, metrics_summary.csv,
     coverage_report.json, cp_thresholds.json

Plotting is in a separate module (plot_cp_phase0i_b.py).

Usage:
    python apply_cp_to_phase0i_b.py \\
        --forecast <forecast_*.parquet> \\
        --truth    <..._test_denormalize.parquet> \\
        --output   <outdir> \\
        --split-id 194 \\
        --alphas 0.5 0.2 0.1 0.05 0.01
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from conformal_prediction import (
    CPResult,
    apply_cqr_split,
    average_width,
    coverage_at,
    empirical_quantiles_from_samples,
    stratify_alpha_levels,
    winkler_interval_score,
)

# Reuse existing CRPS-from-samples
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from wind_forecasting.postprocessing.probabilistic_metrics import (  # noqa: E402
    continuous_ranked_probability_score_samples,
)


logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger("apply_cp")


N_TURBINES = 88
N_COMPONENTS = 2  # horz, vert
N_LEAD = 4        # 0, 15, 30, 45 s
N_SAMPLES = 200


def map_target_to_wt_col(target_idx: int) -> str:
    """target_0..87 → ws_horz_wt001..088; target_88..175 → ws_vert_wt001..088."""
    if target_idx < N_TURBINES:
        return f"ws_horz_wt{target_idx + 1:03d}"
    return f"ws_vert_wt{target_idx - N_TURBINES + 1:03d}"


def load_forecast_long(forecast_path: Path) -> pl.DataFrame:
    """Load the long-format forecast parquet and verify schema."""
    df = pl.read_parquet(forecast_path)
    required = {"time", "sample", "test_idx", "continuity_group"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Forecast parquet missing columns: {missing}")
    return df


def load_truth_split(truth_path: Path, split_id: int) -> pl.DataFrame:
    """Load truth and filter to SPLIT{N}; verify target_0..175 present."""
    df = pl.read_parquet(truth_path)
    item_id = f"SPLIT{split_id}"
    sub = df.filter(pl.col("item_id") == item_id).sort("time")
    if len(sub) == 0:
        raise ValueError(f"No rows for item_id={item_id} in {truth_path}")
    target_cols = [c for c in sub.columns if c.startswith("target_")]
    if len(target_cols) != N_TURBINES * N_COMPONENTS:
        raise ValueError(
            f"Expected {N_TURBINES*N_COMPONENTS} target_ cols, found {len(target_cols)}"
        )
    return sub


def assemble_samples_truth(
    fcst: pl.DataFrame,
    truth: pl.DataFrame,
    target_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build matched (samples, truth, lead_step, valid_time) arrays for one variable.

    Returns:
        samples:    (n_rows, N_SAMPLES)  — model samples per matched forecast row
        truth_arr:  (n_rows,)            — ground truth (m/s) at each row
        lead_step:  (n_rows,)            — 0..N_LEAD-1, lead within test_idx
        valid_time: (n_rows,)            — forecast valid time (datetime64[ns])

    A "row" = (test_idx, lead_step) pair (one forecast valid time). Rows are
    aligned across all 200 samples by pivoting from long form.
    """
    wt_col = map_target_to_wt_col(target_idx)
    target_col = f"target_{target_idx}"

    # Compute lead_step per (test_idx, time) by ranking time within test_idx
    fcst_ranked = (
        fcst.sort(["test_idx", "time"])
        .with_columns(
            ((pl.col("time").rank(method="dense").over("test_idx")) - 1).alias("lead_step")
        )
    )

    # Pivot: rows = (test_idx, lead_step, time), cols = sample, values = wt_col
    pivot = (
        fcst_ranked.select(["test_idx", "lead_step", "time", "sample", wt_col])
        .pivot(values=wt_col, index=["test_idx", "lead_step", "time"], on="sample")
    )
    sample_cols = [c for c in pivot.columns if c not in ("test_idx", "lead_step", "time")]
    sample_cols_int = sorted(sample_cols, key=lambda c: float(c))

    # Inner-join truth in one pass so samples + truth stay row-aligned
    joined = (
        pivot.join(truth.select(["time", target_col]), on="time", how="inner")
        .sort(["test_idx", "lead_step"])
    )
    samples = joined.select(sample_cols_int).to_numpy().astype(np.float64)
    truth_arr = joined[target_col].to_numpy().astype(np.float64)
    lead_step = joined["lead_step"].to_numpy().astype(np.int32)
    valid_time = joined["time"].to_numpy()

    return samples, truth_arr, lead_step, valid_time


def run_cp_for_target(
    samples: np.ndarray,
    truth_arr: np.ndarray,
    lead_step: np.ndarray,
    alphas: list[float],
    calibration_fraction: float,
) -> dict:
    """Per-lead CQR fit; returns nested dict: lead → alpha → result."""
    out: dict = {}
    for lead in range(N_LEAD):
        mask = lead_step == lead
        if mask.sum() < 20:
            continue
        out[int(lead)] = {}
        for alpha in alphas:
            res = apply_cqr_split(
                truth=truth_arr[mask],
                samples=samples[mask],
                alpha=alpha,
                calibration_fraction=calibration_fraction,
                chronological=True,
            )
            out[int(lead)][float(alpha)] = res
    return out


def metrics_for_target(
    samples: np.ndarray,
    truth_arr: np.ndarray,
    lead_step: np.ndarray,
    cp_results: dict,
) -> dict:
    """Headline metrics: CRPS, MAE, RMSE per lead + global. Coverage + width per alpha."""
    per_lead: dict = {}
    for lead in range(N_LEAD):
        m = lead_step == lead
        if m.sum() == 0:
            continue
        s = samples[m]
        y = truth_arr[m]
        mean = s.mean(axis=1)
        mae = float(np.mean(np.abs(mean - y)))
        rmse = float(np.sqrt(np.mean((mean - y) ** 2)))
        crps_raw = float(continuous_ranked_probability_score_samples(y, s))
        d = {"mae": mae, "rmse": rmse, "crps_raw": crps_raw, "n": int(m.sum())}
        if lead in cp_results:
            for alpha, res in cp_results[lead].items():
                test_y = y[res.test_indices]
                d[f"coverage_a{alpha:.3f}"] = float(coverage_at(test_y, res.lower, res.upper))
                d[f"width_a{alpha:.3f}"] = float(average_width(res.lower, res.upper))
                d[f"winkler_a{alpha:.3f}"] = float(
                    winkler_interval_score(test_y, res.lower, res.upper, alpha)
                )
        per_lead[int(lead)] = d
    return per_lead


def persistence_skill(
    samples: np.ndarray,
    truth_arr: np.ndarray,
    lead_step: np.ndarray,
    valid_time: np.ndarray,
) -> dict:
    """% improvement of model CRPS over persistence baseline (truth at lead 0)."""
    # Persistence baseline: previous truth carries forward.
    # For each (test_idx, lead>0) the baseline is truth at lead=0 of that test_idx.
    by_lead = {}
    n_rows = len(truth_arr)
    # Sort by valid_time so within a test_idx, lead 0 precedes lead 1..3
    order = np.argsort(valid_time)
    truth_o = truth_arr[order]; lead_o = lead_step[order]; samples_o = samples[order]
    # Identify lead-0 truth as "previous observation"; carry it forward for the 3 subsequent leads
    # via index alignment within a contiguous block of 4 rows per test_idx
    # (works because rows are sorted by valid_time and within a test_idx lead is monotone)
    for lead in range(N_LEAD):
        m = lead_o == lead
        if m.sum() == 0:
            continue
        # Persistence "forecast" mean = last observed lead-0 truth carried forward
        # Compute by repeating each lead-0 truth value 4 times (one per lead) and slicing
        l0 = truth_o[lead_o == 0]
        if lead == 0:
            persist_mean = l0
        else:
            persist_mean = l0[: m.sum()]
        y = truth_o[m]
        model_mean = samples_o[m].mean(axis=1)
        # Use absolute error as a simple deterministic baseline against persistence
        model_mae = float(np.mean(np.abs(model_mean - y)))
        persist_mae = float(np.mean(np.abs(persist_mean - y)))
        by_lead[int(lead)] = {
            "model_mae": model_mae,
            "persist_mae": persist_mae,
            "skill": float(1.0 - model_mae / persist_mae) if persist_mae > 0 else float("nan"),
        }
    return by_lead


def build_intervals_table(
    target_idx: int,
    cp_results: dict,
    truth_arr: np.ndarray,
    lead_step: np.ndarray,
    valid_time: np.ndarray,
    sample_mean: np.ndarray,
    sample_lo_raw: dict,
    sample_hi_raw: dict,
) -> pl.DataFrame:
    """Long-format DataFrame of (time, target_idx, lead, alpha, lo, hi, truth, mean, raw_lo, raw_hi)."""
    rows = []
    target_col = map_target_to_wt_col(target_idx)
    for lead, alpha_map in cp_results.items():
        for alpha, res in alpha_map.items():
            test_idx_arr = res.test_indices
            mask = lead_step == lead
            mask_idx = np.where(mask)[0]
            test_global_idx = mask_idx[test_idx_arr]
            for j, ti in enumerate(test_global_idx):
                rows.append({
                    "time": valid_time[ti],
                    "target_col": target_col,
                    "target_idx": target_idx,
                    "lead_step": int(lead),
                    "alpha": float(alpha),
                    "lower_cp": float(res.lower[j]),
                    "upper_cp": float(res.upper[j]),
                    "lower_raw": float(sample_lo_raw[(lead, alpha)][test_idx_arr[j]]),
                    "upper_raw": float(sample_hi_raw[(lead, alpha)][test_idx_arr[j]]),
                    "truth": float(truth_arr[ti]),
                    "forecast_mean": float(sample_mean[ti]),
                    "q_hat": float(res.q_hat),
                })
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--forecast", type=Path, required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--split-id", type=int, default=194)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.5, 0.2, 0.1, 0.05, 0.01],
                    help="Miscoverage levels (e.g. 0.1 = 90%% CI)")
    ap.add_argument("--calibration-fraction", type=float, default=0.8)
    ap.add_argument("--n-turbines", type=int, default=N_TURBINES,
                    help="Override for partial runs / debugging")
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "plots").mkdir(exist_ok=True)

    alphas = stratify_alpha_levels(args.alphas)
    log.info("Loading forecast: %s", args.forecast)
    fcst = load_forecast_long(args.forecast)
    log.info("  rows=%d, n_test_idx=%d", len(fcst), fcst["test_idx"].n_unique())

    log.info("Loading truth: %s (SPLIT%d)", args.truth, args.split_id)
    truth = load_truth_split(args.truth, args.split_id)
    log.info("  rows=%d", len(truth))

    # Run CP per target
    all_intervals = []
    per_target_metrics: dict = {}
    per_target_skill: dict = {}
    cp_thresholds: dict = {}

    # n_turbines K selects the first K physical turbines (both horz + vert).
    # target indices: {0..K-1} for horz, {N_TURBINES..N_TURBINES+K-1} for vert.
    k = args.n_turbines
    target_indices = list(range(k)) + list(range(N_TURBINES, N_TURBINES + k))
    n_targets = len(target_indices)
    log.info("Processing %d targets (%d horz + %d vert)", n_targets, k, k)
    for i, ti in enumerate(target_indices):
        if i % 20 == 0:
            log.info("  target %d / %d  (ti=%d, %s)", i, n_targets, ti, map_target_to_wt_col(ti))
        samples, y, lead, vtime = assemble_samples_truth(fcst, truth, ti)
        cp_res = run_cp_for_target(samples, y, lead, alphas, args.calibration_fraction)
        per_target_metrics[ti] = metrics_for_target(samples, y, lead, cp_res)
        per_target_skill[ti] = persistence_skill(samples, y, lead, vtime)

        # Cache raw sample-based intervals for plotting comparisons
        sample_lo_raw: dict = {}
        sample_hi_raw: dict = {}
        for ld in range(N_LEAD):
            m = lead == ld
            for a in alphas:
                if m.sum() < 20:
                    continue
                lo_a, hi_a = empirical_quantiles_from_samples(samples[m], a)
                sample_lo_raw[(ld, a)] = lo_a
                sample_hi_raw[(ld, a)] = hi_a
        sample_mean = samples.mean(axis=1)

        df = build_intervals_table(ti, cp_res, y, lead, vtime, sample_mean,
                                   sample_lo_raw, sample_hi_raw)
        if len(df) > 0:
            all_intervals.append(df)

        # Record thresholds
        cp_thresholds[ti] = {
            f"lead{ld}_a{a:.3f}": res.q_hat
            for ld, am in cp_res.items() for a, res in am.items()
        }

    # Write outputs
    log.info("Writing outputs to %s", args.output)
    if all_intervals:
        big = pl.concat(all_intervals)
        big.write_parquet(args.output / "calibrated_intervals.parquet")
        log.info("  calibrated_intervals.parquet: %d rows", len(big))

    # Coverage report (global + per-stratum summary)
    coverage_report: dict = {"global": {}, "per_target": {}}
    for ti, by_lead in per_target_metrics.items():
        target_summary: dict = {}
        for lead, m in by_lead.items():
            target_summary[f"lead{lead}"] = m
        coverage_report["per_target"][ti] = target_summary
    # Global: average over targets per lead × alpha
    for lead in range(N_LEAD):
        for a in alphas:
            covs = [
                per_target_metrics[ti].get(lead, {}).get(f"coverage_a{a:.3f}")
                for ti in target_indices
                if lead in per_target_metrics.get(ti, {})
            ]
            covs = [c for c in covs if c is not None]
            if covs:
                coverage_report["global"][f"lead{lead}_a{a:.3f}"] = {
                    "mean_coverage": float(np.mean(covs)),
                    "median_coverage": float(np.median(covs)),
                    "min_coverage": float(np.min(covs)),
                    "nominal": float(1.0 - a),
                }

    (args.output / "coverage_report.json").write_text(json.dumps(coverage_report, indent=2))
    log.info("  coverage_report.json")

    (args.output / "cp_thresholds.json").write_text(json.dumps(cp_thresholds, indent=2))
    log.info("  cp_thresholds.json")

    # Metrics summary csv
    metric_rows = []
    for ti, by_lead in per_target_metrics.items():
        for lead, m in by_lead.items():
            row = {"target_idx": ti, "target_col": map_target_to_wt_col(ti),
                   "lead_step": lead, **m}
            if ti in per_target_skill and lead in per_target_skill[ti]:
                row.update({f"skill_{k}": v for k, v in per_target_skill[ti][lead].items()})
            metric_rows.append(row)
    metrics_df = pl.DataFrame(metric_rows)
    metrics_df.write_csv(args.output / "metrics_summary.csv")
    log.info("  metrics_summary.csv")

    # Headline numbers
    if metric_rows:
        pcts = []
        for row in metric_rows:
            if "skill_skill" in row and not (row["skill_skill"] != row["skill_skill"]):
                pcts.append(row["skill_skill"])
        log.info("Done. Mean persistence-skill across all (target,lead): %.4f (%.1f%% positive)",
                 float(np.mean(pcts)) if pcts else float("nan"),
                 100.0 * float(np.mean(np.array(pcts) > 0)) if pcts else float("nan"))


if __name__ == "__main__":
    main()
