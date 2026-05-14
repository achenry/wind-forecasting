"""Plot the RAW model's predictive uncertainty as individual sample
trajectories, without any CP layer. Shows F1 sample-tightness collapse +
F2 time-flatness directly.

Within a test_idx the 200 samples are jointly drawn (copula-coupled), so
connecting same-sample-index across the 4 lead times gives a valid
trajectory. Across test_idxs samples are independent — drawn as separate
bundles.

Vectorized via polars pivot + numpy indexing — ~50x faster than per-row
filters.

Usage:
    python plot_raw_trajectories.py \\
        --forecast <forecast_*.parquet> \\
        --truth    <..._test_denormalize.parquet> \\
        --output   <outdir> \\
        --turbines wt008 wt042 wt075
"""

from __future__ import annotations

import argparse
import logging
from datetime import timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger("raw_trajectories")


def target_col_for_wt(wt: str, component: str = "horz") -> str:
    return f"ws_{component}_{wt}"


def wt_to_target_idx(wt: str, component: str, n_turbines: int = 88) -> int:
    n = int(wt.replace("wt", ""))
    if component == "horz":
        return n - 1
    return n_turbines + n - 1


def build_sample_tensor(
    fcst: pl.DataFrame,
    wt_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pivot once → arrays for fast slicing.

    Returns:
        samples_arr: shape (n_test_idx, n_lead_steps=4, n_samples=200)
        test_idxs:   shape (n_test_idx,) integer keys
        times_arr:   shape (n_test_idx, n_lead_steps) datetime64[ns]
    """
    # Drop test_idxs that don't have full 4 × N_SAMPLES rows (a few edge-of-test ones)
    expected = 4 * 200
    keep = (fcst.group_by("test_idx").len()
            .filter(pl.col("len") == expected)["test_idx"])
    fcst = fcst.filter(pl.col("test_idx").is_in(keep))
    sub = fcst.sort(["test_idx", "time", "sample"]).with_columns(
        ((pl.col("time").rank(method="dense").over("test_idx")) - 1).alias("lead_step")
    )
    pivot = (
        sub.select(["test_idx", "lead_step", "time", "sample", wt_col])
        .pivot(values=wt_col, index=["test_idx", "lead_step", "time"], on="sample")
        .sort(["test_idx", "lead_step"])
    )
    sample_cols = [c for c in pivot.columns
                   if c not in ("test_idx", "lead_step", "time")]
    sample_cols_sorted = sorted(sample_cols, key=lambda c: float(c))

    # Reshape to (n_test_idx, n_lead, n_samples)
    test_idxs = np.array(sorted(sub["test_idx"].unique().to_list()))
    n_lead = pivot["lead_step"].max() + 1
    n_samples = len(sample_cols_sorted)
    samples_arr = pivot.select(sample_cols_sorted).to_numpy().reshape(
        len(test_idxs), n_lead, n_samples
    )
    times_arr = pivot["time"].to_numpy().reshape(len(test_idxs), n_lead)
    return samples_arr, test_idxs, times_arr


def plot_single_60s(
    samples_arr: np.ndarray,
    times_arr: np.ndarray,
    test_idx_row: int,
    truth_at_time: dict,
    wt_col: str,
    output_dir: Path,
) -> Path:
    """One test_idx → 4 time-steps × 200 trajectories."""
    times = times_arr[test_idx_row]
    samples = samples_arr[test_idx_row]  # (4, 200)
    truth = np.array([truth_at_time.get(int(t), np.nan) for t in times])

    fig, ax = plt.subplots(figsize=(11, 6))
    # All 200 trajectories
    for k in range(samples.shape[1]):
        ax.plot(times, samples[:, k], color="#1f77b4", alpha=0.06, lw=0.6)
    mean_path = samples.mean(axis=1)
    p5 = np.percentile(samples, 5, axis=1)
    p95 = np.percentile(samples, 95, axis=1)
    ax.fill_between(times, p5, p95, color="#1f77b4", alpha=0.25,
                    label="5–95% sample percentile")
    ax.plot(times, mean_path, color="#1f77b4", lw=2.5, label="Sample mean")
    ax.plot(times, truth, color="black", lw=2.0, marker="o", markersize=6,
            label="Ground truth")
    width = float(np.max(p95 - p5))
    truth_range = float(np.nanmax(truth) - np.nanmin(truth))
    title = (f"{wt_col} — single 60s forecast (test_idx={test_idx_row})\n"
             f"Raw ensemble width (5-95%): max {width:.4f} m/s  |  "
             f"truth range: {truth_range:.4f} m/s  |  "
             f"ratio: {truth_range/max(width, 1e-9):.1f}×")
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Wind component (m/s)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fname = output_dir / f"raw_{wt_col}_single60s_test{test_idx_row}.png"
    fig.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return fname


def plot_window(
    samples_arr: np.ndarray,
    times_arr: np.ndarray,
    truth_times: np.ndarray,
    truth_vals: np.ndarray,
    test_idx_rows: np.ndarray,
    n_trajectories: int,
    label: str,
    wt_col: str,
    output_dir: Path,
) -> Path:
    """Multiple consecutive test_idxs as separate bundles overlaid on the
    same time axis."""
    if len(test_idx_rows) == 0:
        return None
    fig, ax = plt.subplots(figsize=(14, 6))

    widths_per_window = []
    for row in test_idx_rows:
        times = times_arr[row]
        samples = samples_arr[row]
        # Subsample for legibility
        idx = np.linspace(0, samples.shape[1] - 1, n_trajectories, dtype=int)
        sub_samples = samples[:, idx]
        for k in range(sub_samples.shape[1]):
            ax.plot(times, sub_samples[:, k], color="#1f77b4", alpha=0.08, lw=0.45)
        # Per-bundle mean
        ax.plot(times, samples.mean(axis=1), color="#1f77b4", lw=0.9, alpha=0.65)
        p5 = np.percentile(samples, 5, axis=1)
        p95 = np.percentile(samples, 95, axis=1)
        widths_per_window.append(float(np.max(p95 - p5)))

    # Truth: all points in the window
    t_min = times_arr[test_idx_rows[0]].min()
    t_max = times_arr[test_idx_rows[-1]].max()
    mask = (truth_times >= t_min) & (truth_times <= t_max)
    if mask.any():
        ax.plot(truth_times[mask], truth_vals[mask], color="black", lw=1.2,
                label="Ground truth")

    truth_range = (float(truth_vals[mask].max() - truth_vals[mask].min())
                   if mask.any() else 0.0)
    median_w = float(np.median(widths_per_window)) if widths_per_window else 0.0
    max_w = float(np.max(widths_per_window)) if widths_per_window else 0.0
    title = (f"{wt_col} — {label}: raw sample trajectories\n"
             f"{len(test_idx_rows)} consecutive 60s forecasts × "
             f"{n_trajectories} subsampled trajectories\n"
             f"Truth range across window: {truth_range:.2f} m/s  |  "
             f"per-window ensemble width: median {median_w:.4f}, "
             f"max {max_w:.4f} m/s  |  ratio: {truth_range/max(median_w, 1e-9):.1f}×")
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Wind component (m/s)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.autofmt_xdate()
    fname = output_dir / f"raw_{wt_col}_{label}.png"
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fname


def plot_violin(
    samples_arr: np.ndarray,
    times_arr: np.ndarray,
    truth_times: np.ndarray,
    truth_vals: np.ndarray,
    test_idx_rows: np.ndarray,
    label: str,
    wt_col: str,
    output_dir: Path,
) -> Path:
    """Violin plot: per-timestep sample distribution. Each violin should be a
    vertical line if F1 collapse holds."""
    if len(test_idx_rows) == 0:
        return None
    samples_per_time = []
    times_list = []
    for row in test_idx_rows:
        for j in range(samples_arr.shape[1]):
            samples_per_time.append(samples_arr[row, j])
            times_list.append(times_arr[row, j])
    times_arr_flat = np.array(times_list)
    # Truth at same times
    truth_lookup = dict(zip(truth_times.astype(np.int64), truth_vals))
    truth_at = np.array([truth_lookup.get(int(t), np.nan) for t in times_arr_flat])

    fig, ax = plt.subplots(figsize=(15, 6))
    positions = np.arange(len(times_list))
    parts = ax.violinplot(samples_per_time, positions=positions, widths=0.7,
                          showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#1f77b4")
        pc.set_alpha(0.5)
    ax.plot(positions, truth_at, color="black", lw=1.2, marker="o",
            markersize=3, label="Ground truth")
    stds = [float(np.std(s)) for s in samples_per_time]
    ax.set_title(f"{wt_col} — {label}: per-timestep sample distribution\n"
                 f"Each violin = 200 samples at one time. "
                 f"Median sample-std: {np.median(stds):.4f} m/s, "
                 f"max: {np.max(stds):.4f} m/s",
                 fontsize=10)
    ax.set_ylabel("Wind component (m/s)")
    n_show = min(15, len(times_list))
    show_idx = np.linspace(0, len(times_list) - 1, n_show, dtype=int)
    import pandas as pd
    labels_show = [pd.Timestamp(times_list[i]).strftime("%H:%M:%S") for i in show_idx]
    ax.set_xticks(positions[show_idx])
    ax.set_xticklabels(labels_show, rotation=45, fontsize=8)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fname = output_dir / f"raw_violin_{wt_col}_{label}.png"
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fname


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--forecast", type=Path, required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--split-id", type=int, default=194)
    ap.add_argument("--turbines", nargs="+", default=["wt008", "wt075"])
    ap.add_argument("--components", nargs="+", default=["horz"],
                    choices=["horz", "vert"])
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    log.info("Loading forecast %s", args.forecast)
    fcst = pl.read_parquet(args.forecast)
    log.info("Loading truth %s (SPLIT%d)", args.truth, args.split_id)
    truth_df = (pl.read_parquet(args.truth)
                .filter(pl.col("item_id") == f"SPLIT{args.split_id}")
                .sort("time"))

    files: list[Path] = []
    for wt in args.turbines:
        for comp in args.components:
            wt_col = target_col_for_wt(wt, comp)
            tgt_idx = wt_to_target_idx(wt, comp)
            tgt_col = f"target_{tgt_idx}"
            if wt_col not in fcst.columns or tgt_col not in truth_df.columns:
                log.warning("Skipping %s (missing column)", wt_col)
                continue
            log.info("Processing %s (%s)", wt_col, tgt_col)

            samples_arr, test_idxs, times_arr = build_sample_tensor(fcst, wt_col)
            truth_times = truth_df["time"].to_numpy()
            truth_vals = truth_df[tgt_col].to_numpy()
            truth_at_time = dict(zip(truth_times.astype(np.int64),
                                     truth_vals.astype(float)))

            mid_row = len(test_idxs) // 2
            mid_test_idx = int(test_idxs[mid_row])
            log.info("  center test_idx=%d (row=%d)", mid_test_idx, mid_row)

            # 1) Single 60s forecast
            f = plot_single_60s(samples_arr, times_arr, mid_row, truth_at_time,
                                wt_col, args.output)
            if f: files.append(f)

            # 2) Z1 — 4 consecutive test_idxs (~1 minute)
            rows = np.arange(mid_row, min(mid_row + 4, len(test_idxs)))
            f = plot_window(samples_arr, times_arr, truth_times, truth_vals,
                            rows, n_trajectories=200, label="Z1_1min",
                            wt_col=wt_col, output_dir=args.output)
            if f: files.append(f)

            # 3) Z2 — 20 consecutive (~5 minutes)
            rows = np.arange(mid_row, min(mid_row + 20, len(test_idxs)))
            f = plot_window(samples_arr, times_arr, truth_times, truth_vals,
                            rows, n_trajectories=60, label="Z2_5min",
                            wt_col=wt_col, output_dir=args.output)
            if f: files.append(f)

            # 4) Z3 — 40 consecutive (~10 minutes)
            rows = np.arange(mid_row, min(mid_row + 40, len(test_idxs)))
            f = plot_window(samples_arr, times_arr, truth_times, truth_vals,
                            rows, n_trajectories=30, label="Z3_10min",
                            wt_col=wt_col, output_dir=args.output)
            if f: files.append(f)

            # 5) Z4 — 240 consecutive (~1 hour)
            rows = np.arange(mid_row, min(mid_row + 240, len(test_idxs)))
            f = plot_window(samples_arr, times_arr, truth_times, truth_vals,
                            rows, n_trajectories=10, label="Z4_1hour",
                            wt_col=wt_col, output_dir=args.output)
            if f: files.append(f)

            # 6) Per-timestep violin over a 1-minute window
            rows = np.arange(mid_row, min(mid_row + 4, len(test_idxs)))
            f = plot_violin(samples_arr, times_arr, truth_times, truth_vals,
                            rows, label="Z1_1min", wt_col=wt_col,
                            output_dir=args.output)
            if f: files.append(f)

    log.info("Done. %d files written", len(files))


if __name__ == "__main__":
    main()
