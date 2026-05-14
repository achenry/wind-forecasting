"""Visualization suite for TACTiS-2 Phase 0i-B + CQR-calibrated intervals.

Six groups (A–F) of plots + one index HTML (G). Run after `apply_cp_to_phase0i_b.py`.

Usage:
    python plot_cp_phase0i_b.py \\
        --forecast <forecast_*.parquet> \\
        --truth    <..._test_denormalize.parquet> \\
        --intervals <calibrated_intervals.parquet> \\
        --metrics  <metrics_summary.csv> \\
        --output   <outdir>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from plot_utils import (  # noqa: E402
    binomial_ci,
    build_index_html,
    cp_palette,
    map_target_to_wt_col,
    regime_from_truth_window,
    select_representative_turbines,
    zoom_windows,
)

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger("plot_cp")

N_TURBINES = 88
N_COMPONENTS = 2
N_LEAD = 4
N_SAMPLES = 200


# ---------- Data loaders ----------

def load_intervals(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path)
    # `time` may be ns int — cast to datetime
    if df["time"].dtype == pl.Int64 or df["time"].dtype == pl.Float64:
        df = df.with_columns(pl.col("time").cast(pl.Int64).cast(pl.Datetime("ns")))
    return df


def load_truth(path: Path, split_id: int) -> pl.DataFrame:
    df = pl.read_parquet(path)
    return df.filter(pl.col("item_id") == f"SPLIT{split_id}").sort("time")


def load_metrics(path: Path) -> pl.DataFrame:
    return pl.read_csv(path)


# ---------- GROUP A: Time-series with bands at 5 zoom levels ----------

def plot_timeseries_with_bands(
    intervals: pl.DataFrame,
    selection,            # TurbineSelection
    output_dir: Path,
    n_lead: int = N_LEAD,
) -> list[Path]:
    """For one (target_idx), make 5 zoom-level plots showing truth + bands."""
    target_idx = selection.target_idx
    sub = intervals.filter(pl.col("target_idx") == target_idx).sort("time", "lead_step")
    if len(sub) == 0:
        return []

    # We plot only the lead-0 (most reliable) view for the time-axis plots,
    # because each forecast issuance covers 4 leads and we want a continuous
    # time series view. (Group A focuses on time-evolution.)
    sub = sub.filter(pl.col("lead_step") == 0)
    if len(sub) == 0:
        return []

    # Alpha-independent context: one row per time
    alphas = sorted(sub["alpha"].unique().to_list())
    ctx = (sub.filter(pl.col("alpha") == alphas[0])
           .select(["time", "truth", "forecast_mean"])
           .sort("time"))
    times = ctx["time"].to_numpy()
    truth = ctx["truth"].to_numpy()
    mean = ctx["forecast_mean"].to_numpy()

    # Per-alpha bands: pivot CP bounds + the raw ensemble band for a focus alpha
    pivot_lo = sub.pivot(values="lower_cp", index="time", on="alpha").sort("time")
    pivot_hi = sub.pivot(values="upper_cp", index="time", on="alpha").sort("time")
    cp_los = {a: pivot_lo[str(a)].to_numpy() for a in alphas if str(a) in pivot_lo.columns}
    cp_his = {a: pivot_hi[str(a)].to_numpy() for a in alphas if str(a) in pivot_hi.columns}

    # Raw band only at α=0.1 (focus level) for overlay
    raw_sub = sub.filter(pl.col("alpha") == 0.1).sort("time")
    raw_lo = raw_sub["lower_raw"].to_numpy()
    raw_hi = raw_sub["upper_raw"].to_numpy()

    out_files: list[Path] = []
    n_total = len(times)
    windows = zoom_windows(n_total)
    target_col = selection.target_col

    for w in windows:
        if w.end_idx - w.start_idx < 2:
            continue
        sl = slice(w.start_idx, w.end_idx)
        fig, ax = plt.subplots(figsize=(12, 5))
        # Plot CP bands first (widest to narrowest, so wide is on bottom)
        for a in sorted(cp_los.keys(), reverse=True):
            ax.fill_between(times[sl], cp_los[a][sl], cp_his[a][sl],
                            color=cp_palette["cp_band_levels"].get(a, cp_palette["cp_band"]),
                            alpha=0.35, label=f"CP {int((1-a)*100)}%")
        # Then the raw 90% band overlaid (very thin)
        ax.fill_between(times[sl], raw_lo[sl], raw_hi[sl],
                        color=cp_palette["raw_band"], alpha=0.6,
                        label="Uncalibrated ens 90%")
        ax.plot(times[sl], mean[sl], color=cp_palette["forecast_mean"], lw=1.5,
                label="Forecast mean")
        ax.plot(times[sl], truth[sl], color=cp_palette["truth"], lw=1.2,
                label="Ground truth")
        ax.set_title(f"{target_col} ({selection.regime}) — {w.description}", fontsize=11)
        ax.set_ylabel("Wind component (m/s)")
        # Hide x-axis labels for full / day-scale (too cluttered); show for shorter
        if w.label in ("L0_full", "L1_1day"):
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fname = output_dir / f"groupA_target{target_idx:03d}_{selection.regime}_{w.label}.png"
        fig.savefig(fname, dpi=110, bbox_inches="tight")
        plt.close(fig)
        out_files.append(fname)
    return out_files


# ---------- GROUP B: Side-by-side uncal vs CP-calibrated ----------

def plot_uncal_vs_cp(
    intervals: pl.DataFrame,
    selection,
    output_dir: Path,
    alpha_focus: float = 0.1,
) -> list[Path]:
    """2-panel figure at zoom L2 (1h) — left: uncal bands only; right: CP bands."""
    target_idx = selection.target_idx
    sub = intervals.filter((pl.col("target_idx") == target_idx)
                            & (pl.col("lead_step") == 0)
                            & (pl.col("alpha") == alpha_focus)).sort("time")
    if len(sub) == 0:
        return []
    times = sub["time"].to_numpy()
    truth = sub["truth"].to_numpy()
    mean = sub["forecast_mean"].to_numpy()
    raw_lo = sub["lower_raw"].to_numpy()
    raw_hi = sub["upper_raw"].to_numpy()
    cp_lo = sub["lower_cp"].to_numpy()
    cp_hi = sub["upper_cp"].to_numpy()

    # 1-hour window centered on the middle
    mid = len(times) // 2
    half = min(120, len(times) // 2)
    sl = slice(max(0, mid - half), min(len(times), mid + half))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    for ax, lo, hi, title, color in zip(
        axes,
        [raw_lo, cp_lo],
        [raw_hi, cp_hi],
        ["Uncalibrated ensemble band (90%)", "CP-calibrated band (90%)"],
        [cp_palette["raw_band"], cp_palette["cp_band"]],
    ):
        ax.fill_between(times[sl], lo[sl], hi[sl], color=color, alpha=0.4,
                        label="90% band")
        ax.plot(times[sl], mean[sl], color=cp_palette["forecast_mean"], lw=1.5,
                label="Forecast mean")
        ax.plot(times[sl], truth[sl], color=cp_palette["truth"], lw=1.2,
                label="Ground truth")
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Wind component (m/s)")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.suptitle(f"{selection.target_col} ({selection.regime}) — 1h window, lead 0",
                 fontsize=12)
    fig.autofmt_xdate()
    fname = output_dir / f"groupB_target{target_idx:03d}_{selection.regime}_uncal_vs_cp.png"
    fig.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return [fname]


# ---------- GROUP C: Reliability + PIT + sharpness ----------

def plot_reliability(
    intervals: pl.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Reliability: nominal vs empirical coverage at α ∈ {0.01..0.5}, per lead."""
    alphas = sorted(intervals["alpha"].unique().to_list())
    fig, ax = plt.subplots(figsize=(8, 7))
    leads = sorted(intervals["lead_step"].unique().to_list())
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(leads)))
    for ld, c in zip(leads, colors):
        emp = []
        for a in alphas:
            sub = intervals.filter((pl.col("lead_step") == ld)
                                    & (pl.col("alpha") == a))
            if len(sub) == 0:
                emp.append(np.nan)
                continue
            in_band = ((sub["truth"] >= sub["lower_cp"])
                       & (sub["truth"] <= sub["upper_cp"]))
            emp.append(float(in_band.mean()))
        ax.plot([1 - a for a in alphas], emp, marker="o", color=c,
                label=f"lead {ld} ({(ld+1)*15}s)")
    # Diagonal + binomial CI
    nom = np.linspace(0.01, 0.99, 50)
    ax.plot(nom, nom, "--", color=cp_palette["diagonal"], lw=1)
    n_typ = intervals.filter(pl.col("lead_step") == leads[0]).group_by("alpha").len()["len"].mean()
    if n_typ and not np.isnan(n_typ):
        ci_lo = [binomial_ci(int(n_typ), p)[0] for p in nom]
        ci_hi = [binomial_ci(int(n_typ), p)[1] for p in nom]
        ax.fill_between(nom, ci_lo, ci_hi, color=cp_palette["diagonal"], alpha=0.15,
                        label="95% binomial CI")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("CP reliability diagram (per lead time)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fname = output_dir / "groupC_reliability_diagram.png"
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return [fname]


def plot_pit_histogram(
    intervals: pl.DataFrame,
    output_dir: Path,
    n_bins: int = 20,
) -> list[Path]:
    """PIT-like histogram: rank of truth among samples (raw) vs within CP band (calibrated)."""
    # Approx PIT from CP intervals: u_calibrated = (truth - lower_cp) / (upper_cp - lower_cp)
    # for the α=0.1 band; well-calibrated → not concentrated at 0/1.
    sub = intervals.filter(pl.col("alpha") == 0.1)
    if len(sub) == 0:
        return []
    width = (sub["upper_cp"] - sub["lower_cp"]).to_numpy()
    width = np.where(width < 1e-6, np.nan, width)
    u_cp = ((sub["truth"] - sub["lower_cp"]).to_numpy()) / width
    # Uncalibrated raw PIT
    raw_w = (sub["upper_raw"] - sub["lower_raw"]).to_numpy()
    raw_w = np.where(raw_w < 1e-6, np.nan, raw_w)
    u_raw = ((sub["truth"] - sub["lower_raw"]).to_numpy()) / raw_w

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, u, label in zip(axes, [u_raw, u_cp], ["Uncalibrated", "CP-calibrated"]):
        u_clip = u[np.isfinite(u)]
        ax.hist(np.clip(u_clip, -0.5, 1.5), bins=n_bins, range=(-0.5, 1.5),
                color=cp_palette["forecast_mean"], alpha=0.7, edgecolor="black")
        # Reference: uniform on [0, 1]
        ideal = len(u_clip) / n_bins * (1.0 / 2.0)
        ax.axhline(ideal, color="red", linestyle="--", lw=1, label="Uniform target (in [0,1])")
        # Mark the [0, 1] valid range
        ax.axvline(0.0, color="green", lw=0.7); ax.axvline(1.0, color="green", lw=0.7)
        ax.set_xlabel("PIT (truth position within 90% band)")
        ax.set_ylabel("Count")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("PIT histogram — pre vs post conformal calibration (α=0.1)")
    fname = output_dir / "groupC_pit_histogram.png"
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return [fname]


def plot_sharpness_vs_coverage(
    intervals: pl.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Width vs coverage at each α: sharpness-vs-coverage curve."""
    alphas = sorted(intervals["alpha"].unique().to_list())
    fig, ax = plt.subplots(figsize=(8, 6))
    leads = sorted(intervals["lead_step"].unique().to_list())
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(leads)))
    for ld, c in zip(leads, colors):
        widths = []
        covs = []
        for a in alphas:
            sub = intervals.filter((pl.col("lead_step") == ld) & (pl.col("alpha") == a))
            if len(sub) == 0:
                continue
            w = float((sub["upper_cp"] - sub["lower_cp"]).mean())
            cov = float(((sub["truth"] >= sub["lower_cp"])
                          & (sub["truth"] <= sub["upper_cp"])).mean())
            widths.append(w)
            covs.append(cov)
        ax.plot(covs, widths, marker="o", color=c, label=f"lead {ld}")
    ax.set_xlabel("Empirical coverage")
    ax.set_ylabel("Mean band width (m/s)")
    ax.set_title("Sharpness vs coverage (lower-left = sharper at same coverage)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fname = output_dir / "groupC_sharpness_vs_coverage.png"
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return [fname]


# ---------- GROUP D: Per-stratum coverage diagnostics ----------

def plot_coverage_breakdowns(
    intervals: pl.DataFrame,
    truth_df: pl.DataFrame,
    output_dir: Path,
    alpha_focus: float = 0.1,
) -> list[Path]:
    """Coverage by lead, by turbine (heatmap), by regime."""
    out: list[Path] = []
    leads = sorted(intervals["lead_step"].unique().to_list())
    alphas = sorted(intervals["alpha"].unique().to_list())
    nominal = 1.0 - alpha_focus

    # By lead
    cov_by_lead = []
    for ld in leads:
        sub = intervals.filter((pl.col("lead_step") == ld)
                                & (pl.col("alpha") == alpha_focus))
        if len(sub) == 0:
            cov_by_lead.append(np.nan); continue
        cov_by_lead.append(float(((sub["truth"] >= sub["lower_cp"])
                                  & (sub["truth"] <= sub["upper_cp"])).mean()))
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar([str((ld + 1) * 15) + "s" for ld in leads], cov_by_lead,
                  color=cp_palette["cp_band"], alpha=0.8, edgecolor="black")
    ax.axhline(nominal, color="red", lw=1.5, linestyle="--", label=f"Nominal {nominal:.2f}")
    ax.set_ylabel("Empirical coverage")
    ax.set_xlabel("Lead time")
    ax.set_title(f"Coverage by lead time (α={alpha_focus}, {int(nominal*100)}% nominal)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for bar, c in zip(bars, cov_by_lead):
        if not np.isnan(c):
            ax.text(bar.get_x() + bar.get_width()/2, c + 0.01, f"{c:.3f}",
                    ha="center", fontsize=9)
    fname = output_dir / "groupD_coverage_by_lead.png"
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    out.append(fname)

    # By turbine — heatmap (alphas × turbine indices)
    target_ids = sorted(intervals["target_idx"].unique().to_list())
    grid = np.full((len(alphas), len(target_ids)), np.nan)
    for i, a in enumerate(alphas):
        for j, ti in enumerate(target_ids):
            sub = intervals.filter((pl.col("alpha") == a)
                                    & (pl.col("target_idx") == ti)
                                    & (pl.col("lead_step") == 0))
            if len(sub) == 0:
                continue
            grid[i, j] = float(((sub["truth"] >= sub["lower_cp"])
                                & (sub["truth"] <= sub["upper_cp"])).mean())
    fig, ax = plt.subplots(figsize=(min(20, max(8, len(target_ids) * 0.12)), 5))
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1.0)
    ax.set_yticks(range(len(alphas)))
    ax.set_yticklabels([f"α={a} ({int((1-a)*100)}%)" for a in alphas])
    ax.set_xlabel("Target index (horz then vert)")
    ax.set_title("Coverage heatmap (lead 0 only). Green=good, Red=bad.")
    fig.colorbar(im, ax=ax, label="Empirical coverage")
    fname = output_dir / "groupD_coverage_by_turbine.png"
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    out.append(fname)

    # By regime — robust polars-native version
    # Compute per-time regime by averaging std across many turbine targets, not just one.
    target_cols_in_truth = [c for c in truth_df.columns if c.startswith("target_")]
    if target_cols_in_truth:
        # Use the mean abs-deviation across all horz components as the regime signal
        horz_cols = [c for c in target_cols_in_truth
                     if int(c.split("_")[1]) < N_TURBINES]
        if not horz_cols:
            horz_cols = target_cols_in_truth[:N_TURBINES]
        truth_arr = truth_df.select(horz_cols).to_numpy()  # (n_time, n_turbines)
        # Per-timestep cross-turbine variability (proxy: rolling std of mean across turbines)
        global_signal = truth_arr.mean(axis=1)
        regime = regime_from_truth_window(global_signal, window_size=40)
        regime_df = pl.DataFrame({
            "time": truth_df["time"].to_numpy(),
            "regime": regime,
        })

        sub_all = (intervals
                   .filter((pl.col("alpha") == alpha_focus)
                           & (pl.col("lead_step") == 0))
                   .join(regime_df, on="time", how="inner"))
        if len(sub_all) > 0:
            sub_all = sub_all.with_columns(
                in_band=((pl.col("truth") >= pl.col("lower_cp"))
                         & (pl.col("truth") <= pl.col("upper_cp"))).cast(pl.Float64),
                width=(pl.col("upper_cp") - pl.col("lower_cp")),
            )
            agg = (sub_all.group_by("regime")
                   .agg(pl.col("in_band").mean().alias("coverage"),
                        pl.col("width").mean().alias("width"),
                        pl.len().alias("n")))
            regimes = ["calm", "transitional", "gusty"]
            agg_dict = {r["regime"]: r for r in agg.to_dicts()}
            covs_r = [agg_dict[r]["coverage"] if r in agg_dict else np.nan for r in regimes]
            ws_r = [agg_dict[r]["width"] if r in agg_dict else np.nan for r in regimes]
            ns_r = [agg_dict[r]["n"] if r in agg_dict else 0 for r in regimes]

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            colors = [cp_palette["regime_calm"], "#cccccc", cp_palette["regime_gusty"]]
            x = np.arange(len(regimes))
            axes[0].bar(x, [c if not np.isnan(c) else 0 for c in covs_r],
                        color=colors, edgecolor="black", width=0.6)
            axes[0].axhline(nominal, color="red", lw=1.5, ls="--",
                            label=f"Nominal {nominal:.2f}")
            axes[0].set_xticks(x)
            axes[0].set_xticklabels([f"{r}\n(n={ns_r[i]})" for i, r in enumerate(regimes)])
            axes[0].set_ylabel("Empirical coverage")
            axes[0].set_ylim(0, 1.05)
            axes[0].set_title(f"Coverage by regime (α={alpha_focus})")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3, axis="y")
            for i, c in enumerate(covs_r):
                if not np.isnan(c):
                    axes[0].text(i, c + 0.01, f"{c:.3f}", ha="center", fontsize=9)

            axes[1].bar(x, [w if not np.isnan(w) else 0 for w in ws_r],
                        color=colors, edgecolor="black", width=0.6)
            axes[1].set_xticks(x)
            axes[1].set_xticklabels([f"{r}\n(n={ns_r[i]})" for i, r in enumerate(regimes)])
            axes[1].set_ylabel("Mean band width (m/s)")
            axes[1].set_title(f"Band width by regime (α={alpha_focus})")
            axes[1].grid(True, alpha=0.3, axis="y")
            for i, w in enumerate(ws_r):
                if not np.isnan(w):
                    axes[1].text(i, w + (max(ws_r) * 0.02 if max(ws_r) > 0 else 0.01),
                                 f"{w:.3f}", ha="center", fontsize=9)
            fname = output_dir / "groupD_coverage_and_width_by_regime.png"
            fig.savefig(fname, dpi=120, bbox_inches="tight")
            plt.close(fig)
            out.append(fname)
    return out


# ---------- GROUP E: Residual diagnostics ----------

def plot_residual_diagnostics(
    intervals: pl.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Q-Q of residuals + per-turbine boxplot of residuals."""
    out: list[Path] = []
    sub = intervals.filter((pl.col("alpha") == 0.1) & (pl.col("lead_step") == 0))
    if len(sub) == 0:
        return out
    resid = (sub["truth"] - sub["forecast_mean"]).to_numpy()

    # Q-Q vs Gaussian
    from scipy.stats import probplot  # noqa: WPS433
    fig, ax = plt.subplots(figsize=(7, 7))
    probplot(resid, dist="norm", plot=ax)
    ax.set_title(f"Residual Q-Q plot — lead 0 (n={len(resid)})")
    ax.grid(True, alpha=0.3)
    fname = output_dir / "groupE_residual_qq.png"
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    out.append(fname)

    # Per-turbine boxplot
    target_ids = sorted(sub["target_idx"].unique().to_list())
    data = []
    labels = []
    for ti in target_ids[:40]:  # cap at 40 turbines for legibility
        rs = (sub.filter(pl.col("target_idx") == ti)["truth"]
              - sub.filter(pl.col("target_idx") == ti)["forecast_mean"]).to_numpy()
        if len(rs) > 0:
            data.append(rs)
            labels.append(str(ti))
    if data:
        fig, ax = plt.subplots(figsize=(min(20, max(10, len(data) * 0.4)), 6))
        ax.boxplot(data, tick_labels=labels, showfliers=False)
        ax.axhline(0, color="red", lw=1, ls="--")
        ax.set_xlabel("Target index")
        ax.set_ylabel("Residual (truth − forecast_mean) [m/s]")
        ax.set_title(f"Residual distribution per target (first {len(data)} shown)")
        ax.grid(True, alpha=0.3, axis="y")
        plt.xticks(rotation=45)
        fname = output_dir / "groupE_residual_per_target_box.png"
        fig.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close(fig)
        out.append(fname)

    return out


# ---------- GROUP F: Headline metrics table ----------

def plot_metrics_table(
    metrics_df: pl.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Rendered table summarizing CRPS, MAE, RMSE, coverage, width per lead."""
    out: list[Path] = []
    # Aggregate per lead. Use drop_nulls() before mean so NaN-rows (e.g. lead 0
    # persistence-skill is undefined) don't propagate.
    agg_cols = [c for c in metrics_df.columns if c.startswith("coverage_a")
                or c.startswith("width_a") or c.startswith("winkler_a")
                or c in ("mae", "rmse", "crps_raw", "skill_skill")]
    agg_cols = [c for c in agg_cols if c in metrics_df.columns]
    # Build aggregation excluding NaN per column to avoid NaN propagation
    agg = (metrics_df
           .group_by("lead_step")
           .agg([pl.col(c).drop_nans().mean().alias(c) for c in agg_cols])
           .sort("lead_step"))

    # Render as a simple text-table image
    fig, ax = plt.subplots(figsize=(max(8, 1.5 + 1.5 * len(agg.columns)), 0.5 + 0.4 * len(agg)))
    ax.axis("off")
    rows = []
    headers = list(agg.columns)
    for r in agg.iter_rows():
        rows.append([f"{v:.4f}" if isinstance(v, (int, float)) and not isinstance(v, bool)
                     else str(v) for v in r])
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    ax.set_title("Headline metrics per lead time (mean over targets)", fontsize=12, pad=15)
    fname = output_dir / "groupF_metrics_table.png"
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    out.append(fname)

    # CSV mirror
    agg.write_csv(output_dir / "groupF_metrics_table.csv")
    return out


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--intervals", type=Path, required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--split-id", type=int, default=194)
    ap.add_argument("--k-per-regime", type=int, default=5,
                    help="Turbines per regime (calm/transitional/gusty) for Group A/B")
    args = ap.parse_args()

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    plot_dir = output / "plots"
    plot_dir.mkdir(exist_ok=True)

    log.info("Loading intervals...")
    intervals = load_intervals(args.intervals)
    log.info("  rows=%d, alphas=%s, leads=%s",
             len(intervals),
             sorted(intervals["alpha"].unique().to_list()),
             sorted(intervals["lead_step"].unique().to_list()))

    log.info("Loading truth (SPLIT%d)...", args.split_id)
    truth_df = load_truth(args.truth, args.split_id)

    log.info("Loading metrics...")
    metrics_df = load_metrics(args.metrics)

    # Pick representative turbines
    target_ids = sorted(intervals["target_idx"].unique().to_list())
    selection = select_representative_turbines(truth_df, target_ids,
                                               map_target_to_wt_col,
                                               k_per_regime=args.k_per_regime)
    log.info("Selected %d turbines for visualization", len(selection))

    groups: dict[str, list[Path]] = {}

    # Group A
    log.info("Group A: time-series plots × 5 zooms × %d turbines...", len(selection))
    a_files: list[Path] = []
    for sel in selection:
        a_files += plot_timeseries_with_bands(intervals, sel, plot_dir)
    groups["A. Time-series with bands (5 zoom levels)"] = a_files

    # Group B (3 cherry-picked: one per regime)
    log.info("Group B: uncal vs CP side-by-side (3 representative)...")
    by_regime: dict[str, list] = defaultdict(list)
    for s in selection:
        by_regime[s.regime].append(s)
    b_files: list[Path] = []
    for r, picks in by_regime.items():
        if picks:
            b_files += plot_uncal_vs_cp(intervals, picks[0], plot_dir)
    groups["B. Side-by-side: uncalibrated vs CP-calibrated"] = b_files

    # Group C
    log.info("Group C: calibration diagnostics (reliability, PIT, sharpness)...")
    c_files = []
    c_files += plot_reliability(intervals, plot_dir)
    c_files += plot_pit_histogram(intervals, plot_dir)
    c_files += plot_sharpness_vs_coverage(intervals, plot_dir)
    groups["C. Calibration diagnostics"] = c_files

    # Group D
    log.info("Group D: per-stratum coverage breakdowns...")
    d_files = plot_coverage_breakdowns(intervals, truth_df, plot_dir)
    groups["D. Per-stratum coverage"] = d_files

    # Group E
    log.info("Group E: residual diagnostics...")
    e_files = plot_residual_diagnostics(intervals, plot_dir)
    groups["E. Residual diagnostics"] = e_files

    # Group F
    log.info("Group F: headline metrics table...")
    f_files = plot_metrics_table(metrics_df, plot_dir)
    groups["F. Headline metrics"] = f_files

    # Group G: index HTML
    log.info("Group G: index.html...")
    idx = build_index_html(output, groups)
    log.info("Index: %s", idx)
    n_total = sum(len(v) for v in groups.values())
    log.info("Done. Total plots: %d", n_total)


if __name__ == "__main__":
    main()
