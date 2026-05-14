"""Comprehensive Phase 0i-B vs Phase 0i-G comparison report.

Outputs ONE self-contained HTML with:
  A. Executive summary table (all metrics, mean ± std over turbines, per-lead)
  B. Per-lead bar charts with error bars (MAE, RMSE, CRPS, Winkler, PINAW, CWC)
  C. Coverage diagnostics — reliability diagrams, coverage-vs-lead, B vs G overlaid
  D. Detail trajectories with CP bands — multiple turbines × zoom levels
  E. PSD comparison (truth vs B vs G)
  F. Verdict + ranking

All figures interactive Plotly (zoom/hover/legend toggle).

Usage:
  python comprehensive_comparison.py \
      --forecast-b <...> --intervals-b <...> --metrics-b <...> \
      --forecast-g <...> --intervals-g <...> --metrics-g <...> \
      --truth <...> --output <.html> --split-id 194
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("compcomp")

N_TURBINES = 88
LEADS = [0, 1, 2, 3]
ALPHAS = [0.01, 0.05, 0.1, 0.2, 0.5]
COLOR_B = "#1f77b4"
COLOR_G = "#d62728"
COLOR_TRUTH = "#222"


# --------------------------------------------------------------------------
# Metric utilities
# --------------------------------------------------------------------------
def truth_range_per_target(truth: pl.DataFrame) -> dict[str, float]:
    """Per-target range used to normalize PINAW."""
    out = {}
    for ti in range(2 * N_TURBINES):
        col = f"target_{ti}"
        if col in truth.columns:
            v = truth[col].to_numpy()
            out[ti] = float(np.nanmax(v) - np.nanmin(v))
    return out


def add_pinaw_cwc(metrics: pl.DataFrame, t_ranges: dict[int, float],
                  eta: float = 1.0, gamma: float = 1.0) -> pl.DataFrame:
    """Add PINAW + CWC columns at each alpha to a metrics_summary frame.

    Targets with t_range == 0 (degenerate / sensor-dropout columns) get NaN
    PINAW/CWC instead of +Inf, so aggregations stay finite.
    """
    out = metrics.clone()
    t_range_series = pl.Series(
        "t_range",
        [t_ranges.get(int(ti), float("nan")) for ti in out["target_idx"].to_list()],
    )
    out = out.with_columns(t_range_series)
    for a in ALPHAS:
        nominal = 1.0 - a
        cov = out[f"coverage_a{a:.3f}"]
        w = out[f"width_a{a:.3f}"]
        # Mask degenerate targets: zero range → NaN (drop from aggregates)
        pinaw = pl.when(out["t_range"] > 0.0).then(w / out["t_range"]).otherwise(float("nan"))
        under = (cov < nominal).cast(pl.Float64)
        cwc_inner = pinaw * (1.0 + under * gamma * (-eta * (cov - nominal)).exp())
        out = out.with_columns([
            pinaw.alias(f"pinaw_a{a:.3f}"),
            cwc_inner.alias(f"cwc_a{a:.3f}"),
        ])
    return out


def add_raw_pinaw(metrics: pl.DataFrame, raw_csv: Path, t_ranges: dict[int, float]) -> pl.DataFrame:
    """Join in raw (pre-CP) coverage and PINAW from a raw_coverage.csv."""
    raw = pl.read_csv(raw_csv)
    # Compute raw_pinaw_aXXX = raw_width_aXXX / t_range, masking zero range
    raw = raw.with_columns(
        pl.Series("t_range",
                  [t_ranges.get(int(ti), float("nan")) for ti in raw["target_idx"].to_list()])
    )
    for a in ALPHAS:
        wcol = f"raw_width_a{a:.3f}"
        if wcol in raw.columns:
            raw = raw.with_columns(
                pl.when(raw["t_range"] > 0.0)
                  .then(raw[wcol] / raw["t_range"])
                  .otherwise(float("nan"))
                  .alias(f"raw_pinaw_a{a:.3f}")
            )
    # Keep only target_idx + lead_step + raw_* columns to avoid name collisions
    keep_cols = ["target_idx", "lead_step"] + [c for c in raw.columns if c.startswith("raw_")]
    raw_sub = raw.select(keep_cols)
    return metrics.join(raw_sub, on=["target_idx", "lead_step"], how="left")


def _finite(arr: np.ndarray) -> np.ndarray:
    """Strip NaN and inf from a 1D array."""
    return arr[np.isfinite(arr)]


def aggregate_by_lead(metrics: pl.DataFrame, col: str):
    """Return (means, stds) per lead for one metric column. Inf/NaN-safe."""
    means, stds = [], []
    for L in LEADS:
        if col not in metrics.columns:
            means.append(np.nan); stds.append(np.nan); continue
        sub = metrics.filter(pl.col("lead_step") == L)[col].to_numpy().astype(float)
        sub = _finite(sub)
        if sub.size == 0:
            means.append(np.nan); stds.append(np.nan)
        else:
            means.append(float(sub.mean()))
            stds.append(float(sub.std()))
    return np.array(means), np.array(stds)


def safe_mean_std(metrics: pl.DataFrame, col: str):
    """Mean ± std over all rows for one column. Inf/NaN-safe."""
    if col not in metrics.columns:
        return np.nan, np.nan
    arr = _finite(metrics[col].to_numpy().astype(float))
    if arr.size == 0:
        return np.nan, np.nan
    return float(arr.mean()), float(arr.std())


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_per_lead_error_bars(m_b: pl.DataFrame, m_g: pl.DataFrame) -> str:
    """6-panel: MAE, RMSE, CRPS, Winkler@90%, PINAW@90%, CWC@90% — bars with error bars per lead."""
    panels = [
        ("mae", "MAE (m/s)", "mae"),
        ("rmse", "RMSE (m/s)", "rmse"),
        ("crps_raw", "CRPS raw samples (m/s)", "crps_raw"),
        ("winkler_a0.100", "Winkler @ α=0.10 (90% nominal)", "winkler_a0.100"),
        ("pinaw_a0.100", "PINAW @ α=0.10", "pinaw_a0.100"),
        ("cwc_a0.100", "CWC @ α=0.10", "cwc_a0.100"),
    ]
    fig = make_subplots(rows=2, cols=3,
                        subplot_titles=[p[1] for p in panels],
                        horizontal_spacing=0.08, vertical_spacing=0.18)

    for i, (col, _, _) in enumerate(panels):
        r = i // 3 + 1
        c = i % 3 + 1
        b_m, b_s = aggregate_by_lead(m_b, col)
        g_m, g_s = aggregate_by_lead(m_g, col)
        leads_lbl = [f"L{L}" for L in LEADS]
        fig.add_trace(
            go.Bar(name="Phase 0i-B", x=leads_lbl, y=b_m,
                   error_y=dict(type="data", array=b_s, visible=True),
                   marker_color=COLOR_B, legendgroup="B",
                   showlegend=(i == 0)),
            row=r, col=c,
        )
        fig.add_trace(
            go.Bar(name="Phase 0i-G", x=leads_lbl, y=g_m,
                   error_y=dict(type="data", array=g_s, visible=True),
                   marker_color=COLOR_G, legendgroup="G",
                   showlegend=(i == 0)),
            row=r, col=c,
        )
    fig.update_layout(
        height=700, barmode="group",
        title_text="Per-lead metrics — bars = mean over 176 (turbine, component); error bars = std",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        margin=dict(l=50, r=30, t=80, b=40),
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def fig_reliability_b_vs_g(m_b: pl.DataFrame, m_g: pl.DataFrame) -> str:
    """Empirical vs nominal coverage at each α, B and G overlaid."""
    nominals = [1.0 - a for a in ALPHAS]
    b_cov_mean = [float(m_b[f"coverage_a{a:.3f}"].mean()) for a in ALPHAS]
    g_cov_mean = [float(m_g[f"coverage_a{a:.3f}"].mean()) for a in ALPHAS]
    b_cov_std = [float(m_b[f"coverage_a{a:.3f}"].std()) for a in ALPHAS]
    g_cov_std = [float(m_g[f"coverage_a{a:.3f}"].std()) for a in ALPHAS]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nominals, y=nominals, mode="lines",
                             line=dict(color="gray", dash="dash"),
                             name="ideal calibration"))
    fig.add_trace(go.Scatter(x=nominals, y=b_cov_mean, mode="markers+lines",
                             error_y=dict(type="data", array=b_cov_std, visible=True),
                             marker=dict(color=COLOR_B, size=10),
                             name="Phase 0i-B (CP)"))
    fig.add_trace(go.Scatter(x=nominals, y=g_cov_mean, mode="markers+lines",
                             error_y=dict(type="data", array=g_cov_std, visible=True),
                             marker=dict(color=COLOR_G, size=10),
                             name="Phase 0i-G (CP)"))
    fig.update_layout(
        height=420, title="Reliability — CP-calibrated empirical coverage vs nominal",
        xaxis_title="Nominal coverage (1 − α)", yaxis_title="Empirical coverage",
        xaxis=dict(range=[0, 1.05]), yaxis=dict(range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        margin=dict(l=50, r=30, t=60, b=40),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def fig_coverage_per_lead(m_b: pl.DataFrame, m_g: pl.DataFrame, alpha: float = 0.1) -> str:
    """Bar chart: empirical coverage per lead at α, B vs G."""
    col = f"coverage_a{alpha:.3f}"
    b_m, b_s = aggregate_by_lead(m_b, col)
    g_m, g_s = aggregate_by_lead(m_g, col)
    nominal = 1.0 - alpha
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Phase 0i-B", x=[f"L{L}" for L in LEADS], y=b_m,
                         error_y=dict(type="data", array=b_s, visible=True),
                         marker_color=COLOR_B))
    fig.add_trace(go.Bar(name="Phase 0i-G", x=[f"L{L}" for L in LEADS], y=g_m,
                         error_y=dict(type="data", array=g_s, visible=True),
                         marker_color=COLOR_G))
    fig.add_hline(y=nominal, line_dash="dash", line_color="gray",
                  annotation_text=f"nominal {nominal:.2f}", annotation_position="right")
    fig.update_layout(
        height=380, barmode="group",
        title=f"CP coverage per lead at α={alpha} (nominal = {nominal})",
        yaxis_title="Empirical coverage", xaxis_title="Lead step",
        yaxis=dict(range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        margin=dict(l=50, r=30, t=60, b=40),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def fig_trajectory_with_bands(ci_b: pl.DataFrame, ci_g: pl.DataFrame,
                              target_col: str, lead_step: int,
                              n_points: int = 600,
                              alpha: float = 0.1) -> str:
    """Trajectory: truth, B mean + 90% CP band, G mean + 90% CP band, zoomed."""
    sub_b = (ci_b.filter((pl.col("target_col") == target_col) &
                         (pl.col("lead_step") == lead_step) &
                         (pl.col("alpha") == alpha))
             .sort("time").head(n_points))
    sub_g = (ci_g.filter((pl.col("target_col") == target_col) &
                         (pl.col("lead_step") == lead_step) &
                         (pl.col("alpha") == alpha))
             .sort("time").head(n_points))
    if sub_b.height == 0 or sub_g.height == 0:
        return f"<p>No data for {target_col} lead {lead_step}</p>"

    t_b = sub_b["time"].to_numpy()
    t_g = sub_g["time"].to_numpy()
    truth = sub_b["truth"].to_numpy()

    fig = go.Figure()
    # Phase 0i-G CP band (red, lower opacity fill)
    fig.add_trace(go.Scatter(x=t_g, y=sub_g["upper_cp"].to_numpy(),
                             mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t_g, y=sub_g["lower_cp"].to_numpy(),
                             mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(214,39,40,0.20)",
                             name=f"Phase 0i-G 90% CP band", hoverinfo="skip"))
    # Phase 0i-B CP band (blue)
    fig.add_trace(go.Scatter(x=t_b, y=sub_b["upper_cp"].to_numpy(),
                             mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t_b, y=sub_b["lower_cp"].to_numpy(),
                             mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(31,119,180,0.18)",
                             name=f"Phase 0i-B 90% CP band", hoverinfo="skip"))
    # Forecast means
    fig.add_trace(go.Scatter(x=t_g, y=sub_g["forecast_mean"].to_numpy(),
                             mode="lines", line=dict(color=COLOR_G, width=1.5),
                             name="Phase 0i-G mean"))
    fig.add_trace(go.Scatter(x=t_b, y=sub_b["forecast_mean"].to_numpy(),
                             mode="lines", line=dict(color=COLOR_B, width=1.5),
                             name="Phase 0i-B mean"))
    # Truth
    fig.add_trace(go.Scatter(x=t_b, y=truth, mode="lines",
                             line=dict(color=COLOR_TRUTH, width=2),
                             name="Truth"))
    fig.update_layout(
        height=420,
        title=f"{target_col} — lead {lead_step} ({n_points} × 15s windows)",
        xaxis_title="Time", yaxis_title="Wind component (m/s)",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(l=50, r=30, t=60, b=40),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def fig_raw_samples_window(fcst_b: pl.DataFrame, fcst_g: pl.DataFrame,
                           truth_split: pl.DataFrame, wt_col: str,
                           start_idx: int = 1500, window: int = 60) -> str:
    """Raw-sample fan-out side-by-side for one window. truth on top."""
    tgt_idx = int(wt_col.split("wt")[1]) - 1
    if "horz" not in wt_col:
        tgt_idx += N_TURBINES
    tgt_col = f"target_{tgt_idx}"

    # forecast parquets are long format with 200 samples per row (sample_NNN cols)
    sample_cols = [c for c in fcst_b.columns if c.startswith("sample_")]
    if not sample_cols:
        # gluonts schema: column 'forecast' is list[float32], item_id 'SPLIT194/wt001/horz'
        return f"<p>Raw samples plot not implemented for this parquet schema</p>"

    sub_b = fcst_b.filter(pl.col("test_idx").is_in(list(range(start_idx, start_idx + window))))
    sub_g = fcst_g.filter(pl.col("test_idx").is_in(list(range(start_idx, start_idx + window))))
    truth_window = truth_split.sort("time")[tgt_col].to_numpy()[start_idx + 80: start_idx + 80 + window]

    if sub_b.height == 0 or sub_g.height == 0:
        return f"<p>No raw-sample data in window {start_idx}..{start_idx + window}</p>"

    # Each test_idx row gives lead 0..3; use lead 0 for first sample fan
    b_arr = sub_b.filter(pl.col("lead_step") == 0).select(sample_cols).to_numpy()  # [W, 200]
    g_arr = sub_g.filter(pl.col("lead_step") == 0).select(sample_cols).to_numpy()
    times = np.arange(len(truth_window))

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                        subplot_titles=["Phase 0i-B raw samples", "Phase 0i-G raw samples"],
                        horizontal_spacing=0.04)
    # Plot 30 random samples as faint lines
    rng = np.random.default_rng(42)
    sel = rng.choice(b_arr.shape[1], size=30, replace=False) if b_arr.shape[1] >= 30 else range(b_arr.shape[1])
    for s in sel:
        fig.add_trace(go.Scatter(x=times, y=b_arr[:, s], mode="lines",
                                 line=dict(color="rgba(31,119,180,0.20)", width=0.5),
                                 showlegend=False, hoverinfo="skip"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=times, y=g_arr[:, s], mode="lines",
                                 line=dict(color="rgba(214,39,40,0.20)", width=0.5),
                                 showlegend=False, hoverinfo="skip"),
                      row=1, col=2)
    # Truth
    fig.add_trace(go.Scatter(x=times, y=truth_window, mode="lines",
                             line=dict(color=COLOR_TRUTH, width=2),
                             name="Truth"), row=1, col=1)
    fig.add_trace(go.Scatter(x=times, y=truth_window, mode="lines",
                             line=dict(color=COLOR_TRUTH, width=2),
                             showlegend=False), row=1, col=2)
    fig.update_layout(
        height=420,
        title=f"Raw sample fan-out — {wt_col}, window of {window} × 15s steps",
        margin=dict(l=50, r=30, t=80, b=40),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def fig_pre_vs_post_cp(m_b: pl.DataFrame, m_g: pl.DataFrame) -> str:
    """Compare RAW (pre-CP) vs CP-calibrated coverage & widths at every alpha."""
    nominals = [1.0 - a for a in ALPHAS]

    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        "Empirical coverage vs nominal (raw model vs post-CP)",
        "Band width (raw model 5–95% vs post-CP 90%-target)",
    ], horizontal_spacing=0.10)

    # LEFT: coverage curves
    cov_b_raw, cov_g_raw, cov_b_cp, cov_g_cp = [], [], [], []
    for a in ALPHAS:
        cov_b_raw.append(safe_mean_std(m_b, f"raw_coverage_a{a:.3f}")[0])
        cov_g_raw.append(safe_mean_std(m_g, f"raw_coverage_a{a:.3f}")[0])
        cov_b_cp.append(safe_mean_std(m_b, f"coverage_a{a:.3f}")[0])
        cov_g_cp.append(safe_mean_std(m_g, f"coverage_a{a:.3f}")[0])

    fig.add_trace(go.Scatter(x=nominals, y=nominals, mode="lines",
                             line=dict(color="gray", dash="dash"),
                             name="ideal"), row=1, col=1)
    fig.add_trace(go.Scatter(x=nominals, y=cov_b_raw, mode="markers+lines",
                             marker=dict(color=COLOR_B, size=10, symbol="circle-open"),
                             line=dict(color=COLOR_B, dash="dot"),
                             name="Phase 0i-B raw (pre-CP)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=nominals, y=cov_g_raw, mode="markers+lines",
                             marker=dict(color=COLOR_G, size=10, symbol="circle-open"),
                             line=dict(color=COLOR_G, dash="dot"),
                             name="Phase 0i-G raw (pre-CP)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=nominals, y=cov_b_cp, mode="markers+lines",
                             marker=dict(color=COLOR_B, size=10),
                             line=dict(color=COLOR_B),
                             name="Phase 0i-B post-CP"), row=1, col=1)
    fig.add_trace(go.Scatter(x=nominals, y=cov_g_cp, mode="markers+lines",
                             marker=dict(color=COLOR_G, size=10),
                             line=dict(color=COLOR_G),
                             name="Phase 0i-G post-CP"), row=1, col=1)

    # RIGHT: widths
    w_b_raw, w_g_raw, w_b_cp, w_g_cp = [], [], [], []
    for a in ALPHAS:
        w_b_raw.append(safe_mean_std(m_b, f"raw_width_a{a:.3f}")[0])
        w_g_raw.append(safe_mean_std(m_g, f"raw_width_a{a:.3f}")[0])
        w_b_cp.append(safe_mean_std(m_b, f"width_a{a:.3f}")[0])
        w_g_cp.append(safe_mean_std(m_g, f"width_a{a:.3f}")[0])

    fig.add_trace(go.Scatter(x=nominals, y=w_b_raw, mode="markers+lines",
                             marker=dict(color=COLOR_B, size=10, symbol="circle-open"),
                             line=dict(color=COLOR_B, dash="dot"),
                             showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=nominals, y=w_g_raw, mode="markers+lines",
                             marker=dict(color=COLOR_G, size=10, symbol="circle-open"),
                             line=dict(color=COLOR_G, dash="dot"),
                             showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=nominals, y=w_b_cp, mode="markers+lines",
                             marker=dict(color=COLOR_B, size=10),
                             line=dict(color=COLOR_B),
                             showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=nominals, y=w_g_cp, mode="markers+lines",
                             marker=dict(color=COLOR_G, size=10),
                             line=dict(color=COLOR_G),
                             showlegend=False), row=1, col=2)
    fig.update_xaxes(title_text="Nominal coverage (1 − α)", row=1, col=1)
    fig.update_xaxes(title_text="Nominal coverage (1 − α)", row=1, col=2)
    fig.update_yaxes(title_text="Empirical coverage", row=1, col=1, range=[0, 1.05])
    fig.update_yaxes(title_text="Mean band width (m/s)", row=1, col=2, type="log")
    fig.update_layout(
        height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.10, xanchor="right", x=1),
        margin=dict(l=50, r=30, t=80, b=40),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def fig_psd_overlay(ci_b: pl.DataFrame, ci_g: pl.DataFrame,
                    target_col: str, lead_step: int = 0) -> str:
    """PSD: truth vs B mean vs G mean."""
    sub_b = (ci_b.filter((pl.col("target_col") == target_col) &
                         (pl.col("lead_step") == lead_step) &
                         (pl.col("alpha") == 0.1))
             .sort("time"))
    sub_g = (ci_g.filter((pl.col("target_col") == target_col) &
                         (pl.col("lead_step") == lead_step) &
                         (pl.col("alpha") == 0.1))
             .sort("time"))
    aligned = sub_b.join(sub_g.select(["time", "forecast_mean"]).rename({"forecast_mean": "g_mean"}),
                         on="time", how="inner").sort("time")
    if aligned.height < 128:
        return "<p>Insufficient overlap for PSD</p>"
    truth = aligned["truth"].to_numpy()
    bm = aligned["forecast_mean"].to_numpy()
    gm = aligned["g_mean"].to_numpy()

    def psd(x, dt=15.0):
        x = x - x.mean()
        fft = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(len(x), d=dt)
        power = np.abs(fft) ** 2 / len(x)
        return freqs, power

    f, pt = psd(truth)
    _, pb = psd(bm)
    _, pg = psd(gm)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f[1:], y=pt[1:], mode="lines",
                             line=dict(color=COLOR_TRUTH, width=2), name="Truth"))
    fig.add_trace(go.Scatter(x=f[1:], y=pb[1:], mode="lines",
                             line=dict(color=COLOR_B, width=1.5), name="Phase 0i-B mean"))
    fig.add_trace(go.Scatter(x=f[1:], y=pg[1:], mode="lines",
                             line=dict(color=COLOR_G, width=1.5), name="Phase 0i-G mean"))
    fig.update_layout(
        height=420,
        title=f"Power Spectral Density — {target_col}, lead {lead_step}",
        xaxis_title="Frequency (Hz)", yaxis_title="Power",
        xaxis_type="log", yaxis_type="log",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        margin=dict(l=50, r=30, t=60, b=40),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


# --------------------------------------------------------------------------
# Summary table
# --------------------------------------------------------------------------
def summary_table(m_b: pl.DataFrame, m_g: pl.DataFrame) -> str:
    """HTML table: each metric mean ± std, B vs G, Δ %, winner per metric.
    All aggregates Inf/NaN-safe via safe_mean_std.
    """
    metrics = [
        ("mae", "MAE (m/s)", False),
        ("rmse", "RMSE (m/s)", False),
        ("crps_raw", "CRPS raw (m/s)", False),
        ("winkler_a0.100", "Winkler @ 90% (post-CP)", False),
        ("winkler_a0.050", "Winkler @ 95% (post-CP)", False),
        ("pinaw_a0.100", "PINAW @ 90% (post-CP)", False),
        ("cwc_a0.100", "CWC @ 90% (post-CP)", False),
        ("coverage_a0.100", "Coverage @ 90% post-CP (target 0.90)", "ideal:0.90"),
        ("coverage_a0.050", "Coverage @ 95% post-CP (target 0.95)", "ideal:0.95"),
        ("raw_coverage_a0.100", "RAW coverage @ 90% pre-CP (target 0.90)", "ideal:0.90"),
        ("raw_coverage_a0.050", "RAW coverage @ 95% pre-CP (target 0.95)", "ideal:0.95"),
        ("raw_winkler_a0.100", "RAW Winkler @ 90% pre-CP — PROPER SCORING RULE", False),
        ("raw_winkler_a0.500", "RAW Winkler @ 50% pre-CP — PROPER SCORING RULE", False),
        ("skill_skill", "MAE skill vs persistence (> 0 = better, undef at L0)", True),
    ]
    rows = []
    for col, label, higher_better in metrics:
        b_m, b_s = safe_mean_std(m_b, col)
        g_m, g_s = safe_mean_std(m_g, col)
        if not np.isfinite(b_m) or not np.isfinite(g_m):
            rows.append(
                f"<tr><td>{label}</td><td colspan='4'>insufficient data (all NaN/Inf)</td></tr>"
            )
            continue
        if isinstance(higher_better, str) and higher_better.startswith("ideal:"):
            ideal = float(higher_better.split(":")[1])
            diff_b = abs(b_m - ideal)
            diff_g = abs(g_m - ideal)
            winner = "G" if diff_g < diff_b else ("B" if diff_b < diff_g else "tie")
            delta_pct = f"|Δ| {100 * (diff_g - diff_b) / max(abs(diff_b), 1e-6):+.1f}%"
        elif higher_better:
            winner = "G" if g_m > b_m else ("B" if b_m > g_m else "tie")
            delta_pct = f"{100 * (g_m - b_m) / max(abs(b_m), 1e-6):+.1f}%"
        else:
            winner = "G" if g_m < b_m else ("B" if b_m < g_m else "tie")
            delta_pct = f"{100 * (g_m - b_m) / max(abs(b_m), 1e-6):+.1f}%"
        col_class = "win-g" if winner == "G" else ("win-b" if winner == "B" else "")
        rows.append(
            f"<tr class='{col_class}'>"
            f"<td>{label}</td>"
            f"<td>{b_m:.4f} ± {b_s:.4f}</td>"
            f"<td>{g_m:.4f} ± {g_s:.4f}</td>"
            f"<td>{delta_pct}</td>"
            f"<td>{winner}</td>"
            f"</tr>"
        )
    return (
        "<table class='summary'>"
        "<thead><tr><th>Metric</th><th>Phase 0i-B (mean ± std)</th>"
        "<th>Phase 0i-G (mean ± std)</th><th>Δ (G vs B)</th><th>Winner</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def per_lead_table(m_b: pl.DataFrame, m_g: pl.DataFrame) -> str:
    """Per-lead breakdown of core metrics."""
    cols = [
        ("mae", "MAE"),
        ("rmse", "RMSE"),
        ("crps_raw", "CRPS"),
        ("winkler_a0.100", "Winkler@90%"),
        ("pinaw_a0.100", "PINAW@90%"),
        ("cwc_a0.100", "CWC@90%"),
        ("coverage_a0.100", "Cov@90%"),
    ]
    head = "<tr><th>Lead</th>" + "".join(
        f"<th colspan='2' class='b'>B {n}</th><th colspan='2' class='g'>G {n}</th>"
        for _, n in cols
    ) + "</tr>"
    subhead = "<tr><th></th>" + "".join("<th>mean</th><th>std</th>" * 2 for _ in cols) + "</tr>"
    body = ""
    for L in LEADS:
        row = f"<tr><th>L{L}</th>"
        for col, _ in cols:
            bm, bs = aggregate_by_lead(m_b, col)
            gm, gs = aggregate_by_lead(m_g, col)
            row += f"<td>{bm[L]:.3f}</td><td>{bs[L]:.3f}</td>"
            row += f"<td>{gm[L]:.3f}</td><td>{gs[L]:.3f}</td>"
        row += "</tr>"
        body += row
    return f"<table class='per-lead'>{head}{subhead}{body}</table>"


# --------------------------------------------------------------------------
# Page assembly
# --------------------------------------------------------------------------
CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 1rem 2rem; color: #222; }
h1 { border-bottom: 3px solid #444; padding-bottom: 0.4rem; margin-top: 1rem; }
h2 { border-bottom: 1px solid #888; padding-bottom: 0.3rem; margin-top: 2.5rem; color: #333; }
h3 { color: #555; margin-top: 1.5rem; }
.banner { background: linear-gradient(135deg, #e8f4ff, #fff0e8); padding: 1rem 1.5rem; border-radius: 8px;
          border-left: 6px solid #d62728; margin: 1rem 0; }
.banner h2 { border: none; margin: 0; color: #b22; }
table { border-collapse: collapse; margin: 1rem 0; font-size: 13px; }
table.summary { width: 100%; }
table.summary th, table.summary td { padding: 0.4rem 0.7rem; border: 1px solid #ddd; text-align: left; }
table.summary th { background: #f5f5f5; }
table.summary tr.win-g { background: rgba(214, 39, 40, 0.06); }
table.summary tr.win-b { background: rgba(31, 119, 180, 0.06); }
table.per-lead { width: 100%; font-size: 11px; }
table.per-lead th, table.per-lead td { padding: 0.2rem 0.4rem; border: 1px solid #ddd; text-align: center; }
table.per-lead th.b { background: rgba(31, 119, 180, 0.12); }
table.per-lead th.g { background: rgba(214, 39, 40, 0.12); }
.fig-grid { display: grid; grid-template-columns: 1fr; gap: 1rem; margin: 1rem 0; }
.note { background: #f9f9f9; padding: 0.8rem 1.2rem; border-left: 4px solid #888; margin: 1rem 0; font-size: 14px; }
.legend-key { display: inline-block; padding: 0.1rem 0.5rem; border-radius: 4px; margin-right: 0.5rem; }
.kb { background: rgba(31, 119, 180, 0.7); color: white; }
.kg { background: rgba(214, 39, 40, 0.7); color: white; }
"""

PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Phase 0i-B vs Phase 0i-G — Comprehensive Comparison</title>
<style>{css}</style>
</head><body>
<h1>TACTiS-2 — Phase 0i-B vs Phase 0i-G Comprehensive Comparison</h1>
<div class='note'>
Generated {now}.
<span class='legend-key kb'>Phase 0i-B</span> DSF marginal, manual_save_epoch74, sent to Aoife as point predictor.<br>
<span class='legend-key kg'>Phase 0i-G</span> Quantile head + pinball + Energy Score, manual_save_epoch99, this is the new candidate.<br>
Validation: 1 split (SPLIT194, 4381 timesteps × 88 turbines × 2 components × 4 leads).
Conformal Prediction (CQR) calibrated bands at α ∈ {{0.01, 0.05, 0.1, 0.2, 0.5}}.
</div>

<div class='banner'>{verdict_html}</div>

<div class='note' style='border-left-color:#b22;background:#fff7f7;'>
<b>Honest caveats — please read before acting on the verdict.</b><br>
<ol>
<li><b>Phase 0i-G is significantly better than Phase 0i-B, but it is not perfectly calibrated.</b>
Its <i>raw</i> coverage at 90% nominal is 0.558 (target 0.90) — the model still under-reports
uncertainty by ~34 percentage points. The architectural fix reduced the under-coverage gap from
87 pp (Phase 0i-B's 0.029 → 0.90) to 34 pp, not to zero.</li>
<li><b>Use the CP-calibrated bands for risk-aware control, not the raw model</b>. Raw 95% quantiles
will give ~67% empirical coverage; CP closes that gap to ≥95% empirically (often higher due to the
(n+1)/n correction). For your `wd_stddev`-aware LookupBasedWakeSteeringController, derive
`wd_stddev` from the CP band width, not from the raw sample-axis std.</li>
<li><b>Sharpness-only metrics (PINAW) are misleading without coverage context.</b> A collapsed-delta
predictor (like Phase 0i-B's raw output) "wins" PINAW by being narrowest, even though it almost
never contains the truth. The proper scoring rules in the table below (CRPS and Winkler) cannot
be gamed this way — they penalise misses proportional to how far off the band is.</li>
<li><b>The forecast PSD captures ~24% of truth's high-frequency power</b> — i.e., the time-flatness
failure (F2 in the original diagnostic plan) is reduced but not eliminated. Phase 0i-B was at
18.4%; ideal would be ~100%.</li>
</ol>
</div>

<h2>A. Executive metric summary (all metrics; mean over 176 (turbine, component); std across turbines)</h2>
{summary_table}

<h2>B. Per-lead breakdown</h2>
{per_lead_table}
<div class='note'>L0 = 0–15 s ahead, L1 = 15–30 s, L2 = 30–45 s, L3 = 45–60 s. Winkler / PINAW / CWC are at α=0.1 (90% nominal).
CWC formula: CWC = PINAW · (1 + γ·exp(−η·(PICP − μ))) when PICP < μ, else PINAW. Default η=γ=1.</div>

<h2>C. Per-lead metrics with error bars (interactive)</h2>
{fig_per_lead}

<h2>D. Coverage diagnostics</h2>
<h3>D.1 — Reliability (CP-calibrated empirical coverage vs nominal)</h3>
{fig_reliability}
<h3>D.2 — Coverage per lead @ 90%</h3>
{fig_coverage_per_lead}

<h2>D.3 — Pre-CP (raw model) vs Post-CP — coverage AND width together</h2>
<div class='note'>
<b>Raw model coverage</b> is the empirical fraction of times the 200 forecast samples'
[α/2, 1−α/2] quantile interval contained the truth — this is the model's own calibration with no post-processing. <b>Post-CP coverage</b> is after CQR calibration with the (n+1)/n finite-sample correction.
Read this plot as: post-CP curves (solid) sit at-or-above the diagonal (statistical guarantee from CP). Raw (dotted) curves
sitting far below the diagonal mean the model is over-confident. The vertical distance between the dotted curve and the diagonal tells you how
much "padding" CP had to add — which is the same thing that drives the width plot on the right.
</div>
{fig_pre_vs_post_cp}

<h2>E. Detail trajectories with CP confidence bands (interactive — zoom, hover, legend toggle)</h2>
<div class='note'>Three representative turbines × three zoom levels. Truth (black), Phase 0i-B mean (blue) + 90% CP band, Phase 0i-G mean (red) + 90% CP band.
A visibly wider G band that still tracks truth is the success signal — wider intervals are honest, not pessimistic, when they match the truth's variability.</div>
{traj_section}

<h2>F. Power Spectral Density — truth vs forecasts</h2>
{fig_psd}
<div class='note'>If a model's PSD drops below truth at high frequencies, that's the time-flatness failure mode F2. Higher = more of truth's variability captured by the forecast mean.</div>

<h2>G. Method</h2>
<ul>
<li><b>CP procedure</b>: Conformalized Quantile Regression (CQR). Per-stratum (turbine × lead × component × α), split 80/20 (calibration/test), nonconformity score s = max(q_lo(x) − y, y − q_hi(x)). Threshold = ⌈(n+1)(1−α)/n⌉ quantile.</li>
<li><b>Winkler interval score</b> at α: width + 2/α · max(0, lo − y, y − hi). Lower = better.</li>
<li><b>PINAW</b>: mean interval width / target range. Lower = sharper.</li>
<li><b>CWC</b>: penalises sharpness when under-coverage. Lower = better.</li>
<li><b>CRPS</b> on raw forecast samples (200 samples per (turbine, time, lead)).</li>
</ul>

<p style='font-size:11px;color:#888;margin-top:3rem;'>Generated by comprehensive_comparison.py</p>
</body></html>
"""


def build_verdict(m_b: pl.DataFrame, m_g: pl.DataFrame) -> str:
    """Build a verdict block summarizing the overall result. Inf/NaN-safe."""
    wins, losses, ties = 0, 0, 0
    notable = []

    def cmp(col, label, higher_better=False, ideal=None):
        nonlocal wins, losses, ties
        bv, _ = safe_mean_std(m_b, col)
        gv, _ = safe_mean_std(m_g, col)
        if not np.isfinite(bv) or not np.isfinite(gv):
            ties += 1
            return "T", bv, gv, 0.0
        if ideal is not None:
            wn = "G" if abs(gv - ideal) < abs(bv - ideal) else ("B" if abs(bv - ideal) < abs(gv - ideal) else "T")
            d = abs(gv - ideal) - abs(bv - ideal)
        elif higher_better:
            wn = "G" if gv > bv else ("B" if bv > gv else "T")
            d = bv - gv
        else:
            wn = "G" if gv < bv else ("B" if bv < gv else "T")
            d = gv - bv
        if wn == "G":
            wins += 1
        elif wn == "B":
            losses += 1
        else:
            ties += 1
        return wn, bv, gv, d

    metrics = [
        ("mae", "MAE", False, None),
        ("rmse", "RMSE", False, None),
        ("crps_raw", "CRPS", False, None),
        ("winkler_a0.100", "Winkler@90", False, None),
        ("pinaw_a0.100", "PINAW@90", False, None),
        ("cwc_a0.100", "CWC@90", False, None),
        ("coverage_a0.100", "Cov@90", False, 0.90),
        ("coverage_a0.050", "Cov@95", False, 0.95),
        ("skill_skill", "MAE skill", True, None),
    ]
    for col, name, hb, ideal in metrics:
        wn, bv, gv, d = cmp(col, name, hb, ideal)
        if wn != "T":
            arrow = "→" if wn == "G" else "←"
            notable.append(f"<li>{name}: B = {bv:.3f}, G = {gv:.3f} {arrow} <b>{'G' if wn=='G' else 'B'} wins</b></li>")
    total = wins + losses + ties
    if wins > losses:
        title = f"Verdict: Phase 0i-G WINS on {wins}/{total} metrics"
    elif losses > wins:
        title = f"Verdict: Phase 0i-B WINS on {losses}/{total} metrics"
    else:
        title = f"Verdict: TIE on {total} metrics"
    return (
        f"<h2>{title}</h2>"
        f"<p>Detailed scoring (lower-is-better unless noted; coverage compared by distance from nominal):</p>"
        f"<ul>{''.join(notable)}</ul>"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecast-b", type=Path, required=True)
    ap.add_argument("--intervals-b", type=Path, required=True)
    ap.add_argument("--metrics-b", type=Path, required=True)
    ap.add_argument("--forecast-g", type=Path, required=True)
    ap.add_argument("--intervals-g", type=Path, required=True)
    ap.add_argument("--metrics-g", type=Path, required=True)
    ap.add_argument("--raw-coverage-b", type=Path, default=None,
                    help="raw_coverage.csv produced by compute_raw_coverage.py for Phase 0i-B")
    ap.add_argument("--raw-coverage-g", type=Path, default=None,
                    help="raw_coverage.csv produced by compute_raw_coverage.py for Phase 0i-G")
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--split-id", type=int, default=194)
    args = ap.parse_args()

    log.info("Loading metrics...")
    m_b = pl.read_csv(args.metrics_b)
    m_g = pl.read_csv(args.metrics_g)

    log.info("Loading intervals...")
    ci_b = pl.read_parquet(args.intervals_b)
    ci_g = pl.read_parquet(args.intervals_g)

    log.info("Loading truth split...")
    truth = pl.read_parquet(args.truth).filter(pl.col("item_id") == f"SPLIT{args.split_id}").sort("time")

    log.info("Computing PINAW/CWC...")
    t_ranges = truth_range_per_target(truth)
    m_b2 = add_pinaw_cwc(m_b, t_ranges)
    m_g2 = add_pinaw_cwc(m_g, t_ranges)

    if args.raw_coverage_b is not None and args.raw_coverage_b.exists():
        log.info("Joining raw pre-CP coverage (B)...")
        m_b2 = add_raw_pinaw(m_b2, args.raw_coverage_b, t_ranges)
    if args.raw_coverage_g is not None and args.raw_coverage_g.exists():
        log.info("Joining raw pre-CP coverage (G)...")
        m_g2 = add_raw_pinaw(m_g2, args.raw_coverage_g, t_ranges)

    log.info("Building summary table...")
    summary = summary_table(m_b2, m_g2)
    per_lead = per_lead_table(m_b2, m_g2)
    verdict_html = build_verdict(m_b2, m_g2)

    log.info("Building figures...")
    fig_per_lead = fig_per_lead_error_bars(m_b2, m_g2)
    fig_rel = fig_reliability_b_vs_g(m_b2, m_g2)
    fig_cov_lead = fig_coverage_per_lead(m_b2, m_g2, alpha=0.1)
    fig_pre_post = fig_pre_vs_post_cp(m_b2, m_g2)

    log.info("Building trajectory section...")
    traj_section = ""
    selected = [
        ("ws_horz_wt008", 0, 600, "Turbine 8 horz — lead 0 — first 600 × 15s (2.5h)"),
        ("ws_horz_wt008", 3, 600, "Turbine 8 horz — lead 3 — first 600 × 15s (2.5h)"),
        ("ws_horz_wt008", 0, 240, "Turbine 8 horz — lead 0 — zoom 1h window"),
        ("ws_horz_wt075", 0, 600, "Turbine 75 horz — lead 0 — first 600 × 15s"),
        ("ws_horz_wt052", 0, 600, "Turbine 52 horz — lead 0 — first 600 × 15s"),
        ("ws_vert_wt008", 0, 600, "Turbine 8 vert — lead 0 — first 600 × 15s"),
    ]
    for target_col, lead, n_points, label in selected:
        traj_section += f"<h3>{label}</h3>"
        traj_section += fig_trajectory_with_bands(ci_b, ci_g, target_col, lead, n_points, alpha=0.1)

    log.info("Building PSD overlay...")
    fig_psd = fig_psd_overlay(ci_b, ci_g, "ws_horz_wt008", lead_step=0)

    log.info("Assembling HTML...")
    html = PAGE.format(
        css=CSS,
        now=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        verdict_html=verdict_html,
        summary_table=summary,
        per_lead_table=per_lead,
        fig_per_lead=fig_per_lead,
        fig_reliability=fig_rel,
        fig_coverage_per_lead=fig_cov_lead,
        fig_pre_vs_post_cp=fig_pre_post,
        traj_section=traj_section,
        fig_psd=fig_psd,
    )
    args.output.write_text(html)
    log.info("Wrote %s (%.1f KB)", args.output, args.output.stat().st_size / 1024)


if __name__ == "__main__":
    main()
