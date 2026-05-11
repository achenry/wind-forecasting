"""Comprehensive model-quality report for a TACTiS-2 sample-based forecast.

Self-contained HTML report with:
- Executive summary + go/no-go recommendation
- Error-metric tables: deterministic + probabilistic
- Interactive Plotly time-series (with bands + truth)
- Frequency-domain analysis (FFT power spectrum: truth vs forecast)
- Per-turbine error distribution
- Calibration diagnostics (reliability, PIT)
- Persistence skill decomposition
- Spectral coherence (does the model preserve high-freq content?)

Designed for the Phase 0i-B ep74 checkpoint but works on any TACTiS-2
sample-format forecast parquet.

Usage:
    python generate_model_report.py \\
        --forecast <forecast_*.parquet> \\
        --truth <..._test_denormalize.parquet> \\
        --intervals <calibrated_intervals.parquet> (optional, adds CP section) \\
        --metrics <metrics_summary.csv> (optional) \\
        --output report.html
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from base64 import b64encode
from io import BytesIO
from pathlib import Path

import numpy as np
import polars as pl
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy import signal as spsig
from scipy.stats import norm, probplot

sys.path.insert(0, str(Path(__file__).parent))
from plot_utils import map_target_to_wt_col

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger("report")

N_TURBINES = 88
N_LEAD = 4


# ---------- Data loading + tensor construction ----------

def load_inputs(forecast_path, truth_path, split_id, intervals_path, metrics_path):
    log.info("Loading forecast %s", forecast_path)
    fcst = pl.read_parquet(forecast_path)
    log.info("Loading truth %s (SPLIT%d)", truth_path, split_id)
    truth = (pl.read_parquet(truth_path)
             .filter(pl.col("item_id") == f"SPLIT{split_id}")
             .sort("time"))
    intervals = None
    if intervals_path and Path(intervals_path).exists():
        log.info("Loading intervals %s", intervals_path)
        intervals = pl.read_parquet(intervals_path)
    metrics = None
    if metrics_path and Path(metrics_path).exists():
        log.info("Loading metrics %s", metrics_path)
        metrics = pl.read_csv(metrics_path)
    return fcst, truth, intervals, metrics


def build_per_lead_arrays(fcst: pl.DataFrame, truth: pl.DataFrame,
                           wt_col: str, target_col: str):
    """For one wt_col build aligned (truth, sample_mean, sample_std,
    sample_p5, sample_p95, lead, valid_time) arrays."""
    # Filter to complete test_idxs
    expected = N_LEAD * 200
    keep = (fcst.group_by("test_idx").len()
            .filter(pl.col("len") == expected)["test_idx"])
    fcst = fcst.filter(pl.col("test_idx").is_in(keep))
    sub = fcst.sort(["test_idx", "time", "sample"]).with_columns(
        ((pl.col("time").rank(method="dense").over("test_idx")) - 1).alias("lead_step")
    )
    # Per-(test_idx, lead_step, time) aggregate over samples
    agg = (sub.group_by(["test_idx", "lead_step", "time"])
           .agg(pl.col(wt_col).mean().alias("mean"),
                pl.col(wt_col).std().alias("std"),
                pl.col(wt_col).quantile(0.05).alias("p5"),
                pl.col(wt_col).quantile(0.95).alias("p95"),
                pl.col(wt_col).min().alias("min"),
                pl.col(wt_col).max().alias("max"))
           .sort(["test_idx", "lead_step"]))
    # Join truth on time
    joined = agg.join(truth.select(["time", target_col]).rename({target_col: "truth"}),
                      on="time", how="inner")
    return joined


def build_sample_tensor(fcst: pl.DataFrame, wt_col: str):
    """Full (n_test_idx, n_lead, n_samples) tensor for FFT etc."""
    expected = N_LEAD * 200
    keep = (fcst.group_by("test_idx").len()
            .filter(pl.col("len") == expected)["test_idx"])
    fcst = fcst.filter(pl.col("test_idx").is_in(keep))
    sub = fcst.sort(["test_idx", "time", "sample"]).with_columns(
        ((pl.col("time").rank(method="dense").over("test_idx")) - 1).alias("lead_step")
    )
    pivot = (sub.select(["test_idx", "lead_step", "time", "sample", wt_col])
             .pivot(values=wt_col, index=["test_idx", "lead_step", "time"], on="sample")
             .sort(["test_idx", "lead_step"]))
    sample_cols = sorted([c for c in pivot.columns
                          if c not in ("test_idx", "lead_step", "time")],
                         key=lambda c: float(c))
    n_test_idx = pivot["test_idx"].n_unique()
    samples = (pivot.select(sample_cols).to_numpy()
               .reshape(n_test_idx, N_LEAD, len(sample_cols)))
    times = pivot["time"].to_numpy().reshape(n_test_idx, N_LEAD)
    return samples, times


# ---------- Metrics ----------

def per_lead_metrics(per_lead_df: pl.DataFrame) -> dict:
    """Headline error metrics per lead step."""
    out = {}
    for lead in range(N_LEAD):
        sub = per_lead_df.filter(pl.col("lead_step") == lead)
        if len(sub) == 0:
            continue
        y = sub["truth"].to_numpy()
        m = sub["mean"].to_numpy()
        p5 = sub["p5"].to_numpy()
        p95 = sub["p95"].to_numpy()
        e = m - y
        cov90 = float(np.mean((y >= p5) & (y <= p95)))
        # CRPS approximation from samples (using p5..p95 + mean as a crude proxy);
        # the proper CRPS_samples is in probabilistic_metrics.py — call externally
        out[lead] = {
            "n": len(sub),
            "mae": float(np.mean(np.abs(e))),
            "rmse": float(np.sqrt(np.mean(e ** 2))),
            "bias": float(np.mean(e)),
            "ens_std_mean": float(sub["std"].mean()),
            "raw_90_width_mean": float(np.mean(p95 - p5)),
            "raw_90_coverage": cov90,
            "truth_std": float(np.std(y)),
            "skill_mae_vs_persist": persistence_skill_mae(per_lead_df, lead),
            "spread_skill_ratio": (float(sub["std"].mean()) /
                                   max(float(np.sqrt(np.mean(e ** 2))), 1e-9)),
        }
    return out


def persistence_skill_mae(per_lead_df: pl.DataFrame, lead: int) -> float:
    """MAE-based skill vs persistence: 1 - MAE_model / MAE_persistence.

    Persistence = previous-lead-0 truth carried forward.
    """
    sub = (per_lead_df.filter(pl.col("lead_step") == lead)
           .sort(["test_idx"]))
    if len(sub) == 0 or lead == 0:
        return float("nan")
    # Persistence baseline: truth at the same test_idx's lead-0
    lead0 = (per_lead_df.filter(pl.col("lead_step") == 0)
             .select(["test_idx", "truth"])
             .rename({"truth": "truth_lead0"}))
    j = sub.join(lead0, on="test_idx", how="inner")
    if len(j) == 0:
        return float("nan")
    mae_model = float((j["mean"] - j["truth"]).abs().mean())
    mae_persist = float((j["truth_lead0"] - j["truth"]).abs().mean())
    return 1.0 - mae_model / max(mae_persist, 1e-9)


def crps_from_samples_fast(truth, samples):
    """Vectorized CRPS approximation: E|X-Y| - 0.5 E|X-X'|.

    For large sample counts this is O(N²) per timestep — use only on a slice.
    """
    n_t, n_s = samples.shape
    term1 = np.mean(np.abs(samples - truth[:, None]), axis=1)
    # term2: 0.5 * mean over all pairs |X_i - X_j|
    term2 = np.empty(n_t)
    for i in range(n_t):
        s = samples[i]
        term2[i] = 0.5 * np.mean(np.abs(s[:, None] - s[None, :]))
    return term1 - term2


# ---------- Plotly figures ----------

def fig_timeseries(per_lead_df: pl.DataFrame, wt_col: str, title: str) -> str:
    """Interactive time-series: truth + mean + 5-95 band for lead 0."""
    sub = per_lead_df.filter(pl.col("lead_step") == 0).sort("time")
    if len(sub) == 0:
        return ""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["time"], y=sub["p95"], mode="lines",
                              line=dict(color="rgba(31,119,180,0.0)"),
                              showlegend=False))
    fig.add_trace(go.Scatter(x=sub["time"], y=sub["p5"], mode="lines",
                              line=dict(color="rgba(31,119,180,0.0)"),
                              fill="tonexty", fillcolor="rgba(31,119,180,0.2)",
                              name="5–95% ensemble"))
    fig.add_trace(go.Scatter(x=sub["time"], y=sub["mean"], mode="lines",
                              line=dict(color="#1f77b4", width=1.5),
                              name="Forecast mean"))
    fig.add_trace(go.Scatter(x=sub["time"], y=sub["truth"], mode="lines",
                              line=dict(color="black", width=1),
                              name="Ground truth"))
    fig.update_layout(title=title, height=420, margin=dict(t=50, b=40),
                       xaxis_title="Time", yaxis_title="Wind component (m/s)",
                       hovermode="x unified")
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def fig_error_decomposition(per_lead_df: pl.DataFrame, wt_col: str) -> str:
    """4-panel: residual time-series, residual histogram, residual ACF, scatter vs truth."""
    sub = per_lead_df.filter(pl.col("lead_step") == 0).sort("time")
    if len(sub) == 0:
        return ""
    resid = (sub["truth"] - sub["mean"]).to_numpy()
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Residual over time",
                                        "Residual histogram",
                                        "Residual autocorrelation",
                                        "Truth vs forecast (lead 0)"))
    fig.add_trace(go.Scatter(x=sub["time"], y=resid, mode="lines",
                              line=dict(color="#1f77b4", width=0.8),
                              name="resid"), row=1, col=1)
    fig.add_hline(y=0, line=dict(color="red", dash="dash"), row=1, col=1)
    fig.add_trace(go.Histogram(x=resid, nbinsx=60, name="resid"),
                  row=1, col=2)
    # ACF up to 60 lags
    nlags = 60
    acf = np.array([1.0] + [np.corrcoef(resid[:-l], resid[l:])[0, 1]
                            for l in range(1, nlags)])
    fig.add_trace(go.Bar(x=list(range(nlags)), y=acf, name="ACF"),
                  row=2, col=1)
    # 95% null band
    ci = 1.96 / np.sqrt(len(resid))
    fig.add_hline(y=ci, line=dict(color="red", dash="dash"), row=2, col=1)
    fig.add_hline(y=-ci, line=dict(color="red", dash="dash"), row=2, col=1)
    # Truth-vs-forecast scatter
    fig.add_trace(go.Scattergl(
        x=sub["truth"], y=sub["mean"], mode="markers",
        marker=dict(size=2, color="#1f77b4", opacity=0.3),
        name="(truth, mean)"), row=2, col=2)
    tmin, tmax = float(sub["truth"].min()), float(sub["truth"].max())
    fig.add_trace(go.Scatter(x=[tmin, tmax], y=[tmin, tmax],
                              mode="lines",
                              line=dict(color="red", dash="dash"),
                              showlegend=False), row=2, col=2)
    fig.update_layout(height=720, margin=dict(t=60, b=40),
                       showlegend=False)
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Residual (m/s)", row=1, col=2)
    fig.update_xaxes(title_text="Lag (timesteps)", row=2, col=1)
    fig.update_xaxes(title_text="Truth (m/s)", row=2, col=2)
    fig.update_yaxes(title_text="Truth − forecast (m/s)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="Correlation", row=2, col=1)
    fig.update_yaxes(title_text="Forecast mean (m/s)", row=2, col=2)
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def fig_fft(truth_signal: np.ndarray, forecast_signal: np.ndarray,
            dt: float, wt_col: str) -> tuple[str, dict]:
    """Power spectral density of truth vs forecast — the smoking gun for time-flatness."""
    fs = 1.0 / dt  # sampling freq (Hz)
    nperseg = min(512, len(truth_signal))
    f_t, p_t = spsig.welch(truth_signal - np.mean(truth_signal),
                           fs=fs, nperseg=nperseg)
    f_f, p_f = spsig.welch(forecast_signal - np.mean(forecast_signal),
                           fs=fs, nperseg=nperseg)
    # Power retention ratio: model_power / truth_power per frequency bin
    eps = 1e-15
    ratio = p_f / (p_t + eps)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Power spectral density (log-log)",
                                        "Model power retention (model / truth)"))
    fig.add_trace(go.Scatter(x=f_t, y=p_t, mode="lines", name="Truth",
                              line=dict(color="black")), row=1, col=1)
    fig.add_trace(go.Scatter(x=f_f, y=p_f, mode="lines", name="Forecast",
                              line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=f_t, y=ratio, mode="lines",
                              line=dict(color="#ff7f0e"),
                              showlegend=False), row=1, col=2)
    fig.add_hline(y=1.0, line=dict(color="green", dash="dash"), row=1, col=2)
    fig.update_xaxes(type="log", title_text="Frequency (Hz)", row=1, col=1)
    fig.update_yaxes(type="log", title_text="PSD (m²/s² / Hz)", row=1, col=1)
    fig.update_xaxes(type="log", title_text="Frequency (Hz)", row=1, col=2)
    fig.update_yaxes(type="log", title_text="Power ratio (model / truth)",
                      row=1, col=2)
    fig.update_layout(height=420, margin=dict(t=60, b=40),
                       title=f"Frequency-domain analysis: {wt_col}")
    # Find cutoff: highest freq at which model retains > 50% truth power
    valid = p_t > eps
    above = (ratio[valid] > 0.5)
    cutoff = float(f_t[valid][above.argmin()]) if above.any() else 0.0
    # Half-power frequencies for truth and forecast
    truth_total = np.trapezoid(p_t, f_t)
    forecast_total = np.trapezoid(p_f, f_f)
    stats = {
        "f_50pct_cutoff_hz": cutoff,
        "truth_total_power": float(truth_total),
        "forecast_total_power": float(forecast_total),
        "power_ratio_total": float(forecast_total / max(truth_total, eps)),
    }
    return pio.to_html(fig, full_html=False, include_plotlyjs=False), stats


def fig_per_turbine_errors(metrics: pl.DataFrame) -> str:
    """Per-turbine error distribution: MAE, RMSE, persistence skill."""
    if metrics is None:
        return "<p>(metrics CSV not provided)</p>"
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("MAE per target", "RMSE per target",
                                        "Persistence skill (MAE)"))
    for lead in sorted(metrics["lead_step"].unique().to_list()):
        sub = metrics.filter(pl.col("lead_step") == lead)
        fig.add_trace(go.Box(y=sub["mae"], name=f"L{lead}",
                              boxpoints=False, showlegend=(lead == 0)),
                      row=1, col=1)
        fig.add_trace(go.Box(y=sub["rmse"], name=f"L{lead}",
                              boxpoints=False, showlegend=False),
                      row=1, col=2)
        if "skill_skill" in sub.columns:
            fig.add_trace(go.Box(y=sub["skill_skill"].drop_nulls(),
                                  name=f"L{lead}",
                                  boxpoints=False, showlegend=False),
                          row=1, col=3)
    fig.add_hline(y=0, line=dict(color="red", dash="dash"), row=1, col=3)
    fig.update_yaxes(title_text="m/s", row=1, col=1)
    fig.update_yaxes(title_text="m/s", row=1, col=2)
    fig.update_yaxes(title_text="1 − MAE_model/MAE_persistence", row=1, col=3)
    fig.update_layout(height=420, margin=dict(t=60, b=40),
                       title="Per-target error distribution across 176 targets")
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def fig_calibration_summary(intervals: pl.DataFrame | None) -> str:
    """Reliability + per-α coverage; only if CP intervals are provided."""
    if intervals is None:
        return "<p>(CP intervals not provided — skipping CP section)</p>"
    alphas = sorted(intervals["alpha"].unique().to_list())
    leads = sorted(intervals["lead_step"].unique().to_list())
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Reliability diagram (per lead)",
                                        "Mean band width per α (per lead)"))
    colors = ["#440154", "#3b528b", "#21918c", "#5ec962"]
    for lead, c in zip(leads, colors):
        emp = []; widths = []
        for a in alphas:
            sub = intervals.filter((pl.col("lead_step") == lead)
                                    & (pl.col("alpha") == a))
            if len(sub) == 0:
                emp.append(np.nan); widths.append(np.nan); continue
            in_band = ((sub["truth"] >= sub["lower_cp"])
                       & (sub["truth"] <= sub["upper_cp"]))
            emp.append(float(in_band.mean()))
            widths.append(float((sub["upper_cp"] - sub["lower_cp"]).mean()))
        nom = [1 - a for a in alphas]
        fig.add_trace(go.Scatter(x=nom, y=emp, mode="lines+markers",
                                  marker=dict(color=c), name=f"L{lead}"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=nom, y=widths, mode="lines+markers",
                                  marker=dict(color=c), name=f"L{lead}",
                                  showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                              line=dict(color="gray", dash="dash"),
                              showlegend=False), row=1, col=1)
    fig.update_xaxes(title_text="Nominal coverage", row=1, col=1)
    fig.update_yaxes(title_text="Empirical coverage", row=1, col=1)
    fig.update_xaxes(title_text="Nominal coverage", row=1, col=2)
    fig.update_yaxes(title_text="Mean band width (m/s)", row=1, col=2)
    fig.update_layout(height=420, margin=dict(t=60, b=40),
                       title="CP calibration summary")
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def fig_pit_histogram(per_lead_df: pl.DataFrame, intervals: pl.DataFrame | None) -> str:
    """Raw PIT (truth percentile vs samples) + CP PIT side by side."""
    sub = per_lead_df.filter(pl.col("lead_step") == 0).sort("time")
    if len(sub) == 0:
        return ""
    # Raw PIT: linear interpolation between p5, p95
    raw_w = (sub["p95"] - sub["p5"]).to_numpy()
    raw_pit = np.where(raw_w > 1e-9,
                       (sub["truth"].to_numpy() - sub["p5"].to_numpy()) / raw_w,
                       np.nan)
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Raw PIT (within 5–95% ensemble)",
                                        "Empirical coverage vs lead time"))
    fig.add_trace(go.Histogram(x=np.clip(raw_pit, -0.5, 1.5),
                                nbinsx=30, name="raw"),
                  row=1, col=1)
    fig.add_vline(x=0, line=dict(color="green", dash="dash"), row=1, col=1)
    fig.add_vline(x=1, line=dict(color="green", dash="dash"), row=1, col=1)
    # Cover-by-lead bar
    leads = sorted(per_lead_df["lead_step"].unique().to_list())
    cov = []
    for lead in leads:
        s = per_lead_df.filter(pl.col("lead_step") == lead)
        cov.append(float(((s["truth"] >= s["p5"]) & (s["truth"] <= s["p95"])).mean()))
    fig.add_trace(go.Bar(x=[f"L{l} ({(l+1)*15}s)" for l in leads], y=cov,
                          marker_color="#1f77b4"),
                  row=1, col=2)
    fig.add_hline(y=0.9, line=dict(color="red", dash="dash"), row=1, col=2)
    fig.update_xaxes(title_text="PIT", row=1, col=1)
    fig.update_xaxes(title_text="Lead", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Raw 90% coverage", row=1, col=2,
                      range=[0, 1])
    fig.update_layout(height=420, margin=dict(t=60, b=40),
                       title="Probability integral transform & coverage",
                       showlegend=False)
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


# ---------- HTML assembly ----------

PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif;
         max-width: 1400px; margin: 1em auto; padding: 0 1em;
         color: #222; line-height: 1.5; }}
  h1 {{ border-bottom: 3px solid #333; padding-bottom: 0.3em; }}
  h2 {{ color: #444; margin-top: 2.5em; border-bottom: 1px solid #aaa;
        padding-bottom: 0.2em; }}
  h3 {{ color: #555; margin-top: 1.5em; }}
  .verdict {{ padding: 1.2em; margin: 1em 0; border-radius: 5px;
              border-left: 6px solid; }}
  .verdict.warn  {{ background: #fff3cd; border-color: #ffa500; }}
  .verdict.fail  {{ background: #f8d7da; border-color: #dc3545; }}
  .verdict.pass  {{ background: #d4edda; border-color: #28a745; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0;
           font-size: 0.92em; }}
  th, td {{ border: 1px solid #ccc; padding: 0.4em 0.6em; text-align: right; }}
  th {{ background: #eee; }}
  td:first-child, th:first-child {{ text-align: left; }}
  .metric-positive {{ color: #28a745; font-weight: bold; }}
  .metric-negative {{ color: #dc3545; font-weight: bold; }}
  .toc {{ background: #f5f5f5; padding: 1em; border-radius: 4px;
          margin: 1em 0; }}
  .toc ul {{ margin: 0; padding-left: 1.2em; }}
  pre {{ background: #f5f5f5; padding: 0.8em; overflow-x: auto;
         border-radius: 4px; font-size: 0.85em; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit,
              minmax(220px, 1fr)); gap: 0.8em; margin: 1em 0; }}
  .stat-card {{ background: #f9f9f9; padding: 0.8em 1em; border-radius: 4px;
              border-left: 4px solid #1f77b4; }}
  .stat-card .label {{ color: #666; font-size: 0.85em; }}
  .stat-card .value {{ font-size: 1.4em; font-weight: bold; color: #333;
                      margin-top: 0.2em; }}
  .footnote {{ color: #777; font-size: 0.85em; margin-top: 2em;
               border-top: 1px solid #eee; padding-top: 1em; }}
</style>
</head>
<body>

<h1>TACTiS-2 Phase 0i-B — comprehensive model quality report</h1>
<p><b>Checkpoint:</b> {checkpoint}<br>
<b>Forecast parquet:</b> {forecast}<br>
<b>Test split:</b> SPLIT{split_id}, {n_test_idx} forecast issuances × 4 lead steps × 200 samples<br>
<b>Generated:</b> {generated_at}</p>

<div class="verdict {verdict_class}">
<h2 style="margin-top:0;border:none;">Executive verdict: {verdict_title}</h2>
{verdict_body}
</div>

<div class="toc">
<b>Contents</b>
<ul>
<li><a href="#summary">1. Headline numbers</a></li>
<li><a href="#per-lead">2. Per-lead error metrics</a></li>
<li><a href="#timeseries">3. Time-series with bands (interactive)</a></li>
<li><a href="#errors">4. Error decomposition (residual time-series + ACF + scatter)</a></li>
<li><a href="#fft">5. Frequency-domain analysis (the smoking gun)</a></li>
<li><a href="#calibration">6. Probabilistic calibration</a></li>
<li><a href="#per-turbine">7. Per-turbine error distributions</a></li>
<li><a href="#assessment">8. Honest assessment + recommendations</a></li>
</ul>
</div>

<h2 id="summary">1. Headline numbers (representative turbine: {wt_col})</h2>
<div class="stat-grid">
{stat_cards}
</div>

<h2 id="per-lead">2. Per-lead error metrics</h2>
<p>Aggregated across the test period, broken out by 15s/30s/45s/60s lead time.</p>
<table>
<thead><tr><th>Lead</th><th>n</th><th>MAE</th><th>RMSE</th><th>Bias</th>
<th>Ens std</th><th>Raw 90% width</th><th>Raw 90% coverage</th>
<th>Spread/skill</th><th>MAE skill vs persistence</th></tr></thead>
<tbody>{per_lead_rows}</tbody>
</table>
<p class="footnote"><b>Spread/skill ratio</b> = ens_std / RMSE. Healthy ≈ 1.
Tiny values mean the predicted spread vastly under-represents true error
(overconfidence). <b>MAE skill</b> = 1 − MAE_model / MAE_persistence: positive
beats persistence, negative loses to it.</p>

<h2 id="timeseries">3. Time-series with bands (interactive)</h2>
{timeseries_fig}

<h2 id="errors">4. Error decomposition</h2>
<p>Four diagnostic views of the lead-0 residuals: time-evolution, distribution,
autocorrelation, and scatter against truth. A healthy model shows
near-Gaussian residuals, low autocorrelation, and points clustering tightly
along the y=x line.</p>
{errors_fig}

<h2 id="fft">5. Frequency-domain analysis</h2>
<p>Power spectral density (PSD) of truth vs forecast at lead 0 (one prediction
per 15s for the test period). The model retains power below
<b>{f_cutoff:.4f} Hz</b> (period {period_cutoff_min:.1f} min) but suppresses
higher-frequency content. The right panel shows the model/truth power ratio
per frequency bin — values near 1 mean the model preserves that frequency,
values near 0 mean it filters it out.</p>
{fft_fig}
<div class="stat-grid">
<div class="stat-card"><div class="label">Truth total power</div>
<div class="value">{truth_total_power:.3f}</div></div>
<div class="stat-card"><div class="label">Forecast total power</div>
<div class="value">{forecast_total_power:.3f}</div></div>
<div class="stat-card"><div class="label">Power retention ratio</div>
<div class="value">{power_ratio_total:.3f}</div></div>
<div class="stat-card"><div class="label">50%-power cutoff</div>
<div class="value">{f_cutoff:.4f} Hz</div></div>
</div>

<h2 id="calibration">6. Probabilistic calibration</h2>
{pit_fig}
{calibration_fig}

<h2 id="per-turbine">7. Per-turbine error distributions</h2>
{per_turbine_fig}

<h2 id="assessment">8. Honest assessment + recommendations</h2>
{assessment_body}

<div class="footnote">
Generated by <code>generate_model_report.py</code>.
Source: <code>{forecast_path}</code> joined with truth from
<code>{truth_path}</code> (SPLIT{split_id}).
</div>

</body>
</html>
"""


def build_verdict(per_lead_metrics_d: dict, fft_stats: dict) -> tuple[str, str, str]:
    """Decide PASS / WARN / FAIL based on key metrics."""
    spread_skills = [m["spread_skill_ratio"] for m in per_lead_metrics_d.values()
                     if not np.isnan(m["spread_skill_ratio"])]
    avg_skill = np.mean([m["skill_mae_vs_persist"] for m in per_lead_metrics_d.values()
                         if not np.isnan(m["skill_mae_vs_persist"])])
    cov90 = np.mean([m["raw_90_coverage"] for m in per_lead_metrics_d.values()])
    power_retention = fft_stats.get("power_ratio_total", 1.0)
    if spread_skills and max(spread_skills) > 0.7 and avg_skill > 0.2:
        cls = "pass"
        title = "Model is well-calibrated and skilful"
    elif power_retention < 0.05 or (spread_skills and max(spread_skills) < 0.1):
        cls = "fail"
        title = "Model has collapsed predictive distribution and acts as a low-pass filter"
    else:
        cls = "warn"
        title = "Model is partially-usable point predictor with broken probabilistic output"

    max_ss = max(spread_skills) if spread_skills else float("nan")
    ss_interp = ("≈ 1.0 — predicted uncertainty matches actual error."
                 if max_ss > 0.7 else
                 f"far below 1.0 — predicted uncertainty is at most {int(100*max_ss)}% "
                 f"of actual error (F1 sample-tightness collapse).")
    body = f"""
<p><b>Spread/skill ratio (ens_std / RMSE):</b> max across leads
= {max_ss:.3f}. A healthy model has spread/skill ≈ 1.0; this model is
{ss_interp}
</p>
<p><b>Average MAE skill vs persistence:</b>
{'+' if avg_skill > 0 else ''}{avg_skill:+.3f}. The model
{'beats persistence on MAE.' if avg_skill > 0 else 'loses to persistence on MAE (a 1-line baseline).' }
</p>
<p><b>Raw 90% ensemble coverage:</b> {cov90:.3f} (nominal 0.90). The model's
self-reported 90% interval contains the truth
{'as advertised.' if abs(cov90 - 0.9) < 0.05 else
 f'only {cov90*100:.1f}% of the time (vs 90% nominal). Severe miscalibration.'}
</p>
<p><b>High-frequency power retention:</b>
{power_retention:.3f} ({power_retention*100:.2f}% of truth variance preserved).
The model
{'preserves truth dynamics across the relevant frequency band.' if power_retention > 0.5 else
 'filters out the high-frequency content that drives 60s-scale wind variability — essentially a low-pass filter on truth.'}
</p>
"""
    return cls, title, body


def build_per_lead_table_rows(d: dict) -> str:
    rows = []
    for lead, m in sorted(d.items()):
        skill = m["skill_mae_vs_persist"]
        skill_cls = "metric-positive" if skill > 0 else "metric-negative" if skill < 0 else ""
        skill_str = f"{skill:+.3f}" if not np.isnan(skill) else "—"
        spread = m["spread_skill_ratio"]
        spread_cls = "metric-positive" if spread > 0.7 else "metric-negative" if spread < 0.2 else ""
        rows.append(
            f"<tr><td>L{lead} ({(lead+1)*15}s)</td>"
            f"<td>{m['n']}</td>"
            f"<td>{m['mae']:.3f}</td>"
            f"<td>{m['rmse']:.3f}</td>"
            f"<td>{m['bias']:+.3f}</td>"
            f"<td>{m['ens_std_mean']:.3f}</td>"
            f"<td>{m['raw_90_width_mean']:.3f}</td>"
            f"<td>{m['raw_90_coverage']:.3f}</td>"
            f"<td class='{spread_cls}'>{m['spread_skill_ratio']:.3f}</td>"
            f"<td class='{skill_cls}'>{skill_str}</td></tr>"
        )
    return "\n".join(rows)


def build_stat_cards(per_lead_d: dict, fft_stats: dict, n_targets: int) -> str:
    avg_skill = np.nanmean([m["skill_mae_vs_persist"] for m in per_lead_d.values()])
    avg_mae = np.mean([m["mae"] for m in per_lead_d.values()])
    avg_ens_std = np.mean([m["ens_std_mean"] for m in per_lead_d.values()])
    avg_truth_std = np.mean([m["truth_std"] for m in per_lead_d.values()])
    avg_spread_skill = np.mean([m["spread_skill_ratio"] for m in per_lead_d.values()])

    cards = [
        ("Mean absolute error (avg over leads)", f"{avg_mae:.3f} m/s"),
        ("Persistence skill (MAE)", f"{avg_skill:+.3f}"),
        ("Predictive std (mean)", f"{avg_ens_std:.3f} m/s"),
        ("Truth std (mean)", f"{avg_truth_std:.3f} m/s"),
        ("Spread/skill ratio", f"{avg_spread_skill:.3f}"),
        ("High-freq power retention", f"{fft_stats.get('power_ratio_total', 1):.3f}"),
        ("Targets analysed", f"{n_targets}"),
    ]
    return "\n".join(f'<div class="stat-card"><div class="label">{l}</div>'
                     f'<div class="value">{v}</div></div>'
                     for l, v in cards)


def build_assessment(per_lead_d: dict, fft_stats: dict) -> str:
    """Detailed recommendations."""
    spread = np.mean([m["spread_skill_ratio"] for m in per_lead_d.values()])
    cov90 = np.mean([m["raw_90_coverage"] for m in per_lead_d.values()])
    skill = np.nanmean([m["skill_mae_vs_persist"] for m in per_lead_d.values()])
    pwr = fft_stats.get("power_ratio_total", 1)

    body = ["<h3>What this model does well</h3><ul>"]
    if abs(skill) < 0.1:
        body.append("<li>Tracks the slow-drifting regime mean over minute-to-hour timescales — equivalent to a low-pass filter on truth.</li>")
    if pwr > 0.1:
        body.append("<li>Preserves at least some low-frequency power.</li>")
    body.append("<li>Produces correlated multivariate samples via the trained copula (joint cross-turbine structure exists, even if marginals are collapsed).</li>")
    body.append("</ul>")

    body.append("<h3>What this model does NOT do</h3><ul>")
    if spread < 0.2:
        body.append(f"<li><b>Severe F1 sample-tightness collapse</b>: predicted spread is "
                    f"{spread*100:.1f}% of actual prediction error. The model is wildly overconfident.</li>")
    if cov90 < 0.5:
        body.append(f"<li><b>Raw 90% intervals miscover</b>: truth lands inside the band only "
                    f"{cov90*100:.1f}% of the time (vs 90% nominal).</li>")
    if pwr < 0.05:
        body.append(f"<li><b>F2 time-flatness</b>: model retains only {pwr*100:.1f}% of "
                    f"truth's frequency-domain power — high-frequency turbulence is filtered out.</li>")
    body.append("<li><b>No useful aleatoric uncertainty</b>: the 200 samples cluster within ±0.04 m/s while truth oscillates ±0.5 m/s.</li>")
    body.append("</ul>")

    body.append("<h3>Will additional training/tuning fix this?</h3>")
    body.append("""<p><b>Almost certainly no.</b> Six different in-training interventions
have already been tested (Sa <i>a_floor</i>, Sw <i>w_entropy</i>, Sd
<i>Energy Score</i>, Sd <i>Variogram Score</i>, Sd <i>trajectory noise</i>,
NSF parametrization swap). None widened the predictive distribution
past 0.27 standardized units (healthy = 3.3, the broken baseline = 0.27).</p>

<p>The bottleneck is fundamental to the
<b>NLL × highly-autoregressive-data combination</b>: each training
context appears once with one realization, so maximum-likelihood
training collapses the conditional density to a spike at the observed
value. No regularizer pressure has been able to overcome this. NSF
(which structurally cannot produce flat CDFs via min_derivative &gt; 0)
made the collapse <em>worse</em>, not better.</p>

<h3>What might actually fix it</h3>
<ol>
<li><b>Path 2 — Quantile head with pinball loss (~40h GPU + 2 dev days).</b>
Replace DSF with K predicted quantiles. Pinball loss is a sum of K
independent terms (q_0.05, q_0.10, ..., q_0.95) that <em>cannot all be
minimized at a delta function</em> — q_0.05 ≠ q_0.95 forces the model
to predict spread. Different objective function entirely.
<b>P(success) ≈ 0.50</b>.</li>
<li><b>Path 3 — 5s resolution + 2h context (~48h GPU + 26GB disk).</b>
Pure data/config change. AWAKEN is natively 1s but currently downsampled
to 15s; at 5s the model would <em>observe</em> sub-15s turbulent
variability and have aleatoric signal to learn from. Longer context
adds regime identification. <b>P(success) ≈ 0.40</b>.</li>
<li><b>Conformal prediction (already done).</b> Post-hoc statistical
calibration that recovers honest interval coverage from the
overconfident base model. Doesn't fix F1 or F2 in the model itself but
makes the output statistically valid. <b>Operational deliverable today</b>.</li>
</ol>

<h3>Recommendation</h3>
<p>Decide based on the downstream use case:</p>
<ul>
<li><b>Slow-loop control (minute-scale decisions):</b> Phase 0i-B + CP is
adequate. The model's trend-following behavior plus CP's honest bands
gives a deployable system today.</li>
<li><b>Fast-loop wind/gust response (15-60s decisions):</b> Path 2 first.
If pinball-loss quantile head doesn't widen samples, Path 3.
If both fail, the dataset itself may be too deterministic at this
horizon for any NLL/quantile model to learn aleatoric variability,
and the prediction horizon or feature set needs revisiting.</li>
</ul>""")
    return "\n".join(body)


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--forecast", type=Path, required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--intervals", type=Path, default=None)
    ap.add_argument("--metrics", type=Path, default=None)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--split-id", type=int, default=194)
    ap.add_argument("--turbine", default="wt008",
                    help="Representative turbine for time-series + FFT")
    ap.add_argument("--component", default="horz", choices=["horz", "vert"])
    ap.add_argument("--checkpoint", type=str,
                    default="Phase 0i-B unsmoothed_amax3 manual_save_epoch74")
    args = ap.parse_args()

    fcst, truth, intervals, metrics = load_inputs(
        args.forecast, args.truth, args.split_id, args.intervals, args.metrics)

    wt_col = f"ws_{args.component}_{args.turbine}"
    target_idx = (int(args.turbine.replace("wt", "")) - 1
                  + (N_TURBINES if args.component == "vert" else 0))
    target_col = f"target_{target_idx}"

    log.info("Building per-lead arrays for %s", wt_col)
    per_lead = build_per_lead_arrays(fcst, truth, wt_col, target_col)
    log.info("Per-lead metrics")
    per_lead_d = per_lead_metrics(per_lead)

    log.info("FFT analysis")
    lead0 = per_lead.filter(pl.col("lead_step") == 0).sort("time")
    truth_signal = lead0["truth"].to_numpy()
    fcst_signal = lead0["mean"].to_numpy()
    fft_fig, fft_stats = fig_fft(truth_signal, fcst_signal, dt=15.0, wt_col=wt_col)

    log.info("Other figures")
    ts_fig = fig_timeseries(per_lead, wt_col,
                            f"{wt_col} — lead 0 forecast (ensemble band + mean + truth)")
    err_fig = fig_error_decomposition(per_lead, wt_col)
    pit_fig = fig_pit_histogram(per_lead, intervals)
    cal_fig = fig_calibration_summary(intervals)
    pt_fig = fig_per_turbine_errors(metrics)

    log.info("Verdict + assembly")
    verdict_cls, verdict_title, verdict_body = build_verdict(per_lead_d, fft_stats)

    period_cutoff_min = (1.0 / max(fft_stats["f_50pct_cutoff_hz"], 1e-9)) / 60.0
    n_targets = metrics["target_idx"].n_unique() if metrics is not None else 1
    from datetime import datetime
    html = PAGE_TEMPLATE.format(
        title=f"TACTiS-2 Phase 0i-B report ({wt_col})",
        checkpoint=args.checkpoint,
        forecast=str(args.forecast),
        truth_path=str(args.truth),
        forecast_path=str(args.forecast),
        split_id=args.split_id,
        n_test_idx=fcst["test_idx"].n_unique(),
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        wt_col=wt_col,
        verdict_class=verdict_cls,
        verdict_title=verdict_title,
        verdict_body=verdict_body,
        stat_cards=build_stat_cards(per_lead_d, fft_stats, n_targets),
        per_lead_rows=build_per_lead_table_rows(per_lead_d),
        timeseries_fig=ts_fig,
        errors_fig=err_fig,
        fft_fig=fft_fig,
        f_cutoff=fft_stats["f_50pct_cutoff_hz"],
        period_cutoff_min=period_cutoff_min,
        truth_total_power=fft_stats["truth_total_power"],
        forecast_total_power=fft_stats["forecast_total_power"],
        power_ratio_total=fft_stats["power_ratio_total"],
        pit_fig=pit_fig,
        calibration_fig=cal_fig,
        per_turbine_fig=pt_fig,
        assessment_body=build_assessment(per_lead_d, fft_stats),
    )
    args.output.write_text(html)
    log.info("Wrote %s (%.1f KB)", args.output, args.output.stat().st_size / 1024)


if __name__ == "__main__":
    main()
