"""Head-to-head model comparison report — Phase 0i-G vs Phase 0i-B.

Self-contained HTML with side-by-side metrics, overlaid plots, and an
automated verdict ("model G beats / ties / loses to model B").

Designed to be model-agnostic: works for any two TACTiS-2 forecast parquets
sharing the same truth split. Reuses the metrics + FFT machinery from
`generate_model_report.py` to keep computations consistent.

Usage:
    python compare_models.py \\
        --label-a "Phase 0i-B (DSF + NLL + CP)" \\
        --forecast-a <forecast_b_parquet> \\
        --metrics-a <metrics_summary_b.csv> \\
        --label-b "Phase 0i-G (Quantile + pinball)" \\
        --forecast-b <forecast_g_parquet> \\
        --metrics-b <metrics_summary_g.csv> \\
        --truth <test_denormalize.parquet> \\
        --output comparison.html
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))
from generate_model_report import (  # noqa: E402
    N_LEAD,
    N_TURBINES,
    build_per_lead_arrays,
    fig_fft,
    per_lead_metrics,
)
from plot_utils import map_target_to_wt_col  # noqa: E402

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger("compare")


# ---------- Diff metric computation ----------

def compute_deltas(per_lead_a: dict, per_lead_b: dict) -> dict:
    """B − A per lead, per metric. Positive Δ on error metrics (MAE/RMSE) =
    B is WORSE; positive Δ on skill metrics = B is BETTER."""
    out = {}
    for lead in sorted(set(per_lead_a) & set(per_lead_b)):
        a = per_lead_a[lead]
        b = per_lead_b[lead]
        out[lead] = {
            "delta_mae":            b["mae"] - a["mae"],
            "delta_rmse":           b["rmse"] - a["rmse"],
            "delta_bias":           b["bias"] - a["bias"],
            "delta_ens_std":        b["ens_std_mean"] - a["ens_std_mean"],
            "delta_raw90_width":    b["raw_90_width_mean"] - a["raw_90_width_mean"],
            "delta_raw90_coverage": b["raw_90_coverage"] - a["raw_90_coverage"],
            "delta_spread_skill":   b["spread_skill_ratio"] - a["spread_skill_ratio"],
            "delta_skill":          (b["skill_mae_vs_persist"]
                                       if not np.isnan(b["skill_mae_vs_persist"])
                                       else 0)
                                    - (a["skill_mae_vs_persist"]
                                       if not np.isnan(a["skill_mae_vs_persist"])
                                       else 0),
            "a_mae": a["mae"], "b_mae": b["mae"],
            "a_rmse": a["rmse"], "b_rmse": b["rmse"],
            "a_ens_std": a["ens_std_mean"], "b_ens_std": b["ens_std_mean"],
            "a_raw90_cov": a["raw_90_coverage"], "b_raw90_cov": b["raw_90_coverage"],
            "a_raw90_w": a["raw_90_width_mean"], "b_raw90_w": b["raw_90_width_mean"],
            "a_spread_skill": a["spread_skill_ratio"], "b_spread_skill": b["spread_skill_ratio"],
            "a_skill": a["skill_mae_vs_persist"], "b_skill": b["skill_mae_vs_persist"],
        }
    return out


def verdict(deltas: dict, fft_a: dict, fft_b: dict) -> tuple[str, str, str]:
    """Tally: how many metrics does B beat A on?"""
    wins_b = 0
    wins_a = 0
    ties = 0
    metric_summary = []
    for lead, d in deltas.items():
        for key, direction, label in [
            ("delta_mae",            -1, "MAE (lower better)"),
            ("delta_rmse",           -1, "RMSE (lower better)"),
            ("delta_spread_skill",   +1, "Spread/skill ratio (higher better)"),
            ("delta_skill",          +1, "Persistence skill (higher better)"),
            ("delta_raw90_coverage", "abs", "Raw 90% coverage (closer to 0.9 better)"),
        ]:
            v = d[key]
            if key == "delta_raw90_coverage":
                # Closer to 0.9 wins
                a_dist = abs(d["a_raw90_cov"] - 0.9)
                b_dist = abs(d["b_raw90_cov"] - 0.9)
                if b_dist < a_dist - 1e-3:
                    wins_b += 1
                elif a_dist < b_dist - 1e-3:
                    wins_a += 1
                else:
                    ties += 1
            else:
                signed = v * direction
                if signed > 1e-4:
                    wins_b += 1
                elif signed < -1e-4:
                    wins_a += 1
                else:
                    ties += 1
            metric_summary.append((lead, key, v, label))
    # FFT power retention: B beats A if retains more
    a_power = fft_a.get("power_ratio_total", 1.0)
    b_power = fft_b.get("power_ratio_total", 1.0)
    if b_power > a_power + 1e-3:
        wins_b += 1
    elif b_power > a_power - 1e-3:
        ties += 1
    else:
        wins_a += 1

    if wins_b > wins_a:
        cls = "pass"
        title = (f"Phase 0i-G WINS on {wins_b}/{wins_a+wins_b+ties} metrics — "
                 f"recommend ship as Phase 0i-G upgrade")
    elif wins_a > wins_b:
        cls = "warn"
        title = (f"Phase 0i-G LOSES on {wins_a}/{wins_a+wins_b+ties} metrics — "
                 f"keep Phase 0i-B + CP as operational")
    else:
        cls = "warn"
        title = (f"TIE: {wins_b}/{wins_a+wins_b+ties} for G, {wins_a} for A, "
                 f"{ties} tied — judgment call")

    summary_pts = []
    summary_pts.append(f"<li>G's MAE skill (mean over leads): "
                       f"{np.mean([d['b_skill'] for d in deltas.values() if not np.isnan(d['b_skill'])]):+.4f} "
                       f"vs A: "
                       f"{np.mean([d['a_skill'] for d in deltas.values() if not np.isnan(d['a_skill'])]):+.4f}</li>")
    summary_pts.append(f"<li>G's raw 90% coverage (mean): "
                       f"{np.mean([d['b_raw90_cov'] for d in deltas.values()]):.3f} (nom 0.90) "
                       f"vs A: {np.mean([d['a_raw90_cov'] for d in deltas.values()]):.3f}</li>")
    summary_pts.append(f"<li>G's spread/skill (mean): "
                       f"{np.mean([d['b_spread_skill'] for d in deltas.values()]):.3f} (target ≈ 1.0) "
                       f"vs A: {np.mean([d['a_spread_skill'] for d in deltas.values()]):.3f}</li>")
    summary_pts.append(f"<li>High-freq power retention: "
                       f"G = {b_power:.3f} vs A = {a_power:.3f}</li>")
    body = "<ul>" + "\n".join(summary_pts) + "</ul>"
    return cls, title, body


# ---------- Plotly figures ----------

def fig_overlay_timeseries(per_lead_a, per_lead_b, label_a, label_b, wt_col) -> str:
    """Overlaid lead-0 time-series for one representative turbine."""
    a = per_lead_a.filter(pl.col("lead_step") == 0).sort("time")
    b = per_lead_b.filter(pl.col("lead_step") == 0).sort("time")
    # Inner-join via polars (avoids Datetime↔Int64 is_in mismatch)
    a_aligned = a.join(b.select(["time"]), on="time", how="inner").sort("time")
    b_aligned = b.join(a.select(["time"]), on="time", how="inner").sort("time")
    if len(a_aligned) == 0:
        return "<p>(no overlapping timesteps between the two forecasts)</p>"
    n = len(a_aligned)
    mid = n // 2
    half = min(120, n // 2)
    lo, hi = max(0, mid - half), min(n, mid + half)
    a_sub = a_aligned.slice(lo, hi - lo)
    b_sub = b_aligned.slice(lo, hi - lo)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a_sub["time"], y=a_sub["truth"], mode="lines",
                             line=dict(color="black", width=1.2), name="Truth"))
    fig.add_trace(go.Scatter(x=a_sub["time"], y=a_sub["mean"], mode="lines",
                             line=dict(color="#1f77b4", width=1.5), name=f"{label_a} mean"))
    fig.add_trace(go.Scatter(x=a_sub["time"], y=a_sub["p95"], mode="lines",
                             line=dict(color="rgba(31,119,180,0)"), showlegend=False))
    fig.add_trace(go.Scatter(x=a_sub["time"], y=a_sub["p5"], mode="lines",
                             line=dict(color="rgba(31,119,180,0)"), fill="tonexty",
                             fillcolor="rgba(31,119,180,0.2)",
                             name=f"{label_a} 5–95%"))
    fig.add_trace(go.Scatter(x=b_sub["time"], y=b_sub["mean"], mode="lines",
                             line=dict(color="#ff7f0e", width=1.5), name=f"{label_b} mean"))
    fig.add_trace(go.Scatter(x=b_sub["time"], y=b_sub["p95"], mode="lines",
                             line=dict(color="rgba(255,127,14,0)"), showlegend=False))
    fig.add_trace(go.Scatter(x=b_sub["time"], y=b_sub["p5"], mode="lines",
                             line=dict(color="rgba(255,127,14,0)"), fill="tonexty",
                             fillcolor="rgba(255,127,14,0.2)",
                             name=f"{label_b} 5–95%"))
    fig.update_layout(title=f"{wt_col} — overlaid 1h forecast bands", height=480,
                      xaxis_title="Time", yaxis_title="Wind component (m/s)",
                      hovermode="x unified", margin=dict(t=60, b=40))
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def fig_overlay_psd(truth_signal, fcst_a, fcst_b, label_a, label_b, dt=15.0) -> str:
    """Three-line PSD: truth + both models. Visually shows whether B retains more
    high-frequency power than A."""
    from scipy import signal as spsig
    fs = 1.0 / dt
    nperseg = min(512, len(truth_signal))
    f_t, p_t = spsig.welch(truth_signal - np.mean(truth_signal), fs=fs, nperseg=nperseg)
    f_a, p_a = spsig.welch(fcst_a - np.mean(fcst_a), fs=fs, nperseg=nperseg)
    f_b, p_b = spsig.welch(fcst_b - np.mean(fcst_b), fs=fs, nperseg=nperseg)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f_t, y=p_t, mode="lines",
                             line=dict(color="black", width=2), name="Truth"))
    fig.add_trace(go.Scatter(x=f_a, y=p_a, mode="lines",
                             line=dict(color="#1f77b4", width=1.5), name=label_a))
    fig.add_trace(go.Scatter(x=f_b, y=p_b, mode="lines",
                             line=dict(color="#ff7f0e", width=1.5), name=label_b))
    fig.update_xaxes(type="log", title_text="Frequency (Hz)")
    fig.update_yaxes(type="log", title_text="PSD (m²/s² / Hz)")
    fig.update_layout(title="Power spectral density: truth vs both models",
                      height=460, margin=dict(t=60, b=40),
                      annotations=[dict(
                          text="If A is below truth at high f and B sits above A, B retains more turbulence.",
                          showarrow=False, xref="paper", yref="paper",
                          x=0.02, y=-0.18, font=dict(size=10, color="#555"))])
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def fig_per_lead_delta_bars(deltas: dict) -> str:
    leads = sorted(deltas)
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("Δ MAE (G − B)", "Δ Spread/skill ratio",
                                        "Δ Persistence skill"))
    fig.add_trace(go.Bar(x=[f"L{l}" for l in leads],
                          y=[deltas[l]["delta_mae"] for l in leads],
                          marker_color="#1f77b4"), row=1, col=1)
    fig.add_trace(go.Bar(x=[f"L{l}" for l in leads],
                          y=[deltas[l]["delta_spread_skill"] for l in leads],
                          marker_color="#1f77b4"), row=1, col=2)
    fig.add_trace(go.Bar(x=[f"L{l}" for l in leads],
                          y=[deltas[l]["delta_skill"] for l in leads],
                          marker_color="#1f77b4"), row=1, col=3)
    fig.add_hline(y=0, line=dict(color="red", dash="dash"))
    fig.update_layout(height=380, margin=dict(t=60, b=40), showlegend=False,
                      title="Per-lead delta (positive = Phase 0i-G better)")
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


# ---------- HTML assembly ----------

PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Phase 0i-G vs Phase 0i-B comparison</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 1400px;
         margin: 1em auto; padding: 0 1em; line-height: 1.5; color: #222; }}
  h1, h2 {{ border-bottom: 2px solid #333; padding-bottom: 0.2em; }}
  .verdict {{ padding: 1em; margin: 1em 0; border-radius: 5px;
              border-left: 6px solid; }}
  .verdict.pass {{ background: #d4edda; border-color: #28a745; }}
  .verdict.warn {{ background: #fff3cd; border-color: #ffa500; }}
  .verdict.fail {{ background: #f8d7da; border-color: #dc3545; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0;
           font-size: 0.92em; }}
  th, td {{ border: 1px solid #ccc; padding: 0.4em 0.7em; text-align: right; }}
  th {{ background: #eee; }}
  td:first-child, th:first-child {{ text-align: left; font-weight: bold; }}
  .delta-pos {{ color: #28a745; font-weight: bold; }}
  .delta-neg {{ color: #dc3545; font-weight: bold; }}
  .delta-zero {{ color: #888; }}
  .footnote {{ color: #777; font-size: 0.85em; margin-top: 2em;
               border-top: 1px solid #eee; padding-top: 1em; }}
</style>
</head><body>

<h1>Phase 0i-G vs Phase 0i-B — head-to-head model comparison</h1>
<p><b>Model A:</b> {label_a}<br>
<b>Model B:</b> {label_b}<br>
<b>Generated:</b> {generated_at}</p>

<div class="verdict {verdict_class}">
<h2 style="margin-top:0;border:none;">Verdict: {verdict_title}</h2>
{verdict_body}
</div>

<h2>Per-lead delta table</h2>
<p>"Δ" = (B − A). For error metrics (MAE/RMSE/bias) <em>negative</em> Δ means B is better.
For skill metrics (spread/skill, persistence skill, raw 90% coverage)
<em>positive</em> Δ means B is better.</p>
<table>
<thead><tr>
<th>Lead</th>
<th>A MAE</th><th>B MAE</th><th>Δ MAE</th>
<th>A RMSE</th><th>B RMSE</th><th>Δ RMSE</th>
<th>A ens std</th><th>B ens std</th>
<th>A 90% cov</th><th>B 90% cov</th>
<th>A spread/skill</th><th>B spread/skill</th>
<th>A skill</th><th>B skill</th>
</tr></thead><tbody>
{rows}
</tbody></table>

<h2>Per-lead delta bar chart</h2>
{delta_bars}

<h2>Overlaid time-series (representative turbine, 1-hour zoom)</h2>
{overlay_ts}

<h2>Frequency-domain comparison (truth + both models)</h2>
{overlay_psd}
<p class="footnote">If model A sits well below truth at high frequencies and model
B sits closer to truth, B is retaining more of the turbulence the model is
supposed to forecast.</p>

<div class="footnote">
Reused unchanged from Path 1: <code>generate_model_report.py</code> for metrics,
<code>plot_utils.py</code> for helpers. This comparison report is the only NEW
post-processing artefact created for Phase 0i-G.
</div>

</body></html>
"""


def fmt_delta(v: float, lower_is_better: bool) -> str:
    if abs(v) < 1e-5:
        return f'<span class="delta-zero">{v:+.4f}</span>'
    better = (v < 0) if lower_is_better else (v > 0)
    klass = "delta-pos" if better else "delta-neg"
    return f'<span class="{klass}">{v:+.4f}</span>'


def build_table_rows(deltas: dict) -> str:
    rows = []
    for lead in sorted(deltas):
        d = deltas[lead]
        a_skill_str = f"{d['a_skill']:+.4f}" if not np.isnan(d['a_skill']) else "—"
        b_skill_str = f"{d['b_skill']:+.4f}" if not np.isnan(d['b_skill']) else "—"
        rows.append(
            f"<tr><td>L{lead} ({(lead+1)*15}s)</td>"
            f"<td>{d['a_mae']:.4f}</td><td>{d['b_mae']:.4f}</td>"
            f"<td>{fmt_delta(d['delta_mae'], lower_is_better=True)}</td>"
            f"<td>{d['a_rmse']:.4f}</td><td>{d['b_rmse']:.4f}</td>"
            f"<td>{fmt_delta(d['delta_rmse'], lower_is_better=True)}</td>"
            f"<td>{d['a_ens_std']:.4f}</td><td>{d['b_ens_std']:.4f}</td>"
            f"<td>{d['a_raw90_cov']:.4f}</td><td>{d['b_raw90_cov']:.4f}</td>"
            f"<td>{d['a_spread_skill']:.4f}</td><td>{d['b_spread_skill']:.4f}</td>"
            f"<td>{a_skill_str}</td><td>{b_skill_str}</td></tr>"
        )
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label-a", default="Phase 0i-B")
    ap.add_argument("--forecast-a", type=Path, required=True)
    ap.add_argument("--label-b", default="Phase 0i-G")
    ap.add_argument("--forecast-b", type=Path, required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--split-id", type=int, default=194)
    ap.add_argument("--turbine", default="wt008")
    ap.add_argument("--component", default="horz", choices=["horz", "vert"])
    args = ap.parse_args()

    log.info("Loading forecasts and truth")
    fcst_a = pl.read_parquet(args.forecast_a)
    fcst_b = pl.read_parquet(args.forecast_b)
    truth = (pl.read_parquet(args.truth)
             .filter(pl.col("item_id") == f"SPLIT{args.split_id}")
             .sort("time"))

    wt_col = f"ws_{args.component}_{args.turbine}"
    tgt_idx = (int(args.turbine.replace("wt", "")) - 1
               + (N_TURBINES if args.component == "vert" else 0))
    tgt_col = f"target_{tgt_idx}"

    log.info("Building per-lead arrays and metrics for %s", args.label_a)
    pl_a = build_per_lead_arrays(fcst_a, truth, wt_col, tgt_col)
    per_lead_a = per_lead_metrics(pl_a)
    log.info("Building per-lead arrays and metrics for %s", args.label_b)
    pl_b = build_per_lead_arrays(fcst_b, truth, wt_col, tgt_col)
    per_lead_b = per_lead_metrics(pl_b)

    deltas = compute_deltas(per_lead_a, per_lead_b)

    log.info("FFT analysis on both models")
    lead0_a = pl_a.filter(pl.col("lead_step") == 0).sort("time")
    lead0_b = pl_b.filter(pl.col("lead_step") == 0).sort("time")
    # Polars inner join on time → both arrays aligned and sorted
    a_aligned = lead0_a.join(lead0_b.select(["time"]), on="time", how="inner").sort("time")
    b_aligned = lead0_b.join(lead0_a.select(["time"]), on="time", how="inner").sort("time")
    if len(a_aligned) > 0:
        truth_sig = a_aligned["truth"].to_numpy()
        a_sig = a_aligned["mean"].to_numpy()
        b_sig = b_aligned["mean"].to_numpy()
    else:
        log.warning("No overlap in forecast times; FFT comparison will be uninformative")
        truth_sig = lead0_a["truth"].to_numpy()
        a_sig = lead0_a["mean"].to_numpy()
        b_sig = lead0_b["mean"].to_numpy()

    _, fft_a = fig_fft(truth_sig, a_sig, dt=15.0, wt_col=wt_col)
    _, fft_b = fig_fft(truth_sig, b_sig, dt=15.0, wt_col=wt_col)

    cls, title, body = verdict(deltas, fft_a, fft_b)

    log.info("Building Plotly figures")
    overlay_ts = fig_overlay_timeseries(pl_a, pl_b, args.label_a, args.label_b, wt_col)
    overlay_psd = fig_overlay_psd(truth_sig, a_sig, b_sig, args.label_a, args.label_b)
    delta_bars = fig_per_lead_delta_bars(deltas)

    html = PAGE.format(
        label_a=args.label_a,
        label_b=args.label_b,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        verdict_class=cls,
        verdict_title=title,
        verdict_body=body,
        rows=build_table_rows(deltas),
        delta_bars=delta_bars,
        overlay_ts=overlay_ts,
        overlay_psd=overlay_psd,
    )
    args.output.write_text(html)
    log.info("Wrote %s (%.1f KB)", args.output, args.output.stat().st_size / 1024)


if __name__ == "__main__":
    main()
