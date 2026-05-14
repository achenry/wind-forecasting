"""Plotting helpers for CP visualization suite.

Provides:
    select_representative_turbines  — calm/gusty/transitional sampling
    zoom_windows                    — 5 zoom-level slice computation
    regime_from_truth_window        — per-timestep calm/gusty classification
    cp_palette                      — consistent colors across plots
    build_index_html                — auto-generate single-page HTML index
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl


# Color palette: muted but distinguishable
cp_palette = {
    "truth": "#000000",            # black solid
    "forecast_mean": "#1f77b4",    # tab:blue
    "raw_band": "#7fbcd2",         # light blue
    "cp_band": "#ff7f0e",          # tab:orange (uniform face for CP bands)
    "cp_band_levels": {
        0.5: "#ffcc99",   # 50% — pale
        0.2: "#ffaa66",   # 80% — light orange
        0.1: "#ff7f0e",   # 90% — orange (focus)
        0.05: "#cc5500",  # 95% — dark orange
        0.01: "#660000",  # 99% — burgundy
    },
    "regime_calm": "#a8e6a3",      # pastel green
    "regime_gusty": "#ff9999",     # pastel red
    "diagonal": "#888888",         # mid grey for reference diagonal
}


@dataclass(frozen=True)
class TurbineSelection:
    """A representative turbine picked for plotting."""
    target_idx: int     # 0..175
    target_col: str     # e.g. "ws_horz_wt001"
    regime: str         # "calm" | "gusty" | "transitional"
    truth_std: float    # per-turbine truth std on test set


def select_representative_turbines(
    truth_df: pl.DataFrame,
    target_indices: list[int],
    map_target_to_wt: callable,
    k_per_regime: int = 5,
) -> list[TurbineSelection]:
    """Pick K calm + K gusty + K transitional turbines by truth std.

    Args:
        truth_df:       The truth DataFrame filtered to one SPLIT.
        target_indices: All target indices in scope (e.g. [0..175]).
        map_target_to_wt: target_idx -> wt_col_name string.
        k_per_regime:   How many to pick per regime.
    """
    stds = []
    for ti in target_indices:
        col = f"target_{ti}"
        if col not in truth_df.columns:
            continue
        s = float(truth_df[col].std())
        stds.append((ti, s))
    stds.sort(key=lambda x: x[1])  # ascending by std
    if len(stds) < 3 * k_per_regime:
        # Not enough turbines — just take what we can
        k_per_regime = max(1, len(stds) // 3)

    n = len(stds)
    calm = stds[:k_per_regime]
    gusty = stds[-k_per_regime:]
    mid_start = (n - k_per_regime) // 2
    transitional = stds[mid_start: mid_start + k_per_regime]

    out: list[TurbineSelection] = []
    for ti, s in calm:
        out.append(TurbineSelection(ti, map_target_to_wt(ti), "calm", s))
    for ti, s in transitional:
        out.append(TurbineSelection(ti, map_target_to_wt(ti), "transitional", s))
    for ti, s in gusty:
        out.append(TurbineSelection(ti, map_target_to_wt(ti), "gusty", s))
    return out


@dataclass(frozen=True)
class ZoomWindow:
    label: str
    description: str
    start_idx: int
    end_idx: int


def zoom_windows(
    n_total: int,
    timesteps_per_window: dict[str, int] | None = None,
) -> list[ZoomWindow]:
    """Return 5 zoom windows from full test period down to a single 60s forecast.

    Defaults (at 15s/timestep):
      L0 full         — all n_total points
      L1 1-day        — first 5760 points (24h * 4 quarters/min * 60 min)
      L2 1-hour       — first 240 points (60 min * 4)
      L3 10-min       — first 40 points (10 min * 4)
      L4 60s window   — first 4 points (1 forecast horizon)
    """
    if timesteps_per_window is None:
        timesteps_per_window = {
            "L0_full":       n_total,
            "L1_1day":       min(n_total, 5760),
            "L2_1hour":      min(n_total, 240),
            "L3_10min":      min(n_total, 40),
            "L4_1forecast":  min(n_total, 4),
        }
    out: list[ZoomWindow] = []
    descriptions = {
        "L0_full":       "Full test period",
        "L1_1day":       "1-day window",
        "L2_1hour":      "1-hour window",
        "L3_10min":      "10-minute window",
        "L4_1forecast":  "Single 60s forecast window",
    }
    # All windows start at midpoint - half_size (so we get a representative middle segment)
    mid = n_total // 2
    for k, w in timesteps_per_window.items():
        start = max(0, mid - w // 2)
        end = min(n_total, start + w)
        out.append(ZoomWindow(k, descriptions[k], start, end))
    return out


def regime_from_truth_window(
    truth: np.ndarray,
    window_size: int = 40,
    calm_pct: float = 33.0,
    gusty_pct: float = 67.0,
) -> np.ndarray:
    """Label each timestep 'calm'/'transitional'/'gusty' by rolling truth std.

    Args:
        truth:        (n,) ground truth time series
        window_size:  rolling window for std (default 40 = 10 min at 15s)
        calm_pct, gusty_pct: percentile thresholds of per-timestep std

    Returns:
        (n,) array of string labels.
    """
    n = len(truth)
    if n < window_size:
        return np.full(n, "transitional", dtype="<U13")
    # Rolling std
    s = np.empty(n)
    half = window_size // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half)
        s[i] = float(np.std(truth[lo:hi]))
    p_low = np.percentile(s, calm_pct)
    p_hi = np.percentile(s, gusty_pct)
    labels = np.where(s < p_low, "calm",
                      np.where(s > p_hi, "gusty", "transitional"))
    return labels


def binomial_ci(n: int, p_hat: float, conf: float = 0.95) -> tuple[float, float]:
    """Wilson-style binomial CI for empirical coverage."""
    from scipy.stats import norm  # noqa: WPS433
    z = norm.ppf(1.0 - (1.0 - conf) / 2.0)
    denom = 1.0 + (z**2) / n
    centre = (p_hat + z**2 / (2.0 * n)) / denom
    halfw = (z * np.sqrt(p_hat * (1.0 - p_hat) / n + z**2 / (4.0 * n**2))) / denom
    return max(0.0, centre - halfw), min(1.0, centre + halfw)


def build_index_html(
    output_dir: Path,
    groups: dict[str, list[Path]],
    title: str = "TACTiS-2 Phase 0i-B Conformal Prediction Suite",
) -> Path:
    """Generate a single-page index.html linking all plots in `groups`.

    `groups` keys are section names; values are lists of plot paths (relative
    to output_dir or absolute — both converted to relative for the HTML).
    """
    parts: list[str] = []
    parts.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
    parts.append(f"<title>{title}</title>")
    parts.append("""<style>
        body { font-family: sans-serif; max-width: 1400px; margin: 1em auto; }
        h1 { border-bottom: 3px solid #333; }
        h2 { color: #555; margin-top: 2em; border-bottom: 1px solid #aaa; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 8px; }
        .grid img { width: 100%; height: auto; border: 1px solid #ccc; }
        .caption { font-size: 0.8em; color: #666; text-align: center; }
    </style>""")
    parts.append("</head><body>")
    parts.append(f"<h1>{title}</h1>")
    for section, paths in groups.items():
        parts.append(f"<h2>{section}</h2>")
        parts.append("<div class='grid'>")
        for p in paths:
            try:
                rel = Path(p).relative_to(output_dir)
            except ValueError:
                rel = Path(p)
            parts.append(f"<div><a href='{rel}'><img src='{rel}'/></a>"
                         f"<div class='caption'>{rel.name}</div></div>")
        parts.append("</div>")
    parts.append("</body></html>")
    target = output_dir / "index.html"
    target.write_text("\n".join(parts))
    return target


def map_target_to_wt_col(target_idx: int, n_turbines: int = 88) -> str:
    """Same mapping as apply_cp_to_phase0i_b — kept here for convenience."""
    if target_idx < n_turbines:
        return f"ws_horz_wt{target_idx + 1:03d}"
    return f"ws_vert_wt{target_idx - n_turbines + 1:03d}"
