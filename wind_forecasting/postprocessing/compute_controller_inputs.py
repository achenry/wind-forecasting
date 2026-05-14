"""Pre-compute controller-ready inputs from a TACTiS-2 forecast + CP intervals.

Produces a single parquet with one row per (time, turbine, lead_step):
    time, turbine, lead_step
    u_mean, v_mean                       point forecast components (m/s)
    ws_mean                              mean of sqrt(u^2+v^2) over the 200 samples (m/s)
    wd_mean_deg                          circular mean of atan2(v,u) over the 200 samples (deg)
    wd_stddev_raw_deg                    circular std of direction samples, no CP (deg)
    wd_stddev_calibrated_deg             circular std after CP per-sample rescaling (deg)
    ws_stddev_raw                        std of speed samples, no CP (m/s)
    ws_stddev_calibrated                 std of speed samples after CP rescaling (m/s)

The CP rescaling preserves the (u,v) correlation structure of the raw samples
because each sample is rescaled around its own mean by the per-component CP
scale factor (upper_cp - lower_cp) / (upper_raw - lower_raw). Equivalent to
applying CP marginally to each component and then propagating to direction
via atan2 of the rescaled joint samples.

Direction convention: standard meteorological — atan2(v, u) is the direction the
wind is blowing TOWARDS (the convention of the underlying TACTiS forecast). If
the controller expects FROM-direction, add 180 deg downstream.
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path

import numpy as np
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("controller-inputs")


def circ_mean_std_deg(angles_rad: np.ndarray, axis: int = -1):
    """Circular mean + std in degrees. Input: angles in radians."""
    s = np.sin(angles_rad).mean(axis=axis)
    c = np.cos(angles_rad).mean(axis=axis)
    mean = np.arctan2(s, c)
    R = np.sqrt(s * s + c * c).clip(1e-12, 1.0)
    std = np.sqrt(-2.0 * np.log(R))  # Mardia & Jupp circular std (radians)
    return np.degrees(mean), np.degrees(std)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecast", type=Path, required=True,
                    help="forecast_194.parquet with 200 samples per (time, sample)")
    ap.add_argument("--intervals", type=Path, required=True,
                    help="calibrated_intervals.parquet (provides per-stratum CP scale factors)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--alpha", type=float, default=0.1,
                    help="alpha level whose CP-derived scale factor is used (default 0.1 = 90% nominal)")
    ap.add_argument("--n-turbines", type=int, default=88)
    args = ap.parse_args()

    log.info("Loading forecast %s", args.forecast)
    fcst = pl.read_parquet(args.forecast)
    # Derive lead_step from time-rank within (test_idx, sample)
    fcst = fcst.sort(["test_idx", "sample", "time"]).with_columns(
        ((pl.col("time") - pl.col("time").min().over(["test_idx", "sample"]))
         .dt.total_seconds() // 15).cast(pl.Int32).alias("lead_step")
    )
    log.info("  rows=%d, n_unique_time=%d, n_samples=%d",
             fcst.height, fcst["time"].n_unique(), fcst["sample"].n_unique())

    log.info("Loading CP intervals %s (alpha=%s)", args.intervals, args.alpha)
    ci = pl.read_parquet(args.intervals).filter(pl.col("alpha") == args.alpha)
    # CP file stores time as f64 (unix-ns); forecast file has datetime[ns]. Align.
    if ci.schema["time"] != pl.Datetime("ns"):
        ci = ci.with_columns(pl.col("time").cast(pl.Int64).cast(pl.Datetime("ns")))
    # Per (target_col, lead_step, time): the scale factor for CP
    ci = ci.with_columns([
        (pl.col("upper_cp") - pl.col("lower_cp")).alias("cp_width"),
        (pl.col("upper_raw") - pl.col("lower_raw")).alias("raw_width"),
    ])
    # Avoid divide-by-zero on degenerate (zero-range) targets
    ci = ci.with_columns(
        pl.when(pl.col("raw_width") > 1e-9)
          .then(pl.col("cp_width") / pl.col("raw_width"))
          .otherwise(1.0)
          .alias("scale")
    )
    log.info("  intervals rows=%d, scale median=%.3f",
             ci.height, float(ci["scale"].median()))

    out_rows = []
    for wt in range(1, args.n_turbines + 1):
        if wt % 10 == 0:
            log.info("  turbine %d/%d", wt, args.n_turbines)
        u_col = f"ws_horz_wt{wt:03d}"
        v_col = f"ws_vert_wt{wt:03d}"
        u_target = f"ws_horz_wt{wt:03d}"
        v_target = f"ws_vert_wt{wt:03d}"

        # Pull all 200 samples for this turbine, all (test_idx, lead, time)
        sub = (fcst.select(["test_idx", "sample", "lead_step", "time", u_col, v_col])
               .rename({u_col: "u", v_col: "v"}))

        # Per (test_idx, lead_step, time): collect the 200 samples
        agg = (sub.group_by(["test_idx", "lead_step", "time"])
               .agg([pl.col("u"), pl.col("v")]))  # list columns of length 200

        # Join the CP scale per (target, lead, time) — u side
        scale_u = (ci.filter(pl.col("target_col") == u_target)
                   .select(["lead_step", "time", "scale"])
                   .rename({"scale": "scale_u"}))
        scale_v = (ci.filter(pl.col("target_col") == v_target)
                   .select(["lead_step", "time", "scale"])
                   .rename({"scale": "scale_v"}))
        agg = agg.join(scale_u, on=["lead_step", "time"], how="left")
        agg = agg.join(scale_v, on=["lead_step", "time"], how="left")
        # Drop rows where CP scale is missing (rare boundary effects)
        agg = agg.filter(pl.col("scale_u").is_not_null() & pl.col("scale_v").is_not_null())

        if agg.height == 0:
            continue

        # Convert list columns to numpy [N_rows, 200]
        u_arr = np.array(agg["u"].to_list(), dtype=np.float64)
        v_arr = np.array(agg["v"].to_list(), dtype=np.float64)
        scale_u_arr = agg["scale_u"].to_numpy()
        scale_v_arr = agg["scale_v"].to_numpy()

        u_mean = u_arr.mean(axis=1)
        v_mean = v_arr.mean(axis=1)

        # Rescale samples around their per-row mean by the per-row scale factor
        u_cal = u_mean[:, None] + (u_arr - u_mean[:, None]) * scale_u_arr[:, None]
        v_cal = v_mean[:, None] + (v_arr - v_mean[:, None]) * scale_v_arr[:, None]

        ws_raw = np.sqrt(u_arr ** 2 + v_arr ** 2)
        ws_cal = np.sqrt(u_cal ** 2 + v_cal ** 2)
        ws_mean = ws_raw.mean(axis=1)
        ws_std_raw = ws_raw.std(axis=1)
        ws_std_cal = ws_cal.std(axis=1)

        # Direction
        theta_raw = np.arctan2(v_arr, u_arr)        # [N, 200] radians
        theta_cal = np.arctan2(v_cal, u_cal)
        wd_mean_deg, wd_std_raw_deg = circ_mean_std_deg(theta_raw, axis=1)
        _, wd_std_cal_deg = circ_mean_std_deg(theta_cal, axis=1)

        for i in range(agg.height):
            out_rows.append({
                "time": agg["time"][i],
                "turbine": wt,
                "lead_step": int(agg["lead_step"][i]),
                "u_mean": float(u_mean[i]),
                "v_mean": float(v_mean[i]),
                "ws_mean": float(ws_mean[i]),
                "ws_stddev_raw": float(ws_std_raw[i]),
                "ws_stddev_calibrated": float(ws_std_cal[i]),
                "wd_mean_deg": float(wd_mean_deg[i]),
                "wd_stddev_raw_deg": float(wd_std_raw_deg[i]),
                "wd_stddev_calibrated_deg": float(wd_std_cal_deg[i]),
            })

    out = pl.DataFrame(out_rows)
    out.write_parquet(args.output)
    log.info("Wrote %s (%d rows)", args.output, out.height)
    log.info("Summary stats (mean over all rows):")
    log.info(f"  ws_mean: {float(out['ws_mean'].mean()):.3f} m/s")
    log.info(f"  wd_stddev_raw_deg: {float(out['wd_stddev_raw_deg'].mean()):.3f} deg")
    log.info(f"  wd_stddev_calibrated_deg: {float(out['wd_stddev_calibrated_deg'].mean()):.3f} deg")
    log.info(f"  calibration_factor (cal/raw direction std median): "
             f"{float((out['wd_stddev_calibrated_deg'] / out['wd_stddev_raw_deg']).median()):.3f}")


if __name__ == "__main__":
    main()
