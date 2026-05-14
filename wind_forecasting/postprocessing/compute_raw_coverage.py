"""Compute pre-CP raw empirical coverage and PINAW directly from the 200
   forecast samples, on the real validation split.

This is the model's own probabilistic calibration before CP rescues it.
Output is a CSV with one row per (target_col, lead_step), columns:
  raw_coverage_aXXX, raw_width_aXXX  for alpha in {0.5, 0.2, 0.1, 0.05, 0.01}.
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path

import numpy as np
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("raw-cov")

ALPHAS = [0.01, 0.05, 0.1, 0.2, 0.5]
N_TURB = 88


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecast", type=Path, required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--split-id", type=int, default=194)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    log.info("Loading forecast %s ...", args.forecast)
    fcst = pl.read_parquet(args.forecast)
    log.info("  shape: %d rows × %d cols", fcst.height, len(fcst.columns))

    log.info("Loading truth %s (SPLIT%d) ...", args.truth, args.split_id)
    truth = (pl.read_parquet(args.truth)
             .filter(pl.col("item_id") == f"SPLIT{args.split_id}")
             .sort("time"))
    log.info("  truth rows: %d", truth.height)

    # Each test_idx has 4 future timestamps × 200 samples × 176 wt cols
    # We need: per (test_idx, lead, target_col): the (200,) sample array
    #          per (test_idx, lead, target_col): the truth at that future time

    # Step 1 — derive lead_step from time-rank within test_idx
    log.info("Annotating lead_step...")
    fcst = fcst.sort(["test_idx", "sample", "time"])
    fcst = fcst.with_columns(
        ((pl.col("time") - pl.col("time").min().over(["test_idx", "sample"]))
         .dt.total_seconds() // 15).cast(pl.Int32).alias("lead_step")
    )

    rows = []
    wt_cols = [c for c in fcst.columns if c.startswith("ws_horz_") or c.startswith("ws_vert_")]
    log.info("Processing %d turbine-component columns...", len(wt_cols))

    for tc_idx, tc_col in enumerate(wt_cols):
        # Build target_idx (matches the post-CP CSV: 0..87 horz, 88..175 vert)
        wt_num = int(tc_col.split("wt")[1])
        if tc_col.startswith("ws_horz_"):
            target_idx = wt_num - 1
        else:
            target_idx = wt_num - 1 + N_TURB
        truth_col = f"target_{target_idx}"
        if truth_col not in truth.columns:
            continue

        truth_arr = truth.select(["time", truth_col]).rename({truth_col: "truth"})

        # Pull samples for this turbine + collect quantiles per (test_idx, lead)
        sub = fcst.select(["test_idx", "sample", "lead_step", "time", tc_col]).rename({tc_col: "v"})

        # Quantiles over the 200-sample axis per (test_idx, lead, time)
        q_table = (sub.group_by(["test_idx", "lead_step", "time"])
                   .agg([pl.col("v").quantile(0.005).alias("q0.005"),
                         pl.col("v").quantile(0.025).alias("q0.025"),
                         pl.col("v").quantile(0.05).alias("q0.05"),
                         pl.col("v").quantile(0.1).alias("q0.1"),
                         pl.col("v").quantile(0.25).alias("q0.25"),
                         pl.col("v").quantile(0.75).alias("q0.75"),
                         pl.col("v").quantile(0.9).alias("q0.9"),
                         pl.col("v").quantile(0.95).alias("q0.95"),
                         pl.col("v").quantile(0.975).alias("q0.975"),
                         pl.col("v").quantile(0.995).alias("q0.995")]))
        # Join truth on time
        q_table = q_table.join(truth_arr, on="time", how="inner")

        for L in sorted(q_table["lead_step"].unique().to_list()):
            sl = q_table.filter(pl.col("lead_step") == L)
            if sl.height == 0:
                continue
            t = sl["truth"].to_numpy()
            entry = {"target_idx": target_idx, "target_col": tc_col, "lead_step": int(L), "n": int(sl.height)}
            for a in ALPHAS:
                if a == 0.01:
                    lo, hi = sl["q0.005"].to_numpy(), sl["q0.995"].to_numpy()
                elif a == 0.05:
                    lo, hi = sl["q0.025"].to_numpy(), sl["q0.975"].to_numpy()
                elif a == 0.1:
                    lo, hi = sl["q0.05"].to_numpy(), sl["q0.95"].to_numpy()
                elif a == 0.2:
                    lo, hi = sl["q0.1"].to_numpy(), sl["q0.9"].to_numpy()
                else:  # 0.5
                    lo, hi = sl["q0.25"].to_numpy(), sl["q0.75"].to_numpy()
                cov = float(np.mean((t >= lo) & (t <= hi)))
                wid = float(np.mean(hi - lo))
                miss = np.maximum.reduce([np.zeros_like(t), lo - t, t - hi])
                winkler = float(np.mean((hi - lo) + (2.0 / a) * miss))
                entry[f"raw_coverage_a{a:.3f}"] = cov
                entry[f"raw_width_a{a:.3f}"] = wid
                entry[f"raw_winkler_a{a:.3f}"] = winkler
            rows.append(entry)

        if (tc_idx + 1) % 20 == 0:
            log.info("  %d / %d", tc_idx + 1, len(wt_cols))

    out = pl.from_dicts(rows)
    out.write_csv(args.output)
    log.info("Wrote %s (%d rows)", args.output, out.height)


if __name__ == "__main__":
    main()
