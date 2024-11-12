import polars as pl
import os
x = pl.scan_parquet("/projects/ssc/ahenry/wind_forecasting/awaken_data/kp.turbine.zo2.b0.parquet")
x.sink_parquet(os.path.join("/tmp/scratch", os.environ["SLURM_JOB_ID"], "loaded_data.parquet"))
