import polars as pl
import polars.selectors as cs
import os

def main():
    file_directory = "/scratch/ahenry/kfs3-flash/"
    file_directory = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data/awaken_data"
    filenames = ["awaken_processed_imputed.parquet", 
                 "awaken_processed_smoothed_butterworth.parquet",
                 "awaken_processed_smoothed_normalized.parquet",
                 "awaken_processed_split.parquet"]
    for fn in filenames:
        fp = os.path.join(file_directory, fn)
        pl.read_parquet(fp).with_columns(cs.float().cast(pl.Float32)).write_parquet(fp)
    
if __name__ == "__main__":
    main()