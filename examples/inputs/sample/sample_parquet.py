import polars as pl
import logging
import os
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
INPUT_PATH = "examples/inputs/sample/sample.csv"
OUTPUT_DIR = "examples/inputs/sample"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "sample.parquet")

# Feature names to extract from the sample data
FEATURES = ["turbine_status", "wind_direction", "wind_speed", "power_output", "nacelle_direction"]

def process_sample_csv():
    # Ensure output directory exists
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    try:
        # Read CSV with the unnamed first column
        logging.info(f"🔄 Reading CSV file: {INPUT_PATH}")
        df = pl.read_csv(INPUT_PATH)
        
        # Drop the unnamed index column
        df = df.drop("")
        
        # Convert time column to datetime
        df = df.with_columns([
            pl.col("time").str.to_datetime().alias("time")
        ])
        
        # Handle any infinite values in numeric columns
        numeric_cols = [col for col in df.columns if col != "time"]
        df = df.with_columns([
            pl.col(col).map_elements(lambda x: None if np.isinf(x) else x)
            for col in numeric_cols
        ])
        
        # Forward/backward fill missing values with a reasonable limit
        ffill_limit = 10  # Adjust as needed
        df = df.with_columns([
            pl.col(col)
            .fill_null(strategy="forward", limit=ffill_limit)
            .fill_null(strategy="backward", limit=ffill_limit)
            for col in numeric_cols
        ])
        
        # Remove rows where all numeric columns are null
        df = df.filter(
            pl.any_horizontal(pl.exclude("time").is_not_null())
        )
        
        logging.info(f"📊 Processed dataframe shape: {df.shape}")
        logging.info(f"📊 Columns: {df.columns}")
        
        # Write to Parquet
        logging.info(f"💾 Writing to Parquet: {OUTPUT_PATH}")
        df.write_parquet(OUTPUT_PATH)
        logging.info("✅ Successfully created Parquet file")
        
        # Verify the Parquet file
        df_verify = pl.read_parquet(OUTPUT_PATH)
        logging.info(f"✅ Verification successful. Parquet shape: {df_verify.shape}")
        
    except Exception as e:
        logging.error(f"❌ Error processing file: {str(e)}")
        raise

if __name__ == "__main__":
    process_sample_csv()