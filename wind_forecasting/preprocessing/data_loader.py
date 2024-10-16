### This file contains class and method to: 
### - load the scada data, 
### - convert timestamps to datetime objects
### - convert circular measurements to sinusoidal measurements
### - normalize data

import glob
import os
import logging
from concurrent.futures import ProcessPoolExecutor
import time

import netCDF4 as nc
import polars as pl
import polars.selectors as cs
#from pandas import to_datetime as pd_to_datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor

from pandas import to_datetime as pd_to_datetime # INFO: @Juan 10/16/24 Added pd_to_datetime to avoid conflict with polars to_datetime

SECONDS_PER_MINUTE = np.float64(60)
SECONDS_PER_HOUR = SECONDS_PER_MINUTE * 60
SECONDS_PER_DAY = SECONDS_PER_HOUR * 24
SECONDS_PER_YEAR = SECONDS_PER_DAY * 365  # non-leap year, 365 days

# INFO: @Juan 10/02/24 Set Logging up
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    """_summary_
       - load the scada data, 
       - convert timestamps to datetime objects
       - convert circular measurements to sinusoidal measurements
       - normalize data 
    """
    def __init__(self, data_dir: str = r"/Users/$USER/Documents/toolboxes/wind_forecasting/examples/data",
                 file_signature: str = "kp.turbine.z02.b0.*.*.*.nc",
                 save_path: str = r"/Users/$USER/Documents/toolboxes/wind_forecasting/examples/data/kp.turbine.zo2.b0.raw.parquet",
                 multiprocessor: str | None = None,
                 chunk_size: int = 100000, # INFO: @Juan 10/16/24 Added arg for chunk size. 
                 features: list[str] = None,
                 dt: int | None = 5,
                 data_format: str = "netcdf", # INFO:@Juan 10/14/24  Added arg for data format. Either "netcdf" or "csv" 
                 column_mapping: dict = None): # INFO:@Juan 10/14/24 Added arg for column mapping of csv files. 
        
        self.save_path = save_path
        self.multiprocessor = multiprocessor
        self.dt = dt
        self.data_format = data_format.lower()
        self.column_mapping = column_mapping or {}
        self.chunk_size = chunk_size
        self.features = features or ["time", "turbine_id", "turbine_status", "turbine_availability", "wind_direction", "wind_speed", "power_output"]

        # Get all the wts in the folder @Juan 10/16/24 used os.path.join for OS compatibility
        self.file_paths = glob.glob(os.path.join(data_dir, file_signature))
        if not self.file_paths:
            raise FileExistsError(f"⚠️ File with signature {file_signature} in directory {data_dir} doesn't exist.")

    # INFO: @Juan 10/14/24 Added method to read multiple files based on the file signature. 
    def read_multi_files(self) -> pl.LazyFrame | None:
        start_time = time.time() # INFO: @Juan 10/16/24 Debbuging time measurements
        logging.info(f"✅ Starting read_multi_files with {len(self.file_paths)} files")
        
        if self.multiprocessor:
            executor = MPICommExecutor(MPI.COMM_WORLD, root=0) if self.multiprocessor == "mpi" else ProcessPoolExecutor()
            logging.info(f"🔧 Using {self.multiprocessor} executor with {MPI.COMM_WORLD.Get_size()} processes")
            with executor as exec:
                futures = [exec.submit(self._read_single_file, file_path) for file_path in self.file_paths]
                df_query = [fut.result() for fut in futures if fut.result() is not None]
        else:
            logging.info("🔧 Using single process executor")
            df_query = [self._read_single_file(file_path) for file_path in self.file_paths if self._read_single_file(file_path) is not None]

        logging.info(f"✅ Finished reading individual files. Time elapsed: {time.time() - start_time:.2f} s")

        if (self.multiprocessor == "mpi" and (comm_rank := MPI.COMM_WORLD.Get_rank()) == 0) \
            or (self.multiprocessor != "mpi") or (self.multiprocessor is None):

            if [df for df in df_query if df is not None]:
                # logging.info("🔄 Starting concatenation of DataFrames")
                concat_start = time.time()
                df_query = pl.concat([df for df in df_query if df is not None]).lazy()
                logging.info(f"✅ Finished concatenation. Time elapsed: {time.time() - concat_start:.2f} s")

                # logging.info("🔄 Starting sorting")
                sort_start = time.time()
                df_query = df_query.sort(["turbine_id", "time"])
                logging.info(f"✅ Finished sorting. Time elapsed: {time.time() - sort_start:.2f} s")
          
                # INFO: @Juan 10/16/24 Separate method for writing parquet file
                self._write_parquet(df_query)
                
                logging.info(f"⏱️ Total time elapsed: {time.time() - start_time:.2f} s")
                return df_query #INFO: @Juan 10/16/24 Added .lazy() to the return statement to match the expected return type. Is this necessary?
        
            logging.warning("⚠️ No data frames were created.")
            return None

        logging.info(f"⏱️ Total time elapsed: {time.time() - start_time:.2f} s")
        
    def _write_parquet(self, df_query: pl.LazyFrame):
        logging.info("🔄 Starting Parquet write")
        write_start = time.time()
        
        try:
            # Collect a small sample to check for issues
            sample = df_query.limit(10).collect()
            logging.info(f"Total rows in df_query: {df_query.collect().shape[0]}")
            logging.info(f"Sample data types: {sample.dtypes}")
            logging.info(f"Sample data:\n{sample}")

            # Collect the entire LazyFrame into a DataFrame
            df = df_query.collect()
            
            # Write the DataFrame to a Parquet file
            df.write_parquet(
                self.save_path, 
                row_group_size=100000  # Adjust as needed
            )
            
            logging.info(f"✅ Finished writing Parquet. Time elapsed: {time.time() - write_start:.2f} s")
        except Exception as e:
            logging.error(f"❌ Error during Parquet write: {str(e)}")
            logging.info("Attempting to write as CSV instead...")
            try:
                csv_path = self.save_path.replace('.parquet', '.csv')
                df_query.collect().write_csv(csv_path)
                logging.info(f"✅ Successfully wrote data as CSV to {csv_path}")
            except Exception as csv_e:
                logging.error(f"❌ Error during CSV write: {str(csv_e)}")

    # INFO: @Juan 10/14/24 Added method to read single file based on the file signature. 
    def _read_single_file(self, file_path: str) -> pl.LazyFrame:
        start_time = time.time()
        # logging.info(f"Starting to process {file_path}")
        try:
            if self.data_format == "netcdf":
                result = self._read_single_netcdf(file_path)
            elif self.data_format == "csv":
                result = self._read_single_csv(file_path)
            else:
                raise ValueError(f"❌ Unsupported data format: {self.data_format}")
            
            logging.info(f"✅ Processed {file_path}. Time: {time.time() - start_time:.2f} s")
            return result
        except Exception as e:
            logging.error(f"❌ Error processing {file_path}: {e}")
            return None

    # INFO: @Juan 10/16/24 Added method to read single netcdf file. Use pl.Series to convert the time variable to a polars series. and combined time extraction operations into a single line to remove intermediate variables. Removed try/except block as it is done in the calling method (_read_single_file())
    def _read_single_netcdf(self, file_path: str) -> pl.LazyFrame:
        with nc.Dataset(file_path, 'r') as dataset:
            #TODO: @Juan 10/14/24 Check if this is correct and if pandas can be substituted for polars
            time_var = dataset.variables['date']
            time = pd_to_datetime(nc.num2date(times=time_var[:], 
                                              units=time_var.units, 
                                              calendar=time_var.calendar, 
                                              only_use_cftime_datetimes=False, 
                                              only_use_python_datetimes=True))
            
            data = {
                'turbine_id': [os.path.basename(file_path).split('.')[-2]] * len(time),
                'time': time,
                'turbine_status': dataset.variables['WTUR.TurSt'][:],
                'wind_direction': dataset.variables['WMET.HorWdDir'][:],
                'wind_speed': dataset.variables['WMET.HorWdSpd'][:],
                'power_output': dataset.variables['WTUR.W'][:],
                'nacelle_direction': dataset.variables['WNAC.Dir'][:]
            }

            # remove the rows with all nans (corresponding to rows where excluded columns would have had a value)
            # and bundle all values corresponding to identical time stamps together
            df_query = pl.DataFrame(data).lazy().group_by("turbine_id", "time").agg(
                cs.numeric().drop_nans().first()
            ).fill_nan(None)
            
            df_query = self.reduce_features(df_query)
            if self.dt is not None:
                df_query = self.resample(df_query)
            
            del data  # Free up memory

            # logging.info(f"Processed {file_path}")
            return df_query

    def _read_single_csv(self, file_path: str) -> pl.LazyFrame:
        df = pl.read_csv(file_path, low_memory=False)
        
        # Apply column mapping if provided
        if self.column_mapping:
            df = df.rename(self.column_mapping)
        
        # Convert time column to datetime
        df = df.with_columns(pl.col("time").str.to_datetime())
        
        # Group by turbine_id and time if necessary
        if "turbine_id" in df.columns:
            df = df.group_by("turbine_id", "time").agg(
                cs.numeric().drop_nans().first()
            ).fill_nan(None)
        
        df = self.reduce_features(df)
        if self.dt is not None:
            df = self.resample(df)
        
        logging.info(f"Processed {file_path}")
        return df.lazy()

    def print_netcdf_structure(self, file_path) -> None: #INFO: @Juan 10/02/24 Changed print to logging
        try:
            with nc.Dataset(file_path, 'r') as dataset:
                logging.info(f"NetCDF File: {os.path.basename(file_path)}")
                logging.info("\nGlobal Attributes:")
                for attr in dataset.ncattrs():
                    logging.info(f"  {attr}: {getattr(dataset, attr)}")

                logging.info("\nDimensions:")
                for dim_name, dim in dataset.dimensions.items():
                    logging.info(f"  {dim_name}: {len(dim)}")

                logging.info("\nVariables:")
                for var_name, var in dataset.variables.items():
                    logging.info(f"  {var_name}:")
                    logging.info(f"    Dimensions: {var.dimensions}")
                    logging.info(f"    Shape: {var.shape}")
                    logging.info(f"    Data type: {var.dtype}")
                    logging.info("    Attributes:")
                    for attr in var.ncattrs():
                        logging.info(f"      {attr}: {getattr(var, attr)}")

        except Exception as e:
            logging.error(f"Error reading NetCDF file: {e}")

    def convert_time_to_sin(self, df) -> pl.LazyFrame:
        """_summary_
            convert timestamp to cosine and sinusoidal components
        Returns:
            pl.LazyFrame: _description_
        """
        if self.df is None:
            raise ValueError("⚠️ Data not loaded > call read_multi_netcdf() first.")
        
        self.df = self.df.with_columns([
            pl.col('time').dt.hour().alias('hour'),
            pl.col('time').dt.ordinal_day().alias('day'),
            pl.col('time').dt.year().alias('year'),
        ])

        # Normalize time features using sin/cos for capturing cyclic patterns using Polars vectorized operations
        self.df = self.df.with_columns([
            (2 * np.pi * pl.col('hour') / 24).sin().alias('hour_sin'),
            (2 * np.pi * pl.col('hour') / 24).cos().alias('hour_cos'),
            (2 * np.pi * pl.col('day') / 365).sin().alias('day_sin'),
            (2 * np.pi * pl.col('day') / 365).cos().alias('day_cos'),
            (2 * np.pi * pl.col('year') / 365).sin().alias('year_sin'),
            (2 * np.pi * pl.col('year') / 365).cos().alias('year_cos'),
        ])

        return self.df

    def reduce_features(self, df) -> pl.LazyFrame:
        """_summary_

        Returns:
            pl.LazyFrame: _description_
        """
        df = df.select(self.features).filter(pl.any_horizontal(cs.numeric().is_not_null()))
        return df

    def resample(self, df) -> pl.LazyFrame:
        return df.with_columns(pl.col("time").dt.round(f"{self.dt}s").alias("time"))\
                 .group_by("turbine_id", "time").agg(cs.numeric().drop_nulls().first()).sort(["turbine_id", "time"])

    def normalize_features(self, df) -> pl.LazyFrame:
        """_summary_
            use minmax scaling to normalize non-temporal features
        Returns:
            pl.LazyFrame: _description_
        """
        if df is None:
            raise ValueError("Data not loaded > call read_multi_netcdf() first.")
        
        features_to_normalize = [col for col in self.df.columns
                                 if all(c not in col for c in ['time', 'hour', 'day', 'year'])]
        
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(self.df.select(features_to_normalize).to_numpy())
        normalized_df = pl.DataFrame(normalized_data, schema=features_to_normalize)
        
        df = df.drop(features_to_normalize).hstack(normalized_df)
        return df
    
    def create_sequences(self, df, target_turbine: str, 
                         features: list[str] | None = None, 
                         sequence_length: int = 600, 
                         prediction_horizon: int = 240) -> tuple[np.ndarray, np.ndarray, list[str], int, int]:
        
        features = [col for col in df.columns if col not in [f'TurbineWindMag_{target_turbine}_u', f'TurbineWindMag_{target_turbine}_v']]
        y_columns = [f'TurbineWindMag_{target_turbine}_u', f'TurbineWindMag_{target_turbine}_v']
        
        X_data = df.select(features).collect(streaming=True).to_numpy()
        y_data = df.select(y_columns).collect(streaming=True).to_numpy()
        
        X = np.array([X_data[i:i + sequence_length] for i in range(len(X_data) - sequence_length - prediction_horizon + 1)])
        y = np.array([y_data[i:i + prediction_horizon] for i in range(sequence_length, len(y_data) - prediction_horizon + 1)])
        
        return X, y, features, sequence_length, prediction_horizon

    # INFO: @Juan 10/14/24 Added method to format SMARTEOLE data. (TEMPORARY)
    def format_smarteole_data(self, df: pl.LazyFrame) -> pl.LazyFrame:
        # Implement the formatting logic for SMARTEOLE data
        # This method should apply the necessary transformations as shown in the format_dataframes function
        
        # Example of some transformations:
        df = df.with_columns([
            (pl.col("is_operation_normal_000").cast(pl.Boolean()).not_()).alias("is_operation_normal_000"),
            (pl.col("is_operation_normal_001").cast(pl.Boolean()).not_()).alias("is_operation_normal_001"),
            # ... (repeat for other turbines)
        ])
        
        df = df.with_columns([
            pl.when(pl.col("control_mode") == 0).then("baseline")
              .when(pl.col("control_mode") == 1).then("controlled")
              .otherwise(pl.col("control_mode")).alias("control_mode")
        ])
        
        # Add more transformations as needed
        
        return df

    # INFO: @Juan 10/14/24 Added method to load and process data. (TEMPORARY)
    def load_and_process_data(self) -> pl.LazyFrame:
        df = self.read_multi_files()
        if df is not None:
            df = self.format_smarteole_data(df)
            df = self.convert_time_to_sin(df)
            df = self.normalize_features(df)
        return df

if __name__ == "__main__":
    from sys import platform
    
    if platform == "darwin":
        # DATA_DIR = "/Users/$USER/Documents/toolboxes/wind_forecasting/examples/data"
        DATA_DIR = "examples/inputs/awaken_data"
        # PL_SAVE_PATH = "/Users/$USER/Documents/toolboxes/wind_forecasting/examples/data/kp.turbine.zo2.b0.raw.parquet"
        PL_SAVE_PATH = "examples/inputs/awaken_data/processed/kp.turbine.zo2.b0.raw.parquet"
        # FILE_SIGNATURE = "kp.turbine.z02.b0.20220301.*.*.nc"
        FILE_SIGNATURE = "kp.turbine.z02.b0.*.*.*.nc"
        MULTIPROCESSOR = "cf"
        TURBINE_INPUT_FILEPATH = "/Users/$USER/Documents/toolboxes/wind_forecasting/examples/inputs/ge_282_127.yaml"
        FARM_INPUT_FILEPATH = "/Users/$USER/Documents/toolboxes/wind_forecasting/examples/inputs/gch_KP_v4.yaml"
    elif platform == "linux":
        # DATA_DIR = "/pl/active/paolab/awaken_data/kp.turbine.z02.b0/"
        DATA_DIR = "examples/inputs/awaken_data"
        # PL_SAVE_PATH = "/scratch/alpine/aohe7145/awaken_data/kp.turbine.zo2.b0.raw.parquet"
        PL_SAVE_PATH = "examples/inputs/awaken_data/processed/kp.turbine.zo2.b0.raw.parquet"
        # FILE_SIGNATURE = "kp.turbine.z02.b0.20220301.*.*.nc"
        # FILE_SIGNATURE = "kp.turbine.z02.b0.*.*.*.nc"
        FILE_SIGNATURE = "kp.turbine.z02.b0.20220101.000000.wt001.nc"
        MULTIPROCESSOR = "mpi"
        # TURBINE_INPUT_FILEPATH = "/projects/$USER/toolboxes/wind-forecasting/examples/inputs/ge_282_127.yaml"
        # FARM_INPUT_FILEPATH = "/projects/$USER/toolboxes/wind-forecasting/examples/inputs/gch_KP_v4.yaml"
        TURBINE_INPUT_FILEPATH = "examples/inputs/ge_282_127.yaml"
        FARM_INPUT_FILEPATH = "examples/inputs/gch_KP_v4.yaml"
    
    DT = 5
    RUN_ONCE = (MULTIPROCESSOR == "mpi" and (comm_rank := MPI.COMM_WORLD.Get_rank()) == 0) or (MULTIPROCESSOR != "mpi") or (MULTIPROCESSOR is None)
    
    if FILE_SIGNATURE.endswith(".nc"):
        data_format = "netcdf"
    elif FILE_SIGNATURE.endswith(".csv"):
        data_format = "csv"
    else:
        raise ValueError("Invalid file signature. Please specify either '*.nc' or '*.csv'.")

    if RUN_ONCE:
        try:
            data_loader = DataLoader(
                data_dir=DATA_DIR,
                file_signature=FILE_SIGNATURE,
                save_path=PL_SAVE_PATH,
                multiprocessor=MULTIPROCESSOR,
                dt=DT,
                features=["time", "turbine_id", "turbine_status", "wind_direction", "wind_speed", "power_output", "nacelle_direction"],
                data_format=data_format
            )
            
            if os.path.exists(data_loader.save_path):
                logging.info("🔄 Loading existing Parquet file")
                df_query = pl.scan_parquet(source=data_loader.save_path)
                logging.info("✅ Loaded existing Parquet file successfully")
            else:
                logging.info("🔄 Processing new data files")
                df_query = data_loader.read_multi_files()
                if df_query is not None:
                    try:
                        data_loader._write_parquet(df_query)
                    except Exception as e:
                        logging.error(f"❌ Error during data writing: {str(e)}")
                        logging.info("Attempting to process data in smaller chunks...")
                        chunk_size = 1000000  # Adjust this value as needed
                        for i, chunk in enumerate(df_query.collect(streaming=True).iter_batches(batch_size=chunk_size)):
                            chunk_path = f"{data_loader.save_path}_chunk_{i}.parquet"
                            pl.DataFrame(chunk).write_parquet(chunk_path)
                            logging.info(f"✅ Wrote chunk {i} to {chunk_path}")
                
            logging.info("✅ Script completed successfully")
        except Exception as e:
            logging.error(f"❌ An error occurred: {str(e)}")
