### This file contains class and method to: 
### - load the scada data, 
### - convert timestamps to datetime objects
### - convert circular measurements to sinusoidal measurements
### - normalize data

import glob
import os
import logging
import re

import multiprocessing
import time

import netCDF4 as nc
import polars as pl
import polars.selectors as cs
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor
from concurrent.futures import ProcessPoolExecutor

SECONDS_PER_MINUTE = np.float64(60)
SECONDS_PER_HOUR = SECONDS_PER_MINUTE * 60
SECONDS_PER_DAY = SECONDS_PER_HOUR * 24
SECONDS_PER_YEAR = SECONDS_PER_DAY * 365  # non-leap year, 365 days
FFILL_LIMIT = 10 * SECONDS_PER_MINUTE 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
JOIN_CHUNK = 100 #int(2000)

class DataLoader:
    """_summary_
       - load the scada data, 
       - convert timestamps to datetime objects
       - convert circular measurements to sinusoidal measurements
       - normalize data 
    """
    def __init__(self, 
                 data_dir: str = r"/Users/$USER/Documents/toolboxes/wind_forecasting/examples/data",
                 file_signature: str = "kp.turbine.z02.b0.*.*.*.nc",
                 save_path: str = r"/Users/$USER/Documents/toolboxes/wind_forecasting/examples/data/kp.turbine.zo2.b0.raw.parquet",
                 multiprocessor: str | None = None,
                 chunk_size: int = 100000, 
                 desired_feature_types: list[str] = None,
                 dt: int | None = 5,
                 ffill_limit: int | None = None, 
                 data_format: str = "netcdf", 
                 column_mapping: dict = None, 
                 wide_format: bool = True):
        
        self.data_dir = data_dir
        self.save_path = save_path
        self.multiprocessor = multiprocessor
        self.dt = dt
        self.data_format = data_format.lower()
        self.column_mapping = column_mapping or {}
        self.chunk_size = chunk_size
        self.desired_feature_types = desired_feature_types or ["time", "turbine_id", "turbine_status", "wind_direction", "wind_speed", "power_output", "nacelle_direction"]
        self.wide_format = wide_format
        self.ffill_limit = ffill_limit
        
        # Get all the wts in the folder @Juan 10/16/24 used os.path.join for OS compatibility
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, file_signature)))
    
    def read_multi_files(self) -> pl.LazyFrame | None:
        if self.multiprocessor is not None:
            if self.multiprocessor == "mpi":
                executor = MPICommExecutor(MPI.COMM_WORLD, root=0)
            else:  # "cf" case
                executor = ProcessPoolExecutor()
            with executor as ex:
                if ex is not None:
                    if not self.file_paths:
                        raise FileExistsError(f"⚠️ File with signature {self.file_signature} in directory {self.data_dir} doesn't exist.")
                    
                    futures = [ex.submit(self._read_single_file, f, file_path) for f, file_path in enumerate(self.file_paths)]
                    df_query = [fut.result() for fut in futures]
                    df_query = [(self.file_paths[d], df) for d, df in enumerate(df_query) if df is not None]

                    if df_query:
                        # INFO: @Juan 11/13/24 Added check for data patterns in the names and also added a check for single files
                        join_start = time.time()
                        logging.info(f"✅ Started join of {len(self.file_paths)} files.")
                        
                        # Check if files have date patterns in their names
                        date_pattern = r"\.(\d{8})\."
                        has_date_pattern = any(re.search(date_pattern, fp) for fp, _ in df_query)
                        
                        if has_date_pattern:
                            unique_file_timestamps = sorted(set(re.findall(date_pattern, fp)[0] for fp,_ in df_query 
                                                             if re.search(date_pattern, fp)))
                            df_query = [self._join_dfs(ts, [df for filepath, df in df_query if ts in filepath]) 
                                      for ts in unique_file_timestamps]
                        else:
                            # For single file or files without timestamps, just get the dataframes
                            df_query = [df for _, df in df_query]
                            if len(df_query) == 1:
                                df_query = df_query[0]  # If single file, no need to join
                            else:
                                df_query = pl.concat(df_query, how="diagonal").group_by("time").agg(cs.numeric().mean())
                            
                            # Write directly to final parquet
                            df_query.collect().write_parquet(self.save_path, statistics=False)
                            return pl.scan_parquet(self.save_path)

                        for ts, df in zip(unique_file_timestamps, df_query):
                            # print(ts, df.collect(), sep="\n")
                            df.collect().write_parquet(self.save_path.replace(".parquet", f"_{ts}.parquet"), statistics=False)
                            logging.info(f"Finished writing parquet {ts}")

                        logging.info(f"🔗 Finished join. Time elapsed: {time.time() - join_start:.2f} s")

                        concat_start = time.time()
                        df_query = [pl.scan_parquet(self.save_path.replace(".parquet", f"_{ts}.parquet")) 
                                            for ts in unique_file_timestamps]
                        df_query = pl.concat(df_query, how="diagonal").group_by("time").agg(cs.numeric().mean())
                        logging.info(f"🔗 Finished concat. Time elapsed: {time.time() - concat_start:.2f} s")

                        logging.info(f"Started sorting.")
                        df_query = df_query.sort("time")
                        logging.info(f"Finished sorting.")

                        logging.info(f"Started resampling.") 
                        full_datetime_range = df_query.select(pl.datetime_range(
                            start=df_query.select("time").min().collect().item(),
                            end=df_query.select("time").max().collect().item(),
                            interval=f"{self.dt}s", time_unit=df_query.collect_schema()["time"].time_unit).alias("time"))
                            
                        df_query = full_datetime_range.join(df_query, on="time", how="left").collect(streaming=True).lazy() # NOTE: @Aoife 10/18 make sure all time stamps are included, to interpolate continuously later
                        logging.info(f"Finished resampling.") 

                        logging.info(f"Started forward/backward fill.") 
                        df_query = df_query.fill_null(strategy="forward").fill_null(strategy="backward") # NOTE: @Aoife for KP data, need to fill forward null gaps, don't know about Juan's data
                        logging.info(f"Finished forward/backward fill.")

                        df_query.collect().write_parquet(self.save_path, statistics=False)

                        for ts in unique_file_timestamps:
                            os.remove(self.save_path.replace(".parquet", f"_{ts}.parquet"))
                    
                    return pl.scan_parquet(self.save_path)
        else:
            logging.info(f"🔧 Using single process executor.")
            if not self.file_paths:
                raise FileExistsError(f"⚠️ File with signature {self.file_signature} in directory {self.data_dir} doesn't exist.")
            df_query = [self._read_single_file(f, file_path) for f, file_path in enumerate(self.file_paths)]
            df_query = [df for df in df_query if df is not None]
            return df_query
    
    def sink_parquet(self, df, filepath):
        try:
            df.sink_parquet(filepath, statistics=False)
        except Exception as e:
            logging.info(f"Failed to sink LazyFrame {filepath}")

        logging.info(f"Finished sinking parquet {filepath}") 

    def _join_dfs(self, file_suffix, dfs):
        # logging.info(f"✅ Started joins for {file_suffix}-th collection of files.") 
        all_cols = set()
        first_df = True
        for d, df in enumerate(dfs):
            # df = df.collect()
            new_cols = [col for col in df.collect_schema().names() if col != "time"]
            if first_df:
                df_query = df
                first_df = False
            else:
                existing_cols = list(all_cols.intersection(new_cols))
                if existing_cols:
                    # data for the turbine contained in this frame has already been added, albeit from another day
                    df_query = df_query.join(df, on="time", how="full", coalesce=True)\
                                        .with_columns([pl.coalesce(col, f"{col}_right").alias(col) for col in existing_cols])\
                                        .select(~cs.ends_with("right"))
                                        # .sink_parquet(temp_save_path)
                else:
                    df_query = df_query.join(df, on="time", how="full", coalesce=True)

            all_cols.update(new_cols)
        
        logging.info(f"🔗 Finished joins for {file_suffix}-th collection of files.")
        return df_query

    def _write_parquet(self, df_query: pl.LazyFrame):
        
        write_start = time.time()
        
        try:
            logging.info("📝 Starting Parquet write")
            
            # Ensure the directory exists
            self._ensure_dir_exists(self.save_path)

            # df_query.sink_ipc(self.save_path)
            df_query.sink_parquet(self.save_path, statistics=False)

            # df = pl.scan_parquet(self.save_path)
            logging.info(f"✅ Finished writing Parquet. Time elapsed: {time.time() - write_start:.2f} s")
            
        except PermissionError:
            logging.error("❌🔒 Permission denied when writing Parquet file. Check file permissions.")
        except OSError as e:
            if e.errno == 28:  # No space left on device
                logging.error("❌💽 No space left on device. Free up some disk space and try again.")
            else:
                logging.error(f"❌💻 OS error occurred: {str(e)}")
        except Exception as e:
            logging.error(f"❌ Error during Parquet write: {str(e)}")
            logging.info("📄 Attempting to write as CSV instead...")
            try:
                csv_path = self.save_path.replace('.parquet', '.csv')
                df_query.sink_csv(csv_path)
                logging.info(f"✅ Successfully wrote data as CSV to {csv_path}")
            except Exception as csv_e:
                logging.error(f"❌ Error during CSV write: {str(csv_e)}")
                
    # INFO: @Juan 10/16/24 Added method to ensure the directory exists.
    def _ensure_dir_exists(self, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"📁 Created directory: {directory}")

    # INFO: @Juan 10/14/24 Added method to read single file based on the file signature. 
    def _read_single_file(self, file_number: int, file_path: str) -> pl.LazyFrame:
        start_time = time.time()
        # logging.info(f"Starting to process {file_path}")
        try:
            if self.data_format == "netcdf":
                logging.info(f"📖 Reading NetCDF file: {file_path}")
                with nc.Dataset(file_path, 'r') as ds:
                    # Assuming time is a dimension in the NetCDF file
                    time_var = ds.variables['time']
                    times = nc.num2date(time_var[:], units=time_var.units, calendar=time_var.calendar)
                    df = pl.DataFrame({'time': times}).lazy()

                    for var_name, variable in ds.variables.items():
                        if var_name != 'time':
                            # Check if the variable has a time dimension
                            if 'time' in variable.dimensions:
                                # Flatten the data array and add it as a new column
                                df = df.with_columns(pl.Series(name=var_name, values=variable[:].flatten()))
                            else:
                                logging.warning(f"⚠️  Variable {var_name} does not have a time dimension. Skipping.")
                
                df = self.map_column_names(df)
                df = self.convert_time_to_datetime(df)
                df = self.convert_to_wide_format(df)
                df = self.convert_circular_to_sinusoidal(df)
                df = self.reduce_features(df)
                df = self.resample(df)
                
                return df
            
            elif self.data_format == "csv":
                logging.info(f"📖 Reading CSV file: {file_path}")
                df = pl.scan_csv(file_path)
                
                # Apply column mapping
                df = self.map_column_names(df)
                
                # Convert time to datetime objects
                df = self.convert_time_to_datetime(df)
                
                # Convert to wide format if necessary
                df = self.convert_to_wide_format(df)
                
                # Convert circular measurements to sinusoidal measurements
                df = self.convert_circular_to_sinusoidal(df)
                
                # Reduce features
                df = self.reduce_features(df)
                
                # Resample
                df = self.resample(df)
                
                return df
            
            else:
                raise ValueError(f"❌ Invalid data format: {self.data_format}. Only 'netcdf' and 'csv' are supported.")

        except FileNotFoundError:
            logging.error(f"❌ File not found: {file_path}")
            return None
        except OSError as e:
            logging.error(f"❌ Error reading file: {file_path}\n{e}")
            return None
        except Exception as e:
            logging.error(f"❌ Error processing file: {file_path}\n{e}")
            return None

    def print_netcdf_structure(self, file_path: str):
        """
        Prints the structure and variables of a NetCDF file.

        Args:
            file_path (str): The path to the NetCDF file.
        """
        try:
            with nc.Dataset(file_path, 'r') as ds:
                logging.info(f"Structure of NetCDF file: {file_path}")
                logging.info("Dimensions:")
                for dim_name, dim in ds.dimensions.items():
                    logging.info(f"  {dim_name}: size={len(dim)}")
                
                logging.info("Variables:")
                for var_name, var in ds.variables.items():
                    logging.info(f"  {var_name}:")
                    logging.info(f"    shape: {var.shape}")
                    logging.info(f"    dtype: {var.dtype}")
                    # Print other attributes if needed
                    for attr_name in var.ncattrs():
                        logging.info(f"    {attr_name}: {getattr(var, attr_name)}")
        except FileNotFoundError:
            logging.error(f"❌ File not found: {file_path}")
        except OSError as e:
            logging.error(f"❌ Error reading file: {file_path}\n{e}")
        except Exception as e:
            logging.error(f"❌ Error processing file: {file_path}\n{e}")

    def map_column_names(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Maps column names in the DataFrame according to the provided column mapping.

        Args:
            df (pl.LazyFrame): The input DataFrame.

        Returns:
            pl.LazyFrame: The DataFrame with renamed columns.
        """
        
        # Filter the column mapping for keys that exist in the DataFrame
        existing_column_mapping = {k: v for k, v in self.column_mapping.items() if k in df.columns}
        
        # Rename columns based on the filtered mapping
        df = df.rename(existing_column_mapping)
        
        return df

    def convert_time_to_datetime(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Converts the 'time' column in the DataFrame to datetime objects.

        Args:
            df (pl.LazyFrame): The input DataFrame.

        Returns:
            pl.LazyFrame: The DataFrame with the 'time' column converted to datetime.
        """
        if 'time' in df.columns:
            # Assuming 'time' is in seconds since epoch
            df = df.with_columns(pl.col('time').cast(pl.Datetime))
        else:
            logging.warning("⚠️ 'time' column not found. Skipping time conversion.")
        return df

    def convert_to_wide_format(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Converts the DataFrame to wide format if it's in long format.

        Args:
            df (pl.LazyFrame): The input DataFrame.

        Returns:
            pl.LazyFrame: The DataFrame in wide format.
        """
        if not self.wide_format:
            # Identify unique turbine IDs
            unique_turbine_ids = df.select(pl.col("turbine_id").unique()).collect().to_series().to_list()

            # Pivot for each turbine ID
            df_list = []
            for turbine_id in unique_turbine_ids:
                df_turbine = df.filter(pl.col("turbine_id") == turbine_id)
                
                # Rename columns to include turbine ID as suffix
                df_turbine = df_turbine.rename({col: f"{col}_{turbine_id}" for col in df_turbine.columns if col not in ["time", "turbine_id"]})
                
                df_list.append(df_turbine)

            # Join all DataFrames on 'time'
            df = df_list[0]
            for df_turbine in df_list[1:]:
                df = df.join(df_turbine, on="time", how="outer")

            # Sort by time
            df = df.sort("time")

        return df

    def convert_circular_to_sinusoidal(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Converts circular measurements (like wind direction) to sinusoidal components.

        Args:
            df (pl.LazyFrame): The input DataFrame.

        Returns:
            pl.LazyFrame: The DataFrame with circular measurements converted to sinusoidal.
        """
        circular_cols = [col for col in df.columns if "direction" in col]
        for col in circular_cols:
            # Convert degrees to radians
            rad_col = f"{col}_rad"
            df = df.with_columns((pl.col(col) * np.pi / 180).alias(rad_col))

            # Convert to sinusoidal components
            sin_col = f"{col}_sin"
            cos_col = f"{col}_cos"
            df = df.with_columns([
                pl.col(rad_col).sin().alias(sin_col),
                pl.col(rad_col).cos().alias(cos_col)
            ])

            # Drop the intermediate radians column
            df = df.drop(rad_col)
        
        # Add hour, day, and year sinusoidal features
        df = df.with_columns([
            (2 * np.pi * pl.col('time').dt.hour() / 24).sin().alias('hour_sin'),
            (2 * np.pi * pl.col('time').dt.hour() / 24).cos().alias('hour_cos'),
            (2 * np.pi * pl.col('time').dt.day() / 31).sin().alias('day_sin'),
            (2 * np.pi * pl.col('time').dt.day() / 31).cos().alias('day_cos'),
            (2 * np.pi * pl.col('time').dt.month() / 12).sin().alias('month_sin'),
            (2 * np.pi * pl.col('time').dt.month() / 12).cos().alias('month_cos'),
            (2 * np.pi * pl.col('time').dt.year() / pl.col('time').dt.year().max()).sin().alias('year_sin'),
            (2 * np.pi * pl.col('time').dt.year() / pl.col('time').dt.year().max()).cos().alias('year_cos'),
        ])

        return df

    def reduce_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Reduce the DataFrame to include only the specified features that exist in the DataFrame.
        """
        # Check which desired features are actually in the DataFrame
        available_features = []
        for feature_type in self.desired_feature_types:
            if any(feature_type in col for col in df.columns):
                available_features.append(feature_type)
            else:
                logging.warning(f"⚠️ Desired feature '{feature_type}' not found in data. Skipping.")

        # Select only the available features
        df = df.select([
            col for col in df.columns if any(feature in col for feature in available_features)
        ])

        # Only filter rows if there are numeric columns
        numeric_cols = df.select(cs.numeric()).columns
        if numeric_cols:
            df = df.filter(pl.any_horizontal(pl.col(numeric_cols).is_not_null()))

        # Update the list of available features after reduction
        self.available_features = available_features
        logging.info(f"Available features after reduction: {self.available_features}")
        logging.info(f"Columns after reduce_features: {df.columns}")
        logging.info(f"Shape after reduce_features: {df.shape}")
        return df

    # INFO: @Juan 10/16/24 Modified resampling method to handle both wide and long formats.
    def resample(self, df) -> pl.LazyFrame:
        if self.wide_format:
            return df.with_columns(pl.col("time").dt.round(f"{self.dt}s").alias("time"))\
                     .group_by("time").agg(cs.numeric().drop_nulls().first()).sort("time")
        else:
            return df.with_columns(pl.col("time").dt.round(f"{self.dt}s").alias("time"))\
                     .group_by("turbine_id", "time").agg(cs.numeric().drop_nulls().first()).sort(["turbine_id", "time"])

    def normalize_features(self, df) -> pl.LazyFrame:
        """_summary_
            use minmax scaling to normalize non-temporal features
        Returns:
            pl.LazyFrame: _description_
        """
        if df is None:
            raise ValueError("⚠️ Data not loaded > call read_multi_netcdf() first.")
        
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

########################################################## INPUTS ##########################################################
if __name__ == "__main__":
    from sys import platform
    RELOAD_DATA = True
    PLOT = False

    DT = 5
    DATA_FORMAT = "csv"

    if platform == "darwin" and DATA_FORMAT == "netcdf":
        DATA_DIR = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data"
        # PL_SAVE_PATH = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data/kp.turbine.zo2.b0.raw.parquet"
        # FILE_SIGNATURE = "kp.turbine.z02.b0.*.*.*.nc"
        PL_SAVE_PATH = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data/short_loaded_data.parquet"
        FILE_SIGNATURE = "kp.turbine.z02.b0.202203*1.*.*.nc"
        MULTIPROCESSOR = "cf"
        TURBINE_INPUT_FILEPATH = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/inputs/ge_282_127.yaml"
        FARM_INPUT_FILEPATH = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/inputs/gch_KP_v4.yaml"
        FEATURES = ["time", "turbine_id", "turbine_status", "wind_direction", "wind_speed", "power_output", "nacelle_direction"]
        WIDE_FORMAT = False
        COLUMN_MAPPING = {"date": "time",
                          "turbine_id": "turbine_id",
                          "WTUR.TurSt": "turbine_status",
                          "WMET.HorWdDir": "wind_direction",
                          "WMET.HorWdSpd": "wind_speed",
                          "WTUR.W": "power_output",
                          "WNAC.Dir": "nacelle_direction"
                          }
    elif platform == "linux" and DATA_FORMAT == "netcdf":
        # DATA_DIR = "/pl/active/paolab/awaken_data/kp.turbine.z02.b0/"
        DATA_DIR = "/projects/ssc/ahenry/wind_forecasting/awaken_data/kp.turbine.z02.b0/"
        # PL_SAVE_PATH = "/scratch/alpine/aohe7145/awaken_data/kp.turbine.zo2.b0.raw.parquet"
        # PL_SAVE_PATH = "/projects/ssc/ahenry/wind_forecasting/awaken_data/loaded_data.parquet"
        PL_SAVE_PATH = os.path.join("/tmp/scratch", os.environ["SLURM_JOB_ID"], "loaded_data.parquet")
        # print(f"PL_SAVE_PATH = {PL_SAVE_PATH}")
        FILE_SIGNATURE = "kp.turbine.z02.b0.*.*.*.nc"
        MULTIPROCESSOR = "mpi"
        # TURBINE_INPUT_FILEPATH = "/projects/aohe7145/toolboxes/wind-forecasting/examples/inputs/ge_282_127.yaml"
        TURBINE_INPUT_FILEPATH = "/home/ahenry/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/ge_282_127.yaml"
        # FARM_INPUT_FILEPATH = "/projects/aohe7145/toolboxes/wind-forecasting/examples/inputs/gch_KP_v4.yaml"
        FARM_INPUT_FILEPATH = "/home/ahenry/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/gch_KP_v4.yaml"
        FEATURES = ["time", "turbine_id", "turbine_status", "wind_direction", "wind_speed", "power_output", "nacelle_direction"]
        WIDE_FORMAT = False # not originally in wide format
        COLUMN_MAPPING = {"date": "time",
                          "turbine_id": "turbine_id",
                          "WTUR.TurSt": "turbine_status",
                          "WMET.HorWdDir": "wind_direction",
                          "WMET.HorWdSpd": "wind_speed",
                          "WTUR.W": "power_output",
                          "WNAC.Dir": "nacelle_direction"
                          }
        
    elif platform == "linux" and DATA_FORMAT == "csv":
        # DATA_DIR = "/pl/active/paolab/awaken_data/kp.turbine.z02.b0/"
        # DATA_DIR = "examples/inputs/awaken_data"
        DATA_DIR = "examples/inputs/SMARTEOLE-WFC-open-dataset"
        # PL_SAVE_PATH = "/scratch/alpine/aohe7145/awaken_data/kp.turbine.zo2.b0.raw.parquet"
        PL_SAVE_PATH = "examples/inputs/SMARTEOLE-WFC-open-dataset/processed/SMARTEOLE_WakeSteering_SCADA_1minData.parquet"
        FILE_SIGNATURE = "SMARTEOLE_WakeSteering_SCADA_1minData.csv"
        MULTIPROCESSOR = "cf" # mpi for HPC or "cf" for local computing
        # TURBINE_INPUT_FILEPATH = "/projects/$USER/toolboxes/wind-forecasting/examples/inputs/ge_282_127.yaml"
        # FARM_INPUT_FILEPATH = "/projects/$USER/toolboxes/wind-forecasting/examples/inputs/gch_KP_v4.yaml"
        TURBINE_INPUT_FILEPATH = "examples/inputs/ge_282_127.yaml"
        FARM_INPUT_FILEPATH = "examples/inputs/gch_KP_v4.yaml"
        
        FEATURES = ["time", "active_power", "wind_speed", "nacelle_position", "wind_direction", "derate"]
        WIDE_FORMAT = True
        
        COLUMN_MAPPING = {
            "time": "time",
            **{f"active_power_{i}_avg": f"active_power_{i:03d}" for i in range(1, 8)},
            **{f"wind_speed_{i}_avg": f"wind_speed_{i:03d}" for i in range(1, 8)},
            **{f"nacelle_position_{i}_avg": f"nacelle_direction_{i:03d}" for i in range(1, 8)},
            **{f"wind_direction_{i}_avg": f"wind_direction_{i:03d}" for i in range(1, 8)},
            **{f"derate_{i}": f"derate_{i:03d}" for i in range(1, 8)}
        }
    
    if FILE_SIGNATURE.endswith(".nc"):
        DATA_FORMAT = "netcdf"
    elif FILE_SIGNATURE.endswith(".csv"):
        DATA_FORMAT = "csv"
    else:
        raise ValueError("Invalid file signature. Please specify either '*.nc' or '*.csv'.")
    
    RUN_ONCE = (MULTIPROCESSOR == "mpi" and (comm_rank := MPI.COMM_WORLD.Get_rank()) == 0) or (MULTIPROCESSOR != "mpi") or (MULTIPROCESSOR is None)
    data_loader = DataLoader(
                data_dir=DATA_DIR,
                file_signature=FILE_SIGNATURE,
                save_path=PL_SAVE_PATH,
                multiprocessor=MULTIPROCESSOR,
                dt=DT,
                desired_feature_types=FEATURES,
                data_format=DATA_FORMAT,
                column_mapping=COLUMN_MAPPING,
                wide_format=WIDE_FORMAT,
                ffill_limit=int(60 * 60 * 10 // DT))
    
    if RUN_ONCE:
        
        if not RELOAD_DATA and os.path.exists(data_loader.save_path):
            logging.info("🔄 Loading existing Parquet file")
            df_query = pl.scan_parquet(source=data_loader.save_path)
            logging.info("✅ Loaded existing Parquet file successfully")
        
        logging.info("🔄 Processing new data files")
       
        if MULTIPROCESSOR == "mpi":
            comm_size = MPI.COMM_WORLD.Get_size()
            logging.info(f"🚀 Using MPI executor with {comm_size} processes.")
        else:
            max_workers = multiprocessing.cpu_count()
            logging.info(f"🖥️  Using ProcessPoolExecutor with {max_workers} workers.")
    
    if RUN_ONCE:
        start_time = time.time()
        logging.info(f"✅ Starting read_multi_files with {len(data_loader.file_paths)} files")
    df_query = data_loader.read_multi_files()
    if RUN_ONCE:
        logging.info(f"✅ Finished reading individual files. Time elapsed: {time.time() - start_time:.2f} s")

    if RUN_ONCE:
    
        if df_query is not None:
            # Perform any additional operations on df_query if needed
            logging.info("✅ Data processing completed successfully")
        else:
            logging.warning("⚠️  No data was processed")
        
        logging.info("🎉 Script completed successfully")
