### This file contains class and method to: 
### - load the scada data, 
### - convert timestamps to datetime objects
### - convert circular measurements to sinusoidal measurements
### - normalize data

import glob
import os
import logging
import re
from shutil import rmtree
from psutil import virtual_memory

import time

import netCDF4 as nc
import polars as pl
import polars.selectors as cs
import numpy as np
from sklearn.preprocessing import MinMaxScaler

mpi_exists = False
try:
    from mpi4py import MPI
    from mpi4py.futures import MPICommExecutor
    mpi_exists = True
except:
    print("No MPI available on system.")

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
                #  desired_feature_types: list[str] = None,
                 dt: int | None = 5,
                 ffill_limit: int | None = None, 
                 data_format: str = "netcdf", 
                 feature_mapping: dict = None,
                 turbine_signature: str = None):
        
        self.data_dir = data_dir
        self.save_path = save_path
        self.file_signature = file_signature
        self.multiprocessor = multiprocessor
        self.dt = dt
        self.data_format = data_format.lower()
        self.feature_mapping = feature_mapping
        self.reverse_feature_mapping = dict((src, tgt) for tgt, src in self.feature_mapping.items())

        self.source_features = list(self.feature_mapping.values())
        self.target_features = list(self.feature_mapping.keys())
        
        assert [col in self.target_features for col in ["time", "wind_speed", "nacelle_direction", "power_output"]]
        assert "wind_direction" in self.target_features or ("nacelle_direction" in self.target_features and ("yaw_offset_cw" in self.target_features or "yaw_offset_ccw" in self.target_features)), "if wind_direction is not in the feature_mapping values, then yaw_offset_cw or yaw_offset_ccw must be to compute it from nacelle_direction"
        
        # self.desired_feature_types = desired_feature_types or ["time", "turbine_id", "turbine_status", "wind_direction", "wind_speed", "power_output", "nacelle_direction"]
        
        self.ffill_limit = ffill_limit

        self.turbine_signature = turbine_signature
        self.turbine_ids = set()
        # self.turbine_ids = sorted(list(set(k.split("_")[-1] for k in self.feature_mapping.keys() if re.search(r'\d', k)))) 

        self.target_feature_types = list(set((re.sub(self.turbine_signature, "", k) if re.search(self.turbine_signature, k) else k) for k in self.feature_mapping.keys()))
        
        # Get all the wts in the folder @Juan 10/16/24 used os.path.join for OS compatibility
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, file_signature), recursive=True))
    
    def read_multi_files(self) -> pl.LazyFrame | None:
        read_start = time.time()
        logging.info(f"✅ Started reading {len(self.file_paths)} files.")
        
        temp_save_dir = os.path.join(os.path.dirname(self.save_path), os.path.basename(self.save_path).replace(".parquet", "_temp"))
        if os.path.exists(temp_save_dir):
            rmtree(temp_save_dir)
            # raise Exception(f"Temporary saving directory {temp_save_dir} already exists! Please remove or rename it.")
        os.makedirs(temp_save_dir)
        
        if self.multiprocessor is not None:
            if self.multiprocessor == "mpi" and mpi_exists:
                executor = MPICommExecutor(MPI.COMM_WORLD, root=0)
            else:  # "cf" case
                executor = ProcessPoolExecutor()
            with executor as ex:
                if ex is not None:
                    if not self.file_paths:
                        raise FileExistsError(f"⚠️ File with signature {self.file_signature} in directory {self.data_dir} doesn't exist.")
                    
                    # futures = [ex.submit(self._read_single_file, f, file_path) for f, file_path in enumerate(self.file_paths)]
                    
                    df_query = []
                    file_paths = []
                    batch_idx = 0
                    batch_paths = []
                    for f, file_path in enumerate(self.file_paths):
                        used_ram = virtual_memory().percent 
                        if len(df_query) == 0 or used_ram < 80:
                            logging.info(f"Used RAM = {used_ram}%. Continue processing single files.")
                            res = ex.submit(self._read_single_file, f, file_path).result()
                            if res is not None: 
                                df_query.append(res)
                                file_paths.append(self.file_paths[f])
                        else:
                            # process what we have so far and dump processed lazy frames
                            logging.info(f"Used RAM = {used_ram}%. Pause to batch process.")
                            logging.info(f"🔗 Processing {len(df_query)} files read so farm.")
                            batch_idx += 1
                            batch_paths.append(self.process_batch_files(df_query, file_paths, batch_idx, temp_save_dir))
                            df_query = []
                            file_paths = []
                    
                    if len(df_query):
                        batch_idx += 1
                        batch_paths.append(self.process_batch_files(df_query, file_paths, batch_idx, temp_save_dir))
                    # df_query = [fut.result() for fut in futures]
                    # df_query = [(self.file_paths[d], df) for d, df in enumerate(df_query) if df is not None]
                    
        else:
            logging.info(f"🔧 Using single process executor.")
            if not self.file_paths:
                raise FileExistsError(f"⚠️ File with signature {self.file_signature} in directory {self.data_dir} doesn't exist.")
            # df_query = [self._read_single_file(f, file_path) for f, file_path in enumerate(self.file_paths)]
            # df_query = [(self.file_paths[d], df) for d, df in enumerate(df_query) if df is not None]

            df_query = []
            file_paths = []
            batch_idx = 0
            batch_paths = []
            for f, file_path in enumerate(self.file_paths):
                used_ram = virtual_memory().percent
                if  len(df_query) == 0 or used_ram < 80:
                    logging.info(f"Used RAM = {used_ram}%. Continue processing single files.")
                    res = self._read_single_file(f, file_path)
                    if res is not None: 
                        df_query.append(res)
                        file_paths.append(self.file_paths[f])
                else:
                    # process what we have so far and dump processed lazy frames
                    logging.info(f"Used RAM = {used_ram}%. Pause to batch process.")
                    logging.info(f"🔗 Processing {len(df_query)} files read so farm.")
                    batch_idx += 1
                    batch_paths.append(self.process_batch_files(df_query, file_paths, batch_idx, temp_save_dir))
                    df_query = []
                    file_paths = []
            
            if len(df_query):
                batch_idx += 1
                batch_paths.append(self.process_batch_files(df_query, file_paths, batch_idx, temp_save_dir))
        
        logging.info(f"🔗 Finished reading files. Time elapsed: {time.time() - read_start:.2f} s")
        
        df_query = [pl.scan_parquet(bp) for bp in batch_paths]
        if df_query:
            
            # concatenate intermediary dataframes
            df_query = pl.concat(df_query, how="diagonal").sort("time")
            
            if df_query.select(pl.col("time").diff().slice(1).n_unique()).collect().item() > 1:
                logging.info(f"Started final resampling.") 
                full_datetime_range = df_query.select(pl.datetime_range(
                    start=df_query.select("time").min().collect().item(),
                    end=df_query.select("time").max().collect().item(),
                    interval=f"{self.dt}s", time_unit=df_query.collect_schema()["time"].time_unit).alias("time"))
            
                df_query = full_datetime_range.join(df_query, on="time", how="left")
                logging.info(f"Finished final resampling.") 

            logging.info(f"Started final forward/backward fill.") 
            df_query = df_query.fill_null(strategy="forward").fill_null(strategy="backward").collect(streaming=True).lazy() # NOTE: @Aoife for KP data, need to fill forward null gaps, don't know about Juan's data
            logging.info(f"Finished final forward/backward fill.") 

            # Write to final parquet
            if not os.path.exists(os.path.dirname(self.save_path)):
                os.makedirs(os.path.dirname(self.save_path))
            df_query.sink_parquet(self.save_path, statistics=False)

            # turbine ids found in all files so far
            self.turbine_ids = self.get_turbine_ids(df_query, sort=True)
            logging.info("Final Parquet file saved into %s", self.save_path)
            rmtree(temp_save_dir)
            return df_query
            # return pl.scan_parquet(self.save_path)
        else:
            logging.error("No data successfully processed by read_multi_files.")
            return None
    
    def process_batch_files(self, df_queries, file_paths, i, temp_save_dir):
        # INFO: @Juan 11/13/24 Added check for data patterns in the names and also added a check for single files
        join_start = time.time()
        logging.info(f"✅ Started join of {len(file_paths)} files.")
        
        # Check if files have date patterns in their names
        date_pattern = r"\.(\d{8})\."
        has_date_pattern = any(re.search(date_pattern, fp) for fp in file_paths)
        
        if has_date_pattern:
            unique_file_timestamps = sorted(set(re.findall(date_pattern, fp)[0] for fp in file_paths 
                                                if re.search(date_pattern, fp)))
            df_queries = [self._join_dfs(ts, [df for filepath, df in df_queries if ts in filepath]) 
                        for ts in unique_file_timestamps]

            for ts, df in zip(unique_file_timestamps, df_queries):
                df.collect().write_parquet(self.save_path.replace(".parquet", f"_{ts}.parquet"), statistics=False)
                logging.info(f"Finished writing parquet {ts}")
            
            logging.info(f"🔗 Finished join. Time elapsed: {time.time() - join_start:.2f} s")

            concat_start = time.time()
            df_queries = [pl.scan_parquet(self.save_path.replace(".parquet", f"_{ts}.parquet")) 
                                for ts in unique_file_timestamps]
            df_queries = pl.concat(df_queries, how="diagonal").group_by("time").agg(cs.numeric().mean())
            logging.info(f"🔗 Finished concat. Time elapsed: {time.time() - concat_start:.2f} s")

            for ts in unique_file_timestamps:
                os.remove(self.save_path.replace(".parquet", f"_{ts}.parquet"))
        else:
            # For single file or files without timestamps, just get the dataframes
            if len(df_queries) == 1:
                df_queries = df_queries[0]  # If single file, no need to join
            else:
                df_queries = pl.concat(df_queries, how="diagonal").group_by("time").agg(cs.numeric().mean())

        logging.info(f"Started sorting.")
        df_queries = df_queries.sort("time")
        logging.info(f"Finished sorting.")

        logging.info(f"Started resampling.") 
        full_datetime_range = df_queries.select(pl.datetime_range(
            start=df_queries.select("time").min().collect().item(),
            end=df_queries.select("time").max().collect().item(),
            interval=f"{self.dt}s", time_unit=df_queries.collect_schema()["time"].time_unit).alias("time"))
            
        df_queries = full_datetime_range.join(df_queries, on="time", how="left").collect(streaming=True).lazy() # NOTE: @Aoife 10/18 make sure all time stamps are included, to interpolate continuously later
        logging.info(f"Finished resampling.") 

        logging.info(f"Started forward/backward fill.") 
        df_queries = df_queries.fill_null(strategy="forward").fill_null(strategy="backward").collect(streaming=True).lazy() # NOTE: @Aoife for KP data, need to fill forward null gaps, don't know about Juan's data
        logging.info(f"Finished forward/backward fill.")

        batch_path = os.path.join(temp_save_dir, f"df{i}")
        df_queries.sink_parquet(batch_path, statistics=False)
        return batch_path
    
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

    def get_turbine_ids(self, df_query, sort=False):
        turbine_ids = set()
        for col in df_query.collect_schema().names():
            match = re.search(self.turbine_signature, col)
            if match:
                turbine_ids.add(match.group())

        if sort:
            return sorted(turbine_ids)
        else:
            return turbine_ids

    # INFO: @Juan 10/14/24 Added method to read single file based on the file signature. 
    def _read_single_file(self, file_number: int, file_path: str) -> pl.LazyFrame:
        start_time = time.time()
        # logging.info(f"Starting to process {file_path}")
        try:
            if self.data_format == "netcdf":
                result = self._read_single_netcdf(file_path)
            elif self.data_format == "csv":
                result = self._read_single_csv(file_path)
            elif self.data_format == "parquet":
                result = self._read_single_parquet(file_path)
            else:
                raise ValueError(f"❌ Unsupported data format: {self.data_format}")
            
            logging.info(f"✅ Processed {file_number + 1}-th {file_path}. Time: {time.time() - start_time:.2f} s")
            return result
        except Exception as e:
            logging.error(f"❌ Error processing {file_path}: {e}")
            return None

    # INFO: @Juan 10/16/24 Added method to read single netcdf file. Use pl.Series to convert the time variable to a polars series. and combined time extraction operations into a single line to remove intermediate variables. Removed try/except block as it is done in the calling method (_read_single_file())
    def _read_single_netcdf(self, file_path: str) -> pl.LazyFrame:
        try:
            with nc.Dataset(file_path, 'r') as dataset:
                time_var = dataset.variables[self.feature_mapping["time"]]
                time = nc.num2date(times=time_var[:], 
                                units=time_var.units, 
                                calendar=time_var.calendar, 
                                only_use_cftime_datetimes=False, 
                                only_use_python_datetimes=True)
                
                data = {
                    'turbine_id': [os.path.basename(file_path).split('.')[-2]] * len(time),
                    'time': time.tolist(),  # Convert to Polars datetime
                    'turbine_status': dataset.variables[self.feature_mapping["turbine_status"]][:],
                    'wind_direction': dataset.variables[self.feature_mapping["wind_direction"]][:],
                    'wind_speed': dataset.variables[self.feature_mapping["wind_speed"]][:],
                    'power_output': dataset.variables[self.feature_mapping["power_output"]][:],
                    'nacelle_direction': dataset.variables[self.feature_mapping["nacelle_direction"]][:]
                }

                # self.turbine_ids = self.turbine_ids.union(set(data["turbine_id"]))

                # remove the rows with all nans (corresponding to rows where excluded columns would have had a value)
                # and bundle all values corresponding to identical time stamps together
                # forward fill missing values
                df_query = pl.LazyFrame(data).fill_nan(None)\
                                                .with_columns(pl.col("time").dt.round(f"{self.dt}s").alias("time"))\
                                                .select([cs.contains(feat) for feat in self.target_feature_types])\
                                                .filter(pl.any_horizontal(cs.numeric().is_not_null()))

                # Check if data is already in wide format
                is_already_wide = all(any(f"{feature}_{tid}" in col for col in df_query.collect_schema().names()) 
                    for feature in self.target_feature_types for tid in set(data["turbine_id"]) if feature != "time") 
                
                # pivot table to have columns for each turbine and measurement if not originally in wide format
                # if not self.wide_format:
                if not is_already_wide:
                    pivot_features = [col for col in df_query.collect_schema().names() if col not in ['time', 'turbine_id']]
                    df_query = df_query.collect(streaming=True).pivot(
                        index="time",
                        on="turbine_id",
                        values=pivot_features,
                        aggregate_function=pl.element().drop_nulls().first(),
                        sort_columns=True
                    ).sort("time").lazy()
                else:
                    df_query = df_query.collect(streaming=True).lazy()
                del data  # Free up memory

                return df_query
        except Exception:
            logging.error(f"❌ Error processing netcdf file {file_path}: {str(e)}")
            return None

    def _read_single_csv(self, file_path: str) -> pl.LazyFrame:
        try:
            df_query = pl.scan_csv(file_path, low_memory=False)
            
            available_columns = df_query.collect_schema().names()
            assert all(any(bool(re.search(feat, col)) for col in available_columns) for feat in self.source_features), "All values in feature_mapping must exist in data columns."

            # Select only relevant columns and handle missing values
            df_query = df_query.select(*[cs.matches(feat) for feat in self.source_features])
            source_features = df_query.collect_schema().names()
            
            # just the turbine ids found in this file
            turbine_ids = self.get_turbine_ids(df_query)

            # turbine ids found in all files so far 
            # self.turbine_ids = self.turbine_ids.union(turbine_ids)

            # Apply column mapping after selecting relevant columns
            # df_query = df_query.rename(dict((src, tgt) for tgt, src in zip(self.feature_mapping.keys(), source_features)))
            rename_dict = {}
            for src in source_features:
                feature_type = None 
                tid = None
                for src_signature in self.source_features:
                    if re.search(src_signature, src):
                        feature_type = self.reverse_feature_mapping[src_signature]
                        tid = re.search(self.turbine_signature, src)
                        if tid:
                            tid = tid.group()
                        break
                if feature_type and tid:
                    rename_dict[src] = f"{feature_type}_{tid}"
                elif feature_type:
                    rename_dict[src] = feature_type

            df_query = df_query.rename(rename_dict)
            df_query = df_query.with_columns([pl.col(col).cast(pl.Float64) for col in df_query.collect_schema().names() if any(feat_type in col and feat_type != "time" for feat_type in self.target_features)])
            df_query = df_query.with_columns(cs.numeric().fill_nan(None))
            
            if "wind_direction" not in self.target_features:
                if "nacelle_direction" in self.target_features:
                    if "yaw_offset_cw" in self.target_features:
                        delta = 1
                        direc = "cw" 
                    elif "yaw_offset_ccw" in self.target_features:
                        delta = -1
                        direc = "ccw"
                        
                    df_query = df_query\
                        .with_columns([
                            (pl.col(f"nacelle_direction_{tid}") + delta * pl.col(f"yaw_offset_{direc}_{tid}")).mod(360.0)\
                            .alias(f"wind_direction_{tid}") for tid in turbine_ids
                            ])
                    df_query = df_query.select(pl.exclude("yaw_offset_*"))

            if df_query.collect_schema()["time"] == pl.datatypes.String:
                df_query = df_query.with_columns([
                                        # Convert time column to datetime
                                        pl.col("time").str.to_datetime().alias("time")
                                    ])
            else:
                if df_query.collect_schema()["time"].time_zone is None:
                    df_query = df_query.with_columns(
                        time=pl.col("time").cast(pl.Datetime(time_unit=df_query.collect_schema()["time"].time_unit))
                    )
                else:
                    df_query = df_query.with_columns(
                        time=pl.col("time").dt.convert_time_zone("UTC").cast(pl.Datetime(time_unit=df_query.collect_schema()["time"].time_unit))
                    )
            
            # Check if data is already in wide format
            is_already_wide = all(any(f"{feature}_{tid}" in col for col in df_query.collect_schema().names()) 
                for feature in self.target_feature_types for tid in turbine_ids if feature != "time")

            # remove the rows with all nans (corresponding to rows where excluded columns would have had a value)
            # and bundle all values corresponding to identical time stamps together
            # forward fill missing values
            df_query = df_query.with_columns(pl.col("time").dt.round(f"{self.dt}s").alias("time"))\
                                .select([cs.contains(feat) for feat in self.target_feature_types])\
                                .filter(pl.any_horizontal(cs.numeric().is_not_null()))
            
            # pivot table to have columns for each turbine and measurement if not originally in wide format
            if not is_already_wide:
                pivot_features = [col for col in df_query.collect_schema().names() if col not in ['time', 'turbine_id']]
                df_query = df_query.collect(streaming=True).pivot(
                    index="time",
                    on="turbine_id",
                    values=pivot_features,
                    aggregate_function=pl.element().drop_nulls().first(),
                    sort_columns=True
                ).sort("time").lazy()
            else:
                df_query = df_query.collect(streaming=True).lazy()
            
            return df_query
        
        except Exception as e:
            logging.error(f"❌ Error processing CSV file {file_path}: {str(e)}")
            return None

    def _read_single_parquet(self, file_path: str) -> pl.LazyFrame:
        try:
            df_query = pl.scan_parquet(file_path)
            available_columns = df_query.collect_schema().names()
            assert all(any(bool(re.search(feat, col)) for col in available_columns) for feat in self.source_features), "All values in feature_mapping must exist in data columns."

            # Select only relevant columns and handle missing values
            # df_query = df_query.select(self.source_features)\
            df_query = df_query.select(*[cs.matches(feat) for feat in self.source_features])
            source_features = df_query.collect_schema().names()
            
            # just the turbine ids found in this file
            turbine_ids = self.get_turbine_ids(df_query)
            
            # Apply column mapping after selecting relevant columns
            rename_dict = {}
            for src in source_features:
                feature_type = None
                tid = None
                for src_signature in self.source_features:
                    if re.search(src_signature, src):
                        feature_type = self.reverse_feature_mapping[src_signature]
                        tid = re.search(self.turbine_signature, src)
                        if tid:
                            tid = tid.group()
                if feature_type and tid:
                    rename_dict[src] = f"{feature_type}_{tid}"
                elif feature_type:
                    rename_dict[src] = feature_type

            df_query = df_query.rename(rename_dict)
            
            df_query = df_query.with_columns([pl.col(col).cast(pl.Float64) for col in df_query.collect_schema().names() if any(feat_type in col and feat_type != "time" for feat_type in self.target_features)])
            df_query = df_query.with_columns(cs.numeric().fill_nan(None))
            
            target_features = list(self.target_features)
            if "wind_direction" not in target_features:
                if "nacelle_direction" in target_features:
                    if "yaw_offset_cw" in target_features:
                        delta = 1
                        direc = "cw" 
                    elif "yaw_offset_ccw" in target_features:
                        delta = -1
                        direc = "ccw"
                    
                    df_query = df_query\
                        .with_columns([
                            (pl.col(f"nacelle_direction_{tid}") + delta * pl.col(f"yaw_offset_{direc}_{tid}")).mod(360.0)\
                            .alias(f"wind_direction_{tid}") for tid in turbine_ids
                            ])
                    df_query = df_query.select(pl.exclude("^yaw_offset_.*$"))
                    del target_features[target_features.index(f"yaw_offset_{direc}")]
                    target_features.append("wind_direction")
                    
            if df_query.collect_schema()["time"] == pl.datatypes.String:
                df_query = df_query.with_columns([
                                        # Convert time column to datetime
                                        pl.col("time").str.to_datetime().alias("time")
                                    ])
            else:
                if df_query.collect_schema()["time"].time_zone is None:
                    df_query = df_query.with_columns(
                        time=pl.col("time").cast(pl.Datetime(time_unit=df_query.collect_schema()["time"].time_unit))
                    )
                else:
                    df_query = df_query.with_columns(
                        time=pl.col("time").dt.convert_time_zone("UTC").cast(pl.Datetime(time_unit=df_query.collect_schema()["time"].time_unit))
                    )
            
            # Check if data is already in wide format
            is_already_wide = all(any(f"{feature}_{tid}" in col for col in df_query.collect_schema().names()) 
                for feature in target_features for tid in turbine_ids if feature != "time")

            # remove the rows with all nans (corresponding to rows where excluded columns would have had a value)
            # and bundle all values corresponding to identical time stamps together
            # forward fill missing values
            df_query = df_query.with_columns(pl.col("time").dt.round(f"{self.dt}s").alias("time"))\
                                .select([cs.contains(feat) for feat in target_features])\
                                .filter(pl.any_horizontal(cs.numeric().is_not_null()))
            
            # pivot table to have columns for each turbine and measurement if not originally in wide format
            if not is_already_wide:
                pivot_features = [col for col in df_query.collect_schema().names() if col not in ['time', 'turbine_id']]
                df_query = df_query.collect(streaming=True).pivot(
                    index="time",
                    on="turbine_id",
                    values=pivot_features,
                    aggregate_function=pl.element().drop_nulls().first(),
                    sort_columns=True
                ).sort("time").lazy()
            else:
                df_query = df_query.collect(streaming=True).lazy()
            
            # TODO sink this into temporary storage OR when approaching end of RAM, do final preprocessing step and sink into storage
            return df_query
        
        except Exception as e:
            logging.error(f"❌ Error processing parquet file {file_path}: {str(e)}")
            return None

    # INFO: @Juan 10/16/24 Added method to convert to long format. May need refining!!! #UNTESTED
    def _convert_to_long_format(self, df: pl.LazyFrame) -> pl.LazyFrame:
        # It will only trigger when wide_format is False.
        # Identify the columns that contain turbine-specific data
        logging.info("🔄 Converting data to long format")
        turbine_columns = [col for col in df.columns if col != "time"]
        
        # Melt the DataFrame to convert it to long format
        df_long = df.melt(
            id_vars=["time"], 
            value_vars=turbine_columns,
            variable_name="feature",
            value_name="value"
        )
        
        # Extract turbine_id and feature_name from the 'feature' column
        df_long = df_long.with_columns([
            pl.col("feature").str.extract(r"_(\d+)(?:_avg|$)").alias("turbine_id"),
            pl.col("feature").str.replace(r"_\d+(?:_avg|$)", "").alias("feature_name")
        ])
        
        # Pivot the data to have features as columns
        df_final = df_long.pivot(
            index=["time", "turbine_id"],
            columns="feature_name",
            values="value"
        )
        
        # Ensure turbine_id is a string with leading zeros
        df_final = df_final.with_columns(
            pl.col("turbine_id").cast(pl.Int32).cast(pl.Utf8).str.zfill(3)
        )
        
        logging.info("✅ Data pivoted to long format successfully")
        return df_final
    
    # INFO: @Juan 10/16/24 Added method to convert to wide format.
    def convert_to_wide_format(self, df: pl.LazyFrame) -> pl.LazyFrame:
        logging.info("🔄 Converting data to wide format")
        
        # List of features to pivot (excluding 'time' and 'turbine_id')
        pivot_features = [col for col in df.columns if col not in ['time', 'turbine_id']]
        
        # Pivot the data
        df_wide = df.pivot(
            index="time",
            columns="turbine_id",
            values=pivot_features,
            # aggregate_function="first",
            sort_columns=True
        )
        
        logging.info("✅ Data pivoted to wide format successfully")
        return df_wide

    def print_netcdf_structure(self, file_path) -> None: #INFO: @Juan 10/02/24 Changed print to logging
        try:
            with nc.Dataset(file_path, 'r') as dataset:
                logging.info(f"📊 NetCDF File: {os.path.basename(file_path)}")
                logging.info("\n🌐 Global Attributes:")
                for attr in dataset.ncattrs():
                    logging.info(f"  {attr}: {getattr(dataset, attr)}")

                logging.info("\n📏 Dimensions:")
                for dim_name, dim in dataset.dimensions.items():
                    logging.info(f"  {dim_name}: {len(dim)}")

                logging.info("\n🔢 Variables:")
                for var_name, var in dataset.variables.items():
                    logging.info(f"  {var_name}:")
                    logging.info(f"    Dimensions: {var.dimensions}")
                    logging.info(f"    Shape: {var.shape}")
                    logging.info(f"    Data type: {var.dtype}")
                    logging.info("    Attributes:")
                    for attr in var.ncattrs():
                        logging.info(f"      {attr}: {getattr(var, attr)}")

        except Exception as e:
            logging.error(f"❌ Error reading NetCDF file: {e}")

    # INFO: @Juan 10/02/24 Revamped this method to use Polars functions consistently, vectorized where possible, and using type casting for consistency and performance enhancements.

    def convert_time_to_sin(self, df) -> pl.LazyFrame:
        """_summary_
            convert timestamp to cosine and sinusoidal components
        Returns:
            pl.LazyFrame: _description_
        """
        if df is None:
            raise ValueError("⚠️ Data not loaded > call read_multi_netcdf() first.")
        
        df = self.df.with_columns([
            pl.col('time').dt.hour().alias('hour'),
            pl.col('time').dt.ordinal_day().alias('day'),
            pl.col('time').dt.year().alias('year'),
        ])

        # Normalize time features using sin/cos for capturing cyclic patterns using Polars vectorized operations
        df = df.with_columns([
            (2 * np.pi * pl.col('hour') / 24).sin().alias('hour_sin'),
            (2 * np.pi * pl.col('hour') / 24).cos().alias('hour_cos'),
            (2 * np.pi * pl.col('day') / 365).sin().alias('day_sin'),
            (2 * np.pi * pl.col('day') / 365).cos().alias('day_cos'),
            (2 * np.pi * pl.col('year') / 365).sin().alias('year_sin'),
            (2 * np.pi * pl.col('year') / 365).cos().alias('year_cos'),
        ])

        return df

    # DEBUG: @Juan 10/16/24 Check that this is reducing the features correctly.
    def reduce_features(self, df) -> pl.LazyFrame:
        """
        Reduce the DataFrame to include only the specified features that exist in the DataFrame.
        """
        existing_features = [f for f in self.desired_feature_types if any(f in col for col in df.columns)]
        df = df.select([pl.col(col) for col in df.columns if any(feature in col for feature in existing_features)])
        
        # Only filter rows if there are numeric columns
        numeric_cols = df.select(cs.numeric()).columns
        if numeric_cols:
            df = df.filter(pl.any_horizontal(pl.col(numeric_cols).is_not_null()))
        
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
# if __name__ == "__main__":
#     from sys import platform
#     RELOAD_DATA = True
#     PLOT = False

#     DT = 5
#     DATA_FORMAT = "csv"

#     if platform == "darwin" and DATA_FORMAT == "netcdf":
#         DATA_DIR = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data"
#         # PL_SAVE_PATH = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data/kp.turbine.zo2.b0.raw.parquet"
#         # FILE_SIGNATURE = "kp.turbine.z02.b0.*.*.*.nc"
#         PL_SAVE_PATH = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data/short_loaded_data.parquet"
#         FILE_SIGNATURE = "kp.turbine.z02.b0.202203*1.*.*.nc"
#         MULTIPROCESSOR = "cf"
#         TURBINE_INPUT_FILEPATH = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/inputs/ge_282_127.yaml"
#         FARM_INPUT_FILEPATH = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/inputs/gch_KP_v4.yaml"
#         FEATURES = ["time", "turbine_id", "turbine_status", "wind_direction", "wind_speed", "power_output", "nacelle_direction"]
#         WIDE_FORMAT = False
#         feature_mapping = {"date": "time",
#                           "turbine_id": "turbine_id",
#                           "WTUR.TurSt": "turbine_status",
#                           "WMET.HorWdDir": "wind_direction",
#                           "WMET.HorWdSpd": "wind_speed",
#                           "WTUR.W": "power_output",
#                           "WNAC.Dir": "nacelle_direction"
#                           }
#     elif platform == "linux" and DATA_FORMAT == "netcdf":
#         # DATA_DIR = "/pl/active/paolab/awaken_data/kp.turbine.z02.b0/"
#         DATA_DIR = "/projects/ssc/ahenry/wind_forecasting/awaken_data/kp.turbine.z02.b0/"
#         # PL_SAVE_PATH = "/scratch/alpine/aohe7145/awaken_data/kp.turbine.zo2.b0.raw.parquet"
#         # PL_SAVE_PATH = "/projects/ssc/ahenry/wind_forecasting/awaken_data/loaded_data.parquet"
#         PL_SAVE_PATH = os.path.join("/tmp/scratch", os.environ["SLURM_JOB_ID"], "loaded_data.parquet")
#         # print(f"PL_SAVE_PATH = {PL_SAVE_PATH}")
#         FILE_SIGNATURE = "kp.turbine.z02.b0.*.*.*.nc"
#         MULTIPROCESSOR = "mpi"
#         # TURBINE_INPUT_FILEPATH = "/projects/aohe7145/toolboxes/wind-forecasting/examples/inputs/ge_282_127.yaml"
#         TURBINE_INPUT_FILEPATH = "/home/ahenry/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/ge_282_127.yaml"
#         # FARM_INPUT_FILEPATH = "/projects/aohe7145/toolboxes/wind-forecasting/examples/inputs/gch_KP_v4.yaml"
#         FARM_INPUT_FILEPATH = "/home/ahenry/toolboxes/wind_forecasting_env/wind-forecasting/examples/inputs/gch_KP_v4.yaml"
#         FEATURES = ["time", "turbine_id", "turbine_status", "wind_direction", "wind_speed", "power_output", "nacelle_direction"]
#         WIDE_FORMAT = False # not originally in wide format
#         feature_mapping = {"date": "time",
#                           "turbine_id": "turbine_id",
#                           "WTUR.TurSt": "turbine_status",
#                           "WMET.HorWdDir": "wind_direction",
#                           "WMET.HorWdSpd": "wind_speed",
#                           "WTUR.W": "power_output",
#                           "WNAC.Dir": "nacelle_direction"
#                           }
        
#     elif platform == "linux" and DATA_FORMAT == "csv":
#         # DATA_DIR = "/pl/active/paolab/awaken_data/kp.turbine.z02.b0/"
#         # DATA_DIR = "examples/inputs/awaken_data"
#         DATA_DIR = "examples/inputs/SMARTEOLE-WFC-open-dataset"
#         # PL_SAVE_PATH = "/scratch/alpine/aohe7145/awaken_data/kp.turbine.zo2.b0.raw.parquet"
#         PL_SAVE_PATH = "examples/inputs/SMARTEOLE-WFC-open-dataset/processed/SMARTEOLE_WakeSteering_SCADA_1minData.parquet"
#         FILE_SIGNATURE = "SMARTEOLE_WakeSteering_SCADA_1minData.csv"
#         MULTIPROCESSOR = "cf" # mpi for HPC or "cf" for local computing
#         # TURBINE_INPUT_FILEPATH = "/projects/$USER/toolboxes/wind-forecasting/examples/inputs/ge_282_127.yaml"
#         # FARM_INPUT_FILEPATH = "/projects/$USER/toolboxes/wind-forecasting/examples/inputs/gch_KP_v4.yaml"
#         TURBINE_INPUT_FILEPATH = "examples/inputs/ge_282_127.yaml"
#         FARM_INPUT_FILEPATH = "examples/inputs/gch_KP_v4.yaml"
        
#         FEATURES = ["time", "active_power", "wind_speed", "nacelle_position", "wind_direction", "derate"]
#         WIDE_FORMAT = True
        
#         feature_mapping = {
#             "time": "time",
#             **{f"active_power_{i}_avg": f"active_power_{i:03d}" for i in range(1, 8)},
#             **{f"wind_speed_{i}_avg": f"wind_speed_{i:03d}" for i in range(1, 8)},
#             **{f"nacelle_position_{i}_avg": f"nacelle_position_{i:03d}" for i in range(1, 8)},
#             **{f"wind_direction_{i}_avg": f"wind_direction_{i:03d}" for i in range(1, 8)},
#             **{f"derate_{i}": f"derate_{i:03d}" for i in range(1, 8)}
#         }
    
#     if FILE_SIGNATURE.endswith(".nc"):
#         DATA_FORMAT = "netcdf"
#     elif FILE_SIGNATURE.endswith(".csv"):
#         DATA_FORMAT = "csv"
#     else:
#         raise ValueError("Invalid file signature. Please specify either '*.nc' or '*.csv'.")
    
#     RUN_ONCE = (MULTIPROCESSOR == "mpi" and (comm_rank := MPI.COMM_WORLD.Get_rank()) == 0) or (MULTIPROCESSOR != "mpi") or (MULTIPROCESSOR is None)
#     data_loader = DataLoader(
#                 data_dir=DATA_DIR,
#                 file_signature=FILE_SIGNATURE,
#                 save_path=PL_SAVE_PATH,
#                 multiprocessor=MULTIPROCESSOR,
#                 dt=DT,
#                 desired_feature_types=FEATURES,
#                 data_format=DATA_FORMAT,
#                 feature_mapping=feature_mapping,
#                 wide_format=WIDE_FORMAT,
#                 ffill_limit=int(60 * 60 * 10 // DT))
    
#     if RUN_ONCE:
        
#         if not RELOAD_DATA and os.path.exists(data_loader.save_path):
#             logging.info("🔄 Loading existing Parquet file")
#             df_query = pl.scan_parquet(source=data_loader.save_path)
#             logging.info("✅ Loaded existing Parquet file successfully")
        
#         logging.info("🔄 Processing new data files")
       
#         if MULTIPROCESSOR == "mpi":
#             comm_size = MPI.COMM_WORLD.Get_size()
#             logging.info(f"🚀 Using MPI executor with {comm_size} processes.")
#         else:
#             max_workers = multiprocessing.cpu_count()
#             logging.info(f"🖥️  Using ProcessPoolExecutor with {max_workers} workers.")
    
#     if RUN_ONCE:
#         start_time = time.time()
#         logging.info(f"✅ Starting read_multi_files with {len(data_loader.file_paths)} files")
#     df_query = data_loader.read_multi_files()
#     if RUN_ONCE:
#         logging.info(f"✅ Finished reading individual files. Time elapsed: {time.time() - start_time:.2f} s")

#     if RUN_ONCE:
    
#         if df_query is not None:
#             # Perform any additional operations on df_query if needed
#             logging.info("✅ Data processing completed successfully")
#         else:
#             logging.warning("⚠️  No data was processed")
        
#         logging.info("🎉 Script completed successfully")
