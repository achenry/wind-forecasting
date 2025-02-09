### This file contains class and method to: 
### - load the scada data, 
### - convert timestamps to datetime objects
### - convert circular measurements to sinusoidal measurements
### - normalize data

import glob
import os
from typing import List, Optional
import logging
import re
from shutil import rmtree, move
from psutil import virtual_memory
# from datetime.datetime import strptime
from memory_profiler import profile

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
                 data_dir: List[str],
                 file_signature: List[str],
                 save_path: str,
                 multiprocessor: str | None,
                 dt: int,
                 feature_mapping: List[dict],
                 turbine_signature: List[str],
                 turbine_mapping: Optional[List[dict]],
                 datetime_signature: List[dict],
                 data_format: List[str],
                 ffill_limit: int | None = None, 
                 merge_chunk: int = 100,
                 ram_limit: int = 50):
        
        self.data_dir = data_dir
        self.save_path = save_path
        self.file_signature = file_signature
        self.multiprocessor = multiprocessor
        self.dt = dt
        self.data_format = [df.lower() for df in data_format]
        assert all(df in ["netcdf", "csv", "parquet"] for df in self.data_format)
        self.feature_mapping = feature_mapping
        self.reverse_feature_mapping = [dict((src, tgt) for tgt, src in fm.items()) for fm in self.feature_mapping]
        self.merge_chunk = merge_chunk # number of files above which processed files should be merged/sorted/resampled/filled
        self.ram_limit = ram_limit # percentage of used RAM above which processed files should be merged/sorted/resampled/filled

        self.source_features = [list(fm.values()) for fm in self.feature_mapping]
        self.target_features = set.union(*[set(fm.keys()) for fm in self.feature_mapping])
        
        assert [col in self.target_features for col in ["time", "wind_speed", "nacelle_direction", "power_output"]]
        assert "wind_direction" in self.target_features or ("nacelle_direction" in self.target_features and ("yaw_offset_cw" in self.target_features or "yaw_offset_ccw" in self.target_features)), "if wind_direction is not in the feature_mapping values, then yaw_offset_cw or yaw_offset_ccw must be to compute it from nacelle_direction"
        
        # self.desired_feature_types = desired_feature_types or ["time", "turbine_id", "turbine_status", "wind_direction", "wind_speed", "power_output", "nacelle_direction"]
        
        self.ffill_limit = ffill_limit

        self.turbine_signature = turbine_signature
        self.datetime_signature = [list(ds.items())[0] if ds else None for ds in datetime_signature] # mapping from a regex expression to a datetime format to capture datetime from filepaths
        self.turbine_ids = set()
        self.turbine_mapping = turbine_mapping
        
        if self.turbine_mapping is not None:
            # check for no duplicates in target turbine ids, also check that values are the same for every element in mapping and only check if not None
            assert all(len(set(self.turbine_mapping[0].values()).difference(set(tm.values()))) == 0 for tm in self.turbine_mapping[1:]), "target integer turbine ids must match for each file type in turbine mapping"
            assert all(isinstance(v, int) for tm in self.turbine_mapping for v in tm.values()), "if using turbine_mapping, must map each turbine id to unique integer"
            
        # Get all the wts in the folder @Juan 10/16/24 used os.path.join for OS compatibility
        self.file_paths = [sorted(glob.glob(os.path.join(dd, fs), recursive=True)) for dd, fs in zip(data_dir, file_signature)]
    
    # @profile 
    def read_multi_files(self) -> pl.LazyFrame | None:
        read_start = time.time()
        
        if self.multiprocessor is not None:
            if self.multiprocessor == "mpi" and mpi_exists:
                executor = MPICommExecutor(MPI.COMM_WORLD, root=0)
            else:  # "cf" case
                executor = ProcessPoolExecutor()
            with executor as ex:
   
                if ex is not None:
                    RUN_ONCE = MPI.COMM_WORLD.Get_rank() == 0
    
                    temp_save_dir = os.path.join(os.path.dirname(self.save_path), os.path.basename(self.save_path).replace(".parquet", "_temp"))
                    if RUN_ONCE:
                        logging.info(f"Making temporary directory {temp_save_dir}")
                        if os.path.exists(temp_save_dir):
                            rmtree(temp_save_dir)
                            # raise Exception(f"Temporary saving directory {temp_save_dir} already exists! Please remove or rename it.")
                        os.makedirs(temp_save_dir)
                    
                        if not os.path.exists(os.path.dirname(self.save_path)):
                            logging.info(f"Making directory to save_path {os.path.dirname(self.save_path)}")
                            os.makedirs(os.path.dirname(self.save_path))
                        
                    logging.info(f"✅ Started reading {sum(len(fp) for fp in self.file_paths)} files.")
                    
                    for file_set_idx, fp in enumerate(self.file_paths):
                        if not fp:
                            raise FileExistsError(f"⚠️ File with signature {self.file_signature[file_set_idx]} in directory {self.data_dir[file_set_idx]} doesn't exist.")
                    
                    # futures = [ex.submit(self._read_single_file, f, file_path) for f, file_path in enumerate(self.file_paths)]
                    
                    processed_file_paths = []
                    merge_idx = 0
                    merged_paths = []
                    # init_used_ram = virtual_memory().percent 
                     
                    file_futures = [ex.submit(self._read_single_file, file_set_idx, f, file_path, os.path.join(temp_save_dir, os.path.basename(file_path))) 
                                    for file_set_idx in range(len(self.file_paths)) for f, file_path in enumerate(self.file_paths[file_set_idx])] #4% increase in mem
                    
                    for file_set_idx in range(len(self.file_paths)):
                        for f, file_path in enumerate(self.file_paths[file_set_idx]):
                            used_ram = virtual_memory().percent 
                            
                            # if we have enough ram to continue to process files AND we still have files to process
                            if (len(processed_file_paths) < self.merge_chunk and used_ram < self.ram_limit):
                                logging.info(f"Used RAM = {used_ram}%. Continue adding to buffer of {len(processed_file_paths)} processed single files.")
                                # res = ex.submit(self._read_single_file, f, file_path).result()
                                res = file_futures[f].result() #.5% increase in mem
                                if res is not None: 
                                    processed_file_paths.append(os.path.join(temp_save_dir, os.path.basename(file_path)))
                            
                            if not (len(processed_file_paths) < self.merge_chunk and used_ram < self.ram_limit) \
                                or (f == len(self.file_paths[file_set_idx]) - 1):
                                # process what we have so far and dump processed lazy frames
                                if f == (len(self.file_paths[file_set_idx]) - 1):
                                    logging.info(f"Used RAM = {used_ram}%. Pause for FINAL merge/sort/resample/fill of {len(processed_file_paths)} files read so far.")
                                else:
                                    logging.info(f"Used RAM = {used_ram}%. Pause to merge/sort/resample/fill {len(processed_file_paths)} files read so far.")
                                
                                merged_paths.append(ex.submit(self.merge_multiple_files, file_set_idx, processed_file_paths, merge_idx, temp_save_dir).result())
                                # merged_paths.append(self.merge_multiple_files(file_set_idx, processed_file_paths, merge_idx, temp_save_dir))
                                merge_idx += 1
                                processed_file_paths = []
                    
                    # merged_paths = [fut.result() for fut in merged_paths]
                    
        else:
            temp_save_dir = os.path.join(os.path.dirname(self.save_path), os.path.basename(self.save_path).replace(".parquet", "_temp"))
            if os.path.exists(temp_save_dir):
                rmtree(temp_save_dir)
            if not os.path.exists(os.path.dirname(self.save_path)):
                os.makedirs(os.path.dirname(self.save_path))
                # raise Exception(f"Temporary saving directory {temp_save_dir} already exists! Please remove or rename it.")
            os.makedirs(temp_save_dir)
            logging.info(f"✅ Started reading {sum(len(fp) for fp in self.file_paths)} files.")
            logging.info(f"🔧 Using single process executor.")
            if not self.file_paths:
                raise FileExistsError(f"⚠️ File with signature {self.file_signature} in directory {self.data_dir} doesn't exist.")
            # df_query = [self._read_single_file(f, file_path) for f, file_path in enumerate(self.file_paths)]
            # df_query = [(self.file_paths[d], df) for d, df in enumerate(df_query) if df is not None]

            df_query = []
            processed_file_paths = []
            merge_idx = 0
            merged_paths = []
            for file_set_idx in range(len(self.file_paths)):
                for f, file_path in enumerate(self.file_paths[file_set_idx]):
                    used_ram = virtual_memory().percent
                    if (len(processed_file_paths) < self.merge_chunk and used_ram < self.ram_limit):
                        logging.info(f"Used RAM = {used_ram}%. Continue adding to buffer of {len(processed_file_paths)} processed single files.")
                        res = self._read_single_file(file_set_idx, f, file_path, 
                                                    os.path.join(temp_save_dir, os.path.basename(file_path)))
                        if res is not None:
                            processed_file_paths.append(os.path.join(temp_save_dir, os.path.basename(file_path)))
                    if not (len(processed_file_paths) < self.merge_chunk and used_ram < self.ram_limit) \
                        or (f == len(self.file_paths[file_set_idx]) - 1):
                        # process what we have so far and dump processed lazy frames
                        if f == (len(self.file_paths[file_set_idx]) - 1):
                            logging.info(f"Used RAM = {used_ram}%. Pause for FINAL merge/sort/resample/fill of {len(processed_file_paths)} files read so far.")
                        else:
                            logging.info(f"Used RAM = {used_ram}%. Pause to merge/sort/resample/fill {len(processed_file_paths)} files read so far.")
                        
                        merged_paths.append(self.merge_multiple_files( file_set_idx, processed_file_paths, merge_idx, temp_save_dir))
                        merge_idx += 1
                        processed_file_paths = []
            
        if self.turbine_mapping: # if not none, ie there are multiple filetypes being processed
            self.turbine_signature = "\\d+$"
                
        if len(merged_paths):    
            logging.info(f"🔗 Finished reading files. Time elapsed: {time.time() - read_start:.2f} s")
            if len(merged_paths) > 1: 
                logging.info(f"Concatenating and running final sort/resample/fill.")
                # concatenate intermediary dataframes
                df_query = pl.concat([pl.scan_parquet(bp) for bp in merged_paths], how="diagonal")
                df_query = self.sort_resample_refill(df_query)
                # Write to final parquet
                logging.info(f"Saving final Parquet file into {self.save_path}")
                df_query.sink_parquet(self.save_path, statistics=False)
                
            else:
                logging.info(f"Moving only batch to {self.save_path}.")
                move(merged_paths[0], self.save_path)
                df_query = pl.scan_parquet(self.save_path)
            
            # turbine ids found in all files so far
            self.turbine_ids = self.get_turbine_ids(self.turbine_signature, df_query, sort=True)
                
            logging.info(f"Final Parquet file saved into {self.save_path}")
            
            logging.info(f"Removing temporary storage directory {temp_save_dir}")
            rmtree(temp_save_dir)
            logging.info(f"Removed temporary storage directory {temp_save_dir}")
            
            return df_query
        else:
            logging.error("No data successfully processed by read_multi_files.")
            logging.info(f"Removing temporary storage directory {temp_save_dir}")
            rmtree(temp_save_dir)
            logging.info(f"Removed temporary storage directory {temp_save_dir}")
            return None
   
    # @profile
    def sort_resample_refill(self, df_query):
        
        logging.info(f"Started sorting.")
        df_query = df_query.sort("time")
        logging.info(f"Finished sorting.")
        
        # if df_query.select(pl.col("time").diff().slice(1).n_unique()).collect().item() > 1:
        if df_query.select((pl.col("time").diff().slice(1) == pl.col("time").diff().last()).all()).collect().item():
            logging.info(f"Started resampling.") 
            bounds = df_query.select(pl.col("time").first().alias("first"),
                                     pl.col("time").last().alias("last")).collect()
            df_query = df_query.select(pl.datetime_range(
                                        start=bounds.select("first").item(),
                                        end=bounds.select("last").item(),
                                        interval=f"{self.dt}s", time_unit=df_query.collect_schema()["time"].time_unit).alias("time"))\
                                .join(df_query, on="time", how="left")
            
            # df_query = full_datetime_range.join(df_query, on="time", how="left")
            logging.info(f"Finished resampling.") 

        logging.info(f"Started forward/backward fill.") 
        df_query = df_query.fill_null(strategy="forward").fill_null(strategy="backward").collect().lazy() # NOTE: @Aoife for KP data, need to fill forward null gaps, don't know about Juan's data
        logging.info(f"Finished forward/backward fill.") 
        
        return df_query
    
    # @profile 
    def merge_multiple_files(self, file_set_idx, processed_file_paths, i, temp_save_dir):
        # INFO: @Juan 11/13/24 Added check for data patterns in the names and also added a check for single files
        # if len(processed_file_paths) == 1:
        #     df_queries = [pl.scan_parquet(fp) for fp in processed_file_paths]
        #     df_queries = self.sort_resample_refill(df_queries)
        #     return processed_file_paths[0]
        
        # join_start = time.time()
        logging.info(f"✅ Started join of {len(processed_file_paths)} files.")
        df_queries = [pl.scan_parquet(fp) for fp in processed_file_paths]
        
        # Check if files have date patterns in their names
        # has_date_pattern = self.datetime_signature[file_set_idx] is not None and all(re.search(self.datetime_signature[file_set_idx][0], os.path.basename(fp)) for fp in processed_file_paths)
        # unique_file_timestamps = sorted(set(re.findall(self.datetime_signature[file_set_idx][0], fp)[0] for fp in processed_file_paths 
        #                                         if re.search(self.datetime_signature[file_set_idx][0], fp))) if has_date_pattern else None
        
        # if has_date_pattern and (len(processed_file_paths) > len(unique_file_timestamps)):
        #     # selectively join dataframes for same timestamps but different turbines, then concatenate different time stamps (more efficient less joins)
            
        #     df_queries = [self._join_dfs(ts, [df for filepath, df in zip(processed_file_paths, df_queries) if ts in filepath]) 
        #                 for ts in unique_file_timestamps]

        #     pl.concat(df_queries, how="diagonal").group_by("time").agg(cs.numeric().mean())
            
        #     for ts in unique_file_timestamps:
        #         df_queries.pop(0).collect().write_parquet(os.path.join(temp_save_dir, os.path.basename(self.save_path).replace(".parquet", f"_{ts}.parquet")), statistics=False)
        #         logging.info(f"Finished writing parquet {ts}")
        #     # del df_queries
            
        #     logging.info(f"🔗 Finished join. Time elapsed: {time.time() - join_start:.2f} s")

        #     concat_start = time.time()
        #     df_queries = [pl.scan_parquet(os.path.join(temp_save_dir, os.path.basename(self.save_path).replace(".parquet", f"_{ts}.parquet"))) 
        #                         for ts in unique_file_timestamps]
        #     df_queries = pl.concat(df_queries, how="diagonal").group_by("time").agg(cs.numeric().mean())
        #     logging.info(f"🔗 Finished concat. Time elapsed: {time.time() - concat_start:.2f} s")

        # else:
        
        # For single file or files without timestamps, just get the dataframes
        if len(df_queries) == 1:
            df_queries = df_queries[0]  # If single file, no need to join
        else:
            df_queries = pl.concat(df_queries, how="diagonal").group_by("time").agg(cs.numeric().mean())
        
        logging.info(f"Finished join of {len(processed_file_paths)} files.")
        
        # convert to common turbine_id over multiple filetypes
        if self.turbine_mapping is not None:
            turbine_ids = self.get_turbine_ids(self.turbine_signature[file_set_idx], df_queries, sort=True) # turbine ids available in this collection of file paths (may not represent all)
            assert all(tid in self.turbine_mapping[file_set_idx] for tid in turbine_ids), "check turbine_mapping in parameter, should have n_turbines length of distinct turbine ids, all of which are found in the data" # make sure that turbine mapping accounts for all turbine ids found in files
            df_queries = df_queries.rename({
                col: 
                re.sub(pattern=self.turbine_signature[file_set_idx], 
                    repl=str(self.turbine_mapping[file_set_idx][re.search(self.turbine_signature[file_set_idx], col).group(0)]), 
                    string=col) for col in df_queries.collect_schema().names() if col != "time"})
        
        merged_path = os.path.join(temp_save_dir, f"df_{file_set_idx}_{i}.parquet") 
        self.sort_resample_refill(df_queries).collect().write_parquet(merged_path, statistics=False)
        return merged_path 

    # @profile
    def _join_dfs(self, file_suffix, dfs):
        # logging.info(f"✅ Started joins for {file_suffix}-th collection of files.") 
        all_cols = set()
        first_df = True
        for d, df in enumerate(dfs):
            # df = df.collect()
            new_cols = df.collect_schema().names()
            
            if first_df:
                df_query = df
                first_df = False
            else:
                # existing_cols = list(all_cols.intersection(new_cols))
                existing_cols = list(all_cols.intersection(new_cols))
                if existing_cols:
                    # data for the turbine contained in this frame has already been added, albeit from another day
                    df_query = df_query.join(df, on="time", how="full", coalesce=True)\
                                        .with_columns([pl.coalesce(col, f"{col}_right").alias(col) for col in existing_cols])\
                                        .select(~cs.ends_with("right"))
                else:
                    df_query = df_query.join(df, on="time", how="full", coalesce=True)

            all_cols.update(new_cols)
            all_cols.remove("time")
        
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

    def get_turbine_ids(self, turbine_signature, df_query, sort=False):
        turbine_ids = set()
        if "turbine_id" in df_query.collect_schema().names():
            turbine_ids.update(df_query.select("turbine_id").collect().to_series().unique())
        else:
            for col in df_query.collect_schema().names():
                match = re.search(turbine_signature, col)
                if match:
                    turbine_ids.add(match.group())
        if sort:
            return sorted(turbine_ids)
        else:
            return turbine_ids

    # @profile
    def _read_single_file(self, file_set_idx: int, file_number:int, raw_file_path: str, processed_file_path: str) -> pl.LazyFrame:
        
        try:
            start_time = time.time()
            if self.data_format[file_set_idx] == "netcdf":
                with nc.Dataset(raw_file_path, 'r') as dataset:
                    time_var = dataset.variables[self.feature_mapping[file_set_idx]["time"]]
                    time_var = nc.num2date(times=time_var[:], 
                                    units=time_var.units, 
                                    calendar=time_var.calendar, 
                                    only_use_cftime_datetimes=False, 
                                    only_use_python_datetimes=True)
                     
                    data = {
                        **{
                            'turbine_id': re.findall(self.turbine_signature[file_set_idx], os.path.basename(raw_file_path)) * len(time_var),
                            'time': time_var.tolist(),  # Convert to Polars datetime
                        },
                        **{k: dataset.variables[v][:] for k, v in self.feature_mapping[file_set_idx].items() if k not in ["time", "turbine_id"] and v in dataset.variables}
                    }

                    # self.turbine_ids = self.turbine_ids.union(set(data["turbine_id"]))

                    # remove the rows with all nans (corresponding to rows where excluded columns would have had a value)
                    # and bundle all values corresponding to identical time stamps together
                    # forward fill missing values
                    
                    available_columns = list(data.keys()) 
                    target_features = list(self.feature_mapping[file_set_idx])
                    df_query = pl.LazyFrame(data).fill_nan(None)\
                                                    .with_columns(pl.col("time").dt.round(f"{self.dt}s").alias("time").cast(pl.Datetime(time_unit="us")))\
                                                    .select([cs.contains(feat) for feat in target_features])\
                                                    .filter(pl.any_horizontal(cs.numeric().is_not_null()))
                    # just the turbine ids found in this file
                    turbine_ids = self.get_turbine_ids(self.turbine_signature[file_set_idx], df_query)
                    
            elif self.data_format[file_set_idx] in ["csv", "parquet"]:
                if self.data_format[file_set_idx] == "csv":
                    df_query = pl.scan_csv(raw_file_path, low_memory=False)
                elif self.data_format[file_set_idx] == "parquet":
                    df_query = pl.scan_parquet(raw_file_path) 
                
                available_columns = df_query.collect_schema().names()
                assert all(any(bool(re.search(feat, col)) for col in available_columns) for feat in self.source_features[file_set_idx]), "All values in feature_mapping must exist in data columns."

                # Select only relevant columns and handle missing values
                # df_query = df_query.select(self.source_features)\
                df_query = df_query.select(*[cs.matches(feat) for feat in self.source_features[file_set_idx]])
                source_features = df_query.collect_schema().names()
                
                # just the turbine ids found in this file
                turbine_ids = self.get_turbine_ids(self.turbine_signature[file_set_idx], df_query)
                
                # Apply column mapping after selecting relevant columns
                rename_dict = {}
                for src in source_features:
                    feature_type = None
                    tid = None
                    for src_signature in self.source_features[file_set_idx]:
                        if re.search(src_signature, src):
                            feature_type = self.reverse_feature_mapping[file_set_idx][src_signature]
                            tid = re.search(self.turbine_signature[file_set_idx], src)
                            if tid:
                                tid = tid.group()
                    if feature_type and tid:
                        rename_dict[src] = f"{feature_type}_{tid}"
                    elif feature_type:
                        rename_dict[src] = feature_type

                df_query = df_query.rename(rename_dict)
                
                df_query = df_query.with_columns([pl.col(col).cast(pl.Float64) for col in df_query.collect_schema().names() if any(feat_type in col and feat_type != "time" for feat_type in self.target_features)])
                df_query = df_query.with_columns(cs.numeric().fill_nan(None))
                
                # target_features = list(self.target_features)
                target_features = list(self.feature_mapping[file_set_idx])
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
                        # del self.feature_mapping[file_set_idx][f"yaw_offset_{direc}"]
                        # self.feature_mapping[file_set_idx]["wind_direction"] = ""
                
                time_type = df_query.collect_schema()["time"]
                if time_type == pl.datatypes.String:
                    df_query = df_query.with_columns([
                                            # Convert time column to datetime
                                            pl.col("time").str.to_datetime().alias("time")
                                        ])
                else:
                    if time_type.time_zone is None:
                        df_query = df_query.with_columns(
                            time=pl.col("time").cast(pl.Datetime(time_unit="us"))
                        )
                    else:
                        df_query = df_query.with_columns(
                            time=pl.col("time").dt.convert_time_zone("UTC").cast(pl.Datetime(time_unit="us"))
                        )
                
                # Check if data is already in wide format
                available_columns = df_query.collect_schema().names()
                
                # remove the rows with all nans (corresponding to rows where excluded columns would have had a value)
                # and bundle all values corresponding to identical time stamps together
                # forward fill missing values
                df_query = df_query.with_columns(pl.col("time").dt.round(f"{self.dt}s").alias("time"))\
                                    .select([cs.contains(feat) for feat in target_features])\
                                    .filter(pl.any_horizontal(cs.numeric().is_not_null()))
                
            # pivot table to have columns for each turbine and measurement if not originally in wide format
            is_already_wide = all(f"{feature}_{tid}" in available_columns 
                for feature in target_features for tid in turbine_ids if feature != "time")
            if not is_already_wide:
                pivot_features = [col for col in available_columns if col not in ['time', 'turbine_id']]
                df_query = df_query.collect().pivot(
                    index="time",
                    on="turbine_id",
                    values=pivot_features,
                    aggregate_function=pl.element().drop_nulls().first(),
                    sort_columns=True
                ).lazy().sort("time")
                
            df_query.collect().write_parquet(processed_file_path, statistics=False)
            logging.info(f"✅ Processed {file_number + 1}-th {raw_file_path} and saved to {processed_file_path}. Time: {time.time() - start_time:.2f} s")
            return processed_file_path
        
        except Exception as e:
            logging.error(f"❌ Error processing file {raw_file_path}: {str(e)}")
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