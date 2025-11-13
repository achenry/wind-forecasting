### This file contains class and method to: 
### - load the scada data, 
### - convert timestamps to datetime objects
### - convert circular measurements to sinusoidal measurements
### - normalize data

import glob
import os
import time
from datetime import datetime, timedelta
from typing import List, Optional
from pathlib import Path
import logging
import re
import pickle
import glob

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from shutil import move
from psutil import virtual_memory
# import gc
# from datetime.datetime import strptime
# from memory_profiler import profile

import netCDF4 as nc

import polars as pl
import polars.selectors as cs
import numpy as np
# from sklearn.preprocessing import MinMaxScaler

# mpi_exists = False
# try:
#     from mpi4py import MPI
#     from mpi4py.futures import MPICommExecutor
#     mpi_exists = True
# except:
#     logging.info("No MPI available on system.")


SECONDS_PER_MINUTE = np.float64(60)
SECONDS_PER_HOUR = SECONDS_PER_MINUTE * 60
SECONDS_PER_DAY = SECONDS_PER_HOUR * 24
SECONDS_PER_YEAR = SECONDS_PER_DAY * 365  # non-leap year, 365 days
FFILL_LIMIT = 10 * SECONDS_PER_MINUTE 

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
JOIN_CHUNK = 100 #int(2000)
# logging.info("Hi 6")
class DataLoader:
    """_summary_
       - load the scada data, 
       - convert timestamps to datetime objects
       - convert circular measurements to sinusoidal measurements
       - normalize data 
    """
    def __init__(self, 
                 data_dir: List[Path],
                 file_signature: List[str],
                 save_path: Path,
                 multiprocessor: str | None,
                 dt: int,
                 split_dt: int,
                 min_continuous_duration: int,
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
        self.split_dt = split_dt
        self.min_continuous_duration = min_continuous_duration
        self.data_format = [df.lower() for df in data_format]
        assert all(df in ["netcdf", "csv", "parquet"] for df in self.data_format)
        self.feature_mapping = feature_mapping
        self.reverse_feature_mapping = []
        for fm in self.feature_mapping:
            self.reverse_feature_mapping.append({})
            for tgt, src in fm.items():
                if type(src) is list:
                    for s in src:
                        self.reverse_feature_mapping[-1][s] = tgt
                else:
                    self.reverse_feature_mapping[-1][src] = tgt
        # self.reverse_feature_mapping = [dict((src, tgt) for tgt, src in fm.items()) for fm in self.feature_mapping]
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
            self.turbine_mapping = [dict((k, tm[k]) for k in sorted(tm)) for tm in self.turbine_mapping]
        
        # for dd, fs, ts, tm in zip(data_dir, file_signature, self.turbine_signature, self.turbine_mapping):
        #     if all(len(re.findall(ts, os.path.basename(fp))) 
        #            for fp in glob.glob(os.path.join(dd, fs), recursive=True)):
                # assert merge_chunk % len(tm) == 0, "merge_chunk in yaml config must be a multiple of the number of turbines in turbine_signature, for file formats with files for each turbine id"
        
        # Get all the wts in the folder @Juan 10/16/24 used os.path.join for OS compatibility
        self.file_paths = [sorted(glob.glob(os.path.join(dd, fs), recursive=True),
                                  key=lambda fp: (datetime.strptime(re.search(ds[0], os.path.basename(fp)).group(0), ds[1]) if ds is not None else 0,
                                                    re.search(ts, os.path.basename(fp)).group(0) if len(re.findall(ts, os.path.basename(fp))) else 0)) 
                           for dd, fs, ds, ts in zip(data_dir, file_signature, self.datetime_signature, self.turbine_signature)]
    
    @staticmethod
    def sizeof_fmt(num, suffix='B'):
        ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
        for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
            if abs(num) < 1024.0:
                return "%3.1f %s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f %s%s" % (num, 'Yi', suffix)
     
    def read_multi_files(self, temp_save_dir, read_single_files="all") -> pl.LazyFrame | None:
        read_start = time.time()
        
        if self.multiprocessor is not None:
            # if self.multiprocessor == "mpi" and mpi_exists:
            #     executor = MPICommExecutor(MPI.COMM_WORLD, root=0)
            # else:  # "cf" case
            executor = ProcessPoolExecutor(mp_context=mp.get_context("spawn"), max_workers=int(os.environ.get("MAX_WORKERS", mp.cpu_count())))
            with executor as ex:
                if read_single_files:
                    logging.info(f"✅ Started reading {sum(len(fp) for fp in self.file_paths)} files.")
                    
                    for file_set_idx, fp in enumerate(self.file_paths):
                        if not fp:
                            raise FileExistsError(f"⚠️ File with signature {self.file_signature[file_set_idx]} in directory {self.data_dir[file_set_idx]} doesn't exist.")

                    
                    init_used_ram = virtual_memory().percent
                    assert init_used_ram < self.ram_limit - 5, f"RAM limit in yaml config must be at least 5% greater than initial ram value of {init_used_ram}%."
                    
                    if read_single_files == "all":
                        file_futures = [] #4% increase in mem
                        for file_set_idx in range(len(self.file_paths)):
                            file_futures.append([])
                            for f, file_path in enumerate(self.file_paths[file_set_idx]):
                                processed_path = os.path.join(temp_save_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}.parquet")
                                file_futures[-1].append(ex.submit(self._read_single_file, file_set_idx, f, file_path, 
                                                              processed_path, len(self.file_paths[file_set_idx])))
                    elif read_single_files == "unprocessed":
                        unprocessed_file_path_idx = [[f for f, file_path in enumerate(self.file_paths[file_set_idx]) if not os.path.exists(os.path.join(temp_save_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}.parquet"))] 
                                                     for file_set_idx in range(len(self.file_paths))]
                        
                        file_futures = []
                        for file_set_idx in range(len(self.file_paths)):
                            file_futures.append([])
                            for f, file_path in enumerate(self.file_paths[file_set_idx]):
                                if f in unprocessed_file_path_idx[file_set_idx]:
                                    processed_path = os.path.join(temp_save_dir, 
                                                            f"{os.path.splitext(os.path.basename(file_path))[0]}.parquet")
                                    file_futures[-1].append(ex.submit(self._read_single_file, file_set_idx, f, file_path, 
                                                processed_path, len(self.file_paths[file_set_idx])))
                    
                    for file_set_idx in range(len(self.file_paths)):
                        for f, fut in enumerate(file_futures[file_set_idx]):
                            logging.info(f"Fetching results for file set {file_set_idx} file no {f}.")
                            fut.result()

            logging.info(f"Started fetching results from {sum(len(fp) for fp in self.file_paths)} files.")
            if (read_single_files == "all") or (read_single_files == "unprocessed" and f in unprocessed_file_path_idx[file_set_idx]):
                processed_file_paths = []
                file_set_indices = []
                for file_set_idx in range(len(self.file_paths)):
                    processed_file_paths.append([])
                    file_set_indices.append(file_set_idx)
                    for f, file_path in enumerate(self.file_paths[file_set_idx]):
                        fn = f"{os.path.splitext(os.path.basename(file_path))[0]}.parquet"
                        fp = os.path.join(temp_save_dir, fn)
                        if os.path.exists(fp): 
                            processed_file_paths[-1].append(fp)
            else:
                processed_file_paths = [glob.glob(os.path.join(temp_save_dir, f"{os.path.splitext(self.file_signature[file_set_idx])[0]}.parquet")) for file_set_idx in range(len(self.file_paths))]
                            # logging.info(f"- Adding {fp} to list of processed files")
                        # else:
                        #     logging.warning(f"File {file_path} could not be processed, skipping.")
                    
                    # for name, size in sorted(((name, sys.getsizeof(value)) for name, value in list(
                    #                     locals().items())), key= lambda x: -x[1])[:3]:
                    #     print("{:>30}: {:>8}".format(name, DataLoader.sizeof_fmt(size)))
                    
                    
        else:
            logging.info(f"✅ Started reading {sum(len(fp) for fp in self.file_paths)} files.")
            logging.info(f"🔧 Using single process executor.")
            if not self.file_paths:
                raise FileExistsError(f"⚠️ File with signature {self.file_signature} in directory {self.data_dir} doesn't exist.")
            
            processed_file_paths = []
            for file_set_idx in range(len(self.file_paths)):
                processed_file_paths.append([])
                for f, file_path in enumerate(self.file_paths[file_set_idx]):
                    used_ram = virtual_memory().percent
                    
                    logging.info(f"Used RAM = {used_ram}%. Continue adding to buffer of {len(processed_file_paths)} processed single files.")
                    processed_fp = os.path.join(temp_save_dir, 
                                                        f"{os.path.splitext(os.path.basename(file_path))[0]}.parquet")
                    if read_single_files == "all":
                        res = self._read_single_file(file_set_idx, f, file_path, 
                                                processed_fp, len(self.file_paths[file_set_idx]))
                    elif (read_single_files == "unprocessed") and (not os.path.exists(processed_fp)):
                        res = self._read_single_file(file_set_idx, f, file_path, 
                                                processed_fp, len(self.file_paths[file_set_idx]))
                    else:
                        res = 1
                        
                    # all_columns.update(file_cols)
                    
                    if res is not None:
                        processed_file_paths[-1].append(processed_fp)
            
        if self.turbine_mapping: # if not none, all turbine signatures have been transformed
            self.turbine_signature = "\\d+$"
        else:
            self.turbine_signature = self.turbine_signature[0]
        
        RUN_ONCE = (self.multiprocessor != "mpi") or (self.multiprocessor is None) # or (self.multiprocessor == "mpi" and mpi_exists and (MPI.COMM_WORLD.Get_rank()) == 0)

        if RUN_ONCE:
            # if len(merged_paths):    
            logging.info(f"🔗 Finished reading files. Time elapsed: {time.time() - read_start:.2f} s")
            # if len(merged_paths) > 1:
            if sum(len(pfp) for pfp in processed_file_paths) > 1: 
                logging.info(f"Concatenating and running final sort/resample/fill.")
                
                # df_queries = []
                for file_set_idx in range(len(self.file_paths)):
                    self.merge_multiple_files(file_set_idx, processed_file_paths[file_set_idx], file_set_idx, temp_save_dir, reload=read_single_files=="all")
                
                # Write to final parquet
                logging.info(f"Saving final Parquet file into {self.save_path}, used ram = {virtual_memory().percent}%")
                merged_fp = glob.glob(os.path.join(temp_save_dir, "merged_*_*.parquet"))
                merged_fp = sorted(merged_fp, key=lambda fp: tuple(int(x) for x in re.findall("(?<=merged)_(\\d+)_(\\d+)", os.path.basename(fp))[0]))
                
                if len(merged_fp) == 1:
                    move(merged_fp[0], self.save_path)
                else:
                    pl.scan_parquet(merged_fp, glob=True).sink_parquet(self.save_path, maintain_order=True)
                
            else:
                logging.info(f"Moving only batch to {self.save_path}.")
                move(processed_file_paths[0][0], self.save_path)
            
            df_query = pl.scan_parquet(self.save_path)
            # turbine ids found in all files so far TODO check if sorted
            self.turbine_ids = self.get_turbine_ids(self.turbine_signature, df_query, sort=True)
            
            logging.info(f"Final Parquet file saved into {self.save_path}")
            
            return df_query
    
    def _get_schema(self, fp):
        return pl.scan_parquet(fp).collect_schema()
    
    def _get_time_bounds(self, fp):
        x = pl.scan_parquet(fp).select(pl.col("time").first().alias("start"), pl.col("time").last().alias("end")).collect()
        return x["start"].item(), x["end"].item()
    
    def _split_df(self, df, next_split_indices, j, file_set_idx_offset, n_splits):
        logging.info(f"Splitting {j}th of {n_splits} continuous dataframes. Used RAM = {virtual_memory().percent}%.")
        return df.slice(next_split_indices[0], next_split_indices[1] - next_split_indices[0])\
                        .with_columns(file_set_idx=file_set_idx_offset + j)
    
    def merge_multiple_files(self, file_set_idx, processed_file_paths, i, temp_save_dir, reload):
        
        logging.info(f"Started scanning schema. Used RAM = {virtual_memory().percent}%.")
        if reload or not os.path.exists(os.path.join(temp_save_dir, f"full_schema_{file_set_idx}_{i}.pkl")):
            if self.multiprocessor is not None:
                executor = ProcessPoolExecutor(mp_context=mp.get_context("spawn"), max_workers=int(os.environ.get("MAX_WORKERS", mp.cpu_count())))
                with executor as ex:
                    if ex is not None:
                        schema_futures = [ex.submit(self._get_schema, fp) for fp in processed_file_paths]
                
                full_schema = schema_futures[0].result()
                logging.info(f"  - Schema for {processed_file_paths[0]}: {full_schema}")
                for f, fut in enumerate(schema_futures[1:]):
                    schema = fut.result()
                    logging.info(f"  - Schema for {processed_file_paths[f+1]}: {schema}")
                    full_schema.update(schema)
            else:
                full_schema = pl.scan_parquet(processed_file_paths[0]).collect_schema()
            
                for f, fp in enumerate(processed_file_paths[1:]):
                    logging.info(f"  - Scanning {f}th of {len(processed_file_paths)} files. Used RAM = {virtual_memory().percent}%.")
                    full_schema.update(pl.scan_parquet(fp).collect_schema())
            
            with open(os.path.join(temp_save_dir, f"full_schema_{file_set_idx}_{i}.pkl"), "wb") as fp:
                pickle.dump(full_schema, fp)
                
        else:
            with open(os.path.join(temp_save_dir, f"full_schema_{file_set_idx}_{i}.pkl"), "rb") as fp:
                full_schema = pickle.load(fp)
            logging.info(f"Scanned existing schema: {full_schema}")
            
        logging.info(f"Finished scanning schema. Used RAM = {virtual_memory().percent}%.")
        
        # if os.path.exists(os.path.join(temp_save_dir, f"time_bounds_{file_set_idx}_{i}.pkl")):
        #     with open(os.path.join(temp_save_dir, f"time_bounds_{file_set_idx}_{i}.pkl"), "rb") as fp:
        #         all_time_bounds = pickle.load(fp)
        #     all_time_bounds.write_parquet(os.path.join(temp_save_dir, f"time_bounds_{file_set_idx}_{i}.parquet"))
        
        logging.info(f"Started scanning time bounds. Used RAM = {virtual_memory().percent}%.")
        if reload or not os.path.exists(os.path.join(temp_save_dir, f"time_bounds_{file_set_idx}_{i}.parquet")):
            all_time_bounds = []
            if self.multiprocessor is not None:
                executor = ProcessPoolExecutor(mp_context=mp.get_context("spawn"), max_workers=int(os.environ.get("MAX_WORKERS", mp.cpu_count())))
                with executor as ex:
                    if ex is not None:
                        time_bounds_futures = [ex.submit(self._get_time_bounds, fp) for fp in processed_file_paths]
                for f, fut in enumerate(time_bounds_futures):
                    start, end = fut.result()
                    all_time_bounds.append((start, end, f))
                    logging.info(f"  - Time Bounds for {processed_file_paths[f]}: {all_time_bounds[-1]}")
            else:
                for f, fp in enumerate(processed_file_paths):
                    start, end = self._get_time_bounds(fp)
                    all_time_bounds.append((start, end, f))
                    logging.info(f"  - Time Bounds for {processed_file_paths[f]}: {all_time_bounds[-1]}")
            
            all_time_bounds = pl.DataFrame(all_time_bounds, schema=["start", "end", "file_index"], orient="row").sort("start").drop_nulls()
            all_time_bounds.write_parquet(os.path.join(temp_save_dir, f"time_bounds_{file_set_idx}_{i}.parquet"))
                
        else:
            all_time_bounds = pl.read_parquet(os.path.join(temp_save_dir, f"time_bounds_{file_set_idx}_{i}.parquet"))
            
            logging.info(f"Scanned existing time_bounds.")
            
        logging.info(f"Finished scanning time bounds. Used RAM = {virtual_memory().percent}%.")
        
        if len(processed_file_paths) == 1:
            df_queries = pl.scan_parquet(processed_file_paths[0])  # If single file, no need to join
        else:
            # concatenate and forward fill file groups with continuous time spans
            # split by discontinuity, all df_queries are sorted up to this point
            turbine_ids = set()
            for n in full_schema.names():
                match = re.findall(self.turbine_signature, n)
                if len(match):
                    turbine_ids.add(match[0])
            turbine_ids = sorted(turbine_ids, key=lambda tid: int(re.search("\\d+", tid).group(0)))
            
            if reload or not os.path.exists(os.path.join(temp_save_dir, f"all_grouped_{file_set_idx}_{i}.parquet")):
                logging.info(f"Started grouping of {len(processed_file_paths)} files for file set {file_set_idx}, merge index {i}. Used RAM = {virtual_memory().percent}%.")

                bounds = all_time_bounds.select(pl.col("start").first().alias("first"), pl.col("end").last().alias("last"))
                for tid in turbine_ids:
                    
                    asset_schema = pl.Schema({k: v for k, v in full_schema.items() if k == "time" or re.search(f".*?(?={tid})", k)})
                    # loop through all of this asset's files
                    asset_processed_files = [fp for fp in processed_file_paths if re.findall(tid, os.path.basename(fp))]
                    for fp in asset_processed_files:
                        if all_time_bounds.filter(pl.col("file_index") == processed_file_paths.index(fp)).is_empty():
                            logging.error(f"File {fp} has no all_time_bounds entry! Its index in processed_file_paths is {processed_file_paths.index(fp)}")
                    
                    asset_processed_files = sorted(asset_processed_files, key=lambda fp: all_time_bounds.filter(pl.col("file_index") == processed_file_paths.index(fp)).select("start").item())
                    
                    logging.info(f"  - Grouping {len(asset_processed_files)} files for turbine id {tid} for file set {file_set_idx}, merge index {i}. Used RAM = {virtual_memory().percent}%.")
                    df = pl.scan_parquet(asset_processed_files, schema=asset_schema, missing_columns="insert")\
                            .sort("time")\
                                .group_by("time", maintain_order=True)\
                                .agg(cs.numeric().mean())
                    
                    self._resample_df(df, bounds, fill_null=False)\
                        .sink_parquet(os.path.join(temp_save_dir, f"grouped_{file_set_idx}_{i}_{tid}.parquet"), maintain_order=True)
                
                grouped_file_paths = glob.glob(os.path.join(temp_save_dir, f"grouped_{file_set_idx}_{i}_*.parquet"))
                grouped_file_paths = [fp for fp in grouped_file_paths if re.findall(self.turbine_signature, os.path.basename(fp))]
                grouped_file_paths = sorted(grouped_file_paths, key=lambda fp: re.search(f"(?<=grouped_{file_set_idx}_{i}_)({self.turbine_signature})", os.path.splitext(os.path.basename(fp))[0]).group())
                pl.concat([pl.scan_parquet(fp).select(pl.exclude("time")) if f > 0 else pl.scan_parquet(fp) for f, fp in enumerate(grouped_file_paths) if os.path.getsize(fp)], how="horizontal")\
                  .sink_parquet(os.path.join(temp_save_dir, f"all_grouped_{file_set_idx}_{i}.parquet"), maintain_order=True)
                  
            else:
                logging.info(f"Loading groupings of {len(processed_file_paths)} files for file set {file_set_idx}, merge index {i} from file. Used RAM = {virtual_memory().percent}%.")
            
            df_queries = pl.scan_parquet(os.path.join(temp_save_dir, f"all_grouped_{file_set_idx}_{i}.parquet"))
            
            logging.info(f"Finished grouping {len(processed_file_paths)} files for file set {file_set_idx}, merge index {i}. Used RAM = {virtual_memory().percent}%.")
            
            logging.info(f"Found columns:")
            for col, dtype in full_schema.items():
                logging.info(f"  - {col} | {dtype}")
            
            logging.info(f"Sorting columns based on turbine signature {self.turbine_signature}. Used RAM = {virtual_memory().percent}%.")
            full_schema = full_schema.names()
            del full_schema[full_schema.index("time")]
            full_schema = sorted(full_schema, key=lambda col: (re.search(f".*?(?={self.turbine_signature})", col).group(0), 
                                                int(re.search("\\d+", re.search(self.turbine_signature, col).group(0)).group(0))))
            logging.info(f"Sorting columns in dataframe. Used RAM = {virtual_memory().percent}%.")
            df_queries = df_queries.select(["time"] + full_schema)
            logging.info(f"Finished sorting columns. Used RAM = {virtual_memory().percent}%.")
                
            # convert to common turbine_id over multiple filetypes
            if self.turbine_mapping is not None:
                turbine_ids = self.get_turbine_ids(self.turbine_signature[file_set_idx], df_queries, sort=True) # turbine ids available in this collection of file paths (may not represent all) 
                # make sure that turbine mapping accounts for all turbine ids found in files
                assert all(tid in self.turbine_mapping[file_set_idx] for tid in turbine_ids), \
                    f"""check turbine_mapping in yaml config, should have n_turbines length of distinct turbine ids, 
                    and all ids found in the data, {turbine_ids}, should be included in the keys, {self.turbine_mapping[file_set_idx]}, 
                    for the set of processed file paths, {[os.path.basename(fp) for fp in processed_file_paths]} for file set {file_set_idx}""" 
                
                logging.info(f"Renaming DataFrame for file set {file_set_idx}, merge index {i}, to common turbine_signature. Used RAM = {virtual_memory().percent}%.") 
                df_queries = df_queries.rename({
                    col: 
                    re.sub(pattern=self.turbine_signature[file_set_idx], 
                        repl=str(self.turbine_mapping[file_set_idx][re.search(self.turbine_signature[file_set_idx], col).group(0)]), 
                        string=col) for col in df_queries.collect_schema().names() if col != "time"})
            
            assert os.path.exists(temp_save_dir), f"temp_save_dir={temp_save_dir} is not available for file set {file_set_idx}, merge index {i}"
            
            split_indices_fp = os.path.join(temp_save_dir, f"split_indices_{file_set_idx}_{i}.parquet")
            file_set_idx_offset = sum(len(file_set) for file_set in self.file_paths[:file_set_idx])
            if reload or not os.path.exists(os.path.join(temp_save_dir, f"split_indices_{file_set_idx}_{i}.parquet")):
                logging.info(f"Started generating split_indices for file set {file_set_idx}, merge index {i}. Used RAM = {virtual_memory().percent}%.")
                # fetch only rows where there is at least one non-null numeric value (excluding index column)
                df_queries.with_row_index()\
                          .filter(~pl.all_horizontal(cs.numeric().exclude("index").is_null()))\
                          .select("index", dt=pl.col("time").diff())\
                          .slice(1)\
                          .filter(pl.col("dt") > timedelta(seconds=self.split_dt))\
                          .select("index")\
                          .sink_parquet(split_indices_fp, maintain_order=True)
                
                pl.concat([pl.Series([0]).alias("index").to_frame(),
                            pl.read_parquet(split_indices_fp),
                            pl.Series([df_queries.select(pl.len()).collect().item()]).alias("index").to_frame()], how="vertical_relaxed")\
                  .write_parquet(split_indices_fp)
                
            else:
                logging.info(f"Loading split_indices for file set {file_set_idx}, merge index {i}. Used RAM = {virtual_memory().percent}%.")
            
            split_indices = pl.read_parquet(split_indices_fp)
            n_splits = split_indices.select(pl.len()).item() - 1
            
            logging.info(f"Finished generating split_indices {len(processed_file_paths)} files for file set {file_set_idx}, merge index {i}. Used RAM = {virtual_memory().percent}%.")
            
            logging.info(f"Started splitting and filling {len(processed_file_paths)} files for file set {file_set_idx}, merge index {i}. Used RAM = {virtual_memory().percent}%.")
            
            # df_queries_2 = []
            min_duration = timedelta(seconds=self.min_continuous_duration)
            j = 0
            jj = 0
            while j < n_splits:
                next_split_indices = split_indices.head(2).to_numpy().flatten() # takes 1 min
                dfq = df_queries.head(next_split_indices[1] - next_split_indices[0])
                not_all_null_bounds = dfq.with_row_index().filter(~pl.all_horizontal(cs.numeric().exclude("index").is_null())).select(pl.col("index").min().alias("start"), pl.col("index").max().alias("end")).collect()
                slc_len = not_all_null_bounds["end"].item() - not_all_null_bounds["start"].item() + 1
                dfq = dfq.slice(not_all_null_bounds["start"].item(), slc_len)
                # if (dur := dfq.select(pl.col("time").last() - pl.col("time").first()).collect().item()) < min_duration: # takes 1 min 
                if (dur := slc_len * timedelta(seconds=self.dt)) < min_duration: 
                    logging.info(f"Skipping split and fill {j} of {n_splits} continuous dataframes due to insufficient duration of {dur}. Used RAM = {virtual_memory().percent}%.")
                else:
                    if reload or not os.path.exists(os.path.join(temp_save_dir, f"merged_{file_set_idx_offset + jj}_{i}.parquet")):
                        logging.info(f"Splitting and filling {j}th of {n_splits} continuous dataframes. Used RAM = {virtual_memory().percent}%.")
                        dfq.with_columns(file_set_idx=file_set_idx_offset + jj)\
                           .fill_null(strategy="forward").fill_null(strategy="backward")\
                           .sink_parquet(os.path.join(temp_save_dir, f"merged_{file_set_idx_offset + jj}_{i}.parquet"), maintain_order=True)
                    else:
                        logging.info(f"Loaded existing split/filled {j}th of {n_splits} continuous dataframes. Used RAM = {virtual_memory().percent}%.")
                    jj += 1
                
                df_queries = df_queries.slice(next_split_indices[1] - next_split_indices[0]) 
                split_indices = split_indices.slice(1)
                j += 1
            
            # df_queries = df_queries_2
            logging.info(f"Finished splitting and filling {len(processed_file_paths)} files for file set {file_set_idx}, merge index {i}. Used RAM = {virtual_memory().percent}%.")
            
        return
    
    # other_df = pl.DataFrame({"apple": ["x", "y", "z", "a"], "ham": ["a", "b", "d", "a"]})
    
    def _resample_df(self, df, bounds, fill_null=False):
        # logging.info(f"Started resampling and refill from {bounds['first'].item()} to {bounds['last'].item()} for {i}th of dfs. Used RAM = {virtual_memory().percent}%.") 
        df = df.select(pl.datetime_range(start=bounds["first"].item(),
                                    end=bounds["last"].item(),
                                    interval=f"{self.dt}s", 
                                    time_unit=df.collect_schema()["time"].time_unit).alias("time"))\
          .join(df, on="time", how="left", maintain_order="left")
        
        if fill_null:
          df = df.fill_null(strategy="forward").fill_null(strategy="backward")
        
        return df
        
        # logging.info(f"Finished resampling and refill for {i}th of {num_files_set_indices}. Used RAM = {virtual_memory().percent}%.") 
        # return
    
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
            if len(re.findall("\\d+", list(turbine_ids)[0])):
                return sorted(turbine_ids, key=lambda tid: int(re.search("\\d+", tid).group(0)))
            else:
                return sorted(turbine_ids)
        else:
            return turbine_ids

    
    def _read_single_file(self, file_set_idx: int, file_number:int, raw_file_path: str, processed_file_path: str, num_file_paths: int) -> pl.LazyFrame:
        
        try:
            start_time = time.time()
            if self.data_format[file_set_idx] == "netcdf":
                with nc.Dataset(raw_file_path, 'r') as dataset:
                    
                    # logging.info(f"✅ Scanned {file_number + 1}-th {raw_file_path}")
                    if type(self.feature_mapping[file_set_idx]["time"]) is list:
                        for var_name in self.feature_mapping[file_set_idx]["time"]:
                            if var_name in dataset.variables:
                                time_var = dataset.variables[var_name]
                                break
                    else:
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
                        # **{k: dataset.variables[v][:] for k, v in self.feature_mapping[file_set_idx].items() if k not in ["time", "turbine_id"] and v in dataset.variables}
                    }
                    
                    for k, v in self.feature_mapping[file_set_idx].items():
                        if k not in ["time", "turbine_id"]:
                            if type(v) is list:
                                for vv in v:
                                    if vv in dataset.variables:
                                        data[k] = dataset.variables[vv][:]
                            elif v in dataset.variables:
                                data[k] = dataset.variables[v][:]
                    
                    # If wind_direction variable is not present, calculate it from nacelle_direction and yaw_offset
                    target_features = list(self.feature_mapping[file_set_idx])
                    if "wind_direction" not in target_features:
                        
                        if "nacelle_direction" in target_features:
                            if "yaw_offset_cw" in target_features:
                                delta = 1
                                direc = "cw" 
                            elif "yaw_offset_ccw" in target_features:
                                delta = -1
                                direc = "ccw"
                            else:
                                raise Exception("No wind_direction or yaw_offset variable found in data.")
                            
                            data[f"wind_direction"] = data[f"nacelle_direction"] + delta * data[f"yaw_offset_{direc}"] 
                            del data[f"yaw_offset_{direc}"]
                                
                            del target_features[target_features.index(f"yaw_offset_{direc}")]
                            target_features.append("wind_direction")
                        else:
                            raise Exception("No wind direction, or nacelle direction variable found in data.")

                    # self.turbine_ids = self.turbine_ids.union(set(data["turbine_id"]))

                    # remove the rows with all nans (corresponding to rows where excluded columns would have had a value)
                    # and bundle all values corresponding to identical time stamps together
                    # forward fill missing values
                    
                    available_columns = list(data.keys()) 
                     
                    # counts, bins = np.histogram(x.select(pl.col("time").dt.round(f"{self.dt}s").alias("time").cast(pl.Datetime(time_unit="us")).unique()).sort("time").select(pl.all().diff()).to_pandas()["time"].astype('timedelta64[s]').astype('int').iloc[1:])
                    df_query = pl.LazyFrame(data).fill_nan(None)\
                                                    .with_columns(pl.col("time").dt.round(f"{self.dt}s").alias("time").cast(pl.Datetime(time_unit="us")))\
                                                    .select([cs.contains(feat) for feat in target_features])\
                                                    .filter(pl.any_horizontal(cs.numeric().is_not_null()))
                    # just the turbine ids found in this file
                    turbine_ids = self.get_turbine_ids(self.turbine_signature[file_set_idx], df_query)
                    
                    if len(turbine_ids) == 0:
                        turbine_ids.update(data["turbine_id"])
                    
            elif self.data_format[file_set_idx] in ["csv", "parquet"]:
                if self.data_format[file_set_idx] == "csv":
                    df_query = pl.scan_csv(raw_file_path, low_memory=False)
                elif self.data_format[file_set_idx] == "parquet":
                    df_query = pl.scan_parquet(raw_file_path)
                    
                logging.info(f"✅ Scanned {file_number + 1}-th {raw_file_path}") 
                
                available_columns = df_query.collect_schema().names()
                assert all(any(bool(re.search(feat, col)) for col in available_columns) for feat in self.source_features[file_set_idx]), "All values in feature_mapping must exist in data columns."

                # Select only relevant columns
                df_query = df_query.select(*[cs.matches(feat) for feat in self.source_features[file_set_idx]])
                source_features = df_query.collect_schema().names()
                
                # Get just the turbine ids found in this file
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
                
                # Cast all numeric columns to Float64
                df_query = df_query.with_columns([pl.col(col).cast(pl.Float64) for col in df_query.collect_schema().names() if any(feat_type in col and feat_type != "time" for feat_type in self.target_features)])
                
                # Fill NaN values with None for uniform 'none' denotation
                df_query = df_query.with_columns(cs.numeric().fill_nan(None))
                
                # If wind_direction variable is not present, calculate it from nacelle_direction and yaw_offset
                target_features = list(self.feature_mapping[file_set_idx])
                if "wind_direction" not in target_features:
                    
                    if "nacelle_direction" in target_features:
                        if "yaw_offset_cw" in target_features:
                            delta = 1
                            direc = "cw" 
                        elif "yaw_offset_ccw" in target_features:
                            delta = -1
                            direc = "ccw"
                        else:
                            raise Exception("No wind_direction or yaw_offset variable found in data.")
                        
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
                    else:
                        raise Exception("No wind direction, or nacelle direction variable found in data.")
                    
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
                # counts, bins = np.histogram(df_query.collect().select(pl.col("time").dt.round(f"{self.dt}s").alias("time").cast(pl.Datetime(time_unit="us")).unique()).sort("time").select(pl.all().diff()).to_pandas()["time"].astype('timedelta64[s]').astype('int').iloc[1:])
                df_query = df_query.with_columns(pl.col("time").dt.round(f"{self.dt}s").alias("time"))\
                                    .select([cs.contains(feat) for feat in target_features])
                                    # .filter(pl.any_horizontal(cs.numeric().is_not_null()))
            
            # if df_query.select(pl.len()).collect().item() == 0:
            
            # pivot table to have columns for each turbine and measurement if not originally in wide format
            is_already_wide = all(f"{feature}_{tid}" in available_columns for feature in target_features for tid in turbine_ids if feature != "time")
            
            if is_already_wide:
                df_query = df_query.sort("time").collect()
            else:
                pivot_features = [col for col in available_columns if col not in ['time', 'turbine_id']]
                # unique_cols = [f"{col}_{tid}" for col in pivot_features for tid in turbine_ids]
                # agg_func = lambda col: col.mean()
                # index = pl.col("time")
                # on = pl.col("turbine_id")
                # values = pl.col(pivot_features)
                # df_query = df_query.group_by(index).agg(
                #     agg_func(values.filter(on == value)).alias(value) for value in unique_cols)
                
                df_query = df_query.collect().pivot(
                    index="time",
                    on="turbine_id",
                    values=pivot_features,
                    aggregate_function="mean", # if there are multiple values for a single time stamp, average them #pl.element().drop_nulls().first(),
                    sort_columns=True
                ).sort("time")
            
            if df_query.select(pl.len()).item():
                bounds = df_query.select(pl.col("time").first().alias("first"), pl.col("time").last().alias("last"))
                self._resample_df(df_query.lazy(), bounds, fill_null=False).collect().write_parquet(processed_file_path)
            # else:
            #     df_query.write_parquet(processed_file_path)
            
                assert os.path.exists(os.path.dirname(processed_file_path)), f"Temporary save directory {os.path.dirname(processed_file_path)} does not exist." 
                # df_query.collect().write_parquet(processed_file_path, statistics=False)
                
                logging.info(f"✅ Processed {file_number + 1}-th of {num_file_paths} {raw_file_path} and saved to {processed_file_path}. Time: {time.time() - start_time:.2f} s")
                return processed_file_path #, df_query.collect_schema().names()
            else:
                logging.warning(f"⚠️ No valid data found in {raw_file_path} after processing. Skipping saving to {processed_file_path}.")
                return None
        
        except Exception as e:
            logging.error(f"❌ Error processing file {raw_file_path} and saving to {processed_file_path}: {str(e)}")
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