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
# from pandas import to_datetime as pd_to_datetime # INFO: @Juan 10/16/24 Added pd_to_datetime to avoid conflict with polars to_datetime

SECONDS_PER_MINUTE = np.float64(60)
SECONDS_PER_HOUR = SECONDS_PER_MINUTE * 60
SECONDS_PER_DAY = SECONDS_PER_HOUR * 24
SECONDS_PER_YEAR = SECONDS_PER_DAY * 365  # non-leap year, 365 days
FFILL_LIMIT = 10 * SECONDS_PER_MINUTE 
# pl.Config.set_streaming_chunk_size(None)
# INFO: @Juan 10/02/24 Set Logging up
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
                 turbine_ids: list[str] = None,
                 multiprocessor: str | None = None,
                 chunk_size: int = 100000, # INFO: @Juan 10/16/24 Added arg for chunk size. 
                 desired_feature_types: list[str] = None,
                 dt: int | None = 5,
                 ffill_limit: int | None = None, # INFO:@Aoife 10/18/24 an argument for how many time steps the forward will should populate
                 data_format: str = "netcdf", # INFO:@Juan 10/14/24  Added arg for data format. Either "netcdf" or "csv" 
                 column_mapping: dict = None, # INFO:@Juan 10/14/24 Added arg for column mapping of csv files.
                 wide_format: bool = True): # INFO: @Juan 10/16/24 Added arg for wide format. If true, the data is loaded in wide format. If false, the data is loaded in long format.
        
        # INFO: @Juan 10/16/24 Added arg for data directory. 
        self.data_dir = data_dir
        self.save_path = save_path
        self.multiprocessor = multiprocessor
        self.dt = dt
        self.data_format = data_format.lower()
        self.column_mapping = column_mapping or {}
        self.chunk_size = chunk_size
        self.desired_feature_types = desired_feature_types or ["time", "turbine_id", "turbine_status", "wind_direction", "wind_speed", "power_output", "nacelle_direction"]
        self.wide_format = wide_format
        self.turbine_ids = turbine_ids
        self.ffill_limit = ffill_limit
        
        # Get all the wts in the folder @Juan 10/16/24 used os.path.join for OS compatibility
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, file_signature)))
        if not self.file_paths:
            raise FileExistsError(f"⚠️ File with signature {file_signature} in directory {data_dir} doesn't exist.")

    # INFO: @Juan 10/14/24 Added method to read multiple files based on the file signature. 
    def read_multi_files(self) -> pl.LazyFrame | None:
        if self.multiprocessor is not None:
            if self.multiprocessor == "mpi":
                executor = MPICommExecutor(MPI.COMM_WORLD, root=0)
            else:  # "cf" case
                executor = ProcessPoolExecutor()
            with executor as ex:
                start_time = time.time() # INFO: @Juan 10/16/24 Debbuging time measurements
                logging.info(f"✅ Starting read_multi_files with {len(data_loader.file_paths)} files")
                futures = [ex.submit(self._read_single_file, f, file_path) for f, file_path in enumerate(self.file_paths)]
                df_query = [fut.result() for fut in futures]
                df_query = [df for df in df_query if df is not None]
                logging.info(f"✅ Finished reading individual files. Time elapsed: {time.time() - start_time:.2f} s")
                return df_query
        else:
            logging.info(f"🔧 Using single process executor.")
            df_query = [self._read_single_file(f, file_path) for f, file_path in enumerate(self.file_paths)]
            df_query = [df for df in df_query if df is not None]
            return df_query
    
    def _join_dfs(self, file_suffix, dfs):
        logging.info(f"✅ Started joins for {file_suffix}-th collection of files.") 
        all_cols = set()
        first_df = True
        # temp_save_path = self.save_path.replace(".parquet", f"_{file_suffix}_tmp.parquet")
        save_path = self.save_path.replace(".parquet", f"_{file_suffix}.parquet")
        # df_query = None
        for d, df in enumerate(dfs):
            # df = df.collect()
            new_cols = [col for col in df.collect_schema().names() if col != "time"]
            if first_df:
                df_query = df
                # df.sink_parquet(temp_save_path)
                first_df = False
            else:
                #df_query = pl.scan_parquet(self.save_path.replace(".parquet", f"_{file_suffix}.parquet"))
                # df_query = pl.scan_parquet(save_path)
                existing_cols = list(all_cols.intersection(new_cols))
                if existing_cols:
                    # data for the turbine contained in this frame has already been added, albeit from another day
                    df_query = df_query.join(df, on="time", how="full", coalesce=True)\
                                        .with_columns([pl.coalesce(col, f"{col}_right").alias(col) for col in existing_cols])\
                                        .select(~cs.ends_with("right"))
                                        # .sink_parquet(temp_save_path)
                else:
                    df_query = df_query.join(df, on="time", how="full", coalesce=True)
                            # .sink_parquet(temp_save_path)
                    # df.sort("time").collect()
                    # df_query.filter((pl.col("time") >= df.select("time").min().collect().item()) & (pl.col("time") <= df.select("time").max().collect().item())).sort("time").select(pl.col("time"), cs.contains(df.columns[1].split("_")[-1])).collect()

            all_cols.update(new_cols)
            # df_query = df_query.collect(streaming=True).lazy()
            logging.info(f"🔗 Finished {d}-th join of {len(dfs)} of {file_suffix}-th collection of files.") 
            # os.rename(temp_save_path, save_path)
        df_query.sink_parquet(save_path, statistics=False)
            # df_query.sink_parquet(self.save_path) #, statistics=False)
        
        logging.info(f"🔗 Finished joins for {file_suffix}-th collection of files.")
        # return pl.scan_parquet(save_path)
        # return df_query

    def postprocess_multi_files(self, df_query) -> pl.LazyFrame | None:
        
        # if df_query:
        #     # join dfs of different turbine types and same timestamps, then concat remaining
        #     logging.info(f"✅ Finished reading individual files. Time elapsed: {time.time() - start_time:.2f} s")
        #     # logging.info("🔄 Starting concatenation of DataFrames")
        #     join_start = time.time()
        #     logging.info(f"✅ Started concatenation of {len(self.file_paths)} files.")
        #     unique_file_timestamps = set(re.findall(r"\.(\d{8})\.", fp)[0] for fp in self.file_paths)
        #     dfs_to_concat = []
        #     for ts in unique_file_timestamps:
        #         dfs_to_join = [df for d, df in enumerate(df_query) if ts in self.file_paths[d]]
        #         dfs_to_concat.append(self._join_dfs(ts,dfs_to_join))

        #     logging.info(f"🔗 Finished join. Time elapsed: {time.time() - join_start:.2f} s")
            
        #     concat_start = time.time()
        #     pl.concat(dfs_to_concat, how="vertical").collect().write_parquet(self.save_path, statistics=False)
        #     logging.info(f"🔗 Finished concat. Time elapsed: {time.time() - concat_start:.2f} s")

        #     return pl.scan_parquet(self.save_path)
        # else:
        #     logging.warning("⚠️ No data frames were created.")
        #     return None
         
        if self.multiprocessor is not None:
            if self.multiprocessor == "mpi":
                executor = MPICommExecutor(MPI.COMM_WORLD, root=0)
                # size = comm.Get_size()
            else:  # "cf" case
                executor = ProcessPoolExecutor()
            with executor as ex:
                # join dfs of different turbine types and same timestamps, then concat remaining
                
                # logging.info("🔄 Starting concatenation of DataFrames")
                join_start = time.time()
                logging.info(f"✅ Started join of {len(self.file_paths)} files.")
                unique_file_timestamps = set(re.findall(r"\.(\d{8})\.", fp)[0] for fp in self.file_paths)

                futures = [ex.submit(self._join_dfs, ts, 
                                        [df for d, df in enumerate(df_query) if ts in self.file_paths[d]]) 
                                        for ts in unique_file_timestamps]
                _ = [fut.result() for fut in futures]
                # dfs_to_concat = [fut.result() for fut in futures]

                logging.info(f"🔗 Finished join. Time elapsed: {time.time() - join_start:.2f} s")
                
                concat_start = time.time()
                dfs_to_concat = [pl.scan_parquet(self.save_path.replace(".parquet", f"_{ts}.parquet")) 
                                    for ts in unique_file_timestamps]
                pl.concat(dfs_to_concat, how="vertical").collect().write_parquet(self.save_path, statistics=False)
                logging.info(f"🔗 Finished concat. Time elapsed: {time.time() - concat_start:.2f} s")

                for ts in unique_file_timestamps:
                    os.remove(self.save_path.replace(".parquet", f"_{ts}.parquet"))

                return pl.scan_parquet(self.save_path)
                
        else:
            # join dfs of different turbine types and same timestamps, then concat remaining
            logging.info(f"✅ Finished reading individual files. Time elapsed: {time.time() - start_time:.2f} s")
            # logging.info("🔄 Starting concatenation of DataFrames")
            join_start = time.time()
            logging.info(f"✅ Started join of {len(self.file_paths)} files.")
            unique_file_timestamps = set(re.findall(r"\.(\d{8})\.", fp)[0] for fp in self.file_paths)
            dfs_to_concat = []

            for ts in unique_file_timestamps:
                dfs_to_join = [df for d, df in enumerate(df_query) if ts in self.file_paths[d]]
                dfs_to_concat.append(self._join_dfs(ts,dfs_to_join))

            logging.info(f"🔗 Finished join. Time elapsed: {time.time() - join_start:.2f} s")
            
            concat_start = time.time()
            pl.concat(dfs_to_concat, how="vertical").collect().write_parquet(self.save_path, statistics=False)
            logging.info(f"🔗 Finished concat. Time elapsed: {time.time() - concat_start:.2f} s")

            return pl.scan_parquet(self.save_path)
                    
        #             # df_query = pl.concat([df for df in df_query if df is not None]).lazy()
        #             num_processes = executor._max_workers
        #             df_query_list = df_query
        #             n_dfs = len(df_query_list)
        #             join_chunk = math.ceil(n_dfs / num_processes)
        #             df_slices = [slice(i * join_chunk, (i + 1) * join_chunk, 1) for i in range(math.ceil(n_dfs / join_chunk))]
        #             n_df_slices = len(df_slices)

        #             logging.info(f"✅ Started first concatenation of {len(df_query)} files in {n_df_slices} groups of {join_chunk}.")

        #             futures = [ex.submit(self._join_dfs, i, df_query_list[indices]) for i, indices in enumerate(df_slices)]
        #             _ = [fut.result() for fut in futures]
                    
        #             del df_query_list
        #             logging.info(f"🔗 Finished first concatenation of {len(self.file_paths)} files. Time elapsed: {time.time() - concat_start:.2f} s")

        #             concat_start = time.time()
        #             logging.info(f"✅ Started second concatenation of {n_df_slices} files.")
        #             df_query = self._join_dfs("", 
        #                         [pl.scan_parquet(self.save_path.replace(".parquet", f"_{i}.parquet")) for i in range(n_df_slices)])

        #             logging.info(f"🔗 Finished second concatenation of {n_df_slices} files. Time elapsed: {time.time() - concat_start:.2f} s")

        #             # with open(os.path.join(os.path.dirname(self.save_path), "all_df_query_explan.txt"), "w") as f:
        #             #     f.write(df_query.explain(streaming=True))

        #             # self._write_parquet(df_query)
                    
        #             return df_query #INFO: @Juan 10/16/24 Added .lazy() to the return statement to match the expected return type. Is this necessary?
        #         else:
        #             logging.warning("⚠️ No data frames were created.")
        #             return None
        # else:
        #     if df_query:
        #         # logging.info(f"✅ Finished reading individual files. Time elapsed: {time.time() - start_time:.2f} s")
        #         # # logging.info("🔄 Starting concatenation of DataFrames")
        #         # concat_start = time.time()
        #         # logging.info(f"✅ Started concatenation of {len(self.file_paths)} files.")
        #         # df_query = self._join_dfs("",df_query)

        #         # logging.info(f"🔗 Finished concatenation of {len(self.file_paths)} files. Time elapsed: {time.time() - concat_start:.2f} s")
        #         num_processes = 12 
        #         df_query_list = df_query
        #         n_dfs = len(df_query_list)
        #         join_chunk = math.ceil(n_dfs / num_processes)
        #         df_slices = [slice(i * join_chunk, (i + 1) * join_chunk, 1) for i in range(math.ceil(n_dfs / join_chunk))]
        #         n_df_slices = len(df_slices)

        #         logging.info(f"✅ Started first concatenation of {len(df_query)} files in {n_df_slices} groups of {join_chunk}.")

        #         dfs = []
        #         for i, indices in enumerate(df_slices):
        #             dfs.append(self._join_dfs(i, df_query_list[indices]))
                
        #         del df_query_list
        #         logging.info(f"🔗 Finished first concatenation of {len(self.file_paths)} files. Time elapsed: {time.time() - concat_start:.2f} s")

        #         concat_start = time.time()
        #         logging.info(f"✅ Started second concatenation of {n_df_slices} files.")
        #         df_query = self._join_dfs("", 
        #                     [pl.scan_parquet(self.save_path.replace(".parquet", f"_{i}.parquet")) for i in range(n_df_slices)])

        #         return df_query
        #     else:
        #         logging.warning("⚠️ No data frames were created.")
        #         return None
            
    def _write_parquet(self, df_query: pl.LazyFrame):
        
        write_start = time.time()
        
        try:
            logging.info("📝 Starting Parquet write")
            # Collect a small sample to check for issues
            # sample = df_query.limit(10).collect()
            # total_rows = df_query.select(pl.len()).collect().item()
            # logging.info(f"📊 Total rows in df_query: {total_rows}")
            # logging.info(f"🔢 Sample data types: {sample.dtypes}")
            # logging.info(f"🔍 Sample data:\n{sample}")
            
            # Ensure the directory exists
            self._ensure_dir_exists(self.save_path)

            # df_query.sink_ipc(self.save_path)
            df_query.sink_parquet(self.save_path, statistics=False)
            # df_query.sink_csv(self.save_path.replace(".arrow", ".csv"))

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
                result = self._read_single_netcdf(file_path)
            elif self.data_format == "csv":
                result = self._read_single_csv(file_path)
            else:
                raise ValueError(f"❌ Unsupported data format: {self.data_format}")
            
            logging.info(f"✅ Processed {file_number}-th {file_path}. Time: {time.time() - start_time:.2f} s")
            return result
        except Exception as e:
            logging.error(f"❌ Error processing {file_path}: {e}")
            return None

    # INFO: @Juan 10/16/24 Added method to read single netcdf file. Use pl.Series to convert the time variable to a polars series. and combined time extraction operations into a single line to remove intermediate variables. Removed try/except block as it is done in the calling method (_read_single_file())
    def _read_single_netcdf(self, file_path: str) -> pl.LazyFrame:
        with nc.Dataset(file_path, 'r') as dataset:
            # @Juan 10/14/24 Check if this is correct and if pandas can be substituted for polars
            col_mapping = dict((v, k) for k, v in self.column_mapping.items())
            time_var = dataset.variables[col_mapping["time"]]
            # time = pd_to_datetime(nc.num2date(times=time_var[:], 
            #                                   units=time_var.units, 
            #                                   calendar=time_var.calendar, 
            #                                   only_use_cftime_datetimes=False, 
            #                                   only_use_python_datetimes=True))
            time = nc.num2date(times=time_var[:], 
                               units=time_var.units, 
                               calendar=time_var.calendar, 
                               only_use_cftime_datetimes=False, 
                               only_use_python_datetimes=True)
            
            data = {
                'turbine_id': [os.path.basename(file_path).split('.')[-2]] * len(time),
                'time': time.tolist(),  # Convert to Polars datetime
                'turbine_status': dataset.variables[col_mapping["turbine_status"]][:],
                'wind_direction': dataset.variables[col_mapping["wind_direction"]][:],
                'wind_speed': dataset.variables[col_mapping["wind_speed"]][:],
                'power_output': dataset.variables[col_mapping["power_output"]][:],
                'nacelle_direction': dataset.variables[col_mapping["nacelle_direction"]][:]
            }

            # remove the rows with all nans (corresponding to rows where excluded columns would have had a value)
            # and bundle all values corresponding to identical time stamps together
            # forward fill missing values
            df_query = pl.LazyFrame(data).fill_nan(None)\
                                            .with_columns(pl.col("time").dt.round(f"{self.dt}s").alias("time"))\
                                            .select([cs.contains(feat) for feat in self.desired_feature_types])\
                                            .filter(pl.any_horizontal(cs.numeric().is_not_null()))
                                            # .group_by("turbine_id", "time")\
                                            # .agg(cs.numeric().drop_nulls().first())
                                                # .sort("turbine_id", "time")
            # with open(os.path.join(os.path.dirname(file_path), "ind_df_query_explan.txt"), "w") as f:
                # f.write(df_query.explain(streaming=True))
            
            # pivot table to have columns for each turbine and measurement if not originally in wide format
            if not self.wide_format:
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
            # with open(os.path.join(os.path.dirname(file_path), "ind_df_query_explan.txt"), "w") as f:
            #     f.write(df_query.explain(streaming=True))

            del data  # Free up memory

            # logging.info(f"Processed {file_path}")
            return df_query

    def _read_single_csv(self, file_path: str) -> pl.LazyFrame:
        try:
            df = pl.read_csv(file_path, low_memory=False)
            logging.info(f"Initial CSV columns: {df.columns}")
            logging.info(f"Initial CSV shape: {df.shape}")
            
            if self.column_mapping:
                df = df.rename(self.column_mapping)
                logging.info(f"Columns after mapping: {df.columns}")
            
            # INFO: @Juan 10/16/24 Select only the relevant columns based on self.features
            relevant_columns = ["time"] + [col for col in df.columns if any(feature in col for feature in self.features if feature != "time")]
            df = df.select(relevant_columns)
            logging.info(f"Columns after selecting relevant features: {df.columns}")
            logging.info(f"Shape after selecting relevant features: {df.shape}")
            
            if "time" in df.columns:
                df = df.with_columns(pl.col("time").str.to_datetime())
            else:
                logging.warning("⚠️ 'time' column not found in CSV file.")
            
            # Check if data is already in wide format
            is_already_wide = all(any(feature in col for col in df.columns) for feature in self.features if feature != "time")
            
            # INFO: @Juan 10/16/24 Added explicit check for wide_format to ensure consistent behavior
            # DEBUG: Make sure that this works with SMARTEOLE data.
            if is_already_wide:
                # Extract features based on the provided list
                feature_cols = [col for col in df.columns if any(feature in col for feature in self.features)]
                if "time" not in feature_cols:
                    feature_cols = ["time"] + feature_cols
                df = df.select(feature_cols)
                
                # Convert to long format if needed
                if not self.wide_format:
                    df = self._convert_to_long_format(df)
            else:
                df = df.select(self.features)
                if self.wide_format:
                    df = self.convert_to_wide_format(df)
            
            df = self.reduce_features(df)
            logging.info(f"Shape after reduce_features: {df.shape}")
            
            if self.dt is not None:
                df = self.resample(df)
            
            logging.info(f"📁 Processed {file_path}")
            return df.lazy()
        except Exception as e:
            logging.error(f"❌ Error processing CSV file {file_path}: {str(e)}")
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
        
        # Get unique turbine IDs
        # turbine_ids = df.select(pl.col("turbine_id").unique()).collect().to_series().to_list()
        
        # List of features to pivot (excluding 'time' and 'turbine_id')
        pivot_features = [col for col in df.columns if col not in ['time', 'turbine_id']]
        
        # Create expressions for pivot
        # pivot_exprs = [
        #     pl.col(feature).pivot(
        #         index="time",
        #         columns="turbine_id",
        #         aggregate_function="first",
        #         sort_columns=True
        #     ).prefix(f"{feature}_") for feature in pivot_features
        # ]
        
        # Pivot the data
        df_wide = df.pivot(
            index="time",
            columns="turbine_id",
            values=pivot_features,
            # aggregate_function="first",
            sort_columns=True
        )
        # .select([
        #     pl.col("time"),
        #     *pivot_exprs
        # ])
        
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
        existing_features = [f for f in self.features if any(f in col for col in df.columns)]
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
if __name__ == "__main__":
    from sys import platform
    RELOAD_DATA = True
    PLOT = False

    DT = 5
    DATA_FORMAT = "netcdf"

    if platform == "darwin" and DATA_FORMAT == "netcdf":
        DATA_DIR = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data"
        # PL_SAVE_PATH = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data/kp.turbine.zo2.b0.raw.parquet"
        # FILE_SIGNATURE = "kp.turbine.z02.b0.*.*.*.nc"
        PL_SAVE_PATH = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data/kp.turbine.zo2.b0.raw.parquet"
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
        # PL_SAVE_PATH = "/projects/ssc/ahenry/wind_forecasting/awaken_data/kp.turbine.zo2.b0.raw.parquet"
        PL_SAVE_PATH = os.path.join("/tmp/scratch", os.environ["SLURM_JOB_ID"], "kp.turbine.zo2.b0.parquet")
        # print(f"PL_SAVE_PATH = {PL_SAVE_PATH}")
        FILE_SIGNATURE = "kp.turbine.z02.b0.202203*1.*.*.nc"
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
        # FILE_SIGNATURE = "kp.turbine.z02.b0.20220301.*.*.nc"
        # FILE_SIGNATURE = "kp.turbine.z02.b0.*.*.*.nc"
        # FILE_SIGNATURE = "kp.turbine.z02.b0.20220101.000000.wt001.nc"
        FILE_SIGNATURE = "SMARTEOLE_WakeSteering_SCADA_1minData.csv"
        MULTIPROCESSOR = "cf" # mpi for HPC or "cf" for local computing
        # TURBINE_INPUT_FILEPATH = "/projects/$USER/toolboxes/wind-forecasting/examples/inputs/ge_282_127.yaml"
        # FARM_INPUT_FILEPATH = "/projects/$USER/toolboxes/wind-forecasting/examples/inputs/gch_KP_v4.yaml"
        TURBINE_INPUT_FILEPATH = "examples/inputs/ge_282_127.yaml"
        FARM_INPUT_FILEPATH = "examples/inputs/gch_KP_v4.yaml"
        FEATURES = ["time", "active_power", "wind_speed", "nacelle_position", "wind_direction", "derate"]
        WIDE_FORMAT = True
        
        COLUMN_MAPPING = {
            **{"time": "time"},
            **{f"active_power_{i}_avg": f"active_power_{i:03d}" for i in range(1, 8)},
            **{f"wind_speed_{i}_avg": f"wind_speed_{i:03d}" for i in range(1, 8)},
            **{f"nacelle_position_{i}_avg": f"nacelle_position_{i:03d}" for i in range(1, 8)},
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
            # Note that the order of the columns in the provided schema must match the order of the columns in the CSV being read.
            schema = pl.Schema(dict(sorted(({**{"time": pl.Datetime(time_unit="ms")},
                        **{
                            f"{feat}_{tid}": pl.Float64
                            for feat in FEATURES if feat != "time"
                            for tid in [f"wt{d+1:03d}" for d in range(88)]}
                        }).items())))
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
    
    df_query = data_loader.read_multi_files()

    if df_query:
        df_query = data_loader.postprocess_multi_files(df_query)
    elif RUN_ONCE:
        logging.warning("⚠️ No data frames were created.")

    if RUN_ONCE:
        
        # logging.info(f"⏱️ Total time elapsed: {time.time() - start_time:.2f} s")
    
        if df_query is not None:
            # Perform any additional operations on df_query if needed
            logging.info("✅ Data processing completed successfully")
        else:
            logging.warning("⚠️  No data was processed")
        
        logging.info("🎉 Script completed successfully")
