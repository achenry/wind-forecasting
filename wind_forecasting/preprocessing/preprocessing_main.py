# %%
# ! module load mambaforge or mamba
# ! mamba create -n wind_forecasting_env python=3.12
# ! mamba activate wind_forecasting_env
# ! conda install -c conda-forge jupyterlab mpi4py impi_rt
# git clone https://github.com/achenry/wind-forecasting.git
# git checkout feature/nacelle_calibration
# git submodule update --init --recursive
# ! pip install ./OpenOA # have to change pyproject.toml to allow for python 3.12.7
# ! pip install floris polars windrose netCDF4 statsmodels h5pyd seaborn pyarrow memory_profiler scikit-learn
# ! python -m ipykernel install --user --name=wind_forecasting_env
# ./run_jupyter_preprocessing.sh && http://localhost:7878/lab

import os
import sys
import logging
import argparse
import yaml
import time
import re
from memory_profiler import profile
from shutil import rmtree
import pickle

mpi_exists = False
try:
    from mpi4py import MPI
    mpi_exists = True
except:
    print("No MPI available on system.")

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, parent_dir)

# print(f"Current working directory inside script: {os.getcwd()}")
# print("sys.path inside script:", sys.path)

import multiprocessing

from wind_forecasting.preprocessing.data_loader import DataLoader
from wind_forecasting.preprocessing.data_filter import (DataFilter, 
                                                        add_df_continuity_columns, add_df_agg_continuity_columns, 
                                                        get_continuity_group_index, group_df_by_continuity, 
                                                        merge_adjacent_periods, compute_offsets, safe_mask)
from wind_forecasting.preprocessing.data_inspector import DataInspector
from openoa.utils import plot, filters, power_curve

import polars as pl
import polars.selectors as cs
import numpy as np
import matplotlib
# matplotlib.use('Agg') # Use TkAgg for interactive plots
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from scipy.stats import norm
from floris import FlorisModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ROW_LIMIT = 2 * 60 * 60 * 24 * 30 * 18
ROW_LIMIT = 2 * 24 * 30 * 18

# %%
# @profile
def main():
    if MPI.COMM_WORLD.Get_rank() == 0: 
     logging.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog="WindFarmForecasting")
    parser.add_argument("-cnf", "--config", type=str)
    parser.add_argument("-m", "--multiprocessor", type=str, choices=["cf", "mpi"], required=False, default=None)
    parser.add_argument("-rf", "--regenerate_filters", action="store_true")
    parser.add_argument("-ld", "--reload_data", action="store_true")
    parser.add_argument("-pd", "--preprocess_data", action="store_true")
    parser.add_argument("-p", "--plot", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()
     
    RUN_ONCE = (args.multiprocessor == "mpi" and mpi_exists and (MPI.COMM_WORLD.Get_rank()) == 0) or (args.multiprocessor != "mpi") or (args.multiprocessor is None)
    args.verbose = args.verbose and RUN_ONCE
    args.plot = args.plot and RUN_ONCE
    
    with open(args.config, 'r') as file:
        config  = yaml.safe_load(file)

    config["raw_data_directory"] = [os.path.expanduser(fp) for fp in config["raw_data_directory"]]
    config["processed_data_path"] = os.path.expanduser(config["processed_data_path"])
    config["turbine_input_path"] = os.path.expanduser(config["turbine_input_path"]) 
    config["farm_input_path"] = os.path.expanduser(config["farm_input_path"])
    
    if RUN_ONCE:
        for path_key in ["raw_data_directory", "turbine_input_path", "farm_input_path"]:
            if isinstance(config[path_key], list):
                assert all(os.path.exists(fp) for fp in config[path_key]), f"One of {config[path_key]} doesn't exist."
            else:
                assert os.path.exists(config[path_key]), f"{config[path_key]} doesn't exist."
             
    for path_key in ["raw_data_directory", "processed_data_path", "turbine_input_path", "farm_input_path"]:
        if isinstance(config[path_key], list):
            env_vars = [re.findall(r"(?:^|\/)\$(\w+)(?:\/|$)", d) for d in config[path_key]]
            for file_set_idx in range(len(env_vars)):
                for env_var in env_vars[file_set_idx]:
                    config[path_key][file_set_idx] = config[path_key][file_set_idx].replace(f"${env_var}", os.environ[env_var])
        else:
            env_vars = re.findall(r"(?:^|\/)\$(\w+)(?:\/|$)", config[path_key])
            for env_var in env_vars:
                config[path_key] = config[path_key].replace(f"${env_var}", os.environ[env_var])
    
    # config["filters"] = ["nacelle_calibration", "unresponsive_sensor", "range_flag", "bin_filter", "std_range_flag", "impute_missing_data", "split", "normalize"]
    # config["filters"] = ["split", "impute_missing_data", "normalize"]
            #    ["unresponsive_sensor", "inoperational", "range_flag", "window_range_flag", "bin_filter", "std_range_flag", "split", "impute_missing_data", "normalize"]
    if RUN_ONCE:
        assert all(filt in 
                   ["nacelle_calibration", "unresponsive_sensor", "inoperational", "range_flag", "window_range_flag", "bin_filter", "std_range_flag", "split", "impute_missing_data", "normalize"] 
                   for filt in config["filters"])

    config["data_format"] = []
    for fp in config["raw_data_file_signature"]:
        if fp.endswith(".nc"):
            config["data_format"].append("netcdf")
        elif fp.endswith(".csv"):
            config["data_format"].append("csv")
        elif fp.endswith(".parquet"):
            config["data_format"].append("parquet")
        else:
            raise ValueError("Invalid file signature. Please specify either '*.nc', '*.csv', or '*.parquet'.")
    
    if "turbine_signature" not in config:
       config["turbine_signature"] = [None]
       
    if "datetime_signature" not in config:
       config["datetime_signature"] = [None]
       
    # if we are only processing one file type, there is no need to transform all turbine ids to a common list
    if "turbine_mapping" not in config or (len(config["raw_data_directory"]) == 1):
        config["turbine_mapping"] = None
    elif RUN_ONCE:
        assert all(isinstance(tm, dict) for tm in config["turbine_mapping"])
    
    if RUN_ONCE:
        logging.info("Parsed arguments successfully")
        logging.info("Instantiating DataLoader")
        
    data_loader = DataLoader(
        data_dir=config["raw_data_directory"],
        file_signature=config["raw_data_file_signature"],
        save_path=config["processed_data_path"],
        multiprocessor=args.multiprocessor,
        dt=config["dt"],
        ffill_limit=None,
        data_format=config["data_format"],
        feature_mapping=config["feature_mapping"],
        turbine_signature=config["turbine_signature"],
        turbine_mapping=config["turbine_mapping"],
        datetime_signature=config["datetime_signature"],
        merge_chunk=config["merge_chunk"],
        ram_limit=config["ram_limit"]
    )
    if RUN_ONCE:
        logging.info("Instantiated DataLoader successfully")

    # %%
    # INFO: Print netcdf structure
    if args.verbose and config["data_format"] == "netcdf":
        data_loader.print_netcdf_structure(data_loader.file_paths[0])

    # %%
    if RUN_ONCE:
        if not args.reload_data and os.path.exists(data_loader.save_path):
            # Note that the order of the columns in the provided schema must match the order of the columns in the CSV being read.
            logging.info("🔄 Loading existing Parquet file")
            df_query = pl.scan_parquet(source=data_loader.save_path)
            if data_loader.turbine_mapping is not None:
                # if this data was pulled from multiple files, the turbine ids have all been mapped to integers
                data_loader.turbine_signature = "\\d+$" 
            else:
                data_loader.turbine_signature = data_loader.turbine_signature[0]
            
            df_query = df_query.select([pl.col("time")] 
                                       + [pl.col(c) for c in 
                                          sorted(df_query.select(cs.numeric()).select(pl.exclude("file_set_idx")).collect_schema().names(), 
                                                 key=lambda col: (re.search(f".*?(?={data_loader.turbine_signature})", col).group(0), 
                                                                  int(re.search("\\d+", re.search(data_loader.turbine_signature, col).group(0)).group(0))))])
            # generate turbine ids
            data_loader.turbine_ids = data_loader.get_turbine_ids(data_loader.turbine_signature, df_query, sort=True)

        else:
            if args.multiprocessor == "mpi" and mpi_exists:
                comm_size = MPI.COMM_WORLD.Get_size()
                logging.info(f"🚀 Using MPI executor with {comm_size} processes.")
            elif args.multiprocessor == "cf":
                max_workers = multiprocessing.cpu_count()
                logging.info("🚀  Using ProcessPoolExecutor with %d workers.", max_workers)
            else:
                logging.info("🚀  Using single process executor.")

            logging.info("🔄 Processing new data files with %d files", sum(len(fp) for fp in data_loader.file_paths))
            start_time = time.time()
    
    if args.reload_data or not os.path.exists(data_loader.save_path):
        temp_save_dir = os.path.join(os.path.dirname(data_loader.save_path), os.path.basename(data_loader.save_path).replace(".parquet", "_temp"))
        
        # if os.path.exists(temp_save_dir):
        #     rmtree(temp_save_dir)
            # raise Exception(f"Temporary saving directory {temp_save_dir} already exists! Please remove or rename it.")
        if RUN_ONCE:
            if not os.path.exists(temp_save_dir):
                logging.info(f"Making temporary directory {temp_save_dir}")
                os.makedirs(temp_save_dir)
    
            if not os.path.exists(os.path.dirname(data_loader.save_path)):
                logging.info(f"Making directory to save_path {os.path.dirname(data_loader.save_path)}")
                os.makedirs(os.path.dirname(data_loader.save_path))
                        
        df_query = data_loader.read_multi_files(temp_save_dir, read_single_files=True, first_merge=True, second_merge=True) 
        
        if RUN_ONCE:
            logging.info("✅ Finished reading individual files. Time elapsed: %.2f s", time.time() - start_time)
            logging.info("Parquet file saved into %s", data_loader.save_path)
            
            if os.path.exists(temp_save_dir):
                logging.info(f"Removing temporary storage directory {temp_save_dir}")
                rmtree(temp_save_dir)
                
    elif RUN_ONCE:
        logging.info("✅ Loaded existing Parquet file.")
        
    if not args.preprocess_data:
        return
    
    if RUN_ONCE:
        assert all(any(prefix in col for col in df_query.collect_schema().names()) for prefix in ["time", "wind_speed_", "wind_direction_", "nacelle_direction_", "power_output_"]), "DataFrame must contain columns 'time', then columns with prefixes 'wind_speed_', 'wind_direction_', 'power_output_', 'nacelle_direction_'"
        assert df_query.select("time").collect().to_series().is_sorted(), "Loaded data should be sorted by time!"
        assert all(any(f"{prefix}{tid}" in col for col in df_query.collect_schema().names() if col != "time") for prefix in ["wind_speed_", "wind_direction_", "nacelle_direction_", "power_output_"] for tid in data_loader.turbine_ids), "DataFrame must contain columns with prefixes 'wind_speed_', 'wind_direction_', 'power_output_', 'nacelle_direction_' and suffixes for each turbine id" 

    # df_query = df_query.group_by("time").agg(cs.numeric().mean())
    # df_query.collect().write_parquet(config["processed_data_path"], statistics=False)
    
    if RUN_ONCE:
        data_inspector = DataInspector(
            turbine_input_filepath=config["turbine_input_path"],
            farm_input_filepath=config["farm_input_path"],
            turbine_signature=data_loader.turbine_signature,
            data_format='auto',
            save_dir=os.path.dirname(config["processed_data_path"])
        )
        
        waked_distances = np.array([((data_inspector.fmodel.layout_x[pair[0]] - data_inspector.fmodel.layout_x[pair[1]])**2 + (data_inspector.fmodel.layout_y[pair[0]] - data_inspector.fmodel.layout_y[pair[1]])**2)**0.5 for pair in config["nacelle_calibration_turbine_pairs"]])
    
    if args.debug:
        # .group_by("time", "file_set_idx")\
        df_query = df_query.slice(0, int(3 * 30 * np.timedelta64(1, 'D') / np.timedelta64(data_loader.dt, 's')))\
                           .with_columns(pl.col("time").dt.round(f"{1}m").alias("time"))\
                           .group_by("time")\
                            .agg(cs.numeric().mean()).sort("time")
    
    # %% Plot Wind Farm, Data Distributions
    # df_query.select("time", "wind_direction_1").filter((pl.col("time") > datetime(2020, 5, 24, 4, 30)) & (pl.col("time") < datetime(2020, 5, 24, 6, 30))).collect().to_numpy()[:, 1].flatten() 
    if args.plot:
        # from datetime import datetime
        # df_query = df_query.with_columns(file_set_idx=pl.when(pl.col("time") < pl.lit(datetime(2024,2,20))).then(0).otherwise(1))
        
        logging.info("🔄 Generating plots.")
        # x = pl.concat([df.slice(0, ROW_LIMIT) for df in df_query.collect().partition_by("file_set_idx")], how="vertical").lazy()
        # data_inspector.plot_wind_farm()
        if "file_set_idx" in df_query.collect_schema().names():
            file_set_indices = df_query.select("file_set_idx").unique().collect().to_numpy().flatten()
            df_query2 = df_query.with_columns(pl.col("time").dt.round(f"{1}m").alias("time"))\
                        .group_by("time", "file_set_idx").agg(cs.numeric().mean()).sort("time")\
                        .filter(pl.all_horizontal((cs.starts_with("wind_speed") >= 3) & (cs.starts_with("wind_speed") <= 25)))
        
            
            # data_inspector.plot_wind_rose(df_query2, feature_type="wind_direction", turbine_ids="all", fig_label=f"wind_rose_awaken")
            data_inspector.plot_wind_speed_weibull(df_query2.filter(pl.col("file_set_idx") == 1), turbine_ids="all", fig_label=file_set_idx) 
            
            for file_set_idx in file_set_indices:
                data_inspector.plot_wind_rose(df_query2.filter(pl.col("file_set_idx") == file_set_idx).slice(0, ROW_LIMIT), 
                                            feature_type="wind_direction", turbine_ids="all", fig_label=f"wind_rose_{file_set_idx}")
                # data_inspector.plot_wind_rose(df_query2.filter(pl.col("file_set_idx") == file_set_idx).slice(0, ROW_LIMIT), 
                #                               feature_type="nacelle_direction", turbine_ids="all", fig_label=f"nacelle_rose_{file_set_idx}")
            
            for file_set_idx in file_set_indices:
                data_inspector.plot_wind_speed_power(df_query2.filter(pl.col("file_set_idx") == file_set_idx).slice(0, ROW_LIMIT), turbine_ids=data_loader.turbine_ids, fig_label=file_set_idx)
            
            for file_set_idx in file_set_indices:
                data_inspector.plot_wind_speed_weibull(df_query2.filter(pl.col("file_set_idx") == file_set_idx).slice(0, ROW_LIMIT), turbine_ids="all", fig_label=file_set_idx)
                
            for file_set_idx in file_set_indices:
                fig, _ = plot.column_histograms(data_inspector.collect_data(
                    df=df_query2.filter(pl.col("file_set_idx") == file_set_idx).slice(0, ROW_LIMIT), feature_types=["wind_speed"]), 
                                    return_fig=True)
                fig.savefig(os.path.join(data_inspector.save_dir, f"wind_speed_histogram_{file_set_idx}.png"))
                fig, _ = plot.column_histograms(data_inspector.collect_data(
                    df=df_query2.filter(pl.col("file_set_idx") == file_set_idx).slice(0, ROW_LIMIT), feature_types=["wind_direction"]), 
                                    return_fig=True)
                fig.savefig(os.path.join(data_inspector.save_dir, f"wind_dir_histogram_{file_set_idx}.png"))
        else:
            df_query2 = df_query.with_columns(pl.col("time").dt.round(f"{1}m").alias("time"))\
                        .group_by("time").agg(cs.numeric().mean()).sort("time")\
                        .filter(pl.all_horizontal((cs.starts_with("wind_speed") >= 3) & (cs.starts_with("wind_speed") <= 25)))
            data_inspector.plot_wind_rose(df_query2.slice(0, ROW_LIMIT), 
                                feature_type="wind_direction", turbine_ids="all", fig_label=f"wind_rose_{file_set_idx}")
            data_inspector.plot_wind_rose(df_query2.slice(0, ROW_LIMIT), 
                                feature_type="nacelle_direction", turbine_ids="all", fig_label=f"wind_rose_{file_set_idx}")
                        
        # data_inspector.plot_correlation(df_query.slice(0, ROW_LIMIT), 
        # data_inspector.get_features(df_query.slice(0, ROW_LIMIT), feature_types=["wind_speed", "wind_direction", "nacelle_direction"], 
        #                             turbine_ids=data_loader.turbine_ids))
        # data_inspector.plot_boxplot_wind_speed_direction(df_query.slice(0, ROW_LIMIT), 
        #                                                  turbine_ids=data_loader.turbine_ids)
        # data_inspector.plot_time_series(df_query.slice(0, ROW_LIMIT), 
        #                                 turbine_ids=data_loader.turbine_ids)

        logging.info("✅ Generated plots.")

    # %% check time series
    if args.verbose:
        DataInspector.print_df_state(df_query, ["wind_speed", "wind_direction", "nacelle_direction"])
    if args.plot:
        data_inspector.plot_time_series(df_query.slice(0, ROW_LIMIT), feature_types=["wind_speed", "wind_direction"], turbine_ids=data_loader.turbine_ids, continuity_groups=None, label="original")
    
    # %% OpenOA Data Preparation & Inspection

    # %%
    ws_cols = data_inspector.get_features(df_query, "wind_speed")
    wd_cols = data_inspector.get_features(df_query, "wind_direction")
    pwr_cols = data_inspector.get_features(df_query, "power_output")
    
    # Create a mapping from turbine ID to its index
    turbine_id_to_index = {tid: idx for idx, tid in enumerate(data_loader.turbine_ids)}

    # %%
    data_filter = DataFilter(turbine_signature=data_loader.turbine_signature, turbine_availability_col=None, 
                             turbine_status_col="turbine_status", multiprocessor=args.multiprocessor, data_format='wide')

    # if args.regenerate_filters or args.reload_data or not os.path.exists(config["processed_data_path"].replace(".parquet", "_filtered.parquet")):
    # %% # first filter because need to catch frozen measurements before others are nulled from repeated value.
    # applied_filter = False
    if "unresponsive_sensor" in config["filters"]:
        if RUN_ONCE:
            logging.info("Nullifying unresponsive sensor cells.")
        
            # find stuck sensor measurements for each turbine and set them to null
            # NOTE: this filter must be applied before any cells are nullified st null values aren't considered repeated values
            # find values of wind speed/direction, where there are duplicate values with nulls inbetween
            if args.reload_data or args.regenerate_filters \
                or not all(os.path.exists(config["processed_data_path"].replace(".parquet", f"_frozen_sensors_{feat}.npy")) for feat in ws_cols + wd_cols):
                    
                thr = int(np.timedelta64(config["filters"]["unresponsive_sensor"]["frozen_sensor_limit"], 's') / np.timedelta64(data_loader.dt, 's'))
                frozen_sensors = filters.unresponsive_flag(
                    data_pl=df_query.select(cs.starts_with("wind_speed"), cs.starts_with("wind_direction")), 
                    threshold=thr)
                mask = lambda feat: frozen_sensors(feat).collect().to_numpy().flatten()
                
                for feat in ws_cols + wd_cols:
                    np.save(config["processed_data_path"].replace(".parquet", f"_frozen_sensors_{feat}.npy"), 
                                frozen_sensors(feat).collect().to_numpy().flatten())
                    
            else:
                mask = lambda feat: np.load(config["processed_data_path"].replace(".parquet", f"_frozen_sensors_{feat}.npy")) 
        
            # check time series
            if args.verbose:
                DataInspector.print_pc_remaining_vals(df_query, mask,
                                                        mask_input_features=ws_cols+wd_cols,
                                                        output_features=ws_cols+wd_cols,
                                                        filter_type="unresponsive sensor")
                
            if args.plot:
                
                # for feature_type, features in zip(["wind_speed", "wind_direction"], [ws_cols, wd_cols]):
                #     flag = np.concatenate([mask(feat).select(pl.all().slice(0, ROW_LIMIT)).collect().to_numpy().flatten() for feat in features])
                #     plot.plot_power_curve(
                #         data_inspector.collect_data(df=df_query.head(ROW_LIMIT), feature_types="wind_speed").values.flatten(),
                #         data_inspector.collect_data(df=df_query.head(ROW_LIMIT), feature_types="power_output").values.flatten(),
                #         flag=flag,
                #         flag_labels=(f"{feature_type} Unresponsive Sensors (n={flag.sum():,.0f})", "Normal Turbine Operations"),
                #         xlim=(-1, 15),  # optional input for refining plots
                #         ylim=(-100, 3000),  # optional input for refining plots
                #         legend=True,  # optional flag for adding a legend
                #         scatter_kwargs=dict(alpha=0.4, s=10)  # optional input for refining plots
                #     )
                
                
                data_inspector.plot_nulled_vs_remaining(df_query.slice(0, ROW_LIMIT), mask,
                                                        mask_input_features=ws_cols,
                                                        output_features=ws_cols, 
                                                        feature_types=["wind_speed"], 
                                                        feature_labels=["Wind Speed (m/s) after Unresponsive Sensor Filter"])
                data_inspector.plot_nulled_vs_remaining(df_query.slice(0, ROW_LIMIT), mask,
                                                        mask_input_features=wd_cols,
                                                        output_features=wd_cols, 
                                                        feature_types=["wind_direction"], 
                                                        feature_labels=["Wind Direction ($^\\circ$) after Unresponsive Sensor Filter"])

            # change the values corresponding to frozen sensor measurements to null or interpolate (instead of dropping full row, since other sensors could be functioning properly)
            # fill stuck sensor measurements with Null st they are marked for interpolation later,
        
            threshold = 0.01
            logging.info("Nullifying wind speed/direction frozen sensor measurements in dataframe.")
            df_query = data_filter.conditional_filter(df_query, threshold, mask, 
                                                        mask_input_features=ws_cols+wd_cols,
                                                        output_features=ws_cols+wd_cols, 
                                                        filter_type="unresponsive sensor",
                                                        check_js=False)
            # df_query.select(pl.col("time"), cs.starts_with("wind_speed")).filter(frozen_sensors["wind_speed"].all(axis=1)).collect()
            del frozen_sensors, mask
            df_query.collect().write_parquet(config["processed_data_path"].replace(".parquet", "_filtered.parquet"), statistics=False)
            df_query = pl.scan_parquet(config["processed_data_path"].replace(".parquet", "_filtered.parquet"))
            logging.info("Finished nullifying wind speed/direction frozen sensor measurements in dataframe.")
            
            # check time series
            if args.verbose:
                DataInspector.print_df_state(df_query, ["wind_speed", "wind_direction", "nacelle_direction"])
                
            if args.plot:
                data_inspector.plot_time_series(df_query.slice(0, ROW_LIMIT), feature_types=["wind_speed", "wind_direction"], turbine_ids=data_loader.turbine_ids, continuity_groups=None, label="after_frozen_sensor") 

    # %%
    if "inoperational" in config["filters"] and any(col.startswith("turbine_status") for col in df_query.collect_schema()["names"]): # 10 is normal operation for AWAKEN
        if RUN_ONCE:
            logging.info("Nullifying inoperational turbine cells.")
            
            # check if wind speed/dir measurements from inoperational turbines differ from fully operational
            status_codes = [1]
            mask = lambda tid: ~pl.col(f"turbine_status_{tid}").is_in(status_codes) & pl.col(f"turbine_status_{tid}").is_not_null()

            # check time series
            if args.verbose:
                DataInspector.print_pc_remaining_vals(df_query, mask,
                                                        mask_input_features=sorted(list(data_loader.turbine_ids)) * 2,
                                                        output_features=ws_cols+wd_cols,
                                                        filter_type="inoperational turbine status")
            if args.plot:
                data_inspector.plot_nulled_vs_remaining(df_query.slice(0, ROW_LIMIT), mask, 
                                                        mask_input_features=sorted(list(data_loader.turbine_ids)) * 2,
                                                        output_features=ws_cols+wd_cols, 
                                                        feature_types=["wind_speed", "wind_direction"], 
                                                    feature_labels=["Wind Speed (m/s)", "Wind Direction ($^\\circ$)"])
        
            # loop through each turbine's wind speed and wind direction columns, and compare the distribution of data with and without the inoperational turbines
            # fill out_of_range measurements with Null st they are marked for interpolation via impute or linear/forward fill interpolation later
            threshold = 0.01
            logging.info("Nullifying inoperational turbine measurements in dataframe.")
            # turbine_status_cols = data_inspector.get_features(df_query, "turbine_status")
            df_query = data_filter.conditional_filter(df_query, threshold, mask, 
                                                        mask_input_features=sorted(list(data_loader.turbine_ids))*2, 
                                                        output_features=ws_cols+wd_cols, 
                                                        filter_type="inoperational turbine status", check_js=False)
            del mask
            df_query.collect().write_parquet(config["processed_data_path"].replace(".parquet", "_filtered.parquet"), statistics=False)
            df_query = pl.scan_parquet(config["processed_data_path"].replace(".parquet", "_filtered.parquet"))
            logging.info("Finished nullifying inoperational turbine measurements in dataframe.") 
        
            # check time series
            if args.verbose:
                DataInspector.print_df_state(df_query, ["wind_speed", "wind_direction", "nacelle_direction"])
            if args.plot:
                data_inspector.plot_time_series(df_query.slice(0, ROW_LIMIT), feature_types=["wind_speed", "wind_direction"], turbine_ids=data_loader.turbine_ids, continuity_groups=None, label="after_inoperational")

    # %%
    if "range_flag" in config["filters"]:
        if RUN_ONCE:
            logging.info("Nullifying wind speed out-of-range cells.")
            
            # check for wind speed values that are outside of the acceptable range
            if args.reload_data or args.regenerate_filters or not os.path.exists(config["processed_data_path"].replace(".parquet", "_out_of_range.npy")):
                # Generate out_of_range array
                # Note: OpenOA's range_flag returns True for out-of-range values
                ws = df_query.select(cs.starts_with("wind_speed")).collect().to_pandas()
                out_of_range = (filters.range_flag(ws, 
                                                    lower=config["filters"]["range_flag"]["lower"],
                                                    upper=config["filters"]["range_flag"]["upper"]) & ~ws.isna()).values # range flag includes formerly null values as nan
                del ws
                np.save(config["processed_data_path"].replace(".parquet", "_out_of_range.npy"), out_of_range)
            else:
                out_of_range = np.load(config["processed_data_path"].replace(".parquet", "_out_of_range.npy"))

            # check if wind speed/dir measurements from inoperational turbines differ from fully operational 
            mask = lambda tid: safe_mask(tid, outlier_flag=out_of_range, turbine_id_to_index=turbine_id_to_index)

            # check time series
            if args.verbose:
                DataInspector.print_pc_remaining_vals(df_query, mask, 
                                                        mask_input_features=sorted(data_loader.turbine_ids),
                                                        output_features=ws_cols,
                                                        filter_type="wind speed range")
            if args.plot:
                data_inspector.plot_nulled_vs_remaining(df_query.slice(0, ROW_LIMIT), mask, 
                                                        mask_input_features=sorted(data_loader.turbine_ids),
                                                        output_features=ws_cols, 
                                                        feature_types=["wind_speed"], 
                                                        feature_labels=["Wind Speed (m/s) after Out-of-Range Filter"])
                
                # plot values outside the win speed range 
                target_turbine_idx = np.argsort(out_of_range.sum(axis=0))[-1]
                # target_turbine_idx = np.arange(len(data_loader.turbine_ids))
                fig, axs = plot.plot_power_curve(
                    data_inspector.collect_data(df=df_query, feature_types="wind_speed").to_numpy()[:, target_turbine_idx].flatten(),
                    data_inspector.collect_data(df=df_query, feature_types="power_output").to_numpy()[:, target_turbine_idx].flatten(),
                    flag=out_of_range[:, target_turbine_idx].flatten(),
                    flag_labels=("Anomylous Measurements", "Normal Measurements"),
                    xlim=(-1, 30),
                    ylim=(-100, 3000),
                    legend=True,
                    scatter_kwargs=dict(alpha=0.4, s=10),
                    figure_kwargs=dict(figsize=(10, 6)),
                    return_fig=True
                )
                axs.tick_params(axis="x", labelsize=12*1.5)
                axs.tick_params(axis="y", labelsize=12*1.5)
                axs.xaxis.label.set_size(15*1.5)
                axs.yaxis.label.set_size(15*1.5)
                for t in axs.legend_.get_texts():
                    t.set_size(12*1.5)
                
                plt.tight_layout()
                fig.savefig(os.path.join(data_inspector.save_dir, "wind_speed_out_of_range_awaken.png"), dpi=100)

            # loop through each turbine's wind speed and wind direction columns, and compare the distribution of data with and without the inoperational turbines
            # fill out_of_range measurements with Null st they are marked for interpolation via impute or linear/forward fill interpolation later
            threshold = 0.01
            logging.info("Nullifying wind speed out of range measurements in dataframe.")
            df_query = data_filter.conditional_filter(df_query, threshold, mask, 
                                                        mask_input_features=sorted(data_loader.turbine_ids), 
                                                        output_features=ws_cols, 
                                                        filter_type="wind speed range", check_js=False)
            del out_of_range, mask
            
            df_query.collect().write_parquet(config["processed_data_path"].replace(".parquet", "_filtered.parquet"), statistics=False)
            df_query = pl.scan_parquet(config["processed_data_path"].replace(".parquet", "_filtered.parquet"))
            logging.info("Finished nullifying wind speed out of range measurements in dataframe.") 
        
            if args.verbose:
                DataInspector.print_df_state(df_query, ["wind_speed", "wind_direction", "nacelle_direction"])
                
            if args.plot:
                data_inspector.plot_time_series(df_query.slice(0, ROW_LIMIT), feature_types=["wind_speed", "wind_direction"], turbine_ids=data_loader.turbine_ids, continuity_groups=None, label="after_out_of_range")
    
    # %%
    if "window_range_flag" in config["filters"]:
        if RUN_ONCE:
            logging.info("Nullifying wind speed-power curve out-of-window cells.")
        
        # apply a window range filter to remove data with power values outside of the window from 20 to 3000 kW for wind speeds between 5 and 40 m/s.
        # identifies when turbine is shut down, filtering for normal turbine operation
        print(534)
        if args.reload_data or args.regenerate_filters or not os.path.exists(config["processed_data_path"].replace(".parquet", "_out_of_window.npy")):
            if RUN_ONCE:
                # data_filter.multiprocessor = None
                out_of_window = data_filter.multi_generate_filter(df_query=df_query, filter_func=data_filter._single_generate_window_range_filter,
                                                                    feature_types=["wind_speed", "power_output"], turbine_ids=data_loader.turbine_ids,
                                                                    window_start=config["filters"]["window_range_flag"]["window_start"], 
                                                                    window_end=config["filters"]["window_range_flag"]["window_end"], 
                                                                    value_min=config["filters"]["window_range_flag"]["value_min"] * data_inspector.rated_turbine_power, 
                                                                    value_max=config["filters"]["window_range_flag"]["value_max"] * data_inspector.rated_turbine_power)
                # data_filter.multiprocessor = args.multiprocessor
            
                np.save(config["processed_data_path"].replace(".parquet", "_out_of_window.npy"), out_of_window)
                
        elif RUN_ONCE:
                out_of_window = np.load(config["processed_data_path"].replace(".parquet", "_out_of_window.npy"))
        
        if RUN_ONCE:
            print(547)
            # check if wind speed/dir measurements from inoperational turbines differ from fully operational 
            mask = lambda tid: safe_mask(tid, outlier_flag=out_of_window, turbine_id_to_index=turbine_id_to_index)
            print(550)
            
            if args.verbose:
                DataInspector.print_pc_remaining_vals(df_query, mask,
                                                        mask_input_features=sorted(data_loader.turbine_ids),
                                                        output_features=ws_cols,
                                                        filter_type="power-wind speed window range")
            
            print(557)
            
            if args.plot:
                data_inspector.plot_nulled_vs_remaining(df_query.slice(0, ROW_LIMIT), mask,
                                                        mask_input_features=sorted(data_loader.turbine_ids),
                                                        output_features=ws_cols, 
                                                        feature_types=["wind_speed"], 
                                                        feature_labels=["Wind Speed (m/s) after Wind Speed-Power Out-of-Window Filter"])

                # plot values outside the win speed range 
                target_turbine_idx = np.argsort(out_of_window.sum(axis=0))[-1]
                fig, axs = plot.plot_power_curve(
                    data_inspector.collect_data(df=df_query, feature_types="wind_speed").to_numpy()[:, target_turbine_idx].flatten(),
                    data_inspector.collect_data(df=df_query, feature_types="power_output").to_numpy()[:, target_turbine_idx].flatten(),
                    flag=out_of_window[:, target_turbine_idx].flatten(),
                    flag_labels=("Anomylous Measurements", "Normal Measurements"),
                    xlim=(-1, 30),
                    ylim=(-100, 3000),
                    legend=True,
                    scatter_kwargs=dict(alpha=0.4, s=10),
                    figure_kwargs=dict(figsize=(10, 6)),
                    return_fig=True
                )
                axs.tick_params(axis="x", labelsize=12*1.5)
                axs.tick_params(axis="y", labelsize=12*1.5)
                axs.xaxis.label.set_size(15*1.5)
                axs.yaxis.label.set_size(15*1.5)
                for t in axs.legend_.get_texts():
                    t.set_size(12*1.5)
                
                plt.tight_layout()
                fig.savefig(os.path.join(data_inspector.save_dir, "power_curve_out_of_window_range.png"), dpi=100)

            print(589)
            
            # fill cells corresponding to values that are outside of power-wind speed window range with Null st they are marked for interpolation via impute or linear/forward fill interpolation later
            # loop through each turbine's wind speed and wind direction columns, and compare the distribution of data with and without the inoperational turbines
            threshold = 0.01
            logging.info("Nullifying wind speed-power curve out-of-window measurements in dataframe.")
            df_query = data_filter.conditional_filter(df_query, threshold, mask, 
                                                        mask_input_features=sorted(data_loader.turbine_ids),
                                                        output_features=ws_cols, 
                                                        filter_type="power-wind speed window range", check_js=False)
            print(598)
            del out_of_window, mask
            # need to sink parquet and recollect to avoid recursion limit error
            df_query.collect().write_parquet(config["processed_data_path"].replace(".parquet", "_filtered.parquet"), statistics=False)
            print(601)
            df_query = pl.scan_parquet(config["processed_data_path"].replace(".parquet", "_filtered.parquet"))
            print(603)
            logging.info("Finished nullifying wind speed-power curve out-of-window measurements in dataframe.") 
            
            if args.verbose:
                DataInspector.print_df_state(df_query, ["wind_speed", "wind_direction", "nacelle_direction"])
                
            if args.plot:
                data_inspector.plot_time_series(df_query.slice(0, ROW_LIMIT), feature_types=["wind_speed", "wind_direction"], 
                                                turbine_ids=data_loader.turbine_ids, continuity_groups=None, label="after_out_of_window")
        
    if "bin_filter" in config["filters"]:
        if RUN_ONCE:
            logging.info("Nullifying wind speed-power curve bin-outlier cells.")
        
        # apply a bin filter to remove data with power values outside of an envelope around median power curve at each wind speed
        if args.reload_data or args.regenerate_filters or not os.path.exists(config["processed_data_path"].replace(".parquet", "_bin_outliers.npy")):
            # data_filter.multiprocessor = None
            
            # df_query.select(pl.max_horizontal(cs.starts_with(f"power_output").max())).collect().item()
            bin_outliers = data_filter.multi_generate_filter(df_query=df_query, filter_func=data_filter._single_generate_bin_filter,
                                                                feature_types=["wind_speed", "power_output"], turbine_ids=data_loader.turbine_ids,
                                                                bin_width=config["filters"]["bin_filter"]["bin_width"], 
                                                                threshold=config["filters"]["bin_filter"]["threshold"],
                                                                center_type=config["filters"]["bin_filter"]["center_type"],
                                                                bin_min=config["filters"]["bin_filter"]["bin_min"] * data_inspector.rated_turbine_power,
                                                                bin_max=config["filters"]["bin_filter"]["bin_max"] * data_inspector.rated_turbine_power,
                                                                threshold_type=config["filters"]["bin_filter"]["threshold_type"],
                                                                direction="below"# keep derated cases
                                                                ) 
            # data_filter.multiprocessor = args.multiprocessor
            if RUN_ONCE:
                np.save(config["processed_data_path"].replace(".parquet", "_bin_outliers.npy"), bin_outliers)
        elif RUN_ONCE:
            bin_outliers = np.load(config["processed_data_path"].replace(".parquet", "_bin_outliers.npy"))

        if RUN_ONCE:
            # check if wind speed/dir measurements from inoperational turbines differ from fully operational 
            mask = lambda tid: safe_mask(tid, outlier_flag=bin_outliers, turbine_id_to_index=turbine_id_to_index)
        
            # check time series
            if args.verbose:
                DataInspector.print_pc_remaining_vals(df_query, mask, 
                                                        mask_input_features=sorted(data_loader.turbine_ids),
                                                        output_features=ws_cols,
                                                        filter_type="power-wind speed bin")
            
            if args.plot:
                data_inspector.plot_nulled_vs_remaining(df_query.slice(0, ROW_LIMIT), mask, 
                                                        mask_input_features=sorted(data_loader.turbine_ids),
                                                        output_features=ws_cols, 
                                                        feature_types=["wind_speed"], 
                                                        feature_labels=["Wind Speed (m/s) after Wind Speed-Power Bin Outlier Filter"])

                # plot values outside the power-wind speed bin filter
                target_turbine_idx = np.argsort(bin_outliers.sum(axis=0))[-1]
                other_outputs[0][target_turbine_idx] # TODO plot median, mean, need wind speed bins too...
                fig, axs = plot.plot_power_curve(
                    data_inspector.collect_data(df=df_query, feature_types="wind_speed").to_numpy()[:, target_turbine_idx],
                    data_inspector.collect_data(df=df_query, feature_types="power_output").to_numpy()[:, target_turbine_idx],
                    flag=bin_outliers[:, target_turbine_idx],
                    flag_labels=("Anomylous Measurements", "Normal Measurements"),
                    xlim=(-1, 30),
                    ylim=(-100, 3000),
                    legend=True,
                    scatter_kwargs=dict(alpha=0.4, s=10),
                    figure_kwargs=dict(figsize=(10, 6)),
                    return_fig=True
                )
                axs.tick_params(axis="x", labelsize=12*1.5)
                axs.tick_params(axis="y", labelsize=12*1.5)
                axs.xaxis.label.set_size(15*1.5)
                axs.yaxis.label.set_size(15*1.5)
                for t in axs.legend_.get_texts():
                    t.set_size(12*1.5)
                
                plt.tight_layout()
                fig.savefig(os.path.join(data_inspector.save_dir, "power_curve_bin_outliers_awaken.png"), dpi=100)
                
            # fill cells corresponding to values that are outside of power-wind speed bins with Null st they are marked for interpolation via impute or linear/forward fill interpolation later
            # loop through each turbine's wind speed and wind direction columns, and compare the distribution of data with and without the inoperational turbines
            threshold = 0.01
            logging.info("Nullifying wind speed-power curve bin outlier measurements in dataframe.")
            df_query = data_filter.conditional_filter(df_query, threshold, mask, 
                                                        mask_input_features=sorted(data_loader.turbine_ids),
                                                        output_features=ws_cols, 
                                                        filter_type="power-wind speed bin",
                                                        check_js=False)
            del bin_outliers, mask
            df_query.collect().write_parquet(config["processed_data_path"].replace(".parquet", "_filtered.parquet"), statistics=False)
            df_query = pl.scan_parquet(config["processed_data_path"].replace(".parquet", "_filtered.parquet"))
            logging.info("Finished nullifying wind speed-power curve bin outlier measurements in dataframe.") 
            if args.verbose:
                DataInspector.print_df_state(df_query, ["wind_speed", "wind_direction", "nacelle_direction"])
                
            if args.plot:
                data_inspector.plot_time_series(df_query.slice(0, ROW_LIMIT), feature_types=["wind_speed", "wind_direction"], turbine_ids=data_loader.turbine_ids, continuity_groups=None, label="after_bin_outlier")

    if "nacelle_calibration" in config["filters"]:
        if RUN_ONCE:
            if args.reload_data or args.regenerate_filters or not os.path.exists(config["processed_data_path"].replace(".parquet", "_calibrated.parquet")): 
                
                # Nacelle Calibration 
                # Find and correct wind direction offsets from median wind plant wind direction for each turbine
                logging.info("Subtracting median wind direction from wind direction and nacelle direction measurements.")
                
                # add the 3 degrees back to the wind direction signal
                offset = 3.0
                df_query2 = df_query.with_columns((cs.starts_with("wind_direction") + offset).mod(360.0))
                df_query_10min = df_query2\
                                    .with_columns(pl.col("time").dt.round(f"{10}m").alias("time"))\
                                    .group_by("time").agg(cs.numeric().mean()).sort("time")
                
                wd_median = df_query_10min.select(cs.starts_with("wind_direction").radians().sin().name.suffix("_sin"),
                                                cs.starts_with("wind_direction").radians().cos().name.suffix("_cos"))
                
                wd_median = pl.concat([wd_median.select(pl.concat_list(cs.ends_with("_sin")).list.drop_nulls().list.median().alias("wd_sin_median")), 
                                        wd_median.select(pl.concat_list(cs.ends_with("_cos")).list.drop_nulls().list.median().alias("wd_cos_median"))],
                                    how="horizontal")\
                                    .select(pl.arctan2(pl.col("wd_sin_median"), pl.col("wd_cos_median")).degrees().alias("wd_median"))
                                    # .collect().to_numpy().flatten()
                
                nd_median = df_query_10min.select(cs.starts_with("nacelle_direction").radians().sin().name.suffix("_sin"),
                                                cs.starts_with("nacelle_direction").radians().cos().name.suffix("_cos"))
                nd_median = pl.concat([nd_median.select(pl.concat_list(cs.ends_with("_sin")).list.drop_nulls().list.median().alias("nd_sin_median")), 
                                        nd_median.select(pl.concat_list(cs.ends_with("_cos")).list.drop_nulls().list.median().alias("nd_cos_median"))],
                                    how="horizontal")\
                                    .select(pl.arctan2(pl.col("nd_sin_median"), pl.col("nd_cos_median")).degrees().alias("nd_median"))
                                    # .collect().to_numpy().flatten()

                # df_query_10min = df_query_10min.with_columns(wd_median=wd_median, yaw_median=yaw_median).collect().lazy()
                df_query_10min = pl.concat([df_query_10min, wd_median, nd_median], how="horizontal")
                del wd_median, nd_median
                if args.plot:
                    data_inspector.plot_wind_offset(df_query_10min, "Original", data_loader.turbine_ids)

                # remove biases from median direction

                # df_offsets = {"turbine_id": [], "northing_bias": []}
                if args.reload_data or args.regenerate_filters or not os.path.exists(config["processed_data_path"].replace(".parquet", "_biases.npy")):
                    data_filter.multiprocessor = None
                    biases = data_filter.multi_compute_bias(df_query_10min, data_loader.turbine_ids)
                    data_filter.multiprocessor = args.multiprocessor
                    np.save(config["processed_data_path"].replace(".parquet", "_biases.npy"), biases)
                else:
                    biases = np.load(config["processed_data_path"].replace(".parquet", "_biases.npy"))
                    
                for bias, turbine_id in zip(biases, data_loader.turbine_ids):
                    
                    df_query_10min = df_query_10min.with_columns((pl.col(f"wind_direction_{turbine_id}") - bias).mod(360.0).alias(f"wind_direction_{turbine_id}"), 
                                                                (pl.col(f"nacelle_direction_{turbine_id}") - bias).mod(360.0).alias(f"nacelle_direction_{turbine_id}"))
                    df_query2 = df_query2.with_columns((pl.col(f"wind_direction_{turbine_id}") - bias).mod(360.0).alias(f"wind_direction_{turbine_id}"), 
                                                    (pl.col(f"nacelle_direction_{turbine_id}") - bias).mod(360.0).alias(f"nacelle_direction_{turbine_id}"))

                    print(f"Turbine {turbine_id} bias from median wind direction: {bias} deg")

                # df_offsets = pl.DataFrame(df_offsets)

                if args.plot:
                    data_inspector.plot_wind_offset(df_query_10min, "Corrected", data_loader.turbine_ids)
                    
                # make sure we have corrected the bias between wind direction and yaw position by adding 3 deg. to the wind direction
                if args.verbose and False:
                    bias = 0
                    for turbine_id in data_loader.turbine_ids:
                        turbine_bias = df_query_10min.filter(pl.col(f"power_output_{turbine_id}") >= 0)\
                                        .select("time", f"wind_direction_{turbine_id}", f"nacelle_direction_{turbine_id}")\
                                        .select(bias=(pl.col(f"wind_direction_{turbine_id}") - pl.col(f"nacelle_direction_{turbine_id}")))\
                                        .select(sin=pl.all().radians().sin().mean(), cos=pl.all().radians().cos().mean())\
                                        .select(pl.arctan2(pl.col("sin"), pl.col("cos")).degrees().mod(360.0).alias("bias"))\
                                        .select(pl.when(pl.all() > 180.0).then(pl.all() - 360.0).otherwise(pl.all()))\
                                        .collect().item() or 0
                        bias += turbine_bias
                                    
                        # bias += DataFilter.wrap_180(DataFilter.circ_mean(df.select(pl.col(f"wind_direction_{turbine_id}") - pl.col(f"nacelle_direction_{turbine_id}")).collect().to_numpy().flatten()))
                        
                    print(f"Average Bias = {bias / len(data_loader.turbine_ids)} deg")

                # %%
                # Find offset to true North using wake loss profiles

                logging.info("Finding offset to true North using wake loss profiles.")

                # Find offsets between direction of alignment between pairs of turbines 
                # and direction of peak wake losses. Use the average offset found this way 
                # to identify the Northing correction that should be applied to all turbines 
                # in the wind farm.
                
                dir_offsets = compute_offsets(df_query_10min, data_inspector.fmodel, turbine_ids=data_loader.turbine_ids,
                                            turbine_pairs=config["nacelle_calibration_turbine_pairs"],
                                            plot=args.plot,
                                            save_path=os.path.join(os.path.dirname(config["processed_data_path"]), "pre_correction.png")
                                            )
                
                if dir_offsets:
                    # Apply Northing offset to each turbine
                    dir_offsets = np.mean(dir_offsets)
                    for turbine_id in data_loader.turbine_ids:
                        # df_query_10min = df_query_10min.with_columns((pl.col(f"wind_direction_{turbine_id}") - dir_offsets).mod(360).alias(f"wind_direction_{turbine_id}"),
                        #                                              (pl.col(f"nacelle_direction_{turbine_id}") - dir_offsets).mod(360).alias(f"nacelle_direction_{turbine_id}"))
                        
                        df_query2 = df_query2.with_columns((pl.col(f"wind_direction_{turbine_id}") - dir_offsets).mod(360).alias(f"wind_direction_{turbine_id}"),
                                                        (pl.col(f"nacelle_direction_{turbine_id}") - dir_offsets).mod(360).alias(f"nacelle_direction_{turbine_id}"))

                    # Determine final wind direction correction for each turbine
                    # df_offsets = df_offsets.with_columns(
                    #     northing_bias=(pl.col("northing_bias") + np.mean(dir_offsets)))\
                    #     .with_columns(northing_bias=pl.when(pl.col("northing_bias") > 180.0)\
                    #             .then(pl.col("northing_bias") - 360.0)\
                    #             .otherwise(pl.col("northing_bias"))\
                    #             .round(2))
                    
                    # verify that Northing calibration worked properly
                    # new_dir_offsets = compute_offsets(df_query_10min, data_inspector.fmodel, turbine_ids=data_loader.turbine_ids,
                    #                                 turbine_pairs=config["nacelle_calibration_turbine_pairs"],
                    #                                 # turbine_pairs=[(51,50),(43,42),(41,40),(18,19),(34,33),(22,21),(87,86),(62,63),(33,32),(59,60),(43,42)],
                    #                                 plot=args.plot,
                    #                                 save_path=os.path.join(os.path.dirname(config["processed_data_path"]), "post_correction.png")
                    # ) 
                del df_query_10min
                df_query = df_query2
                
                # need to sink parquet and recollect to avoid recursion limit error
                df_query.collect().write_parquet(config["processed_data_path"].replace(".parquet", "_calibrated.parquet"), statistics=False)
                df_query = pl.scan_parquet(config["processed_data_path"].replace(".parquet", "_calibrated.parquet"))
            else:
                df_query = pl.scan_parquet(config["processed_data_path"].replace(".parquet", "_calibrated.parquet"))

            # %% check time series
            if args.verbose:
                DataInspector.print_df_state(df_query, ["wind_speed", "wind_direction", "nacelle_direction"])
            if args.plot:
                data_inspector.plot_time_series(df_query.slice(0, ROW_LIMIT), feature_types=["wind_speed", "wind_direction"], turbine_ids=data_loader.turbine_ids, continuity_groups=None, label="after_nacelle_calibration")
    
    
    # %% Feature Selection
    if RUN_ONCE:
        logging.info("Selecting features.")
        
        df_query2 = df_query\
            .with_columns(**{f"ws_horz_{tid}": (pl.col(f"wind_speed_{tid}") * ((pl.col(f"wind_direction_{tid}") - 180.0).radians().sin())) for tid in data_loader.turbine_ids})\
            .with_columns(**{f"ws_vert_{tid}": (pl.col(f"wind_speed_{tid}") * ((pl.col(f"wind_direction_{tid}") - 180.0).radians().cos())) for tid in data_loader.turbine_ids})\
            .with_columns(**{f"nd_cos_{tid}": ((pl.col(f"nacelle_direction_{tid}")).radians().cos()) for tid in data_loader.turbine_ids})\
            .with_columns(**{f"nd_sin_{tid}": ((pl.col(f"nacelle_direction_{tid}")).radians().sin()) for tid in data_loader.turbine_ids})\
            .select(pl.col("time"), cs.starts_with("ws_horz"), cs.starts_with("ws_vert"), cs.starts_with("nd_sin"), cs.starts_with("nd_cos"), cs.starts_with("power_output"))
        
        # df_query2.filter(pl.col("continuity_group") == 5).select("time", "ws_horz_1", "ws_vert_1").filter((pl.col("time") > datetime(2020, 5, 24, 4, 30)) & (pl.col("time") < datetime(2020, 5, 24, 6, 30))).collect().to_numpy()[:, 1].flatten() 
        if False:
            wind_dirs = np.arctan2(df_query2.select(cs.starts_with("ws_horz")).collect().to_numpy(), 
                                df_query2.select(cs.starts_with("ws_vert")).collect().to_numpy()) * (180.0 / np.pi)
            wind_dirs[wind_dirs < 0] = 360.0 + wind_dirs[wind_dirs < 0]
            wind_dirs = np.mod(wind_dirs, 360.0)
            assert np.allclose(df_query.select(cs.starts_with("wind_direction")).collect().to_numpy(), wind_dirs)
            wind_mags = (df_query2.select(cs.starts_with("ws_horz")).collect().to_numpy()**2 + df_query2.select(cs.starts_with("ws_vert")).collect().to_numpy()**2)**0.5
            assert np.allclose(df_query.select(cs.starts_with("wind_speed")).collect().to_numpy(), wind_mags)
        
        df_query = df_query2
    
        # %%
        if args.plot:
            logging.info("Plotting time series.")
            feature_types = ["nd_cos", "nd_sin", "ws_horz", "ws_vert"]
            if "continuity_group" in df_query.collect_schema().names():
                continuity_groups = df_query.select(pl.col("continuity_group")).unique().collect().to_numpy().flatten()
                
                data_inspector.plot_time_series(pl.concat([df.slice(0, ROW_LIMIT) for df in df_query.collect().partition_by("continuity_group")], how="vertical").lazy(), 
                                                feature_types=["ws_horz", "ws_vert"], 
                                                turbine_ids=data_loader.turbine_ids, 
                                                continuity_groups=continuity_groups,
                                                label="after_feat_select"
                                                # continuity_groups=continuity_groups
                                                )
            else:
                data_inspector.plot_time_series(df_query.slice(0, ROW_LIMIT), 
                                                feature_types=["ws_horz", "ws_vert"],
                                                turbine_ids=data_loader.turbine_ids, 
                                                continuity_groups=None, 
                                                label="after_feat_select")
        
    # %%
    if "std_range_flag" in config["filters"]:
        if RUN_ONCE:
            logging.info("Nullifying standard deviation outliers.")
            
        # apply a bin filter to remove data with power values outside of an envelope around median power curve at each wind speed
        if args.reload_data or args.regenerate_filters or not os.path.exists(config["processed_data_path"].replace(".parquet", "_std_dev_outliers.pkl")):
            # df_query.select("time", "ws_vert_1").with_row_index().filter(((pl.col("time") > datetime(2020, 5, 23, 20, 45)) & (pl.col("time") < datetime(2020, 5, 23, 21, 45)))).collect().select("index").to_numpy().flatten() 
            # TODO consider neighboring turbines only
            std_dev_outliers = filters.std_range_flag(
                data_pl=df_query.select(cs.starts_with("ws_horz"), cs.starts_with("ws_vert")),
                threshold=config["filters"]["std_range_flag"]["threshold"], 
                over="asset", feature_types=["ws_horz", "ws_vert"],
                # asset_coords={tid: (data_inspector.fmodel.layout_x[t], data_inspector.fmodel.layout_y[t]) for t, tid in enumerate(data_loader.turbine_ids)}
            ) & df_query.select(cs.starts_with("ws_horz").is_not_null(), cs.starts_with("ws_vert").is_not_null()).collect().to_numpy() 
             
            if RUN_ONCE:
                with open(config["processed_data_path"].replace(".parquet", "_std_dev_outliers.pkl"), "wb") as f:  
                    pickle.dump(std_dev_outliers.values, f)

        elif RUN_ONCE:
            # std_dev_outliers = np.load(config["processed_data_path"].replace(".parquet", "_std_dev_outliers.npy"), allow_pickle=True)[()]
            with open(config["processed_data_path"].replace(".parquet", "_std_dev_outliers.pkl"), "rb") as f:  
                std_dev_outliers = pickle.load(f)

        if RUN_ONCE:
            mask = lambda feat: std_dev_outliers[:, (ws_horz_cols + ws_vert_cols).index(feat)]
            
            # check if wind speed/dir measurements from inoperational turbines differ from fully operational
            ws_horz_cols = [col for col in df_query.collect_schema().names() if col.startswith("ws_horz")]
            ws_vert_cols = [col for col in df_query.collect_schema().names() if col.startswith("ws_vert")]
            if args.verbose:
                DataInspector.print_pc_remaining_vals(df_query, mask,
                                                        mask_input_features=ws_horz_cols+ws_vert_cols,
                                                        output_features=ws_horz_cols+ws_vert_cols,
                                                        filter_type="standard deviation")
            
            if args.plot:
                data_inspector.plot_nulled_vs_remaining(df_query.slice(0, ROW_LIMIT), mask,
                                                        mask_input_features=ws_horz_cols,
                                                        output_features=ws_horz_cols, 
                                                        feature_types=["ws_horz"], 
                                                        feature_labels=["Horizontal Wind Speed (m/s) after Standard Deviation Filter"])

                data_inspector.plot_nulled_vs_remaining(df_query.slice(0, ROW_LIMIT), mask,
                                                        mask_input_features=ws_vert_cols,
                                                        output_features=ws_vert_cols,
                                                        feature_types=["ws_vert"], 
                                                        feature_labels=["Vertical Wind Speed (m/s) after Standard Deviation Filter"])
            
            # fill cells corresponding to values that are outside of power-wind speed bins with Null st they are marked for interpolation via impute or linear/forward fill interpolation later
            # loop through each turbine's wind speed and wind direction columns, and compare the distribution of data with and without the inoperational turbines
            threshold = 0.01
            logging.info("Nullifying horizontal/vertical wind speed standard deviation measurements in dataframe.")
            df_query = data_filter.conditional_filter(df_query, threshold, mask, 
                                                        mask_input_features=ws_horz_cols+ws_vert_cols,
                                                        output_features=ws_horz_cols+ws_vert_cols, 
                                                        filter_type="standard deviation", 
                                                        check_js=False)
            del std_dev_outliers
            
            # need to sink parquet and recollect to avoid recursion limit error
            df_query.collect().write_parquet(config["processed_data_path"].replace(".parquet", "_stddev.parquet"), statistics=False)
            df_query = pl.scan_parquet(config["processed_data_path"].replace(".parquet", "_stddev.parquet")) 
            logging.info("Finished nullifying horizontal/vertical wind speed standard deviation measurements in dataframe.") 
            
            # check time series 
            if args.verbose:
                DataInspector.print_df_state(df_query, ["ws_horz", "ws_vert", "nd_cos", "nd_sin"])
                
            if args.plot:
                data_inspector.plot_time_series(df_query.slice(0, ROW_LIMIT), feature_types=["ws_horz", "ws_vert"], turbine_ids=data_loader.turbine_ids, continuity_groups=None, label="after_std_dev")

    # %%
    if "split" in config["filters"]:
        if RUN_ONCE:
            if args.reload_data or args.regenerate_filters or not os.path.exists(config["processed_data_path"].replace(".parquet", "_split.parquet")):
                logging.info("Split dataset during time steps for which many turbines have missing data.")
                
                # if there is a short or long gap for some turbines, impute them using the imputing.impute_all_assets_by_correlation function
                #       else if there is a short or long gap for many turbines, split the dataset
                assert config["filters"]["split"]["missing_col_thr"] <= len(data_loader.turbine_ids) 
                missing_col_thr = config["filters"]["split"]["missing_col_thr"] 
                missing_duration_thr = np.timedelta64(config["filters"]["split"]["missing_duration_thr"], "s")
                minimum_not_missing_duration = np.timedelta64(config["filters"]["split"]["minimum_not_missing_duration"], "s")
                missing_data_cols = ["ws_horz", "ws_vert"]

                # check for any periods of time for which more than 'missing_col_thr' features have missing data
                df_query2 = df_query\
                        .with_columns(*[cs.contains(col).is_null().name.prefix("is_missing_") for col in missing_data_cols])\
                        .with_columns(**{f"num_missing_{col}": pl.sum_horizontal((cs.contains(col) & cs.starts_with("is_missing"))) for col in missing_data_cols})

                # subset of data, indexed by time, which has <= the threshold number of missing columns
                # check that the number of missing wind dir/speed measurements (over all turbines) is less or equal to missing_col_thr (i.e. both the number of missing wind dirs and wind speeds must be <= missing_col_thr)
                df_query_not_missing_times = add_df_continuity_columns(df_query2, 
                                                                    dt=data_loader.dt,
                                                                    mask=pl.all_horizontal(cs.starts_with("num_missing") <= missing_col_thr) 
                                                                    #    mask=pl.sum_horizontal(cs.starts_with("num_missing")) <= missing_col_thr, 
                                                                    )

                # subset of data, indexed by time, which has > the threshold number of missing wind speed or wind dir
                df_query_missing_times = add_df_continuity_columns(df_query2, 
                                                                dt=data_loader.dt,
                                                                mask=pl.any_horizontal(cs.starts_with("num_missing") > missing_col_thr)
                                                                #    mask=pl.sum_horizontal(cs.starts_with("num_missing")) > missing_col_thr, 
                                                                )

                # start times, end times, and durations of each of the continuous subsets of data in df_query_missing_times 
                df_query_not_missing = add_df_agg_continuity_columns(df_query_not_missing_times) 
                df_query_missing = add_df_agg_continuity_columns(df_query_missing_times)

                # start times, end times, and durations of each of the continuous subsets of data in df_query_not_missing_times 
                # AND of each of the continuous subsets of data in df_query_missing_times that are under the threshold duration time 
                df_query_not_missing = pl.concat([df_query_not_missing, 
                                                        df_query_missing.filter(pl.col("duration") <= missing_duration_thr)])\
                                        .sort("start_time")

                df_query_missing = df_query_missing.filter(pl.col("duration") > missing_duration_thr)

                if df_query_not_missing.select(pl.len()).collect().item() == 0:
                    raise Exception("Parameters 'missing_col_thr' or 'missing_duration_thr' are too stringent, can't find any eligible durations of time.")

                df_query_missing = merge_adjacent_periods(agg_df=df_query_missing, dt=data_loader.dt)
                df_query_not_missing = merge_adjacent_periods(agg_df=df_query_not_missing, dt=data_loader.dt)

                df_query_missing = group_df_by_continuity(df=df_query2, agg_df=df_query_missing, missing_data_cols=missing_data_cols)
                df_query_not_missing = group_df_by_continuity(df=df_query2, agg_df=df_query_not_missing, missing_data_cols=missing_data_cols)
                df_query_not_missing = df_query_not_missing.filter(pl.col("duration") >= minimum_not_missing_duration)
                
                df_query = df_query2.select(*[cs.starts_with(feat_type) for feat_type in ["time", "ws_horz", "ws_vert", "nd_cos", "nd_sin", "power_output"]])
                del df_query2
                
                if args.plot:
                    # Plot number of missing wind dir/wind speed data for each wind turbine (missing duration on x axis, turbine id on y axis, color for wind direction/wind speed)
                    from matplotlib import colormaps
                    from matplotlib.ticker import MaxNLocator
                    fig, ax = plt.subplots(1, 1)
                    for feature_type, marker in zip(missing_data_cols, ["o", "^"]):
                        for turbine_id, color in zip(data_loader.turbine_ids, colormaps["tab20c"](np.linspace(0, 1, len(data_loader.turbine_ids)))):
                            df = df_query_missing.select("duration", f"is_missing_{feature_type}_{turbine_id}").collect().to_pandas()
                            ax.scatter(x=df["duration"].dt.seconds / 3600,
                                        y=df[f"is_missing_{feature_type}_{turbine_id}"].astype(int),  
                            marker=marker, label=turbine_id, s=400, color=color)
                    ax.set_title("Occurence of Missing Wind Speed (circle) and Wind Direction (triangle) Values vs. Missing Duration, for each Turbine")
                    ax.set_xlabel("Duration of Missing Values (hrs)")
                    ax.set_ylabel("Number of Missing Values over this Duration")
                    h, l = ax.get_legend_handles_labels()
                    # ax.legend(h[:len(data_loader.turbine_ids)], l[:len(data_loader.turbne_ids)], ncol=8)
                    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

                    # Plot missing duration on x axis, number of missing turbines on y-axis, marker for wind speed vs wind direction,
                    fig, ax = plt.subplots(1, 1)
                    for feature_type, marker in zip(missing_data_cols, ["o", "^"]):
                        df = df_query_missing.select("duration", (cs.contains(feature_type) & cs.starts_with("is_missing")))\
                                                .with_columns(pl.sum_horizontal([f"is_missing_{feature_type}_{tid}" for tid in data_loader.turbine_ids]).alias(f"is_missing_{feature_type}")).collect().to_pandas()
                        ax.scatter(x=df["duration"].dt.seconds / 3600,
                                    y=df[f"is_missing_{feature_type}"].astype(int),  
                        marker=marker, label=feature_type, s=400)
                    ax.set_title("Occurence of Missing Wind Speed (circle) and Wind Direction (triangle) Values vs. Missing Duration, for all Turbines")
                    ax.set_xlabel("Duration of Missing Values [hrs]")
                    ax.set_ylabel("Number of Missing Values over this Duration")
                    h, l = ax.get_legend_handles_labels()
                    # ax.legend(h[:len(missing_data_cols)], l[:len(missing_data_cols)], ncol=8)
                    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

                # if more than 'missing_col_thr' columns are missing data for more than 'missing_timesteps_thr', split the dataset at the point of temporal discontinuity
                # df_query = [df.lazy() for df in df_query.with_columns(get_continuity_group_index(df_query_not_missing).alias("continuity_group"))\
                #                           .filter(pl.col("continuity_group") != -1)\
                #                           .drop(cs.contains("is_missing") | cs.contains("num_missing"))
                #                           .collect(streaming=True)\
                #                           .sort("time")
                #                           .partition_by("continuity_group")]

                # x = df_query.collect().partition_by("continuity_group")
                # x[0].select(pl.any_horizontal(cs.numeric().is_not_null().sum() < 2)).item()
                
                # filter out the continuity groups for which any measurement has 0 non-null values, can't impute then
                df_query_not_missing = df_query_not_missing.select(pl.col("duration"), pl.col("start_time"), pl.col("end_time"), pl.col("continuity_group"), 
                                            cs.starts_with("is_missing") & cs.matches(data_loader.turbine_signature))\
                                    .filter(pl.all_horizontal(cs.starts_with("is_missing") 
                                                            < ((pl.col("duration") / np.timedelta64(data_loader.dt, 's')).cast(pl.Int64))))
                
                # df_query_not_missing.collect().select(pl.col("duration"), pl.col("start_time"), pl.col("end_time"), pl.col("continuity_group"), cs.contains("3"))\
                #                     .select(cs.starts_with("is_missing") / (pl.col("duration") / np.timedelta64(data_loader.dt, 's')).cast(pl.Int64))
                
                df_query = df_query.with_columns(get_continuity_group_index(df_query_not_missing).alias("continuity_group"))\
                                        .filter(pl.col("continuity_group") != -1)\
                                        .drop(cs.contains("is_missing") | cs.contains("num_missing"))\
                                        .sort("time").collect().lazy()

                if df_query.select(pl.len()).collect().item() == 0:
                    logging.warn(f"No remaining data rows after splicing time steps with over {missing_col_thr} missing columns")
                    
                # need to sink parquet and recollect to avoid recursion limit error
                df_query.collect().write_parquet(config["processed_data_path"].replace(".parquet", "_split.parquet"), statistics=False)
                df_query = pl.scan_parquet(config["processed_data_path"].replace(".parquet", "_split.parquet"))
                
                # check each split dataframe a) is continuous in time AND b) has <= than the threshold number of missing columns OR for less than the threshold time span
                # for df in df_query:
                #     assert df.select((pl.col("time").diff(null_behavior="drop") == np.timedelta64(data_loader.dt, "s")).all()).collect(streaming=True).item()
                #     assert (df.select((pl.sum_horizontal([(cs.numeric() & cs.contains(col)).is_null() for col in missing_data_cols]) <= missing_col_thr)).collect(streaming=True)
                #             |  ((df.select("time").max().collect(streaming=True).item() - df.select("time").min().collect(streaming=True).item()) < missing_duration_thr))
        elif RUN_ONCE:
            df_query = pl.scan_parquet(config["processed_data_path"].replace(".parquet", "_split.parquet"))
    elif RUN_ONCE:
        df_query = df_query.with_columns(pl.lit(0).alias("continuity_group"))

    # %% check time series
    if args.verbose:
        DataInspector.print_df_state(df_query, ["ws_horz", "ws_vert", "nd_cos", "nd_sin"])
        
    if args.plot:
        continuity_groups = df_query.select("continuity_group").unique().collect().to_numpy().flatten()
        data_inspector.plot_time_series(pl.concat([df.slice(0, ROW_LIMIT) for df in df_query.collect().partition_by("continuity_group")], how="vertical").lazy(), 
                                        feature_types=["ws_horz", "ws_vert"], 
                                        turbine_ids=data_loader.turbine_ids, 
                                        continuity_groups=continuity_groups, 
                                        label="after_split")
    
    # %%
    # df_query.filter(pl.col("continuity_group") == 5).select("time", "ws_vert_1").filter((pl.col("time") > datetime(2020, 5, 23, 20, 45)) & (pl.col("time") < datetime(2020, 5, 23, 21, 45))).collect().to_numpy()[:, 1].flatten() 
    if "impute_missing_data" in config["filters"]:
        if RUN_ONCE:
             logging.info("Impute/interpolate turbine missing data from correlated measurements.")
        
        if args.reload_data or args.regenerate_filters or not os.path.exists(config["processed_data_path"].replace(".parquet", "_imputed.parquet")):
            
            # else, for each of those split datasets, impute the values using the imputing.impute_all_assets_by_correlation function
            # fill data on single concatenated dataset

            df_query2 = data_filter._fill_single_missing_dataset(
                df_idx=0, 
                df=df_query, 
                impute_missing_features=["ws_horz", "ws_vert"], 
                # impute_missing_features=["wind_direction", "wind_speed"], 
                interpolate_missing_features=["ws_horz", "ws_vert", "nd_cos", "nd_sin"], 
                # interpolate_missing_features=["wind_direction", "wind_speed", "nacelle_direction"], 
                # parallel="feature",
                parallel="turbine_id",
                # parallel=None,
                r2_threshold=config["filters"]["impute_missing_data"]["r2_threshold"])

            if RUN_ONCE:
                df_query = df_query.drop([cs.starts_with(feat) for feat in ["ws_horz", "ws_vert", "nd_cos", "nd_sin", "power_output"]]).join(df_query2, on="time", how="left")
                del df_query2
                df_query.collect().write_parquet(config["processed_data_path"].replace(".parquet", "_imputed.parquet"), statistics=False)
                df_query = pl.scan_parquet(config["processed_data_path"].replace(".parquet", "_imputed.parquet"))
        elif RUN_ONCE:
            df_query = pl.scan_parquet(config["processed_data_path"].replace(".parquet", "_imputed.parquet"))

    # %% check time series
    if args.verbose:
        DataInspector.print_df_state(df_query, ["ws_horz", "ws_vert", "nd_cos", "nd_sin"])
        
    if args.plot:
        continuity_groups = df_query.select("continuity_group").unique().collect().to_numpy().flatten()
        data_inspector.plot_time_series(pl.concat([df.slice(0, ROW_LIMIT) for df in df_query.collect().partition_by("continuity_group")], how="vertical").lazy(), 
                                        feature_types=["ws_horz", "ws_vert"], 
                                        turbine_ids=data_loader.turbine_ids, 
                                        continuity_groups=continuity_groups, 
                                        label="after_impute")
        # df_query.filter(pl.col("continuity_group") == 5).select("time", "wind_direction_1").filter((pl.col("time") > datetime(2020, 5, 24, 4, 30)) & (pl.col("time") < datetime(2020, 5, 24, 6, 30))).collect().to_numpy()[:, 1].flatten()

    if args.plot:
        logging.info("Plotting and fitting target value distribution.")
        data_inspector.plot_data_distribution(df_query, feature_types=["ws_horz", "ws_vert"], turbine_ids=data_loader.turbine_ids, distribution=norm)
       
        logging.info("Power curve fitting.")
        # Get unpivoted data with correct column names
        df_unpivoted = DataInspector.unpivot_dataframe(
            df_query,
            turbine_signature=data_loader.turbine_signature, 
            feature_types=["wind_speed", "power_output"]
        ).head(5000).collect().to_pandas()
        
        logging.info(f"Fitting power curves with {len(df_unpivoted)} valid data points")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        x = np.linspace(0, 20, 100)
        
        # Fit the curves with error handling
        try:
            iec_curve = power_curve.IEC(
                windspeed_col="wind_speed", 
                power_col="power_output",
                data=df_unpivoted
            )
            ax.plot(x, iec_curve(x), color="red", label="IEC", linewidth=3)

            l5p_curve = power_curve.logistic_5_parametric(
                windspeed_col="wind_speed", 
                power_col="power_output",
                data=df_unpivoted
            )
            ax.plot(x, l5p_curve(x), color="blue", label="L5P", linewidth=3)
            
            # Convert sparse matrix to dense if needed
            try:
                spline_curve = power_curve.gam(
                    windspeed_col="wind_speed", 
                    power_col="power_output",
                    data=df_unpivoted,
                    n_splines=20
                )
                ax.plot(x, spline_curve(x), color="green", label="Spline", linewidth=3)
                
            except AttributeError:
                # If sparse matrix error occurs, try with fewer splines
                logging.warning("Sparse matrix error occurred, trying with fewer splines")
            
            # Plot scatter points
            ax.scatter(
                df_unpivoted["wind_speed"],
                df_unpivoted["power_output"],
                alpha=0.1,
                s=10,
                label="Measurements"
            )

            ax.set_xlabel("Wind Speed (m/s)")
            ax.set_ylabel("Power Output (kW)")
            ax.set_title("Power Curve Fitting")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            # plt.show()
            plt.savefig('power_curve.png')
            plt.close()
            
        except Exception as e:
            logging.error(f"Error during power curve fitting: {str(e)}")
        finally:
            del df_unpivoted

    # %%
    if "normalize" in config["filters"]:
        if RUN_ONCE:
            if args.reload_data or args.regenerate_filters or not os.path.exists(config["processed_data_path"].replace(".parquet", "_normalized.parquet")): 
                # Normalization & Feature Selection
                logging.info("Normalizing features.")
                
                # store min/max of each column to rescale later
                feature_types = ["nd_cos", "nd_sin", "ws_horz", "ws_vert"]
                
                norm_vals = {}
                for feature_type in feature_types:
                    norm_vals[f"{feature_type}_max"] = df_query.select(pl.max_horizontal(cs.starts_with(feature_type).max())).collect().item()
                    norm_vals[f"{feature_type}_min"] = df_query.select(pl.min_horizontal(cs.starts_with(feature_type).min())).collect().item()

                norm_vals = pl.DataFrame(norm_vals).select(pl.all().round(2))
                norm_vals.write_csv(config["processed_data_path"].replace(".parquet", "_normalization_consts.csv"))
                
                if "continuity_group" in df_query.collect_schema().names():
                    df_query = df_query.select(pl.col("time"), pl.col("continuity_group"), cs.contains("nd_sin"), cs.contains("nd_cos"), cs.contains("ws_horz"), cs.contains("ws_vert"))
                    time_cols = [pl.col("time"), pl.col("continuity_group")]
                else:
                    time_cols = [pl.col("time")]

                df_query = df_query.select(time_cols 
                                        + [((2.0 * ((cs.starts_with(feature_type) - norm_vals.select(f"{feature_type}_min").item()) 
                                        / (norm_vals.select(f"{feature_type}_max").item() - norm_vals.select(f"{feature_type}_min").item()))) - 1.0).name.keep()
                                        for feature_type in feature_types])
                
                df_query.collect().write_parquet(config["processed_data_path"].replace(".parquet", "_normalized.parquet"), statistics=False)
            else:
                df_query = pl.scan_parquet(config["processed_data_path"].replace(".parquet", "_normalized.parquet"))

if __name__ == "__main__":
    main()