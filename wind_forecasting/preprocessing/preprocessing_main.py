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

from sys import platform
import os
import logging
import argparse
import yaml
import time

mpi_exists = False
try:
    from mpi4py import MPI
    from mpi4py.futures import MPICommExecutor
    mpi_exists = True
except:
    print("No MPI available on system.")

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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg') 

from scipy.stats import norm
from floris import FlorisModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# %%

def main():
    parser = argparse.ArgumentParser(prog="WindFarmForecasting")
    parser.add_argument("-cnf", "--config", type=str)
    parser.add_argument("-m", "--multiprocessor", type=str, choices=["cf", "mpi"], required=False, default=None)
    parser.add_argument("-rf", "--regenerate_filters", action="store_true")
    parser.add_argument("-ld", "--reload_data", action="store_true")
    parser.add_argument("-pd", "--preprocess_data", action="store_true")
    parser.add_argument("-p", "--plot", action="store_true")
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config  = yaml.safe_load(file)

    config["raw_data_directory"] = os.path.expanduser(config["raw_data_directory"])
    config["processed_data_path"] = os.path.expanduser(config["processed_data_path"])
    config["turbine_input_path"] = os.path.expanduser(config["turbine_input_path"]) 
    config["farm_input_path"] = os.path.expanduser(config["farm_input_path"])

    # config["filters"] = ["nacelle_calibration", "unresponsive_sensor", "range_flag", "bin_filter", "std_range_flag", "impute_missing_data", "split", "normalize"]
    # config["filters"] = ["split", "impute_missing_data", "normalize"]
            #    ["unresponsive_sensor", "inoperational", "range_flag", "window_range_flag", "bin_filter", "std_range_flag", "split", "impute_missing_data", "normalize"]
    assert all(filt in ["nacelle_calibration", "unresponsive_sensor", "inoperational", "range_flag", "window_range_flag", "bin_filter", "std_range_flag", "split", "impute_missing_data", "normalize"] for filt in config["filters"])

    if config["raw_data_file_signature"].endswith(".nc"):
        config["data_format"] = "netcdf"
    elif config["raw_data_file_signature"].endswith(".csv"):
        config["data_format"] = "csv"
    elif config["raw_data_file_signature"].endswith(".parquet"):
        config["data_format"] = "parquet" 
    else:
        raise ValueError("Invalid file signature. Please specify either '*.nc', '*.csv', or '*.parquet'.")
    
    if "turbine_signature" not in config:
       config["turbine_signature"] = None 

    RUN_ONCE = (args.multiprocessor == "mpi" and mpi_exists and (comm_rank := MPI.COMM_WORLD.Get_rank()) == 0) or (args.multiprocessor != "mpi") or (args.multiprocessor is None)
    data_loader = DataLoader(
        data_dir=config["raw_data_directory"],
        file_signature=config["raw_data_file_signature"],
        save_path=config["processed_data_path"],
        multiprocessor=args.multiprocessor,
        dt=config["dt"],
        ffill_limit=None,
        data_format=config["data_format"],
        feature_mapping=config["feature_mapping"],
        turbine_signature=config["turbine_signature"]
    )

    # %%
    # INFO: Print netcdf structure
    if config["data_format"] == "netcdf":
        data_loader.print_netcdf_structure(data_loader.file_paths[0])

    # %%
    if RUN_ONCE:
        if not args.reload_data and os.path.exists(data_loader.save_path):
            # Note that the order of the columns in the provided schema must match the order of the columns in the CSV being read.
            logging.info("🔄 Loading existing Parquet file")
            df_query = pl.scan_parquet(source=data_loader.save_path)

            # generate turbine ids
            data_loader.turbine_ids = data_loader.get_turbine_ids(df_query, sort=True)

            logging.info("✅ Loaded existing Parquet file successfully")
        else:
            if args.multiprocessor == "mpi":
                comm_size = MPI.COMM_WORLD.Get_size()
                logging.info(f"🚀 Using MPI executor with {comm_size} processes.")
            else:
                max_workers = multiprocessing.cpu_count()
                logging.info("🖥️  Using ProcessPoolExecutor with %d workers.", max_workers)

            logging.info("🔄 Processing new data files with %d files", len(data_loader.file_paths)) 
    
    if args.reload_data or not os.path.exists(data_loader.save_path):
        df_query = data_loader.read_multi_files()
        
    if not args.preprocess_data:
        return

    assert all(any(prefix in col for col in df_query.collect_schema().names()) for prefix in ["time", "wind_speed_", "wind_direction_", "nacelle_direction_", "power_output_"]), "DataFrame must contain columns 'time', then columns with prefixes 'wind_speed_', 'wind_direction_', 'power_output_', 'nacelle_direction_'"
    assert df_query.select("time").collect().to_series().is_sorted(), "Loaded data should be sorted by time!"
    assert all(any(f"{prefix}{tid}" in col for col in df_query.collect_schema().names() if col != "time") for prefix in ["wind_speed_", "wind_direction_", "nacelle_direction_", "power_output_"] for tid in data_loader.turbine_ids), "DataFrame must contain columns with prefixes 'wind_speed_', 'wind_direction_', 'power_output_', 'nacelle_direction_' and suffixes for each turbine id" 

    # df_query = df_query.group_by("time").agg(cs.numeric().mean())
    # df_query.collect().write_parquet(config["processed_data_path"], statistics=False)
    
    data_inspector = DataInspector(
        turbine_input_filepath=config["turbine_input_path"],
        farm_input_filepath=config["farm_input_path"],
        data_format='auto'
    )

    # %% Plot Wind Farm, Data Distributions
    if args.plot:
        logging.info("🔄 Generating plots.")
        data_inspector.plot_wind_farm()
        data_inspector.plot_wind_speed_power(df_query, turbine_ids=data_loader.turbine_ids[:5])
        data_inspector.plot_wind_speed_weibull(df_query, turbine_ids="all")
        data_inspector.plot_wind_rose(df_query, turbine_ids="all")
        data_inspector.plot_correlation(df_query, 
        data_inspector.get_features(df_query, feature_types=["wind_speed", "wind_direction", "nacelle_direction"], 
                                    turbine_ids=data_loader.turbine_ids[:1]))
        data_inspector.plot_boxplot_wind_speed_direction(df_query, 
                                                         turbine_ids=data_loader.turbine_ids[:1])
        data_inspector.plot_time_series(df_query.head(1000), 
                                        turbine_ids=data_loader.turbine_ids[:1])
        plot.column_histograms(data_inspector.collect_data(df=df_query.head(1000), 
                                    feature_types=data_inspector.get_features(df_query, ["wind_speed", "wind_direction"])))
        logging.info("✅ Generated plots.")

    # %% check time series
    if args.plot:
        DataInspector.print_df_state(df_query, ["wind_speed", "wind_direction", "nacelle_direction"])
        data_inspector.plot_time_series(df_query.head(1000), feature_types=["wind_speed", "wind_direction"], turbine_ids=data_loader.turbine_ids[:1], continuity_groups=None)
    
    # df_query.filter(df_query.select("time").collect().to_numpy().flatten() >= datetime.datetime(2022, 3, 2, 0, 0)).head(10).collect()
    # %%
    if "nacelle_calibration" in config["filters"]:
        if args.reload_data or not os.path.exists(config["processed_data_path"].replace(".parquet", "_calibrated.parquet")): 
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
            wd_median = wd_median.select(wind_direction_sin_median=np.nanmedian(wd_median.select(cs.ends_with("_sin")).collect().to_numpy(), axis=1), 
                                        wind_direction_cos_median=np.nanmedian(wd_median.select(cs.ends_with("_cos")).collect().to_numpy(), axis=1))\
                                .select(pl.arctan2(pl.col("wind_direction_sin_median"), pl.col("wind_direction_cos_median")).degrees().alias("wind_direction_median"))\
                                .collect().to_numpy().flatten()
            
            yaw_median = df_query_10min.select(cs.starts_with("nacelle_direction").radians().sin().name.suffix("_sin"),
                                            cs.starts_with("nacelle_direction").radians().cos().name.suffix("_cos"))
            yaw_median = yaw_median.select(nacelle_direction_sin_median=np.nanmedian(yaw_median.select(cs.ends_with("_sin")).collect().to_numpy(), axis=1), 
                                        nacelle_direction_cos_median=np.nanmedian(yaw_median.select(cs.ends_with("_cos")).collect().to_numpy(), axis=1))\
                                .select(pl.arctan2(pl.col("nacelle_direction_sin_median"), pl.col("nacelle_direction_cos_median")).degrees().alias("nacelle_direction_median"))\
                                .collect().to_numpy().flatten()

            df_query_10min = df_query_10min.with_columns(wd_median=wd_median, yaw_median=yaw_median).collect().lazy()

            if args.plot:
                data_inspector.plot_wind_offset(df_query_10min, "Original", data_loader.turbine_ids)

            # remove biases from median direction

            df_offsets = {"turbine_id": [], "northing_bias": []}
            for turbine_id in data_loader.turbine_ids:
                
                bias = df_query_10min\
                            .filter(pl.col(f"power_output_{turbine_id}") >= 0)\
                            .select("time", f"wind_direction_{turbine_id}", f"nacelle_direction_{turbine_id}", "wd_median", "yaw_median")\
                            .select(wd_bias=(pl.col(f"wind_direction_{turbine_id}") - pl.col("wd_median")), 
                                    yaw_bias=(pl.col(f"nacelle_direction_{turbine_id}") - pl.col("yaw_median")))\
                            .select(pl.all().radians().sin().mean().name.suffix("_sin"), pl.all().radians().cos().mean().name.suffix("_cos"))\
                            .select(wd_bias=pl.arctan2("wd_bias_sin", "wd_bias_cos").degrees().mod(360),
                                    yaw_bias=pl.arctan2("yaw_bias_sin", "yaw_bias_cos").degrees().mod(360))\
                            .select(pl.when(pl.all() > 180.0).then(pl.all() - 360.0).otherwise(pl.all()))

                df_offsets["turbine_id"].append(turbine_id)
                wd_bias = bias.select('wd_bias').collect().item()
                yaw_bias = bias.select("yaw_bias").collect().item()
                bias = (wd_bias or 0) + (yaw_bias or 0)
                bias = 0.5 * bias
                df_offsets["northing_bias"].append(np.round(bias, 2))
                
                df_query_10min = df_query_10min.with_columns((pl.col(f"wind_direction_{turbine_id}") - bias).mod(360.0).alias(f"wind_direction_{turbine_id}"), 
                                                            (pl.col(f"nacelle_direction_{turbine_id}") - bias).mod(360.0).alias(f"nacelle_direction_{turbine_id}"))
                df_query2 = df_query2.with_columns((pl.col(f"wind_direction_{turbine_id}") - bias).mod(360.0).alias(f"wind_direction_{turbine_id}"), 
                                                (pl.col(f"nacelle_direction_{turbine_id}") - bias).mod(360.0).alias(f"nacelle_direction_{turbine_id}"))

                print(f"Turbine {turbine_id} bias from median wind direction: {bias} deg")

            df_offsets = pl.DataFrame(df_offsets)

            if args.plot:
                data_inspector.plot_wind_offset(df_query_10min, "Corrected", data_loader.turbine_ids)
                
            # make sure we have corrected the bias between wind direction and yaw position by adding 3 deg. to the wind direction
            bias = 0
            for turbine_id in data_loader.turbine_ids:
                turbine_bias = df_query_10min.filter(pl.col(f"power_output_{turbine_id}") >= 0)\
                                .select("time", f"wind_direction_{turbine_id}", f"nacelle_direction_{turbine_id}")\
                                .select(bias=(pl.col(f"wind_direction_{turbine_id}") - pl.col(f"nacelle_direction_{turbine_id}")))\
                                .select(sin=pl.all().radians().sin().mean(), cos=pl.all().radians().cos().mean())\
                                .select(pl.arctan2("sin", "cos").degrees().mod(360).alias("bias"))\
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
            fi = FlorisModel(data_inspector.farm_input_filepath)
            
            dir_offsets = compute_offsets(df_query_10min, fi, turbine_ids=data_loader.turbine_ids,
                                        turbine_pairs=config["nacelle_calibration_turbine_pairs"],
                                        # turbine_pairs=[(51,50),(43,42),(41,40),(18,19),(34,33),(22,21),(87,86),(62,63),(33,32),(59,60),(43,42)],
                                        plot=args.plot
                                        #   turbine_pairs=[(61,60),(51,50),(43,42),(41,40),(18,19),(34,33),(17,16),(21,22),(87,86),(62,63),(32,33),(59,60),(42,43)]
                                        )
            
            if dir_offsets:
                # Apply Northing offset to each turbine
                for turbine_id in data_loader.turbine_ids:
                    df_query_10min = df_query_10min.with_columns((pl.col(f"wind_direction_{turbine_id}") - np.mean(dir_offsets)).mod(360).alias(f"wind_direction_{turbine_id}"),
                                                                (pl.col(f"nacelle_direction_{turbine_id}") - np.mean(dir_offsets)).mod(360).alias(f"nacelle_direction_{turbine_id}"))
                    
                    df_query2 = df_query2.with_columns((pl.col(f"wind_direction_{turbine_id}") - np.mean(dir_offsets)).mod(360).alias(f"wind_direction_{turbine_id}"),
                                                    (pl.col(f"nacelle_direction_{turbine_id}") - np.mean(dir_offsets)).mod(360).alias(f"nacelle_direction_{turbine_id}"))

                # Determine final wind direction correction for each turbine
                df_offsets = df_offsets.with_columns(
                    northing_bias=(pl.col("northing_bias") + np.mean(dir_offsets)))\
                    .with_columns(northing_bias=pl.when(pl.col("northing_bias") > 180.0)\
                            .then(pl.col("northing_bias") - 360.0)\
                            .otherwise(pl.col("northing_bias"))\
                            .round(2))
                
                # verify that Northing calibration worked properly
                new_dir_offsets = compute_offsets(df_query_10min, fi, turbine_ids=data_loader.turbine_ids,
                                                turbine_pairs=config["nacelle_calibration_turbine_pairs"],
                                                # turbine_pairs=[(51,50),(43,42),(41,40),(18,19),(34,33),(22,21),(87,86),(62,63),(33,32),(59,60),(43,42)],
                                                plot=args.plot
                ) 

            df_query = df_query2
            df_query.collect().write_parquet(config["processed_data_path"].replace(".parquet", "_calibrated.parquet"), statistics=False)
        else:
            df_query = pl.scan_parquet(config["processed_data_path"].replace(".parquet", "_calibrated.parquet"))

    # %% [markdown]
    # ## OpenOA Data Preparation & Inspection

    # %%
    ws_cols = data_inspector.get_features(df_query, "wind_speed")
    wd_cols = data_inspector.get_features(df_query, "wind_direction")
    pwr_cols = data_inspector.get_features(df_query, "power_output")
    
    # Create a mapping from turbine ID to its index
    turbine_id_to_index = {tid: idx for idx, tid in enumerate(data_loader.turbine_ids)}

    # %% check time series
    if args.plot:
        DataInspector.print_df_state(df_query, ["wind_speed", "wind_direction", "nacelle_direction"])
        data_inspector.plot_time_series(df_query.head(1000), feature_types=["wind_speed", "wind_direction"], turbine_ids=data_loader.turbine_ids[:1], continuity_groups=None)

    # %%
    data_filter = DataFilter(turbine_availability_col=None, turbine_status_col="turbine_status", multiprocessor=args.multiprocessor, data_format='wide')

    if args.regenerate_filters or args.reload_data or not os.path.exists(config["processed_data_path"].replace(".parquet", "_filtered.parquet")):
        # %%
        if "unresponsive_sensor" in config["filters"]:
            logging.info("Nullifying unresponsive sensor cells.")
            # find stuck sensor measurements for each turbine and set them to null
            # this filter must be applied before any cells are nullified st null values aren't considered repeated values
            # find values of wind speed/direction, where there are duplicate values with nulls inbetween
            
            if args.regenerate_filters or not os.path.exists(config["processed_data_path"].replace(".parquet", "_frozen_sensors.npy")):
                thr = int(np.timedelta64(20, 'm') / np.timedelta64(data_loader.dt, 's'))
                frozen_sensors = filters.unresponsive_flag(
                    data_pl=df_query.select(cs.starts_with("wind_speed"), cs.starts_with("wind_direction")), threshold=thr)
                
                frozen_sensors = {"wind_speed": frozen_sensors[ws_cols].values, 
                                  "wind_direction": frozen_sensors[wd_cols].values}
                np.save(config["processed_data_path"].replace(".parquet", "_frozen_sensors.npy"), frozen_sensors)
            else:
                frozen_sensors = np.load(config["processed_data_path"].replace(".parquet", "_frozen_sensors.npy"), allow_pickle=True)[()]

            # df = df_query.select("time", cs.starts_with("wind_speed"), cs.starts_with("wind_direction")).filter(frozen_sensors["wind_speed"].all(axis=1) & frozen_sensors["wind_direction"].all(axis=1))
            # df = df_query.select("time", cs.starts_with("wind_speed"), cs.starts_with("wind_direction")).filter(frozen_sensors["wind_speed"].all(axis=1) & frozen_sensors["wind_direction"].all(axis=1))
            # data_inspector.plot_time_series(df, turbine_ids=data_loader.turbine_ids, feature_types=["wind_speed", "wind_direction"], scatter=True)

            ws_mask = lambda tid: safe_mask(tid, outlier_flag=frozen_sensors["wind_speed"], turbine_id_to_index=turbine_id_to_index)
            wd_mask = lambda tid: safe_mask(tid, outlier_flag=frozen_sensors["wind_direction"], turbine_id_to_index=turbine_id_to_index)
            
            # check time series
            if args.plot:
                for feature_type, mask in frozen_sensors.items():
                    
                    plot.plot_power_curve(
                        data_inspector.collect_data(df=df_query.head(1000), feature_types="wind_speed").values.flatten(),
                        data_inspector.collect_data(df=df_query.head(1000), feature_types="power_output").values.flatten(),
                        flag=mask[:1000, :].flatten(),
                        flag_labels=(f"{feature_type} Unresponsive Sensors (n={mask.sum():,.0f})", "Normal Turbine Operations"),
                        xlim=(-1, 15),  # optional input for refining plots
                        ylim=(-100, 3000),  # optional input for refining plots
                        legend=True,  # optional flag for adding a legend
                        scatter_kwargs=dict(alpha=0.4, s=10)  # optional input for refining plots
                    )
                DataInspector.print_pc_remaining_vals(df_query, ws_cols, ws_mask)
                DataInspector.print_pc_remaining_vals(df_query, wd_cols, wd_mask)
                DataInspector.plot_nulled_vs_remaining(df_query, ws_mask, ws_cols, ["wind_speed"], ["Wind Speed [m/s]"])
                DataInspector.plot_nulled_vs_remaining(df_query, wd_mask, wd_cols, ["wind_direction"], ["Wind Direction [deg]"])

            # change the values corresponding to frozen sensor measurements to null or interpolate (instead of dropping full row, since other sensors could be functioning properly)
            # fill stuck sensor measurements with Null st they are marked for interpolation later,
            threshold = 0.01
            logging.info("Nullifying wind speed frozen sensor measurements in dataframe.")
            df_query = data_filter.conditional_filter(df_query, threshold, ws_mask, ws_cols, check_js=False)
            logging.info("Nullifying wind direction frozen sensor measurements in dataframe.")
            df_query = data_filter.conditional_filter(df_query, threshold, wd_mask, wd_cols, check_js=False)
            # df_query.select(pl.col("time"), cs.starts_with("wind_speed")).filter(frozen_sensors["wind_speed"].all(axis=1)).collect()
            del frozen_sensors, ws_mask, wd_mask
            
            if args.plot:
                DataInspector.print_df_state(df_query, ["wind_speed", "wind_direction", "nacelle_direction"])
                data_inspector.plot_time_series(df_query.head(1000), feature_types=["wind_speed", "wind_direction"], turbine_ids=data_loader.turbine_ids[:1], continuity_groups=None) 

        # %%
        if "inoperational" in config["filters"] and any(col.startswith("turbine_status") for col in df_query.collect_schema()["names"]):
            logging.info("Nullifying inoperational turbine cells.")
            # check if wind speed/dir measurements from inoperational turbines differ from fully operational
            status_codes = [1]
            mask = lambda tid: ~pl.col(f"turbine_status_{tid}").is_in(status_codes) & pl.col(f"turbine_status_{tid}").is_not_null()

            # check time series
            if args.plot:
                DataInspector.print_pc_remaining_vals(df_query, ws_cols + wd_cols, mask)
                DataInspector.plot_nulled_vs_remaining(df_query, mask, ws_cols + wd_cols, ["wind_speed", "wind_direction"], ["Wind Speed [m/s]", "Wind Direction [deg]"])
            
            # loop through each turbine's wind speed and wind direction columns, and compare the distribution of data with and without the inoperational turbines
            # fill out_of_range measurements with Null st they are marked for interpolation via impute or linear/forward fill interpolation later
            threshold = 0.01
            logging.info("Nullifying inoperational turbine measurements in dataframe.")
            df_query = data_filter.conditional_filter(df_query, threshold, mask, ws_cols + wd_cols, check_js=False)
            del mask
            
            if args.plot:
                DataInspector.print_df_state(df_query, ["wind_speed", "wind_direction", "nacelle_direction"])
                data_inspector.plot_time_series(df_query.head(1000), feature_types=["wind_speed", "wind_direction"], turbine_ids=data_loader.turbine_ids[:1], continuity_groups=None)

        # %%
        if "range_flag" in config["filters"]:
            logging.info("Nullifying wind speed out-of-range cells.")

            # check for wind speed values that are outside of the acceptable range
            if args.regenerate_filters or not os.path.exists(config["processed_data_path"].replace(".parquet", "_out_of_range.npy")):
                # Generate out_of_range array
                # Note: OpenOA's range_flag returns True for out-of-range values
                ws = df_query.select(cs.starts_with("wind_speed")).collect().to_pandas()
                out_of_range = (filters.range_flag(ws, lower=0, upper=70) & ~ws.isna()).values # range flag includes formerly null values as nan
                del ws
                np.save(config["processed_data_path"].replace(".parquet", "_out_of_range.npy"), out_of_range)
            else:
                out_of_range = np.load(config["processed_data_path"].replace(".parquet", "_out_of_range.npy"))

            # check if wind speed/dir measurements from inoperational turbines differ from fully operational 
            mask = lambda tid: safe_mask(tid, outlier_flag=out_of_range, turbine_id_to_index=turbine_id_to_index)

            # check time series
            if args.plot:
                DataInspector.print_pc_remaining_vals(df_query, ws_cols, mask)
                DataInspector.plot_nulled_vs_remaining(df_query, mask, ws_cols, ["wind_speed"], ["Wind Speed [m/s]"])

            # loop through each turbine's wind speed and wind direction columns, and compare the distribution of data with and without the inoperational turbines
            # fill out_of_range measurements with Null st they are marked for interpolation via impute or linear/forward fill interpolation later
            threshold = 0.01
            logging.info("Nullifying wind speed out of range measurements in dataframe.")
            df_query = data_filter.conditional_filter(df_query, threshold, mask, ws_cols, check_js=False)
            del out_of_range, mask
            
            if args.plot:
                DataInspector.print_df_state(df_query, ["wind_speed", "wind_direction", "nacelle_direction"])
                data_inspector.plot_time_series(df_query.head(1000), feature_types=["wind_speed", "wind_direction"], turbine_ids=data_loader.turbine_ids[:1], continuity_groups=None)
        
        # %%
        if "window_range_flag" in config["filters"]:
            logging.info("Nullifying wind speed-power curve out-of-window cells.")
            # apply a window range filter to remove data with power values outside of the window from 20 to 3000 kW for wind speeds between 5 and 40 m/s.
            # identifies when turbine is shut down, filtering for normal turbine operation
            if args.regenerate_filters or not os.path.exists(config["processed_data_path"].replace(".parquet", "_out_of_window.npy")):
                data_filter.multiprocessor = None
                out_of_window = data_filter.multi_generate_filter(df_query=df_query, filter_func=data_filter._single_generate_window_range_filter,
                                                                  feature_types=["wind_speed", "power_output"], turbine_ids=data_loader.turbine_ids,
                                                                  window_start=5., window_end=40., value_min=20., value_max=3000.)
                # NOTE AOIFE TO JUAN would it be possible for you to transform column names active power to power_output to in data loader part st all functions work
                data_filter.multiprocessor = args.multiprocessor
                np.save(config["processed_data_path"].replace(".parquet", "_out_of_window.npy"), out_of_window)
            else:
                out_of_window = np.load(config["processed_data_path"].replace(".parquet", "_out_of_window.npy"))

            # check if wind speed/dir measurements from inoperational turbines differ from fully operational 
            mask = lambda tid: safe_mask(tid, outlier_flag=out_of_window, turbine_id_to_index=turbine_id_to_index)

            if args.plot:
                DataInspector.print_pc_remaining_vals(df_query, ws_cols, mask)
                DataInspector.plot_nulled_vs_remaining(df_query, mask, ws_cols, ["wind_speed"], ["Wind Speed [m/s]"])

                # plot values that are outside of power-wind speed range
                plot.plot_power_curve(
                    data_inspector.collect_data(df=df_query, feature_types="wind_speed").to_numpy().flatten(),
                    data_inspector.collect_data(df=df_query, feature_types="power_output").to_numpy().flatten(),
                    flag=out_of_window.flatten(),
                    flag_labels=("Outside Acceptable Window", "Acceptable Power Curve Points"),
                    xlim=(-1, 15),
                    ylim=(-100, 3000),
                    legend=True,
                    scatter_kwargs=dict(alpha=0.4, s=10)
                )
                
            # fill cells corresponding to values that are outside of power-wind speed window range with Null st they are marked for interpolation via impute or linear/forward fill interpolation later
            # loop through each turbine's wind speed and wind direction columns, and compare the distribution of data with and without the inoperational turbines
            threshold = 0.01
            logging.info("Nullifying wind speed-power curve out-of-window measurements in dataframe.")
            df_query = data_filter.conditional_filter(df_query, threshold, mask, ws_cols, check_js=False)
            del out_of_window, mask
            
            if args.plot:
                DataInspector.print_df_state(df_query, ["wind_speed", "wind_direction", "nacelle_direction"])
                data_inspector.plot_time_series(df_query.head(1000), feature_types=["wind_speed", "wind_direction"], turbine_ids=data_loader.turbine_ids[:1], continuity_groups=None)
            

        if "bin_filter" in config["filters"]:
            logging.info("Nullifying wind speed-power curve bin-outlier cells.")
            # apply a bin filter to remove data with power values outside of an envelope around median power curve at each wind speed
            if args.regenerate_filters or not os.path.exists(config["processed_data_path"].replace(".parquet", "_bin_outliers.npy")):
                data_filter.multiprocessor = None
                bin_outliers = data_filter.multi_generate_filter(df_query=df_query, filter_func=data_filter._single_generate_bin_filter,
                                                                  feature_types=["wind_speed", "power_output"], turbine_ids=data_loader.turbine_ids,
                                                                  bin_width=50, threshold=3, center_type="median", 
                                                                  bin_min=20., bin_max=0.90*(df_query.select(pl.max_horizontal(cs.starts_with(f"power_output").max())).collect().item() or 3000.),
                                                                  threshold_type="scalar", direction="below")
                # NOTE AOIFE TO JUAN change acive power to power_output, also better to wrap checks within functions so that they don't create clutter here
                data_filter.multiprocessor = args.multiprocessor
                np.save(config["processed_data_path"].replace(".parquet", "_bin_outliers.npy"), bin_outliers)
            else:
                bin_outliers = np.load(config["processed_data_path"].replace(".parquet", "_bin_outliers.npy"))

            # check if wind speed/dir measurements from inoperational turbines differ from fully operational 
            mask = lambda tid: safe_mask(tid, outlier_flag=bin_outliers, turbine_id_to_index=turbine_id_to_index)
            
            # %% check time series
            if args.plot:
                DataInspector.print_pc_remaining_vals(df_query, ws_cols, mask)
                DataInspector.plot_nulled_vs_remaining(df_query, mask, ws_cols, ["wind_speed"], ["Wind Speed [m/s]"])

                # plot values outside the power-wind speed bin filter
                plot.plot_power_curve(
                    data_inspector.collect_data(df=df_query, feature_types="wind_speed").to_numpy().flatten(),
                    data_inspector.collect_data(df=df_query, feature_types="power_output").to_numpy().flatten(),
                    flag=bin_outliers.flatten(),
                    flag_labels=("Anomylous Data", "Normal Wind Speed Sensor Operation"),
                    xlim=(-1, 15),
                    ylim=(-100, 3000),
                    legend=True,
                    scatter_kwargs=dict(alpha=0.4, s=10)
                )
                
            # fill cells corresponding to values that are outside of power-wind speed bins with Null st they are marked for interpolation via impute or linear/forward fill interpolation later
            # loop through each turbine's wind speed and wind direction columns, and compare the distribution of data with and without the inoperational turbines
            threshold = 0.01
            logging.info("Nullifying wind speed-power curve bin outlier measurements in dataframe.")
            df_query = data_filter.conditional_filter(df_query, threshold, mask, ws_cols, check_js=False)
            del bin_outliers, mask
            
            if args.plot:
                DataInspector.print_df_state(df_query, ["wind_speed", "wind_direction", "nacelle_direction"])
                data_inspector.plot_time_series(df_query.head(1000), feature_types=["wind_speed", "wind_direction"], turbine_ids=data_loader.turbine_ids[:1], continuity_groups=None)

        # %%
        if "std_range_flag" in config["filters"]:
            logging.info("Nullifying standard deviation outliers.")
            # apply a bin filter to remove data with power values outside of an envelope around median power curve at each wind speed
            if args.regenerate_filters or not os.path.exists(config["processed_data_path"].replace(".parquet", "_std_dev_outliers.npy")):
                
                std_dev_outliers = filters.std_range_flag(
                    data_pl=df_query.select(cs.starts_with("wind_speed"), cs.starts_with("wind_direction"))
                ) & df_query.select(cs.starts_with("wind_speed").is_not_null(), cs.starts_with("wind_direction").is_not_null()).collect().to_numpy() 
                
                std_dev_outliers = {"wind_speed": std_dev_outliers[ws_cols].values,
                                    "wind_direction": std_dev_outliers[wd_cols].values}
                
                np.save(config["processed_data_path"].replace(".parquet", "_std_dev_outliers.npy"), std_dev_outliers)
            else:
                std_dev_outliers = np.load(config["processed_data_path"].replace(".parquet", "_std_dev_outliers.npy"), allow_pickle=True)[()]

            # check if wind speed/dir measurements from inoperational turbines differ from fully operational 
            ws_mask = lambda tid: safe_mask(tid, 
                                            outlier_flag=std_dev_outliers["wind_speed"], 
                                            turbine_id_to_index=turbine_id_to_index)
            wd_mask = lambda tid: safe_mask(tid, 
                                            outlier_flag=std_dev_outliers["wind_direction"], 
                                            turbine_id_to_index=turbine_id_to_index)
            
            # check time series
            if args.plot:
                DataInspector.print_pc_remaining_vals(df_query, ws_cols, ws_mask)
                DataInspector.print_pc_remaining_vals(df_query, wd_cols, wd_mask)
                DataInspector.plot_nulled_vs_remaining(df_query, ws_mask, ws_cols, ["wind_speed"], ["Wind Speed [m/s]"])
                DataInspector.plot_nulled_vs_remaining(df_query, wd_mask, wd_cols, ["wind_direction"], ["Wind Direction [deg]"])
            
            # fill cells corresponding to values that are outside of power-wind speed bins with Null st they are marked for interpolation via impute or linear/forward fill interpolation later
            # loop through each turbine's wind speed and wind direction columns, and compare the distribution of data with and without the inoperational turbines
            threshold = 0.01
            logging.info("Nullifying wind speed standard deviation measurements in dataframe.")
            df_query = data_filter.conditional_filter(df_query, threshold, ws_mask, features=ws_cols, check_js=False)
            logging.info("Nullifying wind direction standard deviation measurements in dataframe.")
            df_query = data_filter.conditional_filter(df_query, threshold, wd_mask, features=wd_cols, check_js=False)
            del std_dev_outliers, ws_mask, wd_mask
            
            if args.plot:
                DataInspector.print_df_state(df_query, ["wind_speed", "wind_direction", "nacelle_direction"])
                data_inspector.plot_time_series(df_query.head(1000), feature_types=["wind_speed", "wind_direction"], turbine_ids=data_loader.turbine_ids[:1], continuity_groups=None)
    
        df_query.collect().write_parquet(config["processed_data_path"].replace(".parquet", "_filtered.parquet"), statistics=False)
    else:
        df_query = pl.scan_parquet(config["processed_data_path"].replace(".parquet", "_filtered.parquet"))

    # %% check time series
    if args.plot:
        DataInspector.print_df_state(df_query, ["wind_speed", "wind_direction", "nacelle_direction"])
        data_inspector.plot_time_series(df_query.head(1000), feature_types=["wind_speed", "wind_direction"], turbine_ids=data_loader.turbine_ids[:1], continuity_groups=None)
    
    # %%
    if "split" in config["filters"]:
        if args.reload_data or not os.path.exists(config["processed_data_path"].replace(".parquet", "_split.parquet")):
            logging.info("Split dataset during time steps for which many turbines have missing data.")
            
            # if there is a short or long gap for some turbines, impute them using the imputing.impute_all_assets_by_correlation function
            #       else if there is a short or long gap for many turbines, split the dataset
            missing_col_thr = max(1, int(len(data_loader.turbine_ids) * 1.0))
            missing_duration_thr = np.timedelta64(20, "m")
            minimum_not_missing_duration = np.timedelta64(20, "m")
            missing_data_cols = ["wind_speed", "wind_direction"]

            # check for any periods of time for which more than 'missing_col_thr' features have missing data
            df_query2 = df_query\
                    .with_columns(*[cs.contains(col).is_null().name.prefix("is_missing_") for col in missing_data_cols])\
                    .with_columns(**{f"num_missing_{col}": pl.sum_horizontal((cs.contains(col) & cs.starts_with("is_missing"))) for col in missing_data_cols})

            # subset of data, indexed by time, which has <= the threshold number of missing columns
            df_query_not_missing_times = add_df_continuity_columns(df_query2, mask=pl.sum_horizontal(cs.starts_with("num_missing")) <= missing_col_thr, dt=data_loader.dt)

            # subset of data, indexed by time, which has > the threshold number of missing columns
            df_query_missing_times = add_df_continuity_columns(df_query2, mask=pl.sum_horizontal(cs.starts_with("num_missing")) > missing_col_thr, dt=data_loader.dt)

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
            
            df_query = df_query2.select(*[cs.starts_with(feat_type) for feat_type in data_loader.feature_mapping.keys()])

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
                ax.set_xlabel("Duration of Missing Values [hrs]")
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

            df_query = df_query.with_columns(get_continuity_group_index(df_query_not_missing).alias("continuity_group"))\
                                    .filter(pl.col("continuity_group") != -1)\
                                    .drop(cs.contains("is_missing") | cs.contains("num_missing"))\
                                    .sort("time").collect().lazy()

            if df_query.select(pl.len()).collect().item() == 0:
                logging.warn(f"No remaining data rows after splicing time steps with over {missing_col_thr} missing columns")
            df_query.collect().write_parquet(config["processed_data_path"].replace(".parquet", "_split.parquet"), statistics=False)
            # check each split dataframe a) is continuous in time AND b) has <= than the threshold number of missing columns OR for less than the threshold time span
            # for df in df_query:
            #     assert df.select((pl.col("time").diff(null_behavior="drop") == np.timedelta64(data_loader.dt, "s")).all()).collect(streaming=True).item()
            #     assert (df.select((pl.sum_horizontal([(cs.numeric() & cs.contains(col)).is_null() for col in missing_data_cols]) <= missing_col_thr)).collect(streaming=True)
            #             |  ((df.select("time").max().collect(streaming=True).item() - df.select("time").min().collect(streaming=True).item()) < missing_duration_thr))
        else:
            df_query = pl.scan_parquet(config["processed_data_path"].replace(".parquet", "_split.parquet"))
    else:
        df_query = df_query.with_columns(pl.lit(0).alias("continuity_group"))

    # %% check time series
    if args.plot:
        DataInspector.print_df_state(df_query, ["wind_speed", "wind_direction", "nacelle_direction"])
        continuity_groups = df_query.select("continuity_group").unique().collect().to_numpy().flatten()
        data_inspector.plot_time_series(df_query.head(1000), feature_types=["wind_speed", "wind_direction"], turbine_ids=data_loader.turbine_ids, continuity_groups=continuity_groups) 
    
    # %% 
    if "impute_missing_data" in config["filters"]:
        if args.reload_data or not os.path.exists(config["processed_data_path"].replace(".parquet", "_imputed.parquet")):
            logging.info("Impute/interpolate turbine missing data from correlated measurements.")
            # else, for each of those split datasets, impute the values using the imputing.impute_all_assets_by_correlation function
            # fill data on single concatenated dataset
            df_query2 = data_filter._fill_single_missing_dataset(df_idx=0, df=df_query, impute_missing_features=["wind_speed", "wind_direction"], 
                                                    interpolate_missing_features=["wind_direction", "wind_speed", "nacelle_direction"], 
                                                    parallel="turbine_id")

            df_query = df_query.drop([cs.starts_with(feat) for feat in ["wind_direction", "wind_speed", "nacelle_direction"]]).join(df_query2, on="time", how="left")
            df_query.collect().write_parquet(config["processed_data_path"].replace(".parquet", "_imputed.parquet"), statistics=False)
        else:
            df_query = pl.scan_parquet(config["processed_data_path"].replace(".parquet", "_imputed.parquet"))

    # %% check time series
    if args.plot:
        DataInspector.print_df_state(df_query, ["wind_speed", "wind_direction", "nacelle_direction"])
        continuity_groups = df_query.select("continuity_group").unique().collect().to_numpy().flatten()
        data_inspector.plot_time_series(df_query.head(1000), feature_types=["wind_speed", "wind_direction"], turbine_ids=data_loader.turbine_ids[:1], continuity_groups=continuity_groups)

    # %%
    if "normalize" in config["filters"]:
        if args.reload_data or not os.path.exists(config["processed_data_path"].replace(".parquet", "_normalized.parquet")): 
            # Normalization & Feature Selection
            logging.info("Normalizing and selecting features.")
            df_query = df_query\
                    .with_columns(((cs.starts_with("wind_direction") - 180.).radians().sin()).name.map(lambda c: "wd_sin_" + c.split("_")[-1]),
                                ((cs.starts_with("wind_direction") - 180.).radians().cos()).name.map(lambda c: "wd_cos_" + c.split("_")[-1]))\
                    .with_columns(**{f"ws_horz_{tid}": (pl.col(f"wind_speed_{tid}") * pl.col(f"wd_sin_{tid}")) for tid in data_loader.turbine_ids})\
                    .with_columns(**{f"ws_vert_{tid}": (pl.col(f"wind_speed_{tid}") * pl.col(f"wd_cos_{tid}")) for tid in data_loader.turbine_ids})\
                    .with_columns(**{f"nd_cos_{tid}": ((pl.col(f"nacelle_direction_{tid}") - 180.).radians().cos()) for tid in data_loader.turbine_ids})\
                    .with_columns(**{f"nd_sin_{tid}": ((pl.col(f"nacelle_direction_{tid}") - 180.).radians().sin()) for tid in data_loader.turbine_ids})
            
            if "continuity_group" in df_query.collect_schema().names():
                df_query = df_query.select(pl.col("time"), pl.col("continuity_group"), cs.contains("nd_sin"), cs.contains("nd_cos"), cs.contains("ws_horz"), cs.contains("ws_vert"))
                time_cols = [pl.col("time"), pl.col("continuity_group")]
            else:
                time_cols = [pl.col("time")]

            # store min/max of each column to rescale later
            feature_types = ["nd_cos", "nd_sin", "ws_horz", "ws_vert"]
            
            norm_vals = {}
            for feature_type in feature_types:
                norm_vals[f"{feature_type}_max"] = df_query.select(pl.max_horizontal(cs.starts_with(feature_type).max())).collect().item()
                norm_vals[f"{feature_type}_min"] = df_query.select(pl.min_horizontal(cs.starts_with(feature_type).min())).collect().item()

            norm_vals = pl.DataFrame(norm_vals).select(pl.all().round(2))
            norm_vals.write_csv(config["processed_data_path"].replace(".parquet", "_normalization_consts.csv"))

            df_query = df_query.select(time_cols 
                                    + [((2.0 * ((cs.starts_with(feature_type) - norm_vals.select(f"{feature_type}_min").item()) 
                                    / (norm_vals.select(f"{feature_type}_max").item() - norm_vals.select(f"{feature_type}_min").item()))) - 1.0).name.keep()
                                    for feature_type in feature_types])
            
            df_query.collect().write_parquet(config["processed_data_path"].replace(".parquet", "_normalized.parquet"), statistics=False)
        else:
            df_query = pl.scan_parquet(config["processed_data_path"].replace(".parquet", "_normalized.parquet"))

    # %%
    if args.plot:
        logging.info("Plotting time series.")
        feature_types = ["nd_cos", "nd_sin", "ws_horz", "ws_vert"]
        if "continuity_group" in df_query.collect_schema().names():
            continuity_groups = df_query.select(pl.col("continuity_group")).unique().collect().to_numpy().flatten()
            data_inspector.plot_time_series(df_query, feature_types=["ws_horz", "ws_vert"], 
                                            turbine_ids=data_loader.turbine_ids, continuity_groups=continuity_groups)
        
        data_inspector.plot_time_series(df_query, feature_types=["ws_horz", "ws_vert"],
                                        turbine_ids=data_loader.turbine_ids[:8], continuity_groups=None)

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
            plt.show()
            
        except Exception as e:
            logging.error(f"Error during power curve fitting: {str(e)}")


if __name__ == "__main__":
    main()