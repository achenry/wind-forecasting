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

#%load_ext memory_profiler
from data_loader import DataLoader
from data_filter import DataFilter
from data_inspector import DataInspector
from openoa.utils import plot, filters, power_curve
import polars.selectors as cs
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from sys import platform
import os
import logging
from datetime import timedelta

from scipy.stats import norm
from scipy.optimize import minimize

from floris import FlorisModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# import datetime
# t2 = datetime.datetime(2022, 3, 1, 12, 0, 55)

# %% [markdown]
# ## Print NetCDF Data Structure, Load Data, Transform Datetime Columns
def add_df_continuity_columns(df, mask):
    # change first value of continuous_shifted to false such that add_df_agg_continuity_columns catches it as a start time for a period 
    return df\
            .filter(mask)\
            .with_columns(dt=pl.col("time").diff())\
            .with_columns(dt=pl.when(pl.int_range(0, pl.len()) == 0).then(np.timedelta64(data_loader.dt, "s")).otherwise(pl.col("dt")))\
            .select("time", "dt", cs.starts_with("num_missing"), cs.starts_with("is_missing"))\
            .with_columns(continuous=pl.col("dt")==np.timedelta64(data_loader.dt, "s"))\
            .with_columns(continuous_shifted=pl.col("continuous").shift(-1, fill_value=True))

def add_df_agg_continuity_columns(df):
    # if the continuous flag is True, but the value in the row before it False
    df = df.filter(pl.col("continuous") | (pl.col("continuous") & ~pl.col("continuous_shifted")) |  (~pl.col("continuous") & pl.col("continuous_shifted")))
    start_time_cond = ((pl.col("continuous") & ~pl.col("continuous_shifted"))).shift() | (pl.int_range(0, pl.len()) == 0)
    end_time_cond = (~pl.col("continuous") & pl.col("continuous_shifted"))
    return pl.concat([df.filter(start_time_cond).select(start_time=pl.col("time")), 
                        df.with_columns(end_time=pl.col("time").shift(1)).filter(end_time_cond).select("end_time")], how="horizontal")\
                .with_columns(duration=(pl.col("end_time") - pl.col("start_time")))\
                .sort("start_time")\
            .drop_nulls()

def get_continuity_group_index(df):
    # Create the condition for the group
    group_number = None

    # Create conditions to assign group numbers based on time ranges
    for i, (start, end) in enumerate(df.collect().select("start_time", "end_time").iter_rows()):
        # print(i, start, end, duration)
        time_cond = pl.col("time").is_between(start, end)
        if group_number is None:
            group_number = pl.when(time_cond).then(pl.lit(i))
        else:
            group_number = group_number.when(time_cond).then(pl.lit(i))

    # If no group is matched, assign a default value (e.g., -1) 
    # group_number = group_number.when(pl.col("time") > end).then(pl.lit(i + 1))
    group_number = group_number.otherwise(pl.lit(-1))
    return group_number

def group_df_by_continuity(df, agg_df):
    group_number = get_continuity_group_index(agg_df)

    return pl.concat([agg_df, df.with_columns(group_number.alias("continuity_group"))\
            .filter(pl.col("continuity_group") != -1)\
            .group_by("continuity_group")\
            .agg(cs.starts_with("is_missing").sum())\
            .with_columns([pl.sum_horizontal(cs.contains(col) & cs.starts_with("is_missing")).alias(f"is_missing_{col}") for col in missing_data_cols])\
            .sort("continuity_group")], how="horizontal")

def merge_adjacent_periods(agg_df):
    # merge rows with end times corresponding to start times of the next row into the next row, until no more rows need to be merged
    # loop through and merge as long as the shifted -1 end time + dt == the start time
    all_times = agg_df.select(pl.col("start_time"), pl.col("end_time")).collect()
    data = {"start_time":[], "end_time": []}
    start_time_idx = 0
    for end_time_idx in range(all_times.select(pl.len()).item()):
        end_time = all_times.item(end_time_idx, "end_time") 
        if not (end_time_idx + 1 == all_times.select(pl.len()).item()) and (end_time + timedelta(seconds=data_loader.dt)  == all_times.item(end_time_idx + 1, "start_time")):
            continue
        
        data["start_time"].append(all_times.item(start_time_idx, "start_time"))
        data["end_time"].append(end_time)

        start_time_idx = end_time_idx + 1

    return pl.LazyFrame(data).with_columns((pl.col("end_time") - pl.col("start_time")).alias("duration"))

def plot_wind_offset(full_df, wd_median, title):
    _, ax = plt.subplots(1, 1)
    for turbine_id in data_loader.turbine_ids:
        df = full_df.filter(pl.col(f"power_output_{turbine_id}") >= 0).select("time", f"wind_direction_{turbine_id}").collect()
                            
        ax.plot(df.select("time").to_numpy().flatten(), 
                DataFilter.wrap_180(
                            (df.select(f"wind_direction_{turbine_id}")
                            - wd_median.filter(pl.col(f"power_output_{turbine_id}") >= 0).select(f"wd_median").collect()).to_numpy().flatten()),
                            label=f"{turbine_id}")

    # ax.legend(ncol=8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Wind Direction - Median Wind Direction (deg)")

    ax.set_title(title)

# Optimization function for finding waked direction
def gauss_corr(gauss_params, power_ratio):
    xs = np.array(range(-int((len(power_ratio) - 1) / 2), int((len(power_ratio) + 1) / 2), 1))
    gauss = -1 * gauss_params[2] * np.exp(-0.5 * ((xs - gauss_params[0]) / gauss_params[1])**2) + 1.
    return -1 * np.corrcoef(gauss, power_ratio)[0, 1]

def compute_offsets(df, turbine_pairs:list[tuple[int, int]]=None):
    p_min = 100
    p_max = 2500

    prat_hfwdth = 30

    prat_turbine_pairs = turbine_pairs or [(61,60), (51,50), (43,42), (41,40), (18,19), (34,33), (17,16), (21,22), (87,86), (62,63), (32,33), (59,60), (42,43)]

    dir_offsets = []

    for i in range(len(prat_turbine_pairs)):
        i_up = prat_turbine_pairs[i][0]
        i_down = prat_turbine_pairs[i][1]

        dir_align = np.degrees(np.arctan2(fi.layout_x[i_up] - fi.layout_x[i_down], fi.layout_y[i_up] - fi.layout_y[i_down])) % 360

        # df_sub = df_10min.loc[(df_10min['pow_%03d' % i_up] >= p_min) & (df_10min['pow_%03d' % i_up] <= p_max) & (df_10min['pow_%03d' % i_down] >= 0)]
        tid_up =  f'wt{i_up + 1:03d}'
        tid_down =  f'wt{i_down + 1:03d}'

        if not (any(tid_up in feat for feat in data_loader.available_features) and any(tid_down in feat for feat in data_loader.available_features)):
            continue

        df_sub = df.filter((pl.col(f"power_output_{tid_up}") >= p_min) 
                                & (pl.col(f"power_output_{tid_up}") <= p_max) 
                                & (pl.col(f"power_output_{tid_down}") >= 0))\
                                .select(f"power_output_{tid_up}", f"power_output_{tid_down}", f"wind_direction_{tid_up}", f"wind_direction_{tid_down}")
        
        # df_sub.loc[df_sub['wd_%03d' % i_up] >= 359.5,'wd_%03d' % i_up] = df_sub.loc[df_sub['wd_%03d' % i_up] >= 359.5,'wd_%03d' % i_up] - 360.0
        df_sub = df_sub.with_columns(pl.when((pl.col(f"wind_direction_{tid_up}") >= 359.5))\
                                        .then(pl.col(f"wind_direction_{tid_up}") - 360.0)\
                                        .otherwise(pl.col(f"wind_direction_{tid_up}")))

        # df_sub["wd_round"] = df_sub[f'wd_{i_up:03d}'].round()
        df_sub = df_sub.with_columns(pl.col(f"wind_direction_{tid_up}").round().alias(f"wd_round_{tid_up}"))

        df_sub = df_sub.group_by(f"wd_round_{tid_up}").mean().collect()

        p_ratio = df_sub.select(pl.col(f"power_output_{tid_down}") / pl.col(f"power_output_{tid_up}")).to_numpy().flatten()

        plt.figure()
        _, ax = plt.subplots(1,1)
        ax.plot(p_ratio, label="_nolegend_")
        ax.plot(dir_align * np.ones(2),[0,1.25], 'k--', label="Direction of Alignment")
        ax.grid()

        nadir = np.argmin(p_ratio[np.arange(int(np.round(dir_align)) - prat_hfwdth, int(np.round(dir_align)) + prat_hfwdth + 1) % 360])
        nadir = nadir + int(np.round(dir_align)) - prat_hfwdth

        opt_gauss_params = minimize(gauss_corr, [0, 5.0, 1.0], args=(p_ratio[np.arange(nadir - prat_hfwdth, nadir + prat_hfwdth + 1) % 360]),method='SLSQP')

        xs = np.array(range(-int((60 - 1) / 2),int((60 + 1) / 2),1))
        gauss = -1 * opt_gauss_params.x[2] * np.exp(-0.5 * ((xs - opt_gauss_params.x[0]) / opt_gauss_params.x[1])**2) + 1.

        ax.plot(xs + nadir, gauss,'k',label="_nolegend_")
        ax.plot(2 * [nadir + opt_gauss_params.x[0]], [0,1.25], 'r--',label="Direction of Measured Wake Center")
        ax.set_title(f"Turbine Pair: ({i_up}, {i_down})")
        ax.legend()
        ax.set_xlabel("Wind Direction [deg]")
        ax.set_ylabel("Power Ratio [-]")
        
        dir_offset = DataFilter.wrap_180(nadir + opt_gauss_params.x[0] - dir_align)
        print(f"Direction offset for turbine pair ({tid_up}, {tid_down}) = {dir_offset}")

        dir_offsets = dir_offsets + [dir_offset]

    if dir_offsets:
        print(f"Mean offset = {np.mean(dir_offsets)}")
        print(f"Std. Dev. = {np.std(dir_offsets)}")
        print(f"Min. = {np.min(dir_offsets)}")
        print(f"Max. = {np.max(dir_offsets)}")
        return dir_offsets
    else:
        print("No available turbine pairs!")


#DEBUG: ############################################# MAIN #############################################
# %%
if __name__ == "__main__":
    PLOT = True 
    RELOAD_DATA = False
    REGENERATE_FILTERS = True
    plot_unfiltered = False
    
    if platform == "darwin":
        DATA_DIR = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data"
        PL_SAVE_PATH = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data/kp.turbine.zo2.b0.raw.parquet"
        FILE_SIGNATURE = "kp.turbine.z02.b0.*.*.*.nc"
        PL_SAVE_PATH = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data/filled_data.parquet"
        FILE_SIGNATURE = "kp.turbine.z02.b0.202203*.*.*.nc"
        MULTIPROCESSOR = "cf"
        TURBINE_INPUT_FILEPATH = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/inputs/ge_282_127.yaml"
        FARM_INPUT_FILEPATH = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/inputs/gch_KP_v4.yaml"
        FEATURES = ["time", "turbine_id", "turbine_status", "wind_direction", "wind_speed", "power_output", "nacelle_direction"]
        WIDE_FORMAT = False
        COLUMN_MAPPING = {"time": "date",
                                "turbine_id": "turbine_id",
                                "turbine_status": "WTUR.TurSt",
                                "wind_direction": "WMET.HorWdDir",
                                "wind_speed": "WMET.HorWdSpd",
                                "power_output": "WTUR.W",
                                "nacelle_direction": "WNAC.Dir"
                                }
    elif platform == "linux":
        DATA_DIR = "examples/inputs/SMARTEOLE-WFC-open-dataset"
        PL_SAVE_PATH = "examples/inputs/SMARTEOLE-WFC-open-dataset/processed/SMARTEOLE_WakeSteering_SCADA_1minData.parquet"
        FILE_SIGNATURE = "SMARTEOLE_WakeSteering_SCADA_1minData.csv"
        MULTIPROCESSOR = "cf" # mpi for HPC or "cf" for local computing
        TURBINE_INPUT_FILEPATH = os.path.expanduser("~/wind-forecasting/examples/inputs/turbine_library/mm82.yaml")
        FARM_INPUT_FILEPATH = os.path.expanduser("~/wind-forecasting/examples/inputs/smarteole_farm.yaml")
        FEATURES = ["time", "active_power", "wind_speed", "nacelle_position", "wind_direction", "derate"]
        WIDE_FORMAT = True
        COLUMN_MAPPING = {
            "time": "time",
            **{f"active_power_{i}_avg": f"active_power_{i:03d}" for i in range(1, 8)},
            **{f"wind_speed_{i}_avg": f"wind_speed_{i:03d}" for i in range(1, 8)},
            **{f"nacelle_position_{i}_avg": f"nacelle_position_{i:03d}" for i in range(1, 8)},
            **{f"wind_direction_{i}_avg": f"wind_direction_{i:03d}" for i in range(1, 8)},
            **{f"derate_{i}": f"derate_{i:03d}" for i in range(1, 8)}
        }

    # turbine_ids =  ["wt028", "wt033", "wt073"]
    turbine_ids = None
    #
    if FILE_SIGNATURE.endswith(".nc"):
        DATA_FORMAT = "netcdf"
    elif FILE_SIGNATURE.endswith(".csv"):
        DATA_FORMAT = "csv"
    else:
        raise ValueError("Invalid file signature. Please specify either '*.nc' or '*.csv'.")
    
    DT = 5
    CHUNK_SIZE = 100000
    DATA_FORMAT = "csv"
    FFILL_LIMIT = int(60 * 60 * 10 // DT)
    
    data_loader = DataLoader(
        data_dir=DATA_DIR,
        file_signature=FILE_SIGNATURE,
        save_path=PL_SAVE_PATH,
        turbine_ids=turbine_ids,
        multiprocessor=MULTIPROCESSOR,
        chunk_size=CHUNK_SIZE,
        desired_feature_types=FEATURES,
        dt=DT,
        ffill_limit=FFILL_LIMIT,
        data_format=DATA_FORMAT,
        column_mapping=COLUMN_MAPPING,
        wide_format=WIDE_FORMAT
    )

    # %%
    # INFO: Print netcdf structure
    if DATA_FORMAT == "netcdf":
        data_loader.print_netcdf_structure(data_loader.file_paths[0])

    # %%
    if not RELOAD_DATA and os.path.exists(data_loader.save_path):
        # Note that the order of the columns in the provided schema must match the order of the columns in the CSV being read.
        # schema = pl.Schema(dict(sorted(({**{"time": pl.Datetime(time_unit="ms")},
        #             **{
        #                 f"{feat}_{tid}": pl.Float64
        #                 for feat in FEATURES 
        #                 for tid in [f"wt{d+1:03d}" for d in range(88)]}
        #             }).items())))
        logging.info("🔄 Loading existing Parquet file")
        df_query = pl.scan_parquet(source=data_loader.save_path)
        logging.info("✅ Loaded existing Parquet file successfully")
        data_loader.available_features = sorted(df_query.collect_schema().names())
        data_loader.turbine_ids = sorted(set(col.split("_")[-1] for col in data_loader.available_features if "wt" in col))
    else:
        logging.info("🔄 Processing new data files")
        df_query = data_loader.read_multi_files()
        df_query = data_loader.postprocess_multi_files(df_query)
        logging.info(f"Parquet file saved into {data_loader.save_path}")
        if df_query is not None:
            # Perform any additional operations on df_query if needed
            logging.info("✅ Data processing completed successfully")
        else:
            logging.warning("⚠️ No data was processed")

    # %% [markdown]
    # ## Resampling & Forward/Backward Fill

    # %%
    print(df_query.select("time").min().collect().item(), df_query.select("time").max().collect().item())

    # %% [markdown]
    # ## Plot Wind Farm, Data Distributions

    # %%
    # INFO: @Juan 11/17/24 Added feature mapping to allow for custom feature mapping, which is required for different data sources
    # TODO: Modify this according to the data source
    feature_mapping = {
        "power_output": ["active_power"],
        "wind_speed": ["wind_speed"],
        "wind_direction": ["wind_direction"],
        "nacelle_direction": ["nacelle_position"]
    }

    data_inspector = DataInspector(
        turbine_input_filepath=TURBINE_INPUT_FILEPATH,
        farm_input_filepath=FARM_INPUT_FILEPATH,
        data_format='auto',
        feature_mapping=feature_mapping
    )

    # %%
    if PLOT:
        logging.info("🔄 Generating plots.")
        data_inspector.plot_wind_farm()
        # print("[DEBUG] Available columns:", df_query.collect_schema().names()) # DEBUG
        data_inspector.plot_wind_speed_power(df_query, turbine_ids=["007","006","005","004","003","002","001"])
        data_inspector.plot_wind_speed_weibull(df_query, turbine_ids="all")
        data_inspector.plot_wind_rose(df_query, turbine_ids="all")
        data_inspector.plot_correlation(df_query, 
        data_inspector.get_features(df_query, feature_types=["wind_speed", "wind_direction", "nacelle_direction"], turbine_ids=["007"]))
        data_inspector.plot_boxplot_wind_speed_direction(df_query, turbine_ids=["007"])
        data_inspector.plot_time_series(df_query, turbine_ids=["007"], feature_mapping=feature_mapping)
        plot.column_histograms(data_inspector.collect_data(df=df_query, 
        feature_types=data_inspector.get_features(df_query, ["wind_speed", "wind_direction", "power_output", "nacelle_direction"])))
        logging.info("✅ Generated plots.")

    # %% [markdown]
    # ## OpenOA Data Preparation & Inspection

    # %%
    ws_cols = data_inspector.get_features(df_query, "wind_speed")
    wd_cols = data_inspector.get_features(df_query, "wind_direction")
    pwr_cols = data_inspector.get_features(df_query, "power_output")

    # %%
    print(f"Features of interest = {data_loader.desired_feature_types}")
    print(f"Available features = {data_loader.available_features}")
    # qa.describe(DataInspector.collect_data(df=df_query))

    # %%
    data_filter = DataFilter(turbine_availability_col=None, turbine_status_col="turbine_status", multiprocessor=MULTIPROCESSOR, data_format='wide')

    # %%
    logging.info("Nullifying inoperational turbine cells.")
    # check if wind speed/dir measurements from inoperational turbines differ from fully operational
    status_codes = [1]
    mask = (lambda tid: pl.col(f"turbine_status_{tid}").is_in(status_codes) | pl.col(f"turbine_status_{tid}").is_null()) if any("turbine_status" in col for col in df_query.collect_schema().names()) else (lambda tid: pl.lit(True))
    features = ws_cols

    if PLOT and plot_unfiltered:
        DataInspector.print_pc_unfiltered_vals(df_query, features, mask)
        DataInspector.plot_filtered_vs_unfiltered(df_query, mask, ws_cols + wd_cols, ["wind_speed", "wind_direction"], ["Wind Speed [m/s]", "Wind Direction [deg]"])

    # loop through each turbine's wind speed and wind direction columns, and compare the distribution of data with and without the inoperational turbines
    # fill out_of_range measurements with Null st they are marked for interpolation via impute or linear/forward fill interpolation later
    threshold = 0.01
    df_query = data_filter.conditional_filter(df_query, threshold, mask, ws_cols + wd_cols)

    # %%
    logging.info("Nullifying wind speed out-of-range cells.")
    
    ws_cols = [col for col in df_query.columns if col.startswith("wind_speed_")]
    turbine_ids = sorted([col.split("_")[-1] for col in ws_cols])
    logging.info(f"Extracted turbine IDs from columns: {turbine_ids}")

    # Check for wind speed values that are outside of the acceptable range
    if RELOAD_DATA or REGENERATE_FILTERS or not os.path.exists(os.path.join(DATA_DIR, "out_of_range.npy")):
        logging.info("Processing new out-of-range data.")
        ws = data_inspector.collect_data(df=df_query, feature_types="wind_speed")
        logging.info(f"Wind speed columns: {ws.columns.tolist()}")
        logging.info(f"Range flag input shape: {ws.shape}")

        # Generate out_of_range array
        # Note: OpenOA's range_flag returns True for out-of-range values
        out_of_range_flags = filters.range_flag(ws, lower=0, upper=70)
        out_of_range = out_of_range_flags.values
        logging.info(f"out_of_range shape: {out_of_range.shape}")
        logging.info(f"Out of range count per turbine: {np.sum(out_of_range, axis=0)}")
        del ws
        np.save(os.path.join(DATA_DIR, "out_of_range.npy"), out_of_range)
    else:
        logging.info("Loading existing out-of-range data.")
        out_of_range = np.load(os.path.join(DATA_DIR, "out_of_range.npy"))
        logging.info(f"Loaded out_of_range shape: {out_of_range.shape}")

    # Create a mapping from turbine ID to its index in the out_of_range array
    turbine_id_to_index = {tid: idx for idx, tid in enumerate(turbine_ids)}
    logging.info(f"Turbine ID to index mapping: {turbine_id_to_index}")

    # Define the mask function using the new mapping
    def safe_mask(tid):
        try:
            idx = turbine_id_to_index[tid]
            mask_array = ~out_of_range[:, idx]
            logging.info(f"Mask for turbine {tid} excludes {np.sum(~mask_array)} out of {len(mask_array)} data points")
            return mask_array
        except KeyError:
            logging.error(f"Mask error for turbine {tid}: turbine ID not found in mapping")
            return None

    mask = safe_mask

    # Proceed to use the mask in plotting and analysis
    features = ws_cols

    if PLOT and plot_unfiltered:
        DataInspector.print_pc_unfiltered_vals(df_query, features, mask)
        DataInspector.plot_filtered_vs_unfiltered(df_query, mask, ws_cols, ["wind_speed"], ["Wind Speed [m/s]"])

    # loop through each turbine's wind speed and wind direction columns, and compare the distribution of data with and without the inoperational turbines
    # fill out_of_range measurements with Null st they are marked for interpolation via impute or linear/forward fill interpolation later
    threshold = 0.01
    df_query = data_filter.conditional_filter(df_query, threshold, mask, ws_cols)

    del out_of_range 

    # %%
    # DEBUG: Checkpoint
    logging.info("Nullifying wind speed-power curve out-of-window cells.")
    # apply a window range filter to remove data with power values outside of the window from 20 to 3000 kW for wind speeds between 5 and 40 m/s.
    # identifies when turbine is shut down, filtering for normal turbine operation
    if RELOAD_DATA or not os.path.exists(os.path.join(DATA_DIR, "out_of_window.npy")):
        window_flags = []
        
        for tid in data_loader.turbine_ids:
            # Get wind speed and power data for current turbine
            ws_data = data_inspector.collect_data(df=df_query, 
                                                feature_types=["wind_speed"], 
                                                turbine_ids=[tid])
            power_data = data_inspector.collect_data(df=df_query, 
                                                   feature_types=["active_power"], 
                                                   turbine_ids=[tid])
            
            # Check if we have valid data
            if not ws_data.empty and not power_data.empty:
                ws_col = f"wind_speed_{tid}"
                power_col = f"active_power_{tid}"
                
                # Apply window range flag
                flag = filters.window_range_flag(
                    window_col=ws_data[ws_col],
                    window_start=5., 
                    window_end=40.,
                    value_col=power_data[power_col],
                    value_min=20., 
                    value_max=3000.
                )
                
                # Add null check condition
                null_check = df_query.select(
                    no_nulls=pl.all_horizontal(
                        pl.col(ws_col).is_not_null(), 
                        pl.col(power_col).is_not_null()
                    )
                ).collect()["no_nulls"]
                
                window_flags.append(flag & null_check)
                
        if window_flags:  # Only stack if we have valid flags
            out_of_window = np.stack(window_flags, axis=1)
            np.save(os.path.join(DATA_DIR, "out_of_window.npy"), out_of_window)
            logging.info(f"Saved window flags with shape {out_of_window.shape}")
        else:
            logging.warning("No valid data found for window range filtering")
            out_of_window = None
    else:
        out_of_window = np.load(os.path.join(DATA_DIR, "out_of_window.npy"))
        logging.info(f"Loaded out_of_window shape: {out_of_window.shape}")

    if out_of_window is not None:
        # Create turbine ID mapping
        turbine_id_mapping = {f"{int(tid):03d}": tid for tid in data_loader.turbine_ids}
        
        # Define mask function
        def safe_mask(tid):
            try:
                idx = data_loader.turbine_ids.index(turbine_id_mapping.get(tid, tid))
                mask_array = ~out_of_window[:, idx]
                logging.info(f"Mask for turbine {tid} excludes {np.sum(~mask_array)} out of {len(mask_array)} data points")
                return mask_array
            except (KeyError, ValueError) as e:
                logging.error(f"Mask error for turbine {tid}: {str(e)}")
                return None

        mask = safe_mask
        features = ws_cols

        if PLOT:
            DataInspector.print_pc_unfiltered_vals(df_query, features, mask)
            DataInspector.plot_filtered_vs_unfiltered(df_query, mask, features, ["wind_speed"], ["Wind Speed [m/s]"])

        # Apply filtering
        threshold = 0.01
        df_query = data_filter.conditional_filter(df_query, threshold, mask, features)
    else:
        logging.warning("Skipping window range filtering due to no valid data")

    # %%
    logging.info("Nullifying wind speed-power curve bin-outlier cells.")
    # apply a bin filter to remove data with power values outside of an envelope around median power curve at each wind speed
    if RELOAD_DATA or not os.path.exists(os.path.join(DATA_DIR, "bin_outliers.npy")):
        bin_flags = []
        
        for tid in data_loader.turbine_ids:
            # Get wind speed and power data for current turbine
            ws_data = data_inspector.collect_data(df=df_query, 
                                                feature_types=["wind_speed"], 
                                                turbine_ids=[tid])
            power_data = data_inspector.collect_data(df=df_query, 
                                                   feature_types=["active_power"], 
                                                   turbine_ids=[tid])
            
            # Check if we have valid data
            if not ws_data.empty and not power_data.empty:
                ws_col = f"wind_speed_{tid}"
                power_col = f"active_power_{tid}"
                
                # Apply bin filter
                try:
                    flag = filters.bin_filter(
                        bin_col=power_data[power_col],
                        value_col=ws_data[ws_col],
                        bin_width=50,
                        threshold=3,
                        center_type="median",
                        bin_min=20.,
                        bin_max=0.90 * power_data[power_col].max(),
                        threshold_type="scalar",
                        direction="below"
                    )
                    
                    # Add null check condition
                    null_check = df_query.select(
                        no_nulls=pl.all_horizontal(
                            pl.col(ws_col).is_not_null(),
                            pl.col(power_col).is_not_null()
                        )
                    ).collect()["no_nulls"]
                    
                    bin_flags.append(flag & null_check)
                    logging.info(f"Successfully processed bin filter for turbine {tid}")
                except Exception as e:
                    logging.warning(f"Failed to process bin filter for turbine {tid}: {str(e)}")
                    continue
                
        if bin_flags:  # Only stack if we have valid flags
            bin_outliers = np.stack(bin_flags, axis=1)
            np.save(os.path.join(DATA_DIR, "bin_outliers.npy"), bin_outliers)
            logging.info(f"Saved bin outlier flags with shape {bin_outliers.shape}")
        else:
            logging.warning("No valid data found for bin filtering")
            bin_outliers = None
    else:
        bin_outliers = np.load(os.path.join(DATA_DIR, "bin_outliers.npy"))
        logging.info(f"Loaded bin_outliers shape: {bin_outliers.shape}")

    if bin_outliers is not None:
        # Create turbine ID mapping
        turbine_id_mapping = {f"{int(tid):03d}": tid for tid in data_loader.turbine_ids}
        
        # Define mask function
        def safe_mask(tid):
            try:
                idx = data_loader.turbine_ids.index(turbine_id_mapping.get(tid, tid))
                mask_array = ~bin_outliers[:, idx]
                logging.info(f"Mask for turbine {tid} excludes {np.sum(~mask_array)} out of {len(mask_array)} data points")
                return mask_array
            except (KeyError, ValueError) as e:
                logging.error(f"Mask error for turbine {tid}: {str(e)}")
                return None

        mask = safe_mask
        features = ws_cols

        if PLOT:
            DataInspector.print_pc_unfiltered_vals(df_query, features, mask)
            DataInspector.plot_filtered_vs_unfiltered(df_query, mask, features, ["wind_speed"], ["Wind Speed [m/s]"])

            # Plot values outside the power-wind speed bin filter
            plot.plot_power_curve(
                data_inspector.collect_data(df=df_query, feature_types="wind_speed").to_numpy().flatten(),
                data_inspector.collect_data(df=df_query, feature_types="active_power").to_numpy().flatten(),
                flag=bin_outliers.flatten(),
                flag_labels=("Anomalous Data", "Normal Wind Speed Sensor Operation"),
                xlim=(-1, 15),
                ylim=(-100, 3000),
                legend=True,
                scatter_kwargs=dict(alpha=0.4, s=10)
            )

        # Apply filtering
        threshold = 0.01
        df_query = data_filter.conditional_filter(df_query, threshold, mask, features)
    else:
        logging.warning("Skipping bin filtering due to no valid data")

    # %%
    if PLOT:
        logging.info("Power curve fitting.")
        try:
            # Get unpivoted data with correct column names - do this in chunks
            df_unpivoted = DataInspector.unpivot_dataframe(
                df_query, 
                feature_types=["wind_speed", "active_power"]
            )
            
            # Add safety checks
            if df_unpivoted is not None:
                # Collect and filter data safely
                df_filtered = df_unpivoted.select(
                    "wind_speed",
                    "active_power"
                ).filter(
                    pl.all_horizontal(pl.all().is_not_null())
                )
                
                # Convert to pandas in a controlled way
                df_pandas = df_filtered.collect().to_pandas()
                
                if len(df_pandas) > 0:
                    # Rename columns to match what the power curve functions expect
                    df_pandas = df_pandas.rename(columns={"active_power": "power_output"})
                    
                    logging.info(f"Fitting power curves with {len(df_pandas)} valid data points")
                    
                    # Fit the curves with error handling
                    try:
                        iec_curve = power_curve.IEC(
                            windspeed_col="wind_speed", 
                            power_col="power_output",
                            data=df_pandas
                        )

                        l5p_curve = power_curve.logistic_5_parametric(
                            windspeed_col="wind_speed", 
                            power_col="power_output",
                            data=df_pandas
                        )

                        # Convert sparse matrix to dense if needed
                        try:
                            spline_curve = power_curve.gam(
                                windspeed_col="wind_speed", 
                                power_col="power_output",
                                data=df_pandas,
                                n_splines=20
                            )
                        except AttributeError:
                            # If sparse matrix error occurs, try with fewer splines
                            logging.warning("Sparse matrix error occurred, trying with fewer splines")
                            spline_curve = power_curve.gam(
                                windspeed_col="wind_speed", 
                                power_col="power_output",
                                data=df_pandas,
                                n_splines=10
                            )

                        # Plot the results
                        fig, ax = plt.subplots(figsize=(12, 8))
                        
                        # Plot scatter points
                        ax.scatter(
                            df_pandas["wind_speed"],
                            df_pandas["power_output"],
                            alpha=0.1,
                            s=10,
                            label="Measurements"
                        )

                        # Plot fitted curves
                        x = np.linspace(0, 20, 100)
                        ax.plot(x, iec_curve(x), color="red", label="IEC", linewidth=3)
                        ax.plot(x, spline_curve(x), color="green", label="Spline", linewidth=3)
                        ax.plot(x, l5p_curve(x), color="blue", label="L5P", linewidth=3)

                        ax.set_xlabel("Wind Speed (m/s)")
                        ax.set_ylabel("Power Output (kW)")
                        ax.set_title("Power Curve Fitting")
                        ax.legend()
                        ax.grid(True, alpha=0.3)

                        plt.tight_layout()
                        plt.show()
                        
                    except Exception as e:
                        logging.error(f"Error during power curve fitting: {str(e)}")
                else:
                    logging.warning("No valid data points for power curve fitting after filtering")
            else:
                logging.warning("No data available for power curve fitting")
                
        except Exception as e:
            logging.error(f"Error during data preparation for power curve fitting: {str(e)}")

    # %%
    if False:
        logging.info("Nullifying unresponsive sensor cells.")
        # find stuck sensor measurements for each turbine and set them to null
        frozen_thresholds = [(data_loader.ffill_limit * i) + 1 for i in range(1, 19)]
        print(frozen_thresholds)
        ws_pcs = []
        wd_pcs = []
        pwr_pcs = []
        for thr in frozen_thresholds:
            ws_frozen_sensor = filters.unresponsive_flag(data=DataInspector.collect_data(df=df_query, feature_types="wind_speed"), threshold=thr).values
            wd_frozen_sensor = filters.unresponsive_flag(data=DataInspector.collect_data(df=df_query, feature_types="wind_direction"), threshold=thr).values
            pwr_frozen_sensor = filters.unresponsive_flag(data=DataInspector.collect_data(df=df_query, feature_types="power_output"), threshold=thr).values

            # check if wind speed/dir measurements from inoperational turbines differ from fully operational
            print(f"For a threshold of {thr} for the frozen sensor filters:")
            ws_pcs.append(DataInspector.print_pc_unfiltered_vals(df_query, ws_cols, lambda tid: ws_frozen_sensor[:, data_loader.turbine_ids.index(tid)]))
            wd_pcs.append(DataInspector.print_pc_unfiltered_vals(df_query, wd_cols, lambda tid: wd_frozen_sensor[:, data_loader.turbine_ids.index(tid)]))
            pwr_pcs.append(DataInspector.print_pc_unfiltered_vals(df_query, pwr_cols, lambda tid: pwr_frozen_sensor[:, data_loader.turbine_ids.index(tid)]))

        if PLOT:
            fig, ax = plt.subplots(3, 1, sharex=True)
            for t_idx in range(len(data_loader.turbine_ids)):
                ax[0].scatter(x=frozen_thresholds, y=[x[1][t_idx] for x in ws_pcs], label=data_loader.turbine_ids[t_idx])
                ax[1].scatter(x=frozen_thresholds, y=[x[1][t_idx] for x in wd_pcs], label=data_loader.turbine_ids[t_idx])
                ax[2].scatter(x=frozen_thresholds, y=[x[1][t_idx] for x in pwr_pcs],label=data_loader.turbine_ids[t_idx])

            h, l = ax[0].get_legend_handles_labels()
            ax[0].legend(h[:len(data_loader.turbine_ids)], l[:len(data_loader.turbine_ids)])
            ax[0].set_title("Percentage of Unfrozen Wind Speed Measurements")
            ax[1].set_title("Percentage of Unfrozen Wind Direction Measurements")
            ax[2].set_title("Percentage of Unfrozen Power Output Measurements")

        thr = data_loader.ffill_limit + 1
        features = ws_cols + wd_cols
        # find values of wind speed/direction, where there are duplicate values with nulls inbetween
        mask = lambda tid: ~frozen_sensor[:, data_loader.turbine_ids.index(tid)]

        df_query.select([pl.col(feat).is_not_null().name.suffix("_not_null") for feat in ws_cols])\
                                        .collect(streaming=True).to_pandas()
        
        ws_frozen_sensor = (filters.unresponsive_flag(data=DataInspector.collect_data(df=df_query, feature_types="wind_speed"), threshold=thr)
                                        & df_query.select([pl.col(feat).is_not_null().name.suffix("_not_null") for feat in ws_cols])\
                                        .collect(streaming=True).to_pandas()
                                        ).values
        ws_frozen_sensor
        wd_frozen_sensor = np.stack([(filters.unresponsive_flag(data=DataInspector.collect_data(df=df_query, feature_types="wind_direction"), threshold=thr)
                                        & df_query.select(no_nulls=pl.col(f"wind_speed_{tid}").is_not_null())\
                                        .collect(streaming=True).to_pandas()["no_nulls"]
                                        ).values for tid in data_loader.turbine_ids], axis=1)
        pwr_frozen_sensor = np.stack([(filters.unresponsive_flag(data=DataInspector.collect_data(df=df_query, feature_types="power_output"), threshold=thr)
                                        & df_query.select(no_nulls=pl.col(f"wind_speed_{tid}").is_not_null())\
                                        .collect(streaming=True).to_pandas()["no_nulls"]
                                        ).values for tid in data_loader.turbine_ids], axis=1)

        wd_frozen_sensor = filters.unresponsive_flag(data=DataInspector.collect_data(df=df_query, feature_types="wind_direction"), threshold=thr).values
        pwr_frozen_sensor = filters.unresponsive_flag(data=DataInspector.collect_data(df=df_query, feature_types="power_output"), threshold=thr).values

        ws_mask = lambda tid: ~ws_frozen_sensor[:, data_loader.turbine_ids.index(tid)]
        wd_mask = lambda tid: ~wd_frozen_sensor[:, data_loader.turbine_ids.index(tid)]
        pwr_mask = lambda tid: ~pwr_frozen_sensor[:, data_loader.turbine_ids.index(tid)]


        if PLOT:
            plot.plot_power_curve(
                data_inspector.collect_data(df=df_query, feature_types="wind_speed"),
                data_inspector.collect_data(df=df_query, feature_types="power_output"),
                flag=ws_frozen_sensor,
                flag_labels=(f"Wind Speed Unresponsive Sensors (n={ws_frozen_sensor.sum():,.0f})", "Normal Turbine Operations"),
                xlim=(-1, 15),  # optional input for refining plots
                ylim=(-100, 3000),  # optional input for refining plots
                legend=True,  # optional flag for adding a legend
                scatter_kwargs=dict(alpha=0.4, s=10)  # optional input for refining plots
            )

            plot.plot_power_curve(
                data_inspector.collect_data(df=df_query, feature_types="wind_speed"),
                data_inspector.collect_data(df=df_query, feature_types="power_output"),
                flag=wd_frozen_sensor,
                flag_labels=(f"Wind Direction Unresponsive Sensors (n={wd_frozen_sensor.sum():,.0f})", "Normal Turbine Operations"),
                xlim=(-1, 15),  # optional input for refining plots
                ylim=(-100, 3000),  # optional input for refining plots
                legend=True,  # optional flag for adding a legend
                scatter_kwargs=dict(alpha=0.4, s=10)  # optional input for refining plots
            )

            plot.plot_power_curve(
                data_inspector.collect_data(df=df_query, feature_types="wind_speed"),
                data_inspector.collect_data(df=df_query, feature_types="power_output"),
                flag=pwr_frozen_sensor,
                flag_labels=(f"Power Output Unresponsive Sensors (n={pwr_frozen_sensor.sum():,.0f})", "Normal Turbine Operations"),
                xlim=(-1, 15),  # optional input for refining plots
                ylim=(-100, 3000),  # optional input for refining plots
                legend=True,  # optional flag for adding a legend
                scatter_kwargs=dict(alpha=0.4, s=10)  # optional input for refining plots
            )

        # change the values corresponding to frozen sensor measurements to null or interpolate (instead of dropping full row, since other sensors could be functioning properly)
        # fill stuck sensor measurements with Null st they are marked for interpolation later,
        threshold = 0.01
        df_query = data_filter.conditional_filter(df_query, threshold, mask, features)

        del frozen_sensor

    # %%
    logging.info("Split dataset during time steps for which many turbines have missing data.")
    
    # if there is a short or long gap for some turbines, impute them using the imputing.impute_all_assets_by_correlation function
    #       else if there is a short or long gap for many turbines, split the dataset
    missing_col_thr = max(1, int(len(data_loader.turbine_ids) * 0.05))
    missing_duration_thr = np.timedelta64(5, "m")
    minimum_not_missing_duration = np.timedelta64(20, "m")
    missing_data_cols = ["wind_speed", "wind_direction", "nacelle_direction"]

    # check for any periods of time for which more than 'missing_col_thr' features have missing data
    df_query2 = df_query.with_columns([
        pl.col(col).is_null().name.prefix("is_missing_") 
        for col in df_query.collect_schema().names() 
        if any(feat in col for feat in missing_data_cols)
    ])

    # Ensure we have at least one column before doing sum_horizontal
    if not any(col.startswith("is_missing_") for col in df_query2.collect_schema().names()):
        logging.warning("No missing data columns found to analyze")
        df_query_not_missing = df_query2  # Just continue with original data
    else:
        # subset of data, indexed by time, which has <= the threshold number of missing columns
        df_query_not_missing_times = add_df_continuity_columns(
            df_query2, 
            mask=pl.sum_horizontal([
                pl.col(col) for col in df_query2.collect_schema().names() 
                if col.startswith("is_missing_")
            ]) <= missing_col_thr
        )

        # subset of data, indexed by time, which has > the threshold number of missing columns
        df_query_missing_times = add_df_continuity_columns(
            df_query2, 
            mask=pl.sum_horizontal([
                pl.col(col) for col in df_query2.collect_schema().names() 
                if col.startswith("is_missing_")
            ]) > missing_col_thr
        )

        # start times, end times, and durations of each of the continuous subsets
        df_query_not_missing = add_df_agg_continuity_columns(df_query_not_missing_times)
        df_query_missing = add_df_agg_continuity_columns(df_query_missing_times)

        if df_query_not_missing is None or df_query_missing is None:
            logging.warning("No valid data found after continuity analysis")
            df_query_not_missing = df_query2  # Just continue with original data

    # %%
    logging.info("Impute/interpolate turbine missing dta from correlated measurements.")
    # else, for each of those split datasets, impute the values using the imputing.impute_all_assets_by_correlation function
    # fill data on single concatenated dataset
    df_query2 = data_filter._fill_single_missing_dataset(df_idx=0, df=df_query, impute_missing_features=["wind_speed", "wind_direction"], 
                                            interpolate_missing_features=["wind_speed", "wind_direction", "nacelle_direction"], 
                                            available_features=data_loader.available_features, parallel="turbine_id")

    df_query = df_query.drop(cs.starts_with("wind_speed"), cs.starts_with("wind_direction")).join(df_query2, on="time", how="left")

    # %%
    # Nacelle Calibration 
    # Find and correct wind direction offsets from median wind plant wind direction for each turbine
    logging.info("Subtracting median wind direction from wind direction and nacelle direction measurements.")

    # add the 3 degrees back to the wind direction signal
    offset = 0.0
    df_query2 = df_query.with_columns((cs.starts_with("wind_direction") + offset % 360.0))
    df_query_10min = df_query2\
                        .with_columns(pl.col("time").dt.round(f"{10}m").alias("time"))\
                        .group_by("time").agg(cs.numeric().drop_nulls().mean()).sort("time")

    wd_median = np.nanmedian(df_query_10min.select(cs.starts_with("wind_direction")).collect().to_numpy(), axis=1)
    wd_median = np.degrees(np.arctan2(np.sin(np.radians(wd_median)), np.cos(np.radians(wd_median))))
    wd_median = df_query_10min.select("time", cs.starts_with("wind_direction"), cs.starts_with("power_output")).with_columns(wd_median=wd_median)

    yaw_median = np.nanmedian(df_query_10min.select(cs.starts_with("nacelle_direction")).collect().to_numpy(), axis=1)
    yaw_median = np.degrees(np.arctan2(np.sin(np.radians(yaw_median)), np.cos(np.radians(yaw_median))))
    yaw_median = df_query_10min.select("time", cs.starts_with("nacelle_direction"), cs.starts_with("power_output")).with_columns(yaw_median=yaw_median)

    plot_wind_offset(df_query_10min, wd_median, "Original")

    df_offsets = {"turbine_id": [], "northing_bias": []}

    # remove biases from median direction
    for turbine_id in data_loader.turbine_ids:
        df = df_query_10min.filter(pl.col(f"power_output_{turbine_id}") >= 0)\
                    .select("time", f"wind_direction_{turbine_id}", f"nacelle_direction_{turbine_id}").collect()
                    
        wd_bias = DataFilter.wrap_180(DataFilter.circ_mean(
            (df.select(f"wind_direction_{turbine_id}")
                                        - wd_median.filter(pl.col(f"power_output_{turbine_id}") >= 0).select(f"wd_median").collect()).to_numpy().flatten()))
        yaw_bias = DataFilter.wrap_180(DataFilter.circ_mean(
            (df.select(f"nacelle_direction_{turbine_id}")
                                        - yaw_median.filter(pl.col(f"power_output_{turbine_id}") >= 0).select(f"yaw_median").collect()).to_numpy().flatten()))

        df_offsets["turbine_id"].append(turbine_id)
        bias = -0.5 * (wd_bias + yaw_bias)
        df_offsets["northing_bias"].append(np.round(bias, 2))
        
        df_query_10min = df_query_10min.with_columns((pl.col(f"wind_direction_{turbine_id}") + bias) % 360.0, (pl.col(f"nacelle_direction_{turbine_id}") + bias) % 360.0)
        df_query2 = df_query2.with_columns((pl.col(f"wind_direction_{turbine_id}") + bias) % 360.0, (pl.col(f"nacelle_direction_{turbine_id}") + bias) % 360.0)

        print(f"Turbine {turbine_id} bias from median wind direction: {df_offsets['northing_bias'][-1]} deg.")

    df_offsets = pl.DataFrame(df_offsets)

    plot_wind_offset(df_query_10min, wd_median, "Corrected")

    # make sure we have corrected the bias between wind direction and yaw position by adding 3 deg. to the wind direction
    bias = 0
    for turbine_id in data_loader.turbine_ids:
        df = df_query_10min.filter(pl.col(f"power_output_{turbine_id}") >= 0)\
                    .select("time", f"wind_direction_{turbine_id}", f"nacelle_direction_{turbine_id}")
                    
        bias += DataFilter.wrap_180(DataFilter.circ_mean(df.select(pl.col(f"wind_direction_{turbine_id}") - pl.col(f"nacelle_direction_{turbine_id}")).collect().to_numpy().flatten()))
        
    print(f"Average Bias = {bias / len(data_loader.turbine_ids)} deg")

    # Find offset to true North using wake loss profiles
    logging.info("Finding offset to true North using wake loss profiles.")

    # Find offsets between direction of alignment between pairs of turbines 
    # and direction of peak wake losses. Use the average offset found this way 
    # to identify the Northing correction that should be applied to all turbines 
    # in the wind farm.
    
    fi = FlorisModel(data_inspector.farm_input_filepath)

    dir_offsets = compute_offsets(df_query_10min)

    # Apply Northing offset to each turbine
    for turbine_id in data_loader.turbine_ids:
        df_query_10min = df_query_10min.with_columns((pl.col(f"wind_direction_{turbine_id}") - np.mean(dir_offsets)) % 360)\
                            .with_columns((pl.col(f"nacelle_direction_{turbine_id}") - np.mean(dir_offsets)) % 360)
        
        df_query2 = df_query2.with_columns((pl.col(f"wind_direction_{turbine_id}") - np.mean(dir_offsets)) % 360)\
                            .with_columns((pl.col(f"nacelle_direction_{turbine_id}") - np.mean(dir_offsets)) % 360)

    # Determine final wind direction correction for each turbine
    df_offsets = df_offsets.with_columns(northing_bias=DataFilter.wrap_180(df_offsets.select("northing_bias").to_numpy().flatten() + np.mean(dir_offsets)).round(2))

    # verify that Northing calibration worked properly
    new_dir_offsets = compute_offsets(df_query_10min)

    df_query = df_query2

    # %%
    # Normalization & Feature Selection
    logging.info("Normalizing and selecting features.")
    df_query = df_query\
            .with_columns(((cs.starts_with("wind_direction") - 180.).sin()).name.map(lambda c: "wd_sin_" + c.split("_")[-1]),
                        ((cs.starts_with("wind_direction") - 180.).cos()).name.map(lambda c: "wd_cos_" + c.split("_")[-1]))\
            .with_columns(**{f"ws_horz_{tid}": (pl.col(f"wind_speed_{tid}") * pl.col(f"wd_sin_{tid}")) for tid in data_loader.turbine_ids})\
            .with_columns(**{f"ws_vert_{tid}": (pl.col(f"wind_speed_{tid}") * pl.col(f"wd_cos_{tid}")) for tid in data_loader.turbine_ids})\
            .with_columns(**{f"nd_cos_{tid}": ((pl.col(f"nacelle_direction_{tid}") - 180.).cos()) for tid in data_loader.turbine_ids})\
            .with_columns(**{f"nd_sin_{tid}": ((pl.col(f"nacelle_direction_{tid}") - 180.).sin()) for tid in data_loader.turbine_ids})\
            .drop(cs.starts_with("wind_speed"), cs.starts_with("wind_direction"), cs.starts_with("wd_sin"), cs.starts_with("wd_cos"), cs.starts_with("nacelle_direction"))\
            .select(pl.col("time"), pl.col("continuity_group"), cs.contains("nd"), cs.contains("ws"))

    # store min/max of each column to rescale later
    is_numeric = (cs.contains("ws") | cs.contains("nd"))
    norm_vals = DataInspector.unpivot_dataframe(df_query, feature_types=["nd_cos", "nd_sin", "ws_vert", "ws_horz"]).select(is_numeric.min().round(2).name.suffix("_min"),
                                                                                                                            is_numeric.max().round(2).name.suffix("_max"))
    norm_vals.collect().write_csv(os.path.join(os.path.dirname(PL_SAVE_PATH), "normalization_consts.csv"))

    df_query = DataInspector.pivot_dataframe(DataInspector.unpivot_dataframe(df_query, feature_types=["nd_cos", "nd_sin", "ws_vert", "ws_horz"])\
                .select(pl.col("time"), pl.col("continuity_group"), pl.col("turbine_id"), (is_numeric - is_numeric.min().round(2)) / (is_numeric.max().round(2) - is_numeric.min().round(2)))
                )

    df_query.collect().write_parquet(os.path.join(DATA_DIR, "normalized_data.parquet"))