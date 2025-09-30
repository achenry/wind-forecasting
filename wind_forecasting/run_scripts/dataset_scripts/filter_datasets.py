import numpy as np
from whoc.wind_field.WindField import butterworth_LPF_TFmag
from wind_forecasting.preprocessing.data_module import DataModule
import argparse
import yaml
import polars as pl
import pandas as pd
import os

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def apply_wind_filter(simulation_dt, dataset, filter_func, return_mag_dir=False):
    # FFT of raw wind direction time series
    # freq_vec_dir = np.fft.fft(simulation_dir)
    freq_vec_u = np.fft.fft(simulation_u)
    freq_vec_v = np.fft.fft(simulation_v)
    
    # fc_dir = 0.0011
    fc_mag = 0.0011
    n_lpf = 1
    ts_len = len(simulation_u)
    half_len = int(ts_len / 2)
    fs = (1 / (ts_len * simulation_dt)) * np.arange(1, half_len)
    
    # tf_dir_lpf = butterworth_LPF_TFmag(fs, fc_dir, n_lpf)
    tf_mag_lpf = filter_func(fs, fc_mag, n_lpf)

    # Apply LPF magnitude
    # freq_vec_dir[1:int(ts_len / 2)] *= tf_dir_lpf
    freq_vec_u[1:half_len] *= tf_mag_lpf
    freq_vec_v[1:half_len] *= tf_mag_lpf
    
    if ts_len % 2 == 0:
        freq_vec_u[half_len] = np.sqrt(np.max([filter_func(0.5 / simulation_dt, fc_mag, n_lpf), 0]))
        freq_vec_u[half_len + 1:] *= np.flip(tf_mag_lpf)
        freq_vec_v[half_len] = np.sqrt(np.max([filter_func(0.5 / simulation_dt, fc_mag, n_lpf), 0]))
        freq_vec_v[half_len + 1:] *= np.flip(tf_mag_lpf)
    else:
        freq_vec_u[half_len:half_len+2] = np.sqrt(np.max([filter_func(0.5 / simulation_dt, fc_mag, n_lpf), 0]))
        freq_vec_u[half_len + 2:] *= np.flip(tf_mag_lpf)
        freq_vec_v[half_len:half_len+2] = np.sqrt(np.max([filter_func(0.5 / simulation_dt, fc_mag, n_lpf), 0]))
        freq_vec_v[half_len + 2:] *= np.flip(tf_mag_lpf)

    # START TEST
    # new_simulation_u = np.real(np.fft.ifft(freq_vec_u))[TRUNCATE_STEPS:-TRUNCATE_STEPS]
    # new_simulation_v = np.real(np.fft.ifft(freq_vec_v))[TRUNCATE_STEPS:-TRUNCATE_STEPS]
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(2, 1, figsize=(10,6), sharex=True)
    # axs[0].plot(simulation_u,label="Raw Wind U")
    # axs[0].plot(new_simulation_u,linewidth=2.0,color='r',label="Low-Frequency Wind U")
    # axs[0].legend()
    # axs[1].plot(simulation_v,label="Raw Wind V")
    # axs[1].plot(new_simulation_v,linewidth=2.0,color='r',label="Low-Frequency Wind V")
    # axs[1].legend()
    # plt.grid()
    # END TEST

    # save `originals
    # all_freq_simulation_mag = ((simulation_u**2 + simulation_v**2)**0.5)#[TRUNCATE_STEPS:-TRUNCATE_STEPS]
    # all_freq_simulation_dir = (180.0 + np.rad2deg(np.arctan2(simulation_u, simulation_v)))#[TRUNCATE_STEPS:-TRUNCATE_STEPS]
    # all_freq_simulation_dir[all_freq_simulation_dir < 0] = 360. + all_freq_simulation_dir[all_freq_simulation_dir < 0]
    # all_freq_simulation_dir[all_freq_simulation_dir > 360] = np.mod(all_freq_simulation_dir[all_freq_simulation_dir > 360], 360.) 
    
    # time series of low-frequency wind direction
    simulation_u = np.real(np.fft.ifft(freq_vec_u))#[TRUNCATE_STEPS:-TRUNCATE_STEPS]
    simulation_v = np.real(np.fft.ifft(freq_vec_v))#[TRUNCATE_STEPS:-TRUNCATE_STEPS]
    
    if return_mag_dir:
        simulation_mag = (simulation_u**2 + simulation_v**2)**0.5
        simulation_dir = 180.0 + np.rad2deg(np.arctan2(simulation_u, simulation_v))
        simulation_dir[simulation_dir < 0] = 360. + simulation_dir[simulation_dir < 0]
        simulation_dir[simulation_dir > 360] = np.mod(simulation_dir[simulation_dir > 360], 360.)
        return simulation_mag, simulation_dir
    else:
        return simulation_u, simulation_v
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(prog="filter_datasets.py", description="Apply smoothing filters to SCADA data.")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-st", "--stoptime", default="auto")
    parser.add_argument("-ns", "--n_seeds", default="auto")
    parser.add_argument("-ics", "--filter_controller_signals", action="store_true")
    parser.add_argument("-m", "--multiprocessor", type=str, choices=["mpi", "cf"], help="which multiprocessing backend to use, omit for sequential processing", default=None)
    parser.add_argument("-f", "--filter", type=str, choices=["butterworth", "moving_average", "savitzky_golay"], required=True, default="butterworth")
    parser.add_argument("-mcnf", "--model_config", type=str, required=True, default="")
    parser.add_argument("-dcnf", "--data_config", type=str, required=True, default="")
    parser.add_argument("-rl", "--ram_limit", type=int, required=False, default=75)
    parser.add_argument("-rld", "--reload_data",
                        action="store_true",
                        help="Whether to reload validation all_turbine simulation time step datasets or not.")
    parser.add_argument("-rsd", "--resplit_data",
                        action="store_true",
                        help="Whether to resplit all_turbine simulation time step datasets or not.")
    args = parser.parse_args()
    
    if args.multiprocessor == "mpi":
        from mpi4py import MPI
        from mpi4py.futures import MPICommExecutor
        comm = MPI.COMM_WORLD
    
    RUN_ONCE = (args.multiprocessor == "mpi" and (comm_rank := comm.Get_rank()) == 0) or (args.multiprocessor != "mpi") or (args.multiprocessor is None)
    
    # NOTE make sure this is the model config with the highest prediction length required, for splitting
    logging.info(f"Reading model config file {args.model_config}")
    with open(args.model_config, 'r') as file:
        model_config  = yaml.safe_load(file)
        
    logging.info(f"Reading preprocessing config file {args.data_config}")
    with open(args.data_config, 'r') as file:
        data_config  = yaml.safe_load(file)
    
    if len(data_config["turbine_signature"]) == 1:
        tid2idx_mapping = {str(k): i for i, k in enumerate(data_config["turbine_mapping"][0].keys())}
    else:
        tid2idx_mapping = {str(k): i for i, k in enumerate(data_config["turbine_mapping"][0].values())} # if more than one file type was pulled from, all turbine ids will be transformed into common type
    
    turbine_signature = data_config["turbine_signature"][0] if len(data_config["turbine_signature"]) == 1 else "\\d+"
    
    measurements_timedelta = pd.Timedelta(seconds=args.simulation_timestep)
    
    data_module = DataModule(data_path=model_config["dataset"]["data_path"], 
                                normalization_consts_path=model_config["dataset"]["normalization_consts_path"],
                                use_normalization=False, 
                                n_splits=1, #model_config["dataset"]["n_splits"],
                                continuity_groups=None, train_split=(1.0 - model_config["dataset"]["val_split"] - model_config["dataset"]["test_split"]),
                                val_split=model_config["dataset"]["val_split"], test_split=model_config["dataset"]["test_split"],
                                prediction_length=model_config["dataset"]["prediction_length"], 
                                context_length=model_config["dataset"]["context_length"],
                                target_prefixes=["ws_horz", "ws_vert"], feat_dynamic_real_prefixes=["nd_cos", "nd_sin"],
                                freq=f"{int(measurements_timedelta.total_seconds())}s", 
                                target_suffixes=model_config["dataset"]["target_turbine_ids"],
                                per_turbine_target=False, as_lazyframe=True, dtype=pl.Float32,
                                workers=4,
                                pin_memory=True,
                                persistent_workers=True,
                                verbose=True)
    
    if RUN_ONCE and not os.path.exists(data_module.train_ready_data_path) or args.reload_data:
        data_module.generate_datasets()
        logging.info("Reloading datasets.")
        reload = True
    else:
        logging.info("Reading saved datasets.")
        reload = False

    # data_module.train_ready_data_path=data_module.train_ready_data_path.replace("awaken_data/", "awaken_data/test/")
    # reload = True
    data_module.generate_splits(save=True, reload=reload or args.resplit_data, splits=["train", "val", "test"])