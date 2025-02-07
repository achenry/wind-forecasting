import sys
import argparse
import logging
from memory_profiler import profile

import polars as pl
import matplotlib
import yaml

from wind_forecasting.preprocessing.data_module import DataModule

# Configure logging and matplotlib backend
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if sys.platform == "darwin":
    matplotlib.use('TkAgg')
    
mpi_exists = False
try:
    from mpi4py import MPI
    mpi_exists = True
except:
    print("No MPI available on system.")

# @profile
def main():
    
    # %% PARSE CONFIGURATION
    # parse training/test booleans and config file from command line
    logging.info("Parsing configuration from yaml and command line arguments")
    parser = argparse.ArgumentParser(prog="WindFarmForecasting")
    parser.add_argument("-cnf", "--config", type=str)
    parser.add_argument("-rl", "--reload", action="store_true")
    # pretrained_filename = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/logging/wf_forecasting/lznjshyo/checkpoints/epoch=0-step=50.ckpt"
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config  = yaml.safe_load(file)
    
    if (type(config["dataset"]["target_turbine_ids"]) is str) and (
        (config["dataset"]["target_turbine_ids"].lower() == "none") or (config["dataset"]["target_turbine_ids"].lower() == "all")):
        config["dataset"]["target_turbine_ids"] = None # select all turbines

    # %% CREATE DATASET
    logging.info("Creating datasets")
    data_module = DataModule(data_path=config["dataset"]["data_path"], 
                             save_data_path=config["dataset"]["save_data_path"] if "save_data_path" in config["dataset"] else None, 
                             n_splits=config["dataset"]["n_splits"],
                            continuity_groups=None, train_split=(1.0 - config["dataset"]["val_split"] - config["dataset"]["test_split"]),
                                val_split=config["dataset"]["val_split"], test_split=config["dataset"]["test_split"], 
                                prediction_length=config["dataset"]["prediction_length"], context_length=config["dataset"]["context_length"],
                                target_prefixes=["ws_horz", "ws_vert"], feat_dynamic_real_prefixes=["nd_cos", "nd_sin"],
                                freq=config["dataset"]["resample_freq"], target_suffixes=config["dataset"]["target_turbine_ids"],
                                per_turbine_target=config["dataset"]["per_turbine_target"], dtype=pl.Float32)
    # if RUN_ONCE:
    data_module.generate_datasets()
    # data_module.plot_dataset_splitting()

if __name__ == "__main__":
    main()