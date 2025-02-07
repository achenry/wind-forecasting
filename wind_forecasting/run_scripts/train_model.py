import argparse
import logging
from memory_profiler import profile
import os

import polars as pl
import wandb
import yaml

from gluonts.torch.distributions import LowRankMultivariateNormalOutput
from gluonts.model.forecast_generator import DistributionForecastGenerator
from gluonts.time_feature._base import second_of_minute, minute_of_hour, hour_of_day, day_of_year
from gluonts.transform import ExpectedNumInstanceSampler, ValidationSplitSampler, SequentialSampler

from torch import set_float32_matmul_precision 
set_float32_matmul_precision('medium') # or high to trade off performance for precision

from lightning.pytorch.loggers import WandbLogger
from pytorch_transformer_ts.informer.lightning_module import InformerLightningModule
from pytorch_transformer_ts.informer.estimator import InformerEstimator
from pytorch_transformer_ts.autoformer.estimator import AutoformerEstimator
from pytorch_transformer_ts.autoformer.lightning_module import AutoformerLightningModule
from pytorch_transformer_ts.spacetimeformer.estimator import SpacetimeformerEstimator
from pytorch_transformer_ts.spacetimeformer.lightning_module import SpacetimeformerLightningModule
from wind_forecasting.preprocessing.data_module import DataModule

# Configure logging and matplotlib backend
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

mpi_exists = False
try:
    from mpi4py import MPI
    mpi_exists = True
except:
    print("No MPI available on system.")

# @profile
def main():
    
    RUN_ONCE = (mpi_exists and (MPI.COMM_WORLD.Get_rank()) == 0)
    
    # %% PARSE CONFIGURATION
    # parse training/test booleans and config file from command line
    logging.info("Parsing configuration from yaml and command line arguments")
    parser = argparse.ArgumentParser(prog="WindFarmForecasting")
    parser.add_argument("-cnf", "--config", type=str)
    parser.add_argument("-tr", "--train", action="store_true")
    parser.add_argument("-chk", "--checkpoint", type=str, required=False, default=None)
    parser.add_argument("-m", "--model", type=str, choices=["informer", "autoformer", "spacetimeformer", "tactis"], required=True)
    # pretrained_filename = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/logging/wf_forecasting/lznjshyo/checkpoints/epoch=0-step=50.ckpt"
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config  = yaml.safe_load(file)
        
    # TODO set number of devices/number of nodes based on environment variables

    # TODO create function to check config params and set defaults
    # if config["trainer"]["n_workers"] == "auto":
    #     if "SLURM_GPUS_ON_NODE" in os.environ:
    #         config["trainer"]["n_workers"] = int(os.environ["SLURM_GPUS_ON_NODE"])
    #     else:
    #         config["trainer"]["n_workers"] = mp.cpu_count()
    
    # config["trainer"]["devices"] = 'auto'
    # config["trainer"]["accelerator"] = 'auto'
    
    if (type(config["dataset"]["target_turbine_ids"]) is str) and (
        (config["dataset"]["target_turbine_ids"].lower() == "none") or (config["dataset"]["target_turbine_ids"].lower() == "all")):
        config["dataset"]["target_turbine_ids"] = None # select all turbines

    # %% SETUP LOGGING
    # logging.info("Setting up logging")
    # if not os.path.exists(config["experiment"]["log_dir"]):
    #     os.makedirs(config["experiment"]["log_dir"])
    # wandb_logger = WandbLogger(
    #     project="wf_forecasting",
    #     name=config["experiment"]["run_name"],
    #     log_model=True,
    #     save_dir=config["experiment"]["log_dir"],
    #     config=config
    # )
    # config["trainer"]["logger"] = wandb_logger

    # %% CREATE DATASET
    logging.info("Creating datasets")
    data_module = DataModule(data_path=config["dataset"]["data_path"], n_splits=config["dataset"]["n_splits"],
                            continuity_groups=None, train_split=(1.0 - config["dataset"]["val_split"] - config["dataset"]["test_split"]),
                                val_split=config["dataset"]["val_split"], test_split=config["dataset"]["test_split"], 
                                prediction_length=config["dataset"]["prediction_length"], context_length=config["dataset"]["context_length"],
                                target_prefixes=["ws_horz", "ws_vert"], feat_dynamic_real_prefixes=["nd_cos", "nd_sin"],
                                freq=config["dataset"]["resample_freq"], target_suffixes=config["dataset"]["target_turbine_ids"],
                                    per_turbine_target=config["dataset"]["per_turbine_target"], dtype=pl.Float32)
    # if RUN_ONCE:
    data_module.generate_splits()

    # %% DEFINE ESTIMATOR
    logging.info(f"Declaring estimator {args.model.capitalize()}")
    estimator = globals()[f"{args.model.capitalize()}Estimator"](
        freq=data_module.freq, 
        prediction_length=data_module.prediction_length,
        context_length=data_module.context_length,
        num_feat_dynamic_real=data_module.num_feat_dynamic_real, 
        num_feat_static_cat=data_module.num_feat_static_cat,
        cardinality=data_module.cardinality,
        num_feat_static_real=data_module.num_feat_static_real,
        input_size=data_module.num_target_vars,
        scaling=False,
        # lags_seq=[0, 1],
        batch_size=config["dataset"].setdefault("batch_size", 128),
        num_batches_per_epoch=config["trainer"].setdefault("limit_train_batches", 50), # TODO set this to be arbitrarily high st limit train_batches dominates
        # train_sampler=SequentialSampler(min_past=data_module.context_length, min_future=data_module.prediction_length), # TODO SequentialSampler = terrible results
        # validation_sampler=SequentialSampler(min_past=data_module.context_length, min_future=data_module.prediction_length),
        train_sampler=ExpectedNumInstanceSampler(num_instances=1.0, min_past=data_module.context_length, min_future=data_module.prediction_length), # TODO should be context_len + max(seq_len) to avoid padding..
        validation_sampler=ValidationSplitSampler(min_past=data_module.context_length, min_future=data_module.prediction_length),
        time_features=[second_of_minute, minute_of_hour, hour_of_day, day_of_year],
        distr_output=globals()[config["model"]["distr_output"]["class"]](dim=data_module.num_target_vars, **config["model"]["distr_output"]["kwargs"]),
        trainer_kwargs=config["trainer"],
        **config["model"][args.model]
    )

    # %% TRAIN MODEL
    logging.info("Training model")    
    predictor = estimator.train(
        training_data=data_module.train_dataset,
        validation_data=data_module.val_dataset,
        forecast_generator=DistributionForecastGenerator(estimator.distr_output),
        ckpt_path=args.checkpoint if ((not args.train) and (args.checkpoint is not None) and (os.path.exists(args.checkpoint))) else None
        # shuffle_buffer_length=1024
    )
    
if __name__ == "__main__":
    main()