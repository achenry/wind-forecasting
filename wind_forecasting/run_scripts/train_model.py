import argparse
import logging
from memory_profiler import profile
import os
import sys
import polars as pl
import wandb
import yaml

# Forcefully insert the project root at the beginning of sys.path.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)
sys.path.insert(0, project_root)

from gluonts.torch.distributions import LowRankMultivariateNormalOutput, StudentTOutput
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
from pytorch_transformer_ts.tactis_2.estimator import TACTiSEstimator
from pytorch_transformer_ts.tactis_2.lightning_module import TACTiSLightningModule
from wind_forecasting.preprocessing.data_module import DataModule


# Configure logging and matplotlib backend
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

mpi_exists = False
try:
    from mpi4py import MPI
    mpi_exists = True
except:
    print("No MPI available on system.")

@profile
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
    if RUN_ONCE:
        data_module.generate_datasets()
    data_module.generate_splits()

    # %% DEFINE ESTIMATOR
    if RUN_ONCE:
        logging.info(f"Declaring estimator {args.model.capitalize()}")
    if args.model == "tactis":
        EstimatorClass = TACTiSEstimator
    else:
        EstimatorClass = globals()[f"{args.model.capitalize()}Estimator"]

    estimator = EstimatorClass(
        freq=config["dataset"]["resample_freq"],
        prediction_length=config["dataset"]["prediction_length"],
        context_length=config["dataset"]["context_length"],
        num_feat_dynamic_real=len(data_module.feat_dynamic_real_cols),
        num_feat_static_real=0, # len(data_module.feat_static_real_cols),
        num_feat_static_cat=0, # len(data_module.feat_static_cat_cols),
        cardinality=[], # [len(data_module.turbine_ids)],
        batch_size=config["training"]["batch_size"],
        num_batches_per_epoch=config["training"]["num_batches_per_epoch"],
        trainer_kwargs = config["trainer"],
        model_kwargs = {
            "num_series": data_module.num_turbines if config["dataset"]["per_turbine_target"] else 1,
            "flow_series_embedding_dim": config["model"]["flow_series_embedding_dim"],
            "copula_series_embedding_dim": config["model"]["copula_series_embedding_dim"],
            "flow_input_encoder_layers": config["model"]["flow_input_encoder_layers"],
            "copula_input_encoder_layers": config["model"]["copula_input_encoder_layers"],
            "num_layers_encoder": config["model"]["num_layers_encoder"],
            "num_decoder_layers": config["model"]["num_decoder_layers"],
            "prediction_length": config["dataset"]["prediction_length"],
            "context_length": config["dataset"]["context_length"],
            "num_feat_dynamic_real": len(data_module.feat_dynamic_real_cols),
            "num_feat_static_real": 0,
            "num_feat_static_cat": 0,
            "cardinality": [],
            "distr_output": globals()[config["model"]["distr_output"]["class"]](**config["model"]["distr_output"]["kwargs"]),
            "freq": config["dataset"]["resample_freq"],
        }
    )

    # %% TRAIN MODEL
    if RUN_ONCE:
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