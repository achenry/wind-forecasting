import argparse
import logging
from memory_profiler import profile
import os
import torch
import gc
import random
import numpy as np
from types import SimpleNamespace

import polars as pl
# import wandb
from lightning.pytorch.loggers import WandbLogger
# wandb.login()
# wandb.login(relogin=True)
import yaml

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
from pytorch_transformer_ts.tactis_2.estimator import TACTiS2Estimator as TactisEstimator
from pytorch_transformer_ts.tactis_2.lightning_module import TACTiS2LightningModule as TactisLightningModule
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
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run a model on a dataset")
    parser.add_argument("--config", type=str, help="Path to config file", default="examples/inputs/training_inputs_aoifemac_flasc.yaml")
    parser.add_argument("-md", "--mode", choices=["tune", "train", "test"], required=True,
                        help="Mode to run: 'tune' for hyperparameter optimization with Optuna, 'train' to train a model, 'test' to evaluate a model")
    parser.add_argument("-chk", "--checkpoint", type=str, required=False, default=None)
    parser.add_argument("-m", "--model", type=str, choices=["informer", "autoformer", "spacetimeformer", "tactis"], required=True)
    parser.add_argument("-rt", "--restart_tuning", action="store_true")
    parser.add_argument("-tp", "--use_tuned_parameters", action="store_true")
    parser.add_argument("--tune_first", action="store_true", help="Whether to use tuned parameters", default=False)
    parser.add_argument("--model_path", type=str, help="Path to a saved model checkpoint to load from", default=None)
    parser.add_argument("--predictor_path", type=str, help="Path to a saved predictor for evaluation", default=None)
    parser.add_argument("--seed", type=int, help="Seed for random number generator", default=42)
    parser.add_argument("--save_to", type=str, help="Path to save the predicted output", default=None)

    args = parser.parse_args()
    
    # %% SETUP SEED
    logging.info(f"Setting random seed to {args.seed}")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # %% PARSE CONFIG
    logging.info(f"Parsing configuration from yaml and command line arguments")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    # TODO create function to check config params and set defaults
    assert args.checkpoint is None or args.checkpoint in ["best", "latest"] or os.path.exists(args.checkpoint), "Checkpoint argument, if provided, must equal 'best', 'latest', or an existing checkpoint path."
    # set number of devices/number of nodes based on environment variables
    if "SLURM_NTASKS_PER_NODE" in os.environ:
        config["trainer"]["devices"] = int(os.environ["SLURM_NTASKS_PER_NODE"])
    if "SLURM_NNODES" in os.environ:
        config["trainer"]["num_nodes"] = int(os.environ["SLURM_NNODES"])
    
    if (type(config["dataset"]["target_turbine_ids"]) is str) and (
        (config["dataset"]["target_turbine_ids"].lower() == "none") or (config["dataset"]["target_turbine_ids"].lower() == "all")):
        config["dataset"]["target_turbine_ids"] = None # select all turbines

    # %% SETUP LOGGING
    logging.info("Setting up logging")
    if not os.path.exists(config["experiment"]["log_dir"]):
        os.makedirs(config["experiment"]["log_dir"])
        
    # Get worker info from environment variables
    worker_id = os.environ.get('SLURM_PROCID', '0')
    gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0')

    # Create a unique run name for each worker
    run_name = f"{config['experiment']['run_name']}_worker{worker_id}_gpu{gpu_id}"

    wandb_logger = WandbLogger(
        project="wind_forecasting",
        name=run_name,  # Use the unique run name
        log_model="all",
        # offline=True,
        save_dir=config["experiment"]["log_dir"],
        group=config['experiment']['run_name'],  # Group all workers under the same experiment group
    )
    wandb_logger.log_hyperparams(config)
    config["trainer"]["logger"] = wandb_logger

    # Early in the function, ensure all log directories exist
    log_dir = config.experiment.log_dir
    
    # Create all logging directories
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up wandb directory
    wandb_dir = config.logging.wandb_dir if hasattr(config, 'logging') and hasattr(config.logging, 'wandb_dir') else os.path.join(log_dir, "wandb")
    os.makedirs(wandb_dir, exist_ok=True)
    os.environ["WANDB_DIR"] = wandb_dir
    
    # Set up optuna directory 
    optuna_dir = config.logging.optuna_dir if hasattr(config, 'logging') and hasattr(config.logging, 'optuna_dir') else os.path.join(log_dir, "optuna")
    os.makedirs(optuna_dir, exist_ok=True)
    
    # Set up checkpoint directory
    checkpoint_dir = config.logging.checkpoint_dir if hasattr(config, 'logging') and hasattr(config.logging, 'checkpoint_dir') else os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Update config with these directories if they're not already set
    if not hasattr(config, 'logging'):
        config.logging = SimpleNamespace()
        config.logging.wandb_dir = wandb_dir
        config.logging.optuna_dir = optuna_dir
        config.logging.checkpoint_dir = checkpoint_dir
    
    # Ensure optuna journal_dir is set correctly
    if hasattr(config, 'optuna'):
        config.optuna.journal_dir = optuna_dir
    
    # Ensure trainer default_root_dir is set correctly
    if hasattr(config, 'trainer'):
        config.trainer.default_root_dir = checkpoint_dir
    
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
    if args.mode in ["train", "test"]:
        from wind_forecasting.run_scripts.tuning import get_tuned_params
        if args.use_tuned_parameters:
            try:
                logging.info("Getting tuned parameters")
                tuned_params = get_tuned_params(use_rdb=config["optuna"]["use_rdb"], study_name=f"tuning_{args.model}")
                logging.info(f"Declaring estimator {args.model.capitalize()} with tuned parameters")
                config["dataset"].update({k: v for k, v in tuned_params.items() if k in config["dataset"]})
                config["model"][args.model].update({k: v for k, v in tuned_params.items() if k in config["model"][args.model]})
                config["trainer"].update({k: v for k, v in tuned_params.items() if k in config["trainer"]})
            except FileNotFoundError as e:
                logging.warning(e)
                logging.info(f"Declaring estimator {args.model.capitalize()} with default parameters")
        else:
            logging.info(f"Declaring estimator {args.model.capitalize()} with default parameters")
        
        # Use globals() to fetch the estimator class dynamically (Aoife comment #2)
        EstimatorClass = globals()[f"{args.model.capitalize()}Estimator"]
        estimator = EstimatorClass(
            freq=data_module.freq, 
            prediction_length=data_module.prediction_length,
            num_feat_dynamic_real=data_module.num_feat_dynamic_real, 
            num_feat_static_cat=data_module.num_feat_static_cat,
            cardinality=data_module.cardinality,
            num_feat_static_real=data_module.num_feat_static_real,
            input_size=data_module.num_target_vars,
            scaling=False,
            time_features=[second_of_minute, minute_of_hour, hour_of_day, day_of_year],
            distr_output=globals()[config["model"]["distr_output"]["class"]](dim=data_module.num_target_vars, **config["model"]["distr_output"]["kwargs"]),
            
            batch_size=config["dataset"].setdefault("batch_size", 128),
            num_batches_per_epoch=config["trainer"].setdefault("limit_train_batches", 50),
            context_length=config["dataset"]["context_length"],
            train_sampler=ExpectedNumInstanceSampler(num_instances=1.0, min_past=config["dataset"]["context_length"], min_future=data_module.prediction_length),
            validation_sampler=ValidationSplitSampler(min_past=config["dataset"]["context_length"], min_future=data_module.prediction_length),
            trainer_kwargs=config["trainer"],
            **config["model"][args.model]
        )

    if args.mode == "tune":
        logging.info("Starting Optuna hyperparameter tuning...")
        # Explicitly set which GPU to use based on environment or worker ID
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            device_id = int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0])
            logging.info(f"Using GPU device {device_id}")
            
        # Clear GPU memory before starting
        torch.cuda.empty_cache()
        gc.collect()
        
        # Enhanced GPU memory logging
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            logging.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
            logging.info(f"GPU Memory: {torch.cuda.memory_allocated(device)/1e9:.2f}GB / {torch.cuda.get_device_properties(device).total_memory/1e9:.2f}GB")
        
        # %% TUNE MODEL WITH OPTUNA
        from wind_forecasting.run_scripts.tuning import tune_model
        if not os.path.exists(config["optuna"]["journal_dir"]):
            os.makedirs(config["optuna"]["journal_dir"]) 

        # Use globals() to fetch the module and estimator classes dynamically (Aoife comment #3)
        LightningModuleClass = globals()[f"{args.model.capitalize()}LightningModule"]
        EstimatorClass = globals()[f"{args.model.capitalize()}Estimator"]
        DistrOutputClass = globals()[config["model"]["distr_output"]["class"]]
        
        def handle_trial_with_oom_protection(tuning_objective, trial):
            try:
                # Log GPU memory at the start of each trial
                if torch.cuda.is_available():
                    device = torch.cuda.current_device()
                    logging.info(f"Trial {trial.number} - Starting GPU Memory: {torch.cuda.memory_allocated(device)/1e9:.2f}GB / {torch.cuda.get_device_properties(device).total_memory/1e9:.2f}GB")
                
                result = tuning_objective(trial)
                
                # Log GPU memory after trial completes
                if torch.cuda.is_available():
                    device = torch.cuda.current_device()
                    logging.info(f"Trial {trial.number} - Ending GPU Memory: {torch.cuda.memory_allocated(device)/1e9:.2f}GB / {torch.cuda.get_device_properties(device).total_memory/1e9:.2f}GB")
                    
                # Add explicit garbage collection after each trial
                gc.collect()
                return result
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logging.warning(f"Trial {trial.number} failed with CUDA OOM error")
                    if torch.cuda.is_available():
                        logging.warning(f"OOM at memory usage: {torch.cuda.memory_allocated()/1e9:.2f}GB")
                    # Force garbage collection
                    torch.cuda.empty_cache()
                    gc.collect()
                    # Return a very poor score
                    return float('inf') if config["optuna"]["direction"] == "minimize" else float('-inf')
                raise e
        
        # Pass the OOM protection wrapper to tune_model
        tune_model(model=args.model, config=config, 
                    lightning_module_class=LightningModuleClass, 
                    estimator_class=EstimatorClass,
                    distr_output_class=DistrOutputClass, 
                    data_module=data_module, 
                    max_epochs=config["optuna"]["max_epochs"],
                    limit_train_batches=config["optuna"]["limit_train_batches"],
                    metric=config["optuna"]["metric"],
                    direction=config["optuna"]["direction"],
                    context_length_choices=[int(data_module.prediction_length * i) for i in config["optuna"]["context_length_choice_factors"]],
                    n_trials=config["optuna"]["n_trials"],
                    journal_storage_dir=config["optuna"]["journal_dir"],
                    use_rdb=config["optuna"]["use_rdb"],
                    restart_study=args.restart_tuning,
                    trial_protection_callback=handle_trial_with_oom_protection)  # Add this parameter
        
        # After training completes
        torch.cuda.empty_cache()
        gc.collect()
        logging.info("Optuna hyperparameter tuning completed.")
        
    elif args.mode == "train":
        logging.info("Starting model training...")
        # %% TRAIN MODEL
        logging.info("Training model")
        estimator.train(
            training_data=data_module.train_dataset,
            validation_data=data_module.val_dataset,
            forecast_generator=DistributionForecastGenerator(estimator.distr_output),
            ckpt_path=(args.checkpoint if (args.checkpoint is not None and os.path.exists(args.checkpoint)) else None)
            # shuffle_buffer_length=1024
        )
        # train_output.trainer.checkpoint_callback.best_model_path
        logging.info("Model training completed.")
    elif args.mode == "test":
        logging.info("Starting model testing...")
        # %% TEST MODEL
        from wind_forecasting.run_scripts.testing import test_model
        from glob import glob
        import pandas as pd
        import re
        import shutil
        
        metric = "val_loss_epoch"
        mode = "min"
        
        if args.checkpoint in ["best", "latest"]:
            version_dirs = glob(os.path.join(config["trainer"]["default_root_dir"], "lightning_logs", "version_*"))
            if not version_dirs:
                raise TypeError("Must provide a valid --checkpoint argument to load from.")
        
        if args.checkpoint == "best":
            best_metric_value = float('inf') if mode == "min" else float('-inf')
            for version_dir in version_dirs:
                if not os.path.exists(os.path.join(version_dir, "metrics.csv")):
                    logging.info(f"Metrics table {os.path.join(version_dir, "metrics.csv")} does not exist, removing invalid version dir {version_dir}.")
                    shutil.rmtree(version_dir) 
                    continue
                
                metrics = pd.read_csv(os.path.join(version_dir, "metrics.csv"), index_col=None)

                best_chk_metrics = metrics.loc[metrics[metric].idxmin() if mode == "min" else metrics[metric].idxmax(), 
                                  ["epoch", "step", metric]]
                
                if (mode == "min" and best_chk_metrics[metric] < best_metric_value) \
                    or (mode == "max" and best_chk_metrics[metric] > best_metric_value):
                    checkpoint = os.path.join(version_dir, "checkpoints", 
                                            f"epoch={int(best_chk_metrics['epoch'])}-step={int(best_chk_metrics['step']) + 1}.ckpt")
                    best_metric_value = best_chk_metrics[metric] 

            if os.path.exists(checkpoint):
                logging.info(f"Found best pretrained model: {checkpoint}")
            else:
                raise FileNotFoundError(f"Best checkpoint {checkpoint} does not exist.")
                
        elif args.checkpoint == "latest":
            logging.info("Fetching latest pretrained model...")
            pattern = r'(?<=version_)\d+'
            version_dir = f"version_{max([int(re.search(pattern, vd).group(0)) for vd in version_dirs])}"
            
            checkpoint_paths = glob(os.path.join(version_dir, "checkpoints", f"*.ckpt"))
            checkpoint_stats = [(int(re.search(r'(?<=epoch=)(\d+)', cp).group(0)),
                                 int(re.search(r'(?<=step=)(\d+)', cp).group(0))) for cp in checkpoint_paths]
            checkpoint = checkpoint_paths[checkpoint_stats.index(sorted(checkpoint_stats)[-1])]
            
            if os.path.exists(checkpoint):
                logging.info(f"Found latest pretrained model: {checkpoint}")
            else:
                raise FileNotFoundError(f"Last checkpoint {checkpoint} does not exist.")
             
        elif os.path.exists(args.checkpoint):
            logging.info("Fetching pretrained model...")
            checkpoint = args.checkpoint
            logging.info(f"Found given pretrained model: {checkpoint}")
            
        test_model(data_module=data_module,
                    checkpoint=checkpoint,
                    lightning_module_class=globals()[f"{args.model.capitalize()}LightningModule"], 
                    estimator=estimator, 
                    normalization_consts_path=config["dataset"]["normalization_consts_path"])
        
        logging.info("Model testing completed.")
        
        # %% EXPORT LOGGING DATA
        # api = wandb.Api()
        # run is specified by <entity>/<project>/<run_id>
        # run = api.run("aoife-henry-university-of-colorado-boulder/wind_forecasting/<run_id>")
        
        # save the metrics for the run to a csv file
        # metrics_df = run.history()
        # metrics_df.to_csv("metrics.csv")
        
        # Pull down the accuracy and timestamps for logged metric data  
        # if run.state == "finished":
        #     for i, row in metrics_df.iterrows():
        #     print(row["_timestamp"], row["accuracy"])
        
        # get unsampled metric data
        # history = run.scan_history()
    
if __name__ == "__main__":
    main()
# %%
