import argparse
import logging
from memory_profiler import profile
import os
import torch
import gc
import random
import numpy as np

import polars as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
import yaml

# Internal imports
from wind_forecasting.utils.trial_utils import handle_trial_with_oom_protection
from wind_forecasting.utils.optuna_db_utils import setup_optuna_storage

from gluonts.torch.distributions import LowRankMultivariateNormalOutput, StudentTOutput
from gluonts.model.forecast_generator import DistributionForecastGenerator
from gluonts.time_feature._base import second_of_minute, minute_of_hour, hour_of_day, day_of_year
from gluonts.transform import ExpectedNumInstanceSampler, ValidationSplitSampler, SequentialSampler

from torch import set_float32_matmul_precision
set_float32_matmul_precision('medium') # or high to trade off performance for precision

from pytorch_transformer_ts.informer.lightning_module import InformerLightningModule
from pytorch_transformer_ts.informer.estimator import InformerEstimator
from pytorch_transformer_ts.autoformer.estimator import AutoformerEstimator
from pytorch_transformer_ts.autoformer.lightning_module import AutoformerLightningModule
from pytorch_transformer_ts.spacetimeformer.estimator import SpacetimeformerEstimator
from pytorch_transformer_ts.spacetimeformer.lightning_module import SpacetimeformerLightningModule
from pytorch_transformer_ts.tactis_2.estimator import TACTiS2Estimator as TactisEstimator
from pytorch_transformer_ts.tactis_2.lightning_module import TACTiS2LightningModule as TactisLightningModule
from wind_forecasting.preprocessing.data_module import DataModule
from wind_forecasting.run_scripts.testing import test_model, get_checkpoint

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

mpi_exists = False
try:
    from mpi4py import MPI
    mpi_exists = True
except:
    print("No MPI available on system.")

# @profile
def main():
    
    # Determine worker rank (using WORKER_RANK set in Slurm script, fallback to 0)
    try:
        # Use the WORKER_RANK variable set explicitly in the Slurm script's nohup block
        rank = int(os.environ.get('WORKER_RANK', '0'))
    except ValueError:
        logging.warning("Could not parse WORKER_RANK, assuming rank 0.")
        rank = 0
    logging.info(f"Determined worker rank from WORKER_RANK: {rank}")
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run a model on a dataset")
    parser.add_argument("--config", type=str, help="Path to config file", default="examples/inputs/training_inputs_aoifemac_flasc.yaml")
    parser.add_argument("-md", "--mode", choices=["tune", "train", "test"], required=True,
                        help="Mode to run: 'tune' for hyperparameter optimization with Optuna, 'train' to train a model, 'test' to evaluate a model")
    parser.add_argument("-chk", "--checkpoint", type=str, required=False, default="latest", 
                        help="Which checkpoint to use: can be equal to 'latest', 'best', or an existing checkpoint path.")
    parser.add_argument("-m", "--model", type=str, choices=["informer", "autoformer", "spacetimeformer", "tactis"], required=True)
    parser.add_argument("-rt", "--restart_tuning", action="store_true")
    parser.add_argument("-tp", "--use_tuned_parameters", action="store_true", help="Use parameters tuned from Optuna optimization, otherwise use defaults set in Module class.")
    parser.add_argument("--tune_first", action="store_true", help="Whether to use tuned parameters", default=False)
    parser.add_argument("--model_path", type=str, help="Path to a saved model checkpoint to load from", default=None)
    parser.add_argument("--predictor_path", type=str, help="Path to a saved predictor for evaluation", default=None)
    parser.add_argument("-s", "--seed", type=int, help="Seed for random number generator", default=42)
    parser.add_argument("--save_to", type=str, help="Path to save the predicted output", default=None)
    parser.add_argument("--single_gpu", action="store_true", help="Force using only a single GPU (the one specified by CUDA_VISIBLE_DEVICES)")

    args = parser.parse_args()
    
    # %% SETUP SEED
    logging.info(f"Setting random seed to {args.seed}")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # %% PARSE CONFIG
    logging.info(f"Parsing configuration from yaml and command line arguments")
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
        
    # TODO create function to check config params and set defaults
    assert args.checkpoint is None or args.checkpoint in ["best", "latest"] or os.path.exists(args.checkpoint), "Checkpoint argument, if provided, must equal 'best', 'latest', or an existing checkpoint path."
    # Modify configuration for single GPU mode vs. multi-GPU mode
    if args.single_gpu:
        # Force single GPU configuration when --single_gpu flag is set
        # This ensures each worker only uses the GPU assigned to it via CUDA_VISIBLE_DEVICES
        config["trainer"]["devices"] = 1
        config["trainer"]["strategy"] = "auto"  # Let PyTorch Lightning determine strategy
        logging.info("Single GPU mode enabled: Using devices=1 with auto strategy")
    else:
        # Check if CUDA_VISIBLE_DEVICES is set and contains only a single GPU
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            cuda_devices = os.environ["CUDA_VISIBLE_DEVICES"]
            logging.info(f"CUDA_VISIBLE_DEVICES is set to: '{cuda_devices}'")
            try:
                # Count the number of GPUs specified in CUDA_VISIBLE_DEVICES
                visible_gpus = [idx for idx in cuda_devices.split(',') if idx.strip()]
                num_visible_gpus = len(visible_gpus)
                
                if num_visible_gpus > 0:
                    # Only override if the current configuration doesn't match
                    if "devices" in config["trainer"] and config["trainer"]["devices"] != num_visible_gpus:
                        logging.warning(f"Adjusting trainer.devices from {config['trainer']['devices']} to {num_visible_gpus} based on CUDA_VISIBLE_DEVICES")
                        config["trainer"]["devices"] = num_visible_gpus
                        
                        # If only one GPU is visible, use auto strategy instead of distributed
                        if num_visible_gpus == 1 and "strategy" in config["trainer"] and config["trainer"]["strategy"] != "auto":
                            logging.warning("Setting strategy to 'auto' since only one GPU is visible")
                            config["trainer"]["strategy"] = "auto"
                            
                    # Log actual GPU mapping information
                    if num_visible_gpus == 1:
                        try:
                            actual_gpu = int(visible_gpus[0])
                            device_id = 0  # With CUDA_VISIBLE_DEVICES, first visible GPU is always index 0
                            logging.info(f"Primary GPU is system device {actual_gpu}, mapped to CUDA index {device_id}")
                        except ValueError:
                            logging.warning(f"Could not parse GPU index from CUDA_VISIBLE_DEVICES: {visible_gpus[0]}")
                else:
                    logging.warning("CUDA_VISIBLE_DEVICES is set but no valid GPU indices found")
            except Exception as e:
                logging.warning(f"Error parsing CUDA_VISIBLE_DEVICES: {e}")
        else:
            logging.warning("CUDA_VISIBLE_DEVICES is not set, using default GPU assignment")
        
        # Multi-GPU configuration from SLURM environment variables (if not overridden above)
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

    # Configure WandB to use the correct checkpoint location
    os.environ["WANDB_ARTIFACT_DIR"] = config["logging"]["checkpoint_dir"]
    # This ensures artifacts are saved in the correct checkpoint directory
    
    wandb_parent_dir = config["logging"]["wandb_dir"]
    os.environ["WANDB_DIR"] = wandb_parent_dir
    logging.info(f"WandB will create logs in {os.path.join(wandb_parent_dir, 'wandb')}")
    # Set an explicit run directory to avoid nesting issues
    from datetime import datetime
    unique_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{worker_id}_{gpu_id}"
    run_dir = os.path.join(config["logging"]["wandb_dir"], f"run_{unique_id}")
    os.environ["WANDB_RUN_DIR"] = run_dir
    
    # Create WandB logger with explicit path settings
    wandb_logger = WandbLogger(
        project="wind_forecasting",
        name=run_name,
        log_model="all",
        save_dir=config["logging"]["wandb_dir"],  # Use the dedicated wandb directory
        group=config['experiment']['run_name'],   # Group all workers under the same experiment
        tags=[f"worker_{worker_id}", f"gpu_{gpu_id}", args.model]  # Add tags for easier filtering
    )
    wandb_logger.log_hyperparams(config)
    config["trainer"]["logger"] = wandb_logger

    # Process absolute paths and resolve any variable references in the config
    log_dir = config["experiment"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    
    # Resolve path variables - ensure all paths are absolute and properly structured
    if "logging" not in config:
        config["logging"] = {}
        
    # Set up wandb directory - use absolute path
    if "wandb_dir" in config["logging"]:
        wandb_dir = config["logging"]["wandb_dir"]
    else:
        wandb_dir = os.path.join(log_dir, "wandb")
    os.makedirs(wandb_dir, exist_ok=True)
    
    # Set up optuna directory - use absolute path
    if "optuna_dir" in config["logging"]:
        optuna_dir = config["logging"]["optuna_dir"]
    else:
        optuna_dir = os.path.join(log_dir, "optuna")
    os.makedirs(optuna_dir, exist_ok=True)
    
    # Set up checkpoint directory - use absolute path
    if "checkpoint_dir" in config["logging"]:
        checkpoint_dir = config["logging"]["checkpoint_dir"]
    else:
        checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Update config with normalized absolute paths
    config["logging"]["wandb_dir"] = wandb_dir
    config["logging"]["optuna_dir"] = optuna_dir
    config["logging"]["checkpoint_dir"] = checkpoint_dir
    
    # Configure WandB to use the specified directory structure
    os.environ["WANDB_DIR"] = wandb_dir
    
    # Configure WandB to save runs in a standard location
    os.environ["WANDB_CHECKPOINT_PATH"] = checkpoint_dir
    
    # Ensure optuna journal_dir is set correctly with absolute path
    if "optuna" in config:
        # Only override journal_dir if it's not explicitly set
        if "journal_dir" not in config["optuna"] or config["optuna"]["journal_dir"] is None:
            config["optuna"]["journal_dir"] = optuna_dir
        else:
            # Ensure the directory exists
            os.makedirs(config["optuna"]["journal_dir"], exist_ok=True)
            logging.info(f"Using explicitly defined Optuna journal_dir: {config['optuna']['journal_dir']}")
    
    # Explicitly resolve any variable references in trainer config
    if "trainer" in config:
        if "default_root_dir" in config["trainer"]:
            # Replace ${logging.checkpoint_dir} with the actual path
            if isinstance(config["trainer"]["default_root_dir"], str) and "${" in config["trainer"]["default_root_dir"]:
                config["trainer"]["default_root_dir"] = checkpoint_dir
        else:
            config["trainer"]["default_root_dir"] = checkpoint_dir

    # --- Database Setup & Synchronization ---
    optuna_storage_target, pg_config = setup_optuna_storage(args, config, rank)
    # --- End Database Setup ---

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
            except KeyError as e:
                 logging.warning(f"KeyError accessing Optuna config for tuned params: {e}. Using defaults.")
                 logging.info(f"Declaring estimator {args.model.capitalize()} with default parameters")
        else:
            logging.info(f"Declaring estimator {args.model.capitalize()} with default parameters")
        
        # Use globals() to fetch the estimator class dynamically
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
        # Enhanced GPU device setup and error checking
        if torch.cuda.is_available():
            # Log all available GPUs for debugging
            num_gpus = torch.cuda.device_count()
            all_gpus = [f"{i}:{torch.cuda.get_device_name(i)}" for i in range(num_gpus)]
            logging.info(f"System has {num_gpus} CUDA device(s): {all_gpus}")
            
            # Check CUDA_VISIBLE_DEVICES setting
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                cuda_devices = os.environ["CUDA_VISIBLE_DEVICES"]
                logging.info(f"CUDA_VISIBLE_DEVICES is set to: '{cuda_devices}'")
                try:
                    # Count the number of GPUs specified in CUDA_VISIBLE_DEVICES
                    visible_gpus = [idx for idx in cuda_devices.split(',') if idx.strip()]
                    num_visible_gpus = len(visible_gpus)
                    
                    if num_visible_gpus > 0:
                        # Only override if the current configuration doesn't match
                        if "devices" in config["trainer"] and config["trainer"]["devices"] != num_visible_gpus:
                            logging.warning(f"Adjusting trainer.devices from {config['trainer']['devices']} to {num_visible_gpus} based on CUDA_VISIBLE_DEVICES")
                            config["trainer"]["devices"] = num_visible_gpus
                            
                            # If only one GPU is visible, use auto strategy instead of distributed
                            if num_visible_gpus == 1 and "strategy" in config["trainer"] and config["trainer"]["strategy"] != "auto":
                                logging.warning("Setting strategy to 'auto' since only one GPU is visible")
                                config["trainer"]["strategy"] = "auto"
                                
                        # Log actual GPU mapping information
                        if num_visible_gpus == 1:
                            try:
                                actual_gpu = int(visible_gpus[0])
                                device_id = 0  # With CUDA_VISIBLE_DEVICES, first visible GPU is always index 0
                                logging.info(f"Primary GPU is system device {actual_gpu}, mapped to CUDA index {device_id}")
                            except ValueError:
                                logging.warning(f"Could not parse GPU index from CUDA_VISIBLE_DEVICES: {visible_gpus[0]}")
                    else:
                        logging.warning("CUDA_VISIBLE_DEVICES is set but no valid GPU indices found")
                except Exception as e:
                    logging.warning(f"Error parsing CUDA_VISIBLE_DEVICES: {e}")
            else:
                logging.warning("CUDA_VISIBLE_DEVICES is not set, using default GPU assignment")
            
            # Verify current device setup
            device = torch.cuda.current_device()
            logging.info(f"Using GPU {device}: {torch.cuda.get_device_name(device)}")
            
            # Verify the trainer configuration matches what we expect
            if args.single_gpu and config["trainer"]["devices"] != 1:
                logging.warning(f"--single_gpu flag is set but trainer.devices={config['trainer']['devices']}. Forcing devices=1.")
                config["trainer"]["devices"] = 1
                config["trainer"]["strategy"] = "auto"
            
            # Final check to ensure configuration is valid for available GPUs    
            num_available_gpus = torch.cuda.device_count()
            if config["trainer"]["devices"] > num_available_gpus:
                logging.warning(f"Requested {config['trainer']['devices']} GPUs but only {num_available_gpus} are available. Adjusting trainer.devices.")
                config["trainer"]["devices"] = num_available_gpus
                if num_available_gpus == 1 and "strategy" in config["trainer"] and config["trainer"]["strategy"] != "auto":
                    config["trainer"]["strategy"] = "auto"
                
            logging.info(f"Trainer config: devices={config['trainer']['devices']}, strategy={config['trainer'].get('strategy', 'auto')}")
            
            # Log memory information
            logging.info(f"GPU Memory: {torch.cuda.memory_allocated(device)/1e9:.2f}GB / {torch.cuda.get_device_properties(device).total_memory/1e9:.2f}GB")
            
            # Clear GPU memory before starting
            torch.cuda.empty_cache()
            gc.collect()
        else:
            logging.error("CUDA is not available! This will likely cause the tuning process to fail.")
            logging.error("Please check your CUDA installation and GPU availability.")
        
        # %% TUNE MODEL WITH OPTUNA
        from wind_forecasting.run_scripts.tuning import tune_model
        if not os.path.exists(config["optuna"]["journal_dir"]):
            os.makedirs(config["optuna"]["journal_dir"]) 

        # Use globals() to fetch the module and estimator classes dynamically
        LightningModuleClass = globals()[f"{args.model.capitalize()}LightningModule"]
        EstimatorClass = globals()[f"{args.model.capitalize()}Estimator"]
        DistrOutputClass = globals()[config["model"]["distr_output"]["class"]]
        
        # handle_trial_with_oom_protection is now imported from wind_forecasting.utils.trial_utils
        
        # Normal execution - pass the OOM protection wrapper and constructed storage URL
        tune_model(model=args.model, config=config,
                    optuna_storage_target=optuna_storage_target, # Pass the URL string or storage object
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
                    trial_protection_callback=handle_trial_with_oom_protection,
                    seed=args.seed)
        
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
        from wind_forecasting.run_scripts.testing import test_model, get_checkpoint
        
        # Set up parameters for checkpoint finding
        metric = "val_loss_epoch"
        mode = "min"
        log_dir = config["trainer"]["default_root_dir"]
        
        # Use the get_checkpoint function to handle checkpoint finding
        checkpoint = get_checkpoint(args.checkpoint, metric, mode, log_dir)
            
        test_model(data_module=data_module,
                    checkpoint=checkpoint,
                    lightning_module_class=globals()[f"{args.model.capitalize()}LightningModule"], 
                    estimator=estimator, 
                    normalization_consts_path=config["dataset"]["normalization_consts_path"])
        
        logging.info("Model testing completed.")
        
        # %% EXPORT LOGGING DATA
        # api = wandb.Api()
        # run = api.run("<entity>/<project>/<run_id>")
        # metrics_df = run.history()
        # metrics_df.to_csv("metrics.csv")
        # history = run.scan_history()

if __name__ == "__main__":
    main()