import argparse
import logging
from memory_profiler import profile
import os
import torch
import gc
import random
import numpy as np
from datetime import datetime
from pathlib import Path

import polars as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
import yaml

# Internal imports
from wind_forecasting.utils.trial_utils import handle_trial_with_oom_protection
from wind_forecasting.utils.optuna_db_utils import setup_optuna_storage

from gluonts.torch.distributions import LowRankMultivariateNormalOutput
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
from wind_forecasting.run_scripts.tuning import get_tuned_params

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

mpi_exists = False
try:
    from mpi4py import MPI
    mpi_exists = True
except:
    logging.warning("No MPI available on system.")


def main():
    
    # %% DETERMINE WORKER RANK (using WORKER_RANK set in Slurm script, fallback to 0)
    try:
        # Use the WORKER_RANK variable set explicitly in the Slurm script's nohup block
        rank = int(os.environ.get('WORKER_RANK', '0'))
    except ValueError:
        logging.warning("Could not parse WORKER_RANK, assuming rank 0.")
        rank = 0
    logging.info(f"Determined worker rank from WORKER_RANK: {rank}")
    
    # %% PARSE ARGUMENTS
    parser = argparse.ArgumentParser(description="Run a model on a dataset")
    parser.add_argument("--config", type=str, help="Path to config file", default="examples/inputs/training_inputs_aoifemac_flasc.yaml")
    parser.add_argument("-md", "--mode", choices=["tune", "train", "test"], required=True,
                        help="Mode to run: 'tune' for hyperparameter optimization with Optuna, 'train' to train a model, 'test' to evaluate a model")
    parser.add_argument("-chk", "--checkpoint", type=str, required=False, default=None, 
                        help="Which checkpoint to use: can be equal to 'None' to start afresh with training mode, 'latest', 'best', or an existing checkpoint path.")
    parser.add_argument("-m", "--model", type=str, choices=["informer", "autoformer", "spacetimeformer", "tactis"], required=True)
    parser.add_argument("-rt", "--restart_tuning", action="store_true")
    parser.add_argument("-tp", "--use_tuned_parameters", action="store_true", help="Use parameters tuned from Optuna optimization, otherwise use defaults set in Module class.")
    # parser.add_argument("--tune_first", action="store_true", help="Whether to use tuned parameters", default=False)
    parser.add_argument("--model_path", type=str, help="Path to a saved model checkpoint to load from", default=None)
    # parser.add_argument("--predictor_path", type=str, help="Path to a saved predictor for evaluation", default=None) # JUAN shouldn't need if we just pass filepath, latest, or best to checkpoint parameter
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
        
    # if (type(config["dataset"]["target_turbine_ids"]) is str) and (
    #     (config["dataset"]["target_turbine_ids"].lower() == "none") or (config["dataset"]["target_turbine_ids"].lower() == "all")):
    #     config["dataset"]["target_turbine_ids"] = None # select all turbines
        
    assert args.checkpoint is None or args.checkpoint in ["best", "latest"] or os.path.exists(args.checkpoint), "Checkpoint argument, if provided, must equal 'best', 'latest', or an existing checkpoint path."
    assert (args.mode == "test" and args.checkpoint is not None) or args.mode != "test", "Must provide a checkpoint path, 'latest', or 'best' for checkpoint argument when mode argument=test."
    # %% Modify configuration for single GPU mode vs. multi-GPU mode
    if args.single_gpu:
        # Force single GPU configuration when --single_gpu flag is set
        # This ensures each worker only uses the GPU assigned to it via CUDA_VISIBLE_DEVICES
        config["trainer"]["devices"] = 1
        config["trainer"]["strategy"] = "auto"  # Let PyTorch Lightning determine strategy
        if config["trainer"]["devices"] != 1:
            # Verify the trainer configuration matches what we expect
            logging.warning(f"--single_gpu flag is set but trainer.devices={config['trainer']['devices']}. Forcing devices=1.")
        else:
            logging.info("Single GPU mode enabled: Using devices=1 with auto strategy")
    else:
        # Log all available GPUs for debugging
        num_gpus = torch.cuda.device_count()
        all_gpus = [f"{i}:{torch.cuda.get_device_name(i)}" for i in range(num_gpus)]
        logging.info(f"System has {num_gpus} CUDA device(s): {all_gpus}")
        
        # Verify current device setup
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            logging.info(f"Using GPU {device}: {torch.cuda.get_device_name(device)}")
            
            # Check if CUDA_VISIBLE_DEVICES is set and contains only a single GPU
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                cuda_devices = os.environ["CUDA_VISIBLE_DEVICES"] # Note: must 'export' variable within nohup to find on Kestrel
                logging.info(f"CUDA_VISIBLE_DEVICES is set to: '{cuda_devices}'")
                try:
                    # Count the number of GPUs specified in CUDA_VISIBLE_DEVICES
                    visible_gpus = [idx for idx in cuda_devices.split(',') if idx.strip()]
                    num_visible_gpus = len(visible_gpus)
                    
                    if num_visible_gpus > 0:
                        # Only override if the current configuration doesn't match
                        if config["trainer"]["devices"] != num_visible_gpus:
                            logging.warning(f"Adjusting trainer.devices from {config['trainer']['devices']} to {num_visible_gpus} based on CUDA_VISIBLE_DEVICES")
                            config["trainer"]["devices"] = num_visible_gpus
                            
                            # If only one GPU is visible, use auto strategy instead of distributed
                            if num_visible_gpus == 1 and config["trainer"]["strategy"] != "auto":
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
            
            # Log memory information
            logging.info(f"GPU Memory: {torch.cuda.memory_allocated(device)/1e9:.2f}GB / {torch.cuda.get_device_properties(device).total_memory/1e9:.2f}GB")
            
            # Clear GPU memory before starting
            torch.cuda.empty_cache()
            
        # Final check to ensure configuration is valid for available GPUs
        if isinstance(config["trainer"]["devices"], int) and config["trainer"]["devices"] > num_gpus:
            logging.warning(f"Requested {config['trainer']['devices']} GPUs but only {num_gpus} are available. Adjusting trainer.devices.")
            config["trainer"]["devices"] = num_gpus
            
        if num_gpus == 1 and config["trainer"]["strategy"] != "auto":
            logging.warning(f"Adjusting trainer.strategy from {config['trainer']['strategy']} to 'auto' for single machine GPU.")
            config["trainer"]["strategy"] = "auto"
            
        logging.info(f"Trainer config: devices={config['trainer']['devices']}, strategy={config['trainer'].get('strategy', 'auto')}")
        
        gc.collect()
        
        # Multi-GPU configuration from SLURM environment variables (if not overridden above)
        if "SLURM_NTASKS_PER_NODE" in os.environ:
            config["trainer"]["devices"] = int(os.environ["SLURM_NTASKS_PER_NODE"])
        if "SLURM_NNODES" in os.environ:
            config["trainer"]["num_nodes"] = int(os.environ["SLURM_NNODES"])

    # %% SETUP LOGGING
    logging.info("Setting up logging")
    
    if "logging" not in config:
        config["logging"] = {}
     
    # Set up logging directory - use absolute path, rename logging dirs to group checkpoints and logs by data source and model
    log_dir = os.path.join(config["experiment"]["log_dir"], f"{args.model}_{config['experiment']['run_name']}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up wandb, optuna, checkpoint directories - use absolute paths
    wandb_parent_dir = os.path.join(config["logging"].get("wandb_dir", log_dir))
    wandb_dir = os.path.join(wandb_parent_dir, "wandb") # Configure WandB to use the specified directory structure
    optuna_dir = config["logging"].get("optuna_dir", os.path.join(log_dir, "optuna"))
    # checkpoint_dir = config["logging"].get("checkpoint_dir", os.path.join(log_dir, "checkpoints")) # NOTE: no need, checkpoints are saved by Model Checkpoint callback in loggers save_dir (wandb_parent_dir)
    
    os.makedirs(wandb_parent_dir, exist_ok=True)
    os.makedirs(wandb_dir, exist_ok=True)
    os.makedirs(optuna_dir, exist_ok=True)
    # os.makedirs(checkpoint_dir, exist_ok=True)
    
    logging.info(f"WandB will create logs in {os.path.join(wandb_parent_dir, 'wandb')}")

    # Get worker info from environment variables
    worker_id = os.environ.get('SLURM_PROCID', '0')
    gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0')

    # Create a unique run name for each worker
    run_name = f"{config['experiment']['run_name']}_worker{worker_id}_gpu{gpu_id}"

    # Set an explicit run directory to avoid nesting issues
    unique_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{worker_id}_{gpu_id}"
    run_dir = os.path.join(wandb_parent_dir, f"run_{unique_id}")
    os.environ["WANDB_RUN_DIR"] = run_dir
    
    # Configure WandB to use the correct checkpoint location
    # This ensures artifacts are saved in the correct checkpoint directory
    # os.environ["WANDB_ARTIFACT_DIR"] = checkpoint_dir
    os.environ["WANDB_DIR"] = wandb_parent_dir
    
    # Create WandB logger with explicit path settings
    wandb_logger = WandbLogger(
        project="wind_forecasting",
        name=run_name,
        log_model="all",
        save_dir=wandb_parent_dir,  # Use the dedicated wandb directory
        group=config['experiment']['run_name'],   # Group all workers under the same experiment
        tags=[f"worker_{worker_id}", f"gpu_{gpu_id}", args.model]  # Add tags for easier filtering
    )
    wandb_logger.log_hyperparams(config)
    config["trainer"]["logger"] = wandb_logger

    
    # Update config with normalized absolute paths
    config["logging"]["optuna_dir"] = optuna_dir
    # config["logging"]["checkpoint_dir"] = checkpoint_dir

    os.environ["WANDB_DIR"] = wandb_dir # JUAN QUESTION TODO why reset this
    
    # Configure WandB to save runs in a standard location
    # os.environ["WANDB_CHECKPOINT_PATH"] = checkpoint_dir
    
    # Ensure optuna storage_dir is set correctly with absolute path
    # Only override storage_dir if it's not explicitly set
    if "storage_dir" not in config["optuna"] or config["optuna"]["storage_dir"] is None:
        config["optuna"]["storage_dir"] = optuna_dir
    else:
        # Ensure the directory exists
        os.makedirs(config["optuna"]["storage_dir"], exist_ok=True)
        logging.info(f"Using explicitly defined Optuna storage_dir: {config['optuna']['storage_dir']}")
    
    # Explicitly resolve any variable references in trainer config
    # TODO JUAN it seems messy to replace embedded vars like logging.checkpoint_dir - can we just let the user supply the pathname, check that it exists, and make it absolute?
    # if "default_root_dir" not in config["trainer"]:
    #     # Replace ${logging.checkpoint_dir} with the actual path
    #     if isinstance(config["trainer"]["default_root_dir"], str) and "${logging.checkpoint_dir}" in config["trainer"]["default_root_dir"]:
    #         config["trainer"]["default_root_dir"] = config["trainer"]["default_root_dir"].replace("${logging.checkpoint_dir}", checkpoint_dir)
    # else:
    # config["trainer"]["default_root_dir"] = checkpoint_dir # TODO i think these are saved elsewhere by model checkpoint callback?
    config["trainer"]["default_root_dir"] = log_dir

    # %% CREATE DATASET
    logging.info("Creating datasets")
    data_module = DataModule(data_path=config["dataset"]["data_path"], n_splits=config["dataset"]["n_splits"],
                            continuity_groups=None, train_split=(1.0 - config["dataset"]["val_split"] - config["dataset"]["test_split"]),
                                val_split=config["dataset"]["val_split"], test_split=config["dataset"]["test_split"],
                                prediction_length=config["dataset"]["prediction_length"], context_length=config["dataset"]["context_length"],
                                target_prefixes=["ws_horz", "ws_vert"], feat_dynamic_real_prefixes=["nd_cos", "nd_sin"],
                                freq=config["dataset"]["resample_freq"], target_suffixes=config["dataset"]["target_turbine_ids"],
                                    per_turbine_target=config["dataset"]["per_turbine_target"], as_lazyframe=False, dtype=pl.Float32)
    
    if rank_zero_only.rank == 0:
        logging.info("Preparing data for tuning")
        if not os.path.exists(data_module.train_ready_data_path):
            data_module.generate_datasets()
            reload = True
        else:
            reload = False
    
        data_module.generate_splits(save=True, reload=reload) 
    
    data_module.generate_splits(save=True, reload=False)

    # %% DEFINE ESTIMATOR
    if args.mode in ["train", "test"]:
        found_tuned_params = True
        if args.use_tuned_parameters:
            try:
                logging.info(f"Getting tuned parameters.")
                tuned_params = get_tuned_params(backend=config["optuna"]["storage"]["backend"], 
                                                study_name=f"tuning_{args.model}_{config['experiment']['run_name']}", 
                                                storage_dir=config["optuna"]["storage_dir"])
                config["dataset"].update({k: v for k, v in tuned_params.items() if k in config["dataset"]})
                config["model"][args.model].update({k: v for k, v in tuned_params.items() if k in config["model"][args.model]})
                config["trainer"].update({k: v for k, v in tuned_params.items() if k in config["trainer"]})
            except FileNotFoundError as e:
                logging.warning(e)
                found_tuned_params = False
            except KeyError as e:
                logging.warning(f"KeyError accessing Optuna config for tuned params: {e}. Using defaults.")
                found_tuned_params = False
        else:
            found_tuned_params = False 
        
        if found_tuned_params:
            logging.info(f"Declaring estimator {args.model.capitalize()} with tuned parameters")
        else:
            logging.info(f"Declaring estimator {args.model.capitalize()} with default parameters")
            
        # Set up parameters for checkpoint finding
        metric = "val_loss_epoch"
        mode = "min"
        log_dir = config["trainer"]["default_root_dir"]
        
        # Use the get_checkpoint function to handle checkpoint finding
        checkpoint = get_checkpoint(args.checkpoint, metric, mode, log_dir)
        
        # Use globals() to fetch the estimator class dynamically
        EstimatorClass = globals()[f"{args.model.capitalize()}Estimator"]

        # Prepare all arguments in a dictionary
        estimator_kwargs = {
            "freq": data_module.freq,
            "prediction_length": data_module.prediction_length,
            "num_feat_dynamic_real": data_module.num_feat_dynamic_real,
            "num_feat_static_cat": data_module.num_feat_static_cat,
            "cardinality": data_module.cardinality,
            "num_feat_static_real": data_module.num_feat_static_real,
            "input_size": data_module.num_target_vars,
            "scaling": False, # Scaling handled externally or internally by TACTiS
            "lags_seq": [0], # TACTiS doesn't typically use lags
            "time_features": [second_of_minute, minute_of_hour, hour_of_day, day_of_year],
            "batch_size": config["dataset"].setdefault("batch_size", 128),
            "num_batches_per_epoch": config["trainer"].setdefault("limit_train_batches", 1000),
            "context_length": data_module.context_length,
            "train_sampler": ExpectedNumInstanceSampler(num_instances=1.0, min_past=data_module.context_length, min_future=data_module.prediction_length),
            "validation_sampler": ValidationSplitSampler(min_past=data_module.context_length, min_future=data_module.prediction_length),
            "trainer_kwargs": config["trainer"],
        }

        # Add model-specific arguments from the config YAML
        estimator_kwargs.update(config["model"][args.model])

        if args.model != 'tactis':
            estimator_kwargs["distr_output"] = globals()[config["model"]["distr_output"]["class"]](dim=data_module.num_target_vars, **config["model"]["distr_output"]["kwargs"])
        elif 'distr_output' in estimator_kwargs:
             del estimator_kwargs['distr_output']

        estimator = EstimatorClass(**estimator_kwargs)

    if args.mode == "tune":
        # %% SETUP & SYNCHRONIZE DATABASE
        # Extract necessary parameters for DB setup explicitly
        study_name = f"tuning_{args.model}_{config['experiment']['run_name']}"
        optuna_cfg = config["optuna"]
        storage_cfg = optuna_cfg.get("storage", {})
        logging_cfg = config["logging"]
        experiment_cfg = config["experiment"]

        # Resolve paths relative to project root and substitute known variables
        project_root = experiment_cfg.get("project_root", os.getcwd())

        # make paths absolute
        def resolve_path(base_path, path_input):
            if not path_input: return None
            # Convert potential Path object back to string if needed
            path_str = str(path_input)
            abs_path = Path(path_str)
            if not abs_path.is_absolute():
                abs_path = Path(base_path) / abs_path
            return str(abs_path.resolve())

        # Resolve paths with direct substitution
        optuna_dir_from_config = logging_cfg.get("optuna_dir")
        resolved_optuna_dir = resolve_path(project_root, optuna_dir_from_config)
        if not resolved_optuna_dir:
             raise ValueError("logging.optuna_dir is required but not found or resolved.")

        pgdata_path_from_config = storage_cfg.get("pgdata_path")
        resolved_pgdata_path = resolve_path(project_root, pgdata_path_from_config)

        socket_dir_base_from_config = storage_cfg.get("socket_dir_base")
        if not socket_dir_base_from_config:
             socket_dir_base_str = os.path.join(resolved_optuna_dir, "sockets")
        else:
             socket_dir_base_str = str(socket_dir_base_from_config).replace("${logging.optuna_dir}", resolved_optuna_dir)
        resolved_socket_dir_base = resolve_path(project_root, socket_dir_base_str) # Make absolute

        sync_dir_from_config = storage_cfg.get("sync_dir")
        if not sync_dir_from_config:
             # Default value uses the resolved optuna_dir
             sync_dir_str = os.path.join(resolved_optuna_dir, "sync")
        else:
             # Substitute directly if the variable exists
             sync_dir_str = str(sync_dir_from_config).replace("${logging.optuna_dir}", resolved_optuna_dir)
        resolved_sync_dir = resolve_path(project_root, sync_dir_str) # Make absolute

        db_setup_params = {
            "backend": storage_cfg.get("backend", "sqlite"),
            "project_root": project_root,
            "pgdata_path": resolved_pgdata_path,
            "study_name": study_name,
            "use_socket": storage_cfg.get("use_socket", True),
            "use_tcp": storage_cfg.get("use_tcp", False),
            "db_host": storage_cfg.get("db_host", "localhost"),
            "db_port": storage_cfg.get("db_port", 5432),
            "db_name": storage_cfg.get("db_name", "optuna_study_db"),
            "db_user": storage_cfg.get("db_user", "optuna_user"),
            "run_cmd_shell": storage_cfg.get("run_cmd_shell", False),
            "socket_dir_base": resolved_socket_dir_base,
            "sync_dir": resolved_sync_dir,
            "storage_dir": resolved_optuna_dir, # For non-postgres backends
            "sqlite_path": storage_cfg.get("sqlite_path"), # For sqlite
            "sqlite_wal": storage_cfg.get("sqlite_wal", True), # For sqlite
            "sqlite_timeout": storage_cfg.get("sqlite_timeout", 600), # For sqlite
        }

        optuna_storage = setup_optuna_storage(
            db_setup_params=db_setup_params,
            study_name=study_name,
            restart_tuning=args.restart_tuning,
            rank=rank
        )

        logging.info("Starting Optuna hyperparameter tuning...")
        
        # %% TUNE MODEL WITH OPTUNA
        from wind_forecasting.run_scripts.tuning import tune_model

        # Use globals() to fetch the module and estimator classes dynamically
        LightningModuleClass = globals()[f"{args.model.capitalize()}LightningModule"]
        EstimatorClass = globals()[f"{args.model.capitalize()}Estimator"]
        DistrOutputClass = globals()[config["model"]["distr_output"]["class"]]
        
        # Normal execution - pass the OOM protection wrapper and constructed storage URL
        tune_model(model=args.model, config=config, # Pass full config here for model/trainer params
                   study_name=study_name,
                   optuna_storage=optuna_storage, # Pass the constructed storage object
                   lightning_module_class=LightningModuleClass,
                   estimator_class=EstimatorClass,
                   distr_output_class=DistrOutputClass,
                   data_module=data_module,
                   max_epochs=config["optuna"]["max_epochs"],
                   limit_train_batches=config["optuna"]["limit_train_batches"],
                   metric=config["optuna"]["metric"],
                   direction=config["optuna"]["direction"],
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
            ckpt_path=checkpoint,
            shuffle_buffer_length=1024
        )
        # train_output.trainer.checkpoint_callback.best_model_path
        logging.info("Model training completed.")
    elif args.mode == "test":
        logging.info("Starting model testing...")
        # %% TEST MODEL
       
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