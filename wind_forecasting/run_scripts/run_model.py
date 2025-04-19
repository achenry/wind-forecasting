import argparse
# from calendar import c
import logging
from memory_profiler import profile
import os
import torch
import gc
import random
import numpy as np
from datetime import datetime
from pathlib import Path
import platform
import subprocess

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

torch.set_float32_matmul_precision('medium') # or high to trade off performance for precision

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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [Early] %(message)s')

mpi_exists = False
try:
    from mpi4py import MPI
    mpi_exists = True
except:
    logging.warning("No MPI available on system.")


def main():
    
    # %% Determine rank using SLURM_PROCID from srun
    if 'SLURM_PROCID' in os.environ:
        try:
            rank = int(os.environ['SLURM_PROCID'])
            logging.info(f"Determined worker rank from SLURM_PROCID: {rank}")
        except ValueError:
            logging.warning("Could not parse SLURM_PROCID, falling back to WORKER_RANK.")
            rank = int(os.environ.get('WORKER_RANK', '0')) # Fallback
    else:
        # Fallback for single-node or non-srun launch (old script) with WORKER_RANK
        try:
            rank = int(os.environ.get('WORKER_RANK', '0'))
            logging.info(f"Determined worker rank from WORKER_RANK (SLURM_PROCID not set): {rank}")
        except ValueError:
            logging.warning("Could not parse WORKER_RANK, assuming rank 0.")
            rank = 0

    # %% CONFIGURE RANK-SPECIFIC LOGGING
    try:
        job_id = os.environ.get('SLURM_JOB_ID', 'unknown_job')
        base_log_dir = os.environ.get('LOG_DIR', './logging')
        log_dir_path = os.path.join(base_log_dir, 'slurm_logs', job_id)
        os.makedirs(log_dir_path, exist_ok=True)

        log_file_path = os.path.join(log_dir_path, f'worker_{rank}_{job_id}.py.log')
        logger = logging.getLogger()
        handler_exists = any(
            isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == log_file_path
            for h in logger.handlers
        )
        if not handler_exists:
            logger.setLevel(logging.INFO)
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
                handler.close()
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(logging.INFO)
            
            formatter = logging.Formatter(f'%(asctime)s - %(levelname)s - Rank {rank} - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            logging.info(f"Initialized rank-specific logging to: {log_file_path}")
        else:
            logging.info(f"Rank-specific logging handler already configured for: {log_file_path}")

    except Exception as e:
         # Fallback to basic console logging if file setup fails
         if not logging.getLogger().hasHandlers():
             logging.basicConfig(level=logging.INFO, format=f'%(asctime)s - %(levelname)s - Rank {rank} - %(message)s')
         logging.error(f"Failed to configure rank-specific file logging. Error: {e}. Falling back to console logging.", exc_info=True)
    
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
    # parser.add_argument("--single_gpu", action="store_true", help="Force using only a single GPU (the one specified by CUDA_VISIBLE_DEVICES)") # Deprecated: Use srun with SLURM env vars + devices: auto

    args = parser.parse_args()
    
    # %% SETUP SEED
    base_seed = args.seed
    logging.info(f"Using base random seed: {base_seed}")
    torch.manual_seed(base_seed)
    random.seed(base_seed)
    np.random.seed(base_seed)
    
    # %% PARSE CONFIG
    logging.info(f"Parsing configuration from yaml and command line arguments")
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
        
    # if (type(config["dataset"]["target_turbine_ids"]) is str) and (
    #     (config["dataset"]["target_turbine_ids"].lower() == "none") or (config["dataset"]["target_turbine_ids"].lower() == "all")):
    #     config["dataset"]["target_turbine_ids"] = None # select all turbines
        
    assert args.checkpoint is None or args.checkpoint in ["best", "latest"] or os.path.exists(args.checkpoint), "Checkpoint argument, if provided, must equal 'best', 'latest', or an existing checkpoint path."
    assert (args.mode == "test" and args.checkpoint is not None) or args.mode != "test", "Must provide a checkpoint path, 'latest', or 'best' for checkpoint argument when mode argument=test."
    # %% Configure Trainer based on YAML and Environment (SLURM)
    logging.info(f"Trainer Config (from YAML): accelerator={config['trainer'].get('accelerator')}, strategy={config['trainer'].get('strategy')}")

    # Let Lightning determine devices based on SLURM vars when set to 'auto'
    if config['trainer'].get('devices') == 'auto':
        logging.info("Trainer 'devices' set to 'auto'. Relying on PyTorch Lightning and SLURM environment.")
    else:
        logging.info(f"Trainer 'devices' set to: {config['trainer'].get('devices')}")

    # Set num_nodes based on SLURM if available, otherwise use YAML default (usually 1)
    if "SLURM_NNODES" in os.environ:
        try:
            slurm_nodes = int(os.environ["SLURM_NNODES"])
            if config['trainer'].get('num_nodes') != slurm_nodes:
                 logging.warning(f"Overriding trainer num_nodes ({config['trainer'].get('num_nodes')}) with SLURM_NNODES ({slurm_nodes})")
                 config['trainer']['num_nodes'] = slurm_nodes
            else:
                 logging.info(f"Using num_nodes={slurm_nodes} (from SLURM_NNODES)")
        except ValueError:
            logging.warning(f"Could not parse SLURM_NNODES ({os.environ['SLURM_NNODES']}). Using num_nodes from YAML: {config['trainer'].get('num_nodes')}")
    else:
        logging.info(f"Using num_nodes={config['trainer'].get('num_nodes')} (from YAML, SLURM_NNODES not set)")

    # Set rank-specific seed for reproducibility across processes
    process_seed = base_seed + rank
    logging.info(f"Setting process-specific seed for rank {rank} to {process_seed}")
    torch.manual_seed(process_seed)
    random.seed(process_seed)
    np.random.seed(process_seed)
    # Ensure CUDA seeds are set if using GPU
    if config['trainer'].get('accelerator') == 'gpu' and torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)
        logging.info(f"Set CUDA seeds for rank {rank}")

    # Log final effective trainer configuration
    logging.info(f"Effective Trainer Config: accelerator={config['trainer'].get('accelerator')}, "
                 f"strategy={config['trainer'].get('strategy')}, "
                 f"devices={config['trainer'].get('devices')}, "
                 f"num_nodes={config['trainer'].get('num_nodes')}")

    # Clear GPU cache if using GPU
    if config['trainer'].get('accelerator') == 'gpu' and torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # %% SETUP LOGGING
    logging.info("Setting up logging")
    
    if "logging" not in config:
        config["logging"] = {}
     
    # Set up logging directory - use absolute path, rename logging dirs to group checkpoints and logs by data source and model
    log_dir = config["experiment"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up wandb, optuna, checkpoint directories - use absolute paths
    wandb_dir = config["logging"]["wandb_dir"] = config["logging"].get("wandb_dir", os.path.join(log_dir, "wandb"))
    optuna_dir = config["logging"]["optuna_dir"] = config["logging"].get("optuna_dir", os.path.join(log_dir, "optuna"))
    checkpoint_dir = config["logging"]["checkpoint_dir"] = config["logging"].get("checkpoint_dir", os.path.join(log_dir, "checkpoints")) # NOTE: no need, checkpoints are saved by Model Checkpoint callback in loggers save_dir (wandb_dir)

    os.makedirs(wandb_dir, exist_ok=True)
    os.makedirs(optuna_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    logging.info(f"WandB will create logs in {wandb_dir}")
    logging.info(f"Optuna will create logs in {optuna_dir}")
    logging.info(f"Checkpoints will be saved in {checkpoint_dir}")

    # Get worker info from environment variables
    worker_id = os.environ.get('SLURM_PROCID', '0')
    gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0')

    # Create a unique run name for each worker
    project_name = config['experiment'].get('project_name', 'wind_forecasting')
    run_name = f"{config['experiment']['username']}_{args.model}_{args.mode}_{gpu_id}"

    # Set an explicit run directory to avoid nesting issues
    unique_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{worker_id}_{gpu_id}"
    run_dir = os.path.join(wandb_dir, f"run_{unique_id}")
    os.environ["WANDB_RUN_DIR"] = run_dir
    
    # Configure WandB to use the correct checkpoint location
    # This ensures artifacts are saved in the correct checkpoint directory
    os.environ["WANDB_ARTIFACT_DIR"] = checkpoint_dir
    os.environ["WANDB_DIR"] = wandb_dir
    
    # Fetch GitHub repo URL and current commit and set WandB environment variables
    project_root = config['experiment'].get('project_root', os.getcwd())
    git_info = {}
    try:
        remote_url_bytes = subprocess.check_output(['git', 'config', '--get', 'remote.origin.url'], cwd=project_root, stderr=subprocess.STDOUT).strip()
        remote_url = remote_url_bytes.decode('utf-8')
        # Convert SSH URL to HTTPS if necessary
        if remote_url.startswith("git@"):
            remote_url = remote_url.replace(":", "/").replace("git@", "https://")
        if remote_url.endswith(".git"):
            remote_url = remote_url[:-4]

        commit_hash_bytes = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=project_root, stderr=subprocess.STDOUT).strip()
        commit_hash = commit_hash_bytes.decode('utf-8')

        git_info = {"url": remote_url, "commit": commit_hash}
        logging.info(f"Fetched Git Info - URL: {remote_url}, Commit: {commit_hash}")

    except subprocess.CalledProcessError as e:
        logging.warning(f"Could not get Git info: {e.output.decode('utf-8').strip()}. Git info might not be logged in WandB.")
    except FileNotFoundError:
        logging.warning("'git' command not found. Cannot log Git info.")
    except Exception as e:
        logging.warning(f"An unexpected error occurred while fetching Git info: {e}. Git info might not be logged in WandB.", exc_info=True)

    # Prepare logger config with only relevant model and dynamic info
    model_config = config['model'].get(args.model, {})
    dynamic_info = {
        'seed': args.seed,
        'rank': rank,
        'gpu_id': gpu_id,
        'devices': config['trainer'].get('devices'),
        'strategy': config['trainer'].get('strategy'),
        'torch_version': torch.__version__,
        'python_version': platform.python_version(),
    }
    logger_config = {
        'experiment': config.get('experiment', {}),
        'dataset': config.get('dataset', {}),
        'trainer': config.get('trainer', {}),
        'model': model_config,
        **dynamic_info,
        "git_info": git_info
    }

    # Create WandB logger only for train/test modes
    if args.mode in ["train", "test"]:
        wandb_logger = WandbLogger(            
            project=project_name, # Project name in WandB, set in config
            entity=config['logging'].get('entity'),
            group=config['experiment']['run_name'],   # Group all workers under the same experiment
            name=run_name, # Unique name for the run, can also take config for hyperparameters. Keep brief
            dir=wandb_dir, # Directory for saving logs and metadata
            log_model="all",            
            job_type=args.mode,
            mode=config['logging'].get('wandb_mode', 'online'), # Configurable wandb mode
            id=unique_id, # Unique ID for the run, can also use config hyperaparameters for comparison later
            notes=config['experiment'].get('notes'),
            tags=[f"gpu_{gpu_id}", args.model, args.mode] + config['experiment'].get('extra_tags', []),
            config=logger_config,            
            save_code=config['logging'].get('save_code', False)
        )
        config["trainer"]["logger"] = wandb_logger
    else:
        # For tuning mode, set logger to None
        config["trainer"]["logger"] = None


    # Update config with normalized absolute paths
    config["logging"]["optuna_dir"] = optuna_dir
    # config["logging"]["checkpoint_dir"] = checkpoint_dir

    
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
        metric = config.get("trainer", {}).get("monitor_metric", "val_loss")
        mode = config.get("optuna", {}).get("direction", "minimize")
        mode = "min" if mode == "minimize" else "max" if mode == "maximize" else "min"
        
        log_dir = config["trainer"]["default_root_dir"]
        logging.info(f"Checkpoint selection: Monitoring metric '{metric}' with mode '{mode}' in directory '{log_dir}'")
        
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
                   n_trials_per_worker=config["optuna"]["n_trials_per_worker"],
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