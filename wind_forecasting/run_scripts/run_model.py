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
import platform
import subprocess

import polars as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
import yaml
from lightning.pytorch.strategies import DDPStrategy # Ensure import

# Internal imports
from wind_forecasting.utils.trial_utils import handle_trial_with_oom_protection
from wind_forecasting.utils.optuna_db_utils import setup_optuna_storage

from gluonts.torch.distributions import LowRankMultivariateNormalOutput
from gluonts.model.forecast_generator import DistributionForecastGenerator, SampleForecastGenerator
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
from wind_forecasting.run_scripts.tuning import get_tuned_params, generate_df_setup_params

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
    parser.add_argument("-utp", "--use_tuned_parameters", action="store_true", help="Use parameters tuned from Optuna optimization, otherwise use defaults set in Module class.")
    parser.add_argument("-tp", "--tuning_phase", type=int, default=1, help="Index of tuning phase to use, gets passed to get_params estimator class methods. For tuning with multiple phases.")
    # parser.add_argument("--tune_first", action="store_true", help="Whether to use tuned parameters", default=False)
    parser.add_argument("--model_path", type=str, help="Path to a saved model checkpoint to load from", default=None)
    # parser.add_argument("--predictor_path", type=str, help="Path to a saved predictor for evaluation", default=None) # JUAN shouldn't need if we just pass filepath, latest, or best to checkpoint parameter
    parser.add_argument("-s", "--seed", type=int, help="Seed for random number generator", default=42)
    parser.add_argument("--save_to", type=str, help="Path to save the predicted output", default=None)
    parser.add_argument("--single_gpu", action="store_true", help="Force using only a single GPU (the one specified by CUDA_VISIBLE_DEVICES)")
    parser.add_argument("-or", "--override", nargs="*", help="List of hyperparameters to override from YAML config instead of using tuned values", default=[])

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
    
    # Store the original YAML config to access original values later if needed for overrides
    original_yaml_config = yaml.safe_load(open(args.config, "r"))
    
    if args.override:
        logging.info(f"Parameters specified for override from YAML: {args.override}")
        
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
                        
                        # Check if strategy needs special handling for TACTiS DDP
                        # Use .get() with default to avoid KeyError if 'strategy' not in config['trainer']
                        current_strategy_setting = config.get("trainer", {}).get("strategy", "auto")

                        if isinstance(current_strategy_setting, str) and current_strategy_setting.lower() == "ddp" and args.model == "tactis":
                             logging.warning("Instantiating DDPStrategy with find_unused_parameters=True for TACTiS-2.")
                             # Instantiate the strategy object with the flag
                             strategy_object = DDPStrategy(find_unused_parameters=True)
                             # Store the object back into the config dictionary
                             # Ensure config['trainer'] exists
                             if "trainer" not in config: config["trainer"] = {}
                             config["trainer"]["strategy"] = strategy_object
                        # else:
                             # Keep the original strategy setting (e.g., 'auto', 'ddp_spawn', or maybe already an object)
                             # No change needed to config["trainer"]["strategy"]
                             # logging.info(f"Using strategy setting from config: {current_strategy_setting}")
                        
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
    log_dir = config["experiment"]["log_dir"]
    # Set up wandb, optuna, checkpoint directories - use absolute paths

    wandb_dir = config["logging"]["wandb_dir"] = config["logging"].get("wandb_dir", os.path.join(log_dir, "wandb"))
    optuna_dir = config["logging"]["optuna_dir"] = config["logging"].get("optuna_dir", os.path.join(log_dir, "optuna"))
    # TODO: do we need this, checkpoints are saved by Model Checkpoint callback in loggers save_dir (wandb_dir)
    # config["trainer"]["default_root_dir"] = checkpoint_dir = config["logging"]["checkpoint_dir"] = config["logging"].get("checkpoint_dir", os.path.join(log_dir, "checkpoints")) 
    
    os.makedirs(wandb_dir, exist_ok=True)
    os.makedirs(optuna_dir, exist_ok=True)
    # os.makedirs(checkpoint_dir, exist_ok=True)
    
    logging.info(f"WandB will create logs in {wandb_dir}")
    logging.info(f"Optuna will create logs in {optuna_dir}")
    # logging.info(f"Checkpoints will be saved in {checkpoint_dir}")

    # Get worker info from environment variables
    worker_id = os.environ.get('SLURM_PROCID', '0')
    gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0')

    # Create a unique run name for each worker
    project_name = f"{config['experiment'].get('project_name', 'wind_forecasting')}_{args.model}"
    run_name = f"{config['experiment']['username']}_{args.model}_{args.mode}_{gpu_id}"

    # Set an explicit run directory to avoid nesting issues
    unique_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{worker_id}_{gpu_id}"
    run_dir = os.path.join(wandb_dir, f"run_{unique_id}")
    
    # Configure WandB to use the correct checkpoint location
    # This ensures artifacts are saved in the correct checkpoint directory
    os.environ["WANDB_RUN_DIR"] = run_dir
    # os.environ["WANDB_ARTIFACT_DIR"] = os.environ["WANDB_CHECKPOINT_PATH"] = checkpoint_dir
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
    checkpoint_dir = os.path.join(log_dir, project_name, unique_id)
    # Create WandB logger only for train/test modes
    if args.mode in ["train", "test"]:
        wandb_logger = WandbLogger(
            project=project_name, # Project name in WandB, set in config
            entity=config['logging'].get('entity'),
            group=config['experiment']['run_name'],   # Group all workers under the same experiment
            name=run_name, # Unique name for the run, can also take config for hyperparameters. Keep brief
            save_dir=wandb_dir, # Directory for saving logs and metadata
            log_model=False,
            job_type=args.mode,
            mode=config['logging'].get('wandb_mode', 'online'), # Configurable wandb mode
            id=unique_id,
            tags=[f"gpu_{gpu_id}", args.model, args.mode] + config['experiment'].get('extra_tags', []),
            config=logger_config,
            save_code=config['logging'].get('save_code', False)
        )
        config["trainer"]["logger"] = wandb_logger
    else:
        # For tuning mode, set logger to None
        config["trainer"]["logger"] = None
    
    # Explicitly resolve any variable references in trainer config
    
    # Ensure default_root_dir exists and is set correctly
    config["trainer"]["default_root_dir"] = checkpoint_dir

    # %% CREATE DATASET
    # Dynamically set DataLoader workers based on SLURM_CPUS_PER_TASK
    cpus_per_task_str = os.environ.get('SLURM_CPUS_PER_TASK', '1') # Default to 1 CPU if var not set
    try:
        cpus_per_task = int(cpus_per_task_str)
        if cpus_per_task <= 0:
             logging.warning(f"SLURM_CPUS_PER_TASK is non-positive ({cpus_per_task}), defaulting cpus_per_task to 1.")
             cpus_per_task = 1
    except ValueError:
        logging.warning(f"Could not parse SLURM_CPUS_PER_TASK ('{cpus_per_task_str}'), defaulting cpus_per_task to 1.")
        cpus_per_task = 1

    num_workers = max(0, cpus_per_task - 1)
    if "trainer" not in config:
        config["trainer"] = {}

    # Set DataLoader parameters within the trainer config
    logging.info(f"Determined SLURM_CPUS_PER_TASK={cpus_per_task}. Setting num_workers = {num_workers}.")

    logging.info("Creating datasets")
    data_module = DataModule(
        data_path=config["dataset"]["data_path"],
        n_splits=config["dataset"]["n_splits"],
        continuity_groups=None,
        train_split=(1.0 - config["dataset"]["val_split"] - config["dataset"]["test_split"]),
        val_split=config["dataset"]["val_split"],
        test_split=config["dataset"]["test_split"],
        prediction_length=config["dataset"]["prediction_length"],
        context_length=config["dataset"]["context_length"],
        target_prefixes=["ws_horz", "ws_vert"],
        feat_dynamic_real_prefixes=["nd_cos", "nd_sin"],
        freq=config["dataset"]["resample_freq"],
        target_suffixes=config["dataset"]["target_turbine_ids"],
        per_turbine_target=config["dataset"]["per_turbine_target"],
        as_lazyframe=False,
        dtype=pl.Float32,
        batch_size=config["dataset"].get("batch_size", 128),
        workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        verbose=True
    )

    if rank_zero_only.rank == 0:
        logging.info("Preparing data for tuning")
        if not os.path.exists(data_module.train_ready_data_path):
            data_module.generate_datasets()
            reload = True
        else:
            reload = False

        data_module.generate_splits(save=True, reload=reload, splits=["train", "val", "test"])
    
    data_module.generate_splits(save=True, reload=False, splits=["train", "val", "test"])

    # %% DEFINE ESTIMATOR
    # Initialize storage and connection info variables
    optuna_storage = None
    db_connection_info = None # Will hold pg_config if PostgreSQL is used
    db_setup_params = None # Initialize

    if args.mode == "tune" or (args.mode == "train" and args.use_tuned_parameters) or (args.mode == "test" and args.use_tuned_parameters):
        # %% SETUP OPTUNA STORAGE (PostgreSQL for tuning, SQLite for loading tuned params)
        
        # Generate DB setup parameters regardless of mode (needed for study name)
        logging.info("Generating Optuna DB setup parameters...")
        db_setup_params = generate_df_setup_params(args.model, config)
        
        # Determine if restart_tuning should be overridden
        # We never want to restart/delete the study when just loading parameters
        effective_restart_tuning = args.restart_tuning
        if args.mode in ["train", "test"] and args.use_tuned_parameters:
            logging.info(f"Mode is '{args.mode}' with --use_tuned_parameters. Ensuring restart_tuning is False.")
            effective_restart_tuning = False
        elif args.mode == "tune":
            logging.info(f"Mode is 'tune'. Using restart_tuning={args.restart_tuning} from command line.")
        else:
            # Should not happen with the main condition, but safety check
             logging.warning(f"Unexpected combination: mode={args.mode}, use_tuned_parameters={args.use_tuned_parameters}. Defaulting restart_tuning to False.")
             effective_restart_tuning = False

        logging.info(f"Setting up Optuna storage (rank {rank}) using backend from config...")
        # Call setup_optuna_storage using config-derived params and effective restart flag
        optuna_storage, db_connection_info = setup_optuna_storage(
            db_setup_params=db_setup_params,
            restart_tuning=effective_restart_tuning, # Use the potentially overridden flag
            rank=rank
            # No force_sqlite_path argument anymore
        )
        logging.info(f"Optuna storage setup complete. Storage type: {type(optuna_storage).__name__}")
        if db_connection_info:
             logging.info("PostgreSQL connection info returned (likely tuning mode).")
        else:
             logging.info("No connection info returned (likely SQLite or Journal mode).")

    if args.mode in ["train", "test"]:
        
        # TODO refactor this to just use hparams from checkpoint
        found_tuned_params = False # Initialize to False
        tuned_params = {} # Initialize empty dict

        if args.use_tuned_parameters:
            if optuna_storage and db_setup_params:
                try:
                    logging.info(f"Attempting to load tuned parameters from study '{db_setup_params['study_name']}' using provided storage.")
                    # Pass the specific study name from db_setup_params
                    tuned_params = get_tuned_params(optuna_storage, db_setup_params["study_name"])
                    logging.info(f"Successfully loaded {len(tuned_params)} tuned parameters.")
                    
                    # Apply loaded parameters to the config
                    config["model"]["distr_output"]["kwargs"].update({k: v for k, v in tuned_params.items() if k in config["model"]["distr_output"]["kwargs"]})
                    config["dataset"].update({k: v for k, v in tuned_params.items() if k in config["dataset"]})
                    config["model"][args.model].update({k: v for k, v in tuned_params.items() if k in config["model"][args.model]})
                    config["trainer"].update({k: v for k, v in tuned_params.items() if k in config["trainer"]})
                    
                    # Apply overrides from YAML config for any parameters specified with --override
                    if args.override and args.use_tuned_parameters:
                        logging.info("Applying YAML overrides for specified parameters:")
                        
                        for param_name in args.override:
                            # Handle TACTiS-specific parameters in model.tactis section
                            if param_name in ["gradient_clip_val_stage1", "gradient_clip_val_stage2"] and args.model == "tactis":
                                if "model" in original_yaml_config and "tactis" in original_yaml_config["model"] and param_name in original_yaml_config["model"]["tactis"]:
                                    original_value = original_yaml_config["model"]["tactis"][param_name]
                                    
                                    # Ensure nested dictionaries exist
                                    if "model" not in config: config["model"] = {}
                                    if "tactis" not in config["model"]: config["model"]["tactis"] = {}
                                    
                                    # Apply the override
                                    config["model"]["tactis"][param_name] = original_value
                                    logging.info(f"  - Overriding model.tactis.{param_name} with YAML value: {original_value}")
                            
                            # Handle trainer parameters
                            elif param_name in ["gradient_clip_val"] and "trainer" in original_yaml_config and param_name in original_yaml_config["trainer"]:
                                original_value = original_yaml_config["trainer"][param_name]
                                config["trainer"][param_name] = original_value
                                logging.info(f"  - Overriding trainer.{param_name} with YAML value: {original_value}")
                            
                            # Handle dataset parameters
                            elif param_name in original_yaml_config.get("dataset", {}):
                                original_value = original_yaml_config["dataset"][param_name]
                                config["dataset"][param_name] = original_value
                                logging.info(f"  - Overriding dataset.{param_name} with YAML value: {original_value}")
                            
                            # Handle general model parameters for current model
                            elif param_name in original_yaml_config.get("model", {}).get(args.model, {}):
                                original_value = original_yaml_config["model"][args.model][param_name]
                                config["model"][args.model][param_name] = original_value
                                logging.info(f"  - Overriding model.{args.model}.{param_name} with YAML value: {original_value}")
                            
                            else:
                                logging.warning(f"  - Parameter '{param_name}' not found in original YAML config, cannot override")
                    
                    context_length_factor = tuned_params.get('context_length_factor', config["dataset"].get("context_length_factor", 2)) # Default to config or 2 if not in trial/config
                    context_length = int(context_length_factor * data_module.prediction_length)
                    
                    # Mark as found only if loading succeeds without exceptions
                    found_tuned_params = True
                    
                except FileNotFoundError as e:
                    logging.warning(f"Could not find Optuna study or parameters: {e}. Using default parameters.")
                    found_tuned_params = False
                except KeyError as e:
                    logging.warning(f"KeyError accessing Optuna config/study for tuned params: {e}. Using default parameters.")
                    found_tuned_params = False
                except Exception as e: # Catch other potential errors during loading
                    logging.error(f"An unexpected error occurred while loading tuned parameters: {e}", exc_info=True)
                    found_tuned_params = False
        else:
            found_tuned_params = False 
        
        if found_tuned_params:
            logging.info(f"Declaring estimator {args.model.capitalize()} with tuned parameters")
        else:
            logging.info(f"Declaring estimator {args.model.capitalize()} with default parameters")
            if "context_length_factor" in config["model"][args.model]:
                context_length = int(config["model"][args.model]["context_length_factor"] * data_module.prediction_length)
                del config["model"][args.model]["context_length_factor"]
            else:
             context_length = data_module.context_length
            
        # Set up parameters for checkpoint finding
        metric = config.get("trainer", {}).get("monitor_metric", "val_loss")
        mode = config.get("optuna", {}).get("direction", "minimize")
        mode_mapping = {"minimize": "min", "maximize": "max"}
        mode = mode_mapping.get(mode, "min")
        
        logging.info(f"Checkpoint selection: Monitoring metric '{metric}' with mode '{mode}' in directory '{checkpoint_dir}'")
        
        # Use the get_checkpoint function to handle checkpoint finding
        checkpoint = get_checkpoint(args.checkpoint, metric, mode, checkpoint_dir)
        
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
            "batch_size": data_module.batch_size,
            "num_batches_per_epoch": config["trainer"].setdefault("limit_train_batches", 1000),
            "context_length": context_length,
            "train_sampler": ExpectedNumInstanceSampler(num_instances=1.0, min_past=context_length, min_future=data_module.prediction_length),
            "validation_sampler": ValidationSplitSampler(min_past=context_length, min_future=data_module.prediction_length),
            "trainer_kwargs": config["trainer"],
        }

        # Add model-specific arguments from the config YAML
        estimator_kwargs.update(config["model"][args.model])

        if args.model != 'tactis':
            estimator_kwargs["distr_output"] = globals()[config["model"]["distr_output"]["class"]](dim=data_module.num_target_vars, **config["model"]["distr_output"]["kwargs"])
        elif 'distr_output' in estimator_kwargs:
             del estimator_kwargs['distr_output']

        estimator = EstimatorClass(**estimator_kwargs)
        
        # Conditionally Create Forecast Generator
        if args.model == 'tactis':
            # TACTiS uses SampleForecastGenerator internally for prediction
            # because its foweard pass returns samples not distribution parameters
            logging.info(f"Using SampleForecastGenerator for TACTiS model.")
            forecast_generator = SampleForecastGenerator()
        else:
            # Other models use DistributionForecastGenerator based on their distr_output
            logging.info(f"Using DistributionForecastGenerator for {args.model} model.")
            # Ensure estimator has distr_output before accessing
            if not hasattr(estimator, 'distr_output'):
                raise AttributeError(f"Estimator for model '{args.model}' is missing 'distr_output' attribute needed for DistributionForecastGenerator.")
            forecast_generator = DistributionForecastGenerator(estimator.distr_output)
            
    if args.mode == "tune":
        logging.info("Starting Optuna hyperparameter tuning...")
        
        # %% TUNE MODEL WITH OPTUNA
        from wind_forecasting.run_scripts.tuning import tune_model

        # Use globals() to fetch the module and estimator classes dynamically
        LightningModuleClass = globals()[f"{args.model.capitalize()}LightningModule"]
        EstimatorClass = globals()[f"{args.model.capitalize()}Estimator"]
        DistrOutputClass = globals()[config["model"]["distr_output"]["class"]]
        
        try:
            callbacks_config = config.get('callbacks', {})
            mc_config = callbacks_config.get('model_checkpoint', {})
            mc_init_args = mc_config.get('init_args', {})

            if 'dirpath' in mc_init_args:
                original_dirpath = mc_init_args['dirpath']
                resolved_dirpath = original_dirpath

                if "${logging.checkpoint_dir}" in resolved_dirpath:
                    base_checkpoint_dir = config.get('logging', {}).get('checkpoint_dir')
                    if base_checkpoint_dir:
                        project_root = config.get('experiment', {}).get('project_root', '.')
                        abs_project_root = os.path.abspath(project_root)
                        abs_base_checkpoint_dir = os.path.abspath(os.path.join(abs_project_root, base_checkpoint_dir))
                        resolved_dirpath = resolved_dirpath.replace("${logging.checkpoint_dir}", abs_base_checkpoint_dir)
                        logging.info(f"Resolved '${{logging.checkpoint_dir}}' to '{abs_base_checkpoint_dir}'")
                    else:
                        logging.warning("Cannot resolve ${logging.checkpoint_dir}.")

                if not os.path.isabs(resolved_dirpath):
                     project_root = config.get('experiment', {}).get('project_root', '.')
                     abs_project_root = os.path.abspath(project_root)
                     abs_resolved_dirpath = os.path.abspath(os.path.join(abs_project_root, resolved_dirpath))
                else:
                     abs_resolved_dirpath = os.path.abspath(resolved_dirpath)

                config['callbacks']['model_checkpoint']['init_args']['dirpath'] = abs_resolved_dirpath

                os.makedirs(abs_resolved_dirpath, exist_ok=True)
            else:
                logging.warning("No 'dirpath' found in ModelCheckpoint configuration in YAML.")

        except KeyError as e:
            logging.warning(f"Could not check ModelCheckpoint in config: {e}")
        except Exception as e:
            logging.error(f"Error during checkpoint path resolution: {e}", exc_info=True)

        # Instantiate callbacks from configuration
        import importlib

        logging.info("Instantiating callbacks from configuration...")
        instantiated_callbacks = []
        if 'callbacks' in config and isinstance(config['callbacks'], dict):
            for cb_name, cb_config in config['callbacks'].items():
                if isinstance(cb_config, dict) and 'class_path' in cb_config:
                    try:
                        module_path, class_name = cb_config['class_path'].rsplit('.', 1)
                        CallbackClass = getattr(importlib.import_module(module_path), class_name)
                        init_args = cb_config.get('init_args', {})

                        # Special handling for ModelCheckpoint dirpath
                        if class_name == "ModelCheckpoint" and 'dirpath' in init_args and init_args['dirpath'] is not None:
                            if isinstance(init_args['dirpath'], str) and '${logging.checkpoint_dir}' in init_args['dirpath']:
                                init_args['dirpath'] = init_args['dirpath'].replace('${logging.checkpoint_dir}', checkpoint_dir)
                            elif not os.path.isabs(init_args['dirpath']):
                                project_root = config.get('experiment', {}).get('project_root', '.')
                                init_args['dirpath'] = os.path.abspath(os.path.join(project_root, init_args['dirpath']))

                        callback_instance = CallbackClass(**init_args)
                        instantiated_callbacks.append(callback_instance)
                        logging.info(f"Instantiated callback: {class_name}")

                    except Exception as e:
                        logging.error(f"Error instantiating callback {cb_name}: {e}")
                elif isinstance(cb_config, bool) and cb_config is True:
                    # Handle simple boolean flag callbacks
                    if cb_name == "progress_bar":
                        from lightning.pytorch.callbacks import RichProgressBar
                        instantiated_callbacks.append(RichProgressBar())
                    elif cb_name == "lr_monitor":
                        from lightning.pytorch.callbacks import LearningRateMonitor
                        lr_config = config['callbacks'].get('lr_monitor_config', {})
                        instantiated_callbacks.append(LearningRateMonitor(**lr_config))

            # Replace the dictionary in the config with the list of instances
            config['callbacks'] = instantiated_callbacks
            logging.info(f"Replaced config['callbacks'] with {len(instantiated_callbacks)} instantiated callback objects.")

        # Normal execution - pass the OOM protection wrapper and constructed storage URL
        tune_model(model=args.model, config=config, # Pass full config here for model/trainer params
                   study_name=db_setup_params["study_name"],
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
                   seed=args.seed, tuning_phase=args.tuning_phase,
                   restart_tuning=args.restart_tuning) # Add restart_tuning parameter
        
        # After training completes
        torch.cuda.empty_cache()
        gc.collect()
        logging.info("Optuna hyperparameter tuning completed.")
        
    elif args.mode == "train":
        logging.info("Starting model training...")
        # %% TRAIN MODEL
        # TODO lightning_logs still being created by something  in log dir...
        logging.info("Training model")
        estimator.train(
            training_data=data_module.train_dataset,
            validation_data=data_module.val_dataset,
            forecast_generator=forecast_generator,
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
                    forecast_generator=forecast_generator,
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