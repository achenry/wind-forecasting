import argparse
# from calendar import c
import logging
from memory_profiler import profile
import os
import re
import torch
import gc
import random
import json
import numpy as np
from datetime import datetime
import platform
import subprocess
import inspect

import polars as pl
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
import yaml
from lightning.pytorch.strategies import DDPStrategy # Ensure import

# Internal imports
from wind_forecasting.tuning.utils.trial_utils import handle_trial_with_oom_protection
from wind_forecasting.utils.optuna_storage import setup_optuna_storage

from gluonts.torch.distributions import LowRankMultivariateNormalOutput
from gluonts.model.forecast_generator import DistributionForecastGenerator, SampleForecastGenerator
from gluonts.time_feature._base import second_of_minute, minute_of_hour, hour_of_day, day_of_year
from gluonts.transform import ExpectedNumInstanceSampler, ValidationSplitSampler, SequentialSampler

torch.set_float32_matmul_precision('medium') # or high to trade off performance for precision

from wind_forecasting.utils.callbacks import DeadNeuronMonitor
from pytorch_transformer_ts.informer.lightning_module import InformerLightningModule
from pytorch_transformer_ts.informer.estimator import InformerEstimator
from pytorch_transformer_ts.autoformer.estimator import AutoformerEstimator
from pytorch_transformer_ts.autoformer.lightning_module import AutoformerLightningModule
from pytorch_transformer_ts.spacetimeformer.estimator import SpacetimeformerEstimator
from pytorch_transformer_ts.spacetimeformer.lightning_module import SpacetimeformerLightningModule
from pytorch_transformer_ts.tactis_2.estimator import TACTiS2Estimator as TactisEstimator
from pytorch_transformer_ts.tactis_2.lightning_module import TACTiS2LightningModule as TactisLightningModule
from wind_forecasting.preprocessing.data_module import DataModule
from wind_forecasting.run_scripts.testing import test_model, get_checkpoint, load_estimator_from_checkpoint
from wind_forecasting.tuning import get_tuned_params
from wind_forecasting.utils.optuna_config_utils import generate_db_setup_params

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
    parser.add_argument("-rl", "--reload_data", action="store_true", help="Whether to reload train/test/val datasets from preprocessed parquets or not.")


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
    original_yaml_config = yaml.safe_load(open(args.config, "r")) # Keep this if needed for the --use_tuned_params logic below
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

            # else:
                    # Keep the original strategy setting (e.g., 'auto', 'ddp_spawn', or maybe already an object)
                    # No change needed to config["trainer"]["strategy"]
                    # logging.info(f"Using strategy setting from config: {current_strategy_setting}")

            
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

    # Check if strategy needs special handling for TACTiS DDP
    # Use .get() with default to avoid KeyError if 'strategy' not in config['trainer']
    current_strategy_setting = config.get("trainer", {}).get("strategy", "auto")

    if isinstance(current_strategy_setting, str) and current_strategy_setting.lower() == "ddp" and args.model in ["tactis", "spacetimeformer"]:
        logging.warning("Instantiating DDPStrategy with find_unused_parameters=True for TACTiS-2.")
        # Instantiate the strategy object with the flag
        strategy_object = DDPStrategy(find_unused_parameters=True)
        # Store the object back into the config dictionary
        # Ensure config['trainer'] exists
        if "trainer" not in config: config["trainer"] = {}
        config["trainer"]["strategy"] = strategy_object
        
    # %% SETUP LOGGING
    logging.info("Setting up logging")

    if "logging" not in config:
        config["logging"] = {}

    # Set up logging directory - use absolute path, rename logging dirs to group checkpoints and logs by data source and model
    log_dir = config["experiment"]["log_dir"]
    # Set up wandb, optuna, checkpoint directories - use absolute paths

    wandb_dir = config["logging"]["wandb_dir"] = config["logging"].get("wandb_dir", log_dir)
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
    run_dir = os.path.join(wandb_dir, "wandb", f"run_{unique_id}")

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

    # Add global gradient clipping for automatic optimization
    config.setdefault("trainer", {})
    
    # Check if model has stage-specific gradient clipping (like TACTiS)
    has_stage_specific_clipping = (
        args.model == "tactis" and 
        "tactis" in config.get("model", {}) and
        any(k.startswith("gradient_clip_val_stage") for k in config["model"]["tactis"])
    )
    
    if has_stage_specific_clipping:
        # For models with stage-specific clipping, ensure no global gradient_clip_val interferes
        if "gradient_clip_val" in config["trainer"]:
            logging.info(f"Model {args.model} uses stage-specific gradient clipping. Removing trainer.gradient_clip_val={config['trainer']['gradient_clip_val']}")
            del config["trainer"]["gradient_clip_val"]
        else:
            logging.info(f"Model {args.model} will use stage-specific gradient clipping")
    else:
        # For other models, use global gradient clipping
        original_gcv = config["trainer"].get("gradient_clip_val")
        effective_gcv = config["trainer"].setdefault("gradient_clip_val", 1.0)
        
        if original_gcv is None:
            logging.info(f"Gradient_clip_val not specified in config, defaulting to {effective_gcv} for automatic optimization.")
        else:
            logging.info(f"Using gradient_clip_val: {original_gcv} from configuration for automatic optimization.")

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

    # Set DataLoader parameters within the trainer config
    logging.info(f"Determined SLURM_CPUS_PER_TASK={cpus_per_task}. Setting num_workers = {num_workers}.")

    logging.info("Creating datasets")
    use_normalization = False if args.model == "tactis" else config["dataset"].get("normalize", True)
    logging.info(f"Instantiating DataModule with normalized={use_normalization} (Forced False for TACTiS-2 which requires denormalized input)")
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
        normalized=use_normalization,  # TACTiS-2 requires denormalized input for internal scaling
        normalization_consts_path=config["dataset"]["normalization_consts_path"], # Needed for denormalization
        batch_size=config["dataset"].get("batch_size", 128),
        workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        verbose=True
    )
    
    # Use globals() to fetch the module and estimator classes dynamically
    LightningModuleClass = globals()[f"{args.model.capitalize()}LightningModule"]
    EstimatorClass = globals()[f"{args.model.capitalize()}Estimator"]
    DistrOutputClass = globals()[config["model"]["distr_output"]["class"]]

    # data_module.train_dataset = [ds for ds in data_module.train_dataset if ds["item_id"].endswith("SPLIT0")]
    
    # %% DEFINE ESTIMATOR
    # Initialize storage and connection info variables
    optuna_storage = None
    db_connection_info = None # Will hold pg_config if PostgreSQL is used
    db_setup_params = None # Initialize

    if args.mode == "tune" or (args.mode == "train" and args.use_tuned_parameters) or (args.mode == "test" and args.use_tuned_parameters):
        # %% SETUP OPTUNA STORAGE (PostgreSQL for tuning, SQLite for loading tuned params)

        # Generate DB setup parameters regardless of mode (needed for study name)
        logging.info("Generating Optuna DB setup parameters...")
        db_setup_params = generate_db_setup_params(args.model, config)

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
        
        # get parameters expected by estimator and trainer
        estimator_sig = inspect.signature(EstimatorClass.__init__)
        estimator_params = [param.name for param in estimator_sig.parameters.values()]
        
        trainer_sig = inspect.signature(Trainer.__init__)
        trainer_params = [param.name for param in trainer_sig.parameters.values()]
       
        # get default params
        model_hparams = config["model"].get(args.model, {})
        
        # get tuned params
        found_tuned_params = True
        if args.use_tuned_parameters:
            try:
                logging.info(f"Getting tuned parameters.")
                
                tuned_params = get_tuned_params(optuna_storage, db_setup_params["base_study_prefix"])
                
                # tuned_params = {'context_length_factor': 3, 'batch_size': 256, 'num_encoder_layers': 2, 'num_decoder_layers': 2, 'dim_feedforward': 2048, 'n_heads': 6, 'factor': 1, 'moving_avg': 21, 'lr': 4.7651748046751395e-05, 'weight_decay': 0.0, 'dropout': 0.0982708428790269}
                # tuned_params = {'context_length_factor': 2, 'batch_size': 128, 'num_encoder_layers': 2, 'num_decoder_layers': 3, 'd_model': 128, 'n_heads': 6}
                
                config["model"]["distr_output"]["kwargs"].update({k: v for k, v in tuned_params.items() if k in config["model"]["distr_output"]["kwargs"]})
                config["dataset"].update({k: v for k, v in tuned_params.items() if k in config["dataset"]})
                # config["model"][args.model].update({k: v for k, v in tuned_params.items() if k in config["model"][args.model]})
                
                config["trainer"].update({k: v for k, v in tuned_params.items() if k in trainer_params})
                
                data_module.batch_size = config["dataset"]["batch_size"]
                
                model_hparams.update(
                    {k: v for k, v in tuned_params.items() if k in estimator_params})
                
                context_length_factor = tuned_params.get('context_length_factor', config["dataset"].get("context_length_factor", None)) # Default to config or 2 if not in trial/config
                if context_length_factor:
                    data_module.context_length = int(context_length_factor * data_module.prediction_length)
                    logging.info(f"Setting context_length to {context_length_factor} times the prediction length {data_module.prediction_length} = {data_module.context_length} from tuned parameters.")
                else:
                    data_module.context_length = config["dataset"]["context_length"]
                    logging.info(f"Setting context_length to default value {data_module.context_length} from default values.")
                
                data_module.freq = config["dataset"]["resample_freq"]
            except FileNotFoundError as e:
                logging.warning(e)
                found_tuned_params = False
            except KeyError as e:
                logging.warning(f"KeyError accessing Optuna config for tuned params: {e}. Using defaults.")
                found_tuned_params = False
        else:
            found_tuned_params = False 
            
        # TODO HIGH lr and weight_decay are not being set properly during tuning or training!!!
        
        if found_tuned_params:
            logging.info(f"Updating estimator {args.model.capitalize()} kwargs with tuned parameters {tuned_params}")
        else:
            logging.info(f"Updating estimator {args.model.capitalize()} kwargs with default parameters")
            if "context_length_factor" in config["model"][args.model]:
                data_module.context_length = int(config["model"][args.model]["context_length_factor"] * data_module.prediction_length)
                del config["model"][args.model]["context_length_factor"]
            
        # Use the get_checkpoint function to handle checkpoint finding
        # TODO can we grab the checkpoint from the winning hyperparameter trial?
        if args.checkpoint:
            # Set up parameters for checkpoint finding
            metric = config.get("trainer", {}).get("monitor_metric", "val_loss")
            mode = config.get("optuna", {}).get("direction", "minimize")
            mode_mapping = {"minimize": "min", "maximize": "max"}
            mode = mode_mapping.get(mode, "min")

            base_checkpoint_dir = os.path.join(log_dir, project_name)
            logging.info(f"Checkpoint selection: Monitoring metric '{metric}' with mode '{mode}' in directory '{base_checkpoint_dir}'")
        
            checkpoint_path = get_checkpoint(args.checkpoint, metric, mode, base_checkpoint_dir)
            checkpoint_hparams = load_estimator_from_checkpoint(checkpoint_path, LightningModuleClass, config, args.model)
            
            model_hparams.update(checkpoint_hparams["init_args"]["model_config"])
            
            # Update DataModule params
            data_module.prediction_length = checkpoint_hparams["prediction_length_int"]
            data_module.context_length = checkpoint_hparams["context_length_int"]
            data_module.freq = checkpoint_hparams["freq_str"]
            
            # check if any new hyperparameters are incompatible with data_module
            # core_data_module_params = ["num_feat_dynamic_real", "num_feat_static_real", "num_feat_static_cat",
            #                       "cardinality", "embedding_dimension"]
            # data_module_sig = inspect.signature(DataModule.__init__)
            # data_module_params = [param.name for param in data_module_sig.parameters.values()]
            
            # TODO JUAN we don't expect these to be equal, num_feat_dynamic_real is changed internally in estimator.py:create_lightning_module, 
            # num_feat_static_real and num_feat_static_cat are set to the max of the data_module value and 1,
            # cardinality is a list and a tuple, so this needs to be handled differently.
            # incompatible_params = []
            # for param in core_data_module_params:
            #     if ((param in checkpoint_hparams["init_args"]["model_config"]) 
            #         and (checkpoint_hparams["init_args"]["model_config"][param] is not None)
            #         and (getattr(data_module, param) is not None) 
            #         and (checkpoint_hparams["init_args"]["model_config"][param] != getattr(data_module, param))):
            #         incompatible_params.append((param, checkpoint_hparams["init_args"]["model_config"][param], getattr(data_module, param)))
            
            # if incompatible_params:
            #     raise TypeError(f"Checkpoint parameters and data module parameters {incompatible_params} are incompatible.")
            
            logging.info(f"Updating estimator {args.model.capitalize()} kwargs with checkpoint parameters {checkpoint_hparams['init_args']['model_config']}.")
        else:
            checkpoint_path = None
            
        # Apply command-line overrides AFTER potentially loading tuned params
        if args.override:
            logging.info(f"Applying command-line overrides (final step): {args.override}")
            for override_item in args.override:
                try:
                    if '=' in override_item:
                        # Case 1: key=value provided - Use command-line value
                        key_path, value_str = override_item.split('=', 1)
                        keys = key_path.split('.')

                        # Try to parse value (int, float, bool, or string)
                        try:
                            value = yaml.safe_load(value_str) # Handles basic types
                        except yaml.YAMLError:
                            value = value_str # Keep as string if parsing fails

                        if keys[0] == "model":
                            keys[1] = args.model # Ensure the first key is the model name
                        
                        key_path = '.'.join(keys)  # Reconstruct the key path for logging
                        
                        # Navigate nested dictionary and set value
                        d = config
                        for key in keys[:-1]:
                            d = d.setdefault(key, {})
                            if not isinstance(d, dict):
                                logging.warning(f"Overriding non-dictionary key '{key}' in path '{key_path}'. Existing value will be replaced.")
                                parent_d = config
                                for parent_key in keys[:keys.index(key)]: parent_d = parent_d[parent_key]
                                parent_d[key] = {}
                                d = parent_d[key]

                        last_key = keys[-1]
                        d[last_key] = value
                        logging.info(f"  - Final override applied (from command line): '{key_path}' = {value} (type: {type(value).__name__})")

                    else:
                        # Case 2: Only key provided - Revert to original YAML value
                        key_path = override_item
                        keys = key_path.split('.')
                        if keys[0] == "model":
                            keys[1] = args.model # Ensure the first key is the model name
                        key_path = '.'.join(keys)  # Reconstruct the key path for logging
                        
                        # Navigate original YAML config to get the value
                        original_d = original_yaml_config
                        found_original = True
                        for key in keys:
                            if isinstance(original_d, dict) and key in original_d:
                                original_d = original_d[key]
                            else:
                                logging.warning(f"  - Override key '{key_path}' not found in original YAML config. Cannot revert.")
                                found_original = False
                                break

                        if found_original:
                            original_value = original_d # The value found at the end of the path

                            # Navigate current config to set the value
                            d = config
                            for key in keys[:-1]:
                                d = d.setdefault(key, {})
                                if not isinstance(d, dict):
                                     # This case is less likely when reverting, but handle defensively
                                     logging.warning(f"Overriding non-dictionary key '{key}' in path '{key_path}' while reverting. Existing value will be replaced.")
                                     parent_d = config
                                     for parent_key in keys[:keys.index(key)]: parent_d = parent_d[parent_key]
                                     parent_d[key] = {}
                                     d = parent_d[key]

                            last_key = keys[-1]
                            d[last_key] = original_value
                            logging.info(f"  - Final override applied (reverted to YAML): '{key_path}' = {original_value} (type: {type(original_value).__name__})")

                except ValueError:
                     # This error should now only happen if split fails unexpectedly
                     logging.warning(f"  - Skipping invalid override format: '{override_item}'.")
                except Exception as e:
                    logging.error(f"  - Error applying override '{override_item}': {e}", exc_info=True)
            
            if "context_length_factor" in config["dataset"]:
                data_module.context_length = int(config["dataset"]["context_length_factor"] * data_module.prediction_length)
                data_module.context_length = int(2 * data_module.prediction_length)
        
            if "batch_size" in config["dataset"]:
                data_module.batch_size = config["dataset"]["batch_size"]
                
            if "sampler" in config["dataset"]:
                data_module.sampler = config["dataset"]["sampler"]
                
        if args.mode == "train" and args.checkpoint is not None:
            logging.info("Restarting training from checkpoint, updating max_epochs accordingly.")
            if os.path.basename(checkpoint_path) == "last.ckpt":
                if torch.cuda.is_available():
                    device = None
                else:
                    device = "cpu"
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                last_epoch = checkpoint.get("epoch", 0)
            else:
                last_epoch = int(re.search("(?<=epoch=)\\d+", os.path.basename(checkpoint_path)).group())
            
            config["trainer"]["max_epochs"] += last_epoch
        
        # --- Instantiate Callbacks ---
        # We need to do this BEFORE creating the estimator,
        # so the instantiated list can be placed in trainer_kwargs.
        import importlib
        logging.info("Instantiating callbacks from configuration...")
        instantiated_callbacks = []
        if 'callbacks' in config and isinstance(config['callbacks'], dict):
            for cb_name, cb_config in config['callbacks'].items():
                 # Skip disabled callbacks explicitly marked as enabled: false
                 if isinstance(cb_config, dict) and cb_config.get('enabled', True) is False:
                      logging.info(f"Skipping disabled callback: {cb_name}")
                      continue

                 # Handle DeadNeuronMonitor specifically based on 'enabled' flag
                 if cb_name == 'dead_neuron_monitor' and isinstance(cb_config, dict) and cb_config.get('enabled', False) is True:
                     try:
                         callback_instance = DeadNeuronMonitor()
                         instantiated_callbacks.append(callback_instance)
                         logging.info("Instantiated callback: DeadNeuronMonitor")
                         continue # Skip to the next callback in the loop
                     except Exception as e:
                         logging.error(f"Error instantiating DeadNeuronMonitor: {e}", exc_info=True)
                         continue # Continue to the next callback even if this one failed

                 if isinstance(cb_config, dict) and 'class_path' in cb_config:
                    try:
                        module_path, class_name = cb_config['class_path'].rsplit('.', 1)
                        CallbackClass = getattr(importlib.import_module(module_path), class_name)
                        init_args = cb_config.get('init_args', {})

                        # Resolve dirpath for ModelCheckpoint if necessary
                        if class_name == "ModelCheckpoint" and 'dirpath' in init_args:
                            if isinstance(init_args['dirpath'], str) and '${logging.checkpoint_dir}' in init_args['dirpath']:
                                # Note: checkpoint_dir was defined earlier based on unique_id
                                init_args['dirpath'] = init_args['dirpath'].replace('${logging.checkpoint_dir}', checkpoint_dir)
                            # Ensure absolute path if not already
                            if not os.path.isabs(init_args['dirpath']):
                                project_root = config.get('experiment', {}).get('project_root', '.')
                                init_args['dirpath'] = os.path.abspath(os.path.join(project_root, init_args['dirpath']))
                            os.makedirs(init_args['dirpath'], exist_ok=True) # Ensure dir exists

                        callback_instance = CallbackClass(**init_args)
                        instantiated_callbacks.append(callback_instance)
                        logging.info(f"Instantiated callback: {class_name}")
                    except Exception as e:
                        logging.error(f"Error instantiating callback {cb_name}: {e}", exc_info=True)
                 # Handle simple boolean flags if needed (though YAML structure is preferred)
                 # elif isinstance(cb_config, bool) and cb_config is True: ...
        else:
             logging.info("No callbacks dictionary found in config or it's not a dictionary.")
        # --- End Callback Instantiation ---
        
        # Ensure trainer_kwargs exists and add the instantiated callbacks list
        config.setdefault("trainer", {})
        config["trainer"]["callbacks"] = instantiated_callbacks
        logging.info(f"Assigned {len(instantiated_callbacks)} callbacks to config['trainer']['callbacks'].")

        # Prepare all arguments in a dictionary for the Estimator
    
    # wait until data_module attributes have been updated to generate splits
    if rank_zero_only.rank == 0:
        logging.info("Preparing data for tuning")
        if args.reload_data or not os.path.exists(data_module.train_ready_data_path):
            data_module.generate_datasets()
            reload = True
        else:
            reload = False
       
    else:
        reload = False
    
    # other ranks should wait for this one 
    # Pass the rank determined at the beginning of main() to handle both tuning and training modes
    data_module.generate_splits(save=True, reload=reload, splits=["train", "val", "test"], rank=rank)
    
    if args.mode in ["train", "test"]:
        estimator_kwargs = {
            "freq": data_module.freq,
            "prediction_length": data_module.prediction_length,
            "num_feat_dynamic_real": data_module.num_feat_dynamic_real,
            "num_feat_static_cat": data_module.num_feat_static_cat,
            "cardinality": data_module.cardinality,
            "num_feat_static_real": data_module.num_feat_static_real,
            "input_size": data_module.num_target_vars,
            "scaling": "False", #if model_hparams.get("scaling", "True") == "True" else False, # TODO back to std, ALLOW US TO SPECIFY SCALING, ALSO WHY STRING NOT B00L Scaling handled externally or internally by TACTiS
            "lags_seq": [0], #model_hparams.get("lags_seq", [0]), #TODOconfig["model"][args.model]["lags_seq"], # TACTiS doesn't typically use lags
            "time_features": [second_of_minute, minute_of_hour, hour_of_day, day_of_year],
            "batch_size": data_module.batch_size,
            "num_batches_per_epoch": config["trainer"].get("limit_train_batches", 1000),
            "context_length": data_module.context_length,
            "train_sampler": SequentialSampler(min_past=data_module.context_length, min_future=data_module.prediction_length)
                if config["dataset"].get("sampler", "sequential") == "sequential"
                else ExpectedNumInstanceSampler(num_instances=1.0, min_past=data_module.context_length, min_future=data_module.prediction_length),
            "validation_sampler": ValidationSplitSampler(min_past=data_module.context_length, min_future=data_module.prediction_length),
            "trainer_kwargs": config["trainer"],
        }
        
        n_training_samples = 0
        for ds in data_module.train_dataset:
            a, b = estimator_kwargs["train_sampler"]._get_bounds(ds["target"])
            n_training_samples += (b - a + 1)
        
        n_training_steps = np.ceil(n_training_samples / data_module.batch_size).astype(int)
        assert estimator_kwargs["num_batches_per_epoch"] is None or isinstance(estimator_kwargs["num_batches_per_epoch"], int)
        if estimator_kwargs["num_batches_per_epoch"] is not None:
            n_training_steps = min(n_training_steps, estimator_kwargs["num_batches_per_epoch"])
        
        # TODO JUAN PATCH FOR TACTIS
        estimator_kwargs["num_batches_per_epoch"] = n_training_steps
            
        # Log warning if using random sampler with null limit_train_batches
        if (config["dataset"].get("sampler", "sequential") == "random" and
            estimator_kwargs["num_batches_per_epoch"] is None):
            logging.warning("Using random sampler (ExpectedNumInstanceSampler) with limit_train_batches=null. "
                          "Consider setting an explicit integer value for limit_train_batches to avoid potential issues.")
        
        if  "d_model" in model_hparams: # and "dim_feedforward" not in model_hparams
            # set dim_feedforward to 4x the d_model found in this trial
            model_hparams["dim_feedforward"] = model_hparams["d_model"] * 4
            logging.info(f"Updating estimator {args.model.capitalize()} dim_feedforward with 4x d_model = {model_hparams['dim_feedforward']}")
        elif "d_model" in estimator_params and estimator_sig.parameters["d_model"].default is not inspect.Parameter.empty:
            # if d_model is not contained in the trial but is a paramter, get the default
            model_hparams["dim_feedforward"] = estimator_sig.parameters["d_model"].default * 4
            logging.info(f"Updating estimator {args.model.capitalize()} dim_feedforward with 4x estimator default d_model = {model_hparams['dim_feedforward']}")

        # Add model-specific arguments. Note that some params, such as num_feat_dynamic_real, are changed within Model, and so can't be used for estimator class
        estimator_kwargs.update({k: v for k, v in model_hparams.items() if k in estimator_params and not hasattr(data_module, k) })
        
        # Add distr_output only if the model is NOT tactis
        if args.model != 'tactis' and "distr_output" not in estimator_kwargs:
            estimator_kwargs["distr_output"] = DistrOutputClass(dim=data_module.num_target_vars, **config["model"]["distr_output"]["kwargs"]) # TODO or checkpoint_hparams["model_config"]["distr_output"]
        elif args.model == "spacetimeformer":
            for k in estimator_kwargs["distr_output"].args_dim:
                estimator_kwargs["distr_output"].args_dim[k] *= estimator_kwargs["input_size"]
            
        # elif 'distr_output' in estimator_kwargs:
        #      del estimator_kwargs['distr_output']
        
        logging.info(f"Using final estimator_kwargs:\n {estimator_kwargs}")
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
        from wind_forecasting.tuning import tune_model
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

        # Callbacks from config will be handled by MLTuningObjective.

        # Normal execution - pass the OOM protection wrapper and constructed storage URL
        tune_model(model=args.model, config=config, # Pass full config here for model/trainer params
                   study_name=db_setup_params["base_study_prefix"],
                   optuna_storage=optuna_storage, # Pass the constructed storage object
                   lightning_module_class=LightningModuleClass,
                   estimator_class=EstimatorClass,
                   distr_output_class=DistrOutputClass,
                   data_module=data_module,
                   max_epochs=config["optuna"]["max_epochs"],
                   # Use base_limit_train_batches if available, otherwise fallback to limit_train_batches (or None)
                   limit_train_batches=config["optuna"].get("base_limit_train_batches", config["optuna"].get("limit_train_batches", None)),
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
        # Callbacks are now instantiated and added to estimator_kwargs above
        logging.info(f"Training model with a total of {n_training_steps} training steps.")
        estimator.train(
            training_data=data_module.train_dataset,
            validation_data=data_module.val_dataset,
            forecast_generator=forecast_generator,
            ckpt_path=checkpoint_path
            # shuffle_buffer_length=1024
        )
        # train_output.trainer.checkpoint_callback.best_model_path
        logging.info("Model training completed.")
    elif args.mode == "test":
        logging.info("Starting model testing...")
        # %% TEST MODEL

        test_model(data_module=data_module,
                    checkpoint=checkpoint_path,
                    lightning_module_class=LightningModuleClass, 
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