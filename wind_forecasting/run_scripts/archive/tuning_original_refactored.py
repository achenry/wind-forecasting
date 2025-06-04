import os
from lightning.pytorch.utilities.model_summary import summarize
from gluonts.evaluation import MultivariateEvaluator
from gluonts.model.forecast_generator import DistributionForecastGenerator, SampleForecastGenerator
from gluonts.time_feature._base import second_of_minute, minute_of_hour, hour_of_day, day_of_year
from gluonts.transform import ExpectedNumInstanceSampler, SequentialSampler, ValidationSplitSampler
import logging
import torch
import gc
import time # Added for load_study retry delay
import inspect
from itertools import product
import subprocess
from datetime import datetime
import re # Added for regex operations
import pandas as pd # Added for pd.Timedelta
# Imports for Optuna
import optuna
from optuna import create_study, load_study
from optuna.study import MaxTrialsCallback
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner, PercentilePruner, PatientPruner, SuccessiveHalvingPruner, NopPruner

import lightning.pytorch as pl # Import pl alias
from optuna.trial import TrialState # Added for checking trial status
import wandb
from lightning.pytorch.loggers import WandbLogger

# Import utility modules
from wind_forecasting.utils.callbacks import DeadNeuronMonitor, SafePruningCallback
from wind_forecasting.utils.optuna_sampler_pruner_utils import OptunaSamplerPrunerPersistence
from wind_forecasting.utils.optuna_visualization import launch_optuna_dashboard, log_optuna_visualizations_to_wandb
from wind_forecasting.utils.optuna_table import log_detailed_trials_table_to_wandb
from wind_forecasting.utils.trial_utils import handle_trial_with_oom_protection
from wind_forecasting.utils.path_utils import resolve_path, flatten_dict
from wind_forecasting.utils.tuning_config_utils import generate_db_setup_params, generate_optuna_dashboard_command
from wind_forecasting.utils.checkpoint_utils import (
    load_checkpoint, parse_epoch_from_checkpoint_path, determine_tactis_stage,
    extract_hyperparameters, prepare_model_init_args, load_model_state, set_tactis_stage
)
from wind_forecasting.utils.metrics_utils import (
    extract_metric_value, compute_evaluation_metrics,
    update_metrics_with_checkpoint_score, validate_metrics_for_return
)
from wind_forecasting.utils.tuning_helpers import (
    set_trial_seeds, update_data_module_params, regenerate_data_splits,
    prepare_feedforward_params, calculate_dynamic_limit_train_batches,
    create_trial_checkpoint_callback, setup_trial_callbacks
)

import random
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Functions moved to utility modules:
# - generate_db_setup_params -> tuning_config_utils.py
# - resolve_path -> path_utils.py  
# - flatten_dict -> path_utils.py
# - generate_optuna_dashboard_command -> tuning_config_utils.py
# - SafePruningCallback -> callbacks.py

class MLTuningObjective:
    def __init__(self, *, model, config, lightning_module_class, estimator_class,
                 distr_output_class, max_epochs, limit_train_batches, data_module,
                 metric, seed=42, tuning_phase=0, dynamic_params=None, study_config_params=None):
        self.model = model
        self.config = config
        self.lightning_module_class = lightning_module_class
        self.estimator_class = estimator_class
        self.distr_output_class = distr_output_class
        self.data_module = data_module
        self.metric = metric # TODO unused
        self.evaluator = MultivariateEvaluator(num_workers=None, custom_eval_fn=None)
        # self.metrics = []
        self.seed = seed
        self.tuning_phase = tuning_phase
        self.dynamic_params = dynamic_params
        self.study_config_params = study_config_params or {}

        self.config["trainer"]["max_epochs"] = max_epochs
        self.config["trainer"]["limit_train_batches"] = limit_train_batches
        # Don't override val_check_interval - let it use the YAML config value
        
        # Debug logging for max_epochs issue
        logging.info(f"MLTuningObjective.__init__: Setting trainer max_epochs={max_epochs}, limit_train_batches={limit_train_batches}")
        # self.config["trainer"]["val_check_interval"] = limit_train_batches
        
        # Store base values for dynamic calculation
        self.base_limit_train_batches = self.config["optuna"].get("base_limit_train_batches")
        self.base_batch_size = self.config["dataset"].get("base_batch_size")
        
        # Store pruning configuration
        self.pruning_enabled = "pruning" in config["optuna"] and config["optuna"]["pruning"].get("enabled", False)
        if self.pruning_enabled:
            logging.info(f"Pruning is enabled using {config['optuna']['pruning'].get('type', 'hyperband')} pruner")
        else:
            logging.info("Pruning is disabled")

        # Add GPU monitoring
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.device = torch.cuda.current_device()
            self.gpu_name = torch.cuda.get_device_name(self.device)
            self.total_memory = torch.cuda.get_device_properties(self.device).total_memory
            logging.info(f"GPU monitoring initialized for {self.gpu_name}")

    def log_gpu_stats(self, stage=""):
        """Log GPU memory usage at different stages of training"""
        if not self.gpu_available:
            return

        # Memory in GB
        allocated = torch.cuda.memory_allocated(self.device) / 1e9
        reserved = torch.cuda.memory_reserved(self.device) / 1e9
        max_allocated = torch.cuda.max_memory_allocated(self.device) / 1e9
        total = self.total_memory / 1e9

        # Calculate utilization percentage
        utilization_percent = (allocated / total) * 100

        logging.info(f"GPU Stats {stage}: "
                    f"Current Memory: {allocated:.2f}GB ({utilization_percent:.1f}%), "
                    f"Reserved: {reserved:.2f}GB, "
                    f"Peak: {max_allocated:.2f}GB, "
                    f"Total: {total:.2f}GB")

    def __call__(self, trial):
        # Set random seeds for reproducibility
        trial_seed = set_trial_seeds(trial.number, self.seed)

        # Initialize wandb logger for this trial only on rank 0
        wandb_logger_trial = None # Initialize to None for non-rank-0 workers

        # Log GPU stats at the beginning of the trial
        self.log_gpu_stats(stage=f"Trial {trial.number} Start")
      
        params = self.estimator_class.get_params(trial, self.tuning_phase,
                                                 dynamic_kwargs=self.dynamic_params)

        # Update DataModule parameters based on trial tuned parameters
        needs_split_regeneration = update_data_module_params(
            self.data_module, params, self.config, trial.number
        )
        
        # Regenerate splits if needed
        if needs_split_regeneration:
            regenerate_data_splits(self.data_module, trial.number)
        else:
            logging.info(f"Trial {trial.number}: No DataModule parameter changes. Using existing splits.")
        # Prepare feedforward parameters based on d_model
        prepare_feedforward_params(params, self.estimator_class, self.config, self.model)
        
        logging.info(f"Testing params {tuple((k, v) for k, v in params.items())}")

        # Calculate dynamic limit_train_batches if base values are available
        dynamic_limit_train_batches = calculate_dynamic_limit_train_batches(
            params, self.config, self.base_limit_train_batches, self.base_batch_size
        )
        
        if dynamic_limit_train_batches is not None:
            # Update trainer config with dynamic value
            self.config["trainer"]["limit_train_batches"] = dynamic_limit_train_batches
            
            # Scale val_check_interval proportionally with limit_train_batches to maintain same validation frequency
            base_val_check_interval = self.config["trainer"].get("val_check_interval", 5000)
            scaling_factor = self.base_batch_size / current_batch_size
            dynamic_val_check_interval = max(1, round(base_val_check_interval * scaling_factor))
            
            # Ensure val_check_interval doesn't exceed limit_train_batches
            if dynamic_val_check_interval > dynamic_limit_train_batches:
                dynamic_val_check_interval = dynamic_limit_train_batches
                logging.info(f"Capped val_check_interval to limit_train_batches={dynamic_limit_train_batches}")
            
            self.config["trainer"]["val_check_interval"] = dynamic_val_check_interval
            logging.info(f"Scaled val_check_interval: {base_val_check_interval} -> {dynamic_val_check_interval} (scaling_factor={scaling_factor:.2f})")
        else:
            logging.info(f"Using static limit_train_batches: {self.config['trainer']['limit_train_batches']}")
            
            # Even with static values, ensure val_check_interval is valid
            limit_train_batches = self.config["trainer"]["limit_train_batches"]
            val_check_interval = self.config["trainer"].get("val_check_interval", 5000)
            
            if val_check_interval > limit_train_batches:
                self.config["trainer"]["val_check_interval"] = limit_train_batches
                logging.info(f"Adjusted val_check_interval from {val_check_interval} to {limit_train_batches} (must be <= limit_train_batches)")

        self.config["model"]["distr_output"]["kwargs"].update({k: v for k, v in params.items() if k in self.config["model"]["distr_output"]["kwargs"]})
        self.config["dataset"].update({k: v for k, v in params.items() if k in self.config["dataset"]})
        self.config["model"][self.model].update({k: v for k, v in params.items() if k in self.config["model"][self.model]})
        self.config["trainer"].update({k: v for k, v in params.items() if k in self.config["trainer"]})
        
        # Create trial-specific checkpoint callback
        trial_checkpoint_callback = create_trial_checkpoint_callback(
            trial.number, self.config, self.model
        )
        
        # Setup all callbacks for this trial
        final_callbacks = setup_trial_callbacks(
            trial, self.config, self.model, self.pruning_enabled, trial_checkpoint_callback
        )

        trial_trainer_kwargs = {k: v for k, v in self.config["trainer"].items() if k != 'callbacks'}
        trial_trainer_kwargs["callbacks"] = final_callbacks
        
        # Log the trainer kwargs to debug max_epochs issue
        logging.info(f"Trial {trial.number}: trial_trainer_kwargs = {trial_trainer_kwargs}")
        logging.info(f"Trial {trial.number}: max_epochs from trainer_kwargs = {trial_trainer_kwargs.get('max_epochs', 'NOT SET')}")
        logging.info(f"Trial {trial.number}: limit_train_batches from trainer_kwargs = {trial_trainer_kwargs.get('limit_train_batches', 'NOT SET')}")
        
        # CRITICAL DEBUG: Check if limit_train_batches is being interpreted as a fraction
        ltb = trial_trainer_kwargs.get('limit_train_batches')
        if ltb is not None and isinstance(ltb, (int, float)):
            if ltb < 1.0:
                logging.warning(f"Trial {trial.number}: limit_train_batches={ltb} is < 1.0 - PyTorch Lightning will interpret this as a FRACTION of the dataset!")
            else:
                logging.info(f"Trial {trial.number}: limit_train_batches={ltb} will be interpreted as an absolute number of batches")

        logging.debug(f"Final callbacks passed to estimator: {[type(cb).__name__ for cb in trial_trainer_kwargs['callbacks']]}")

        # Remove monitor_metric if it exists, as it's handled by ModelCheckpoint
        if "monitor_metric" in trial_trainer_kwargs:
            del trial_trainer_kwargs["monitor_metric"]

        # Initialize W&B for ALL workers
        try:

            # Clean and flatten the parameters for logging without duplicates
            cleaned_params = {}
            model_prefix = f"{self.model}."
            config_prefix = "model_config."

            # First pass: collect all non-prefixed keys and model-prefixed keys
            non_prefixed_params = {}
            model_prefixed_params = {}
            
            for k, v in trial.params.items():
                # Remove model prefix if present
                if k.startswith(model_prefix):
                    stripped_key = k[len(model_prefix):]
                    model_prefixed_params[stripped_key] = v
                else:
                    non_prefixed_params[k] = v

            # Second pass: merge with priority to non-prefixed params
            for k, v in non_prefixed_params.items():
                cleaned_params[k] = v
                
            # Add model-prefixed params only if not already in cleaned_params
            for k, v in model_prefixed_params.items():
                if k.startswith(config_prefix):
                    base_key = k[len(config_prefix):]
                    if base_key not in cleaned_params:
                        cleaned_params[base_key] = v
                else:
                    if k not in cleaned_params:
                        cleaned_params[k] = v

            # Add study config params to WandB config
            cleaned_params.update(self.study_config_params)

            # Add the dynamically calculated limit_train_batches for this trial
            current_batch_size = params.get('batch_size', self.config["dataset"].get("batch_size", 128))
            if self.base_limit_train_batches is not None and self.base_batch_size is not None and self.base_batch_size > 0:
                calculated_limit = max(1, round(self.base_limit_train_batches * self.base_batch_size / current_batch_size))
                cleaned_params["actual_limit_train_batches"] = calculated_limit
            else:
                cleaned_params["actual_limit_train_batches"] = self.config["trainer"]["limit_train_batches"]

            cleaned_params["optuna_trial_number"] = trial.number

            project_name = f"{self.config['experiment'].get('project_name', 'wind_forecasting')}_{self.model}"
            group_name = self.config['experiment']['run_name']
            # Construct unique run name and tags
            run_name = f"{self.config['experiment']['run_name']}_rank_{os.environ.get('WORKER_RANK', '0')}_trial_{trial.number}"
            
            # Initialize a new W&B run for this specific trial
            wandb.init(
                # Core identification
                project=project_name,
                entity=self.config['logging'].get('entity', None),
                group=group_name,
                name=run_name,
                job_type="optuna_trial",
                dir=self.config['logging']['wandb_dir'],
                # Configuration and Metadata
                config=cleaned_params, # Use the cleaned dictionary
                tags=[self.model] + self.config['experiment'].get('extra_tags', []),
                notes=f"Optuna trial {trial.number} (Rank {os.environ.get('WORKER_RANK', '0')}) for study: {self.config['experiment'].get('notes', '')}",
                # Logging and Behavior
                save_code=self.config['optuna'].get('save_trial_code', False),
                mode=self.config['logging'].get('wandb_mode', 'online'),
                reinit="finish_previous"
            )
            logging.info(f"Rank {os.environ.get('WORKER_RANK', '0')}: Initialized W&B run '{run_name}' for trial {trial.number}")

            # Create a WandbLogger using the current W&B run
            # log_model=False as we only want metrics for this trial logger
            wandb_logger_trial = WandbLogger(log_model=False, experiment=wandb.run, save_dir=self.config['logging']['wandb_dir'])
            logging.info(f"Rank {os.environ.get('WORKER_RANK', '0')}: Created WandbLogger for trial {trial.number}")

            # Add the trial-specific logger to the trainer kwargs for this worker
            trial_trainer_kwargs["logger"] = wandb_logger_trial
            logging.info(f"Rank {os.environ.get('WORKER_RANK', '0')}: Added trial-specific WandbLogger to trainer_kwargs for trial {trial.number}")

        except Exception as e:
            logging.error(f"Rank {os.environ.get('WORKER_RANK', '0')}: Failed to initialize W&B or create logger for trial {trial.number}: {e}", exc_info=True)
            # Ensure wandb_logger_trial remains None if setup fails
            wandb_logger_trial = None

        # Verify GPU configuration before creating estimator
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            logging.info(f"Creating estimator using GPU {device}: {torch.cuda.get_device_name(device)}")

            # Ensure we have the right GPU configuration in trainer_kwargs
            if "devices" in self.config["trainer"] and self.config["trainer"]["devices"] > 1:
                if "CUDA_VISIBLE_DEVICES" in os.environ and len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 1:
                    logging.warning(f"Overriding trainer devices={self.config['trainer']['devices']} to 1 due to CUDA_VISIBLE_DEVICES")
                    self.config["trainer"]["devices"] = 1
                    self.config["trainer"]["strategy"] = "auto"
        else:
            logging.warning("No CUDA available for estimator creation")

        context_length_factor = params.get('context_length_factor', self.config["dataset"].get("context_length_factor", 2)) # Default to config or 2 if not in trial/config
        context_length = int(context_length_factor * self.data_module.prediction_length)

        # Estimator Arguments to handle difference between models
        estimator_kwargs = {
            "freq": self.data_module.freq,
            "prediction_length": self.data_module.prediction_length,
            "context_length": context_length,
            "num_feat_dynamic_real": self.data_module.num_feat_dynamic_real,
            "num_feat_static_cat": self.data_module.num_feat_static_cat,
            "cardinality": self.data_module.cardinality,
            "num_feat_static_real": self.data_module.num_feat_static_real,
            "input_size": self.data_module.num_target_vars,
            "scaling": False,
            "lags_seq": [0],
            "use_lazyframe": False,
            "batch_size": current_batch_size,  # Use the current batch size from params, not config
            "num_batches_per_epoch": trial_trainer_kwargs["limit_train_batches"], # Use value from trial_trainer_kwargs
            "base_batch_size_for_scheduler_steps": self.config["dataset"].get("base_batch_size", 512), # Use base_batch_size from config
            "base_limit_train_batches": self.base_limit_train_batches, # Pass base_limit_train_batches for conditional scaling
            "train_sampler": SequentialSampler(min_past=context_length, min_future=self.data_module.prediction_length)
                if self.config["optuna"].get("sampler", "random") == "sequential"
                else ExpectedNumInstanceSampler(num_instances=1.0, min_past=context_length, min_future=self.data_module.prediction_length),
            "validation_sampler": ValidationSplitSampler(min_past=context_length, min_future=self.data_module.prediction_length),
            "time_features": [second_of_minute, minute_of_hour, hour_of_day, day_of_year],
            # Include distr_output initially, will be removed conditionally
            "distr_output": self.distr_output_class(dim=self.data_module.num_target_vars, **self.config["model"]["distr_output"]["kwargs"]),
            "trainer_kwargs": trial_trainer_kwargs, # Pass the trial-specific kwargs
            "num_parallel_samples": self.config["model"][self.model].get("num_parallel_samples", 100) if self.model == 'tactis' else 100, # Default 100 if not specified
        }
        
        # Debug logging for epoch calculation issue
        logging.info(f"Trial {trial.number}: Critical DataLoader parameters:")
        logging.info(f"  - batch_size: {estimator_kwargs['batch_size']}")
        logging.info(f"  - num_batches_per_epoch: {estimator_kwargs['num_batches_per_epoch']}")
        logging.info(f"  - trainer max_epochs: {trial_trainer_kwargs.get('max_epochs', 'NOT SET')}")
        logging.info(f"  - trainer limit_train_batches: {trial_trainer_kwargs.get('limit_train_batches', 'NOT SET')}")
        logging.info(f"  - Expected total batches: {trial_trainer_kwargs.get('max_epochs', 0) * estimator_kwargs['num_batches_per_epoch']}")
        
        # Calculate actual number of training samples
        n_training_samples = 0
        for ds in self.data_module.train_dataset:
            a, b = estimator_kwargs["train_sampler"]._get_bounds(ds["target"])
            n_training_samples += (b - a + 1)
        actual_batches = np.ceil(n_training_samples / estimator_kwargs['batch_size']).astype(int)
        logging.info(f"  - Total training samples: {n_training_samples}")
        logging.info(f"  - Actual batches from data: {actual_batches}")
        logging.info(f"  - DataLoader will provide: min({actual_batches}, {estimator_kwargs['num_batches_per_epoch']}) = {min(actual_batches, estimator_kwargs['num_batches_per_epoch'])} batches/epoch")
        
        # CRITICAL: Check if we're using ExpectedNumInstanceSampler with Cyclic wrapper
        if isinstance(estimator_kwargs["train_sampler"], ExpectedNumInstanceSampler):
            logging.info(f"  - Using ExpectedNumInstanceSampler with Cyclic wrapper - batches will cycle indefinitely")
            logging.info(f"  - DataLoader will provide EXACTLY {estimator_kwargs['num_batches_per_epoch']} batches per epoch")
        # Add model-specific arguments from the default config YAML
        # CRITICAL: Preserve the dynamically calculated num_batches_per_epoch
        dynamic_num_batches = estimator_kwargs["num_batches_per_epoch"]
        estimator_kwargs.update(self.config["model"][self.model])
        estimator_kwargs["num_batches_per_epoch"] = dynamic_num_batches  # Restore dynamic value
        
        if "num_batches_per_epoch" not in self.config["model"][self.model]:
            self.config["model"][self.model]["num_batches_per_epoch"] = dynamic_num_batches
            logging.info(f"Trial {trial.number}: Added num_batches_per_epoch={dynamic_num_batches} to self.config['model'][self.model] for checkpointing stability.")
        else:
            logging.warning(f"Trial {trial.number}: Overriding config num_batches_per_epoch={self.config['model'][self.model].get('num_batches_per_epoch')} with dynamic value={dynamic_num_batches}")

        # Add model-specific tunable hyperparameters suggested by Optuna trial
        # Extract estimator parameters from the class signature
        estimator_sig = inspect.signature(self.estimator_class.__init__)
        estimator_params = [param.name for param in estimator_sig.parameters.values()]
        valid_estimator_params = set(estimator_params)
        filtered_params = {
            k: v for k, v in params.items()
            if k in valid_estimator_params and k != 'context_length_factor'
        }
        logging.info(f"Trial {trial.number}: Updating estimator_kwargs with filtered params: {list(filtered_params.keys())}")
        estimator_kwargs.update(filtered_params)

        # TACTiS manages its own distribution output internally, remove if present
        if self.model == 'tactis' and 'distr_output' in estimator_kwargs:
            estimator_kwargs.pop('distr_output')

        # Get the metric key from config
        metric_to_return = self.config.get("trainer", {}).get("monitor_metric", "val_loss") # Default to val_loss
        
        logging.info(f"Trial {trial.number}: Instantiating estimator '{self.model}' with final args: {list(estimator_kwargs.keys())}")
        logging.info(f"Trial {trial.number}: FINAL num_batches_per_epoch being passed to estimator: {estimator_kwargs.get('num_batches_per_epoch', 'NOT SET')}")
        
        # Debug max_epochs issue - log the actual trainer_kwargs being passed
        if "trainer_kwargs" in estimator_kwargs:
            logging.info(f"Trial {trial.number}: trainer_kwargs being passed to estimator: {estimator_kwargs['trainer_kwargs']}")
            logging.info(f"Trial {trial.number}: max_epochs in trainer_kwargs: {estimator_kwargs['trainer_kwargs'].get('max_epochs', 'NOT SET')}")
            logging.info(f"Trial {trial.number}: limit_train_batches in trainer_kwargs: {estimator_kwargs['trainer_kwargs'].get('limit_train_batches', 'NOT SET')}")

        agg_metrics = None

        try:
            # Create estimator
            try:
                estimator = self.estimator_class(**estimator_kwargs)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logging.error(f"Trial {trial.number} - CUDA OOM during estimator creation: {str(e)}")
                    if wandb.run is not None:
                        wandb.finish(exit_code=1)
                    raise optuna.exceptions.TrialPruned(f"Trial pruned due to CUDA OOM during estimator creation: {e}")
                raise
            except Exception as e:
                logging.error(f"Trial {trial.number} - Error creating estimator: {str(e)}", exc_info=True)
                raise

            # Log GPU stats before training
            self.log_gpu_stats(stage=f"Trial {trial.number} Before Training")

            # Create Forecast Generator
            try:
                if self.model == 'tactis':
                    logging.info(f"Trial {trial.number}: Using SampleForecastGenerator for TACTiS model.")
                    forecast_generator = SampleForecastGenerator()
                else:
                    logging.info(f"Trial {trial.number}: Using DistributionForecastGenerator for {self.model} model.")
                    if not hasattr(estimator, 'distr_output'):
                        raise AttributeError(f"Estimator for model '{self.model}' is missing 'distr_output' attribute needed for DistributionForecastGenerator.")
                    forecast_generator = DistributionForecastGenerator(estimator.distr_output)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logging.error(f"Trial {trial.number} - CUDA OOM during forecast generator creation: {str(e)}")
                    if wandb.run is not None:
                        wandb.finish(exit_code=1)
                    raise optuna.exceptions.TrialPruned(f"Trial pruned due to CUDA OOM during forecast generator creation: {e}")
                raise
            except Exception as e:
                logging.error(f"Trial {trial.number} - Error creating forecast generator: {str(e)}", exc_info=True)
                raise

            # Train Model
            try:
                estimator.train(
                    training_data=self.data_module.train_dataset,
                    validation_data=self.data_module.val_dataset,
                    forecast_generator=forecast_generator
                )
            except optuna.exceptions.TrialPruned as e:
                logging.info(f"Trial {trial.number} pruned by Optuna callback during training: {str(e)}")
                raise
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logging.error(f"Trial {trial.number} - CUDA OOM during training: {str(e)}")
                    if wandb.run is not None:
                        wandb.finish(exit_code=1)
                    raise optuna.exceptions.TrialPruned(f"Trial pruned due to CUDA OOM during training: {e}")
                raise
            except Exception as e:
                if "MisconfigurationException" in str(type(e)):
                    logging.error(f"Trial {trial.number} - MisconfigurationException detected: {str(e)}")
                    if wandb.run is not None:
                        wandb.finish(exit_code=1)
                raise

            # Log GPU stats after training
            self.log_gpu_stats(stage=f"Trial {trial.number} After Training")
            
            # Checkpoint Loading and Processing
            checkpoint = None
            epoch_number = None
            correct_stage = None
            
            try:
                checkpoint_path = trial_checkpoint_callback.best_model_path
                logging.info(f"Trial {trial.number} - Using best checkpoint from trial-specific callback: {checkpoint_path}")
                
                # Load checkpoint
                checkpoint = load_checkpoint(checkpoint_path, trial.number)
                
                # Parse epoch number
                epoch_number = parse_epoch_from_checkpoint_path(checkpoint_path, trial.number)
                
                # Determine TACTiS stage if applicable
                if self.model == 'tactis':
                    logging.info(f"Trial {trial.number} - Model is TACTiS. Attempting to determine stage for re-instantiation.")
                    stage2_start_epoch = estimator_kwargs.get('stage2_start_epoch')
                    correct_stage = determine_tactis_stage(epoch_number, stage2_start_epoch, trial.number)
                else:
                    logging.info(f"Trial {trial.number} - Model is {self.model} (not TACTiS). Skipping stage determination.")
            
            except (FileNotFoundError, RuntimeError, ValueError, KeyError) as e:
                logging.error(f"Trial {trial.number} - Error during checkpoint loading/parsing/stage determination: {str(e)}", exc_info=True)
            except Exception as e:
                logging.error(f"Trial {trial.number} - Unexpected error before hyperparameter extraction: {e}", exc_info=True)
                raise RuntimeError(f"Trial {trial.number}: Unexpected error before hyperparameter extraction: {e}") from e
            
            # Extract hyperparameters and prepare model initialization
            try:
                hparams = extract_hyperparameters(checkpoint, checkpoint_path, trial.number)
                
                init_args = prepare_model_init_args(
                    hparams, self.lightning_module_class, self.config, self.model, trial.number
                )
                
                instantiation_stage_info = "N/A (Not TACTiS)"
                if self.model == 'tactis':
                    if correct_stage is not None:
                        instantiation_stage_info = f"Stage {correct_stage} (Determined, will be set post-init)"
                    else:
                        instantiation_stage_info = "Stage Unknown (Determination Failed)"
                
                logging.info(f"Re-instantiating {self.lightning_module_class.__name__} ({self.model}) for metric retrieval. Stage Info: {instantiation_stage_info}")
                logging.debug(f"Trial {trial.number} - Using init_args for re-instantiation: { {k: type(v).__name__ if not isinstance(v, (str, int, float, bool, list, dict, tuple)) else v for k, v in init_args.items()} }")
            
            except (KeyError, RuntimeError) as e:
                logging.error(f"Trial {trial.number} - Error preparing model initialization: {str(e)}", exc_info=True)
                raise
            
            # Instantiate Model and Load State Dict
            try:
                model = self.lightning_module_class(**init_args)
                
                if self.model == 'tactis' and correct_stage is not None:
                    set_tactis_stage(model, correct_stage, trial.number)
                
                load_model_state(model, checkpoint, trial.number)
            
            except Exception as e:
                stage_at_error = init_args.get('initial_stage', 'Unknown')
                logging.error(f"Trial {trial.number} - Unexpected error instantiating model (with initial_stage={stage_at_error}) or loading state_dict: {str(e)}", exc_info=True)
                raise RuntimeError(f"Error instantiating model (stage {stage_at_error}) or loading state_dict in trial {trial.number}: {str(e)}") from e
            
            # Evaluate model if not using val_loss
            if metric_to_return != "val_loss":
                # Create predictor
                try:
                    transformation = estimator.create_transformation(use_lazyframe=False)
                    predictor = estimator.create_predictor(transformation, model,
                                                            forecast_generator=forecast_generator)
                except Exception as e:
                    logging.error(f"Trial {trial.number} - Error creating predictor: {str(e)}", exc_info=True)
                    raise RuntimeError(f"Error creating predictor in trial {trial.number}: {str(e)}") from e
                
                # Compute evaluation metrics
                agg_metrics = compute_evaluation_metrics(
                    predictor, self.data_module.val_dataset, self.model,
                    self.evaluator, self.data_module.num_target_vars, trial.number
                )
            else:
                agg_metrics = {}
            
            # Update metrics with checkpoint score
            update_metrics_with_checkpoint_score(agg_metrics, trial_checkpoint_callback, trial.number)
            logging.info(f"Trial {trial.number} - Aggregated metrics calculated: {list(agg_metrics.keys())}")

            # Log GPU stats at the end of the trial
            self.log_gpu_stats(stage=f"Trial {trial.number} End")

            # Force garbage collection at the end of each trial
            gc.collect()
            torch.cuda.empty_cache()

        finally:
            # Always attempt to finish if a wandb run object exists for this process
            if wandb.run is not None:
                current_run_name = wandb.run.name # Get name before finishing
                logging.info(f"Rank {os.environ.get('WORKER_RANK', 'N/A')}: Finishing trial-specific W&B run '{current_run_name}' for trial {trial.number if 'trial' in locals() else 'unknown'}")
                wandb.finish()

        # Return metric to Optuna
        metric_to_return = self.config.get("trainer", {}).get("monitor_metric", "val_loss")
        
        metric_value = validate_metrics_for_return(agg_metrics, metric_to_return, trial.number)
        logging.info(f"Trial {trial.number} - This metric is from the trial-specific checkpoint: {os.path.basename(checkpoint_path)}")
        return metric_value


def get_tuned_params(storage, study_name):
    logging.info(f"Getting storage for Optuna prefix study name {study_name}.")
    
    available_studies = [study.study_name for study in storage.get_all_studies()]
    if study_name in available_studies:
        full_study_name = study_name
    else:
        full_study_name = sorted(storage.get_all_studies(), key=lambda study: int(re.search(f"(?<={study_name}_)(\\d+)", study.study_name).group()))[-1].study_name
    
    try:
        study_id = storage.get_study_id_from_name(full_study_name)
        logging.info(f"Found storage for Optuna full study name {full_study_name}.")
    except Exception:
        raise FileNotFoundError(f"Optuna study {full_study_name} not found. Please run tune_hyperparameters_multi for all outputs first. Available studies are: {available_studies}.")
    # self.model[output].set_params(**storage.get_best_trial(study_id).params)
    # storage.get_all_studies()[0]._study_id
    # estimato = self.create_model(**storage.get_best_trial(study_id).params)
    return storage.get_best_trial(study_id).params

# Update signature: Add optuna_storage_url, remove storage_dir, use_rdb, restart_study
def tune_model(model, config, study_name, optuna_storage, lightning_module_class, estimator_class,
               max_epochs, limit_train_batches,
               distr_output_class, data_module,
               metric="val_loss", direction="minimize", n_trials_per_worker=10, total_study_trials=100,
               trial_protection_callback=None, seed=42, tuning_phase=0, restart_tuning=False, optimize_callbacks=None,):

    # Log safely without credentials if they were included (they aren't for socket trust)
    if hasattr(optuna_storage, "url"):
        log_storage_url = optuna_storage.url.split('@')[0] + '@...' if '@' in optuna_storage.url else optuna_storage.url
        logging.info(f"Using Optuna storage URL: {log_storage_url}")

    # NOTE: Restarting the study is now handled in the Slurm script by deleting the PGDATA directory

    # Configure pruner based on settings
    pruner = None
    if "pruning" in config["optuna"] and config["optuna"]["pruning"].get("enabled", False):
        pruning_type = config["optuna"]["pruning"].get("type", "hyperband").lower()
        logging.info(f"Configuring pruner: type={pruning_type}")

        if pruning_type == "patient":
            patience = config["optuna"]["pruning"].get("patience", 0)
            min_delta = config["optuna"]["pruning"].get("min_delta", 0.0)

            # Configure wrapped pruner if specified
            wrapped_config = config["optuna"]["pruning"].get("wrapped_pruner")
            wrapped_pruner_instance = None

            if wrapped_config and isinstance(wrapped_config, dict):
                wrapped_type = wrapped_config.get("type", "").lower()
                logging.info(f"Configuring wrapped pruner of type: {wrapped_type}")

                if wrapped_type == "percentile":
                    percentile = wrapped_config.get("percentile", 50.0)
                    n_startup_trials = wrapped_config.get("n_startup_trials", 4)
                    n_warmup_steps = wrapped_config.get("n_warmup_steps", 12)
                    interval_steps = wrapped_config.get("interval_steps", 1)
                    n_min_trials = wrapped_config.get("n_min_trials", 1)

                    wrapped_pruner_instance = PercentilePruner(
                        percentile=percentile,
                        n_startup_trials=n_startup_trials,
                        n_warmup_steps=n_warmup_steps,
                        interval_steps=interval_steps,
                        n_min_trials=n_min_trials
                    )
                    logging.info(f"Created wrapped PercentilePruner with percentile={percentile}, n_startup_trials={n_startup_trials}, n_warmup_steps={n_warmup_steps}")
                    
                elif wrapped_type == "successivehalving":
                    min_resource = wrapped_config.get("min_resource", 2)
                    reduction_factor = wrapped_config.get("reduction_factor", 2)
                    min_early_stopping_rate = wrapped_config.get("min_early_stopping_rate", 0)
                    bootstrap_count = wrapped_config.get("bootstrap_count", 0)

                    wrapped_pruner_instance = SuccessiveHalvingPruner(
                        min_resource=min_resource,
                        reduction_factor=reduction_factor,
                        min_early_stopping_rate=min_early_stopping_rate,
                        bootstrap_count=bootstrap_count
                    )
                    logging.info(f"Created wrapped SuccessiveHalvingPruner with min_resource={min_resource}, reduction_factor={reduction_factor}, min_early_stopping_rate={min_early_stopping_rate}, bootstrap_count={bootstrap_count}, bootstrap_count={bootstrap_count}")
                
                else:
                    logging.warning(f"Unknown wrapped pruner type: {wrapped_type}. Defaulting to NopPruner.")
                    wrapped_pruner_instance = NopPruner()
            else:
                logging.warning("No wrapped pruner configuration found. Defaulting to NopPruner.")
                wrapped_pruner_instance = NopPruner()
            
            # If no valid wrapped pruner is configured, use NopPruner
            if wrapped_pruner_instance is None:
                logging.warning("No valid wrapped pruner configuration found. PatientPruner will wrap NopPruner.")
                wrapped_pruner_instance = NopPruner()

            # Create PatientPruner wrapping the configured pruner
            pruner = PatientPruner(
                wrapped_pruner=wrapped_pruner_instance,
                patience=patience,
                min_delta=min_delta
            )
            logging.info(f"Created PatientPruner with patience={patience}, min_delta={min_delta} wrapping {type(wrapped_pruner_instance).__name__}")

        elif pruning_type == "hyperband":
            min_resource = config["optuna"]["pruning"].get("min_resource", 2)
            max_resource = config["optuna"]["pruning"].get("max_resource", max_epochs)
            reduction_factor = config["optuna"]["pruning"].get("reduction_factor", 2)
            bootstrap_count = config["optuna"]["pruning"].get("bootstrap_count", 0)
            
            pruner = HyperbandPruner(
                min_resource=min_resource,
                max_resource=max_resource,
                reduction_factor=reduction_factor,
                bootstrap_count=bootstrap_count
            )
            logging.info(f"Created HyperbandPruner with min_resource={min_resource}, max_resource={max_resource}, reduction_factor={reduction_factor}, bootstrap_count={bootstrap_count}")

        elif pruning_type == "successivehalving":
            min_resource = config["optuna"]["pruning"].get("min_resource", 2)
            reduction_factor = config["optuna"]["pruning"].get("reduction_factor", 2)
            min_early_stopping_rate = config["optuna"]["pruning"].get("min_early_stopping_rate", 0)
            bootstrap_count = config["optuna"]["pruning"].get("bootstrap_count", 0)

            pruner = SuccessiveHalvingPruner(
                min_resource=min_resource,
                reduction_factor=reduction_factor,
                min_early_stopping_rate=min_early_stopping_rate,
                bootstrap_count=bootstrap_count
            )
            logging.info(f"Created SuccessiveHalvingPruner with min_resource={min_resource}, reduction_factor={reduction_factor}, min_early_stopping_rate={min_early_stopping_rate}, bootstrap_count={bootstrap_count}, bootstrap_count={bootstrap_count}")

        elif pruning_type == "percentile":
            percentile = config["optuna"]["pruning"].get("percentile", 25)
            n_startup_trials = config["optuna"]["pruning"].get("n_startup_trials", 5)
            n_warmup_steps = config["optuna"]["pruning"].get("n_warmup_steps", 2)
            interval_steps = config["optuna"]["pruning"].get("interval_steps", 1)
            n_min_trials = config["optuna"]["pruning"].get("n_min_trials", 1)

            pruner = PercentilePruner(
                percentile=percentile,
                n_startup_trials=n_startup_trials,
                n_warmup_steps=n_warmup_steps,
                interval_steps=interval_steps,
                n_min_trials=n_min_trials
            )
            logging.info(f"Created PercentilePruner with percentile={percentile}, n_startup_trials={n_startup_trials}, n_warmup_steps={n_warmup_steps}")

        else:
            logging.warning(f"Unknown pruner type: {pruning_type}, using no pruning")
            pruner = NopPruner()
    else:
        logging.info("Pruning is disabled, using NopPruner")
        pruner = NopPruner()

    # Get worker ID for study creation/loading logic
    # Use WORKER_RANK consistent with run_model.py. Default to '0' if not set.
    worker_id = os.environ.get('WORKER_RANK', '0')

    # Generate unique study name based on restart_tuning flag

    base_study_prefix = study_name
    if restart_tuning:
        job_id = os.environ.get('SLURM_JOB_ID')
        if job_id:
            # If running in SLURM, use the job ID
            final_study_name = f"{base_study_prefix}_{job_id}"
        else:
            # Otherwise use a timestamp
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            final_study_name = f"{base_study_prefix}_{timestamp}"
        logging.info(f"Creating a new study with unique name: {final_study_name}")
    else:
        # If not restarting, use the base name to resume existing study
        final_study_name = base_study_prefix
        logging.info(f"Using existing study name to resume: {final_study_name}")

    # Define pickle directory for sampler/pruner persistence
    pickle_dir = os.path.join(config.get("logging", {}).get("optuna_dir", "logging/optuna"), "pickles")
    
    # Instantiate the persistence utility
    sampler_pruner_persistence = OptunaSamplerPrunerPersistence(config, seed)

    # Get sampler and pruner objects using pickling logic
    try:
        sampler, pruner_for_study = sampler_pruner_persistence.get_sampler_pruner_objects(
            worker_id, pruner, restart_tuning, final_study_name, optuna_storage, pickle_dir
        )
    except Exception as e:
        logging.error(f"Worker {worker_id}: Error getting sampler/pruner objects: {str(e)}", exc_info=True)
        raise

    # Create study on rank 0, load on other ranks
    study = None # Initialize study variable
    try:
        if worker_id == '0':
            logging.info(f"Rank 0: Creating/loading Optuna study '{final_study_name}' with pruner: {type(pruner_for_study).__name__}")
            study = create_study(
                study_name=final_study_name,
                storage=optuna_storage,
                direction=direction,
                load_if_exists=not restart_tuning, # Only load if not restarting
                sampler=sampler,
                pruner=pruner_for_study
            )
            logging.info(f"Rank 0: Study '{final_study_name}' created or loaded successfully.")
        else:
            # Non-rank-0 workers MUST load the study created by rank 0
            logging.info(f"Rank {worker_id}: Attempting to load existing Optuna study '{final_study_name}'")
            # Add a small delay and retry mechanism for loading, in case rank 0 is slightly delayed
            max_retries = 6 # Increased retries slightly
            retry_delay = 10 # Increased delay slightly
            for attempt in range(max_retries):
                try:
                    study = load_study(
                        study_name=final_study_name,
                        storage=optuna_storage,
                        sampler=sampler,
                        pruner=pruner_for_study
                    )
                    logging.info(f"Rank {worker_id}: Study '{final_study_name}' loaded successfully on attempt {attempt+1}.")
                    break # Exit loop on success
                except KeyError as e: # Optuna <3.0 raises KeyError if study doesn't exist yet
                     if attempt < max_retries - 1:
                          logging.warning(f"Rank {worker_id}: Study '{final_study_name}' not found yet (attempt {attempt+1}/{max_retries}). Retrying in {retry_delay}s... Error: {e}")
                          time.sleep(retry_delay)
                     else:
                          logging.error(f"Rank {worker_id}: Failed to load study '{final_study_name}' after {max_retries} attempts (KeyError). Aborting.")
                          raise
                except Exception as e: # Catch other potential loading errors (e.g., DB connection issues)
                     logging.error(f"Rank {worker_id}: An unexpected error occurred while loading study '{final_study_name}' on attempt {attempt+1}: {e}", exc_info=True)
                     # Decide whether to retry on other errors or raise immediately
                     if attempt < max_retries - 1:
                          logging.warning(f"Retrying in {retry_delay}s...")
                          time.sleep(retry_delay)
                     else:
                          logging.error(f"Rank {worker_id}: Failed to load study '{final_study_name}' after {max_retries} attempts due to persistent errors. Aborting.")
                          raise # Re-raise other errors after retries

            # Check if study was successfully loaded after the loop
            if study is None:
                 # This condition should ideally be caught by the error handling within the loop, but added for safety.
                 raise RuntimeError(f"Rank {worker_id}: Could not load study '{final_study_name}' after multiple retries.")

    except Exception as e:
        # Log error with rank information
        logging.error(f"Rank {worker_id}: Error creating/loading study '{final_study_name}': {str(e)}", exc_info=True)
        # Log storage URL safely
        if hasattr(optuna_storage, "url"):
            log_storage_url_safe = str(optuna_storage.url).split('@')[0] + '@...' if '@' in str(optuna_storage.url) else str(optuna_storage.url)
            logging.error(f"Error details - Type: {type(e).__name__}, Storage: {log_storage_url_safe}")
        else:
            logging.error(f"Error details - Type: {type(e).__name__}, Storage: Journal")
        raise

    # Define study_config_params for all workers
    study_config_params = {
        "dataset_per_turbine_target": config["dataset"].get("per_turbine_target"),
        "optuna_sampler": config["optuna"].get("sampler"),
        "optuna_pruner_type": config["optuna"]["pruning"].get("type") if "pruning" in config["optuna"] else None,
        "optuna_max_epochs": config["optuna"].get("max_epochs"),
        "optuna_base_limit_train_batches": config["optuna"].get("base_limit_train_batches"),
        "optuna_limit_train_batches": config["optuna"].get("limit_train_batches"),  # Legacy fallback
        "dataset_base_batch_size": config["dataset"].get("base_batch_size"),
        "optuna_metric": config["optuna"].get("metric"),
        "dataset_data_path": config["dataset"].get("data_path"),
        "dataset_resample_freq": config["dataset"].get("resample_freq"),
        "dataset_test_split": config["dataset"].get("test_split"),
        "dataset_val_split": config["dataset"].get("val_split")
    }

    # Add TACTiS-specific parameters if model is TACTiS
    if model == 'tactis':
        # Adjust TACTiS scheduler parameters based on dataset and training settings
        per_turbine_target = config["dataset"].get("per_turbine_target", False)
        # Use base_limit_train_batches if available, otherwise fallback to limit_train_batches
        limit_train_batches = config["optuna"].get("base_limit_train_batches") or config["optuna"].get("limit_train_batches")
        num_turbines = 1 # Default to 1 in case target_suffixes is not available or per_turbine_target is True
        if not data_module.per_turbine_target and data_module.target_suffixes is not None:
            try:
                num_turbines = len(data_module.target_suffixes)
                if num_turbines == 0:
                     logging.warning("data_module.target_suffixes is empty, defaulting num_turbines to 1 for adjustment.")
                     num_turbines = 1
            except TypeError:
                 logging.warning("data_module.target_suffixes is not a sequence, defaulting num_turbines to 1 for adjustment.")
                 num_turbines = 1
        elif data_module.per_turbine_target:
             num_turbines = 1
        else:
             logging.warning("data_module.target_suffixes is None, defaulting num_turbines to 1 for adjustment.")
             num_turbines = 1

        if num_turbines <= 0:
             logging.error(f"Calculated num_turbines is invalid ({num_turbines}). Defaulting to 1 for adjustment.")
             num_turbines = 1

        base_warmup_s1 = config["model"]["tactis"].get("warmup_steps_s1")
        base_decay_s1 = config["model"]["tactis"].get("steps_to_decay_s1")
        base_warmup_s2 = config["model"]["tactis"].get("warmup_steps_s2")
        base_decay_s2 = config["model"]["tactis"].get("steps_to_decay_s2")

        adj_warmup_s1 = base_warmup_s1
        adj_decay_s1 = base_decay_s1
        adj_warmup_s2 = base_warmup_s2
        adj_decay_s2 = base_decay_s2

        if not per_turbine_target and limit_train_batches is None:
            logging.info(f"Adjusting TACTiS scheduler params: per_turbine_target={per_turbine_target}, limit_train_batches={limit_train_batches}. Dividing by num_turbines={num_turbines}.")
            if base_warmup_s1 is not None:
                adj_warmup_s1 = round(base_warmup_s1 / num_turbines)
            if base_decay_s1 is not None:
                adj_decay_s1 = round(base_decay_s1 / num_turbines)
            if base_warmup_s2 is not None:
                adj_warmup_s2 = round(base_warmup_s2 / num_turbines)
            if base_decay_s2 is not None:
                adj_decay_s2 = round(base_decay_s2 / num_turbines)
            
            # Update the config itself so MLTuningObjective uses these adjusted values
            config["model"]["tactis"]["warmup_steps_s1"] = adj_warmup_s1
            config["model"]["tactis"]["steps_to_decay_s1"] = adj_decay_s1
            config["model"]["tactis"]["warmup_steps_s2"] = adj_warmup_s2
            config["model"]["tactis"]["steps_to_decay_s2"] = adj_decay_s2
            logging.info(f"Adjusted TACTiS scheduler params: warmup_s1={adj_warmup_s1}, decay_s1={adj_decay_s1}, warmup_s2={adj_warmup_s2}, decay_s2={adj_decay_s2}")
        else:
            logging.info(f"Using base TACTiS scheduler params: per_turbine_target={per_turbine_target}, limit_train_batches={limit_train_batches}.")

        # These will now pick up the potentially adjusted values from the config
        tactis_params_for_logging = {
            "model_tactis_stage2_start_epoch": config["model"][model].get("stage2_start_epoch"), # Unchanged by this logic
            "model_tactis_warmup_steps_s1": config["model"][model].get("warmup_steps_s1"),
            "model_tactis_warmup_steps_s2": config["model"][model].get("warmup_steps_s2"),
            "model_tactis_steps_to_decay_s1": config["model"][model].get("steps_to_decay_s1"),
            "model_tactis_steps_to_decay_s2": config["model"][model].get("steps_to_decay_s2"),
            "model_tactis_eta_min_fraction_s1": config["model"][model].get("eta_min_fraction_s1"),
            "model_tactis_eta_min_fraction_s2": config["model"][model].get("eta_min_fraction_s2")
        }
        study_config_params.update(tactis_params_for_logging)

    # Set study user attributes from config (only on rank 0)
    if worker_id == '0':
        for key, value in study_config_params.items():
            if value is not None:
                study.set_user_attr(key, value)

        logging.info(f"Set study user attributes: {list(study_config_params.keys())}")

        # --- Launch Dashboard (Rank 0 only) ---
        # if hasattr(optuna_storage, "url"):
        #     launch_optuna_dashboard(config, optuna_storage.url) # Call imported function
        # --------------------------------------

    # Worker ID already fetched above for study creation/loading
    dynamic_params = None

    if tuning_phase == 0 and worker_id == "0":
        resample_freq_choices = config["optuna"].get("resample_freq_choices", [int(data_module.freq[:-1])])
        fixed_per_turbine = config.get("dataset", {}).get("per_turbine_target", False)
        logging.info(f"Rank 0: DataModule 'per_turbine_target' fixed to: {fixed_per_turbine} for pre-computation.")

        original_dm_freq = data_module.freq
        original_dm_per_turbine = data_module.per_turbine_target
        original_dm_pred_len = data_module.prediction_length
        original_dm_ctx_len = data_module.context_length

        logging.info("Rank 0: Starting pre-computation of base resampled Parquet files.")
        for resample_freq_seconds in resample_freq_choices:
            current_freq_str = f"{resample_freq_seconds}s"
            logging.info(f"Rank 0: Checking/generating base resampled Parquet for freq={current_freq_str}, per_turbine={fixed_per_turbine}.")
            
            data_module.freq = current_freq_str
            data_module.per_turbine_target = fixed_per_turbine
            
            original_prediction_len_seconds_config = config["dataset"]["prediction_length"]
            data_module.prediction_length = int(pd.Timedelta(original_prediction_len_seconds_config, unit="s") / pd.Timedelta(data_module.freq))
            
            data_module.set_train_ready_path()

            if not os.path.exists(data_module.train_ready_data_path):
                logging.info(f"Rank 0: Base resampled Parquet {data_module.train_ready_data_path} not found. Calling DataModule.generate_datasets().")
                data_module.generate_datasets()
            else:
                logging.info(f"Rank 0: Base resampled Parquet {data_module.train_ready_data_path} already exists.")

        data_module.freq = original_dm_freq
        data_module.per_turbine_target = original_dm_per_turbine
        data_module.prediction_length = original_dm_pred_len
        data_module.context_length = original_dm_ctx_len
        data_module.set_train_ready_path()

        logging.info("Rank 0: Finished pre-computation of base resampled Parquet files.")
        dynamic_params = {"resample_freq": resample_freq_choices}

    logging.info(f"Worker {worker_id}: Participating in Optuna study {final_study_name}")

    # get from config
    resample_freq_choices = config.get("optuna", {}).get("resample_freq_choices", None)
    if resample_freq_choices is None:
        logging.warning("'optuna.resample_freq_choices' not found in config. Default to 60s.")
        resample_freq_choices = [60]

    tuning_objective = MLTuningObjective(model=model, config=config,
                                        lightning_module_class=lightning_module_class,
                                        estimator_class=estimator_class,
                                        distr_output_class=distr_output_class,
                                        max_epochs=max_epochs,
                                        limit_train_batches=limit_train_batches,
                                        data_module=data_module,
                                        metric=metric,
                                        seed=seed,
                                        tuning_phase=tuning_phase,
                                        dynamic_params=dynamic_params,
                                        study_config_params=study_config_params)

    # Use the trial protection callback if provided
    objective_fn = (lambda trial: trial_protection_callback(tuning_objective, trial)) if trial_protection_callback else tuning_objective

    # WandB integration deprecated
    if optimize_callbacks is None:
        optimize_callbacks = []
    elif not isinstance(optimize_callbacks, list):
        optimize_callbacks = [optimize_callbacks]

    # Add sampler/pruner state checkpointing callback for crash recovery
    # Save after every trial completion to preserve TPESampler algorithmic state
    sampler_checkpoint_callback = sampler_pruner_persistence.create_trial_completion_callback(
        worker_id=worker_id,
        save_frequency=1  # Save after every trial - overhead is minimal compared to trial duration
    )
    optimize_callbacks.append(sampler_checkpoint_callback)
    logging.info(f"Worker {worker_id}: Added sampler state checkpointing after every trial completion")

    try:
        n_trials_per_worker = config["optuna"].get("n_trials_per_worker", 10)
        total_study_trials_config = config["optuna"].get("total_study_trials")
        
        n_trials_setting_for_optimize = None
        
        # Determine number of trials to run
        if isinstance(total_study_trials_config, int) and total_study_trials_config > 0:
            total_study_trials = total_study_trials_config
            study.set_user_attr("total_study_trials", total_study_trials)
            logging.info(f"Set global trial limit to {total_study_trials} trials.")
            n_trials_setting_for_optimize = None
            
            max_trials_cb = MaxTrialsCallback(
                n_trials=total_study_trials,
                states=(TrialState.COMPLETE, TrialState.PRUNED) # INFO: Do not count failed trials
            )
            optimize_callbacks.append(max_trials_cb)
            logging.info(f"MaxTrialsCallback added for {total_study_trials} trials.")
        else:
            # Fall back to per-worker limit if no global limit is set
            n_trials_setting_for_optimize = n_trials_per_worker
            logging.info(f"No valid global trial limit found (value: {total_study_trials_config}). Using per-worker limit of {n_trials_per_worker}.")
            n_trials_setting_for_optimize = n_trials_per_worker
        
        # Let Optuna handle trial distribution - each worker will ask the storage for a trial
        # Show progress bar only on rank 0 to avoid cluttered logs
        study.optimize(
            objective_fn,
            n_trials=n_trials_setting_for_optimize,
            callbacks=optimize_callbacks,
            show_progress_bar=(worker_id=='0')
        )
    except KeyError as e:
        logging.error(f"Configuration key missing: {e}")
    except Exception as e:
        logging.error(f"Worker {worker_id}: Failed during study optimization: {str(e)}", exc_info=True)
        raise

    if worker_id == '0' and study:
        logging.info("Rank 0: Starting W&B summary run creation.")

        # Wait for all expected trials to complete
        num_workers = int(os.environ.get('WORLD_SIZE', 1))
        
        if total_study_trials:
            expected_total_trials = total_study_trials
            logging.info(f"Rank 0: Expecting a maximum of {expected_total_trials} trials (global limit).")
        else:
            expected_total_trials = num_workers * n_trials_per_worker
            logging.info(f"Rank 0: Expecting a total of {expected_total_trials} trials ({num_workers} workers * {n_trials_per_worker} trials/worker).")

        logging.info("Rank 0: Waiting for all expected Optuna trials to reach a terminal state...")
        wait_interval_seconds = 30
        while True:
            # Refresh trials from storage
            all_trials_current = study.get_trials(deepcopy=False)
            finished_trials = [t for t in all_trials_current if t.state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)]
            num_finished = len(finished_trials)
            num_total_in_db = len(all_trials_current) # Current count in DB

            logging.info(f"Rank 0: Trial status check: {num_finished} finished / {num_total_in_db} in DB (expected total: {expected_total_trials}).")

            if num_finished >= expected_total_trials:
                logging.info(f"Rank 0: All {expected_total_trials} expected trials have reached a terminal state.")
                break
            elif num_total_in_db > expected_total_trials and num_finished >= expected_total_trials:
                 logging.warning(f"Rank 0: Found {num_total_in_db} trials in DB (expected {expected_total_trials}), but {num_finished} finished trials meet the expectation.")
                 break

            logging.info(f"Rank 0: Still waiting for trials to finish ({num_finished}/{expected_total_trials}). Sleeping for {wait_interval_seconds} seconds...")
            time.sleep(wait_interval_seconds)

        try:
            # Fetch best trial *before* initializing summary run
            best_trial = None
            try:
                best_trial = study.best_trial
                logging.info(f"Rank 0: Fetched best trial: Number={best_trial.number}, Value={best_trial.value}")
            except ValueError:
                logging.warning("Rank 0: Could not retrieve best trial (likely no trials completed successfully).")
            except Exception as e_best_trial:
                logging.error(f"Rank 0: Error fetching best trial: {e_best_trial}", exc_info=True)

            # Fetch Git info directly using subprocess
            remote_url = None
            commit_hash = None
            try:
                # Get remote URL
                remote_url_bytes = subprocess.check_output(['git', 'config', '--get', 'remote.origin.url'], stderr=subprocess.STDOUT).strip()
                remote_url = remote_url_bytes.decode('utf-8')
                # Convert SSH URL to HTTPS URL if necessary
                if remote_url.startswith("git@"):
                    remote_url = remote_url.replace(":", "/").replace("git@", "https://")
                # Remove .git suffix AFTER potential conversion
                if remote_url.endswith(".git"):
                    remote_url = remote_url[:-4]

                # Get commit hash
                commit_hash_bytes = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT).strip()
                commit_hash = commit_hash_bytes.decode('utf-8')
                logging.info(f"Rank 0: Fetched Git Info - URL: {remote_url}, Commit: {commit_hash}")
            except subprocess.CalledProcessError as e:
                logging.warning(f"Rank 0: Could not get Git info: {e.output.decode('utf-8').strip()}")
            except FileNotFoundError:
                logging.warning("Rank 0: 'git' command not found. Cannot log Git info.")
            except Exception as e_git:
                 logging.error(f"Rank 0: An unexpected error occurred while fetching Git info: {e_git}", exc_info=True)

            git_info_config = {}
            if remote_url and commit_hash:
                 git_info_config = {"git_info": {"url": remote_url, "commit": commit_hash}}
            else:
                 logging.warning("Rank 0: Git info could not be fully determined. Logging summary run without it.")


            # Determine summary run name
            base_run_name = config['experiment']['run_name']
            if best_trial:
                run_name = f"RESULTS_{base_run_name}_tuning_phase{tuning_phase}_best_trial_{best_trial.number}_{final_study_name}"
            else:
                run_name = f"RESULTS_{base_run_name}_tuning_phase{tuning_phase}_optuna_summary_{final_study_name}"

            project_name = f"{config['experiment'].get('project_name', 'wind_forecasting')}_{model}_tuning_phase{tuning_phase}"
            group_name = config['experiment']['run_name']
            wandb_dir = config['logging'].get('wandb_dir', './logging/wandb')
            tags = [model, "optuna_summary"] + config['experiment'].get('extra_tags', [])

            # Ensure wandb is not already initialized in a weird state (shouldn't be, but safety check)
            if wandb.run is not None:
                logging.warning(f"Rank 0: Found an existing W&B run ({wandb.run.id}) before starting summary run. Finishing it.")
                wandb.finish()
            # INFO: This wandb.init is to initialize W&B summary run at the end

            wandb.init(
                name=run_name,
                project=project_name,
                group=group_name,
                job_type="optuna_summary",
                dir=wandb_dir,
                tags=tags,
                config=git_info_config,
                reinit="create_new" # Let explicit wandb.finish() calls handle run separation
            )
            logging.info(f"Rank 0: Initialized W&B summary run: {wandb.run.name} (ID: {wandb.run.id}) with Git info: {git_info_config}")

            try:
                # Log Optuna visualizations using the helper function
                logging.info("Rank 0: Logging Optuna visualizations to W&B summary run...")
                log_optuna_visualizations_to_wandb(study, wandb.run)
                logging.info("Rank 0: Finished logging Optuna visualizations to W&B.")

                # Log Detailed Trials Table using the helper function
                log_detailed_trials_table_to_wandb(study, wandb.run)

            except Exception as e_log:
                 logging.error(f"Rank 0: Error during logging visualizations or trial table to W&B summary run: {e_log}", exc_info=True)
            finally:
                # Ensure W&B run is finished even if logging fails
                if wandb.run is not None:
                    logging.info(f"Rank 0: Finishing W&B summary run: {wandb.run.name}")
                    wandb.finish()
                else:
                    logging.warning("Rank 0: No active W&B run found to finish in the finally block.")

        except Exception as e_init:
            logging.error(f"Rank 0: Failed to initialize W&B summary run: {e_init}", exc_info=True)
            # Ensure wandb is cleaned up if initialization failed partially
            if wandb.run is not None:
                wandb.finish()

    # All workers log their contribution
    logging.info(f"Worker {worker_id} completed optimization")

    # Generate visualizations if enabled (only rank 0 should do this)
    if worker_id == '0' and config.get("optuna", {}).get("visualization", {}).get("enabled", False):
        if study:
            try:
                from wind_forecasting.utils.optuna_visualization import generate_visualizations
                # Import the path resolution helper from db_utils or optuna_db_utils
                from wind_forecasting.utils.db_utils import _resolve_path

                vis_config = config["optuna"]["visualization"]

                # Resolve the output directory using the helper function and full config
                default_vis_path = os.path.join(config.get("logging", {}).get("optuna_dir", "logging/optuna"), "visualizations")
                # Pass vis_config as the dict containing 'output_dir', key 'output_dir', and the full 'config'
                visualization_dir = _resolve_path(vis_config, "output_dir", full_config=config, default=default_vis_path)

                if not visualization_dir:
                     logging.error("Rank 0: Could not determine visualization output directory. Skipping visualization.")
                else:
                    logging.info(f"Rank 0: Resolved visualization output directory: {visualization_dir}")
                    os.makedirs(visualization_dir, exist_ok=True) # Ensure directory exists

                    # Generate plots
                    logging.info(f"Rank 0: Generating Optuna visualizations in {visualization_dir}")
                    summary_path = generate_visualizations(study, visualization_dir, vis_config) # Pass vis_config

                    if summary_path:
                        logging.info(f"Rank 0: Generated Optuna visualizations - summary available at: {summary_path}")
                    else:
                        logging.warning("Rank 0: No visualizations were generated - study may not have enough completed trials or an error occurred.")

            except ImportError:
                 logging.warning("Rank 0: Could not import visualization modules. Skipping visualization generation.")
            except Exception as e:
                logging.error(f"Rank 0: Failed to generate Optuna visualizations: {e}", exc_info=True)
        else:
             logging.warning("Rank 0: Study object not available, cannot generate visualizations.")

    # Log best trial details (only rank 0)
    if worker_id == '0' and study: # Check if study object exists
        if len(study.trials) > 0:
            logging.info("Number of finished trials: {}".format(len(study.trials)))
            logging.info("Best trial:")
            trial = study.best_trial
            logging.info("  Value: {}".format(trial.value))
            logging.info("  Params: ")
            for key, value in trial.params.items():
                logging.info("    {}: {}".format(key, value))
        else:
            logging.warning("No trials were completed")

        # Generate and print Optuna Dashboard command
        db_setup_params = generate_db_setup_params(model, config)
        dashboard_command_output = generate_optuna_dashboard_command(db_setup_params, final_study_name)
        logging.info(dashboard_command_output)

    return study.best_params