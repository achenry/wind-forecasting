"""
MLTuningObjective class for Optuna hyperparameter optimization.

This module contains the main objective function used by Optuna for tuning
wind forecasting models. It handles trial execution, model training, 
evaluation, and logging.
"""

import os
import logging
import torch
import gc
import inspect
import numpy as np
import pandas as pd
import wandb
from lightning.pytorch.loggers import WandbLogger
import optuna

from gluonts.evaluation import MultivariateEvaluator
from gluonts.model.forecast_generator import DistributionForecastGenerator, SampleForecastGenerator
from gluonts.time_feature._base import second_of_minute, minute_of_hour, hour_of_day, day_of_year
from gluonts.transform import ExpectedNumInstanceSampler, SequentialSampler, ValidationSplitSampler

from wind_forecasting.tuning.utils.helpers import (
    set_trial_seeds, update_data_module_params, regenerate_data_splits,
    prepare_feedforward_params, calculate_dynamic_limit_train_batches,
    create_trial_checkpoint_callback, setup_trial_callbacks
)
from wind_forecasting.tuning.utils.checkpoint_utils import (
    load_checkpoint, parse_epoch_from_checkpoint_path, determine_tactis_stage,
    extract_hyperparameters, prepare_model_init_args, load_model_state, set_tactis_stage
)
from wind_forecasting.tuning.utils.metrics_utils import (
    extract_metric_value, compute_evaluation_metrics,
    update_metrics_with_checkpoint_score, validate_metrics_for_return
)


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

        # TODO this has been updated to not change splits for different context_length_factors, for consistency, may cause issues for longer context_length_factors
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
        
        # Get current batch size for calculations
        current_batch_size = params.get('batch_size', self.config["dataset"].get("batch_size", 128))
        
        if dynamic_limit_train_batches is not None:
            # Update trainer config with dynamic value
            self.config["trainer"]["limit_train_batches"] = dynamic_limit_train_batches
            
            # Scale val_check_interval proportionally with limit_train_batches to maintain same validation frequency
            base_val_check_interval = self.config["trainer"].get("val_check_interval", 5000)
            scaling_factor = self.base_batch_size / current_batch_size
            # BUGFIX Preserve float type for epoch validation
            if isinstance(base_val_check_interval, float) and base_val_check_interval <= 1.0:
                dynamic_val_check_interval = base_val_check_interval
            else:
                dynamic_val_check_interval = max(1.0, round(base_val_check_interval * scaling_factor))
            
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

            project_name = f"tune_{self.config['experiment'].get('project_name', 'wind_forecasting')}_{self.model}"
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
        
        # Extract estimator parameters from the class signature
        estimator_sig = inspect.signature(self.estimator_class.__init__)
        estimator_params = [param.name for param in estimator_sig.parameters.values()]
        valid_estimator_params = set(estimator_params)
        
        if any("gradient_clip_val" in k for k in trial_trainer_kwargs) and any("gradient_clip_val" in k for k in valid_estimator_params):
            gc_keys = [k for k in trial_trainer_kwargs if "gradient_clip_val" in k]
            for k in gc_keys:
                del trial_trainer_kwargs[k]
        
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
        
        # Calculate actual number of training samples TODO this will not work with expectednumsampler...
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
        
        estimator_kwargs.update({k: v for k, v in self.config["model"][self.model].items() if k in valid_estimator_params})
        estimator_kwargs["num_batches_per_epoch"] = dynamic_num_batches  # Restore dynamic value
        
        if "num_batches_per_epoch" not in self.config["model"][self.model]:
            self.config["model"][self.model]["num_batches_per_epoch"] = dynamic_num_batches
            logging.info(f"Trial {trial.number}: Added num_batches_per_epoch={dynamic_num_batches} to self.config['model'][self.model] for checkpointing stability.")
        else:
            logging.warning(f"Trial {trial.number}: Overriding config num_batches_per_epoch={self.config['model'][self.model].get('num_batches_per_epoch')} with dynamic value={dynamic_num_batches}")

        # Add model-specific tunable hyperparameters suggested by Optuna trial

        filtered_params = {
            k: v for k, v in params.items()
            if k in valid_estimator_params
        }
        logging.info(f"Trial {trial.number}: Updating estimator_kwargs with filtered params: {list(filtered_params.keys())}")
        estimator_kwargs.update(filtered_params)

        # TACTiS manages its own distribution output internally, remove if present
        if self.model == 'tactis' and 'distr_output' in estimator_kwargs:
            estimator_kwargs.pop('distr_output')
        
        # Add use_pytorch_dataloader flag if specified in dataset config
        if "use_pytorch_dataloader" in self.config["dataset"]:
            estimator_kwargs["use_pytorch_dataloader"] = self.config["dataset"]["use_pytorch_dataloader"]
            logging.info(f"Trial {trial.number}: Setting use_pytorch_dataloader={self.config['dataset']['use_pytorch_dataloader']} from config")

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
                # Check if we should use PyTorch dataloaders
                use_pytorch_dataloader = self.config["dataset"].get("use_pytorch_dataloader", False)
                
                if use_pytorch_dataloader:
                    # For PyTorch dataloaders, pass file paths instead of datasets
                    train_data_path = self.data_module.get_split_file_path("train")
                    val_data_path = self.data_module.get_split_file_path("val")
                    
                    logging.info(f"Trial {trial.number}: Using PyTorch DataLoader with file paths:")
                    logging.info(f"  Training data: {train_data_path}")
                    logging.info(f"  Validation data: {val_data_path}")
                    
                    estimator.train(
                        training_data=train_data_path,
                        validation_data=val_data_path,
                        forecast_generator=forecast_generator,
                        # Pass additional kwargs that might be needed for PyTorch dataloaders
                        num_workers=4,
                        pin_memory=True,
                        persistent_workers=True,
                        skip_indices=self.data_module.prediction_length # TODO this should be configurable how many indices in sequential sampler to skip for validation
                    )
                else:
                    # Original GluonTS data loading
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