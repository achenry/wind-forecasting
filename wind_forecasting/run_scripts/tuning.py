import os
from lightning.pytorch.utilities.model_summary import summarize
from gluonts.evaluation import MultivariateEvaluator, make_evaluation_predictions
from gluonts.model.forecast_generator import DistributionForecastGenerator
from gluonts.time_feature._base import second_of_minute, minute_of_hour, hour_of_day, day_of_year
from gluonts.transform import ExpectedNumInstanceSampler, ValidationSplitSampler
import logging
import torch
import gc
import time # Added for load_study retry delay
# Imports for Optuna
import optuna # Import the base optuna module for type hints
from optuna import create_study, load_study
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner, MedianPruner, PercentilePruner, NopPruner
from optuna.integration import PyTorchLightningPruningCallback
import lightning.pytorch as pl # Import pl alias

from wind_forecasting.utils.optuna_visualization import launch_optuna_dashboard
from wind_forecasting.utils.trial_utils import handle_trial_with_oom_protection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Wrapper class to safely pass the Optuna pruning callback to PyTorch Lightning
class SafePruningCallback(pl.Callback):
    def __init__(self, trial: optuna.trial.Trial, monitor: str):
        super().__init__()
        # Instantiate the actual Optuna callback internally
        self.optuna_pruning_callback = PyTorchLightningPruningCallback(trial, monitor)

    # Delegate the relevant callback method(s)
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # Call the corresponding method on the wrapped Optuna callback
        self.optuna_pruning_callback.on_validation_end(trainer, pl_module)

    # Delegate check_pruned if needed
    def check_pruned(self) -> None:
        self.optuna_pruning_callback.check_pruned()

class MLTuningObjective:
    def __init__(self, *, model, config, lightning_module_class, estimator_class, distr_output_class, max_epochs, limit_train_batches, data_module, metric, context_length_choices, seed=42):
        self.model = model
        self.config = config
        self.lightning_module_class = lightning_module_class
        self.estimator_class = estimator_class
        self.distr_output_class = distr_output_class
        self.data_module = data_module
        self.metric = metric
        self.evaluator = MultivariateEvaluator(num_workers=None, custom_eval_fn=None)
        self.context_length_choices = context_length_choices
        self.metrics = []
        self.seed = seed

        self.config["trainer"]["max_epochs"] = max_epochs
        self.config["trainer"]["limit_train_batches"] = limit_train_batches

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
        # Set random seeds for reproducibility within each trial
        # Use different but deterministic seeds for each trial by combining base seed with trial number
        trial_seed = self.seed + trial.number
        torch.manual_seed(trial_seed)
        torch.cuda.manual_seed_all(trial_seed)
        import random
        import numpy as np
        random.seed(trial_seed)
        np.random.seed(trial_seed)
        logging.info(f"Set random seed for trial {trial.number} to {trial_seed}")

        # Log GPU stats at the beginning of the trial
        self.log_gpu_stats(stage=f"Trial {trial.number} Start")

        # params = self.get_params(trial)
        params = self.estimator_class.get_params(trial, self.context_length_choices)
        self.config["dataset"].update({k: v for k, v in params.items() if k in self.config["dataset"]})
        self.config["model"][self.model].update({k: v for k, v in params.items() if k in self.config["model"][self.model]})
        self.config["trainer"].update({k: v for k, v in params.items() if k in self.config["trainer"]})

        # Configure trainer_kwargs to include pruning callback if enabled
        # Make a copy of callbacks to avoid modifying the original list across trials
        current_callbacks = list(self.config["trainer"].get("callbacks", []))
        if self.pruning_enabled:
            # Create the SAFE wrapper for the PyTorch Lightning pruning callback
            pruning_monitor_metric = "val_loss"
            pruning_callback = SafePruningCallback(
                trial,
                monitor=pruning_monitor_metric
            )
            current_callbacks.append(pruning_callback)
            logging.info(f"Added pruning callback for trial {trial.number}, monitoring '{pruning_monitor_metric}' (Objective metric: '{self.metric}')")

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

        # Create a copy of trainer_kwargs for this trial to avoid side effects
        trial_trainer_kwargs = self.config["trainer"].copy()
        trial_trainer_kwargs["callbacks"] = current_callbacks # Use the potentially modified list

        estimator = self.estimator_class(
            freq=self.data_module.freq,
            prediction_length=self.data_module.prediction_length,
            context_length=self.config["dataset"]["context_length"],
            num_feat_dynamic_real=self.data_module.num_feat_dynamic_real,
            num_feat_static_cat=self.data_module.num_feat_static_cat,
            cardinality=self.data_module.cardinality,
            num_feat_static_real=self.data_module.num_feat_static_real,
            input_size=self.data_module.num_target_vars,
            scaling=False,
            lags_seq=[0], 
            batch_size=self.config["dataset"].setdefault("batch_size", 128),
            num_batches_per_epoch=trial_trainer_kwargs["limit_train_batches"], # Use value from trial_trainer_kwargs
            train_sampler=ExpectedNumInstanceSampler(num_instances=1.0, min_past=self.config["dataset"]["context_length"], min_future=self.data_module.prediction_length),
            validation_sampler=ValidationSplitSampler(min_past=self.config["dataset"]["context_length"], min_future=self.data_module.prediction_length),
            time_features=[second_of_minute, minute_of_hour, hour_of_day, day_of_year],
            distr_output=self.distr_output_class(dim=self.data_module.num_target_vars, **self.config["model"]["distr_output"]["kwargs"]),
            trainer_kwargs=trial_trainer_kwargs, # Pass the trial-specific kwargs
            **self.config["model"][self.model]
        )

        # Log GPU stats before training
        self.log_gpu_stats(stage=f"Trial {trial.number} Before Training")

        train_output = estimator.train(
            training_data=self.data_module.train_dataset,
            validation_data=self.data_module.val_dataset,
            forecast_generator=DistributionForecastGenerator(estimator.distr_output)
            # Note: The trainer_kwargs including callbacks are passed internally by the estimator
        )

        # Log GPU stats after training
        self.log_gpu_stats(stage=f"Trial {trial.number} After Training")

        model = self.lightning_module_class.load_from_checkpoint(train_output.trainer.checkpoint_callback.best_model_path)
        transformation = estimator.create_transformation(use_lazyframe=False)
        predictor = estimator.create_predictor(transformation, model,
                                                forecast_generator=DistributionForecastGenerator(estimator.distr_output))

        forecast_it, ts_it = make_evaluation_predictions(
            dataset=self.data_module.val_dataset,  # FIXED: Using validation data instead of test data
            predictor=predictor,
            output_distr_params=True
        )

        forecasts = list(forecast_it)
        tss = list(ts_it)

        # --- BUG Debugging: Inspect forecasts and distribution parameters before evaluation ---
        logging.info(f"Trial {trial.number}: Inspecting {len(forecasts)} forecasts before evaluation...")
        problematic_indices = []
        for i, (ts, forecast) in enumerate(zip(tss, forecasts)):
            try:
                # Attempt the operation that might fail (related to distribution/copy_dim)
                # Accessing distribution parameters might be enough to trigger calculation
                _ = forecast.distribution.mean # Access a property to potentially trigger internal calculations
                _ = forecast.distribution.variance # Access variance
                # Explicitly try the copy_dim operation if needed, though accessing params might suffice
                # _ = forecast.copy_dim(0)
            except torch._C._LinAlgError as e:
                logging.error(f"Trial {trial.number}: LinAlgError encountered inspecting forecast index {i} (start_date: {forecast.start_date})")
                logging.error(f"Error: {e}")
                problematic_indices.append(i)
                # Log distribution parameters if possible
                try:
                    dist = forecast.distribution
                    if hasattr(dist, 'cov_factor'):
                         logging.error(f"  cov_factor shape: {dist.cov_factor.shape}, has NaNs: {torch.isnan(dist.cov_factor).any()}, has Infs: {torch.isinf(dist.cov_factor).any()}")
                         # Log a small sample of the data if helpful, be careful with large tensors
                         # logging.error(f"  cov_factor sample: {dist.cov_factor.flatten()[:10]}")
                    if hasattr(dist, 'cov_diag'):
                         logging.error(f"  cov_diag shape: {dist.cov_diag.shape}, has NaNs: {torch.isnan(dist.cov_diag).any()}, has Infs: {torch.isinf(dist.cov_diag).any()}")
                         logging.error(f"  cov_diag values: {dist.cov_diag.flatten()}") # Diag is usually smaller
                except Exception as log_e:
                    logging.error(f"  Could not log distribution parameters: {log_e}")
            except Exception as other_e:
                 logging.warning(f"Trial {trial.number}: Unexpected error inspecting forecast index {i}: {other_e}")
                 problematic_indices.append(i) # Mark as problematic even for other errors

        if problematic_indices:
             logging.warning(f"Trial {trial.number}: Found {len(problematic_indices)} potentially problematic forecasts at indices: {problematic_indices}")
        else:
             logging.info(f"Trial {trial.number}: No immediate errors found during forecast inspection.")
        # --- End Debugging ---
        
        agg_metrics, _ = self.evaluator(iter(tss), iter(forecasts), num_series=self.data_module.num_target_vars)
        agg_metrics["trainable_parameters"] = summarize(estimator.create_lightning_module()).trainable_parameters
        self.metrics.append(agg_metrics.copy())

        # Log available metrics for debugging
        logging.info(f"Trial {trial.number} - Aggregated metrics calculated: {list(agg_metrics.keys())}")


        # Log GPU stats at the end of the trial
        self.log_gpu_stats(stage=f"Trial {trial.number} End")

        # Force garbage collection at the end of each trial
        gc.collect()
        torch.cuda.empty_cache()

        # Return the specified metric, with error handling
        try:
            metric_value = agg_metrics[self.metric]
            logging.info(f"Trial {trial.number} - Returning metric '{self.metric}': {metric_value}")
            return metric_value
        except KeyError:
            logging.error(f"Trial {trial.number} - Metric '{self.metric}' not found in calculated metrics: {list(agg_metrics.keys())}")
            # Return a value indicating failure based on optimization direction
            return float('inf') if self.config["optuna"]["direction"] == "minimize" else float('-inf')


# Update signature: Accept optuna_storage_target (URL string or storage object)
def tune_model(model, config, optuna_storage_target, lightning_module_class, estimator_class,
               max_epochs, limit_train_batches,
               distr_output_class, data_module, context_length_choices,
               metric="mean_wQuantileLoss", direction="minimize", n_trials=10,
               trial_protection_callback=None, seed=42):

    # Ensure WandB is correctly initialized with the proper directory
    if "logging" in config and "wandb_dir" in config["logging"]:
        wandb_dir = config["logging"]["wandb_dir"]
        os.makedirs(wandb_dir, exist_ok=True)
        os.environ["WANDB_DIR"] = wandb_dir
        logging.info(f"Set WANDB_DIR to {wandb_dir}")
        logging.info(f"WandB will create logs in {os.path.join(wandb_dir, 'wandb')}")

    study_name = config["optuna"]["study_name"]
    # Log storage target safely
    if isinstance(optuna_storage_target, str):
        # log safely without credentials
        log_storage_info = optuna_storage_target.split('@')[0] + '@...' if '@' in optuna_storage_target else optuna_storage_target
        logging.info(f"Using Optuna storage URL: {log_storage_info}")
    elif isinstance(optuna_storage_target, optuna.storages.BaseStorage):
        # It's a storage object
        log_storage_info = f"Optuna storage object: {type(optuna_storage_target).__name__}"
        if hasattr(optuna_storage_target, 'storage') and hasattr(optuna_storage_target.storage, 'path'):
             # for JournalStorage
             log_storage_info += f" (path: {optuna_storage_target.storage.path})"
        logging.info(log_storage_info)
    else:
        logging.warning(f"Unknown Optuna storage target type: {type(optuna_storage_target)}")
        log_storage_info = str(optuna_storage_target) # Fallback to string representation

    # NOTE: Restarting the study is now handled in the Slurm script by deleting the PGDATA directory
    # if the --restart_tuning flag is set. No specific handling needed here.

    # Use the provided storage target (URL string or object) directly
    storage_target = optuna_storage_target

    # Configure pruner based on settings
    pruner = None
    if "pruning" in config["optuna"] and config["optuna"]["pruning"].get("enabled", False):
        pruning_type = config["optuna"]["pruning"].get("type", "hyperband").lower()
        min_resource = config["optuna"]["pruning"].get("min_resource", 2)

        logging.info(f"Configuring pruner: type={pruning_type}, min_resource={min_resource}")

        if pruning_type == "hyperband":
            reduction_factor = config["optuna"]["pruning"].get("reduction_factor", 3)
            pruner = HyperbandPruner(
                min_resource=min_resource,
                max_resource=max_epochs,
                reduction_factor=reduction_factor
            )
            logging.info(f"Created HyperbandPruner with min_resource={min_resource}, max_resource={max_epochs}, reduction_factor={reduction_factor}")
        elif pruning_type == "median":
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=min_resource)
            logging.info(f"Created MedianPruner with n_startup_trials=5, n_warmup_steps={min_resource}")
        elif pruning_type == "percentile":
            percentile = config["optuna"]["pruning"].get("percentile", 25)
            pruner = PercentilePruner(percentile=percentile, n_startup_trials=5, n_warmup_steps=min_resource)
            logging.info(f"Created PercentilePruner with percentile={percentile}, n_startup_trials=5, n_warmup_steps={min_resource}")
        else:
            logging.warning(f"Unknown pruner type: {pruning_type}, using no pruning")
            pruner = NopPruner()
    else:
        logging.info("Pruning is disabled, using NopPruner")
        pruner = NopPruner()

    # Get worker ID for study creation/loading logic
    # Use WORKER_RANK consistent with run_model.py. Default to '0' if not set.
    worker_id = os.environ.get('WORKER_RANK', '0')

    # Create study on rank 0, load on other ranks
    study = None # Initialize study variable
    try:
        if worker_id == '0':
            logging.info(f"Rank 0: Creating/loading Optuna study '{study_name}' with pruner: {type(pruner).__name__}")
            study = create_study(
                study_name=study_name,
                storage=storage_target, # Pass URL string or storage object
                direction=direction,
                load_if_exists=True, # Rank 0 handles creation or loading
                sampler=TPESampler(seed=seed),
                pruner=pruner
            )
            logging.info(f"Rank 0: Study '{study_name}' created or loaded successfully.")

            # --- Launch Dashboard (Rank 0 only) ---
            # Launch dashboard only if storage is a URL
            if isinstance(storage_target, str):
                 launch_optuna_dashboard(config, storage_target)
            else:
                 logging.info("Skipping Optuna dashboard launch (not supported for non-URL storage types like JournalStorage).")
            # --------------------------------------
        else:
            # Non-rank-0 workers MUST load the study created by rank 0
            logging.info(f"Rank {worker_id}: Attempting to load existing Optuna study '{study_name}'")
            # Add a small delay and retry mechanism for loading, in case rank 0 is slightly delayed
            max_retries = 6 # Increased retries slightly
            retry_delay = 10 # Increased delay slightly
            for attempt in range(max_retries):
                try:
                    study = load_study(
                        study_name=study_name,
                        storage=storage_target, # Pass URL string or storage object
                        sampler=TPESampler(seed=seed), # Sampler might be needed for load_study too
                        pruner=pruner
                    )
                    logging.info(f"Rank {worker_id}: Study '{study_name}' loaded successfully on attempt {attempt+1}.")
                    break # Exit loop on success
                except KeyError as e: # Optuna <3.0 raises KeyError if study doesn't exist yet
                     if attempt < max_retries - 1:
                          logging.warning(f"Rank {worker_id}: Study '{study_name}' not found yet (attempt {attempt+1}/{max_retries}). Retrying in {retry_delay}s... Error: {e}")
                          time.sleep(retry_delay)
                     else:
                          logging.error(f"Rank {worker_id}: Failed to load study '{study_name}' after {max_retries} attempts (KeyError). Aborting.")
                          raise
                except Exception as e: # Catch other potential loading errors (e.g., DB connection issues)
                     logging.error(f"Rank {worker_id}: An unexpected error occurred while loading study '{study_name}' on attempt {attempt+1}: {e}", exc_info=True)
                     # Decide whether to retry on other errors or raise immediately
                     if attempt < max_retries - 1:
                          logging.warning(f"Retrying in {retry_delay}s...")
                          time.sleep(retry_delay)
                     else:
                          logging.error(f"Rank {worker_id}: Failed to load study '{study_name}' after {max_retries} attempts due to persistent errors. Aborting.")
                          raise # Re-raise other errors after retries

            # Check if study was successfully loaded after the loop
            if study is None:
                 # This condition should ideally be caught by the error handling within the loop, but added for safety.
                 raise RuntimeError(f"Rank {worker_id}: Could not load study '{study_name}' after multiple retries.")

    except Exception as e:
        # Log error with rank information
        logging.error(f"Rank {worker_id}: Error creating/loading study '{study_name}': {str(e)}", exc_info=True)
        # Log storage URL safely
        if isinstance(storage_target, str):
             log_storage_info_safe = storage_target.split('@')[0] + '@...' if '@' in storage_target else storage_target
        elif isinstance(storage_target, optuna.storages.BaseStorage):
             log_storage_info_safe = f"Optuna storage object: {type(storage_target).__name__}"
             if hasattr(storage_target, 'storage') and hasattr(storage_target.storage, 'path'):
                  log_storage_info_safe += f" (path: {storage_target.storage.path})"
        else:
             log_storage_info_safe = str(storage_target)
        logging.error(f"Error details - Type: {type(e).__name__}, Storage: {log_storage_info_safe}")
        raise

    # Worker ID already fetched above for study creation/loading

    logging.info(f"Worker {worker_id}: Participating in Optuna study {study_name}")

    tuning_objective = MLTuningObjective(model=model, config=config,
                                        lightning_module_class=lightning_module_class,
                                        estimator_class=estimator_class,
                                        distr_output_class=distr_output_class,
                                        max_epochs=max_epochs,
                                        limit_train_batches=limit_train_batches,
                                        data_module=data_module,
                                        context_length_choices=context_length_choices,
                                        metric=metric,
                                        seed=seed)

    # Use the trial protection callback if provided
    objective_fn = (lambda trial: trial_protection_callback(tuning_objective, trial)) if trial_protection_callback else tuning_objective

    try:
        # Let Optuna handle trial distribution - each worker will ask the storage for a trial
        # Show progress bar only on rank 0 to avoid cluttered logs
        study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=(worker_id=='0'))
    except Exception as e:
        logging.error(f"Worker {worker_id}: Failed during study optimization: {str(e)}", exc_info=True)
        # Optionally, report trial as failed if possible? Optuna might handle this internally.
        # Consider adding: if 'trial' in locals(): trial.report(float('inf'), step=0); trial.storage.set_trial_state(trial._trial_id, optuna.trial.TrialState.FAIL)
        raise

    # All workers log their contribution
    logging.info(f"Worker {worker_id} completed optimization")

    # Generate visualizations if enabled (only rank 0 should do this)
    if worker_id == '0' and config.get("optuna", {}).get("visualization", {}).get("enabled", False):
        # Ensure study object is available (it should be, unless loading failed catastrophically)
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

    return study.best_params