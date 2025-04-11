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
import inspect
# Imports for Optuna
import optuna # Import the base optuna module for type hints
from optuna import create_study, load_study
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner, MedianPruner, PercentilePruner, NopPruner
from optuna.integration import PyTorchLightningPruningCallback
import lightning.pytorch as pl # Import pl alias

from optuna import create_study
from mysql.connector import connect as sql_connect
from optuna.storages import JournalStorage, RDBStorage
from optuna.storages.journal import JournalFileBackend

from wind_forecasting.utils.optuna_visualization import launch_optuna_dashboard
from wind_forecasting.utils.trial_utils import handle_trial_with_oom_protection

import random
import numpy as np

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
        
        random.seed(trial_seed)
        np.random.seed(trial_seed)
        logging.info(f"Set random seed for trial {trial.number} to {trial_seed}")

        # Log GPU stats at the beginning of the trial
        self.log_gpu_stats(stage=f"Trial {trial.number} Start")

        # params = self.get_params(trial) # TODO PUT BACK
        # params = self.estimator_class.get_params(trial, self.context_length_choices)
        
        estimator_sig = inspect.signature(self.estimator_class.__init__)
        estimator_params = [param.name for param in estimator_sig.parameters.values()]
        # TODO PUT BACK
        # if "dim_feedforward" not in params and "d_model" in params:
        #     # set dim_feedforward to 4x the d_model found in this trial 
        #     params["dim_feedforward"] = params["d_model"] * 4
        # elif "d_model" in estimator_params and estimator_sig.parameters["d_model"].default is not inspect.Parameter.empty:
        #     # if d_model is not contained in the trial but is a paramter, get the default
        #     params["dim_feedforward"] = estimator_sig.parameters["d_model"].default * 4
        
        # logging.info(f"Testing params {tuple((k, v) for k, v in params.items())}")
        
        # self.config["dataset"].update({k: v for k, v in params.items() if k in self.config["dataset"]})
        # self.config["model"][self.model].update({k: v for k, v in params.items() if k in self.config["model"][self.model]})
        # self.config["trainer"].update({k: v for k, v in params.items() if k in self.config["trainer"]})

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

        # context_length = int(pd.Timedelta(self.config["dataset"]["context_length"], unit="s") / pd.Timedelta(self.data_module.freq))
        # TODO test informer {'context_length': 90, 'batch_size': 32, 'num_encoder_layers': 3, 'num_decoder_layers': 3, 'd_model': 128, 'n_heads': 8}
        self.config["dataset"]["batch_size"] = 32
        self.config["model"][self.model]["num_encoder_layers"] = 3
        self.config["model"][self.model]["num_decoder_layers"] = 3
        self.config["model"][self.model]["d_model"] = 128
        self.config["model"][self.model]["n_heads"] = 8
        self.config["model"][self.model]["dim_feedforward"] = 128 * 4
        self.config["dataset"]["context_length"] = 15
        
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
            use_lazyframe=False,
            batch_size=self.config["dataset"].get("batch_size", 128),
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

         # TODO PUT BACK
        # train_output = estimator.train(
        #     training_data=self.data_module.train_dataset,
        #     # validation_data=self.data_module.val_dataset, # omit since it is used to validate by optuna
        #     forecast_generator=DistributionForecastGenerator(estimator.distr_output)
        #     # Note: The trainer_kwargs including callbacks are passed internally by the estimator
        # )

        # Log GPU stats after training
        self.log_gpu_stats(stage=f"Trial {trial.number} After Training")

        # /Users/ahenry/Documents/toolboxes/wind_forecasting/examples/logging/informer_aoifemac_awaken/wind_forecasting/i0w51is7/checkpoints/epoch=9-step=10000.ckpt
        # model = self.lightning_module_class.load_from_checkpoint(train_output.trainer.checkpoint_callback.best_model_path) # TODO PUT BACK
        
        if os.path.exists("/Users/ahenry/Downloads/epoch=9-step=10000.ckpt"):
            checkpoint = "/Users/ahenry/Downloads/epoch=9-step=10000.ckpt"
        else:
            checkpoint = "/projects/ssc/ahenry/wind_forecasting/logging/informer_kestrel_awaken_per_turbine/wind_forecasting/j5v9brpu/checkpoints/epoch=9-step=10000.ckpt"
        model = self.lightning_module_class.load_from_checkpoint(checkpoint)
        transformation = estimator.create_transformation(use_lazyframe=False)
        predictor = estimator.create_predictor(transformation, model,
                                                forecast_generator=DistributionForecastGenerator(estimator.distr_output))

        forecast_it, ts_it = make_evaluation_predictions(
            dataset=self.data_module.val_dataset,
            predictor=predictor,
            output_distr_params={"loc": "mean", "cov_factor": "cov_factor", "cov_diag": "cov_diag"}
        )
        # from itertools import islice # TODO REMOVE
        # forecasts = list(islice(forecast_it, 0, 2))
        # tss = list(islice(ts_it, 0, 2))
        # forecasts = list(forecast_it)
        # tss = list(ts_it)
        forecasts = forecast_it
        tss = ts_it
        agg_metrics, _ = self.evaluator(tss, forecasts, num_series=self.data_module.num_target_vars)
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

def get_storage(backend, study_name, storage_dir=None):
    if backend == "mysql":
        logging.info(f"Connecting to RDB database {study_name}")
        # try:
        db = sql_connect(host="localhost", user="root",
                        database=study_name)       
        # except Exception: 
        #     db = sql_connect(host="localhost", user="root")
        #     cursor = db.cursor()
        #     cursor.execute(f"CREATE DATABASE {study_name}") 
        # finally:
        storage = RDBStorage(url=f"mysql://{db.user}@{db.server_host}:{db.server_port}/{study_name}")
    elif backend == "sqlite":
        # SQLite with WAL mode - using a simpler URL format
        # os.makedirs(storage_dir, exist_ok=True)
        db_path = os.path.join(storage_dir, f"{study_name}.db")

        # Use a simplified connection string format that Optuna expects
        storage_url = f"sqlite:///{db_path}"

        # Check if database already exists and initialize WAL mode directly
        if not os.path.exists(db_path):
            try:
                import sqlite3
                # Create the database manually first with WAL settings
                conn = sqlite3.connect(db_path, timeout=60000)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA cache_size=10000")
                conn.execute("PRAGMA temp_store=MEMORY")
                conn.execute("PRAGMA busy_timeout=60000")
                conn.execute("PRAGMA wal_autocheckpoint=1000")
                conn.commit()
                conn.close()
                logging.info(f"Created SQLite database with WAL mode at {db_path}")
            except Exception as e:
                logging.error(f"Error initializing SQLite database: {e}")
                
        storage = RDBStorage(url=storage_url)
        
    elif backend == "journal":
        logging.info(f"Connecting to Journal database {study_name}")
        storage = JournalStorage(JournalFileBackend(os.path.join(storage_dir, f"{study_name}.db")))
    
    return storage

def get_tuned_params(study_name, backend, storage_dir):
    logging.info(f"Allocating storage for Optuna study {study_name}.")  
    storage = get_storage(backend=backend, study_name=study_name, storage_dir=storage_dir)
    try:
        study_id = storage.get_study_id_from_name(study_name)
    except Exception:
        raise FileNotFoundError(f"Optuna study {study_name} not found. Please run tune_hyperparameters_multi for all outputs first.")
    # self.model[output].set_params(**storage.get_best_trial(study_id).params)
    # storage.get_all_studies()[0]._study_id
    # estimato = self.create_model(**storage.get_best_trial(study_id).params)
    return storage.get_best_trial(study_id).params 

# Update signature: Add optuna_storage_url, remove storage_dir, use_rdb, restart_study
def tune_model(model, config, study_name, optuna_storage, lightning_module_class, estimator_class,
               max_epochs, limit_train_batches,
               distr_output_class, data_module, context_length_choices,
               metric="mean_wQuantileLoss", direction="minimize", n_trials=10,
               trial_protection_callback=None, seed=42):

    # Log safely without credentials if they were included (they aren't for socket trust)
    if hasattr(optuna_storage, "url"):
        log_storage_url = optuna_storage.url.split('@')[0] + '@...' if '@' in optuna_storage.url else optuna_storage.url
        logging.info(f"Using Optuna storage URL: {log_storage_url}")

    # NOTE: Restarting the study is now handled in the Slurm script by deleting the PGDATA directory
   
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
                storage=optuna_storage,
                direction=direction,
                load_if_exists=True, # Rank 0 handles creation or loading
                sampler=TPESampler(seed=seed),
                pruner=pruner
            )
            logging.info(f"Rank 0: Study '{study_name}' created or loaded successfully.")

            # --- Launch Dashboard (Rank 0 only) ---
            if hasattr(optuna_storage, "url"):
                launch_optuna_dashboard(config, optuna_storage.url) # Call imported function
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
                        storage=optuna_storage,
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
        if hasattr(optuna_storage, "url"):
            log_storage_url_safe = str(optuna_storage.url).split('@')[0] + '@...' if '@' in str(optuna_storage.url) else str(optuna_storage.url)
            logging.error(f"Error details - Type: {type(e).__name__}, Storage: {log_storage_url_safe}")
        else:
            logging.error(f"Error details - Type: {type(e).__name__}, Storage: Journal")
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