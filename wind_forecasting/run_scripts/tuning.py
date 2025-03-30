import os
from pytorch_lightning.utilities.model_summary import summarize
from gluonts.evaluation import MultivariateEvaluator, make_evaluation_predictions
from gluonts.model.forecast_generator import DistributionForecastGenerator
from gluonts.time_feature._base import second_of_minute, minute_of_hour, hour_of_day, day_of_year
from gluonts.transform import ExpectedNumInstanceSampler, ValidationSplitSampler
import logging
import torch
import gc

# Imports for Optuna
import optuna # Import the base optuna module for type hints
from optuna import create_study, load_study
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner, MedianPruner, PercentilePruner, NopPruner
from optuna.integration import PyTorchLightningPruningCallback
import lightning.pytorch as pl # Import pl alias

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
        if self.pruning_enabled:
            # Initialize callbacks list if it doesn't exist
            if "callbacks" not in self.config["trainer"]:
                self.config["trainer"]["callbacks"] = []
            elif self.config["trainer"]["callbacks"] is None:
                self.config["trainer"]["callbacks"] = []
            
            # Check if callbacks is already a list, if not convert it
            if not isinstance(self.config["trainer"]["callbacks"], list):
                self.config["trainer"]["callbacks"] = [self.config["trainer"]["callbacks"]]
            
            # Create the SAFE wrapper for the PyTorch Lightning pruning callback
            # This ensures it inherits directly from pl.Callback
            pruning_callback = SafePruningCallback(
                trial,
                monitor=self.metric  # Use the same metric for pruning as for optimization
            )
            
            # Add to callbacks list
            self.config["trainer"]["callbacks"].append(pruning_callback)
            logging.info(f"Added pruning callback for trial {trial.number}, monitoring {self.metric}")
        
        # Verify GPU configuration before creating estimator
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            logging.info(f"Creating estimator using GPU {device}: {torch.cuda.get_device_name(device)}")
            
            # Ensure we have the right GPU configuration in trainer_kwargs
            # This helps avoid the "You requested gpu: [0, 1, 2, 3] But your machine only has: [0]" error
            if "devices" in self.config["trainer"] and self.config["trainer"]["devices"] > 1:
                if "CUDA_VISIBLE_DEVICES" in os.environ and len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 1:
                    logging.warning(f"Overriding trainer devices={self.config['trainer']['devices']} to 1 due to CUDA_VISIBLE_DEVICES")
                    self.config["trainer"]["devices"] = 1
                    self.config["trainer"]["strategy"] = "auto"
        else:
            logging.warning("No CUDA available for estimator creation")
    
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

            batch_size=self.config["dataset"].setdefault("batch_size", 128),
            num_batches_per_epoch=self.config["trainer"]["limit_train_batches"],
            train_sampler=ExpectedNumInstanceSampler(num_instances=1.0, min_past=self.config["dataset"]["context_length"], min_future=self.data_module.prediction_length),
            validation_sampler=ValidationSplitSampler(min_past=self.config["dataset"]["context_length"], min_future=self.data_module.prediction_length),
            time_features=[second_of_minute, minute_of_hour, hour_of_day, day_of_year],
            distr_output=self.distr_output_class(dim=self.data_module.num_target_vars, **self.config["model"]["distr_output"]["kwargs"]),
            trainer_kwargs=self.config["trainer"],
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
        agg_metrics, _ = self.evaluator(iter(tss), iter(forecasts), num_series=self.data_module.num_target_vars)
        agg_metrics["trainable_parameters"] = summarize(estimator.create_lightning_module()).trainable_parameters
        self.metrics.append(agg_metrics.copy())

        # Log available metrics for debugging
        logging.info(f"Trial {trial.number} - Aggregated metrics calculated: {list(agg_metrics.keys())}")

        # Checkpoint the WAL file every 5 trials to prevent excessive growth
        if trial.number % 5 == 0 and hasattr(trial.study, '_storage') and hasattr(trial.study._storage, '_url'):
            try:
                if hasattr(trial.study._storage, '_url') and 'sqlite' in trial.study._storage._url:
                    import sqlite3
                    db_path = trial.study._storage._url.replace('sqlite:///', '').split('?')[0]
                    conn = sqlite3.connect(db_path)
                    conn.execute("PRAGMA wal_checkpoint(RESTART)")
                    conn.close()
                    logging.info(f"Forced WAL checkpoint at trial {trial.number}")
            except Exception as e:
                logging.warning(f"Failed to checkpoint WAL file: {e}")
        
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


def get_storage(use_rdb, study_name, journal_storage_dir=None):
    """
    Get storage for Optuna studies.
    
    Args:
        use_rdb: Whether to use SQLite storage
        study_name: Name of the study
        journal_storage_dir: Directory to store journal files
        
    Returns:
        Storage URL string for Optuna
    """
    if use_rdb:
        # For parallel execution, we need to handle SQLite concurrency issues
        os.makedirs(journal_storage_dir, exist_ok=True)
        
        # Get worker ID to potentially create worker-specific database files
        worker_id = os.environ.get('SLURM_PROCID', '0')
        
        # Use a single database with optimized locking parameters for SLURM parallel jobs
        db_path = os.path.join(journal_storage_dir, f"{study_name}.db")
        
        # Optuna connection pooling is disabled by default in sqlite
        # These optimized parameters significantly improve parallel access:
        # - timeout=600: 10 minutes wait on locks (prevents early failures)
        # - isolation_level=IMMEDIATE: Reduces deadlocks by starting transaction immediately
        # - connect_args: Additional SQLite configuration for better concurrency
        storage_url = f"sqlite:///{db_path}?timeout=600&isolation_level=IMMEDIATE"
        
        # Log the database file size for monitoring
        if os.path.exists(db_path):
            db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
            logging.info(f"SQLite database size: {db_size_mb:.2f} MB")
        
        # Check if database already exists and initialize WAL mode directly
        if not os.path.exists(db_path):
            try:
                import sqlite3
                # Create the database manually first with WAL settings
                conn = sqlite3.connect(db_path, timeout=300000)  # 5 minutes timeout
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")  # Less durability but faster
                conn.execute("PRAGMA cache_size=10000")    # Larger cache
                conn.execute("PRAGMA temp_store=MEMORY")   # Use memory for temp operations
                conn.execute("PRAGMA busy_timeout=300000") # 5 minutes busy timeout
                conn.execute("PRAGMA wal_autocheckpoint=1000") # Less frequent checkpoints
                conn.execute("PRAGMA mmap_size=30000000000") # Memory mapping for large DB
                conn.commit()
                conn.close()
                logging.info(f"Created SQLite database with optimized settings at {db_path}")
            except Exception as e:
                logging.error(f"Error initializing SQLite database: {e}")
                
        # Log worker assignment for debugging
        logging.info(f"Worker {worker_id} using SQLite database at {db_path}")
        
        return storage_url
    else:
        # Journal storage implementation with URL format
        os.makedirs(journal_storage_dir, exist_ok=True)
        journal_file = os.path.join(journal_storage_dir, f"{study_name}.journal")
        
        # Create a valid storage URL for the journal file
        # Format: journal:///path/to/journal/file
        storage_url = f"journal:///{journal_file}"
        logging.info(f"Using journal storage at {journal_file}")
        
        return storage_url
    
def check_wal_mode(db_path):
    """Verify that WAL mode is working on the filesystem."""
    import sqlite3
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode")
        result = cursor.fetchone()[0]
        conn.close()
        
        if result.upper() != 'WAL':
            logging.warning(f"WAL mode not supported on {db_path}! Using {result} instead.")
            return False
        else:
            logging.info(f"Successfully confirmed WAL mode on {db_path}")
            return True
    except Exception as e:
        logging.error(f"Error checking WAL mode: {e}")
        return False

def get_tuned_params(use_rdb, study_name, journal_storage_dir=None):
    storage_url = get_storage(use_rdb=use_rdb, study_name=study_name, journal_storage_dir=journal_storage_dir)
    try:
        from optuna import load_study
        study = load_study(study_name=study_name, storage=storage_url)
        return study.best_trial.params
    except Exception as e:
        logging.error(f"Error retrieving tuned parameters: {e}")
        raise FileNotFoundError(f"Optuna study {study_name} not found. Please run tune_hyperparameters_multi for all outputs first.")

def tune_model(model, config, lightning_module_class, estimator_class, 
               max_epochs, limit_train_batches, 
               distr_output_class, data_module, context_length_choices, 
               journal_storage_dir, use_rdb=True, restart_study=False, metric="mean_wQuantileLoss", 
               direction="minimize", n_trials=10, trial_protection_callback=None, seed=42):
    
    # Make sure the journal directory exists
    os.makedirs(journal_storage_dir, exist_ok=True)
    
    # Ensure WandB is correctly initialized with the proper directory
    if "logging" in config and "wandb_dir" in config["logging"]:
        wandb_dir = config["logging"]["wandb_dir"]
        os.makedirs(wandb_dir, exist_ok=True)
        os.environ["WANDB_DIR"] = wandb_dir
        logging.info(f"Set WANDB_DIR to {wandb_dir}")
        logging.info(f"WandB will create logs in {os.path.join(wandb_dir, 'wandb')}")
    
    study_name = config["optuna"]["study_name"]
    logging.info(f"Allocating storage for Optuna study {study_name} in {journal_storage_dir}")  
    
    # Handle restarting the study differently for SQLite
    if restart_study:
        if use_rdb:
            # For SQLite, we can drop and recreate the DB
            db_path = os.path.join(journal_storage_dir, f"{study_name}.db")
            wal_path = f"{db_path}-wal"
            shm_path = f"{db_path}-shm"
            
            # Remove the database and associated WAL files if they exist
            for path in [db_path, wal_path, shm_path]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                        logging.info(f"Deleted {path}")
                    except Exception as e:
                        logging.warning(f"Could not delete {path}: {e}")
        else:
            # Original journal file handling
            journal_file = os.path.join(journal_storage_dir, f"{study_name}.journal")
            if os.path.exists(journal_file):
                try:
                    os.remove(journal_file)
                    logging.info(f"Deleted journal file {journal_file}")
                except Exception as e:
                    logging.warning(f"Could not delete journal file {journal_file}: {e}")
    
    # Get storage after potential deletions
    storage_url = get_storage(use_rdb=use_rdb, study_name=study_name, journal_storage_dir=journal_storage_dir)
    
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
    
    # Create or load the study with standard hyperparameters
    try:
        logging.info(f"Creating Optuna study {study_name} with pruner: {type(pruner).__name__}")
        study = create_study(
            study_name=study_name,
            storage=storage_url,
            direction=direction,
            load_if_exists=True,
            sampler=TPESampler(seed=seed),  # Use the seed provided as an argument
            pruner=pruner  # Add the pruner
        )
        logging.info(f"Study successfully created or loaded: {study_name}")
    except Exception as e:
        logging.error(f"Error creating study: {str(e)}")
        logging.error(f"Error type: {type(e).__name__}")
        logging.error(f"Storage type: {type(storage_url).__name__}")
        logging.error(f"Storage value: {storage_url}")
        raise
    
    # Get worker ID for logging
    worker_id = os.environ.get('SLURM_PROCID', '0')
    
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
        # Let Optuna handle trial distribution - each worker will get trials automatically
        study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=True)
    except Exception as e:
        logging.error(f"Worker {worker_id} failed with error: {str(e)}")
        logging.error(f"Error details: {type(e).__name__}")
        raise

    # All workers log their contribution
    logging.info(f"Worker {worker_id} completed optimization")
    
    # Generate visualizations if enabled (primary worker only)
    if worker_id == '0' and "visualization" in config["optuna"] and config["optuna"]["visualization"].get("enabled", False):
        try:
            from wind_forecasting.utils.optuna_visualization import generate_visualizations
            
            # Determine output directory
            visualization_dir = config["optuna"]["visualization"].get("output_dir")
            if not visualization_dir:
                visualization_dir = os.path.join(journal_storage_dir, "visualizations")
            
            # Expand variable references if present
            if isinstance(visualization_dir, str) and "${" in visualization_dir:
                if "${logging.optuna_dir}" in visualization_dir:
                    visualization_dir = visualization_dir.replace("${logging.optuna_dir}", journal_storage_dir)
            
            # Generate plots
            logging.info(f"Generating Optuna visualizations in {visualization_dir}")
            summary_path = generate_visualizations(study, visualization_dir, config["optuna"]["visualization"])
            
            if summary_path:
                logging.info(f"Generated Optuna visualizations - summary available at: {summary_path}")
            else:
                logging.warning("No visualizations were generated - study may not have enough completed trials")
        except Exception as e:
            logging.error(f"Failed to generate Optuna visualizations: {e}")
            import traceback
            logging.error(traceback.format_exc())
    
    # Only log best trial once
    if worker_id == '0':
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