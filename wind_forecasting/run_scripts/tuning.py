import os
from pytorch_lightning.utilities.model_summary import summarize
from gluonts.evaluation import MultivariateEvaluator, make_evaluation_predictions
from gluonts.model.forecast_generator import DistributionForecastGenerator
from gluonts.time_feature._base import second_of_minute, minute_of_hour, hour_of_day, day_of_year
from gluonts.transform import ExpectedNumInstanceSampler, ValidationSplitSampler
import logging
import torch
import gc

from mysql.connector import connect as sql_connect
from optuna import create_study
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MLTuningObjective:
    def __init__(self, *, model, config, lightning_module_class, estimator_class, distr_output_class, max_epochs, limit_train_batches, data_module, metric, context_length_choices):
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

        self.config["trainer"]["max_epochs"] = max_epochs
        self.config["trainer"]["limit_train_batches"] = limit_train_batches
        
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
        # Log GPU stats at the beginning of the trial
        self.log_gpu_stats(stage=f"Trial {trial.number} Start")
        
        # params = self.get_params(trial)
        params = self.estimator_class.get_params(trial, self.context_length_choices)
        self.config["dataset"].update({k: v for k, v in params.items() if k in self.config["dataset"]})
        self.config["model"][self.model].update({k: v for k, v in params.items() if k in self.config["model"][self.model]})
        self.config["trainer"].update({k: v for k, v in params.items() if k in self.config["trainer"]})
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
        )
        
        # Log GPU stats after training
        self.log_gpu_stats(stage=f"Trial {trial.number} After Training")
        
        model = self.lightning_module_class.load_from_checkpoint(train_output.trainer.checkpoint_callback.best_model_path)
        transformation = estimator.create_transformation(use_lazyframe=False)
        predictor = estimator.create_predictor(transformation, model, 
                                                forecast_generator=DistributionForecastGenerator(estimator.distr_output))
        
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=self.data_module.test_dataset, 
            predictor=predictor,
            output_distr_params=True
        )
        
        forecasts = list(forecast_it)
        tss = list(ts_it)
        agg_metrics, _ = self.evaluator(iter(tss), iter(forecasts), num_series=self.data_module.num_target_vars)
        agg_metrics["trainable_parameters"] = summarize(estimator.create_lightning_module()).trainable_parameters
        self.metrics.append(agg_metrics.copy())
        
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
        
        return agg_metrics[self.metric]


def get_storage(use_rdb, study_name, journal_storage_dir=None):
    """
    Get storage for Optuna studies.
    
    Args:
        use_rdb: Whether to use SQLite storage
        study_name: Name of the study
        journal_storage_dir: Directory to store journal files
        
    Returns:
        Storage object for Optuna
    """
    if use_rdb:
        # SQLite with WAL mode
        db_path = os.path.join(journal_storage_dir, f"{study_name}.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Check if the database file exists
        db_exists = os.path.exists(db_path)
        
        # SQLite connection string with WAL mode
        storage_url = f"sqlite:///{db_path}?journal_mode=WAL&synchronous=NORMAL&timeout=60&busy_timeout=60000"
                
        # Manually create and initialize the database with WAL mode if it doesn't exist
        if not db_exists:
            import sqlite3
            conn = sqlite3.connect(db_path, timeout=60)
            # Enable WAL mode and optimize settings
            settings = [
                ("PRAGMA journal_mode=WAL", "Set WAL mode"),
                ("PRAGMA synchronous=NORMAL", "Balance between durability and performance"),
                ("PRAGMA cache_size=10000", "Increase cache size for better performance"),
                ("PRAGMA temp_store=MEMORY", "Store temp tables in memory"),
                ("PRAGMA mmap_size=30000000000", "Memory map DB up to 30GB if beneficial"),
                ("PRAGMA busy_timeout=60000", "Wait up to 60 seconds on busy database"),
                ("PRAGMA wal_autocheckpoint=1000", "Checkpoint after 1000 pages (default)")
            ]
            for command, description in settings:
                try:
                    result = conn.execute(command).fetchone()[0]
                    logging.info(f"SQLite {description}: {result}")
                except Exception as e:
                    logging.warning(f"Failed to execute {command}: {e}")
            
            conn.commit()
            conn.close()
            
            # Verify WAL mode is working
            check_wal_mode(db_path)
            
        return storage_url
    else:
        # Existing journal storage implementation
        journal_file = os.path.join(journal_storage_dir, f"{study_name}.journal")
        storage = JournalStorage(
            JournalFileBackend(journal_file)
        )
        return storage
    
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

def get_tuned_params(use_rdb, study_name):
    storage = get_storage(use_rdb=use_rdb, study_name=study_name)
    try:
        study_id = storage.get_study_id_from_name(study_name)
    except Exception:
        raise FileNotFoundError(f"Optuna study {study_name} not found. Please run tune_hyperparameters_multi for all outputs first.")
    # self.model[output].set_params(**storage.get_best_trial(study_id).params)
    # storage.get_all_studies()[0]._study_id
    # estimato = self.create_model(**storage.get_best_trial(study_id).params)
    return storage.get_best_trial(study_id).params 

def tune_model(model, config, lightning_module_class, estimator_class, 
               max_epochs, limit_train_batches, 
               distr_output_class, data_module, context_length_choices, 
               journal_storage_dir, use_rdb=True, restart_study=False, metric="mean_wQuantileLoss", 
               direction="minimize", n_trials=10, trial_protection_callback=None):
    
    # Make sure the journal directory exists
    os.makedirs(journal_storage_dir, exist_ok=True)
    
    # Ensure WandB is correctly initialized with the proper directory
    if "logging" in config and "wandb_dir" in config["logging"]:
        wandb_dir = config["logging"]["wandb_dir"]
        os.makedirs(wandb_dir, exist_ok=True)
        os.environ["WANDB_DIR"] = wandb_dir
    
    study_name = config["optuna"]["study_name"]
    logging.info(f"Allocating storage for Optuna study {study_name}.")  
    storage = get_storage(use_rdb=use_rdb, study_name=study_name, journal_storage_dir=journal_storage_dir)
    
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
            
            # Reinitialize storage
            storage = get_storage(use_rdb=use_rdb, study_name=study_name, journal_storage_dir=journal_storage_dir)
        else:
            # Original journal file handling
            logging.info(f"Deleting existing Optuna studies {storage.get_all_studies()}.")  
            for s in storage.get_all_studies():
                storage.delete_study(s._study_id)
    
    # Create or load the study with standard hyperparameters
    try:
        logging.info(f"Creating Optuna study {study_name}.")
        study = create_study(
            study_name=study_name,
            storage=storage,
            direction=direction,
            load_if_exists=True,
            sampler=TPESampler(seed=42)
        )
    except Exception as e:
        logging.error(f"Error creating study: {e}")
        raise
    
    # Get worker ID for logging
    worker_id = os.environ.get('SLURM_PROCID', '0')
    
    logging.info(f"Worker {worker_id}: Optimizing Optuna study {study_name}.")
    
    tuning_objective = MLTuningObjective(model=model, config=config, 
                                        lightning_module_class=lightning_module_class,
                                        estimator_class=estimator_class, 
                                        distr_output_class=distr_output_class,
                                        max_epochs=max_epochs,
                                        limit_train_batches=limit_train_batches,
                                        data_module=data_module, 
                                        context_length_choices=context_length_choices, 
                                        metric=metric)
    
    # Each worker contributes trials to the shared study
    n_trials_per_worker = max(1, n_trials // int(os.environ.get('SLURM_NTASKS', '1')))
    logging.info(f"Worker {worker_id} will run {n_trials_per_worker} trials")
    
    # Use the trial protection callback if provided
    objective_fn = (lambda trial: trial_protection_callback(tuning_objective, trial)) if trial_protection_callback else tuning_objective
    
    try:
        study.optimize(objective_fn, n_trials=n_trials_per_worker, show_progress_bar=True)
    except Exception as e:
        logging.error(f"Worker {worker_id} failed with error: {e}")
        logging.error(f"Error details: {type(e).__name__}")
        raise

    if worker_id == '0':  # Only the first worker prints the final results
        logging.info("Number of finished trials: {}".format(len(study.trials)))
        logging.info("Best trial:")
        trial = study.best_trial
        logging.info("  Value: {}".format(trial.value))
        logging.info("  Params: ")
        for key, value in trial.params.items():
            logging.info("    {}: {}".format(key, value))
        
    return study.best_params