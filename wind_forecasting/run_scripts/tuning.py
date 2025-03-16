import os
from pytorch_lightning.utilities.model_summary import summarize
from gluonts.evaluation import MultivariateEvaluator, make_evaluation_predictions
from gluonts.model.forecast_generator import DistributionForecastGenerator
from gluonts.time_feature._base import second_of_minute, minute_of_hour, hour_of_day, day_of_year
from gluonts.transform import ExpectedNumInstanceSampler, ValidationSplitSampler
import logging

from mysql.connector import connect as sql_connect
from optuna import create_study
from optuna.storages import JournalStorage, RDBStorage
from optuna.storages.journal import JournalFileBackend
from optuna.samplers import TPESampler

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
    
    # def get_params(self, trial) -> dict:
    #     return {
    #     "context_length": trial.suggest_int("context_length", self.data_module.prediction_length, self.data_module.prediction_length*7, 4),
    #     "max_epochs": trial.suggest_int("max_epochs", 1, 10, 2),
    #     "batch_size": trial.suggest_int("batch_size", 128, 256, 64),
    #     "num_encoder_layers": trial.suggest_int("num_encoder_layers", 2, 16,4),
    #     "num_decoder_layers": trial.suggest_int("num_decoder_layers", 2, 16,4),
    #      "num_batches_per_epoch":trial.suggest_int("num_batches_per_epoch", 100, 200, 100),   
    #     }
     
    def __call__(self, trial):
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
        
        train_output = estimator.train(
            training_data=self.data_module.train_dataset,
            validation_data=self.data_module.val_dataset,
            forecast_generator=DistributionForecastGenerator(estimator.distr_output)
        )
        
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
        return agg_metrics[self.metric]


def get_storage(storage_type, study_name, journal_storage_dir=None):
    if storage_type == "mysql":
        logging.info(f"Connecting to RDB database {study_name}")
        try:
            db = sql_connect(host="localhost", user="root",
                            database=study_name)       
        except Exception: 
            db = sql_connect(host="localhost", user="root")
            cursor = db.cursor()
            cursor.execute(f"CREATE DATABASE {study_name}") 
        finally:
            storage = RDBStorage(url=f"mysql://{db.user}@{db.server_host}:{db.server_port}/{study_name}")
    elif storage_type == "sqlite":
        # SQLite with WAL mode - using a simpler URL format
        os.makedirs(journal_storage_dir, exist_ok=True)
        db_path = os.path.join(journal_storage_dir, f"{study_name}.db")

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
        
    elif storage_type == "journal":
        logging.info(f"Connecting to Journal database {study_name}")
        storage = JournalStorage(JournalFileBackend(os.path.join(journal_storage_dir, f"{study_name}.log")))
    
    return storage

def get_tuned_params(storage_type, study_name):
    storage = get_storage(storage_type=storage_type, study_name=study_name)
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
               journal_storage_dir, storage_type="sqlite", restart_tuning=False, metric="mean_wQuantileLoss", 
               direction="minimize", n_trials=10):
    
    assert storage_type in ["sqlite", "mysql", "journal"]
    study_name = f"tuning_{model}"
    logging.info(f"Allocating storage for Optuna study {study_name}.")  
    storage = get_storage(storage_type=storage_type, study_name=study_name, journal_storage_dir=journal_storage_dir)
    if restart_tuning:
        logging.info(f"Deleting existing Optuna studies {storage.get_all_studies()}.")  
        for s in storage.get_all_studies():
            storage.delete_study(s._study_id)
            
    logging.info(f"Creating Optuna study {study_name}.")  
    study = create_study(study_name=study_name,
                         storage=storage,
                         direction=direction,
                         load_if_exists=True,
                         sampler=TPESampler())
    
    logging.info(f"Optimizing Optuna study {study_name}.") 
    tuning_objective = MLTuningObjective(model=model, config=config, 
                                        lightning_module_class=lightning_module_class,
                                        estimator_class=estimator_class, 
                                        distr_output_class=distr_output_class,
                                        max_epochs=max_epochs,
                                        limit_train_batches=limit_train_batches,
                                        data_module=data_module, 
                                        context_length_choices=context_length_choices, 
                                        metric=metric)
    study.optimize(tuning_objective, n_trials=n_trials, show_progress_bar=True)

    logging.info("Number of finished trials: {}".format(len(study.trials)))

    logging.info("Best trial:")
    trial = study.best_trial

    logging.info("  Value: {}".format(trial.value))

    logging.info("  Params: ")
    for key, value in trial.params.items():
        logging.info("    {}: {}".format(key, value))
        
    return study.best_params