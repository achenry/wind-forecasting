import optuna
from pytorch_lightning.utilities.model_summary import summarize
from gluonts.evaluation import MultivariateEvaluator, make_evaluation_predictions
from gluonts.model.forecast_generator import DistributionForecastGenerator
from gluonts.time_feature._base import second_of_minute, minute_of_hour, hour_of_day, day_of_year
from gluonts.transform import ExpectedNumInstanceSampler, ValidationSplitSampler

class MLTuningObjective:
    def __init__(self, *, model, config, estimator_class, data_module, metric_type, context_length_choices):
        self.model = model
        self.config = config
        self.estimator_class = estimator_class
        self.data_module = data_module
        self.metric_type = metric_type
        self.evaluator = MultivariateEvaluator(num_workers=None, custom_eval_fn=None)
        self.context_length_choices = context_length_choices
        self.metrics = []
    
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
        self.config["model"].update({k: v for k, v in params.items() if k in self.config["model"]})
        self.config["trainer"].update({k: v for k, v in params.items() if k in self.config["trainer"]})
        estimator = self.estimator_class(
            freq=self.data_module.freq,
            prediction_length=self.data_module_prediction_length,
            context_length=params["context_length"],
            num_feat_dynamic_real=self.data_module.num_feat_dynamic_real, 
            num_feat_static_cat=self.data_module.num_feat_static_cat,
            cardinality=self.data_module.cardinality,
            num_feat_static_real=self.data_module.num_feat_static_real,
            input_size=self.data_module.num_target_vars,
            scaling=False,
            
            batch_size=self.config["dataset"].setdefault("batch_size", 128),
            # num_batches_per_epoch=self.config["trainer"].setdefault("limit_train_batches", 50), # or in params
            train_sampler=ExpectedNumInstanceSampler(num_instances=1.0, min_past=params["context_length"], min_future=self.data_module.prediction_length),
            validation_sampler=ValidationSplitSampler(min_past=params["context_length"], min_future=self.data_module.prediction_length),
            time_features=[second_of_minute, minute_of_hour, hour_of_day, day_of_year],
            distr_output=globals()[self.config["model"]["distr_output"]["class"]](dim=self.data_module.num_target_vars, **self.config["model"]["distr_output"]["kwargs"]),
            trainer_kwargs=self.config["trainer"],
            **self.config["model"][self.model]
        )
        
        predictor = estimator.train(
            training_data=self.data_module.train_dataset,
            validation_data=self.data_module.val_dataset,
            forecast_generator=DistributionForecastGenerator(estimator.distr_output)
        )
        
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=self.data_module.test_dataset, 
            predictor=predictor
        )
        
        forecasts = list(forecast_it)
        tss = list(ts_it)
        # TODO check how this is being computed for a DistributionForecastGenerator
        agg_metrics, _ = self.evaluator(iter(tss), iter(forecasts))
        agg_metrics["trainable_parameters"] = summarize(estimator.create_lightning_module()).trainable_parameters
        self.metrics.append(agg_metrics.copy())
        return agg_metrics[self.metric_type]

def tune_model(model, config, estimator_class, data_module, context_length_choices, metric_type="mean_wQuantileLoss", n_trials=10):
    study = optuna.create_study(direction="minimize")
    study.optimize(MLTuningObjective(model=model, config=config, estimator_class=estimator_class, data_module=data_module, context_length_choices=context_length_choices, metric_type=metric_type), n_trials=n_trials)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))