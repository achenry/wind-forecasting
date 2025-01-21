import os
import sys
import argparse
from collections import defaultdict
import logging
import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import colormaps, dates as mdates
import wandb
import yaml

from gluonts.torch.distributions import LowRankMultivariateNormalOutput
from gluonts.model.forecast_generator import DistributionForecastGenerator
from gluonts.evaluation import MultivariateEvaluator, make_evaluation_predictions
from gluonts.time_feature._base import second_of_minute, minute_of_hour, hour_of_day, day_of_year
from gluonts.transform import ExpectedNumInstanceSampler, SequentialSampler

from lightning.pytorch.loggers import WandbLogger
from pytorch_transformer_ts.informer.lightning_module import InformerLightningModule
from pytorch_transformer_ts.informer.estimator import InformerEstimator
from pytorch_transformer_ts.autoformer.estimator import AutoformerEstimator
from pytorch_transformer_ts.autoformer.lightning_module import AutoformerLightningModule
from pytorch_transformer_ts.spacetimeformer.estimator import SpacetimeformerEstimator
from pytorch_transformer_ts.spacetimeformer.lightning_module import SpacetimeformerLightningModule
from wind_forecasting.preprocessing.data_module import DataModule
from wind_forecasting.postprocessing.probabilistic_metrics import continuous_ranked_probability_score, reliability, resolution, uncertainty, sharpness, pi_coverage_probability, pi_normalized_average_width, coverage_width_criterion 

# Configure logging and matplotlib backend
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if sys.platform == "darwin":
    matplotlib.use('TkAgg')
    
mpi_exists = False
try:
    from mpi4py import MPI
    mpi_exists = True
except:
    print("No MPI available on system.")


if __name__ == "__main__":
    
    RUN_ONCE = (mpi_exists and (MPI.COMM_WORLD.Get_rank()) == 0)
    
    # %% PARSE CONFIGURATION
    # parse training/test booleans and config file from command line
    logging.info("Parsing configuration from yaml and command line arguments")
    parser = argparse.ArgumentParser(prog="WindFarmForecasting")
    parser.add_argument("-cnf", "--config", type=str)
    parser.add_argument("-tr", "--train", action="store_true")
    parser.add_argument("-te", "--test", action="store_true")
    parser.add_argument("-chk", "--checkpoint", type=str, default="")
    parser.add_argument("-m", "--model", type=str, choices=["informer", "autoformer", "spacetimeformer", "tactis"], required=True)
    # pretrained_filename = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/logging/wf_forecasting/lznjshyo/checkpoints/epoch=0-step=50.ckpt"
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config  = yaml.safe_load(file)
        
    # TODO set number of devices/number of nodes based on environment variables

    # TODO create function to check config params and set defaults
    # if config["trainer"]["n_workers"] == "auto":
    #     if "SLURM_GPUS_ON_NODE" in os.environ:
    #         config["trainer"]["n_workers"] = int(os.environ["SLURM_GPUS_ON_NODE"])
    #     else:
    #         config["trainer"]["n_workers"] = mp.cpu_count()
    
    # config["trainer"]["devices"] = 'auto'
    # config["trainer"]["accelerator"] = 'auto'
    
    if (type(config["dataset"]["target_turbine_ids"]) is str) and (
        (config["dataset"]["target_turbine_ids"].lower() == "none") or (config["dataset"]["target_turbine_ids"].lower() == "all")):
        config["dataset"]["target_turbine_ids"] = None # select all turbines

    # %% SETUP LOGGING
    # logging.info("Setting up logging")
    # if not os.path.exists(config["experiment"]["log_dir"]):
    #     os.makedirs(config["experiment"]["log_dir"])
    # wandb_logger = WandbLogger(
    #     project="wf_forecasting",
    #     name=config["experiment"]["run_name"],
    #     log_model=True,
    #     save_dir=config["experiment"]["log_dir"],
    #     config=config
    # )
    # config["trainer"]["logger"] = wandb_logger

    # %% CREATE DATASET
    logging.info("Creating datasets")
    data_module = DataModule(data_path=config["dataset"]["data_path"], n_splits=config["dataset"]["n_splits"],
                            continuity_groups=None, train_split=(1.0 - config["dataset"]["val_split"] - config["dataset"]["test_split"]),
                                val_split=config["dataset"]["val_split"], test_split=config["dataset"]["test_split"], 
                                prediction_length=config["dataset"]["prediction_length"], context_length=config["dataset"]["context_length"],
                                target_prefixes=["ws_horz", "ws_vert"], feat_dynamic_real_prefixes=["nd_cos", "nd_sin"],
                                freq=config["dataset"]["resample_freq"], target_suffixes=config["dataset"]["target_turbine_ids"],
                                per_turbine_target=config["dataset"]["per_turbine_target"])
    # if RUN_ONCE:
    data_module.generate_datasets()
        # data_module.plot_dataset_splitting()

    # %% DEFINE ESTIMATOR
    logging.info("Declaring estimator")
    estimator = globals()[f"{args.model.capitalize()}Estimator"](
        freq=data_module.freq, 
        prediction_length=data_module.prediction_length,
        context_length=data_module.context_length,
        num_feat_dynamic_real=data_module.num_feat_dynamic_real, 
        num_feat_static_cat=data_module.num_feat_static_cat,
        cardinality=data_module.cardinality,
        num_feat_static_real=data_module.num_feat_static_real,
        input_size=data_module.num_target_vars,
        scaling=False,
        lags_seq=[0, 1],
        batch_size=128,
        num_batches_per_epoch=config["trainer"].setdefault("limit_train_batches", 50), # TODO set this to be arbitrarily high st limit train_batches dominates
        # train_sampler=SequentialSampler(min_past=data_module.context_length, min_future=data_module.prediction_length), # TODO SequentialSampler = terrible results
        # validation_sampler=SequentialSampler(min_past=data_module.context_length, min_future=data_module.prediction_length),
        
        # validation_sampler=ExpectedNumInstanceSampler(num_instances=1.0, min_future=data_module.prediction_length),
        # dim_feedforward=config["model"][args.model]["dim_feedforward"],
        # d_model=config["model"][args.model]["d_model"],
        # num_encoder_layers=config["model"][args.model]["num_layers"],
        # num_decoder_layers=config["model"][args.model]["num_layers"],
        # n_heads=config["model"]["num_heads"],
        activation="relu",
        time_features=[second_of_minute, minute_of_hour, hour_of_day, day_of_year],
        distr_output=LowRankMultivariateNormalOutput(dim=data_module.num_target_vars, rank=8),
        trainer_kwargs=config["trainer"],
        **config["model"][args.model]
    )

    # %% TRAIN MODEL
    if args.train:
        # TODO add possibilty to add checkpoint here
        if RUN_ONCE:
            logging.info("Training model")
        predictor = estimator.train(
            training_data=data_module.train_dataset,
            validation_data=data_module.val_dataset,
            forecast_generator=DistributionForecastGenerator(estimator.distr_output)
            # ckpt_path=config["trainer"]
            # shuffle_buffer_length=1024
        )
    
    # %% TEST MODEL
    # set forecast_generator to DistributionForecastGenerator to access mean and variance in InformerEstimator.create_predictor
    if RUN_ONCE:
        if args.test:
            normalization_consts = pd.read_csv(config["dataset"]["normalization_consts_path"], index_col=None)
            if not args.train and os.path.exists(args.checkpoint):
                logging.info("Found pretrained model, loading...")
                model = globals()[f"{args.model.capitalize()}LightningModule"].load_from_checkpoint(args.checkpoint)
                transformation = estimator.create_transformation()
                predictor = estimator.create_predictor(transformation, model, 
                                                        forecast_generator=DistributionForecastGenerator(estimator.distr_output))
            elif not args.train:
                raise TypeError("Must train model with --train flag or provide a --checkpoint argument to load from.")

            logging.info("Making evaluation predictions") 
            forecast_it, ts_it = make_evaluation_predictions(
                dataset=data_module.test_dataset,
                predictor=predictor,
                output_distr_params=True
            )

            forecasts = list(forecast_it)
            tss = list(ts_it)
            
            # %%
            # num_workers is limited to 10 if cpu has more cores
            num_workers = min(mp.cpu_count(), 10)
            # TODO add custom evaluation functions eg Continuous Ranked Probability Score, Quantile Loss, Pinball Loss/Quantile Score (same as Quantile Loss?), 
            # Energy Score, PI Coverage Probability (PICP), PI Normalized Average Width (PINAW), Coverage Width Criterion (CWC), Winkler/IntervalScore(IS)
            
            custom_eval_fn={
                        "PICP": (pi_coverage_probability, "mean", "mean"),
                        "PINAW": (pi_normalized_average_width, "mean", "mean"),
                        "CWC": (coverage_width_criterion, "mean", "mean"),
                        "CRPS": (continuous_ranked_probability_score, "mean", "mean"),
            }
            evaluator = MultivariateEvaluator(num_workers=None, 
                custom_eval_fn=None
            )

            # %% COMPUTE AGGREGATE METRICS
            agg_metrics, ts_metrics = evaluator(iter(tss), iter(forecasts), num_series=data_module.num_target_vars)

            # %% PLOT TEST PREDICTIONS
            agg_df = defaultdict(list)
            # for t, target in enumerate(target_cols):
            for k, v in agg_metrics.items():
                if "_" in k and k.split("_")[0].isdigit():
                    target_idx = int(k.split("_")[0])
                    perf_metric = k.split('_')[1]
                    if data_module.per_turbine_target:
                        target_metric = data_module.target_cols[target_idx]
                        print(f"Performance metric {perf_metric} for target {target_metric} = {v}")
                    else:
                        turbine_id = data_module.target_cols[target_idx].split("_")[-1]
                        target_metric = "_".join(data_module.target_cols[target_idx].split("_")[:-1])
                        print(f"Performance metric {perf_metric} for target {target_metric} and turbine {turbine_id} = {v}")
                        agg_df["turbine_id"].append(turbine_id)
                    
                    agg_df["target_metric"].append(target_metric)
                    agg_df["perf_metric"].append(perf_metric)
                    agg_df["values"].append(v)

            agg_df = pd.DataFrame(agg_df)
            agg_df = pd.pivot(agg_df, columns="perf_metric") #, values="values", index=["target_metric", "turbine_id"])

            # %%
            # forecasts[0].distribution.loc.cpu().numpy()
            # forecasts[0].distribution.cov_diag.cpu().numpy()
            num_forecasts = 4
            fig, axs = plt.subplots(min(len(forecasts), num_forecasts), len(data_module.target_prefixes), figsize=(6, 12))
            axs = axs.flatten()
            def errorbar(vec):
                print(vec)
                return vec["loc"] - 3 * vec["std_dev"], vec["loc"] + 3 * vec["std_dev"]
            # axx = axs.ravel()
            seq_len, target_dim = tss[0].shape
            ax_idx = 0
            for idx, (forecast, ts) in enumerate(zip(forecasts, tss)):
                if idx == num_forecasts:
                    break
                # for dim in range(min(len(axs), target_dim)):
                for o, output_type in enumerate(data_module.target_prefixes):
                    ax = axs[ax_idx]
                    
                    # TODO change plotting for perturbine case
                    col_idx = [c for c, col in enumerate(data_module.target_cols) if output_type in col]
                    col_names = [col for col in data_module.target_cols if output_type in col]
                    true_df = ts[-data_module.context_length - data_module.prediction_length:][col_idx]\
                                    .rename(columns={c: cname for c, cname in zip(col_idx, col_names)})
                    
                    if data_module.per_turbine_target:
                        true_df = true_df.assign(turbine_id=pd.Categorical([data_module.static_features.iloc[idx]["turbine_id"]
                                                                for t in range(data_module.context_length + data_module.prediction_length)]))
                        # pred_turbine_id = pd.Categorical([data_module.static_features.iloc[idx]["turbine_id"]
                        #                                         for t in range(data_module.prediction_length)])
                    else:
                        true_df = pd.concat([
                            true_df[col].to_frame()\
                                        .rename(columns={col: output_type})\
                                        .assign(turbine_id=pd.Categorical([col.split("_")[-1] 
                                                                for t in range(data_module.context_length + data_module.prediction_length)])) 
                                                                for col in col_names], axis=0)
                        pred_turbine_id = pd.Categorical([col.split("_")[-1] for col in col_names for t in range(data_module.prediction_length)])
                    
                    true_df = true_df.reset_index(names="time").sort_values(["turbine_id", "time"])
                    true_df["time"] = true_df["time"].dt.to_timestamp()
                    sns.lineplot(data=true_df, x="time", y=output_type, hue="turbine_id", ax=ax)
                    # .plot(ax=ax)

                    # (quantile, target_dim, seq_len)
                    # pred_df = pd.DataFrame(
                    #     {q: forecasts[0].quantile(q)[dim] for q in [0.1, 0.5, 0.9]},
                    #     index=forecasts[0].index,
                    # )

                    # (n_stds, target_dim, seq_len)
                    if data_module.per_turbine_target:
                        pred_df = pd.DataFrame(
                            {
                                "loc": forecast.distribution.loc[:, col_idx].transpose(0, 1).reshape(-1, 1).cpu().numpy().flatten(),
                                "std_dev": np.sqrt(forecast.distribution.cov_diag[:, col_idx].transpose(0, 1).reshape(-1, 1).cpu().numpy()).flatten()
                            },
                            index=np.tile(forecast.index, (len(col_names),)),
                        ).reset_index(names="time").sort_values(["time"])
                    else:
                        pred_df = pd.DataFrame(
                            {
                                "turbine_id": pred_turbine_id,
                                "loc": forecast.distribution.loc[:, col_idx].transpose(0, 1).reshape(-1, 1).cpu().numpy().flatten(),
                                "std_dev": np.sqrt(forecast.distribution.cov_diag[:, col_idx].transpose(0, 1).reshape(-1, 1).cpu().numpy()).flatten()
                            },
                            index=np.tile(forecast.index, (len(col_names),)),
                        ).reset_index(names="time").sort_values(["turbine_id", "time"])
                    
                    pred_df["time"] = pred_df["time"].dt.to_timestamp()

                    if data_module.per_turbine_target:
                        sns.lineplot(data=pred_df, x="time", y="loc", ax=ax, linestyle="dashed")
                        ax.fill_between(
                            forecast.index.to_timestamp(), pred_df["loc"] - 1*pred_df["std_dev"], pred_df["loc"] + 1*pred_df["std_dev"], alpha=0.2, 
                        )
                    else:
                        sns.lineplot(data=pred_df, x="time", y="loc", hue="turbine_id", ax=ax, linestyle="dashed")
                        for t, tid in enumerate(pd.unique(pred_df["turbine_id"])):
                            # color = loc_ax.get_lines()[t].get_color()
                            tid_df = pred_df.loc[pred_df["turbine_id"] == tid, :]
                            color = sns.color_palette()[t]
                            ax.fill_between(
                                forecast.index.to_timestamp(), tid_df["loc"] - 1*tid_df["std_dev"], tid_df["loc"] + 1*tid_df["std_dev"], alpha=0.2, color=color
                            )

                    # pred_df["loc"].plot(ax=ax, color='g')
                    ax.legend([], [], frameon=False)
                    ax_idx += 1
            h, l = axs[0].get_legend_handles_labels()
            axs[0].legend(h[:len(data_module.target_suffixes)], l[:len(data_module.target_suffixes)])
            plt.show()
            
            print("here")