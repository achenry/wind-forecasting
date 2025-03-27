import os
from collections import defaultdict
import logging
from memory_profiler import profile
from glob import glob
import re
from torch import load as torch_load
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from gluonts.torch.distributions import LowRankMultivariateNormalOutput
from gluonts.model.forecast_generator import DistributionForecastGenerator
from gluonts.evaluation import MultivariateEvaluator, make_evaluation_predictions

from pytorch_transformer_ts.informer.lightning_module import InformerLightningModule
from pytorch_transformer_ts.informer.estimator import InformerEstimator
from pytorch_transformer_ts.autoformer.estimator import AutoformerEstimator
from pytorch_transformer_ts.autoformer.lightning_module import AutoformerLightningModule
from pytorch_transformer_ts.spacetimeformer.estimator import SpacetimeformerEstimator
from pytorch_transformer_ts.spacetimeformer.lightning_module import SpacetimeformerLightningModule
from wind_forecasting.preprocessing.data_module import DataModule
from gluonts.time_feature._base import second_of_minute, minute_of_hour, hour_of_day, day_of_year
from gluonts.transform import ExpectedNumInstanceSampler, ValidationSplitSampler, SequentialSampler
from wind_forecasting.postprocessing.probabilistic_metrics import continuous_ranked_probability_score, reliability, resolution, uncertainty, sharpness, pi_coverage_probability, pi_normalized_average_width, coverage_width_criterion 

# Configure logging and matplotlib backend
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# if sys.platform == "darwin":
#     matplotlib.use('TkAgg')
    
mpi_exists = False
try:
    from mpi4py import MPI
    mpi_exists = True
except:
    print("No MPI available on system.")

# @profile
def test_model(*, data_module, checkpoint, lightning_module_class, normalization_consts_path, estimator):
    # TODO denormalize at end
    normalization_consts = pd.read_csv(normalization_consts_path, index_col=None)
    if os.path.exists(checkpoint):
        logging.info("Found pretrained model, loading...")
        model = lightning_module_class.load_from_checkpoint(checkpoint)
        transformation = estimator.create_transformation(use_lazyframe=False)
        predictor = estimator.create_predictor(transformation, model, 
                                                forecast_generator=DistributionForecastGenerator(estimator.distr_output))
    else:
        raise TypeError("Must provide a --checkpoint argument to load from.")

    logging.info("Making evaluation predictions") 
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=data_module.test_dataset,
        predictor=predictor,
        output_distr_params=True
    )

    forecasts = list(forecast_it)
    tss = list(ts_it)
    
    # %%
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

def get_checkpoint(checkpoint, metric, mode, log_dir):
    if checkpoint in ["best", "latest"]:
        checkpoint_paths = glob(os.path.join(log_dir, "*/*/*.ckpt"))
        # version_dirs = glob(os.path.join(log_dir, "*"))
        if len(checkpoint_paths) == 0:
            raise FileNotFoundError(f"There are not checkpoint files in {log_dir}.")
        
    elif not os.path.exists(checkpoint):
        raise FileNotFoundError("Must provide a valid --checkpoint argument to load from.")
        
    if checkpoint == "best":
        best_metric_value = float('inf') if mode == "min" else float('-inf')
        best_checkpoint_path = None
        for checkpoint_path in checkpoint_paths:
            checkpoint = torch_load(checkpoint_path, weights_only=False)
            mc_callback = [cb_vals for cb_key, cb_vals in checkpoint["callbacks"].items() if "ModelCheckpoint" in cb_key][0]
            if (mode == "min" and mc_callback["best_model_score"] < best_metric_value) or (mode == "max" and mc_callback["best_model_score"] > best_metric_value):
                best_metric_value = mc_callback["best_model_score"]
                best_checkpoint_path = mc_callback["best_model_path"]
            
        if best_checkpoint_path and os.path.exists(best_checkpoint_path):
            logging.info(f"Found best pretrained model: {best_checkpoint_path}")
        else:
            raise FileNotFoundError(f"Best checkpoint {best_checkpoint_path} does not exist.")
        
        return best_checkpoint_path
    elif checkpoint == "latest":
        logging.info("Fetching latest pretrained model...")
        
        latest_checkpoint_path = sorted(checkpoint_paths, key=lambda chk_path: datetime.fromtimestamp(os.stat(chk_path).st_mtime))[-1]
        
        if os.path.exists(latest_checkpoint_path):
            logging.info(f"Found latest pretrained model: {latest_checkpoint_path}")
        else:
            raise FileNotFoundError(f"Latest checkpoint {latest_checkpoint_path} does not exist.")
        return latest_checkpoint_path
        
    else:
        logging.info("Fetching pretrained model...")
        if os.path.exists(checkpoint):
            logging.info(f"Found given pretrained model: {checkpoint_path}")
            checkpoint_path = checkpoint
        else:
            raise FileNotFoundError(f"Given checkpoint {checkpoint} does not exist.")
        return checkpoint_path