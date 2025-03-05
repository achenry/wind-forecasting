"""_summary_
This file contains functions to: filter the data: power > 0, inoperation flag, one-sided power curve filtering, stuck sensor, nacelle orientation
Returns:
    _type_: _description_
"""

import logging
from datetime import timedelta
import os

import numpy as np
import polars as pl
import polars.selectors as cs
# from scipy.stats import entropy
# from scipy.spatial.distance import jensenshannon
# from scipy.special import kl_div
from openoa.utils import imputing, filters
mpi_exists = False
try:
    from mpi4py import MPI
    from mpi4py.futures import MPICommExecutor
    mpi_exists = True
except:
    print("No MPI available on system.")
    
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from scipy.optimize import minimize

import matplotlib.pyplot as plt

factor = 1.5
# factor = 3.0 # single column
plt.rc('font', size=12*factor)          # controls default text sizes
plt.rc('axes', titlesize=20*factor)     # fontsize of the axes title
plt.rc('axes', labelsize=15*factor)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=12*factor)    # fontsize of the xtick labels
plt.rc('ytick', labelsize=12*factor)    # fontsize of the ytick labels
plt.rc('legend', fontsize=12*factor)    # legend fontsize
plt.rc('legend', title_fontsize=14*factor)  # legend title fontsize

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataFilter:
    """_summary_
    """
    def __init__(self, turbine_signature, turbine_availability_col=None, turbine_status_col=None, data_format='wide', multiprocessor=None):
        self.turbine_signature = turbine_signature
        self.turbine_availability_col = turbine_availability_col
        self.turbine_status_col = turbine_status_col
        self.data_format = data_format
        self.multiprocessor = multiprocessor
        self.data_format = data_format

    def filter_inoperational(self, df, status_codes=None, availability_codes=None, include_nan=True) -> pl.LazyFrame:
        """
        Filter inoperational turbines based on status and availability codes.
        If the status or availability columns don't exist, skip that part of the filtering.

        Args:
            status_codes (list): List of status codes to include (e.g., [1, 3])
            availability_codes (list): List of availability codes to include (e.g., [100, 50])
            include_nan (bool): Whether to include NaN values in the filter

        Returns:
            pl.LazyFrame: Filtered dataframe
        """
        # Check if the columns actually exist in the dataframe
        cols = df.collect_schema().names()
        
        # Update status and availability flags based on column existence
        if self.turbine_status_col and not any(self.turbine_status_col in col for col in cols):
            logging.info(f"Status column '{self.turbine_status_col}' not found in data, skipping status filtering")
            status_codes = None
            
        if self.turbine_availability_col and not any(self.turbine_availability_col in col for col in cols):
            logging.info(f"Availability column '{self.turbine_availability_col}' not found in data, skipping availability filtering")
            availability_codes = None
        
        # If neither column exists, return the original dataframe
        if status_codes is None and availability_codes is None:
            logging.info("No status or availability columns found, skipping inoperational filtering")
            return df
        
        # Create masks for filtering based on data format
        if self.data_format == 'wide':
            return self._filter_inoperational_wide(df, status_codes, availability_codes, include_nan)
        else:
            return self._filter_inoperational_long(df, status_codes, availability_codes, include_nan)

    def _filter_inoperational_wide(self, df, status_codes, availability_codes, include_nan):
        include_status_mask = status_codes is not None and self.turbine_status_col is not None
        include_availability_mask = availability_codes is not None and self.turbine_availability_col is not None

        if include_status_mask and include_availability_mask:
            return df.with_columns(
                pl.when(cs.starts_with(self.turbine_status_col).is_in(status_codes) 
                        | (cs.starts_with(self.turbine_status_col).is_null() if include_nan else False))
                .then(cs.starts_with(self.turbine_status_col)),
                pl.when(cs.starts_with(self.turbine_availability_col).is_in(availability_codes) 
                        | (cs.starts_with(self.turbine_availability_col).is_null() if include_nan else False))
                .then(cs.starts_with(self.turbine_availability_col))
            )
        elif include_status_mask:
            return df.with_columns(
                pl.when(cs.starts_with(self.turbine_status_col).is_in(status_codes) 
                        | (cs.starts_with(self.turbine_status_col).is_null() if include_nan else False))
                .then(cs.starts_with(self.turbine_status_col))
            )
        elif include_availability_mask:
            return df.with_columns(
                pl.when(cs.starts_with(self.turbine_availability_col).is_in(availability_codes) 
                        | (cs.starts_with(self.turbine_availability_col).is_null() if include_nan else False))
                .then(cs.starts_with(self.turbine_availability_col))
            )
        return df

    def _filter_inoperational_long(self, df, status_codes, availability_codes, include_nan):
        include_status_mask = status_codes is not None and self.turbine_status_col is not None
        include_availability_mask = availability_codes is not None and self.turbine_availability_col is not None

        if include_status_mask and include_availability_mask:
            return df.filter(
                (pl.col(self.turbine_status_col).is_in(status_codes) | (pl.col(self.turbine_status_col).is_null() if include_nan else False))
                & (pl.col(self.turbine_availability_col).is_in(availability_codes) | (pl.col(self.turbine_availability_col).is_null() if include_nan else False))
            )
        elif include_status_mask:
            return df.filter(
                pl.col(self.turbine_status_col).is_in(status_codes) 
                | (pl.col(self.turbine_status_col).is_null() if include_nan else False)
            )
        elif include_availability_mask:
            return df.filter(
                pl.col(self.turbine_availability_col).is_in(availability_codes) 
                | (pl.col(self.turbine_availability_col).is_null() if include_nan else False)
            )
        return df
    
    def _single_generate_window_range_filter(self, df_query, tid, **kwargs):
        mask = filters.window_range_flag(window_col=df_query.select(f"wind_speed_{tid}").collect().to_pandas()[f"wind_speed_{tid}"],
                                         value_col=df_query.select(f"power_output_{tid}").collect().to_pandas()[f"power_output_{tid}"],
                                         **kwargs).values
        mask &= df_query.select(pl.all_horizontal(pl.all().is_not_null())).collect().to_numpy().flatten()
                                                
        logging.info(f"Finished generating out of window filter for {df_query.collect_schema().names()}")
        return mask

    def _single_generate_bin_filter(self, df_query, tid, **kwargs):
        mask = filters.bin_filter(bin_col=f"power_output_{tid}", value_col=f"wind_speed_{tid}", 
                                  data=df_query.select(f"wind_speed_{tid}", f"power_output_{tid}").collect().to_pandas(),
                                  **kwargs).values
        mask &= df_query.select(pl.all_horizontal(pl.all().is_not_null())).collect().to_numpy().flatten()
                                                
        logging.info(f"Finished generating wind speed-power curve bin-outlier filter for {df_query.collect_schema().names()}")
        return mask
    
    def _single_generate_std_range_filter(self, df_query, tid, **kwargs):
        mask = filters.std_range_flag(data_pl=df_query.collect().to_pandas(), **kwargs).values
        mask &= df_query.select(pl.all().is_not_null()).collect().to_numpy()
                                                
        logging.info(f"Finished generating std out of range filter for {df_query.collect_schema().names()}")
        return mask 
    
    def multi_generate_filter(self, df_query, filter_func, feature_types, turbine_ids, **kwargs):
        if self.multiprocessor:
            if self.multiprocessor == "mpi" and mpi_exists:
                executor = MPICommExecutor(MPI.COMM_WORLD, root=0)
                logging.info(f"🚀 Using MPI executor with {MPI.COMM_WORLD.Get_size()} processes")
            else:  # "cf" case
                max_workers = multiprocessing.cpu_count()
                executor = ProcessPoolExecutor(max_workers=max_workers)
                logging.info(f"🖥️  Using ProcessPoolExecutor with {max_workers} workers")
            
            with executor as ex:
                futures = [ex.submit(filter_func, 
                                    df_query=df_query.select([pl.col(f"{feat_type}_{tid}") for feat_type in feature_types]), 
                                    tid=tid, **kwargs) for tid in turbine_ids]
                masks = [fut.result() for fut in futures]
                
                return np.stack(masks, axis=1) 
        else:
            logging.info("🔧 Using single process executor")
            res = []
            for tid in turbine_ids:
                mask = filter_func(
                    df_query=df_query.select([pl.col(f"{feat_type}_{tid}") for feat_type in feature_types]), tid=tid, **kwargs)
                res.append(mask)
                    
            return np.stack(res, axis=1)

    def _single_compute_bias(self, df_query, tid):
        bias = df_query\
                    .filter(pl.col(f"power_output_{tid}") >= 0)\
                    .select("time", f"wind_direction_{tid}", f"nacelle_direction_{tid}", "wd_median", "nd_median")\
                    .select(wd_bias=(pl.col(f"wind_direction_{tid}") - pl.col("wd_median")), 
                            nd_bias=(pl.col(f"nacelle_direction_{tid}") - pl.col("nd_median")))\
                    .select(pl.all().radians().sin().mean().name.suffix("_sin"), pl.all().radians().cos().mean().name.suffix("_cos"))\
                    .select(wd_bias=pl.arctan2("wd_bias_sin", "wd_bias_cos").degrees().mod(360),
                            nd_bias=pl.arctan2("nd_bias_sin", "nd_bias_cos").degrees().mod(360))\
                    .select(pl.when(pl.all() > 180.0).then(pl.all() - 360.0).otherwise(pl.all()))\
                    .select(0.5 * (pl.col("wd_bias").fill_null(0) + pl.col("nd_bias").fill_null(0))).collect().item()

        # df_offsets["turbine_id"].append(turbine_id)
        # bias = 0.5 * ((bias.select("wd_bias").item() or 0) + (bias.select("nd_bias").item() or 0))
         
        logging.info(f"Finished computing wind/nacelle direction bias for turbine {tid}")
        return bias
    
    def multi_compute_bias(self, df_query, turbine_ids):
        if self.multiprocessor:
            if self.multiprocessor == "mpi" and mpi_exists:
                executor = MPICommExecutor(MPI.COMM_WORLD, root=0)
                logging.info(f"🚀 Using MPI executor with {MPI.COMM_WORLD.Get_size()} processes")
            else:  # "cf" case
                max_workers = multiprocessing.cpu_count()
                executor = ProcessPoolExecutor(max_workers=max_workers)
                logging.info(f"🖥️  Using ProcessPoolExecutor with {max_workers} workers")
            
            with executor as ex:
                futures = [ex.submit(self._single_compute_bias, 
                                    df_query=df_query.select(
                    "time", "nd_median", "wd_median", f"power_output_{tid}", f"wind_direction_{tid}", f"nacelle_direction_{tid}"), 
                                    tid=tid) for tid in turbine_ids]
                biases = [fut.result() for fut in futures]
                
                return biases
        else:
            logging.info("🔧 Using single process executor")
            biases = []
            for tid in turbine_ids:
                bias = self._single_compute_bias(
                    df_query=df_query.select(
                        "time", "nd_median", "wd_median", f"power_output_{tid}", f"wind_direction_{tid}", f"nacelle_direction_{tid}"), 
                    tid=tid)
                biases.append(bias)
                    
            return biases
    
    def fill_multi_missing_datasets(self, dfs, impute_missing_features, interpolate_missing_features, r2_threshold):
        if self.multiprocessor:
            if self.multiprocessor == "mpi" and mpi_exists:
                executor = MPICommExecutor(MPI.COMM_WORLD, root=0)
                logging.info(f"🚀 Using MPI executor with {MPI.COMM_WORLD.Get_size()} processes")
            else:  # "cf" case
                max_workers = multiprocessing.cpu_count()
                executor = ProcessPoolExecutor(max_workers=max_workers)
                logging.info(f"🖥️  Using ProcessPoolExecutor with {max_workers} workers")
            
            with executor as ex:
                futures = [ex.submit(self._fill_single_missing_dataset, df_idx=df_idx, df=df, 
                impute_missing_features=impute_missing_features, interpolate_missing_features=interpolate_missing_features,
                parallel="turbine_id", r2_threshold=r2_threshold) 
                for df_idx, df in enumerate(dfs)]
                return [fut.result() for fut in futures if fut.result() is not None]
        else:
            logging.info("🔧 Using single process executor")
            return [self._fill_single_missing_dataset(df_idx=df_idx, df=df, impute_missing_features=impute_missing_features, 
            interpolate_missing_features=interpolate_missing_features, parallel="turbine_id", r2_threshold=r2_threshold) 
            for df_idx, df in enumerate(dfs)]
    
    def _impute_single_missing_dataset(self, df_idx, df, impute_missing_features, r2_threshold, parallel=False):

        if parallel == "feature":
            if self.multiprocessor == "mpi" and mpi_exists:
                executor = MPICommExecutor(MPI.COMM_WORLD, root=0)
                logging.info(f"🚀 Using MPI executor with {MPI.COMM_WORLD.Get_size()} processes")
            else:  # "cf" case
                max_workers = multiprocessing.cpu_count()
                executor = ProcessPoolExecutor(max_workers=max_workers)
                logging.info(f"🖥️  Using ProcessPoolExecutor with {max_workers} workers")
            
            with executor as ex:
                futures = {feature: ex.submit(imputing.impute_all_assets_by_correlation,
                                              data_pl=df.select("time", cs.starts_with(f"{feature}_")), data_pd=None, 
                                    #  data_pd=unpivot_df.select(["time", "turbine_id", feature]).collect().to_pandas().set_index(["time", "turbine_id"]),
                                     impute_col=feature, reference_col=feature,
                                     asset_id_col="turbine_id", method="linear", r2_threshold=r2_threshold) for feature in impute_missing_features}
                
                for k, v in futures.items():
                    df = df.update(v.result(), on="time")
        elif parallel == "turbine_id":
            for feature in impute_missing_features:
                features_pl = ["time", cs.starts_with(f"{feature}_")]

                imputed_vals = imputing.impute_all_assets_by_correlation(
                    # data_pd=unpivot_df.select(features).collect().to_pandas().set_index(["time", "turbine_id"]),
                    data_pl=df.select(features_pl), data_pd=None,
                                                            impute_col=feature, reference_col=feature,
                                                            asset_id_col="turbine_id", method="linear", 
                                                            multiprocessor=self.multiprocessor,
                                                            r2_threshold=r2_threshold)
                
                df = df.update(imputed_vals, on="time")
                logging.info(f"Imputed feature {feature} in DataFrame {df_idx}.")
        else:
            for feature in impute_missing_features:
                features_pl = ["time", cs.starts_with(f"{feature}_")]

                imputed_vals = imputing.impute_all_assets_by_correlation(
                                                            data_pl=df.select(features_pl), data_pd=None,
                                                            impute_col=feature, reference_col=feature,
                                                            asset_id_col="turbine_id", method="linear", multiprocessor=None,
                                                            r2_threshold=r2_threshold)
                
                df = df.update(imputed_vals, on="time")
                logging.info(f"Imputed feature {feature} in DataFrame {df_idx}.") 
        return df

    def _fill_single_missing_dataset(self, df_idx, df, impute_missing_features, interpolate_missing_features, r2_threshold, parallel=None):
        
        df = self._impute_single_missing_dataset(df_idx, df, impute_missing_features, r2_threshold=r2_threshold, parallel=parallel)

        # if any column is all nulls ... can't be imputed
        df = df.with_columns([cs.starts_with(f"{feat}_").interpolate().fill_null(strategy="forward").fill_null(strategy="backward") for feat in interpolate_missing_features])

        if df.select(pl.any_horizontal(pl.all().is_null().sum())).collect().item():
            missing_columns = df.select(col.name for col in 
                                        df.select(cs.numeric().null_count() > 0).collect()
                                        if col.all()).collect_schema().names()
            logging.warning(f"Warning, there are still nulls in dataframe {df_idx} in columns {missing_columns}!")

        return df
    
    @staticmethod
    def _compute_probs(data, n=10): 
        h, e = np.histogram(data, n)
        p = h / data.shape[0]
        return e, p

    @staticmethod
    def _support_intersection(p, q): 
        sup_int = (
            list(
                filter(
                    lambda x: (x[0] != 0) & (x[1] != 0), zip(p, q)
                )
            )
        )
        return sup_int

    @staticmethod
    def _get_probs(list_of_tuples): 
        p = np.array([p[0] for p in list_of_tuples])
        q = np.array([p[1] for p in list_of_tuples])
        return p, q

    @staticmethod
    def _kl_divergence(p, q): 
        return np.sum(p * np.log(p / q))

    @staticmethod
    def _js_divergence(p, q):
        m = (1./2.) * (p + q)
        return (1./2.) * __class__._kl_divergence(p, m) + (1./2.) * __class__._kl_divergence(q, m)

    @staticmethod
    def _compute_kl_divergence(train_sample, test_sample, n_bins=10): 
        """
        Computes the KL Divergence using the support 
        intersection between two different samples
        """
        e, p = __class__._compute_probs(train_sample, n=n_bins)
        _, q = __class__._compute_probs(test_sample, n=e)

        list_of_tuples = __class__._support_intersection(p, q)
        p, q = __class__._get_probs(list_of_tuples)
        
        return __class__._kl_divergence(p, q)

    @staticmethod
    def _compute_js_divergence(train_sample, test_sample, n_bins=100): 
        """
        Computes the JS Divergence using the support 
        intersection between two different samples
        """
        e, p = __class__._compute_probs(train_sample, n=n_bins)
        _, q = __class__._compute_probs(test_sample, n=e)
        
        list_of_tuples = __class__._support_intersection(p, q)
        p, q = __class__._get_probs(list_of_tuples)
        
        return __class__._js_divergence(p, q)

    def conditional_filter(self, df, threshold, mask, mask_input_features, output_features, check_js=True):
        """
        only applies mask to features if the Jensen-Shannon metric between filtered and unfiltered 
        data exceeds a threshold
        """
        if self.data_format == 'wide':
            return self._conditional_filter_wide(df, threshold, mask, mask_input_features, output_features, check_js)
        else:
            return self._conditional_filter_long(df, threshold, mask, mask_input_features, output_features, check_js)

    def _conditional_filter_wide(self, df, threshold, mask, mask_input_features, output_features, check_js):
        if check_js:
            js_scores = []
            for inp_feat, opt_feat in zip(mask_input_features, output_features):
                filt_expr = mask(inp_feat)
                js_score = self._compute_js_divergence(
                    train_sample=df.filter(filt_expr).select(opt_feat).drop_nulls().collect().to_numpy().flatten(),
                    test_sample=df.select(opt_feat).drop_nulls().collect().to_numpy().flatten()
                )
                logging.info(f"JS Score for feature {opt_feat} = {js_score}")
                js_scores.append(js_score)
                
                if js_score > threshold:
                    # new_data = 
                    # df = df.with_columns(**{feat: 
                    #     ma.filled(ma.array(df.select(pl.col(feat)).collect().to_numpy().flatten(), mask=mask(tid), fill_value=np.nan))
                    #                         }).with_columns(pl.col(feat).fill_nan(None).alias(feat))
                    df = df.with_columns(pl.when(pl.Series(filt_expr)).then(None).otherwise(pl.col(opt_feat)).alias(opt_feat))
                
                # if js_score > threshold:
                    # df = df.with_columns(pl.when(mask(tid)).then(pl.col(feat)).otherwise(None).alias(feat))
            # df = df.with_columns({feat: pl.when(mask(feat.split("_")[-1]) & js_score > threshold).then(pl.col(feat)).otherwise(None) for js_score, feat in zip(js_scores, features)})
                    
        else:
            # df = df.with_columns({feat: pl.when(mask(feat.split("_")[-1])).then(pl.col(feat)).otherwise(None) for feat in features})
            for inp_feat, opt_feat in zip(mask_input_features, output_features):
                filt_expr = mask(inp_feat)
                # new_data = ma.filled(ma.array(df.select(pl.col(feat)).collect().to_numpy().flatten(), mask=mask(tid), fill_value=np.nan))
                df = df.with_columns(pl.when(pl.Series(filt_expr)).then(None).otherwise(pl.col(opt_feat)).alias(opt_feat))
                # df = df.with_columns(**{feat: 
                #     ma.filled(ma.array(df.select(pl.col(feat)).collect().to_numpy().flatten(), mask=filt_expr, fill_value=np.nan))
                #     }).with_columns(pl.col(feat).fill_nan(None).alias(feat))
                # df = df.with_columns(pl.when(mask(tid)).then(pl.col(feat)).otherwise(None).alias(feat))
                logging.info(f"Applied filter to feature {opt_feat}.")
        
        return df

    def _conditional_filter_long(self, df, threshold, mask, mask_input_features, output_features, check_js):
        # TODO test this
        if check_js:
            js_scores = []
            for inp_feat, opt_feat in zip(mask_input_features, output_features):
                filt_expr = mask(inp_feat).collect().to_numpy().flatten()
                js_score = self._compute_js_divergence(
                    train_sample=df.filter(pl.Series(filt_expr)).select(opt_feat).drop_nulls().collect().to_numpy().flatten(),
                    test_sample=df.select(opt_feat).drop_nulls().collect().to_numpy().flatten()
                )
                logging.info(f"JS Score for feature {feat} = {js_score}")
                js_scores.append(js_score)
                # if js_score > threshold:
                #     df = df.with_columns(pl.when(mask).then(pl.col(feat)).otherwise(None).alias(feat))
                #     
                
            df = df.with_columns(**{opt_feat: pl.when(mask(inp_feat) & js_score > threshold).then(None).otherwise(pl.col(opt_feat)) for js_score, inp_feat, opt_feat in zip(js_scores, mask_input_features, output_features)})
        else:
            df = df.with_columns(**{opt_feat: pl.when(mask(inp_feat)).then(None).otherwise(pl.col(opt_feat)).alias(opt_feat) for inp_feat, opt_feat in zip(mask_input_features, output_features)})
            
        return df
    
    @staticmethod
    def wrap_180(x):
        """
        Converts an angle or array of angles in degrees to the range -180 to +180 degrees.

        Args:
            x (:obj:`float` or :obj:`numpy.ndarray`): Input angle(s) (degrees)

        Returns:
            :obj:`float` or :obj:`numpy.ndarray`: The input angle(s) converted to the range -180 to +180 degrees (degrees)
        """
        input_type = type(x)
        if isinstance(x, pl.Series):
            x = x.to_numpy()
        x = x % 360.0
        x = np.where(x > 180.0, x - 360.0, x)
        return x if input_type != float else float(x)

    @staticmethod
    def circ_mean(x):
        """
        Computes the circular mean of an angle or array of angles in degrees.

        Args:
            x (:obj:`float` or :obj:`numpy.ndarray`): Input angle(s) (degrees)

        Returns:
            :obj:`float`: The circular mean of the input angle(s) (degrees)
        """
        if isinstance(x, pl.Series):
            x = x.to_numpy()
        y = (np.degrees(np.arctan2(np.nanmean(np.sin(np.radians(x))), np.nanmean(np.cos(np.radians(x))))) % 360.0)
        return y
    
def add_df_continuity_columns(df, mask, dt):
    # change first value of continuous_shifted to false such that add_df_agg_continuity_columns catches it as a start time for a period
    return df\
            .filter(mask)\
            .with_columns(dt=pl.col("time").diff())\
            .with_columns(dt=pl.when(pl.int_range(0, pl.len()) == 0).then(np.timedelta64(dt, "s")).otherwise(pl.col("dt")))\
            .select("time", "dt", cs.starts_with("num_missing"), cs.starts_with("is_missing"))\
            .with_columns(continuous=pl.col("dt")==np.timedelta64(dt, "s"))\
            .with_columns(continuous_shifted=pl.col("continuous").shift(-1, fill_value=True))

def add_df_agg_continuity_columns(df):
    # if the continuous flag is True, but the value in the row before it False
    #.cast(df.collect_schema()["time"]))\
    df = df.filter(pl.col("continuous") | (~pl.col("continuous") & pl.col("continuous_shifted")))
    start_time_cond = ((pl.col("continuous") & ~pl.col("continuous_shifted"))).shift() | (pl.int_range(0, pl.len()) == 0)
    end_time_cond = (~pl.col("continuous") & pl.col("continuous_shifted")) | (pl.int_range(0, pl.len()) == pl.len() - 1) 
    return pl.concat([df.filter(start_time_cond).select(start_time=pl.col("time")), 
                        df.with_columns(end_time=pl.col("time").shift(1)).filter(end_time_cond).select("end_time")], how="horizontal")\
                .with_columns(duration=(pl.col("end_time") - pl.col("start_time")))\
                .sort("start_time")\
            .drop_nulls()

def get_continuity_group_index(df):
    # Create the condition for the group
    group_number = None

    # Create conditions to assign group numbers based on time ranges
    for i, (start, end) in enumerate(df.collect().select("start_time", "end_time").iter_rows()):
        # print(i, start, end, duration)
        time_cond = pl.col("time").is_between(start, end)
        if group_number is None:
            group_number = pl.when(time_cond).then(pl.lit(i))
        else:
            group_number = group_number.when(time_cond).then(pl.lit(i))

    # If no group is matched, assign a default value (e.g., -1) 
    # group_number = group_number.when(pl.col("time") > end).then(pl.lit(i + 1))
    if group_number is None:
        group_number = pl.when(True).then(pl.lit(-1))
    else:
        group_number = group_number.otherwise(pl.lit(-1))
    return group_number

def group_df_by_continuity(df, agg_df, missing_data_cols):
    
    group_number = get_continuity_group_index(agg_df)

    return pl.concat([agg_df, df.with_columns(group_number.alias("continuity_group"))\
            .filter(pl.col("continuity_group") != -1)\
            .group_by("continuity_group")\
            .agg(cs.starts_with("is_missing").sum())\
            .with_columns([pl.sum_horizontal(cs.contains(col) & cs.starts_with("is_missing")).alias(f"is_missing_{col}") for col in missing_data_cols])\
            .sort("continuity_group")], how="horizontal")

def merge_adjacent_periods(agg_df, dt):
    # merge rows with end times corresponding to start times of the next row into the next row, until no more rows need to be merged
    # loop through and merge as long as the shifted -1 end time + dt == the start time
    all_times = agg_df.select(pl.col("start_time"), pl.col("end_time")).collect()
    data = {"start_time":[], "end_time": []}
    start_time_idx = 0
    for end_time_idx in range(all_times.select(pl.len()).item()):
        end_time = all_times.item(end_time_idx, "end_time") 
        if not (end_time_idx + 1 == all_times.select(pl.len()).item()) and (end_time + timedelta(seconds=dt)  == all_times.item(end_time_idx + 1, "start_time")):
            continue
        
        data["start_time"].append(all_times.item(start_time_idx, "start_time"))
        data["end_time"].append(end_time)

        start_time_idx = end_time_idx + 1

    return pl.LazyFrame(data, schema={
        "start_time": pl.Datetime(time_unit=agg_df.collect_schema()["start_time"].time_unit), 
        "end_time": pl.Datetime(time_unit=agg_df.collect_schema()["end_time"].time_unit)})\
             .with_columns((pl.col("end_time") - pl.col("start_time")).alias("duration"))

# Optimization function for finding waked direction
def gauss_corr(gauss_params, power_ratio):
    xs = np.arange(-int((len(power_ratio) - 1) / 2), int((len(power_ratio) + 1) / 2), 1)
    gauss = -1 * gauss_params[2] * np.exp(-0.5 * ((xs - gauss_params[0]) / gauss_params[1])**2) + 1.
    # maximize the correlation between the gaussian curve parameterized here and the power_ratio i.e. fit the gaussian curve to the power ratio
    return -1 * np.corrcoef(gauss, power_ratio)[0, 1] 

def compute_offsets(df, fi, turbine_ids, turbine_pairs:list[tuple[int, int]]=None, plot=False, save_path=None):
    p_min = 100
    p_max = 2500
    prat_hfwdth = 30
    # zero indexed
    prat_turbine_pairs = turbine_pairs 

    dir_offsets = []

    for i in range(len(prat_turbine_pairs)):
        i_up = prat_turbine_pairs[i][0]
        i_down = prat_turbine_pairs[i][1]

        # compute the angle of the vector pointing from the downstream turbine to the upstream turbine (CW from north)
        dir_align = np.degrees(np.arctan2(fi.layout_x[i_up] - fi.layout_x[i_down], fi.layout_y[i_up] - fi.layout_y[i_down])) % 360

        tid_up = turbine_ids[i_up]
        tid_down = turbine_ids[i_down]

        # get the subset of power outputs and wind directions for which the downstream power is positive and the upstream power is within a given range
        df_sub = df.filter((pl.col(f"power_output_{tid_up}") >= p_min) 
                                & (pl.col(f"power_output_{tid_up}") <= p_max) 
                                & (pl.col(f"power_output_{tid_down}") >= 0))\
                                .select(f"power_output_{tid_up}", f"power_output_{tid_down}", f"wind_direction_{tid_up}", f"wind_direction_{tid_down}")
                                
        # average the turbine powers by the upstream wind direction nearest integer
        df_sub = df_sub.with_columns(pl.when((pl.col(f"wind_direction_{tid_up}") >= 359.5))\
                                        .then(pl.col(f"wind_direction_{tid_up}") - 360.0)\
                                        .otherwise(pl.col(f"wind_direction_{tid_up}")),
                                        pl.col(f"wind_direction_{tid_up}").round().alias(f"wd_round"))\
                        .group_by(f"wd_round").agg(pl.all().mean()).sort("wd_round")

        # compute the power ratio downstream power to upstream power for each integer wind direction
        df_sub = df_sub.select(pl.col(f"wd_round"), (pl.col(f"power_output_{tid_down}") / pl.col(f"power_output_{tid_up}")).alias("p_ratio")).collect()

        if plot or True:
            fig, ax = plt.subplots(1,1, figsize=(10, 6))
            ax.plot(df_sub.select("wd_round").to_numpy().flatten(), df_sub.select("p_ratio").to_numpy().flatten(), label="_nolegend_")
            ax.plot(dir_align * np.ones(2),[0, 1.25], 'k--', label="Direction of Alignment")
            ax.grid()
            # ax.set_title(f"Turbine Pair: ({turbine_ids[i_up]}, {turbine_ids[i_down]})")
            ax.set_xlabel("Wind Direction [$^\\circ$]")
            ax.set_ylabel("Power Ratio [-]")

        # get the range of wind directions around that of alignment (according to turbine layout), when the nadir should occur
        wd_range = np.arange(int(np.round(dir_align)) - prat_hfwdth,int(np.round(dir_align)) + prat_hfwdth + 1) % 360
        if len(set(wd_range) & set(df_sub.select(f"wd_round").to_numpy().flatten())) != len(wd_range):
            logging.info(f"Cannot compute nadir for turbine pair {turbine_ids[i_up]}, {turbine_ids[i_down]}")
            continue
        
        # get the power ratios that occur in the range around the aligned wind direction, 
        # get the rounded wind direction corresponding to the power nadir, then add the direction of alignment and subtract the prat halfwidth
        nadir = df_sub.filter(pl.col("wd_round").is_in(wd_range)) \
                        .filter(pl.col("p_ratio") == pl.col("p_ratio").min()) \
                        .select(pl.col("wd_round")).item()
        
        wd_range = np.arange(nadir - prat_hfwdth, nadir + prat_hfwdth + 1) % 360
        if len(set(wd_range) & set(df_sub.select(f"wd_round").to_numpy().flatten())) != len(wd_range):
            logging.info(f"Cannot compute nadir for turbine pair {turbine_ids[i_up]}, {turbine_ids[i_down]}")
            continue
        
        # get parameters of gaussian trough that fits power ratio for wind direction approx perpendicular to dir_align TODO?
        opt_gauss_params = minimize(gauss_corr, [0, 5.0, 1.0], 
                                    args=(df_sub.filter(pl.col("wd_round").is_in(wd_range))\
                                               .select("p_ratio").to_numpy().flatten()), method='SLSQP')

        # range around -/+ 30 degrees
        xs = np.arange(-int((2*prat_hfwdth - 1) / 2),int((2*prat_hfwdth + 1) / 2),1)
        gauss = -1 * opt_gauss_params.x[2] * np.exp(-0.5 * ((xs - opt_gauss_params.x[0]) / opt_gauss_params.x[1])**2) + 1.

        if plot or True:
            ax.plot(xs + nadir, gauss,'k',label="_nolegend_")
            ax.plot(2 * [nadir + opt_gauss_params.x[0]], [0, 1.25], 'r--',label="Direction of Measured Power Nadir")
            ax.legend()
            fig.savefig(save_path.replace(".png", f"_{i_up}_{i_down}.png"), dpi=100)
            plt.close() 
        
        dir_offset = DataFilter.wrap_180(nadir + opt_gauss_params.x[0] - dir_align)
        print(f"Direction offset for turbine pair ({tid_up}, {tid_down}) = {dir_offset}")

        dir_offsets.append(dir_offset)

    if dir_offsets:
        print(f"Mean offset = {np.mean(dir_offsets)}")
        print(f"Std. Dev. = {np.std(dir_offsets)}")
        print(f"Min. = {np.min(dir_offsets)}")
        print(f"Max. = {np.max(dir_offsets)}")
        
    else:
        print("No available turbine pairs!")
    return dir_offsets

# Define the mask function using the new mapping
def safe_mask(tid, outlier_flag, turbine_id_to_index, flag_format="numpy"):
    try:
        if flag_format == "numpy":
            idx = turbine_id_to_index[tid]
            mask_array = outlier_flag[:, idx]
        else:
            mask_array = outlier_flag.select(cs.ends_with(f"_{tid}"))
        # logging.info(f"Mask for turbine {tid} includes {np.sum(~mask_array)} out of {len(mask_array)} data points")
        return mask_array
    except KeyError:
        logging.error(f"Mask error for turbine {tid}: turbine ID not found in mapping")
        return None