"""_summary_
This file contains functions to: filter the data: power > 0, inoperation flag, one-sided power curve filtering, stuck sensor, nacelle orientation
Returns:
    _type_: _description_
"""

import logging

import numpy as np
import polars as pl
import polars.selectors as cs
from scipy.interpolate import CubicSpline
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from scipy.special import kl_div
from openoa.utils import imputing

from data_inspector import DataInspector

from mpi4py import MPI
from mpi4py.futures import MPICommExecutor
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# from line_profiler import profile
# from memory_profiler import profile

class DataFilter:
    """_summary_
    """
    def __init__(self, turbine_availability_col=None, turbine_status_col=None, data_format='wide', multiprocessor=None):
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

    def fill_multi_missing_datasets(self, dfs, impute_missing_features, interpolate_missing_features, available_features):
        if self.multiprocessor:
            if self.multiprocessor == "mpi":
                executor = MPICommExecutor(MPI.COMM_WORLD, root=0)
                logging.info(f"🚀 Using MPI executor with {MPI.COMM_WORLD.Get_size()} processes")
            else:  # "cf" case
                max_workers = multiprocessing.cpu_count()
                executor = ProcessPoolExecutor(max_workers=max_workers)
                logging.info(f"🖥️  Using ProcessPoolExecutor with {max_workers} workers")
            
            with executor as ex:
                futures = [ex.submit(self._fill_single_missing_dataset, df_idx=df_idx, df=df, 
                impute_missing_features=impute_missing_features, interpolate_missing_features=interpolate_missing_features,
                 available_features=available_features, parallel="turbine_id") 
                for df_idx, df in enumerate(dfs)]
                return [fut.result() for fut in futures if fut.result() is not None]
        else:
            logging.info("🔧 Using single process executor")
            return [self._fill_single_missing_dataset(df_idx=df_idx, df=df, impute_missing_features=impute_missing_features, 
            interpolate_missing_features=interpolate_missing_features, available_features=available_features, parallel="turbine_id") 
            for df_idx, df in enumerate(dfs)]
    
    def _impute_single_missing_dataset(self, df_idx, df, impute_missing_features, parallel=False):
        # unpivot_df = DataInspector.unpivot_dataframe(df, impute_missing_features)

        if parallel == "feature":
            if self.multiprocessor == "mpi":
                executor = MPICommExecutor(MPI.COMM_WORLD, root=0)
                logging.info(f"🚀 Using MPI executor with {MPI.COMM_WORLD.Get_size()} processes")
            else:  # "cf" case
                max_workers = multiprocessing.cpu_count()
                executor = ProcessPoolExecutor(max_workers=max_workers)
                logging.info(f"🖥️  Using ProcessPoolExecutor with {max_workers} workers")
            
            with executor as ex:
                futures = {feature: ex.submit(imputing.impute_all_assets_by_correlation,
                                              data_pl=df.select("time", cs.starts_with(feature)), data_pd=None, 
                                    #  data_pd=unpivot_df.select(["time", "turbine_id", feature]).collect().to_pandas().set_index(["time", "turbine_id"]),
                                     impute_col=feature, reference_col=feature,
                                     asset_id_col="turbine_id", method="linear") for feature in impute_missing_features}
                
                for k, v in futures.items():
                    df = df.update(v, on="time")
                # unpivot_df = unpivot_df.with_columns({k: v.result() for k, v in futures.items()}).fill_nan(None)
        elif parallel == "turbine_id":
            for feature in impute_missing_features:
                # n_nulls_before = unpivot_df.select(cs.contains(feature)).select(pl.sum_horizontal(pl.all().is_null()).sum()).collect().item()
                # print(f"# Missing values before imputation = {n_nulls_before}")
                
                # other_feature = feature
                # features_pd = set(["time", "turbine_id", feature, other_feature])
                # features_pl = ["time"] + [cs.starts_with(feat) for feat in set([feature, other_feature])]
                features_pl = ["time", cs.starts_with(feature)]

                imputed_vals = imputing.impute_all_assets_by_correlation(
                    # data_pd=unpivot_df.select(features).collect().to_pandas().set_index(["time", "turbine_id"]),
                    data_pl=df.select(features_pl), data_pd=None,
                                                            impute_col=feature, reference_col=feature,
                                                            asset_id_col="turbine_id", method="linear", 
                                                            multiprocessor=self.multiprocessor)
                
                df = df.update(imputed_vals, on="time")
                # n_nulls_after = unpivot_df.select(cs.contains(feature)).select(pl.sum_horizontal(pl.all().is_null()).sum()).collect().item()
                # print(f"# Missing values after imputation = {n_nulls_after}")
                logging.info(f"Imputed feature {feature} in DataFrame {df_idx}.")
                # logging.info(f"Successfully imputed {n_nulls_before - n_nulls_after} cells for feature {feature} in DataFrame {df_idx}.")
        else:
            for feature in impute_missing_features:
                # n_nulls_before = unpivot_df.select(cs.contains(feature)).select(pl.sum_horizontal(pl.all().is_null()).sum()).collect().item()
                # print(f"# Missing values before imputation = {n_nulls_before}")
                
                # other_feature = feature
                # features = set(["time", "turbine_id", feature, other_feature])
                features_pl = ["time", cs.starts_with(feature)]

                imputed_vals = imputing.impute_all_assets_by_correlation(
                                                            data_pl=df.select(features_pl), data_pd=None,
                                                            impute_col=feature, reference_col=feature,
                                                            asset_id_col="turbine_id", method="linear", multiprocessor=None)
                
                df = df.update(imputed_vals, on="time")
                # n_nulls_after = unpivot_df.select(cs.contains(feature)).select(pl.sum_horizontal(pl.all().is_null()).sum()).collect().item()
                # print(f"# Missing values after imputation = {n_nulls_after}")
                logging.info(f"Imputed feature {feature} in DataFrame {df_idx}.") 
        return df 

    def _fill_single_missing_dataset(self, df_idx, df, impute_missing_features, interpolate_missing_features, available_features, parallel=None):
        
        df = self._impute_single_missing_dataset(df_idx, df, impute_missing_features, parallel=parallel)

        # if any column is all nulls ... can't be imputed
        df = df.with_columns([cs.starts_with(feat).interpolate().fill_null(strategy="forward").fill_null(strategy="backward") for feat in interpolate_missing_features])

        if df.select(pl.any_horizontal(pl.all().is_null().sum())).collect().item():
            raise Exception(f"Error, there are still nulls in dataframe {df_idx}!")

        return df

    def resolve_missing_data(self, df, how="linear_interp", features=None) -> pl.LazyFrame:
        """_summary_
        option 1) interpolate via linear, or forward
        """
        if how == "forward_fill":
            return df.fill_null(strategy="forward")
        elif how == "linear_interp":
            return df.with_columns(pl.col(features).interpolate())
        # return df.with_columns(pl.col(features).map_batches(partial(self._interpolate_series, df=df, how=how)))\
        #          .fill_nan(None)
            
    def _interpolate_series(self, ser, df, how):
        """_summary_

        Args:
            ser (_type_): _description_
            how (_type_): _description_

        Returns:
            _type_: _description_
        """
        xp = df["time"].filter(ser.is_not_null())
        fp = ser.filter(ser.is_not_null())
        x = df["time"]

        if how == "forward_fill":
            return ser.fill_null(strategy="forward")

        if how == "linear_interp":
            return np.interp(x, xp, fp, left=None, right=None)
        
        if how == "cubic_interp":
            return CubicSpline(xp, fp, extrapolate=False)(x)
    
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

    def conditional_filter(self, df, threshold, mask, features):
        """
        only applies mask to features if the Jensen-Shannon metric between filtered and unfiltered 
        data exceeds a threshold
        """
        if self.data_format == 'wide':
            return self._conditional_filter_wide(df, threshold, mask, features)
        else:
            return self._conditional_filter_long(df, threshold, mask, features)

    def _conditional_filter_wide(self, df, threshold, mask, features):
        for feat in features:
            tid = feat.split("_")[-1]
            
            js_score = self._compute_js_divergence(
                train_sample=df.filter(mask(tid)).select(feat).drop_nulls().collect().to_numpy().flatten(),
                test_sample=df.select(feat).drop_nulls().collect().to_numpy().flatten()
            )
  
            if js_score > threshold:
                df = df.with_columns(pl.when(mask(tid)).then(pl.col(feat)).otherwise(None).alias(feat))

            logging.info(f"JS Score for feature {feat} = {js_score}")
        
        return df

    def _conditional_filter_long(self, df, threshold, mask, features):
        for feat in features:
            js_score = self._compute_js_divergence(
                train_sample=df.filter(mask).select(feat).drop_nulls().collect().to_numpy().flatten(),
                test_sample=df.select(feat).drop_nulls().collect().to_numpy().flatten()
            )
            
            if js_score > threshold:
                df = df.with_columns(pl.when(mask).then(pl.col(feat)).otherwise(None).alias(feat))

            logging.info(f"JS Score for feature {feat} = {js_score}")
        
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

