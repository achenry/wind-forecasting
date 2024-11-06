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
    def __init__(self, turbine_availability_col=None, turbine_status_col=None, multiprocessor=None):
        self.turbine_availability_col = turbine_availability_col
        self.turbine_status_col = turbine_status_col
        self.multiprocessor = multiprocessor

    def filter_inoperational(self, df, status_codes=None, availability_codes=None, include_nan=True) -> pl.LazyFrame:
        """
        status_codes (list): List of status codes to include (e.g., [1, 3])
        availability_codes (list): List of availability codes to include (e.g., [100, 50])
        include_nan (bool): Whether to include NaN values in the filter
        """
        
        # Create masks for filtering
        include_status_mask = status_codes is not None and self.turbine_status_col is not None
        include_availability_mask = availability_codes is not None and self.turbine_availability_col is not None

        # Combine masks 
        if include_status_mask and include_availability_mask:
            return df.with_columns(pl.when(cs.starts_with(self.turbine_status_col).is_in(status_codes) 
                                            | (cs.starts_with(self.turbine_status_col).is_null() if include_nan else False))\
                                     .then(cs.starts_with(self.turbine_status_col)),
                                    pl.when(cs.starts_with(self.turbine_availability_col).is_in(availability_codes) 
                                            | (cs.starts_with(self.turbine_availabilty_col).is_null() if include_nan else False))\
                                     .then(cs.starts_with(self.turbine_availability_col)) )
        elif include_status_mask:
            return df.with_columns(pl.when(cs.starts_with(self.turbine_status_col).is_in(status_codes) 
                                            | (cs.starts_with(self.turbine_status_col).is_null() if include_nan else False))\
                                     .then(cs.starts_with(self.turbine_status_col)))
        elif include_availability_mask:
            # return df.filter(cs.starts_with(self.turbine_availability_col).is_in(availability_codes) 
            #                  | (cs.starts_with(self.turbine_availability_col).is_null() if include_nan else False))
            return df.with_columns(pl.when(cs.starts_with(self.turbine_availability_col).is_in(availability_codes) 
                                            | (cs.starts_with(self.turbine_availabilty_col).is_null() if include_nan else False))\
                                     .then(cs.starts_with(self.turbine_availability_col)))
        
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
        unpivot_df = DataInspector.unpivot_dataframe(df, impute_missing_features)

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
                                     data=unpivot_df.select(["time", "turbine_id", feature]).collect().to_pandas().set_index(["time", "turbine_id"]),
                                     impute_col=feature, reference_col=feature,
                                     asset_id_col="turbine_id", method="linear") for feature in impute_missing_features}

                unpivot_df = unpivot_df.with_columns({k: v.result() for k, v in futures.items()}).fill_nan(None)
        else:
            for feature in impute_missing_features:
                # n_nulls_before = unpivot_df.select(cs.contains(feature)).select(pl.sum_horizontal(pl.all().is_null()).sum()).collect().item()
                # print(f"# Missing values before imputation = {n_nulls_before}")
                
                other_feature = feature
                features = set(["time", "turbine_id", feature, other_feature])

                imputed_vals = imputing.impute_all_assets_by_correlation(
                    data=unpivot_df.select(features).collect().to_pandas().set_index(["time", "turbine_id"]),
                                                            impute_col=feature, reference_col=other_feature,
                                                            asset_id_col="turbine_id", method="linear", multiprocessor=self.multiprocessor).to_numpy()
                
                unpivot_df = unpivot_df.with_columns({feature: imputed_vals}).fill_nan(None)
                # n_nulls_after = unpivot_df.select(cs.contains(feature)).select(pl.sum_horizontal(pl.all().is_null()).sum()).collect().item()
                # print(f"# Missing values after imputation = {n_nulls_after}")
                logging.info(f"Imputed feature {feature} in DataFrame {df_idx}.")
                # logging.info(f"Successfully imputed {n_nulls_before - n_nulls_after} cells for feature {feature} in DataFrame {df_idx}.")
        return DataInspector.pivot_dataframe(unpivot_df)

    def _fill_single_missing_dataset(self, df_idx, df, impute_missing_features, interpolate_missing_features, available_features, parallel=None):
        
        df = self._impute_single_missing_dataset(df_idx, df, impute_missing_features, parallel=parallel)

        # n_nulls_before = df.select([cs.contains(feat) for feat in interpolate_missing_features]).select(pl.sum_horizontal(pl.all().is_null()).sum()).collect().item()
        # print(f"# Missing values before interpolation = {n_nulls_before}")
        # TODO if any column is all nulls ... can't be imputed
        df = df.with_columns([cs.starts_with(feat).interpolate().fill_null(strategy="forward").fill_null(strategy="backward") for feat in interpolate_missing_features])
                                 
        # df = df.with_columns(pl.col("wind_speed_wt030").fill_null(strategy="forward"), pl.col("wind_speed_wt081").fill_null(strategy="forward"))
        
        # .fill_null(strategy="backward"))
        # n_nulls_after = df.select([cs.contains(feat) for feat in interpolate_missing_features]).select(pl.sum_horizontal(pl.all().is_null()).sum()).collect().item()
        # print(f"# Missing values after interpolation = {n_nulls_after}")
        # logging.info(f"Successfully interpolated {n_nulls_before - n_nulls_after} cells in DataFrame {df_idx}.")

        if df.filter(pl.any_horizontal(pl.all().is_null())).select(pl.len()).collect().item():
            print(f"Error, there are still nulls in dataframe {df_idx}!")

        # for feature in interpolate_missing_features:       
        #     # if not imputed:
        #     # allow interpolation from its own colums, (and others if that is not possible using interpolate_by?)
        #     # df = df.with_columns(pl.col(feature).interpolate())
        #     for turbine_feature in [f for f in available_features if feature in f]:
        #         # interpolate remaining null values with forward fill or backward fill
        #         if df.filter(pl.col(turbine_feature).is_null()).select(pl.len()).collect().item():
        #             df = df.with_columns(pl.col(turbine_feature).interpolate())
        #             # logging.info(f"Successfully interpolated feature {turbine_feature} in DataFrame {df_idx}.")
            
        #         # fill any remaining null values with forward fill or backward fill
        #         if df.filter(pl.col(turbine_feature).is_null()).select(pl.len()).collect().item():

        #             df = df.with_columns(pl.col(turbine_feature).fill_null(strategy="forward"))\
        #                 .with_columns(pl.col(turbine_feature).fill_null(strategy="backward"))

        #             logging.info(f"Successfully filled feature {turbine_feature} in DataFrame {df_idx}.")
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
        only applies mask to features if the Jensen-Shannon metric between filtered and unfiltered data exceeds a threshold
        """
        for feat in features:
            tid = feat.split("_")[-1]
            js_score = __class__._compute_js_divergence(
                train_sample=df.filter(mask(tid)).select(feat).drop_nulls().collect().to_numpy().flatten(),
                test_sample=df.select(feat).drop_nulls().collect().to_numpy().flatten()
                )
            
            if js_score > threshold:
                df = df.with_columns(pl.when(mask(tid)).then(feat).alias(feat))

            print(f"JS Score for feature {feat} = {js_score}")
        
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

        x = x % 360.0  # convert to range 0 to 360 degrees
        x = np.where(x > 180.0, x - 360.0, x)
        return x if input_type != float else float(x)

    @staticmethod
    def circ_mean(x):
        y = (
                np.degrees(
                    np.arctan2(
                        np.nanmean(np.sin(np.radians(x))),
                        np.nanmean(np.cos(np.radians(x))),
                    )
                )
                % 360.0
            )
        
        return y
