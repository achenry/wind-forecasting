"""_summary_
This file contains functions to: filter the data: power > 0, inoperation flag, one-sided power curve filtering, stuck sensor, nacelle orientation
Returns:
    _type_: _description_
"""

from functools import partial

import numpy as np
import polars as pl
import polars.selectors as cs
from scipy.interpolate import CubicSpline
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from scipy.special import kl_div

# from line_profiler import profile
from memory_profiler import profile

class DataFilter:
    """_summary_
    """
    def __init__(self, turbine_availability_col=None, turbine_status_col=None):
        self.turbine_availability_col = turbine_availability_col
        self.turbine_status_col = turbine_status_col

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
