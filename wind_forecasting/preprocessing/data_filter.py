"""_summary_
This file contains functions to: filter the data: power > 0, inoperation flag, one-sided power curve filtering, stuck sensor, nacelle orientation
Returns:
    _type_: _description_
"""

from functools import partial

import numpy as np
import polars as pl
from scipy.interpolate import CubicSpline

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
            return df.filter((pl.col(self.turbine_status_col).is_in(status_codes) 
                              | (pl.col(self.turbine_status_col).is_null() if include_nan else False)) 
                             & (pl.col(self.turbine_availability_col).is_in(availability_codes) 
                                | (pl.col(self.turbine_availability_col).is_null() if include_nan else False)))
        elif include_status_mask:
            return df.filter(pl.col(self.turbine_status_col).is_in(status_codes)
                             | (pl.col(self.turbine_status_col).is_null() if include_nan else False))
        elif include_availability_mask:
            return df.filter(pl.col(self.turbine_availability_col).is_in(availability_codes) 
                             | (pl.col(self.turbine_availability_col).is_null() if include_nan else False))

    
    def resolve_missing_data(self, df, how="linear_interp", features=None) -> pl.LazyFrame:
        """_summary_
        option 1) interpolate via linear, or forward
        option 2) remove rows TODO may need to split into multiple datasets
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
