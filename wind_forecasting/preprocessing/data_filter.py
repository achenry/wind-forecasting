"""_summary_
This file contains functions to: filter the data: power > 0, inoperation flag, one-sided power curve filtering, stuck sensor, nacelle orientation
Returns:
    _type_: _description_
"""

from functools import partial

import numpy as np
import polars as pl
from openoa.utils import filters, power_curve, plot
import polars.selectors as cs
from scipy.interpolate import CubicSpline

class DataFilter:
    """_summary_
    """
    def __init__(self, turbine_availability_col=None, turbine_status_col=None):
        self.turbine_availability_col = turbine_availability_col
        self.turbine_status_col = turbine_status_col
    
    def resolve_missing_data(self, df, how="linear_interp", features=None) -> pl.LazyFrame:
        """_summary_
        option 1) interpolate via linear, or forward
        """
        
        return df.with_columns(pl.col(features).map_batches(partial(self._interpolate_series, df=df, how=how)))\
                 .fill_nan(None)
    
    def _interpolate_series(self, ser, df, how):
        """_summary_

        Args:
            ser (_type_): _description_
            how (_type_): _description_

        Returns:
            _type_: _description_
        """
        xp = df["time"].filter(ser.is_not_nan() & ser.is_not_null())
        fp = ser.filter(ser.is_not_nan() & ser.is_not_null())
        x = df["time"]

        if how == "forward_fill":
            return ser.fill_null(strategy="forward")

        if how == "linear_interp":
            return np.interp(x, xp, fp, left=np.nan, right=np.nan)
        
        if how == "cubic_interp":
            return CubicSpline(xp, fp, extrapolate=False)(x)

    def filter_inoperational(self, df, status_codes=None, availability_codes=None) -> pl.LazyFrame:
        """
        status_codes (list): List of status codes to include (e.g., [1, 3])
        availability_codes (list): List of availability codes to include (e.g., [100, 50])
        include_nan (bool): Whether to include NaN values in the filter
        """
        
        # Create masks for filtering
        include_status_mask = status_codes is not None and self.turbine_status_col is not None
        include_availability_mask = availability_codes is not None and self.turbine_availability_col is not None

        if include_status_mask:
            status_mask = df.col(self.turbine_status_col).is_in(status_codes)
        
        if include_availability_mask:
            availability_mask = df.col(self.turbine_availability_col).is_in(availability_codes)
        
        # Combine masks
        if include_status_mask and include_availability_mask:
            combined_mask = status_mask & availability_mask
        elif include_status_mask:
            combined_mask = status_mask
        elif include_availability_mask:
            combined_mask = availability_mask
        
        return df.filter(combined_mask)
    
if __name__ == "__main__":
    from wind_forecasting.preprocessing.data_loader import DataLoader

    DATA_DIR = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data"
    FILE_SIGNATURE = "kp.turbine.z02.b0.20220301.*.wt073.nc"
    MULTIPROCESSOR = None

    data_loader = DataLoader(data_dir=DATA_DIR, file_signature=FILE_SIGNATURE, multiprocessor=MULTIPROCESSOR)
    df = data_loader.read_multi_netcdf()

    data_filter = DataFilter(raw_df=df)
    df = data_filter.filter_turbine_data(status_codes=[1], availability_codes=[100], include_nan=True)
    inter_df = data_filter.resolve_missing_data(features=["wind_speed"])
