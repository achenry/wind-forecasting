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
    def __init__(self, raw_df, turbine_availability_col=None, turbine_status_col=None):
        self.df = raw_df
        self.turbine_availability_col = turbine_availability_col
        self.turbine_status_col = turbine_status_col

    def filter_inoperational(self, status_codes=None, availability_codes=None, include_nan=False) -> pl.DataFrame:
        """
        status_codes (list): List of status codes to include (e.g., [1, 3])
        availability_codes (list): List of availability codes to include (e.g., [100, 50])
        include_nan (bool): Whether to include NaN values in the filter
        """
        
        # Forward fill NaN values TODO what to do about NaN status and availability
        # self.df = self.df.with_columns([
        #     pl.col("turbine_status").forward_fill(),
        #     pl.col("turbine_availability").forward_fill()
        # ])
        
        # Create masks for filtering
        status_mask = [True] * self.df.shape[0]
        availability_mask = [True] * self.df.shape[0]
        
        if status_codes is not None and self.turbine_status_col is not None:
            status_mask = self.df[self.turbine_status_col].is_in(status_codes)
        
        if availability_codes is not None and self.turbine_availability_col is not None:
            availability_mask = self.df[self.turbine_availability_col].is_in(availability_codes)
        
        # Combine masks
        combined_mask = pl.Series(status_mask & availability_mask)
        
        if include_nan:
            combined_mask |= pl.Series((self.df['turbine_status'].is_nan() | self.df['turbine_status'].is_null()) 
                                       & (self.df['turbine_availability'].is_nan() | self.df['turbine_availability'].is_null()))
            
        
        self.df = self.df.filter(combined_mask)
        return self.df
    
    def resolve_missing_data(self, how="linear_interp", features=None) -> pl.DataFrame:
        """_summary_
        option 1) interpolate via linear, or forward
        option 2) remove rows TODO may need to split into multiple datasets
        """
        # self.df = self.df.with_columns((pl.col(features) & cs.numeric()).map_batches(partial(self._interpolate_series, how=how)))
        self.df = self.df.with_columns(pl.col(features).map_batches(partial(self._interpolate_series, how=how)))
        self.df = self.df.fill_nan(None)
        return self.df
            
    def _interpolate_series(self, ser, how):
        """_summary_

        Args:
            ser (_type_): _description_
            how (_type_): _description_

        Returns:
            _type_: _description_
        """
        xp = self.df["time"].filter(ser.is_not_nan() & ser.is_not_null())
        fp = ser.filter(ser.is_not_nan() & ser.is_not_null())
        x = self.df["time"]

        if how == "linear_interp":
            return np.interp(x, xp, fp, left=np.nan, right=np.nan)
        
        if how == "cubic_interp":
            return CubicSpline(xp, fp, extrapolate=False)(x) 
        # return self.df.select(pl.Float64().map_batches(lambda ser: np.interp(self.df["Time"], self.df.select(["Time"]).filter(ser.is_not_nan()), ser.filter(ser.is_not_nan()))))
    
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