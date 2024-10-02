### This file contains functions to: 
### - filter the data: power > 0, inoperation flag, one-sided power curve filtering, stuck sensor, nacelle orientation

import openoa
import polars as pl
import numpy as np

class DataFilter:
    def __init__(self, raw_df):
        self.df = raw_df.clone()

    def filter_turbine_data(self, status_codes=None, availability_codes=None, include_nan=False):
        """
        status_codes (list): List of status codes to include (e.g., [1, 3])
        availability_codes (list): List of availability codes to include (e.g., [100, 50])
        include_nan (bool): Whether to include NaN values in the filter
        """
        
        # Forward fill NaN values
        self.df = self.df.with_columns([
            pl.col("turbine_status").forward_fill(),
            pl.col("turbine_availability").forward_fill()
        ])
        
        # Create masks for filtering
        status_mask = [True] * self.df.shape[0]
        availability_mask = [True] * self.df.shape[0]
        
        if status_codes is not None:
            status_mask = self.df['turbine_status'].is_in(status_codes)
        
        if availability_codes is not None:
            availability_mask = self.df['turbine_availability'].is_in(availability_codes)
        
        # Combine masks
        combined_mask = np.array(status_mask) & np.array(availability_mask)
        
        if include_nan:
            combined_mask |= (self.df['turbine_status'].is_nan() & self.df['turbine_availability'].is_nan()).to_numpy()
        
        self.df = self.df.filter(combined_mask)
        return self.df