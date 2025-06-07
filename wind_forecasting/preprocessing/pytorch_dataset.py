"""
PyTorch Dataset for wind forecasting data that enables distributed training.

This module provides a PyTorch Dataset that loads pre-split data from pickle files
and applies necessary transformations for model training.
"""

import logging
import pickle
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class WindForecastingDataset(Dataset):
    """
    PyTorch Dataset for wind forecasting data.
    
    Loads pre-split data from pickle files and applies transformations
    needed for training (time features, windowing).
    
    Parameters
    ----------
    data_path : str
        Path to the pickle file containing the dataset
    context_length : int
        Number of past time steps to use as context
    prediction_length : int
        Number of future time steps to predict
    time_features : List[callable]
        List of time feature functions to apply
    """
    
    def __init__(
        self,
        data_path: str,
        context_length: int,
        prediction_length: int,
        time_features: Optional[List] = None,
        sampler: Optional[Any] = None,  # GluonTS sampler instance
    ):
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.time_features = time_features or []
        self.sampler = sampler
        
        # Load data from pickle
        logger.info(f"Loading dataset from {data_path}")
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        logger.info(f"Loaded {len(self.data)} time series")
        
        # Validate data format
        if len(self.data) > 0:
            sample = self.data[0]
            assert 'target' in sample, "Dataset must contain 'target' field"
            assert 'start' in sample, "Dataset must contain 'start' field"
            
        # If sampler provided, pre-compute all windows using it
        if self.sampler is not None:
            self.windows = []
            for ts_idx, sample in enumerate(self.data):
                target = sample['target']
                # Get indices from sampler (like SequentialSampler which returns all valid indices)
                indices = self.sampler(target)
                # Store (time_series_index, time_index) pairs
                for t in indices:
                    self.windows.append((ts_idx, t))
            logger.info(f"Sampler created {len(self.windows)} training windows from {len(self.data)} time series")
        else:
            # Fallback: one random window per time series (original behavior)
            self.windows = None
            logger.info(f"No sampler provided, will use random sampling (1 window per time series)")
            
    def __len__(self) -> int:
        if self.windows is not None:
            return len(self.windows)
        else:
            return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample with transformations applied.
        
        Returns
        -------
        Dictionary containing:
            - past_target: (context_length, num_series)
            - future_target: (prediction_length, num_series)
            - past_time_feat: (context_length, num_time_features)
            - future_time_feat: (prediction_length, num_time_features)
            - past_observed_values: (context_length, num_series)
            - future_observed_values: (prediction_length, num_series)
            - feat_static_cat: (num_static_cat,)
            - feat_static_real: (num_static_real,)
        """
        if self.windows is not None:
            # Use pre-computed window from sampler
            ts_idx, t = self.windows[idx]
            sample = self.data[ts_idx]
        else:
            # Fallback: random sampling (original behavior)
            sample = self.data[idx]
            target = sample['target']
            _, ts_length = target.shape
            
            # Sample a time point for splitting past/future
            min_time = self.context_length
            max_time = ts_length - self.prediction_length
            
            if max_time <= min_time:
                raise ValueError(f"Time series too short: length={ts_length}, "
                               f"required={self.context_length + self.prediction_length}")
            
            t = np.random.randint(min_time, max_time + 1)
        
        # Extract data
        target = sample['target']  # Shape: (num_series, time_steps)
        start_period = sample['start']
        _, ts_length = target.shape
        
        # Split into past and future windows
        past_target = target[:, t - self.context_length:t]
        future_target = target[:, t:t + self.prediction_length]
        
        # Create time features
        time_index = pd.period_range(
            start=start_period,
            periods=ts_length,
            freq=start_period.freq
        )
        
        # Apply time feature transformations
        past_time_feat = self._create_time_features(
            time_index[t - self.context_length:t]
        )
        future_time_feat = self._create_time_features(
            time_index[t:t + self.prediction_length]
        )
        
        # Create observed values indicator (1 for observed, 0 for missing)
        past_observed = ~np.isnan(past_target)
        future_observed = ~np.isnan(future_target)
        
        # Handle NaN values
        past_target = np.nan_to_num(past_target, 0.0)
        future_target = np.nan_to_num(future_target, 0.0)
        
        # Get static features
        feat_static_cat = sample.get('feat_static_cat', [0])
        feat_static_real = sample.get('feat_static_real', [0.0])
        
        # Get dynamic features if available
        if 'feat_dynamic_real' in sample:
            feat_dynamic_real = sample['feat_dynamic_real']
            past_dynamic = feat_dynamic_real[:, t - self.context_length:t]
            future_dynamic = feat_dynamic_real[:, t:t + self.prediction_length]
            
            # Stack with time features
            past_time_feat = np.vstack([past_time_feat, past_dynamic])
            future_time_feat = np.vstack([future_time_feat, future_dynamic])
        
        # Convert to tensors and transpose to (time, features)
        return {
            'past_target': torch.from_numpy(past_target.T).float(),
            'future_target': torch.from_numpy(future_target.T).float(),
            'past_time_feat': torch.from_numpy(past_time_feat.T).float(),
            'future_time_feat': torch.from_numpy(future_time_feat.T).float(),
            'past_observed_values': torch.from_numpy(past_observed.T).float(),
            'future_observed_values': torch.from_numpy(future_observed.T).float(),
            'feat_static_cat': torch.tensor(feat_static_cat, dtype=torch.long),
            'feat_static_real': torch.tensor(feat_static_real, dtype=torch.float),
        }
    
    def _create_time_features(self, time_index: pd.PeriodIndex) -> np.ndarray:
        """
        Create time features from time index.
        
        Parameters
        ----------
        time_index : pd.PeriodIndex
            Time index to create features from
            
        Returns
        -------
        np.ndarray
            Shape (num_features, time_steps)
        """
        if not self.time_features:
            # Default to empty features
            return np.zeros((0, len(time_index)))
        
        features = []
        for feat_func in self.time_features:
            # Apply time feature function
            feat = feat_func(time_index)
            features.append(feat)
            
        return np.array(features)


class WindForecastingInferenceDataset(WindForecastingDataset):
    """
    Dataset for inference that creates windows at all valid time points.
    
    Unlike the training dataset which samples random windows, this creates
    all possible windows for complete evaluation.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Pre-compute all valid windows
        self.windows = []
        for data_idx, sample in enumerate(self.data):
            target = sample['target']
            _, ts_length = target.shape
            
            # Find all valid time points
            min_time = self.context_length
            max_time = ts_length - self.prediction_length
            
            if max_time >= min_time:
                for t in range(min_time, max_time + 1):
                    self.windows.append((data_idx, t))
                    
        logger.info(f"Created {len(self.windows)} inference windows from {len(self.data)} time series")
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a specific window for inference."""
        data_idx, t = self.windows[idx]
        sample = self.data[data_idx]
        
        # Same processing as parent class but with fixed time point
        target = sample['target']
        start_period = sample['start']
        
        # Split into past and future windows at fixed time point t
        past_target = target[:, t - self.context_length:t]
        future_target = target[:, t:t + self.prediction_length]
        
        # Rest of processing is identical to parent class
        # ... (same as parent __getitem__ but without random sampling)
        
        # Create time features
        _, ts_length = target.shape
        time_index = pd.period_range(
            start=start_period,
            periods=ts_length,
            freq=start_period.freq
        )
        
        past_time_feat = self._create_time_features(
            time_index[t - self.context_length:t]
        )
        future_time_feat = self._create_time_features(
            time_index[t:t + self.prediction_length]
        )
        
        # Create observed values indicator
        past_observed = ~np.isnan(past_target)
        future_observed = ~np.isnan(future_target)
        
        # Handle NaN values
        past_target = np.nan_to_num(past_target, 0.0)
        future_target = np.nan_to_num(future_target, 0.0)
        
        # Get static features
        feat_static_cat = sample.get('feat_static_cat', [0])
        feat_static_real = sample.get('feat_static_real', [0.0])
        
        # Get dynamic features if available
        if 'feat_dynamic_real' in sample:
            feat_dynamic_real = sample['feat_dynamic_real']
            past_dynamic = feat_dynamic_real[:, t - self.context_length:t]
            future_dynamic = feat_dynamic_real[:, t:t + self.prediction_length]
            
            # Stack with time features
            past_time_feat = np.vstack([past_time_feat, past_dynamic])
            future_time_feat = np.vstack([future_time_feat, future_dynamic])
        
        # Convert to tensors
        return {
            'past_target': torch.from_numpy(past_target.T).float(),
            'future_target': torch.from_numpy(future_target.T).float(),
            'past_time_feat': torch.from_numpy(past_time_feat.T).float(),
            'future_time_feat': torch.from_numpy(future_time_feat.T).float(),
            'past_observed_values': torch.from_numpy(past_observed.T).float(),
            'future_observed_values': torch.from_numpy(future_observed.T).float(),
            'feat_static_cat': torch.tensor(feat_static_cat, dtype=torch.long),
            'feat_static_real': torch.tensor(feat_static_real, dtype=torch.float),
        }