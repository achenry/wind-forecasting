"""
PyTorch Dataset for wind forecasting data that enables distributed training.

This module provides a PyTorch Dataset that loads pre-split data from pickle files
and applies necessary transformations for model training.
"""

import logging
import pickle
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset
import torch.distributed as dist
from itertools import cycle, islice

logger = logging.getLogger(__name__)


class WindForecastingDataset(IterableDataset):
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
        repeat: bool = False,
        skip_indices: int = 1
    ):
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.time_features = time_features or []
        self.sampler = sampler
        self.repeat = repeat
        self.skip_indices = skip_indices
        
        # Load data from pickle
        logger.info(f"Loading dataset from {data_path}")
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
            
        self.n_datasets = len(self.data)
        logger.info(f"Loaded {self.n_datasets} time series")
        
        # Validate data format
        if len(self.data) > 0:
            sample = self.data[0]
            assert 'target' in sample, "Dataset must contain 'target' field"
            assert 'start' in sample, "Dataset must contain 'start' field"
        
        if self.repeat:
            self.data = cycle(self.data)
        else:
            self.data = iter(self.data)
            
        self.dataset_idx = 0
        # These attributes will be configured by the worker_init_fn.
        # Set defaults for the non-worker/non-distributed case.
        self.worker_shard_start = 0
        self.worker_shard_step = 1
        
    def __iter__(self):
        # return islice(self._base_iter(), self.worker_shard_start, None, self.worker_shard_step)
        
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
        logger.info(f"rank={rank}, world_size={world_size}")
                
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None: # Main process, num_workers=0 case
            return islice(self._base_iter(), rank, None, world_size)
        else: # In a worker process
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            
            global_num_workers = num_workers * world_size
            global_worker_id = rank * num_workers + worker_id
            logger.info(f"global_worker_id={global_worker_id}, global_num_workers={global_num_workers}")

            return islice(self._base_iter(), global_worker_id, None, global_num_workers)
        
    def _base_iter(self):
    
        """
        Get an iterable of samples with transformations applied.
        
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
        
        for entry in self.data:
            
            sampled_indices = self.sampler(entry['target'])[:, :, self.skip_indices]
            
            if len(sampled_indices) == 0:
                continue
            
            for idx in sampled_indices:
                # Extract data
                target = entry['target']  # Shape: (num_series, time_steps)
                start_period = entry['start']
                _, ts_length = target.shape
                
                # Find all valid time points
                min_time = self.context_length
                max_time = ts_length - self.prediction_length
                
                if max_time < min_time:
                    continue
                
                # Split into past and future windows
                past_target = target[:, idx - self.context_length:idx]
                future_target = target[:, idx:idx + self.prediction_length]
                
                # Create time features
                time_index = pd.period_range(
                    start=start_period,
                    periods=ts_length,
                    freq=start_period.freq
                )
                
                # Apply time feature transformations
                past_time_feat = self._create_time_features(
                    time_index[idx - self.context_length:idx]
                )
                future_time_feat = self._create_time_features(
                    time_index[idx:idx + self.prediction_length]
                )
                
                # Create observed values indicator (1 for observed, 0 for missing)
                past_observed = ~np.isnan(past_target)
                future_observed = ~np.isnan(future_target)
                
                # Handle NaN values
                past_target = np.nan_to_num(past_target, 0.0)
                future_target = np.nan_to_num(future_target, 0.0)
                
                # Get static features
                feat_static_cat = entry.get('feat_static_cat', [0])
                feat_static_real = entry.get('feat_static_real', [0.0])
                
                # Get dynamic features if available
                if 'feat_dynamic_real' in entry:
                    feat_dynamic_real = entry['feat_dynamic_real']
                    past_dynamic = feat_dynamic_real[:, idx - self.context_length:idx]
                    future_dynamic = feat_dynamic_real[:, idx:idx + self.prediction_length]
                    
                    # Stack with time features
                    past_time_feat = np.vstack([past_time_feat, past_dynamic])
                    future_time_feat = np.vstack([future_time_feat, future_dynamic])
                
                # Convert to tensors and transpose to (time, features)
                yield {
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
        
    def _base_iter(self):
        """Get a specific window for inference."""
        
        for entry in self.data:
            # Same processing as parent class but with fixed time point
            target = entry['target']
            start_period = entry['start']
            ts_length = target.shape[1]
        
            # Find all valid time points
            min_time = self.context_length
            max_time = ts_length - self.prediction_length
            
            # Fill time indices
            sample_indices = np.arange(min_time, max_time + 1, self.skip_indices)
            
            if len(sample_indices) == 0:
                continue
            
            for idx in sample_indices:
                # Split into past and future windows at fixed time point t
                past_target = target[:, idx - self.context_length:idx]
                future_target = target[:, idx:idx + self.prediction_length]
                
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
                    time_index[idx - self.context_length:idx]
                )
                future_time_feat = self._create_time_features(
                    time_index[idx:idx + self.prediction_length]
                )
                
                # Create observed values indicator
                past_observed = ~np.isnan(past_target)
                future_observed = ~np.isnan(future_target)
                
                # Handle NaN values
                past_target = np.nan_to_num(past_target, 0.0)
                future_target = np.nan_to_num(future_target, 0.0)
                
                # Get static features
                feat_static_cat = entry.get('feat_static_cat', [0])
                feat_static_real = entry.get('feat_static_real', [0.0])
                
                # Get dynamic features if available
                if 'feat_dynamic_real' in entry:
                    feat_dynamic_real = entry['feat_dynamic_real']
                    past_dynamic = feat_dynamic_real[:, idx - self.context_length:idx]
                    future_dynamic = feat_dynamic_real[:, idx:idx + self.prediction_length]
                    
                    # Stack with time features
                    past_time_feat = np.vstack([past_time_feat, past_dynamic])
                    future_time_feat = np.vstack([future_time_feat, future_dynamic])
                
                # Convert to tensors
                yield {
                    'past_target': torch.from_numpy(past_target.T).float(),
                    'future_target': torch.from_numpy(future_target.T).float(),
                    'past_time_feat': torch.from_numpy(past_time_feat.T).float(),
                    'future_time_feat': torch.from_numpy(future_time_feat.T).float(),
                    'past_observed_values': torch.from_numpy(past_observed.T).float(),
                    'future_observed_values': torch.from_numpy(future_observed.T).float(),
                    'feat_static_cat': torch.tensor(feat_static_cat, dtype=torch.long),
                    'feat_static_real': torch.tensor(feat_static_real, dtype=torch.float),
                }
    
    def __iter__(self) -> Dict[str, torch.Tensor]:
        
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
            
        logger.info(f"rank={rank}, world_size={world_size}")
            
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None: # Main process, num_workers=0 case
            return islice(self._base_iter(), rank, None, world_size)
        else: # In a worker process
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            
            global_num_workers = num_workers * world_size
            global_worker_id = rank * num_workers + worker_id
            logger.info(f"global_worker_id={global_worker_id}, global_num_workers={global_num_workers}")

            return islice(self._base_iter(), global_worker_id, None, global_num_workers)