"""
PyTorch Dataset for wind forecasting data that enables distributed training.

This module provides a PyTorch Dataset that loads pre-split data from pickle files
and applies necessary transformations for model training.
"""

import logging
import pickle
from typing import List, Optional, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset
from itertools import cycle, islice
from torch.utils.data import DataLoader
import lightning.pytorch as pl

logger = logging.getLogger(__name__)

class WindForecastingDatamodule(pl.LightningDataModule):
    def __init__(self, train_data_path, val_data_path, train_sampler, 
                 context_length, prediction_length, time_features, val_sampler=None, 
                 train_repeat=True, val_repeat=False,
                 batch_size=32, num_workers=4, pin_memory=True):
        super().__init__()
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.time_features = time_features
        
        self.train_repeat = train_repeat
        self.val_repeat = val_repeat
        
        # These will be set in the setup() hook
        self.world_size = 1
        self.rank = 0

    def setup(self, stage: str):
        # This hook is called on each DDP process, so `self.trainer` is available.
        if self.trainer:
            self.rank = self.trainer.global_rank
            self.world_size = self.trainer.world_size

    def train_dataloader(self):
        # The Trainer calls this after setup() on each DDP process
        train_dataset = WindForecastingDataset(
                data_path=self.train_data_path,
                context_length=self.context_length,
                prediction_length=self.prediction_length,
                time_features=self.time_features,
                sampler=self.train_sampler,  # GluonTS sampler instance
                repeat=self.train_repeat,
                skip_indices=1,
                world_size=self.world_size,
                rank=self.rank)
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Will be overridden by DistributedSampler in DDP
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers>0,
            # drop_last=True,  # Important for DDP to avoid uneven batch sizes
        )
        
    def val_dataloader(self):
        # This is also called at the correct time.
        # It will correctly shard the validation data across all GPUs and workers.
        # if self.val_dataset is None:
        #     return None
        
        val_dataset = WindForecastingInferenceDataset(
                data_path=self.val_data_path,
                context_length=self.context_length,
                prediction_length=self.prediction_length,
                time_features=self.time_features,
                sampler=self.val_sampler,  # GluonTS sampler instance
                repeat=self.val_repeat,
                skip_indices=self.prediction_length,
                world_size=self.world_size,
                rank=self.rank)
        
        if val_dataset is None:
            return None
        
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Never shuffle validation data
            # worker_init_fn=self.__class__._worker_init_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers>0,
            # drop_last=False,  # Keep all validation samples
        )

    def test_dataloader(self):
        # The same pattern applies for testing.
        if self.test_dataset is None:
            return None
            
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Never shuffle validation data
            # worker_init_fn=self.__class__._worker_init_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers>0,
            # drop_last=False,  # Keep all validation samples
        )


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
        skip_indices: int = 1,
        rank: int = 0,
        world_size: int = 1
    ):
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.time_features = time_features or []
        self.sampler = sampler
        self.repeat = repeat
        self.skip_indices = skip_indices
        
        # Store world_size and rank passed from the DataModule
        self.world_size = world_size
        self.rank = rank
        
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
            
        self.dataset_idx = 0
        
    def __iter__(self):
        
        if self.world_size > 1:
            logger.info(f"Using distributed training with rank={self.rank}, world_size={self.world_size}")
        else:
            logger.info(f"Using single-rank training with rank={self.rank}, world_size={self.world_size}")
                
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None: # Main process, num_workers=0 case
            logger.info(f"training worker_info is None, on main process fetching islice {self.rank}:None:{self.world_size}")
            return islice(self._base_iter(), self.rank, None, self.world_size)
        else: # In a worker process
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            
            global_num_workers = num_workers * self.world_size
            global_worker_id = self.rank * num_workers + worker_id
            logger.info(f"training worker {worker_info.id} of {num_workers}, fetching islice {global_worker_id}:None:{global_num_workers}")

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
        
        # Create a NEW, FRESH iterator from the source list every time.
        data_iterator = iter(self.data)
        
        # Apply cycle() here if needed, on the new iterator.
        if self.repeat:
            data_iterator = cycle(data_iterator)
        
        for entry in data_iterator:
            
            sampled_indices = self.sampler(entry['target'])[::self.skip_indices]
            
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
        
        data_iterator = self.data
        self.samples = []
        for entry in data_iterator:
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
                self.samples.append({
                    'past_target': torch.from_numpy(past_target.T).float(),
                    'future_target': torch.from_numpy(future_target.T).float(),
                    'past_time_feat': torch.from_numpy(past_time_feat.T).float(),
                    'future_time_feat': torch.from_numpy(future_time_feat.T).float(),
                    'past_observed_values': torch.from_numpy(past_observed.T).float(),
                    'future_observed_values': torch.from_numpy(future_observed.T).float(),
                    'feat_static_cat': torch.tensor(feat_static_cat, dtype=torch.long),
                    'feat_static_real': torch.tensor(feat_static_real, dtype=torch.float),
                })
        
    def _base_iter(self):
        """Get a specific window for inference."""
        for s in self.samples:
            yield s