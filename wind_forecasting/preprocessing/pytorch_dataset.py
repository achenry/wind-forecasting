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
from itertools import cycle

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

        # Load data from pickle - keep as list for partitioning
        logger.info(f"Loading dataset from {data_path}")
        with open(data_path, 'rb') as f:
            self.full_data_list = pickle.load(f)

        self.n_datasets = len(self.full_data_list)
        logger.info(f"Loaded {self.n_datasets} time series")

        # Validate data format
        if len(self.full_data_list) > 0:
            sample = self.full_data_list[0]
            assert 'target' in sample, "Dataset must contain 'target' field"
            assert 'start' in sample, "Dataset must contain 'start' field"

        # Cache for __len__ calculation
        self._total_samples = None

        self.dataset_idx = 0
        # These attributes are no longer used but kept for compatibility
        self.worker_shard_start = 0
        self.worker_shard_step = 1

    def __iter__(self):
        """
        Create iterator with proper data partitioning for DDP and multi-worker DataLoader.

        Each worker processes a unique subset of time series based on its global worker ID.
        This eliminates redundant computation across workers.
        """
        import os
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:  # Main process, num_workers=0 case
            global_worker_id = rank
            global_num_workers = world_size
        else:  # In a worker process
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

            global_num_workers = num_workers * world_size
            global_worker_id = rank * num_workers + worker_id

        logger.info(f"rank={rank}, world_size={world_size}")
        logger.info(f"global_worker_id={global_worker_id}, global_num_workers={global_num_workers}")

        # CRITICAL FIX: Partition time series by global_worker_id
        # Each worker gets every global_num_workers-th time series
        # This eliminates 21× redundant computation from the old islice approach
        worker_time_series = [
            entry for idx, entry in enumerate(self.full_data_list)
            if idx % global_num_workers == global_worker_id
        ]

        logger.info(
            f"Worker {global_worker_id}/{global_num_workers} assigned "
            f"{len(worker_time_series)}/{self.n_datasets} time series"
        )

        # Convert to iterator with repeat if needed
        if self.repeat:
            worker_data_iter = cycle(worker_time_series)
        else:
            worker_data_iter = iter(worker_time_series)

        # Process only THIS worker's time series (no islice filtering!)
        return self._base_iter(worker_data_iter)

    def _base_iter(self, data_iter):
        """
        Get an iterable of samples with transformations applied.

        Parameters
        ----------
        data_iter : iterator
            Iterator over time series entries assigned to this worker

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

        for entry in data_iter:

            # Get sampled indices from the sampler
            if self.sampler is not None:
                # GluonTS samplers work on 1D arrays (time dimension)
                # They return a 1D array of sampled time indices
                target = entry['target']  # Shape: (num_series, time_steps)
                _, ts_length = target.shape

                # The sampler needs the time series length, not the actual data
                # Create a dummy 1D target for the sampler
                dummy_target = np.zeros(ts_length)
                sampled_indices = self.sampler(dummy_target)

                # Apply skip_indices if needed
                if self.skip_indices > 1:
                    sampled_indices = sampled_indices[::self.skip_indices]
            else:
                # If no sampler, use all valid time points
                target = entry['target']
                _, ts_length = target.shape
                min_time = self.context_length
                max_time = ts_length - self.prediction_length
                if max_time < min_time:
                    continue
                sampled_indices = np.arange(min_time, max_time + 1, self.skip_indices)

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

    def _calculate_total_samples(self):
        """
        Calculate total number of samples across all time series.

        This is used for __len__ to support progress bars.
        """
        if self._total_samples is not None:
            return self._total_samples

        total = 0
        for entry in self.full_data_list:
            target = entry['target']
            _, ts_length = target.shape

            if self.sampler is not None:
                # Use sampler to determine valid indices
                dummy_target = np.zeros(ts_length)
                sampled_indices = self.sampler(dummy_target)

                if self.skip_indices > 1:
                    sampled_indices = sampled_indices[::self.skip_indices]

                total += len(sampled_indices)
            else:
                # Manual calculation
                min_time = self.context_length
                max_time = ts_length - self.prediction_length

                if max_time >= min_time:
                    num_valid = (max_time - min_time + 1)
                    total += (num_valid + self.skip_indices - 1) // self.skip_indices

        self._total_samples = total
        logger.info(f"Calculated total samples: {total}")
        return total

    def __len__(self):
        """
        Return total number of samples in dataset.

        This enables progress bars to show X/Y instead of X/?.
        Note: In DDP, Lightning will automatically adjust this per-rank.
        """
        return self._calculate_total_samples()

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

    def __iter__(self) -> Dict[str, torch.Tensor]:
        """
        Create iterator with proper data partitioning for DDP and multi-worker DataLoader.

        Each worker processes a unique subset of time series based on its global worker ID.
        """
        import os
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:  # Main process, num_workers=0 case
            global_worker_id = rank
            global_num_workers = world_size
        else:  # In a worker process
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

            global_num_workers = num_workers * world_size
            global_worker_id = rank * num_workers + worker_id

        logger.info(f"rank={rank}, world_size={world_size}")
        logger.info(f"global_worker_id={global_worker_id}, global_num_workers={global_num_workers}")

        # CRITICAL FIX: Partition time series by global_worker_id
        worker_time_series = [
            entry for idx, entry in enumerate(self.full_data_list)
            if idx % global_num_workers == global_worker_id
        ]

        logger.info(
            f"Worker {global_worker_id}/{global_num_workers} assigned "
            f"{len(worker_time_series)}/{self.n_datasets} time series"
        )

        # Convert to iterator with repeat if needed
        if self.repeat:
            worker_data_iter = cycle(worker_time_series)
        else:
            worker_data_iter = iter(worker_time_series)

        # Process only THIS worker's time series
        return self._base_iter_inference(worker_data_iter)

    def _base_iter_inference(self, data_iter):
        """Get samples for inference from assigned time series."""

        for entry in data_iter:
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
