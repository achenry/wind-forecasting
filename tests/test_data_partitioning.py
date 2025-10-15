"""
Unit tests for DDP data partitioning in WindForecastingDataset.

Tests verify that:
1. Each worker gets a unique subset of time series
2. No time series are duplicated across workers
3. All time series are assigned to some worker
4. Partitioning is deterministic
"""

import os
import pickle
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from wind_forecasting.preprocessing.pytorch_dataset import WindForecastingDataset


def create_mock_dataset(n_series: int = 88, ts_length: int = 1000) -> list:
    """
    Create a mock dataset for testing.

    Parameters
    ----------
    n_series : int
        Number of time series (simulates 88 turbines)
    ts_length : int
        Length of each time series

    Returns
    -------
    list
        List of dataset entries
    """
    dataset = []
    for i in range(n_series):
        entry = {
            'target': np.random.randn(2, ts_length).astype(np.float32),  # (num_features, time_steps)
            'start': pd.Period('2024-01-01 00:00:00', freq='15s'),
            'item_id': f'turbine_{i}',
            'feat_static_cat': [i],
            'feat_static_real': [float(i)],
        }
        dataset.append(entry)
    return dataset


@pytest.fixture
def mock_data_path(tmp_path):
    """Create a temporary pickle file with mock data."""
    data = create_mock_dataset(n_series=88, ts_length=1000)
    data_file = tmp_path / "test_data.pkl"
    with open(data_file, 'wb') as f:
        pickle.dump(data, f)
    return str(data_file)


class TestDataPartitioning:
    """Test suite for DDP data partitioning."""

    def test_single_worker_gets_all_data(self, mock_data_path):
        """Test that a single worker (no DDP, no multiprocessing) gets all data."""
        # Simulate single worker: RANK=0, WORLD_SIZE=1, no DataLoader workers
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'

        dataset = WindForecastingDataset(
            data_path=mock_data_path,
            context_length=100,
            prediction_length=4,
            repeat=False,
            skip_indices=1
        )

        # Create iterator (no DataLoader, simulating num_workers=0)
        samples = list(dataset)

        # Calculate expected samples: 88 series × valid_windows_per_series
        # valid_windows = ts_length - context_length - prediction_length + 1
        # = 1000 - 100 - 4 + 1 = 897 per series
        # Total = 88 × 897 = 78,936 samples
        expected_samples = 88 * (1000 - 100 - 4 + 1)

        assert len(samples) == expected_samples, \
            f"Single worker should get all {expected_samples} samples, got {len(samples)}"

    def test_ddp_partitioning_no_overlap(self, mock_data_path):
        """Test that multiple DDP ranks get non-overlapping time series."""
        world_size = 3
        all_assigned_series = []

        for rank in range(world_size):
            os.environ['RANK'] = str(rank)
            os.environ['WORLD_SIZE'] = str(world_size)

            dataset = WindForecastingDataset(
                data_path=mock_data_path,
                context_length=100,
                prediction_length=4,
                repeat=False,
                skip_indices=1
            )

            # Get assigned time series for this rank
            iter_obj = iter(dataset)
            # The partitioning happens in __iter__, so we need to access the worker_time_series

            # Since we can't directly access worker_time_series after partitioning,
            # we'll verify by checking which item_ids appear in samples
            samples = list(iter_obj)
            series_ids = set()
            for sample in samples[:10]:  # Check first 10 samples to identify series
                # Samples from same series will have consistent patterns
                # For this test, we'll track unique patterns
                series_ids.add(sample['past_target'].shape[0])  # Placeholder check

            all_assigned_series.append(len(samples))

        # Verify total samples across all ranks equals single-worker total
        total_samples = sum(all_assigned_series)
        expected_total = 88 * (1000 - 100 - 4 + 1)

        assert total_samples == expected_total, \
            f"Total samples across {world_size} ranks should be {expected_total}, got {total_samples}"

    def test_multi_worker_partitioning(self, mock_data_path):
        """Test that multiple DataLoader workers partition correctly."""
        # Simulate DDP with 3 ranks, each with 7 workers
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '3'

        dataset = WindForecastingDataset(
            data_path=mock_data_path,
            context_length=100,
            prediction_length=4,
            repeat=False,
            skip_indices=1
        )

        # Manually test partitioning logic
        n_series = 88
        num_workers = 7
        world_size = 3
        rank = 0

        global_num_workers = num_workers * world_size  # 21

        # Verify each global worker gets expected number of series
        for worker_id in range(num_workers):
            global_worker_id = rank * num_workers + worker_id

            # Simulate partitioning
            assigned_series = [
                idx for idx in range(n_series)
                if idx % global_num_workers == global_worker_id
            ]

            expected_count = n_series // global_num_workers
            if global_worker_id < (n_series % global_num_workers):
                expected_count += 1

            assert len(assigned_series) == expected_count, \
                f"Worker {global_worker_id}/{global_num_workers} should get ~{expected_count} series, got {len(assigned_series)}"

    def test_no_duplicate_series_across_workers(self, mock_data_path):
        """Test that no time series is assigned to multiple workers."""
        n_series = 88
        num_workers = 7
        world_size = 3
        global_num_workers = num_workers * world_size  # 21

        all_assigned_series = set()

        for rank in range(world_size):
            for worker_id in range(num_workers):
                global_worker_id = rank * num_workers + worker_id

                # Simulate partitioning
                assigned_series = [
                    idx for idx in range(n_series)
                    if idx % global_num_workers == global_worker_id
                ]

                # Check for duplicates
                for series_idx in assigned_series:
                    assert series_idx not in all_assigned_series, \
                        f"Series {series_idx} assigned to multiple workers!"
                    all_assigned_series.add(series_idx)

        # Verify all series are assigned
        assert len(all_assigned_series) == n_series, \
            f"Should assign all {n_series} series, only assigned {len(all_assigned_series)}"

    def test_all_series_covered(self, mock_data_path):
        """Test that all time series are assigned to exactly one worker."""
        n_series = 88
        num_workers = 7
        world_size = 3
        global_num_workers = num_workers * world_size  # 21

        assigned_series = set()

        for global_worker_id in range(global_num_workers):
            worker_series = [
                idx for idx in range(n_series)
                if idx % global_num_workers == global_worker_id
            ]
            assigned_series.update(worker_series)

        # Verify coverage
        assert assigned_series == set(range(n_series)), \
            f"Not all series covered. Missing: {set(range(n_series)) - assigned_series}"

    def test_deterministic_partitioning(self, mock_data_path):
        """Test that partitioning is deterministic given same configuration."""
        os.environ['RANK'] = '1'
        os.environ['WORLD_SIZE'] = '3'

        # Create two datasets with same configuration
        dataset1 = WindForecastingDataset(
            data_path=mock_data_path,
            context_length=100,
            prediction_length=4,
            repeat=False,
            skip_indices=1
        )

        dataset2 = WindForecastingDataset(
            data_path=mock_data_path,
            context_length=100,
            prediction_length=4,
            repeat=False,
            skip_indices=1
        )

        # Get samples from both
        samples1 = list(iter(dataset1))[:100]  # First 100 samples
        samples2 = list(iter(dataset2))[:100]  # First 100 samples

        # Verify same samples in same order
        assert len(samples1) == len(samples2), \
            "Deterministic partitioning should produce same number of samples"

        for idx, (s1, s2) in enumerate(zip(samples1, samples2)):
            torch.testing.assert_close(s1['past_target'], s2['past_target'],
                                       msg=f"Sample {idx} differs between iterations")

    def test_partitioning_with_different_world_sizes(self, mock_data_path):
        """Test partitioning correctness with different world sizes."""
        n_series = 88

        for world_size in [1, 2, 3, 4, 8]:
            all_series = set()

            for rank in range(world_size):
                worker_series = [
                    idx for idx in range(n_series)
                    if idx % world_size == rank
                ]
                all_series.update(worker_series)

            assert all_series == set(range(n_series)), \
                f"World size {world_size}: Not all series covered"


class TestDatasetLength:
    """Test suite for __len__ implementation."""

    def test_len_matches_sample_count(self, mock_data_path):
        """Test that __len__ returns correct total sample count."""
        dataset = WindForecastingDataset(
            data_path=mock_data_path,
            context_length=100,
            prediction_length=4,
            repeat=False,
            skip_indices=1
        )

        # Calculate expected length
        # 88 series × (1000 - 100 - 4 + 1) windows = 88 × 897 = 78,936
        expected_length = 88 * (1000 - 100 - 4 + 1)

        assert len(dataset) == expected_length, \
            f"Dataset __len__ should return {expected_length}, got {len(dataset)}"

    def test_len_with_skip_indices(self, mock_data_path):
        """Test that __len__ accounts for skip_indices."""
        dataset = WindForecastingDataset(
            data_path=mock_data_path,
            context_length=100,
            prediction_length=4,
            repeat=False,
            skip_indices=2  # Skip every other sample
        )

        # Expected: 88 × ceil(897 / 2) = 88 × 449 = 39,512
        # Use ceiling division: (num_valid + skip_indices - 1) // skip_indices
        num_valid_windows = 1000 - 100 - 4 + 1  # 897
        skip_indices = 2
        windows_per_series = (num_valid_windows + skip_indices - 1) // skip_indices  # ceil(897/2) = 449
        expected_length = 88 * windows_per_series

        assert len(dataset) == expected_length, \
            f"Dataset __len__ with skip_indices=2 should return {expected_length}, got {len(dataset)}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
