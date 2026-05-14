"""
Unit tests for DataModule property accessors.

Tests the train_dataset, val_dataset, and test_dataset properties that provide
backward-compatible access to self.datasets["train"] etc.
"""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

# Import with patching to avoid full initialization
import sys
sys.path.insert(0, '/user/taed7566/Forecasting/wind-forecasting')


class TestDataModuleProperties:
    """Test suite for DataModule property accessors."""

    @pytest.fixture
    def mock_data_module(self):
        """Create a minimal DataModule instance for testing without full initialization."""
        from wind_forecasting.preprocessing.data_module import DataModule

        # Patch __post_init__ to avoid full initialization
        with patch.object(DataModule, '__post_init__', lambda self: None):
            dm = DataModule(
                data_path="/fake/path/data.parquet",
                n_splits=3,
                continuity_groups=None,
                train_split=0.7,
                val_split=0.15,
                test_split=0.15,
                prediction_length=60,
                context_length=120,
                target_prefixes=["ws_horz", "ws_vert"],
                target_suffixes=["wt001"],
                feat_dynamic_real_prefixes=["nd_cos", "nd_sin"],
                freq="10s",
                per_turbine_target=True,
            )
        return dm

    def test_train_dataset_property_returns_correct_value(self, mock_data_module):
        """Test that train_dataset property returns datasets['train']."""
        mock_train_data = [{"target": np.random.randn(3, 100)}]
        mock_data_module.datasets = {
            "train": mock_train_data,
            "val": [],
            "test": []
        }

        assert mock_data_module.train_dataset is mock_train_data

    def test_val_dataset_property_returns_correct_value(self, mock_data_module):
        """Test that val_dataset property returns datasets['val']."""
        mock_val_data = [{"target": np.random.randn(3, 50)}]
        mock_data_module.datasets = {
            "train": [],
            "val": mock_val_data,
            "test": []
        }

        assert mock_data_module.val_dataset is mock_val_data

    def test_test_dataset_property_returns_correct_value(self, mock_data_module):
        """Test that test_dataset property returns datasets['test']."""
        mock_test_data = [{"target": np.random.randn(3, 30)}]
        mock_data_module.datasets = {
            "train": [],
            "val": [],
            "test": mock_test_data
        }

        assert mock_data_module.test_dataset is mock_test_data

    def test_properties_return_none_when_datasets_not_set(self, mock_data_module):
        """Test that properties return None when datasets dict doesn't exist."""
        # datasets attribute not set (simulating state before generate_splits)
        assert mock_data_module.train_dataset is None
        assert mock_data_module.val_dataset is None
        assert mock_data_module.test_dataset is None

    def test_properties_return_none_when_datasets_is_none(self, mock_data_module):
        """Test that properties return None when datasets is explicitly None."""
        mock_data_module.datasets = None

        assert mock_data_module.train_dataset is None
        assert mock_data_module.val_dataset is None
        assert mock_data_module.test_dataset is None

    def test_properties_return_none_for_missing_keys(self, mock_data_module):
        """Test that properties return None when specific key is missing."""
        mock_data_module.datasets = {"train": []}  # val and test missing

        assert mock_data_module.train_dataset == []
        assert mock_data_module.val_dataset is None
        assert mock_data_module.test_dataset is None

    def test_properties_work_with_empty_lists(self, mock_data_module):
        """Test that properties correctly return empty lists."""
        mock_data_module.datasets = {
            "train": [],
            "val": [],
            "test": []
        }

        assert mock_data_module.train_dataset == []
        assert mock_data_module.val_dataset == []
        assert mock_data_module.test_dataset == []

    def test_properties_work_with_list_of_dicts(self, mock_data_module):
        """Test that properties work with list-of-dict format (non-lazyframe mode)."""
        train_data = [
            {"target": np.random.randn(3, 100), "item_id": "item_0"},
            {"target": np.random.randn(3, 100), "item_id": "item_1"},
        ]
        val_data = [
            {"target": np.random.randn(3, 50), "item_id": "item_0"},
        ]

        mock_data_module.datasets = {
            "train": train_data,
            "val": val_data,
            "test": []
        }

        assert mock_data_module.train_dataset is train_data
        assert len(mock_data_module.train_dataset) == 2
        assert mock_data_module.val_dataset is val_data
        assert len(mock_data_module.val_dataset) == 1


class TestWindForecastingDatamoduleTestDataset:
    """Test the test_dataset property in WindForecastingDatamodule."""

    def test_test_dataset_returns_none_by_default(self):
        """Test that test_dataset returns None when _test_dataset is not set."""
        from wind_forecasting.preprocessing.pytorch_dataset import WindForecastingDatamodule

        # Create a minimal mock to avoid full initialization
        with patch.object(WindForecastingDatamodule, '__init__', lambda self: None):
            dm = WindForecastingDatamodule.__new__(WindForecastingDatamodule)

        assert dm.test_dataset is None

    def test_test_dataset_returns_value_when_set(self):
        """Test that test_dataset returns _test_dataset when it's set."""
        from wind_forecasting.preprocessing.pytorch_dataset import WindForecastingDatamodule

        with patch.object(WindForecastingDatamodule, '__init__', lambda self: None):
            dm = WindForecastingDatamodule.__new__(WindForecastingDatamodule)
            dm._test_dataset = "mock_test_data"

        assert dm.test_dataset == "mock_test_data"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
