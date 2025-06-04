#!/usr/bin/env python3
"""
Quick validation test for the refactored tuning.py functionality.

This script performs essential checks to ensure the refactoring is working correctly
without running full integration tests that might interfere with production.
"""

import sys
import os
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the wind-forecasting module to Python path
sys.path.insert(0, '/fs/dss/home/taed7566/Forecasting/wind-forecasting')

def test_imports():
    """Test that all refactored modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        # Test utility module imports
        from wind_forecasting.tuning.path_utils import resolve_path, flatten_dict
        from wind_forecasting.tuning.config_utils import generate_db_setup_params, generate_optuna_dashboard_command
        from wind_forecasting.tuning.checkpoint_utils import (
            load_checkpoint, parse_epoch_from_checkpoint_path, determine_tactis_stage,
            extract_hyperparameters, prepare_model_init_args, load_model_state, set_tactis_stage
        )
        from wind_forecasting.tuning.metrics_utils import (
            extract_metric_value, compute_evaluation_metrics,
            update_metrics_with_checkpoint_score, validate_metrics_for_return
        )
        from wind_forecasting.tuning.helpers import (
            set_trial_seeds, update_data_module_params, regenerate_data_splits,
            prepare_feedforward_params, calculate_dynamic_limit_train_batches,
            create_trial_checkpoint_callback, setup_trial_callbacks
        )
        from wind_forecasting.tuning.callbacks import SafePruningCallback
        
        # Test main tuning module import
        from wind_forecasting.tuning import MLTuningObjective, tune_model
        
        logger.info("✓ All imports successful")
        return True
        
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of utility functions."""
    logger.info("Testing basic functionality...")
    
    try:
        from wind_forecasting.tuning.path_utils import resolve_path, flatten_dict
        from wind_forecasting.tuning.helpers import set_trial_seeds, calculate_dynamic_limit_train_batches
        from wind_forecasting.tuning.checkpoint_utils import parse_epoch_from_checkpoint_path, determine_tactis_stage
        from wind_forecasting.tuning.metrics_utils import extract_metric_value
        
        # Test path utilities
        path_result = resolve_path("/base", "relative/path")
        assert path_result == "/base/relative/path", f"Expected '/base/relative/path', got '{path_result}'"
        
        dict_result = flatten_dict({"a": {"b": {"c": "value"}}})
        expected = {"a.b.c": "value"}
        assert dict_result == expected, f"Expected {expected}, got {dict_result}"
        
        # Test seed setting
        seed_result = set_trial_seeds(5, 42)
        assert seed_result == 47, f"Expected 47, got {seed_result}"
        
        # Test dynamic batch calculation
        params = {'batch_size': 64}
        config = {'dataset': {'batch_size': 32}}
        batch_result = calculate_dynamic_limit_train_batches(params, config, 100, 32)
        assert batch_result == 50, f"Expected 50, got {batch_result}"
        
        # Test checkpoint path parsing
        checkpoint_path = "trial_1_epoch=15-step=300-val_loss=0.25.ckpt"
        epoch_result = parse_epoch_from_checkpoint_path(checkpoint_path, 1)
        assert epoch_result == 15, f"Expected 15, got {epoch_result}"
        
        # Test TACTiS stage determination
        stage_result = determine_tactis_stage(8, 5, 1)
        assert stage_result == 2, f"Expected 2, got {stage_result}"
        
        # Test metric extraction
        import torch
        metrics = {"val_loss": torch.tensor(0.5)}
        metric_result = extract_metric_value(metrics, "val_loss", 1)
        assert abs(metric_result - 0.5) < 1e-6, f"Expected 0.5, got {metric_result}"
        
        logger.info("✓ All basic functionality tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Basic functionality test failed: {e}")
        return False

def test_config_generation():
    """Test configuration generation functions."""
    logger.info("Testing configuration generation...")
    
    try:
        from wind_forecasting.tuning.config_utils import generate_db_setup_params, generate_optuna_dashboard_command
        
        # Test database setup parameters
        config = {
            "experiment": {"run_name": "test_run", "project_root": "/tmp"},
            "logging": {"optuna_dir": "optuna"},
            "optuna": {
                "storage": {
                    "backend": "postgresql",
                    "pgdata_instance_name": "test_instance",
                    "socket_dir_base": "${logging.optuna_dir}/sockets",
                    "sync_dir": "${logging.optuna_dir}/sync",
                    "database_name": "optuna_test"
                }
            }
        }
        
        db_params = generate_db_setup_params("tactis", config)
        
        # Debug: print actual keys
        logger.info(f"Generated db_params keys: {list(db_params.keys())}")
        
        # Verify expected keys are present
        expected_keys = ["backend", "project_root", "pgdata_path", "socket_dir_base", 
                        "sync_dir", "pgdata_instance_name", "base_study_prefix"]
        
        for key in expected_keys:
            assert key in db_params, f"Missing key in db_params: {key}"
        
        # Check for backend-specific keys
        if db_params["backend"] == "postgresql":
            # For PostgreSQL, optuna_db_name should be present
            assert "optuna_db_name" in db_params, "Missing optuna_db_name for PostgreSQL backend"
        
        assert db_params["backend"] == "postgresql"
        assert "test_instance" in db_params["pgdata_path"]
        assert "tuning_tactis_test_run" in db_params["base_study_prefix"]
        assert db_params["optuna_db_name"] == "optuna_test"
        
        # Test dashboard command generation
        dashboard_cmd = generate_optuna_dashboard_command(db_params, "test_study")
        logger.info(f"Generated dashboard command: {dashboard_cmd}")
        assert "optuna-monitor" in dashboard_cmd
        assert "postgresql" in dashboard_cmd
        assert "test_study" in dashboard_cmd
        assert "optuna_test" in dashboard_cmd  # Check for the database name
        
        logger.info("✓ Configuration generation tests passed")
        return True
        
    except Exception as e:
        import traceback
        logger.error(f"✗ Configuration generation test failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_mltunin_objective_creation():
    """Test MLTuningObjective can be created without errors."""
    logger.info("Testing MLTuningObjective creation...")
    
    try:
        from wind_forecasting.tuning import MLTuningObjective
        from unittest.mock import Mock
        
        # Create minimal mock configuration
        config = {
            "trainer": {"max_epochs": 5, "limit_train_batches": 10, "monitor_metric": "val_loss"},
            "dataset": {"batch_size": 32, "prediction_length": 600, "context_length_factor": 2},
            "optuna": {"direction": "minimize"},
            "model": {"tactis": {"stage2_start_epoch": 3}},
            "logging": {"checkpoint_dir": "/tmp", "chkp_dir_suffix": "_test"},
            "experiment": {"run_name": "test"}
        }
        
        # Mock required components
        lightning_module_class = Mock()
        estimator_class = Mock()
        distr_output_class = Mock()
        data_module = Mock()
        data_module.freq = "60s"
        data_module.prediction_length = 10
        data_module.context_length = 20
        data_module.per_turbine_target = False
        data_module.batch_size = 32
        
        # Create objective (should not raise errors)
        objective = MLTuningObjective(
            model="tactis",
            config=config,
            lightning_module_class=lightning_module_class,
            estimator_class=estimator_class,
            distr_output_class=distr_output_class,
            max_epochs=5,
            limit_train_batches=10,
            data_module=data_module,
            metric="val_loss",
            seed=42
        )
        
        # Verify basic attributes
        assert objective.model == "tactis"
        assert objective.seed == 42
        assert objective.config["trainer"]["max_epochs"] == 5
        
        logger.info("✓ MLTuningObjective creation test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ MLTuningObjective creation test failed: {e}")
        return False

def test_file_structure():
    """Test that all expected files exist with correct structure."""
    logger.info("Testing file structure...")
    
    try:
        base_path = "/fs/dss/home/taed7566/Forecasting/wind-forecasting"
        
        # Check main tuning subpackage files
        tuning_files = [
            "wind_forecasting/tuning/__init__.py",
            "wind_forecasting/tuning/core.py",
            "wind_forecasting/tuning/objective.py",
            "wind_forecasting/tuning/utils.py"
        ]
        
        for tuning_file in tuning_files:
            full_path = os.path.join(base_path, tuning_file)
            assert os.path.exists(full_path), f"Tuning subpackage file not found: {full_path}"
        
        # Check remaining general utility modules (tuning-specific moved to tuning subpackage)
        util_files = [
            "wind_forecasting/tuning/path_utils.py",
            "wind_forecasting/tuning/checkpoint_utils.py",
            "wind_forecasting/tuning/metrics_utils.py", 
            "wind_forecasting/tuning/storage.py",
            "wind_forecasting/tuning/helpers.py",
            "wind_forecasting/tuning/config_utils.py",
            "wind_forecasting/tuning/trial_utils.py",
            "wind_forecasting/utils/callbacks.py"
        ]
        
        for util_file in util_files:
            full_path = os.path.join(base_path, util_file)
            assert os.path.exists(full_path), f"Utility file not found: {full_path}"
        
        # Check file sizes (should be non-empty)
        all_files = [os.path.join(base_path, f) for f in tuning_files + util_files]
        for file_path in all_files:
            size = os.path.getsize(file_path)
            assert size > 0, f"File is empty: {file_path}"
        
        logger.info("✓ File structure test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ File structure test failed: {e}")
        return False

def run_all_tests():
    """Run all validation tests."""
    logger.info("Starting refactor validation tests...")
    
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Configuration Generation", test_config_generation),
        ("MLTuningObjective Creation", test_mltunin_objective_creation),
    ]
    
    results = {}
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} Test ---")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed += 1
        except Exception as e:
            logger.error(f"✗ {test_name} test encountered an error: {e}")
            results[test_name] = False
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*60)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Refactored tuning.py is ready for use.")
        return True
    else:
        logger.error("⚠️  Some tests failed. Review the output above for details.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)