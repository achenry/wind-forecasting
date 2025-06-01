#!/usr/bin/env python3
"""
Validation script for the refactored tuning.py functionality.

This script runs comprehensive tests to ensure the refactored code maintains
identical functionality to the original implementation. It's designed to be
safe and not interfere with production tuning jobs.
"""

import os
import sys
import subprocess
import tempfile
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check that all required dependencies are available."""
    try:
        import pytest
        import torch
        import numpy as np
        import pandas as pd
        logger.info("✓ All required dependencies found")
        return True
    except ImportError as e:
        logger.error(f"✗ Missing dependency: {e}")
        return False

def run_unit_tests():
    """Run unit tests for individual utility modules."""
    logger.info("Running unit tests for utility modules...")
    
    test_files = [
        "tests/test_tuning_refactor.py::TestPathUtils",
        "tests/test_tuning_refactor.py::TestTuningConfigUtils", 
        "tests/test_tuning_refactor.py::TestCheckpointUtils",
        "tests/test_tuning_refactor.py::TestMetricsUtils",
        "tests/test_tuning_refactor.py::TestTuningHelpers",
        "tests/test_tuning_refactor.py::TestSafePruningCallback"
    ]
    
    results = {}
    for test_class in test_files:
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", test_class, "-v", "--tb=short"
            ], capture_output=True, text=True, cwd="/fs/dss/home/taed7566/Forecasting/wind-forecasting")
            
            results[test_class] = {
                "success": result.returncode == 0,
                "output": result.stdout,
                "errors": result.stderr
            }
            
            if result.returncode == 0:
                logger.info(f"✓ {test_class.split('::')[-1]} passed")
            else:
                logger.error(f"✗ {test_class.split('::')[-1]} failed")
                
        except Exception as e:
            logger.error(f"✗ Error running {test_class}: {e}")
            results[test_class] = {"success": False, "error": str(e)}
    
    return results

def run_integration_tests():
    """Run integration tests to verify end-to-end functionality."""
    logger.info("Running integration tests...")
    
    test_files = [
        "tests/test_tuning_integration.py::TestOriginalVsRefactoredBehavior",
        "tests/test_tuning_integration.py::TestProductionScenarios"
    ]
    
    results = {}
    for test_class in test_files:
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", test_class, "-v", "--tb=short"
            ], capture_output=True, text=True, cwd="/fs/dss/home/taed7566/Forecasting/wind-forecasting")
            
            results[test_class] = {
                "success": result.returncode == 0,
                "output": result.stdout,
                "errors": result.stderr
            }
            
            if result.returncode == 0:
                logger.info(f"✓ {test_class.split('::')[-1]} passed")
            else:
                logger.error(f"✗ {test_class.split('::')[-1]} failed")
                
        except Exception as e:
            logger.error(f"✗ Error running {test_class}: {e}")
            results[test_class] = {"success": False, "error": str(e)}
    
    return results

def validate_import_structure():
    """Validate that all imports work correctly in the refactored code."""
    logger.info("Validating import structure...")
    
    try:
        # Test importing the main tuning module
        sys.path.insert(0, '/fs/dss/home/taed7566/Forecasting/wind-forecasting')
        
        from wind_forecasting.run_scripts.tuning import MLTuningObjective, tune_model
        from wind_forecasting.utils.path_utils import resolve_path, flatten_dict
        from wind_forecasting.utils.tuning_config_utils import generate_db_setup_params
        from wind_forecasting.utils.checkpoint_utils import load_checkpoint
        from wind_forecasting.utils.metrics_utils import extract_metric_value
        from wind_forecasting.utils.tuning_helpers import set_trial_seeds
        from wind_forecasting.utils.callbacks import SafePruningCallback
        
        logger.info("✓ All imports successful")
        return True
        
    except ImportError as e:
        logger.error(f"✗ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error during import: {e}")
        return False

def validate_function_signatures():
    """Validate that function signatures match expected interfaces."""
    logger.info("Validating function signatures...")
    
    try:
        import inspect
        from wind_forecasting.utils.path_utils import resolve_path, flatten_dict
        from wind_forecasting.utils.tuning_config_utils import generate_db_setup_params
        
        # Check key function signatures
        sig_resolve_path = inspect.signature(resolve_path)
        expected_params = ['base_path', 'path_input']
        actual_params = list(sig_resolve_path.parameters.keys())
        assert expected_params == actual_params, f"resolve_path signature mismatch: {actual_params}"
        
        sig_flatten_dict = inspect.signature(flatten_dict)
        expected_params = ['d', 'parent_key', 'sep']
        actual_params = list(sig_flatten_dict.parameters.keys())
        assert expected_params == actual_params, f"flatten_dict signature mismatch: {actual_params}"
        
        sig_generate_db = inspect.signature(generate_db_setup_params)
        expected_params = ['model', 'model_config']
        actual_params = list(sig_generate_db.parameters.keys())
        assert expected_params == actual_params, f"generate_db_setup_params signature mismatch: {actual_params}"
        
        logger.info("✓ All function signatures validated")
        return True
        
    except Exception as e:
        logger.error(f"✗ Function signature validation failed: {e}")
        return False

def run_smoke_test():
    """Run a minimal smoke test to ensure basic functionality works."""
    logger.info("Running smoke test...")
    
    try:
        from wind_forecasting.utils.path_utils import resolve_path, flatten_dict
        from wind_forecasting.utils.tuning_helpers import set_trial_seeds
        
        # Test basic functionality
        path_result = resolve_path("/base", "relative")
        assert path_result == "/base/relative"
        
        dict_result = flatten_dict({"a": {"b": "value"}})
        assert dict_result == {"a.b": "value"}
        
        # Test seed setting (should not raise exceptions)
        seed_result = set_trial_seeds(1, 42)
        assert seed_result == 43
        
        logger.info("✓ Smoke test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Smoke test failed: {e}")
        return False

def check_file_structure():
    """Check that all expected files exist."""
    logger.info("Checking file structure...")
    
    base_path = Path("/fs/dss/home/taed7566/Forecasting/wind-forecasting")
    
    expected_files = [
        "wind_forecasting/run_scripts/tuning.py",
        "wind_forecasting/utils/path_utils.py",
        "wind_forecasting/utils/tuning_config_utils.py",
        "wind_forecasting/utils/checkpoint_utils.py",
        "wind_forecasting/utils/metrics_utils.py",
        "wind_forecasting/utils/tuning_helpers.py",
        "wind_forecasting/utils/callbacks.py"
    ]
    
    missing_files = []
    for file_path in expected_files:
        full_path = base_path / file_path
        if not full_path.exists():
            missing_files.append(str(full_path))
    
    if missing_files:
        logger.error(f"✗ Missing files: {missing_files}")
        return False
    
    logger.info("✓ All expected files found")
    return True

def generate_report(results):
    """Generate a comprehensive validation report."""
    logger.info("Generating validation report...")
    
    report = []
    report.append("=" * 60)
    report.append("TUNING REFACTOR VALIDATION REPORT")
    report.append("=" * 60)
    report.append("")
    
    total_tests = 0
    passed_tests = 0
    
    for category, category_results in results.items():
        report.append(f"{category.upper()}:")
        report.append("-" * 40)
        
        if isinstance(category_results, dict):
            for test_name, test_result in category_results.items():
                if isinstance(test_result, dict) and 'success' in test_result:
                    status = "PASS" if test_result['success'] else "FAIL"
                    test_display_name = test_name.split('::')[-1] if '::' in test_name else test_name
                    report.append(f"  {test_display_name}: {status}")
                    total_tests += 1
                    if test_result['success']:
                        passed_tests += 1
        else:
            status = "PASS" if category_results else "FAIL"
            report.append(f"  {category}: {status}")
            total_tests += 1
            if category_results:
                passed_tests += 1
        
        report.append("")
    
    report.append("SUMMARY:")
    report.append("-" * 40)
    report.append(f"Total tests: {total_tests}")
    report.append(f"Passed: {passed_tests}")
    report.append(f"Failed: {total_tests - passed_tests}")
    report.append(f"Success rate: {(passed_tests / total_tests * 100):.1f}%" if total_tests > 0 else "N/A")
    report.append("")
    
    if passed_tests == total_tests:
        report.append("🎉 ALL TESTS PASSED - Refactoring validation successful!")
        report.append("The refactored tuning.py is ready for production use.")
    else:
        report.append("⚠️  SOME TESTS FAILED - Review required before production use.")
        report.append("Check the detailed output above for specific issues.")
    
    report.append("=" * 60)
    
    return "\n".join(report)

def main():
    """Main validation function."""
    logger.info("Starting tuning refactor validation...")
    
    # Check dependencies first
    if not check_dependencies():
        logger.error("Dependency check failed. Please install missing packages.")
        return False
    
    # Collect all validation results
    results = {}
    
    # Basic checks
    results['file_structure'] = check_file_structure()
    results['import_structure'] = validate_import_structure()
    results['function_signatures'] = validate_function_signatures()
    results['smoke_test'] = run_smoke_test()
    
    # Run comprehensive tests
    results['unit_tests'] = run_unit_tests()
    results['integration_tests'] = run_integration_tests()
    
    # Generate and display report
    report = generate_report(results)
    print(report)
    
    # Write report to file
    report_file = "/fs/dss/home/taed7566/Forecasting/wind-forecasting/tuning_validation_report.txt"
    try:
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"Validation report saved to: {report_file}")
    except Exception as e:
        logger.warning(f"Could not save report to file: {e}")
    
    # Return overall success
    all_basic_checks = all([
        results['file_structure'],
        results['import_structure'], 
        results['function_signatures'],
        results['smoke_test']
    ])
    
    unit_test_success = all(test['success'] for test in results['unit_tests'].values() if isinstance(test, dict))
    integration_test_success = all(test['success'] for test in results['integration_tests'].values() if isinstance(test, dict))
    
    overall_success = all_basic_checks and unit_test_success and integration_test_success
    
    if overall_success:
        logger.info("🎉 VALIDATION SUCCESSFUL - Refactored tuning.py is ready!")
    else:
        logger.error("⚠️  VALIDATION FAILED - Issues need to be resolved")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)