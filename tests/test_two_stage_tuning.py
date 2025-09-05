#!/usr/bin/env python
"""
Test script for two-stage TACTiS tuning implementation.
This validates that the lock_skip_copula mechanism and Stage 1/2 separation works correctly.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add wind-forecasting to path
sys.path.insert(0, '/fs/dss/home/taed7566/Forecasting/wind-forecasting')
sys.path.insert(0, '/fs/dss/home/taed7566/Forecasting/pytorch-transformer-ts')

def test_lock_skip_copula():
    """Test that lock_skip_copula parameter prevents automatic skip_copula updates."""
    print("\n=== Testing lock_skip_copula mechanism ===")
    
    try:
        from pytorch_transformer_ts.tactis_2.tactis import TACTiS
        import torch
        
        # Test 1: Without lock_skip_copula (default behavior)
        print("\nTest 1: Default behavior (lock_skip_copula=False)")
        model1 = TACTiS(
            prediction_length=10,
            context_length=20,
            distr_output=None,  # Will use default
            skip_copula=True,
            lock_skip_copula=False
        )
        
        print(f"  Initial skip_copula: {model1.skip_copula}")
        model1.set_stage(2)
        print(f"  After set_stage(2): {model1.skip_copula}")
        assert model1.skip_copula == False, "skip_copula should be False in Stage 2 without lock"
        print("  ✓ Automatic skip_copula update working")
        
        # Test 2: With lock_skip_copula (new behavior)
        print("\nTest 2: Locked behavior (lock_skip_copula=True)")
        model2 = TACTiS(
            prediction_length=10,
            context_length=20,
            distr_output=None,
            skip_copula=True,
            lock_skip_copula=True
        )
        
        print(f"  Initial skip_copula: {model2.skip_copula}")
        model2.set_stage(2)
        print(f"  After set_stage(2): {model2.skip_copula}")
        assert model2.skip_copula == True, "skip_copula should remain True when locked"
        print("  ✓ lock_skip_copula prevents automatic update")
        
        # Test 3: Stage 2 with skip_copula=False and lock
        print("\nTest 3: Stage 2 configuration (skip_copula=False, locked)")
        model3 = TACTiS(
            prediction_length=10,
            context_length=20,
            distr_output=None,
            skip_copula=False,
            lock_skip_copula=True,
            initial_stage=2
        )
        
        print(f"  Initial skip_copula: {model3.skip_copula}")
        print(f"  Initial stage: {model3.current_stage}")
        assert model3.skip_copula == False, "skip_copula should be False for Stage 2"
        assert model3.current_stage == 2, "Should start in Stage 2"
        print("  ✓ Stage 2 configuration correct")
        
        print("\n✓ All lock_skip_copula tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_stage1_study_argument():
    """Test that stage1_study argument is properly handled."""
    print("\n=== Testing stage1_study argument handling ===")
    
    try:
        from wind_forecasting.tuning.objective import MLTuningObjective
        from wind_forecasting.tuning.core import OptunaStudyConfig
        import optuna
        
        # Create a mock storage
        storage = optuna.storages.InMemoryStorage()
        
        # Create a mock Stage 1 study
        stage1_study_name = "test_stage1_study"
        stage1_study = optuna.create_study(
            study_name=stage1_study_name,
            storage=storage,
            direction="minimize"
        )
        
        # Add a trial to Stage 1
        def objective(trial):
            return trial.suggest_float("test_param", 0.0, 1.0)
        
        stage1_study.optimize(objective, n_trials=1)
        
        print(f"  Created Stage 1 study: {stage1_study_name}")
        print(f"  Stage 1 best value: {stage1_study.best_value}")
        
        # Test that MLTuningObjective can handle stage1_study_name
        print("\n  Testing MLTuningObjective with stage1_study_name...")
        
        # Create minimal config
        class MockConfig:
            def __init__(self):
                self.optuna = OptunaStudyConfig(
                    n_trials_per_worker=1,
                    total_study_trials=1,
                    metric="val_loss",
                    direction="minimize"
                )
                self.logging = type('obj', (object,), {
                    'checkpoint_dir': tempfile.mkdtemp(),
                    'chkp_dir_suffix': '_test'
                })()
                self.model = type('obj', (object,), {
                    'tactis': {
                        'skip_copula': False,
                        'lock_skip_copula': True,
                        'initial_stage': 2
                    }
                })()
        
        config = MockConfig()
        
        # This should not raise an error
        objective_fn = MLTuningObjective(
            config=config,
            model_name="tactis",
            seed=42,
            storage=storage,
            stage1_study_name=stage1_study_name,
            limit_train_batches=100,
            val_check_interval=1.0,
            num_batches_per_epoch=100
        )
        
        print("  ✓ MLTuningObjective accepts stage1_study_name")
        
        # Clean up
        shutil.rmtree(config.logging.checkpoint_dir)
        
        print("\n✓ All stage1_study tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_files():
    """Test that Stage 1 and Stage 2 config files have correct settings."""
    print("\n=== Testing configuration files ===")
    
    try:
        import yaml
        
        # Test Stage 1 config
        print("\nTest Stage 1 configuration:")
        stage1_path = "/fs/dss/home/taed7566/Forecasting/wind-forecasting/config/training/training_inputs_juan_awaken_tune_storm_pred60_stage1.yaml"
        with open(stage1_path, 'r') as f:
            stage1_config = yaml.safe_load(f)
        
        tactis1 = stage1_config['model']['tactis']
        assert tactis1['skip_copula'] == True, "Stage 1 should have skip_copula=True"
        assert tactis1['lock_skip_copula'] == True, "Stage 1 should have lock_skip_copula=True"
        assert tactis1['initial_stage'] == 1, "Stage 1 should start in stage 1"
        assert tactis1['stage2_start_epoch'] == 999, "Stage 1 should never reach Stage 2"
        assert stage1_config['trainer']['max_epochs'] == 25, "Stage 1 should train for 25 epochs"
        print("  ✓ Stage 1 config correct")
        
        # Test Stage 2 config
        print("\nTest Stage 2 configuration:")
        stage2_path = "/fs/dss/home/taed7566/Forecasting/wind-forecasting/config/training/training_inputs_juan_awaken_tune_storm_pred60_stage2.yaml"
        with open(stage2_path, 'r') as f:
            stage2_config = yaml.safe_load(f)
        
        tactis2 = stage2_config['model']['tactis']
        assert tactis2['skip_copula'] == False, "Stage 2 should have skip_copula=False"
        assert tactis2['lock_skip_copula'] == True, "Stage 2 should have lock_skip_copula=True"
        assert tactis2['initial_stage'] == 2, "Stage 2 should start in stage 2"
        assert tactis2['stage2_start_epoch'] == 0, "Stage 2 should immediately be in Stage 2"
        assert stage2_config['trainer']['max_epochs'] == 30, "Stage 2 should train for 30 epochs"
        print("  ✓ Stage 2 config correct")
        
        print("\n✓ All configuration tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Two-Stage TACTiS Tuning Implementation Test Suite")
    print("=" * 60)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_lock_skip_copula()
    all_passed &= test_stage1_study_argument()
    all_passed &= test_config_files()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("\nThe two-stage tuning implementation is ready to use.")
        print("\nNext steps:")
        print("1. Submit Stage 1 tuning: sbatch tune_model_storm_awaken_p60_stage1.sh")
        print("2. Wait for Stage 1 to complete and note the study name from logs")
        print("3. Set STAGE1_STUDY_NAME environment variable")
        print("4. Submit Stage 2 tuning: sbatch tune_model_storm_awaken_p60_stage2.sh")
    else:
        print("✗ SOME TESTS FAILED")
        print("Please review the errors above and fix the implementation.")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())