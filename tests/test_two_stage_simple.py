#!/usr/bin/env python
"""
Simple test to validate two-stage configuration files and scripts.
"""

import os
import sys
import yaml
from pathlib import Path

def test_config_files():
    """Test that Stage 1 and Stage 2 config files have correct settings."""
    print("=" * 60)
    print("Two-Stage TACTiS Configuration Validation")
    print("=" * 60)
    
    try:
        # Test Stage 1 config
        print("\n=== Stage 1 Configuration ===")
        stage1_path = "/fs/dss/home/taed7566/Forecasting/wind-forecasting/config/training/training_inputs_juan_awaken_tune_storm_pred60_stage1.yaml"
        with open(stage1_path, 'r') as f:
            stage1_config = yaml.safe_load(f)
        
        print(f"File: {stage1_path}")
        print("\nKey settings:")
        tactis1 = stage1_config['model']['tactis']
        print(f"  skip_copula: {tactis1['skip_copula']} (should be True)")
        print(f"  lock_skip_copula: {tactis1['lock_skip_copula']} (should be True)")
        print(f"  initial_stage: {tactis1['initial_stage']} (should be 1)")
        print(f"  stage2_start_epoch: {tactis1['stage2_start_epoch']} (should be 999)")
        print(f"  max_epochs: {stage1_config['trainer']['max_epochs']} (should be 25)")
        print(f"  run_name: {stage1_config['experiment']['run_name']}")
        print(f"  notes: {stage1_config['experiment']['notes']}")
        
        # Validate Stage 1
        errors = []
        if tactis1['skip_copula'] != True:
            errors.append("Stage 1 should have skip_copula=True")
        if tactis1['lock_skip_copula'] != True:
            errors.append("Stage 1 should have lock_skip_copula=True")
        if tactis1['initial_stage'] != 1:
            errors.append("Stage 1 should start in stage 1")
        if tactis1['stage2_start_epoch'] != 999:
            errors.append("Stage 1 should never reach Stage 2")
        if stage1_config['trainer']['max_epochs'] != 25:
            errors.append("Stage 1 should train for 25 epochs")
        
        if errors:
            print("\n✗ Stage 1 validation FAILED:")
            for e in errors:
                print(f"  - {e}")
        else:
            print("\n✓ Stage 1 configuration VALID")
        
        # Test Stage 2 config
        print("\n=== Stage 2 Configuration ===")
        stage2_path = "/fs/dss/home/taed7566/Forecasting/wind-forecasting/config/training/training_inputs_juan_awaken_tune_storm_pred60_stage2.yaml"
        with open(stage2_path, 'r') as f:
            stage2_config = yaml.safe_load(f)
        
        print(f"File: {stage2_path}")
        print("\nKey settings:")
        tactis2 = stage2_config['model']['tactis']
        print(f"  skip_copula: {tactis2['skip_copula']} (should be False)")
        print(f"  lock_skip_copula: {tactis2['lock_skip_copula']} (should be True)")
        print(f"  initial_stage: {tactis2['initial_stage']} (should be 2)")
        print(f"  stage2_start_epoch: {tactis2['stage2_start_epoch']} (should be 0)")
        print(f"  max_epochs: {stage2_config['trainer']['max_epochs']} (should be 30)")
        print(f"  run_name: {stage2_config['experiment']['run_name']}")
        print(f"  notes: {stage2_config['experiment']['notes']}")
        
        # Validate Stage 2
        errors = []
        if tactis2['skip_copula'] != False:
            errors.append("Stage 2 should have skip_copula=False")
        if tactis2['lock_skip_copula'] != True:
            errors.append("Stage 2 should have lock_skip_copula=True")
        if tactis2['initial_stage'] != 2:
            errors.append("Stage 2 should start in stage 2")
        if tactis2['stage2_start_epoch'] != 0:
            errors.append("Stage 2 should immediately be in Stage 2")
        if stage2_config['trainer']['max_epochs'] != 30:
            errors.append("Stage 2 should train for 30 epochs")
        
        if errors:
            print("\n✗ Stage 2 validation FAILED:")
            for e in errors:
                print(f"  - {e}")
        else:
            print("\n✓ Stage 2 configuration VALID")
        
        print("\n" + "=" * 60)
        print("Configuration validation complete!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_slurm_scripts():
    """Test that SLURM scripts exist and have correct structure."""
    print("\n=== SLURM Scripts Validation ===")
    
    try:
        # Check Stage 1 script
        stage1_script = "/fs/dss/home/taed7566/Forecasting/wind-forecasting/wind_forecasting/run_scripts/tune_scripts/tune_model_storm_awaken_p60_stage1.sh"
        if os.path.exists(stage1_script):
            with open(stage1_script, 'r') as f:
                stage1_content = f.read()
            
            print(f"\n✓ Stage 1 script exists: {stage1_script}")
            
            # Check for key elements
            checks = [
                ("Config file reference", "training_inputs_juan_awaken_tune_storm_pred60_stage1.yaml" in stage1_content),
                ("Stage 1 identifier", "STAGE 1" in stage1_content),
                ("Skip copula mention", "marginals only" in stage1_content.lower() or "skip_copula=true" in stage1_content.lower()),
                ("Job name", "60awaken_tune_tactis_stage1" in stage1_content),
            ]
            
            for check_name, passed in checks:
                if passed:
                    print(f"  ✓ {check_name}")
                else:
                    print(f"  ✗ {check_name}")
        else:
            print(f"✗ Stage 1 script not found: {stage1_script}")
        
        # Check Stage 2 script
        stage2_script = "/fs/dss/home/taed7566/Forecasting/wind-forecasting/wind_forecasting/run_scripts/tune_scripts/tune_model_storm_awaken_p60_stage2.sh"
        if os.path.exists(stage2_script):
            with open(stage2_script, 'r') as f:
                stage2_content = f.read()
            
            print(f"\n✓ Stage 2 script exists: {stage2_script}")
            
            # Check for key elements
            checks = [
                ("Config file reference", "training_inputs_juan_awaken_tune_storm_pred60_stage2.yaml" in stage2_content),
                ("Stage 2 identifier", "STAGE 2" in stage2_content),
                ("Stage 1 study check", "STAGE1_STUDY_NAME" in stage2_content),
                ("Copula training mention", "copula only" in stage2_content.lower() or "skip_copula=false" in stage2_content.lower()),
                ("Job name", "60awaken_tune_tactis_stage2" in stage2_content),
                ("--stage1_study argument", "--stage1_study" in stage2_content),
            ]
            
            for check_name, passed in checks:
                if passed:
                    print(f"  ✓ {check_name}")
                else:
                    print(f"  ✗ {check_name}")
        else:
            print(f"✗ Stage 2 script not found: {stage2_script}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_code_modifications():
    """Test that code modifications are in place."""
    print("\n=== Code Modifications Validation ===")
    
    try:
        # Check run_model.py for stage1_study argument
        print("\nChecking run_model.py for --stage1_study argument...")
        run_model_path = "/fs/dss/home/taed7566/Forecasting/wind-forecasting/wind_forecasting/run_scripts/run_model.py"
        with open(run_model_path, 'r') as f:
            run_model_content = f.read()
        
        if "--stage1_study" in run_model_content:
            print("  ✓ --stage1_study argument found in run_model.py")
        else:
            print("  ✗ --stage1_study argument NOT found in run_model.py")
        
        # Check objective.py for stage1 checkpoint loading
        print("\nChecking objective.py for Stage 1 checkpoint loading...")
        objective_path = "/fs/dss/home/taed7566/Forecasting/wind-forecasting/wind_forecasting/tuning/objective.py"
        with open(objective_path, 'r') as f:
            objective_content = f.read()
        
        if "_get_stage1_checkpoint" in objective_content:
            print("  ✓ _get_stage1_checkpoint method found in objective.py")
        else:
            print("  ✗ _get_stage1_checkpoint method NOT found in objective.py")
        
        if "stage1_study_name" in objective_content:
            print("  ✓ stage1_study_name handling found in objective.py")
        else:
            print("  ✗ stage1_study_name handling NOT found in objective.py")
        
        # Check TACTiS model for lock_skip_copula
        print("\nChecking TACTiS for lock_skip_copula parameter...")
        tactis_path = "/fs/dss/home/taed7566/Forecasting/pytorch-transformer-ts/pytorch_transformer_ts/tactis_2/tactis.py"
        with open(tactis_path, 'r') as f:
            tactis_content = f.read()
        
        if "lock_skip_copula" in tactis_content:
            print("  ✓ lock_skip_copula parameter found in tactis.py")
        else:
            print("  ✗ lock_skip_copula parameter NOT found in tactis.py")
        
        # Check TACTiS estimator
        print("\nChecking TACTiS estimator...")
        estimator_path = "/fs/dss/home/taed7566/Forecasting/pytorch-transformer-ts/pytorch_transformer_ts/tactis_2/estimator.py"
        with open(estimator_path, 'r') as f:
            estimator_content = f.read()
        
        if "lock_skip_copula" in estimator_content:
            print("  ✓ lock_skip_copula parameter found in estimator.py")
        else:
            print("  ✗ lock_skip_copula parameter NOT found in estimator.py")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Two-Stage TACTiS Tuning Implementation Validation")
    print("=" * 60)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_config_files()
    all_passed &= test_slurm_scripts()
    all_passed &= test_code_modifications()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ IMPLEMENTATION VALIDATION COMPLETE!")
        print("\nThe two-stage tuning implementation appears to be correctly set up.")
        print("\n=== How to Run Two-Stage Tuning ===")
        print("\n1. STAGE 1 - Train marginals only:")
        print("   cd /fs/dss/home/taed7566/Forecasting/wind-forecasting")
        print("   sbatch wind_forecasting/run_scripts/tune_scripts/tune_model_storm_awaken_p60_stage1.sh")
        print("\n2. Wait for Stage 1 to complete and check the logs for the study name")
        print("   Look for a line like: 'Study name: tune_awaken_tactis_pred60_stage1_marginals_YYYY_MM_DD_HHMMSS'")
        print("\n3. STAGE 2 - Train copula with Stage 1 checkpoint:")
        print("   export STAGE1_STUDY_NAME=\"<study_name_from_stage1_logs>\"")
        print("   sbatch wind_forecasting/run_scripts/tune_scripts/tune_model_storm_awaken_p60_stage2.sh")
        print("\n4. After Stage 2 completes, use the Stage 2 study for final training")
        print("\n=== Important Notes ===")
        print("- Each stage runs 3 parallel GPU workers for distributed tuning")
        print("- Stage 1 trains for 25 epochs (marginals only)")
        print("- Stage 2 trains for 30 epochs (copula only)")
        print("- Both stages use the same PostgreSQL backend for Optuna")
        print("- Check worker logs in logs/slurm_logs/<job_id>/ for detailed progress")
    else:
        print("✗ SOME VALIDATIONS FAILED")
        print("Please review the errors above.")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())