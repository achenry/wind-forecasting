"""
Test that Stage 2 tuning doesn't waste TPE modeling capacity on fixed parameters.

This test verifies:
1. Common architectural params are NOT suggested by Optuna when Stage 1 values exist
2. Fixed params are used directly from Stage 1
3. TPE only models the search space for parameters actually being tuned
4. Backward compatibility: Stage 1 and sequential training still work

Run with: pytest tests/test_stage2_tpe_efficiency.py -v
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pytorch_transformer_ts.tactis_2.estimator import TACTiS2Estimator


class TestStage2TPEEfficiency:
    """Test that fixed params don't waste TPE capacity."""

    def test_stage2_with_fixed_params_no_suggestions(self):
        """Verify that when stage1_fixed_params provided, common params are NOT suggested."""

        # Create mock trial that tracks suggest_* calls
        mock_trial = Mock()
        suggest_calls = []

        def track_suggest_categorical(name, choices):
            suggest_calls.append(('categorical', name, choices))
            # These should NOT be suggested when fixed params provided
            forbidden_params = ['context_length_factor', 'encoder_type', 'batch_size', 'dropout_rate']
            if name in forbidden_params:
                raise AssertionError(f"Unexpected suggest_categorical call for '{name}' - should be fixed!")

            # Legitimate Stage 2 params that should be suggested
            if name == 'stage2_activation_function':
                return 'relu'
            elif name == 'copula_embedding_dim_per_head':
                return 64
            elif name == 'copula_series_embedding_dim':
                return 128
            elif name == 'ac_mlp_dim':
                return 128
            elif 'weight_decay' in name or 'gradient_clip' in name:
                return choices[0] if choices else 0.0
            return 'relu'

        def track_suggest_int(name, low, high):
            suggest_calls.append(('int', name, (low, high)))
            if 'copula_num_heads' in name:
                return 3
            elif 'copula_num_layers' in name:
                return 2
            elif 'copula_input_encoder_layers' in name:
                return 3
            elif 'ac_mlp_num_layers' in name:
                return 4
            return 2

        def track_suggest_float(name, low, high, log=False):
            suggest_calls.append(('float', name, (low, high), log))
            if 'lr_stage2' in name:
                return 5e-5
            elif 'eta_min_fraction' in name:
                return 0.005
            return 1e-5

        mock_trial.suggest_categorical = track_suggest_categorical
        mock_trial.suggest_int = track_suggest_int
        mock_trial.suggest_float = track_suggest_float

        # Stage 1 fixed parameters
        stage1_fixed = {
            'context_length_factor': 25,
            'encoder_type': 'temporal',
            'batch_size': 64,
            'dropout_rate': 0.008
        }

        dynamic_kwargs = {
            'stage1_fixed_params': stage1_fixed
        }

        # Call get_params for Stage 2 with fixed params
        params = TACTiS2Estimator.get_params(
            trial=mock_trial,
            tuning_phase=2,
            dynamic_kwargs=dynamic_kwargs
        )

        # Assertions: Common params should be in output but NOT suggested by Optuna
        assert params['context_length_factor'] == 25, "Should use fixed context_length_factor"
        assert params['encoder_type'] == 'temporal', "Should use fixed encoder_type"
        assert params['batch_size'] == 64, "Should use fixed batch_size"
        assert params['dropout_rate'] == 0.008, "Should use fixed dropout_rate"

        # Verify NO suggestions were made for common params
        suggested_param_names = [call[1] for call in suggest_calls]
        assert 'context_length_factor' not in suggested_param_names, "Should NOT suggest context_length_factor"
        assert 'encoder_type' not in suggested_param_names, "Should NOT suggest encoder_type"
        assert 'batch_size' not in suggested_param_names, "Should NOT suggest batch_size"
        assert 'dropout_rate' not in suggested_param_names, "Should NOT suggest dropout_rate"

        # Verify Stage 2 params WERE suggested (should be tuned)
        assert 'copula_num_heads' in suggested_param_names, "Should suggest copula_num_heads"
        assert 'ac_mlp_num_layers' in suggested_param_names, "Should suggest ac_mlp_num_layers"
        assert 'lr_stage2' in suggested_param_names, "Should suggest lr_stage2"

        print(f"✓ Stage 2 with fixed params: {len(suggest_calls)} suggestions (common params NOT suggested)")
        print(f"  Fixed params used: {list(stage1_fixed.keys())}")
        print(f"  Tuned params suggested: {len(suggested_param_names)}")

    def test_stage1_normal_suggestions(self):
        """Verify Stage 1 still suggests common params normally."""

        mock_trial = Mock()
        suggest_calls = []

        def track_suggest_categorical(name, choices):
            suggest_calls.append(('categorical', name))
            if name == 'context_length_factor':
                return 20
            elif name == 'encoder_type':
                return 'standard'
            elif name == 'batch_size':
                return 128
            elif name == 'dropout_rate':
                return 0.01
            elif name == 'stage1_activation_function':
                return 'relu'
            elif 'embedding_dim' in name:
                return 64
            elif 'hidden_dim' in name:
                return 128
            elif 'num_bins' in name:
                return 100
            return 'relu'

        def track_suggest_int(name, low, high):
            suggest_calls.append(('int', name))
            return 3

        def track_suggest_float(name, low, high, log=False):
            suggest_calls.append(('float', name))
            return 5e-4

        mock_trial.suggest_categorical = track_suggest_categorical
        mock_trial.suggest_int = track_suggest_int
        mock_trial.suggest_float = track_suggest_float

        # Call get_params for Stage 1 (no fixed params)
        params = TACTiS2Estimator.get_params(
            trial=mock_trial,
            tuning_phase=1,
            dynamic_kwargs={}
        )

        # Verify common params WERE suggested (normal Stage 1 behavior)
        suggested_param_names = [call[1] for call in suggest_calls]
        assert 'context_length_factor' in suggested_param_names, "Stage 1 should suggest context_length_factor"
        assert 'encoder_type' in suggested_param_names, "Stage 1 should suggest encoder_type"
        assert 'batch_size' in suggested_param_names, "Stage 1 should suggest batch_size"
        assert 'dropout_rate' in suggested_param_names, "Stage 1 should suggest dropout_rate"

        # Verify params in output
        assert params['context_length_factor'] == 20
        assert params['encoder_type'] == 'standard'
        assert params['batch_size'] == 128
        assert params['dropout_rate'] == 0.01

        print(f"✓ Stage 1 normal operation: {len(suggest_calls)} suggestions (all params suggested)")

    def test_stage2_without_fixed_params_fallback(self):
        """Verify Stage 2 without fixed params still suggests (backward compatibility)."""

        mock_trial = Mock()
        suggest_calls = []

        def track_suggest_categorical(name, choices):
            suggest_calls.append(('categorical', name))
            if name == 'context_length_factor':
                return 15
            elif name == 'encoder_type':
                return 'temporal'
            elif name == 'batch_size':
                return 256
            elif name == 'dropout_rate':
                return 0.007
            elif 'activation_function' in name:
                return 'relu'
            elif 'embedding_dim' in name:
                return 32
            elif 'mlp_dim' in name:
                return 64
            return 'relu'

        def track_suggest_int(name, low, high):
            suggest_calls.append(('int', name))
            return 2

        def track_suggest_float(name, low, high, log=False):
            suggest_calls.append(('float', name))
            return 1e-5

        mock_trial.suggest_categorical = track_suggest_categorical
        mock_trial.suggest_int = track_suggest_int
        mock_trial.suggest_float = track_suggest_float

        # Call get_params for Stage 2 WITHOUT fixed params (sequential training scenario)
        params = TACTiS2Estimator.get_params(
            trial=mock_trial,
            tuning_phase=2,
            dynamic_kwargs={}  # No stage1_fixed_params
        )

        # Verify common params WERE suggested (fallback behavior)
        suggested_param_names = [call[1] for call in suggest_calls]
        assert 'context_length_factor' in suggested_param_names, "Should suggest when no fixed params"
        assert 'encoder_type' in suggested_param_names, "Should suggest when no fixed params"
        assert 'batch_size' in suggested_param_names, "Should suggest when no fixed params"
        assert 'dropout_rate' in suggested_param_names, "Should suggest when no fixed params"

        print(f"✓ Stage 2 without fixed params (sequential training): {len(suggest_calls)} suggestions")


class TestTPESearchSpaceReduction:
    """Test that the fix actually reduces TPE search space dimensionality."""

    def test_search_space_dimensionality_reduction(self):
        """Verify that fixed params reduce the effective search space."""

        # Count suggestions for Stage 2 WITH fixed params
        mock_trial_fixed = Mock()
        fixed_suggestions = []

        def track_fixed(name, *args, **kwargs):
            fixed_suggestions.append(name)
            if 'num' in name or 'layers' in name:
                return 3
            elif 'lr' in name or 'eta' in name or 'weight' in name or 'clip' in name:
                return 1e-5
            return 'relu' if 'function' in name else 64

        mock_trial_fixed.suggest_categorical = lambda name, choices: track_fixed(name)
        mock_trial_fixed.suggest_int = lambda name, low, high: track_fixed(name)
        mock_trial_fixed.suggest_float = lambda name, low, high, log=False: track_fixed(name)

        stage1_fixed = {
            'context_length_factor': 25,
            'encoder_type': 'temporal',
            'batch_size': 64,
            'dropout_rate': 0.008
        }

        params_fixed = TACTiS2Estimator.get_params(
            trial=mock_trial_fixed,
            tuning_phase=2,
            dynamic_kwargs={'stage1_fixed_params': stage1_fixed}
        )

        # Count suggestions for Stage 2 WITHOUT fixed params
        mock_trial_normal = Mock()
        normal_suggestions = []

        def track_normal(name, *args, **kwargs):
            normal_suggestions.append(name)
            if 'num' in name or 'layers' in name:
                return 3
            elif 'lr' in name or 'eta' in name or 'weight' in name or 'clip' in name:
                return 1e-5
            return 'temporal' if name == 'encoder_type' else 64

        mock_trial_normal.suggest_categorical = lambda name, choices: track_normal(name)
        mock_trial_normal.suggest_int = lambda name, low, high: track_normal(name)
        mock_trial_normal.suggest_float = lambda name, low, high, log=False: track_normal(name)

        params_normal = TACTiS2Estimator.get_params(
            trial=mock_trial_normal,
            tuning_phase=2,
            dynamic_kwargs={}
        )

        # Compare dimensionality
        dimensionality_reduction = len(normal_suggestions) - len(fixed_suggestions)

        print(f"\n{'='*70}")
        print(f"TPE SEARCH SPACE COMPARISON")
        print(f"{'='*70}")
        print(f"Stage 2 WITH fixed params:    {len(fixed_suggestions):2d} dimensions")
        print(f"Stage 2 WITHOUT fixed params:  {len(normal_suggestions):2d} dimensions")
        print(f"Dimensionality reduction:      {dimensionality_reduction:2d} dimensions ({dimensionality_reduction/len(normal_suggestions)*100:.1f}%)")
        print(f"\nFixed params not in search space: {list(stage1_fixed.keys())}")
        print(f"{'='*70}")

        # Assertion: Should have fewer dimensions with fixed params
        assert len(fixed_suggestions) < len(normal_suggestions), \
            f"Fixed params should reduce search space: {len(fixed_suggestions)} vs {len(normal_suggestions)}"

        # Should reduce by exactly 4 dimensions (the 4 common params)
        assert dimensionality_reduction == 4, \
            f"Should reduce by 4 dimensions (common params), got {dimensionality_reduction}"

        print(f"\n✓ Search space reduced by 4 dimensions as expected")


def run_all_tests():
    """Run all tests and print summary."""
    print("\n" + "="*70)
    print("TPE EFFICIENCY FIX VALIDATION")
    print("="*70 + "\n")

    test_classes = [
        TestStage2TPEEfficiency,
        TestTPESearchSpaceReduction,
    ]

    total_tests = 0
    passed_tests = 0
    failed_tests = []

    for test_class in test_classes:
        print(f"\n{'─'*70}")
        print(f"Running {test_class.__name__}")
        print(f"{'─'*70}\n")

        test_methods = [method for method in dir(test_class) if method.startswith('test_')]

        for test_method in test_methods:
            total_tests += 1
            test_name = test_method.replace('_', ' ').title()

            try:
                test_instance = test_class()
                method = getattr(test_instance, test_method)
                method()
                passed_tests += 1
                print(f"  ✓ {test_name}\n")
            except Exception as e:
                failed_tests.append((test_class.__name__, test_name, str(e)))
                print(f"  ✗ {test_name}")
                print(f"    Error: {e}\n")

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")

    if failed_tests:
        print("\nFailed tests:")
        for class_name, test_name, error in failed_tests:
            print(f"  - {class_name}.{test_name}")
            print(f"    {error}")
    else:
        print("\n🎉 All tests passed!")
        print("\n✅ TPE efficiency fix working correctly:")
        print("   - Stage 2 with Stage 1 params: No wasted suggestions")
        print("   - 4D search space reduction achieved")
        print("   - Backward compatibility maintained")

    print("="*70 + "\n")

    return len(failed_tests) == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
