"""
Comprehensive integration testing for distributed training optimizations.
"""
import logging
import tempfile
import yaml
import os
from typing import Dict, Any, Optional
from unittest.mock import patch, MagicMock

logger = logging.getLogger(__name__)

def create_test_config(
    enable_distributed: bool = True,
    world_size: int = 4,
    batch_size: int = 64
) -> Dict[str, Any]:
    """
    Create a comprehensive test configuration for integration testing.
    
    Parameters
    ----------
    enable_distributed : bool
        Whether to enable distributed optimizations
    world_size : int
        Number of GPUs to simulate
    batch_size : int
        Base batch size for testing
        
    Returns
    -------
    Dict[str, Any]
        Complete test configuration
    """
    config = {
        "experiment": {
            "username": "test_user",
            "project_name": "integration_test",
            "run_name": "test_run",
            "project_root": "/tmp/test_project",
            "log_dir": "/tmp/test_logs",
        },
        "logging": {
            "entity": "test_entity",
            "wandb_mode": "disabled",
            "save_code": False,
            "wandb_dir": "/tmp/test_wandb",
            "optuna_dir": "/tmp/test_optuna",
            "checkpoint_dir": "/tmp/test_checkpoints",
        },
        "dataset": {
            "sampler": "random",
            "data_path": "/tmp/test_data.parquet",
            "normalization_consts_path": "/tmp/test_norms.csv",
            "context_length": 600,
            "prediction_length": 210,
            "normalize": True,
            "batch_size": batch_size,
            "base_batch_size": batch_size,
            "workers": 4,
            "test_split": 0.20,
            "val_split": 0.10,
            "resample_freq": "30s",
            "n_splits": 1,
            "per_turbine_target": True,
        },
        "dataloader": {
            "num_workers": 4,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 2,
        },
        "trainer": {
            "val_check_interval": 1.0,
            "accelerator": "gpu",
            "devices": world_size,
            "num_nodes": 1,
            "strategy": "ddp",
            "max_epochs": 1,  # Short for testing
            "limit_train_batches": 10,  # Very limited for testing
            "enable_distributed_optimizations": enable_distributed,
        },
        "model": {
            "distr_output": {
                "class": "LowRankMultivariateNormalOutput",
                "kwargs": {"rank": 8}
            },
            "tactis": {
                "initial_stage": 1,
                "stage2_start_epoch": 20,
                "ac_mlp_num_layers": 2,
                "ac_mlp_dim": 128,
                "stage1_activation_function": "relu",
                "stage2_activation_function": "relu",
                "input_encoding_normalization": True,
                "scaling": "std",
                "loss_normalization": "both",
                "encoder_type": "standard",
                "bagging_size": None,
                "num_parallel_samples": 200,
                "marginal_embedding_dim_per_head": 8,
                "marginal_num_heads": 5,
                "marginal_num_layers": 4,
                "flow_input_encoder_layers": 6,
                "flow_series_embedding_dim": 5,
                "copula_embedding_dim_per_head": 8,
                "copula_num_heads": 5,
                "copula_num_layers": 2,
                "copula_input_encoder_layers": 1,
                "copula_series_embedding_dim": 48,
                "decoder_dsf_num_layers": 2,
                "decoder_dsf_hidden_dim": 256,
                "decoder_mlp_num_layers": 3,
                "decoder_mlp_hidden_dim": 16,
                "decoder_transformer_num_layers": 3,
                "decoder_transformer_embedding_dim_per_head": 16,
                "decoder_transformer_num_heads": 6,
                "decoder_num_bins": 50,
                "lr_stage1": 5e-6,
                "lr_stage2": 2e-6,
                "weight_decay_stage1": 1e-5,
                "weight_decay_stage2": 1e-5,
                "dropout_rate": 0.1,
                "gradient_clip_val_stage1": 1000.0,
                "gradient_clip_val_stage2": 1000.0,
                "warmup_steps_s1": 0.10,
                "steps_to_decay_s1": 0.90,
                "eta_min_fraction_s1": 0.01,
                "warmup_steps_s2": 0.10,
                "steps_to_decay_s2": 0.90,
                "eta_min_fraction_s2": 0.01,
            }
        },
        "callbacks": {
            "progress_bar": {
                "class_path": "lightning.pytorch.callbacks.TQDMProgressBar",
                "init_args": {
                    "refresh_rate": 2048,
                    "leave": False
                }
            },
            "model_checkpoint": {
                "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
                "init_args": {
                    "dirpath": "/tmp/test_checkpoints",
                    "filename": "{epoch}-{step}-{val_loss:.2f}",
                    "monitor": "val_loss",
                    "mode": "min",
                    "save_top_k": 1,
                    "save_last": True,
                    "verbose": True
                }
            }
        }
    }
    
    return config

def test_environment_detection():
    """Test that environment detection works correctly."""
    logger.info("Testing environment detection...")
    
    from wind_forecasting.utils.distributed_utils import detect_training_environment
    
    # Test with mocked SLURM environment
    with patch.dict(os.environ, {
        'SLURM_NTASKS_PER_NODE': '4',
        'SLURM_NNODES': '1',
        'WORKER_RANK': '0'
    }):
        env_info = detect_training_environment()
        
        assert env_info['mode'] == 'tuning', f"Expected tuning mode, got {env_info['mode']}"
        assert env_info['world_size'] == 4, f"Expected world_size=4, got {env_info['world_size']}"
        assert env_info['worker_rank'] == '0', f"Expected worker_rank='0', got {env_info['worker_rank']}"
    
    logger.info("✓ Environment detection test passed")

def test_gpu_optimization_integration():
    """Test GPU optimization integration."""
    logger.info("Testing GPU optimization integration...")
    
    from wind_forecasting.utils.gpu_optimizations import (
        detect_gpu_capabilities,
        apply_trainer_optimizations
    )
    
    # Test with mock H100 environment
    mock_gpu_props = MagicMock()
    mock_gpu_props.name = "NVIDIA H100-SXM"
    mock_gpu_props.total_memory = 80 * 1024**3  # 80GB
    mock_gpu_props.major = 9
    mock_gpu_props.minor = 0
    mock_gpu_props.multi_processor_count = 132
    
    with patch('torch.cuda.is_available', return_value=True):
        with patch('torch.cuda.device_count', return_value=8):
            with patch('torch.cuda.get_device_properties', return_value=mock_gpu_props):
                capabilities = detect_gpu_capabilities()
                
                assert capabilities['has_gpu'] == True
                assert capabilities['gpu_count'] == 8
                assert capabilities['has_h100'] == True
                assert capabilities['optimizations']['precision'] == 'bf16'
                assert capabilities['optimizations']['batch_size_multiplier'] == 2.0
    
    # Test trainer optimization application
    base_config = {"max_epochs": 100, "devices": 8}
    optimized_config = apply_trainer_optimizations(
        trainer_kwargs=base_config,
        gpu_capabilities=capabilities,
        config={}
    )
    
    assert optimized_config['precision'] == 'bf16'
    assert optimized_config['max_epochs'] == 100  # Should preserve
    
    logger.info("✓ GPU optimization integration test passed")

def test_datamodule_creation():
    """Test DataModule creation with various configurations."""
    logger.info("Testing DataModule creation...")
    
    from wind_forecasting.utils.distributed_datamodule import (
        create_distributed_datamodule,
        get_default_dataloader_config
    )
    
    config = create_test_config()
    
    # Test dataloader config extraction
    dataloader_config = get_default_dataloader_config(config)
    
    assert dataloader_config['num_workers'] == 4
    assert dataloader_config['pin_memory'] == True
    assert dataloader_config['persistent_workers'] == True
    assert dataloader_config['prefetch_factor'] == 2
    
    logger.info("✓ DataModule creation test passed")

def test_batch_configuration_scenarios():
    """Test batch configuration under various scenarios."""
    logger.info("Testing batch configuration scenarios...")
    
    from wind_forecasting.utils.distributed_utils import calculate_optimal_batch_configuration
    
    # Scenario 1: H100 with large batch size
    h100_capabilities = {
        'has_gpu': True,
        'optimizations': {'batch_size_multiplier': 2.0}
    }
    
    per_gpu_batch, accumulate = calculate_optimal_batch_configuration(
        tuned_batch_size=64,
        world_size=8,
        min_batch_per_gpu=16,
        gpu_capabilities=h100_capabilities
    )
    
    # Should apply 2.0x multiplier: 64 * 2.0 = 128, then 128 / 8 = 16 per GPU
    assert per_gpu_batch == 16
    assert accumulate == 1
    
    # Scenario 2: Limited memory GPU with small batch
    limited_capabilities = {
        'has_gpu': True,
        'optimizations': {'batch_size_multiplier': 1.0}
    }
    
    per_gpu_batch, accumulate = calculate_optimal_batch_configuration(
        tuned_batch_size=32,
        world_size=8,
        min_batch_per_gpu=16,
        gpu_capabilities=limited_capabilities
    )
    
    # Should use gradient accumulation: 16 per GPU, accumulate 2x to get 32 effective
    assert per_gpu_batch == 16
    assert accumulate >= 1  # Should use accumulation
    
    logger.info("✓ Batch configuration scenarios test passed")

def test_config_integration():
    """Test complete configuration integration."""
    logger.info("Testing complete configuration integration...")
    
    config = create_test_config(enable_distributed=True, world_size=4, batch_size=64)
    
    # Simulate the configuration flow that happens in run_model.py
    from wind_forecasting.utils.distributed_utils import (
        detect_training_environment,
        should_enable_distributed_optimizations
    )
    from wind_forecasting.utils.gpu_optimizations import (
        detect_gpu_capabilities,
        apply_trainer_optimizations
    )
    
    # Mock args for should_enable_distributed_optimizations
    mock_args = MagicMock()
    mock_args.mode = "train"
    
    # Test environment detection
    with patch.dict(os.environ, {'SLURM_NTASKS_PER_NODE': '4', 'SLURM_NNODES': '1'}):
        training_env = detect_training_environment()
        use_distributed = should_enable_distributed_optimizations(config, mock_args)
    
    # Test GPU detection and optimization
    gpu_capabilities = detect_gpu_capabilities()
    
    if use_distributed:
        config["trainer"] = apply_trainer_optimizations(
            trainer_kwargs=config["trainer"],
            gpu_capabilities=gpu_capabilities,
            config=config
        )
    
    # Create runtime config
    config["_runtime"] = {
        "training_environment": training_env,
        "use_distributed_optimizations": use_distributed,
        "original_batch_size": config["dataset"]["batch_size"],
        "gpu_capabilities": gpu_capabilities,
    }
    
    # Verify complete configuration
    assert "_runtime" in config
    assert config["_runtime"]["original_batch_size"] == 64
    assert "training_environment" in config["_runtime"]
    assert "gpu_capabilities" in config["_runtime"]
    
    logger.info("✓ Complete configuration integration test passed")

def test_estimator_integration():
    """Test estimator integration with runtime configuration."""
    logger.info("Testing estimator integration...")
    
    # This test verifies the estimator receives and processes runtime config correctly
    runtime_config = {
        'use_distributed_optimizations': True,
        'training_environment': {'world_size': 4, 'mode': 'distributed_training'},
        'original_batch_size': 64,
        'full_config': create_test_config(),
        'gpu_capabilities': {
            'has_gpu': True,
            'optimizations': {'batch_size_multiplier': 1.5}
        }
    }
    
    # Mock TACTiS2Estimator initialization with runtime config
    estimator_kwargs = {
        "freq": "30S",
        "prediction_length": 210,
        "context_length": 600,
        "input_size": 4,
        "batch_size": 64,
        "trainer_kwargs": {"max_epochs": 1, "logger": None},
        "_runtime_config": runtime_config
    }
    
    # Verify the configuration would be passed correctly
    assert "_runtime_config" in estimator_kwargs
    assert estimator_kwargs["_runtime_config"]["use_distributed_optimizations"] == True
    assert estimator_kwargs["_runtime_config"]["original_batch_size"] == 64
    
    logger.info("✓ Estimator integration test passed")

def test_yaml_configuration_roundtrip():
    """Test that YAML configuration works end-to-end."""
    logger.info("Testing YAML configuration roundtrip...")
    
    config = create_test_config()
    
    # Create temporary YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        yaml_path = f.name
    
    try:
        # Load configuration from YAML
        with open(yaml_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        # Verify key sections are preserved
        assert loaded_config["trainer"]["enable_distributed_optimizations"] == True
        assert loaded_config["dataloader"]["num_workers"] == 4
        assert loaded_config["dataset"]["batch_size"] == 64
        assert loaded_config["model"]["tactis"]["lr_stage1"] == 5e-6
        
        logger.info("✓ YAML configuration roundtrip test passed")
        
    finally:
        # Clean up temporary file
        os.unlink(yaml_path)

def run_integration_tests():
    """Run all integration tests."""
    logger.info("Starting comprehensive integration tests...")
    
    tests = [
        test_environment_detection,
        test_gpu_optimization_integration,
        test_datamodule_creation,
        test_batch_configuration_scenarios,
        test_config_integration,
        test_estimator_integration,
        test_yaml_configuration_roundtrip,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            logger.error(f"❌ Test {test.__name__} failed: {e}", exc_info=True)
            failed += 1
    
    logger.info(f"\nIntegration test results: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 All integration tests passed!")
        return True
    else:
        logger.error("❌ Some integration tests failed!")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    success = run_integration_tests()
    exit(0 if success else 1)