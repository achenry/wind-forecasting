# Distributed Training Implementation Guide

This document provides a comprehensive guide to the distributed training optimizations implemented for the wind forecasting framework.

## Overview

The distributed training system solves the core bottleneck where GluonTS's `as_stacked_batches()` doesn't support proper data sharding across multiple GPUs. The solution provides:

- **Automatic distributed training detection**
- **Lightning DataModule for proper batch sharding**  
- **GPU-specific optimizations (H100/A100/etc.)**
- **Backward compatibility with existing workflows**
- **Seamless integration with tuning and training modes**

## Key Components

### 1. Environment Detection (`distributed_utils.py`)
- Detects SLURM environment and distributed training setup
- Distinguishes between tuning workers and distributed training
- Automatically enables optimizations when appropriate

### 2. GPU Optimizations (`gpu_optimizations.py`)
- GPU capability detection (H100, A100, Tensor Cores, etc.)
- Automatic precision selection (BF16 for H100/A100, FP16 for others)
- Memory-based batch size scaling and gradient checkpointing
- CUDA/NCCL environment optimizations

### 3. Lightning DataModule (`distributed_datamodule.py`)
- Converts GluonTS datasets to PyTorch format efficiently
- Leverages Lightning's automatic DistributedSampler injection
- Configurable through YAML settings
- Handles transformation caching (happens once per training)

### 4. Conditional Integration (`TACTiS2Estimator`)
- Conditional logic in `train_model()` method
- Uses DataModule when distributed optimizations enabled
- Falls back to traditional GluonTS loaders otherwise
- Zero breaking changes to existing code

## Configuration

### YAML Configuration
Add these sections to your training configuration:

```yaml
# Enable distributed optimizations
trainer:
  enable_distributed_optimizations: true
  devices: auto
  strategy: ddp
  precision: null  # Auto-detected based on GPU

# DataLoader configuration for distributed training
dataloader:
  num_workers: null         # Auto-detect from SLURM_CPUS_PER_TASK
  pin_memory: true         # Use pinned memory for faster GPU transfers
  persistent_workers: true # Keep workers alive between epochs
  prefetch_factor: 2       # Number of batches to prefetch per worker
```

### GPU Optimizations
The system automatically detects GPU capabilities and applies optimizations:

- **H100**: BF16 precision, 2.0x batch multiplier, fused optimizers
- **A100**: BF16 precision, 1.5x batch multiplier, torch.compile  
- **Tensor Core GPUs**: FP16 precision, mixed precision training
- **Limited Memory**: Gradient checkpointing, conservative batch sizes

## Usage

### Training Mode
```bash
# Standard distributed training (4 GPUs)
sbatch train_awaken_storm_210.sh

# The system automatically:
# 1. Detects 4-GPU environment
# 2. Enables Lightning DataModule
# 3. Applies H100/A100 optimizations
# 4. Shards batches across GPUs properly
# 5. Uses gradient accumulation if needed
```

### Tuning Mode
```bash
# Hyperparameter tuning (uses traditional loaders)
sbatch tune_job_submit.sh

# The system automatically:
# 1. Detects tuning mode
# 2. Uses traditional GluonTS loaders
# 3. Maintains existing tuning behavior
```

### Single GPU Training
```bash
# Single GPU (uses traditional loaders)
python run_model.py --config config.yaml --mode train --single_gpu

# The system automatically:
# 1. Detects single GPU environment
# 2. Uses traditional GluonTS loaders
# 3. Maintains existing behavior
```

## Performance Benefits

### Before (Traditional GluonTS)
- Each GPU gets full `batch_size=64` (no sharding)
- 4 GPUs = 4×64 = 256 effective batch size (unintended!)
- No distributed sampling coordination
- Training slower due to larger effective batch

### After (Lightning DataModule)
- Each GPU gets `batch_size=16` (proper sharding) 
- 4 GPUs = 4×16 = 64 effective batch size (as intended!)
- Automatic DistributedSampler coordination
- H100 optimizations: BF16 precision, 2x batch multiplier
- Training much faster with optimal batch sizes

## Architecture Details

### Data Flow
1. **Environment Detection**: Detects SLURM/distributed environment
2. **GPU Detection**: Identifies GPU capabilities and optimizations  
3. **Configuration**: Updates trainer and DataLoader settings
4. **Runtime Config**: Passes settings to estimator
5. **Conditional Training**: Estimator chooses DataModule vs traditional loaders
6. **Distributed Training**: Lightning handles DistributedSampler automatically

### Batch Configuration Logic
```python
# Example for H100 with tuned batch_size=64, world_size=4
original_batch_size = 64
gpu_multiplier = 2.0  # H100 optimization
adjusted_batch_size = 64 * 2.0 = 128
per_gpu_batch_size = 128 / 4 = 32
accumulate_grad_batches = 1
effective_batch_size = 32 * 4 * 1 = 128
```

### Error Handling
- Graceful fallback to traditional mode if issues occur
- Comprehensive logging for debugging
- Maintains backward compatibility always

## Files Modified

### Core Implementation
- `wind_forecasting/utils/distributed_utils.py` - Environment detection
- `wind_forecasting/utils/gpu_optimizations.py` - GPU optimizations  
- `wind_forecasting/utils/distributed_datamodule.py` - Lightning DataModule
- `wind_forecasting/run_scripts/run_model.py` - Integration logic
- `pytorch_transformer_ts/tactis_2/estimator.py` - Conditional training

### Configuration
- `config/training/training_inputs_juan_awaken_tune_storm_pred210.yaml` - Added DataLoader settings

### Testing
- `wind_forecasting/utils/integration_testing.py` - Comprehensive tests

## Troubleshooting

### Common Issues

1. **Feature disabled**: Check `trainer.enable_distributed_optimizations: true`
2. **Wrong batch sizes**: Verify `original_batch_size` in logs
3. **Memory issues**: Check GPU memory optimization logs
4. **Slow training**: Verify distributed environment detection

### Debug Logging
Look for these log messages:
```
Training environment detected: {'mode': 'distributed_training', 'world_size': 4, ...}
Distributed optimizations: ENABLED
GPU Optimization Summary: GPUs: 4 × NVIDIA H100-SXM, Precision: bf16
Using Lightning DataModule for distributed training
Distributed batch config: 32 per GPU, 1 accumulation
```

### Verification
Run integration tests:
```bash
python wind_forecasting/utils/integration_testing.py
```

## Future Enhancements

- **FP8 Training**: When PyTorch adds FP8 support for H100
- **Model Parallelism**: For extremely large models
- **Dynamic Batching**: Adaptive batch sizes during training
- **Multi-Node Support**: Scaling beyond single-node clusters

## Performance Monitoring

Monitor these metrics:
- **GPU Utilization**: Should be high (>90%) on all GPUs
- **Memory Usage**: Should be balanced across GPUs  
- **Batch Processing Time**: Should be consistent
- **Loss Convergence**: Should match single-GPU results

The distributed training system is designed to maximize performance on modern GPU clusters while maintaining full compatibility with existing workflows.