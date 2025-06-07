"""
GPU-specific optimizations for distributed training.
"""
import logging
import torch
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def detect_gpu_capabilities():
    """
    Detect GPU capabilities and return optimization recommendations.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing GPU info and optimization recommendations
    """
    if not torch.cuda.is_available():
        return {
            'has_gpu': False,
            'gpu_count': 0,
            'optimizations': {}
        }
    
    gpu_count = torch.cuda.device_count()
    gpu_info = []
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        gpu_info.append({
            'device_id': i,
            'name': props.name,
            'total_memory': props.total_memory,
            'major': props.major,
            'minor': props.minor,
            'multi_processor_count': props.multi_processor_count,
        })
    
    # Detect GPU generation and capabilities
    has_h100 = any('H100' in gpu['name'] for gpu in gpu_info)
    has_a100 = any('A100' in gpu['name'] for gpu in gpu_info)
    has_ampere_or_newer = any(gpu['major'] >= 8 for gpu in gpu_info)
    has_tensor_cores = any(gpu['major'] >= 7 for gpu in gpu_info)
    
    # Memory information
    total_memory = sum(gpu['total_memory'] for gpu in gpu_info)
    min_memory = min(gpu['total_memory'] for gpu in gpu_info) if gpu_info else 0
    
    logger.info(f"Detected {gpu_count} GPU(s): {[gpu['name'] for gpu in gpu_info]}")
    logger.info(f"Total GPU memory: {total_memory / 1e9:.1f}GB")
    
    return {
        'has_gpu': True,
        'gpu_count': gpu_count,
        'gpu_info': gpu_info,
        'has_h100': has_h100,
        'has_a100': has_a100,
        'has_ampere_or_newer': has_ampere_or_newer,
        'has_tensor_cores': has_tensor_cores,
        'total_memory': total_memory,
        'min_memory': min_memory,
        'optimizations': _get_optimization_recommendations(
            has_h100, has_a100, has_ampere_or_newer, has_tensor_cores, min_memory
        )
    }

def _get_optimization_recommendations(
    has_h100: bool,
    has_a100: bool, 
    has_ampere_or_newer: bool,
    has_tensor_cores: bool,
    min_memory: int
) -> Dict[str, Any]:
    """Get optimization recommendations based on GPU capabilities."""
    optimizations = {}
    
    # Precision recommendations
    if has_h100:
        # H100 has excellent BF16 support and FP8 capabilities
        optimizations['precision'] = 'bf16'
        optimizations['use_fp8'] = True  # For future PyTorch support
        logger.info("H100 detected: Recommending BF16 precision with FP8 capability")
    elif has_a100 or has_ampere_or_newer:
        # A100 and other Ampere+ GPUs have good BF16 support
        optimizations['precision'] = 'bf16'
        optimizations['use_fp8'] = False
        logger.info("Ampere+ GPU detected: Recommending BF16 precision")
    elif has_tensor_cores:
        # Tensor Core GPUs benefit from mixed precision
        optimizations['precision'] = '16'
        optimizations['use_fp8'] = False
        logger.info("Tensor Core GPU detected: Recommending FP16 precision")
    else:
        # Fallback to FP32 for older GPUs
        optimizations['precision'] = '32'
        optimizations['use_fp8'] = False
        logger.info("Older GPU detected: Using FP32 precision")
    
    # Memory optimizations
    memory_gb = min_memory / 1e9
    if memory_gb >= 80:  # H100/A100 80GB
        optimizations['batch_size_multiplier'] = 2.0
        optimizations['gradient_checkpointing'] = False
        logger.info("High memory GPU: Enabling larger batch sizes")
    elif memory_gb >= 40:  # A100 40GB
        optimizations['batch_size_multiplier'] = 1.5
        optimizations['gradient_checkpointing'] = False
        logger.info("Medium memory GPU: Moderate batch size increase")
    else:  # Lower memory GPUs
        optimizations['batch_size_multiplier'] = 1.0
        optimizations['gradient_checkpointing'] = True
        logger.info("Limited memory GPU: Enabling gradient checkpointing")
    
    # Optimizer optimizations
    if has_h100 or has_a100:
        # High-end GPUs can handle fused optimizers
        optimizations['use_fused_optimizer'] = True
        optimizations['optimizer_type'] = 'adamw_fused'
        logger.info("High-end GPU: Enabling fused optimizer")
    else:
        optimizations['use_fused_optimizer'] = False
        optimizations['optimizer_type'] = 'adamw'
    
    # Compilation optimizations
    if has_ampere_or_newer:
        # Modern GPUs benefit from torch.compile
        optimizations['use_torch_compile'] = True
        optimizations['compile_mode'] = 'default'
        logger.info("Modern GPU: Enabling torch.compile")
    else:
        optimizations['use_torch_compile'] = False
    
    return optimizations

def apply_trainer_optimizations(
    trainer_kwargs: Dict[str, Any],
    gpu_capabilities: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Apply GPU-specific optimizations to trainer configuration.
    
    Parameters
    ----------
    trainer_kwargs : Dict[str, Any]
        Base trainer configuration
    gpu_capabilities : Dict[str, Any]
        GPU capabilities from detect_gpu_capabilities()
    config : Dict[str, Any]
        Full configuration dictionary
        
    Returns
    -------
    Dict[str, Any]
        Optimized trainer configuration
    """
    if not gpu_capabilities['has_gpu']:
        logger.info("No GPU detected, skipping GPU optimizations")
        return trainer_kwargs
    
    optimizations = gpu_capabilities['optimizations']
    optimized_kwargs = trainer_kwargs.copy()
    
    # Apply precision optimization
    precision = optimizations.get('precision', '32')
    if precision != '32':
        optimized_kwargs['precision'] = precision
        logger.info(f"Set trainer precision to {precision}")
    
    # Apply memory optimizations
    if optimizations.get('gradient_checkpointing', False):
        # This would need to be handled at the model level
        logger.info("Gradient checkpointing recommended (needs model-level implementation)")
    
    # Apply compilation optimizations
    if optimizations.get('use_torch_compile', False):
        # Note: torch.compile needs to be applied to the model, not trainer
        logger.info("torch.compile recommended (needs model-level implementation)")
    
    # Set matmul precision for better performance on modern GPUs
    if gpu_capabilities['has_ampere_or_newer']:
        torch.set_float32_matmul_precision('medium')
        logger.info("Set matmul precision to 'medium' for Ampere+ GPUs")
    
    return optimized_kwargs

def calculate_optimal_distributed_config(
    base_batch_size: int,
    gpu_capabilities: Dict[str, Any],
    world_size: int,
    target_memory_utilization: float = 0.85
) -> Dict[str, Any]:
    """
    Calculate optimal distributed training configuration.
    
    Parameters
    ----------
    base_batch_size : int
        Base batch size from hyperparameter tuning
    gpu_capabilities : Dict[str, Any]
        GPU capabilities from detect_gpu_capabilities()
    world_size : int
        Number of GPUs for distributed training
    target_memory_utilization : float
        Target GPU memory utilization (0.0-1.0)
        
    Returns
    -------
    Dict[str, Any]
        Optimal distributed configuration
    """
    if not gpu_capabilities['has_gpu']:
        return {
            'per_gpu_batch_size': base_batch_size,
            'accumulate_grad_batches': 1,
            'effective_batch_size': base_batch_size,
        }
    
    optimizations = gpu_capabilities['optimizations']
    
    # Apply batch size multiplier based on GPU memory
    batch_multiplier = optimizations.get('batch_size_multiplier', 1.0)
    adjusted_base_batch = int(base_batch_size * batch_multiplier)
    
    # Calculate per-GPU batch size
    if world_size <= 1:
        per_gpu_batch_size = adjusted_base_batch
        accumulate_grad_batches = 1
    else:
        # Try to fit the adjusted batch size across GPUs
        per_gpu_batch_size = max(1, adjusted_base_batch // world_size)
        
        # Use gradient accumulation if needed to maintain effective batch size
        desired_total_batch = adjusted_base_batch
        actual_total_per_step = per_gpu_batch_size * world_size
        accumulate_grad_batches = max(1, desired_total_batch // actual_total_per_step)
    
    effective_batch_size = per_gpu_batch_size * world_size * accumulate_grad_batches
    
    config = {
        'per_gpu_batch_size': per_gpu_batch_size,
        'accumulate_grad_batches': accumulate_grad_batches,
        'effective_batch_size': effective_batch_size,
        'batch_multiplier': batch_multiplier,
        'optimization_applied': True,
    }
    
    logger.info(f"Distributed batch config: {per_gpu_batch_size} per GPU × {world_size} GPUs × {accumulate_grad_batches} accumulation = {effective_batch_size} effective")
    
    return config

def setup_distributed_environment():
    """
    Set up optimal distributed training environment variables and settings.
    """
    if torch.cuda.is_available():
        # Enable optimal CUDA settings for distributed training
        import os
        
        # NCCL optimizations for modern GPUs
        os.environ.setdefault('NCCL_ALGO', 'Tree')
        os.environ.setdefault('NCCL_PROTO', 'Simple')
        
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        logger.info("Applied CUDA and NCCL optimizations for distributed training")
    
def log_optimization_summary(gpu_capabilities: Dict[str, Any]):
    """Log a summary of applied optimizations."""
    if not gpu_capabilities['has_gpu']:
        logger.info("GPU Optimization Summary: No GPU detected, using CPU")
        return
    
    optimizations = gpu_capabilities['optimizations']
    
    logger.info("GPU Optimization Summary:")
    logger.info(f"  GPUs: {gpu_capabilities['gpu_count']} × {gpu_capabilities['gpu_info'][0]['name'] if gpu_capabilities['gpu_info'] else 'Unknown'}")
    logger.info(f"  Total Memory: {gpu_capabilities['total_memory'] / 1e9:.1f}GB")
    logger.info(f"  Precision: {optimizations.get('precision', '32')}")
    logger.info(f"  Fused Optimizer: {optimizations.get('use_fused_optimizer', False)}")
    logger.info(f"  Torch Compile: {optimizations.get('use_torch_compile', False)}")
    logger.info(f"  Gradient Checkpointing: {optimizations.get('gradient_checkpointing', False)}")
    logger.info(f"  Batch Multiplier: {optimizations.get('batch_size_multiplier', 1.0):.1f}x")