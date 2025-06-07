"""
Distributed training utilities for wind forecasting framework.
"""
import logging
import os
import torch
from typing import Tuple, Dict, Any, Optional
from lightning.pytorch.strategies import DDPStrategy

logger = logging.getLogger(__name__)

def detect_training_environment() -> Dict[str, Any]:
    """
    Detect the current training environment and return configuration.
    
    Returns
    -------
    Dict containing:
        - mode: 'single_gpu', 'tuning', 'distributed_training'
        - world_size: number of processes for distributed training
        - is_distributed: whether we're in true distributed mode
        - should_use_lightning_datamodule: whether to use Lightning DataModule
    """
    # Check if we're using distributed pytorch
    is_distributed_initialized = (
        torch.distributed.is_available() and 
        torch.distributed.is_initialized()
    )
    
    # Get SLURM environment info
    slurm_ntasks = os.environ.get('SLURM_NTASKS_PER_NODE', '1')
    slurm_nnodes = os.environ.get('SLURM_NNODES', '1')
    worker_rank = os.environ.get('WORKER_RANK', None)
    
    try:
        ntasks_per_node = int(slurm_ntasks)
        nnodes = int(slurm_nnodes)
        world_size = ntasks_per_node * nnodes
    except (ValueError, TypeError):
        world_size = 1
    
    # Determine mode
    if worker_rank is not None and not is_distributed_initialized:
        # Independent Optuna workers
        mode = 'tuning'
        should_use_lightning_datamodule = False
    elif world_size > 1 and is_distributed_initialized:
        # True distributed training
        mode = 'distributed_training'
        should_use_lightning_datamodule = True
    elif world_size > 1:
        # Multi-GPU but not yet initialized (will be distributed)
        mode = 'distributed_training'
        should_use_lightning_datamodule = True
    else:
        # Single GPU
        mode = 'single_gpu'
        should_use_lightning_datamodule = False
    
    result = {
        'mode': mode,
        'world_size': world_size,
        'is_distributed': mode == 'distributed_training',
        'should_use_lightning_datamodule': should_use_lightning_datamodule,
        'worker_rank': worker_rank,
        'slurm_ntasks_per_node': ntasks_per_node,
        'slurm_nnodes': nnodes,
    }
    
    logger.info(f"Training environment detected: {result}")
    return result

def should_enable_distributed_optimizations(config: Dict[str, Any], args) -> bool:
    """
    Determine if distributed optimizations should be enabled.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    args : argparse.Namespace
        Command line arguments
        
    Returns
    -------
    bool
        Whether to enable distributed optimizations
    """
    # Feature flag from config (allows disabling for testing)
    feature_enabled = config.get("trainer", {}).get("enable_distributed_optimizations", True)
    
    if not feature_enabled:
        logger.info("Distributed optimizations disabled by config flag")
        return False
    
    # Only enable for training mode (not tuning)
    if args.mode != "train":
        return False
    
    # Check environment
    env_info = detect_training_environment()
    return env_info['should_use_lightning_datamodule']

def calculate_optimal_batch_configuration(
    tuned_batch_size: int, 
    world_size: int,
    min_batch_per_gpu: int = 16,
    gpu_capabilities: Optional[Dict[str, Any]] = None
) -> Tuple[int, int]:
    """
    Calculate optimal per-GPU batch size and gradient accumulation.
    
    Parameters
    ----------
    tuned_batch_size : int
        The optimal batch size determined by hyperparameter tuning
    world_size : int
        Number of processes in distributed training
    min_batch_per_gpu : int
        Minimum batch size per GPU for training stability
    gpu_capabilities : Optional[Dict[str, Any]]
        GPU capabilities from detect_gpu_capabilities()
        
    Returns
    -------
    Tuple[int, int]
        (per_gpu_batch_size, accumulate_grad_batches)
    """
    if world_size <= 1:
        return tuned_batch_size, 1
    
    # Apply GPU-specific batch size optimizations if available
    if gpu_capabilities and gpu_capabilities.get('has_gpu', False):
        optimizations = gpu_capabilities.get('optimizations', {})
        batch_multiplier = optimizations.get('batch_size_multiplier', 1.0)
        adjusted_batch_size = int(tuned_batch_size * batch_multiplier)
        logger.info(f"Applied GPU batch multiplier {batch_multiplier:.1f}x: {tuned_batch_size} -> {adjusted_batch_size}")
    else:
        adjusted_batch_size = tuned_batch_size
    
    # Simple division if possible
    if adjusted_batch_size >= min_batch_per_gpu * world_size:
        per_gpu_batch = adjusted_batch_size // world_size
        accumulate_batches = 1
    else:
        # Use gradient accumulation to maintain effective batch size
        per_gpu_batch = min_batch_per_gpu
        total_desired = adjusted_batch_size
        total_per_step = per_gpu_batch * world_size
        accumulate_batches = max(1, total_desired // total_per_step)
    
    effective_batch = per_gpu_batch * world_size * accumulate_batches
    
    logger.info(
        f"Batch configuration: {per_gpu_batch} per GPU × {world_size} GPUs "
        f"× {accumulate_batches} accumulation = {effective_batch} effective "
        f"(target was {tuned_batch_size})"
    )
    
    return per_gpu_batch, accumulate_batches