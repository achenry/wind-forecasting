"""
Distributed Lightning DataModule for wind forecasting.
"""
import logging
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule
from typing import Optional, Dict, Any

from wind_forecasting.utils.dataset_adapters import create_torch_dataset_from_gluonts
from pytorch_transformer_ts.tactis_2.estimator import TRAINING_INPUT_NAMES, PREDICTION_INPUT_NAMES

class SimpleDataset:
    """Simple dataset wrapper for list of samples."""
    def __init__(self, samples):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

logger = logging.getLogger(__name__)

def get_default_dataloader_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get DataLoader configuration from config with sensible defaults.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
        
    Returns
    -------
    Dict[str, Any]
        DataLoader configuration parameters
    """
    # Get num_workers from SLURM environment or config
    cpus_per_task_str = os.environ.get('SLURM_CPUS_PER_TASK', None)
    
    if cpus_per_task_str:
        try:
            cpus_per_task = int(cpus_per_task_str)
            default_workers = max(0, cpus_per_task - 1)  # Leave 1 for main process
        except ValueError:
            default_workers = 4
    else:
        default_workers = 4
    
    # Get dataloader settings from config with defaults
    dataloader_config = config.get("dataloader", {})
    
    # Handle explicit None/null values in config
    config_num_workers = dataloader_config.get("num_workers", default_workers)
    if config_num_workers is None:
        config_num_workers = default_workers
    
    return {
        "num_workers": config_num_workers,
        "pin_memory": dataloader_config.get("pin_memory", torch.cuda.is_available()),
        "persistent_workers": dataloader_config.get("persistent_workers", True),
        "prefetch_factor": dataloader_config.get("prefetch_factor", 2),
    }

class DistributedWindForecastingDataModule(LightningDataModule):
    """
    Lightning DataModule optimized for distributed wind forecasting training.
    
    This module leverages Lightning's automatic DistributedSampler injection
    while preserving all existing GluonTS transformation logic.
    """
    
    def __init__(
        self,
        estimator,
        train_dataset,
        val_dataset,
        per_gpu_batch_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        **kwargs
    ):
        """
        Parameters
        ----------
        estimator : TACTiS2Estimator
            The estimator with transformation pipeline
        train_dataset : GluonTSDataset
            Training dataset from DataModule
        val_dataset : GluonTSDataset
            Validation dataset from DataModule
        per_gpu_batch_size : int
            Batch size per GPU (already calculated for world size)
        num_workers : int
            Number of DataLoader workers per GPU
        pin_memory : bool
            Whether to use pinned memory
        persistent_workers : bool
            Whether to keep workers alive between epochs
        prefetch_factor : int
            Number of batches to prefetch per worker
        """
        super().__init__()
        
        # Store parameters
        self.estimator = estimator
        self.original_train_dataset = train_dataset
        self.original_val_dataset = val_dataset
        self.per_gpu_batch_size = per_gpu_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        
        # Store for setup
        self.train_torch_dataset = None
        self.val_torch_dataset = None
        
        logger.info(
            f"DistributedWindForecastingDataModule initialized: "
            f"batch_size={per_gpu_batch_size}, workers={num_workers}, "
            f"pin_memory={pin_memory}, prefetch={prefetch_factor}"
        )
    
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets. Called on every process in DDP.
        
        Parameters
        ----------
        stage : str, optional
            'fit', 'validate', 'test', or 'predict'
        """
        if stage == "fit" or stage is None:
            logger.info("Setting up training and validation datasets...")
            
            # Convert already-transformed GluonTS datasets to PyTorch format
            # The data is already transformed by the estimator, so we just need to convert formats
            self.train_torch_dataset = self._convert_transformed_dataset_to_torch(
                self.original_train_dataset
            )
            
            if self.original_val_dataset is not None:
                self.val_torch_dataset = self._convert_transformed_dataset_to_torch(
                    self.original_val_dataset
                )
            else:
                self.val_torch_dataset = None
            
            logger.info(
                f"Setup complete: train={len(self.train_torch_dataset)}, "
                f"val={len(self.val_torch_dataset) if self.val_torch_dataset else 0} samples"
            )
    
    def _convert_transformed_dataset_to_torch(self, transformed_dataset):
        """
        Convert already-transformed GluonTS dataset to PyTorch dataset.
        
        This is simpler than the general adapters since the data is already 
        transformed and we just need to convert to PyTorch tensors.
        """
        logger.info("Converting transformed GluonTS dataset to PyTorch format...")
        samples = []
        
        for sample in transformed_dataset:
            processed_sample = {}
            
            # Convert all fields to tensors
            for field_name, value in sample.items():
                if not isinstance(value, torch.Tensor):
                    if isinstance(value, np.ndarray):
                        tensor_value = torch.from_numpy(value)
                    else:
                        try:
                            # Try to convert to tensor
                            tensor_value = torch.tensor(value)
                        except (TypeError, ValueError, RuntimeError) as e:
                            # Handle special cases that can't be converted to tensors
                            if hasattr(value, '__class__') and 'Period' in str(value.__class__):
                                # Skip pandas Period objects - they're not needed for training
                                continue
                            elif isinstance(value, str):
                                # Keep strings as-is
                                tensor_value = value
                            else:
                                # Log warning for debugging and keep original value
                                logger.warning(f"Could not convert field '{field_name}' to tensor: {e}. Keeping original value.")
                                tensor_value = value
                else:
                    tensor_value = value
                
                processed_sample[field_name] = tensor_value
            
            samples.append(processed_sample)
        
        logger.info(f"Converted {len(samples)} samples to PyTorch format")
        if samples:
            logger.info(f"Sample fields: {list(samples[0].keys())}")
        
        return SimpleDataset(samples)
    
    def train_dataloader(self) -> DataLoader:
        """
        Create training DataLoader.
        Lightning automatically handles DistributedSampler injection.
        """
        return DataLoader(
            self.train_torch_dataset,
            batch_size=self.per_gpu_batch_size,
            shuffle=True,  # Lightning converts this to DistributedSampler in DDP
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=True,  # Ensure consistent batch sizes across GPUs
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader."""
        return DataLoader(
            self.val_torch_dataset,
            batch_size=self.per_gpu_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=False,
        )
    
    def teardown(self, stage: Optional[str] = None):
        """Cleanup after training."""
        # Clear cached datasets to free memory
        self.train_torch_dataset = None
        self.val_torch_dataset = None
        logger.info("DataModule teardown complete")

def create_distributed_datamodule(
    estimator,
    train_dataset,
    val_dataset,
    per_gpu_batch_size: int,
    config: Dict[str, Any],
) -> DistributedWindForecastingDataModule:
    """
    Factory function to create distributed DataModule.
    
    Parameters
    ----------
    estimator : TACTiS2Estimator
        The estimator
    train_dataset, val_dataset : GluonTSDataset
        Datasets from existing DataModule
    per_gpu_batch_size : int
        Calculated batch size per GPU
    config : dict
        Full configuration dictionary
        
    Returns
    -------
    DistributedWindForecastingDataModule
        Configured DataModule
    """
    # Get DataLoader configuration from config
    dataloader_config = get_default_dataloader_config(config)
    
    return DistributedWindForecastingDataModule(
        estimator=estimator,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        per_gpu_batch_size=per_gpu_batch_size,
        **dataloader_config,
    )