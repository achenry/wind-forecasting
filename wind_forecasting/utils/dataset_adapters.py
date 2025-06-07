"""
Dataset adapters for converting between different dataset formats.
"""
import logging
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Iterator
from torch.utils.data import Dataset, IterableDataset
from gluonts.dataset.common import Dataset as GluonTSDataset
from gluonts.transform import Transformation

logger = logging.getLogger(__name__)

class GluonTSIterableDataset(IterableDataset):
    """
    Efficient adapter that converts GluonTS datasets to PyTorch IterableDataset.
    
    This avoids pre-loading all samples and instead transforms them on-the-fly,
    which is more memory efficient for large datasets.
    """
    
    def __init__(
        self, 
        gluonts_dataset: GluonTSDataset,
        transformation: Transformation,
        field_names: List[str],
        is_train: bool = True,
    ):
        """
        Parameters
        ----------
        gluonts_dataset : GluonTSDataset
            The original GluonTS dataset
        transformation : Transformation
            GluonTS transformation pipeline (from estimator)
        field_names : List[str]
            List of field names to extract (e.g., TRAINING_INPUT_NAMES)
        is_train : bool
            Whether this is for training (affects transformation)
        """
        super().__init__()
        self.gluonts_dataset = gluonts_dataset
        self.transformation = transformation
        self.field_names = field_names
        self.is_train = is_train
        
        logger.info(f"Created GluonTSIterableDataset with {len(field_names)} fields")
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterate over the dataset, applying transformations on-the-fly.
        
        Returns
        -------
        Iterator[Dict[str, torch.Tensor]]
            Iterator over samples as dictionaries of tensors
        """
        # Apply transformation to the dataset
        transformed_iter = self.transformation.apply(
            self.gluonts_dataset, 
            is_train=self.is_train
        )
        
        # Yield samples one by one, converting to tensors
        for raw_sample in transformed_iter:
            processed_sample = {}
            
            # Process ALL fields in the sample, not just field_names
            # This ensures we don't lose required fields like 'feat_dynamic_real'
            for field_name, value in raw_sample.items():
                # Convert to tensor if not already
                if not isinstance(value, torch.Tensor):
                    if isinstance(value, np.ndarray):
                        tensor_value = torch.from_numpy(value)
                    else:
                        try:
                            tensor_value = torch.tensor(value)
                        except (TypeError, ValueError):
                            # Some fields might not be convertible to tensors (e.g., strings)
                            # Keep them as-is
                            tensor_value = value
                else:
                    tensor_value = value
                
                processed_sample[field_name] = tensor_value
            
            # Log missing critical fields for debugging
            if not any(field in processed_sample for field in self.field_names):
                logger.warning(f"None of the expected fields {self.field_names} found in sample. "
                             f"Available fields: {list(processed_sample.keys())}")
            
            yield processed_sample

class GluonTSMapDataset(Dataset):
    """
    Map-style dataset adapter for cases where we need random access.
    
    This pre-loads samples but only does transformation once during initialization.
    Use this when you need the dataset length or random access by index.
    """
    
    def __init__(
        self, 
        gluonts_dataset: GluonTSDataset,
        transformation: Transformation,
        field_names: List[str],
        is_train: bool = True,
    ):
        """
        Parameters
        ----------
        gluonts_dataset : GluonTSDataset
            The original GluonTS dataset
        transformation : Transformation
            GluonTS transformation pipeline (from estimator)
        field_names : List[str]
            List of field names to extract (e.g., TRAINING_INPUT_NAMES)
        is_train : bool
            Whether this is for training (affects transformation)
        """
        super().__init__()
        self.field_names = field_names
        self.is_train = is_train
        
        # Pre-transform all samples once during initialization
        logger.info("Pre-transforming GluonTS dataset (this happens once)...")
        self._samples = self._prepare_samples(gluonts_dataset, transformation)
        logger.info(f"Pre-transformation complete: {len(self._samples)} samples ready")
    
    def _prepare_samples(self, gluonts_dataset: GluonTSDataset, transformation: Transformation) -> List[Dict[str, torch.Tensor]]:
        """Pre-apply transformations and convert to tensors."""
        transformed_iter = transformation.apply(gluonts_dataset, is_train=self.is_train)
        samples = []
        
        for raw_sample in transformed_iter:
            processed_sample = {}
            
            # Process ALL fields in the sample, not just field_names
            # This ensures we don't lose required fields like 'feat_dynamic_real'
            for field_name, value in raw_sample.items():
                # Convert to tensor if not already
                if not isinstance(value, torch.Tensor):
                    if isinstance(value, np.ndarray):
                        tensor_value = torch.from_numpy(value)
                    else:
                        try:
                            tensor_value = torch.tensor(value)
                        except (TypeError, ValueError):
                            # Some fields might not be convertible to tensors (e.g., strings)
                            # Keep them as-is
                            tensor_value = value
                else:
                    tensor_value = value
                
                processed_sample[field_name] = tensor_value
            
            samples.append(processed_sample)
        
        return samples
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self._samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample by index."""
        return self._samples[idx]

def create_torch_dataset_from_gluonts(
    gluonts_dataset: GluonTSDataset,
    estimator,
    field_names: List[str],
    is_train: bool = True,
    use_iterable: bool = False,
) -> Dataset:
    """
    Convenience function to create PyTorch dataset from GluonTS dataset.
    
    Parameters
    ----------
    gluonts_dataset : GluonTSDataset
        The GluonTS dataset
    estimator : TACTiS2Estimator
        The estimator with transformation pipeline
    field_names : List[str]
        Field names to extract
    is_train : bool
        Whether this is for training
    use_iterable : bool
        If True, use IterableDataset (memory efficient)
        If False, use map-style Dataset (allows random access)
        
    Returns
    -------
    Dataset
        PyTorch-compatible dataset
    """
    transformation = estimator.create_transformation()
    
    if use_iterable:
        return GluonTSIterableDataset(
            gluonts_dataset=gluonts_dataset,
            transformation=transformation,
            field_names=field_names,
            is_train=is_train,
        )
    else:
        return GluonTSMapDataset(
            gluonts_dataset=gluonts_dataset,
            transformation=transformation,
            field_names=field_names,
            is_train=is_train,
        )