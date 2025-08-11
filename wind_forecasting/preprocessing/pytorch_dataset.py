"""
PyTorch Dataset for wind forecasting data that enables distributed training.

This module provides a PyTorch Dataset that loads pre-split data from pickle files
and applies necessary transformations for model training.
"""

import logging
import pickle
# from memory_profiler import profile
from line_profiler import profile
from typing import List, Optional, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, Dataset
from itertools import cycle, islice
from torch.utils.data import DataLoader
import lightning.pytorch as L
import polars as pl
import polars.selectors as cs
from itertools import pairwise

logger = logging.getLogger(__name__)

class WindForecastingDatamodule(L.LightningDataModule):
    def __init__(self, train_data_path, val_data_path, train_sampler, 
                 context_length, prediction_length, time_features, val_sampler=None, 
                 train_repeat=False, val_repeat=False,
                 batch_size=32, num_workers=4, pin_memory=True):
        super().__init__()
        # self.train_data_path = train_data_path
        # self.val_data_path = val_data_path
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.time_features = time_features
        
        self.train_repeat = train_repeat
        self.val_repeat = val_repeat
        
        self.feat_static_real = torch.tensor([0]).float()
        
        # self.train_data_path = train_data_path
        # self.val_data_path = val_data_path
        
        # These will be set in the setup() hook
        self.world_size = 1
        self.rank = 0
        
        # Load data from pickle
        logger.info(f"Loading dataset from {train_data_path}")
        if train_data_path.endswith('.pkl'):
            with open(train_data_path, 'rb') as f:
                self.train_data = pickle.load(f) # TODO HIGH REST OF CODE WON'T WORK WITH PICKLE
        else:
            self.train_data = pl.read_parquet(train_data_path)
              
        logger.info(f"Loading dataset from {val_data_path}")
        if val_data_path.endswith('.pkl'):
            with open(val_data_path, 'rb') as f:
                self.val_data = pickle.load(f)
        else:
            self.val_data = pl.read_parquet(val_data_path)

    def _create_time_features(self, time_index: pd.PeriodIndex) -> np.ndarray:
        """
        Create time features from time index.
        
        Parameters
        ----------
        time_index : pd.PeriodIndex
            Time index to create features from
            
        Returns
        -------
        np.ndarray
            Shape (num_features, time_steps)
        """
        logging.info(f"Creating time features on rank {self.rank} for world_size {self.world_size}.")
        if not self.time_features:
            # Default to empty features
            return np.zeros((len(time_index), 0))
        
        features = []
        for feat_func in self.time_features:
            # Apply time feature function
            feat = feat_func(time_index)
            features.append(feat)
            
        return np.vstack(features).T

    def setup(self, stage: str):
        # This hook is called on each DDP process, so `self.trainer` is available.
        
        if self.trainer:
            self.rank = self.trainer.global_rank
            self.world_size = self.trainer.world_size
            
        logging.info(f"Running WindForecastingDatamodule.setup() on rank {self.rank} for world_size {self.world_size}")
        
        for split in ["train", "val"]:
            logging.info(f"Setting {split} attributes in WindForecastingDatamodule.setup() on rank {self.rank} for world_size {self.world_size}")
            
            data = getattr(self, f"{split}_data")
            
            if isinstance(data, pl.DataFrame):
                # NOTE: below works, testing other
                # # join with lenghts of each continuous time series in the dataset
                # # ensure data is ordered by item_id
                # time_addr = data.group_by("item_id", maintain_order=True).agg(pl.len())["len"].to_numpy()
                # # setattr(self, f"{split}_ds_addr", torch.from_numpy(np.arange(len(time_addr - 1))))
                # ds_addr = torch.from_numpy(np.arange(len(time_addr - 1)))
                # time_addr = np.insert(time_addr.cumsum(), 0, 0)
                
                # setattr(self, 
                #     f"{split}_data_time", 
                #     torch.from_numpy(self._create_time_features(
                #     pd.to_datetime(data.select(pl.col("time")).to_numpy().squeeze())
                # )))
                # setattr(self, 
                #     f"{split}_data_target", 
                #     torch.from_numpy(data.select(cs.starts_with("target_")).to_numpy().T))
                # setattr(self, 
                #     f"{split}_data_fdr", 
                #     torch.from_numpy(data.select(cs.starts_with("feat_dynamic_real_")).to_numpy().T))
                # setattr(self, 
                #         f"{split}_data_fsc", 
                # torch.from_numpy(np.vstack(np.concatenate(
                # data.select(pl.col("feat_static_cat")).with_row_index().filter(pl.col("index").is_in(time_addr[:-1]))["feat_static_cat"].to_numpy()))))
                # # self.time_addr = torch.from_numpy(self.time_addr)
                # setattr(self, f"{split}_time_addr", torch.from_numpy(time_addr))
                
                ds = {
                    "time_addr": data.group_by("item_id", maintain_order=True).agg(pl.len())["len"].to_numpy(),
                    "feat_dynamic_real": torch.from_numpy(data.select(cs.starts_with("feat_dynamic_real_")).to_numpy()),
                    "time": torch.from_numpy(self._create_time_features(
                    pd.to_datetime(data.select(pl.col("time")).to_numpy().squeeze()))),
                    "target": torch.from_numpy(data.select(cs.starts_with("target_")).to_numpy()),
                }
                
                ds["ds_addr"] = np.arange(len(ds["time_addr"] - 1))
                ds["time_addr"] = np.insert(ds["time_addr"].cumsum(), 0, 0)
                ds["feat_static_cat"] = torch.from_numpy(
                        np.vstack(np.concatenate(
                                    data.select(pl.col("feat_static_cat"))\
                                               .with_row_index()\
                                                   .filter(pl.col("index").is_in(ds["time_addr"][:-1]))["feat_static_cat"].to_numpy()
                                    ))).long()
                ds["observed"] = (~torch.isnan(ds["target"])).float()
                ds["target"] = torch.nan_to_num(ds["target"], 0.0).float()
                ds["time"] = torch.hstack([ds["time"], ds["feat_dynamic_real"]]).float()
                ds["feat_static_real"] = torch.tensor([0]).float()
                del ds["feat_dynamic_real"]
                
            else:
                raise NotImplementedError
            
            setattr(self, f"{split}_data", ds)
        

    def train_dataloader(self):
        # The Trainer calls this after setup() on each DDP process
        logging.info(f"Instantiating WindForecastingDataset in WindForecastingDatamodule.train_dataloader() on rank {self.rank} for world_size {self.world_size}")
        
        train_dataset = WindForecastingDataset(
                data=self.train_data,
                context_length=self.context_length,
                prediction_length=self.prediction_length,
                # time_features=self.time_features,
                sampler=self.train_sampler,  # GluonTS sampler instance
                repeat=self.train_repeat,
                skip_indices=1,
                world_size=self.world_size,
                rank=self.rank)
        
        logging.info(f"Returning DataLoader in WindForecastingDatamodule.train_dataloader() on rank {self.rank} for world_size {self.world_size}")
        
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Will be overridden by DistributedSampler in DDP
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers>0,
            prefetch_factor=2 if (self.num_workers > 0) else None
            # drop_last=True,  # Important for DDP to avoid uneven batch sizes
        )
        
    def val_dataloader(self):
        # This is also called at the correct time.
        # It will correctly shard the validation data across all GPUs and workers.
        
        logging.info(f"Instantiating WindForecastingDataset in WindForecastingDatamodule.val_dataloader() on rank {self.rank} for world_size {self.world_size}")
        
        val_dataset = WindForecastingInferenceDataset(
                data=self.val_data,
                context_length=self.context_length,
                prediction_length=self.prediction_length,
                sampler=self.val_sampler,  # GluonTS sampler instance
                repeat=self.val_repeat,
                skip_indices=self.prediction_length,
                world_size=self.world_size,
                rank=self.rank)
        
        if val_dataset is None:
            return None
        
        logging.info(f"Returning DataLoader in WindForecastingDatamodule.val_dataloader() on rank {self.rank} for world_size {self.world_size}")
        
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Never shuffle validation data
            # worker_init_fn=self.__class__._worker_init_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers>0,
            prefetch_factor=2 if (self.num_workers > 0) else None
            # drop_last=False,  # Keep all validation samples
        )

    def test_dataloader(self):
        # The same pattern applies for testing.
        if self.test_dataset is None:
            return None
            
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Never shuffle validation data
            # worker_init_fn=self.__class__._worker_init_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers>0,
            prefetch_factor=2 if (self.num_workers > 0) else None
            # drop_last=False,  # Keep all validation samples
        )


# def _serialize(data):
#     buffer = pickle.dumps(data, protocol=-1)
#     return np.frombuffer(buffer, dtype=np.uint8)
    
# class WindForecastingDataset(IterableDataset):
class WindForecastingDataset(IterableDataset):
    """
    PyTorch Dataset for wind forecasting data.
    
    Loads pre-split data from pickle files and applies transformations
    needed for training (time features, windowing).
    
    Parameters
    ----------
    data_path : str
        Path to the pickle file containing the dataset
    context_length : int
        Number of past time steps to use as context
    prediction_length : int
        Number of future time steps to predict
    time_features : List[callable]
        List of time feature functions to apply
    """
    
    # @profile
    def __init__(
        self,
        data: dict,
        context_length: int,
        prediction_length: int,
        sampler: Optional[Any] = None,  # GluonTS sampler instance
        repeat: bool = False,
        skip_indices: int = 1,
        rank: int = 0,
        world_size: int = 1
    ):
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.sampler = sampler
        self.repeat = repeat
        
        self.skip_indices = skip_indices
        
        # Store world_size and rank passed from the DataModule
        self.world_size = world_size
        self.rank = rank
        
        self.data_time = data["time"]
        self.data_target = data["target"]
        # self.data_fdr = data["feat_dynamic_real"]
        self.data_fsc = data["feat_static_cat"]
        self.time_addr = data["time_addr"]
        self.ds_addr = data["ds_addr"]
        self.data_observed = data["observed"]
        self.data_feat_static_real = data["feat_static_real"]
        
        logging.info(f"Instantiating data attributes in WindForecastingDataset.__init__ with rank = {self.rank} and world_size = {self.world_size}")
    
    @profile
    def __iter__(self):
        
        # if self.world_size > 1:
        #     logger.info(f"Using distributed training with rank={self.rank}, world_size={self.world_size}")
        # else:
        #     logger.info(f"Using single-rank training with rank={self.rank}, world_size={self.world_size}")
                
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None: # Main process, num_workers=0 case
            # logger.info(f"training worker_info is None, on main process fetching islice {self.rank}:None:{self.world_size}")
            return islice(self._base_iter(), self.rank, None, self.world_size)
        else: # In a worker process
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            
            global_num_workers = num_workers * self.world_size
            global_worker_id = self.rank * num_workers + worker_id
            # logger.info(f"training worker {worker_info.id} of {num_workers}, fetching islice {global_worker_id}:None:{global_num_workers}")

            return islice(self._base_iter(), global_worker_id, None, global_num_workers)
    
    @profile
    def _base_iter(self):
    
        """
        Get an iterable of samples with transformations applied.
        
        Returns
        -------
        Dictionary containing:
            - past_target: (context_length, num_series)
            - future_target: (prediction_length, num_series)
            - past_time_feat: (context_length, num_time_features)
            - future_time_feat: (prediction_length, num_time_features)
            - past_observed_values: (context_length, num_series)
            - future_observed_values: (prediction_length, num_series)
            - feat_static_cat: (num_static_cat,)
            - feat_static_real: (num_static_real,)
        """
        # logging.info(f"Running WindForecastingDataset._base_iter() on rank {self.rank} with world_size {self.world_size}")
        
        # Apply cycle() here if needed, on the new iterator.
        time_addr = cycle(self.time_addr) if self.repeat else self.time_addr
        ds_addr = cycle(self.ds_addr) if self.repeat else self.ds_addr
        
        for ds_idx, (start_addr, end_addr) in zip(ds_addr, pairwise(time_addr)):
            
            time = self.data_time[start_addr:end_addr, :]
            target = self.data_target[start_addr:end_addr, :]
            feat_static_cat = self.data_fsc[ds_idx, :]
            observed = self.data_observed[start_addr:end_addr, :]
            
            sampled_indices = self.sampler(target.T)[::self.skip_indices]
            
            for idx in sampled_indices:
                # Extract data
                # ts_length = target.shape[1]
                
                # Find all valid time points
                # min_time = self.context_length
                # max_time = ts_length - self.prediction_length
                
                if target.shape[0] - self.prediction_length < self.context_length:
                    continue
                
                # Split into past and future windows
                context_slice, pred_slice = slice(idx - self.context_length, idx), slice(idx, idx + self.prediction_length)
                
                # past_target, future_target = target[context_slice, :], target[pred_slice, :]
                
                # past_time_feat, future_time_feat = time[context_slice, :], time[pred_slice, :]
                
                # past_observed, future_observed = observed[context_slice, :], observed[pred_slice, :]
                
                yield (
                    target[context_slice, :],
                    target[pred_slice, :],
                    time[context_slice, :],
                    time[pred_slice, :],
                    observed[context_slice, :],
                    observed[pred_slice, :],
                    feat_static_cat,
                    self.data_feat_static_real,
                )
                # np.array(['past_target', 'future_target', 'past_time_feat', 
                #              'future_time_feat', 'past_observed_values', 'future_observed_values',
                #              'feat_static_cat', 'feat_static_real']).astype(np.string_)


class WindForecastingInferenceDataset(WindForecastingDataset):
    """
    Dataset for inference that creates windows at all valid time points.
    
    Unlike the training dataset which samples random windows, this creates
    all possible windows for complete evaluation.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @profile
    def _base_iter(self):
        """Get a specific window for inference."""
        
        # logging.info(f"Running WindForecastingInferenceDataset._base_iter() on rank {self.rank} with world_size {self.world_size}")
        
        for ds_idx, (start_addr, end_addr) in zip(self.ds_addr, pairwise(self.time_addr)):
            
            time = self.data_time[start_addr:end_addr, :]
            target = self.data_target[start_addr:end_addr, :]
            observed = self.data_observed[start_addr:end_addr, :]
            # feat_dynamic_real = self.data_fdr[start_addr:end_addr, :]
            feat_static_cat = self.data_fsc[ds_idx, :]
            
            # start_addr = end_addr
            
            # Same processing as parent class but with fixed time point
            # ts_length = target.shape[1]
        
            # Find all valid time points
            # min_time = self.context_length
            # max_time = ts_length - self.prediction_length
            
            # Fill time indices
            sample_indices = np.arange(self.context_length, target.shape[0] - self.prediction_length + 1, self.skip_indices)
            
            for idx in sample_indices:
                # Split into past and future windows at fixed time point t
                context_slice = slice(idx - self.context_length, idx)
                pred_slice = slice(idx, idx + self.prediction_length)
                
                # past_target, future_target = target[context_slice, :], target[pred_slice, :]
                
                # past_time_feat, future_time_feat = time[context_slice, :], time[pred_slice, :]
                
                # past_observed, future_observed = observed[context_slice, :], observed[pred_slice, :]
                
                # Convert to tensors
                yield (
                    target[context_slice, :],
                    target[pred_slice, :],
                    time[context_slice, :],
                    time[pred_slice, :],
                    observed[context_slice, :],
                    observed[pred_slice, :],
                    feat_static_cat,
                    self.data_feat_static_real
                )
                # np.array(["past_target", "future_target", "past_time_feat", 
                #              "future_time_feat", "past_observed_values", "future_observed_values",
                #              "feat_static_cat", "feat_static_real"]).astype(np.string_)