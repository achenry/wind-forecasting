"""
PyTorch Dataset for wind forecasting data that enables distributed training.

This module provides a PyTorch Dataset that loads pre-split data from pickle files
and applies necessary transformations for model training.
"""

import logging
import pickle
from memory_profiler import profile
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

# def make_df(data):
    
#     dfs = []
#     for sample in data:
#         # Convert each sample to a DataFrame
#         df = {}
#         for k in sample:
#             if k in ["target", "feat_dynamic_real"]:
#                 for c in range(sample[k].shape[0]):
#                     df[f"{k}_{c}"] = sample[k][c, :]
#             elif k in ["feat_static_cat", "feat_static_real", "item_id"]:
#                 df[k] = [sample[k]] * sample["target"].shape[1]
#             elif k == "start":
#                 # df[k] = pd.period_range(
#                 #     start=sample[k],
#                 #     periods=sample["target"].shape[1],
#                 #     freq=sample[k].freq
#                 # )
#                 df["time"] = pl.datetime_range(
#                     start=sample[k].start_time,
#                     end=sample[k].start_time + (pd.Timedelta(sample[k].freq) * sample["target"].shape[1]),
#                     interval=sample[k].freqstr,
#                     eager=True,
#                     time_unit="ns",
#                     closed="left"
#                 ).alias("time")
                
#         dfs.append(pl.DataFrame(df))
        
#     # Concatenate all DataFrames into one
#     return pl.concat(dfs, how="vertical")

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
        
        # Load data from pickle
        logger.info(f"Loading dataset from {train_data_path}")
        if train_data_path.endswith('.pkl'):
            with open(train_data_path, 'rb') as f:
                self.train_data = pickle.load(f) # TODO HIGH REST OF CODE WON'T WORK WITH PICKLE
        else:
            self.train_data = pl.read_parquet(train_data_path)
            
            
        logger.info(f"Loading dataset from {val_data_path}")
        if train_data_path.endswith('.pkl'):
            with open(val_data_path, 'rb') as f:
                self.val_data = pickle.load(f)
        else:
            self.val_data = pl.read_parquet(val_data_path)
        
        self.train_repeat = train_repeat
        self.val_repeat = val_repeat
        
        # These will be set in the setup() hook
        self.world_size = 1
        self.rank = 0

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
        if not self.time_features:
            # Default to empty features
            return np.zeros((0, len(time_index)))
        
        features = []
        for feat_func in self.time_features:
            # Apply time feature function
            feat = feat_func(time_index)
            features.append(feat)
            
        return np.array(features)

    def setup(self, stage: str):
        # This hook is called on each DDP process, so `self.trainer` is available.
        if self.trainer:
            self.rank = self.trainer.global_rank
            self.world_size = self.trainer.world_size
        
        for split in ["train", "val"]:
            data = getattr(self, f"{split}_data")
            
            if isinstance(data, pl.DataFrame):
                # join with lenghts of each continuous time series in the dataset
                # ensure data is ordered by item_id
                time_addr = data.group_by("item_id", maintain_order=True).agg(pl.len())["len"].to_numpy()
                setattr(self, f"{split}_ds_addr", torch.from_numpy(np.arange(len(time_addr - 1))))
                time_addr = np.insert(time_addr.cumsum(), 0, 0)
                
                # the address corresponding to each new item_id time series
                # self.ds_addr = self.ds_addr.unique(maintain_order=True)["count"].to_numpy()
                # self.data_time, self.data_target, self.data_fdr, self.data_fsc = (
                    # data.select(pl.col("time")).to_numpy().squeeze(), 
                setattr(self, 
                    f"{split}_data_time", 
                    torch.from_numpy(self._create_time_features(
                    pd.to_datetime(data.select(pl.col("time")).to_numpy().squeeze())
                )))
                setattr(self, 
                    f"{split}_data_target", 
                    torch.from_numpy(data.select(cs.starts_with("target_")).to_numpy().T))
                setattr(self, 
                    f"{split}_data_fdr", 
                    torch.from_numpy(data.select(cs.starts_with("feat_dynamic_real_")).to_numpy().T))
                setattr(self, 
                        f"{split}_data_fsc", 
                torch.from_numpy(np.vstack(np.concatenate(
                data.select(pl.col("feat_static_cat")).with_row_index().filter(pl.col("index").is_in(time_addr[:-1]))["feat_static_cat"].to_numpy()))))
                # self.time_addr = torch.from_numpy(self.time_addr)
                setattr(self, f"{split}_time_addr", torch.from_numpy(time_addr))
                
            else:
                raise NotImplementedError
        

    def train_dataloader(self):
        # The Trainer calls this after setup() on each DDP process
        train_dataset = WindForecastingDataset(
                data_time=self.train_data_time,
                data_target=self.train_data_target,
                data_fdr=self.train_data_fdr,
                data_fsc=self.train_data_fsc,
                ds_addr=self.train_ds_addr,
                time_addr=self.train_time_addr,
                context_length=self.context_length,
                prediction_length=self.prediction_length,
                # time_features=self.time_features,
                sampler=self.train_sampler,  # GluonTS sampler instance
                repeat=self.train_repeat,
                skip_indices=1,
                world_size=self.world_size,
                rank=self.rank)
        
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
        # if self.val_dataset is None:
        #     return None
        
        val_dataset = WindForecastingInferenceDataset(
                data_time=self.val_data_time,
                data_target=self.val_data_target,
                data_fdr=self.val_data_fdr,
                data_fsc=self.val_data_fsc,
                ds_addr=self.val_ds_addr,
                time_addr=self.val_time_addr,
                context_length=self.context_length,
                prediction_length=self.prediction_length,
                # time_features=self.time_features,
                sampler=self.val_sampler,  # GluonTS sampler instance
                repeat=self.val_repeat,
                skip_indices=self.prediction_length,
                world_size=self.world_size,
                rank=self.rank)
        
        if val_dataset is None:
            return None
        
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


def _serialize(data):
    buffer = pickle.dumps(data, protocol=-1)
    return np.frombuffer(buffer, dtype=np.uint8)
    
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
    
    @profile
    def __init__(
        self,
        data_time: torch.Tensor,
        data_target: torch.Tensor,
        data_fdr: torch.Tensor,
        data_fsc: torch.Tensor,
        time_addr: torch.Tensor,
        ds_addr: torch.Tensor,
        context_length: int,
        prediction_length: int,
        # time_features: Optional[List] = None,
        sampler: Optional[Any] = None,  # GluonTS sampler instance
        repeat: bool = False,
        skip_indices: int = 1,
        rank: int = 0,
        world_size: int = 1
    ):
        self.context_length = context_length
        self.prediction_length = prediction_length
        # self.time_features = time_features or []
        self.sampler = sampler
        self.repeat = repeat
        
        # if self.repeat:
        #     logger.info(f"Dataset will repeat indefinitely")
        # else:
        #     logger.info(f"Dataset will not repeat, will stop after one pass")
            
        self.skip_indices = skip_indices
        
        # Store world_size and rank passed from the DataModule
        self.world_size = world_size
        self.rank = rank
        
        self.data_time = data_time
        self.data_target = data_target
        self.data_fdr = data_fdr
        self.data_fsc = data_fsc
        self.time_addr = time_addr
        self.ds_addr = ds_addr
        
        # Validate data format
        # if len(self.data) > 0:
        #     sample = self.data[0]
            # assert 'target' in sample, "Dataset must contain 'target' field"
            # assert 'start' in sample, "Dataset must contain 'start' field"
        
        # assert len(np.unique([sample["target"].shape[0] for sample in self.data])) == 1, "All samples must have the same number of target variables"
        # assert len(np.unique([sample["feat_dynamic_real"].shape[0] for sample in self.data])) == 1, "All samples must have the same number of dynamic features"
        # assert all([sample["target"].shape[1] == sample["feat_dynamic_real"].shape[1] for sample in self.data]), "length of target and feat_dynamic_real must match"
        
        # self.num_target = self.data[0]['target'].shape[0]
        # self.num_feat_dynamic_real = self.data[0]['feat_dynamic_real'].shape[0]
        
        
        logging.info(f"Instantiating data attributes in WindForecastingDataset.__init__ with rank = {self.rank} and world_size = {self.world_size}")

        
        
            # TODO HIGH this will only work for dataframe not for numpy pkl, add earlier code back in
        #     self.ds_addr = data.group_by("item_id", maintain_order=True).agg(pl.len())["len"].to_numpy()
        #     # data, self.ds_addr = pl.align_frames(data, 
        #     #                                      data.select(pl.col("item_id").value_counts()).unnest("item_id"), 
        #     #                                      how="left", on="item_id")
            
        #     # the address corresponding to each new item_id time series
        #     # self.ds_addr = self.ds_addr.unique(maintain_order=True)["count"].to_numpy()
        #     self.data_time = pd.period_range(
        #                 start=data["start"],
        #                 periods=data[],
        #                 freq=start_period.freq
        #             )
        #     self.data_target = data.select(cs.starts_with("target_")).to_numpy().T
        #     self.data_fdr = data.select(cs.starts_with("feat_dynamic_real_")).to_numpy().T
        #     self.data_fsc = np.vstack(np.concatenate(
        #         data.select(pl.col("feat_static_cat")).with_row_index().filter(pl.col("index").is_in(self.ds_addr))["feat_static_cat"].to_numpy()))
            
        #     self.ds_addr = self.ds_addr.cumsum()
        
        # del data
        
        # serialize into torch tensors to reduce memory usage
        # self._data_keys = list(self.data[0].keys())
        # for k in self._data_keys:
        #     if isinstance(self.data[0][k], (np.ndarray, list)):
        #         setattr(self, f"_data_{k}", 
        #                     # [_serialize(item) for sample in self.data for item in np.array(sample[k]).flatten()]
        #                     [_serialize(np.array(sample[k]).flatten()) for sample in self.data]
        #                 )
        #         setattr(self, f"_dim_{k}", np.array(self.data[0][k]).shape[0])
        #        # setattr(self, f"_addr_{k}", torch.from_numpy(np.cumsum(np.asarray([len(item) for item in getattr(self, f"_data_{k}")], dtype=np.int64))))
        #     else:
        #         setattr(self, f"_data_{k}",  [_serialize(sample[k]) for sample in self.data])

        # # at this point, len(self._data_target) == len(self.data) i.e. an item for every time series sample
        
        # self._addr = torch.from_numpy(
        #     np.vstack([np.cumsum(np.asarray([len(item) for item in getattr(self, f"_data_{k}")], dtype=np.int64)) 
        #                for k in self._data_keys ]))
        #             #    if isinstance(self.data[0][k], (np.ndarray, list))]))
        
        # for k in self._data_keys:
        #     if isinstance(getattr(self, f"_data_{k}"), list):
        #         setattr(self, f"_data_{k}", 
        #                 torch.from_numpy(np.concatenate(getattr(self, f"_data_{k}"))))
            # elif isinstance(getattr(self, f"_data_{k}"), np.ndarray):
            #     setattr(self, f"_data_{k}", torch.from_numpy(getattr(self, f"_data_{k}")))
        # self.n_datasets = len(self.ds_addr)
        # if isinstance(self.data, pl.DataFrame):
        #     # self.n_datasets = self.data.select(pl.col("item_id").n_unique()).item()
        #     self.n_datasets = len(self.ds_addr)
        # else:
            # self.n_datasets = len(self.data)
        # logger.info(f"Loaded {self.n_datasets} time series")

        # del self.data # Free memory after loading
        # self.dataset_idx = 0
    
    @profile
    def __iter__(self):
        
        if self.world_size > 1:
            logger.info(f"Using distributed training with rank={self.rank}, world_size={self.world_size}")
        else:
            logger.info(f"Using single-rank training with rank={self.rank}, world_size={self.world_size}")
                
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None: # Main process, num_workers=0 case
            logger.info(f"training worker_info is None, on main process fetching islice {self.rank}:None:{self.world_size}")
            return islice(self._base_iter(), self.rank, None, self.world_size)
        else: # In a worker process
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            
            global_num_workers = num_workers * self.world_size
            global_worker_id = self.rank * num_workers + worker_id
            logger.info(f"training worker {worker_info.id} of {num_workers}, fetching islice {global_worker_id}:None:{global_num_workers}")

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
        
        # Create a NEW, FRESH iterator from the source list every time.
        # data_iterator = iter(self.data)
        
        # Apply cycle() here if needed, on the new iterator.
        # if self.repeat:
        #     data_iterator = cycle(data_iterator)
        
        # iterators = {}
        # for k in self._data_keys:
        #     iterators[k] = iter(getattr(self, f"_data_{k}"))
        
        # addr = self._addr.numpy()
        # addr = {
        #     k: addr[i, :] for i, k in enumerate(self._data_keys)
        # }
            
        # addr_iterator = zip(iter(self._data_start), iter(addr["start"]), iter(addr["target"]), iter(addr["feat_dynamic_real"]), iter(addr["feat_static_cat"]))
        
        # Apply cycle() here if needed, on the new iterator.
        time_addr = cycle(self.time_addr) if self.repeat else self.time_addr
        ds_addr = cycle(self.ds_addr) if self.repeat else self.ds_addr
        
        # to reconstruct the original data, you can use:
        # np.reshape(self.data[0]["target"].flatten(), (self._dim_target, -1)),
        
        # for entry in data_iterator:
        # ds_idx = 0
        # for start_period, end_addr_start, end_addr_target, end_addr_fdr, end_addr_fsc in addr_iterator:
        # for ds in self.data.partition_by("item_id"):
        for ds_idx, (start_addr, end_addr) in zip(ds_addr, pairwise(time_addr)):
            
            # if ds_idx == 0:
            #     start_addr_start = start_addr_target = start_addr_fdr = start_addr_fsc = 0
            # else:
            #     start_addr_start, start_addr_target, start_addr_fdr, start_addr_fsc = (
            #         last_addr_start, last_addr_target, last_addr_fdr, last_addr_fsc
            #     )
            
            # item_id = pickle.loads(memoryview(self._data_item_id[start_addr_item_id:end_addr_item_id].numpy()))
            # start_period = pickle.loads(memoryview(self._data_start[start_addr_start:end_addr_start].numpy()))
            # target =  pickle.loads(memoryview(self._data_target[start_addr_target:end_addr_target].numpy())).reshape((self._dim_target, -1))
            # # start_period = self._data_start[ds_idx]
            # feat_static_cat = pickle.loads(memoryview(self._data_feat_static_cat[start_addr_fsc:end_addr_fsc].numpy()))
            # feat_dynamic_real = pickle.loads(memoryview(self._data_feat_dynamic_real[start_addr_fdr:end_addr_fdr].numpy())).reshape((self._dim_feat_dynamic_real, -1))
            # feat_static_real = [0.0]
            
            # last_addr_start, last_addr_target, last_addr_fdr, last_addr_fsc = (
            #     end_addr_start, end_addr_target, end_addr_fdr, end_addr_fsc
            #     )
            
            # time = pd.to_datetime(ds.select("time").to_numpy().squeeze())
            # target = ds.select(cs.starts_with("target_")).to_numpy().T
            # feat_dynamic_real = ds.select(cs.starts_with("feat_dynamic_real_")).to_numpy().T
            # feat_static_cat = ds.select(pl.col("feat_static_cat")).head(1).to_numpy()[0,0]
            # feat_static_real = np.array([0])
            
            time = self.data_time[:, start_addr:end_addr]
            target = self.data_target[:, start_addr:end_addr]
            feat_dynamic_real = self.data_fdr[:, start_addr:end_addr]
            feat_static_cat = self.data_fsc[ds_idx, :]
            feat_static_real = torch.tensor([0])
            
            sampled_indices = self.sampler(target)[::self.skip_indices]
            
            # ds_idx = -1
            
            if len(sampled_indices) == 0:
                continue
            
            for idx in sampled_indices:
                # Extract data
                _, ts_length = target.shape
                
                # Find all valid time points
                min_time = self.context_length
                max_time = ts_length - self.prediction_length
                
                if max_time < min_time:
                    continue
                
                # Split into past and future windows
                past_target = target[:, idx - self.context_length:idx]
                future_target = target[:, idx:idx + self.prediction_length]
                
                # Create time features
                # time_index = pd.period_range(
                #     start=start_period,
                #     periods=ts_length,
                #     freq=start_period.freq
                # )
                
                # Apply time feature transformations
                # past_time_feat = self._create_time_features(
                #     time[idx - self.context_length:idx]
                # )
                # future_time_feat = self._create_time_features(
                #     time[idx:idx + self.prediction_length]
                # )
                past_time_feat = time[:, idx - self.context_length:idx]
                future_time_feat = time[:, idx:idx + self.prediction_length]
                
                # Create observed values indicator (1 for observed, 0 for missing)
                past_observed = ~torch.isnan(past_target)
                future_observed = ~torch.isnan(future_target)
                
                # Handle NaN values
                past_target = torch.nan_to_num(past_target, 0.0)
                future_target = torch.nan_to_num(future_target, 0.0)
                
                # Get static features
                # feat_static_cat = entry.get('feat_static_cat', [0])
                # feat_static_real = entry.get('feat_static_real', [0.0])
                
                # Get dynamic features if available
                # if 'feat_dynamic_real' in entry:
                    # feat_dynamic_real = entry['feat_dynamic_real']
                past_dynamic = feat_dynamic_real[:, idx - self.context_length:idx]
                future_dynamic = feat_dynamic_real[:, idx:idx + self.prediction_length]
                
                # Stack with time features
                past_time_feat = torch.vstack([past_time_feat, past_dynamic])
                future_time_feat = torch.vstack([future_time_feat, future_dynamic])
                
                # Convert to tensors and transpose to (time, features)
                # yield {
                #     'past_target': torch.from_numpy(past_target.T).float(),
                #     'future_target': torch.from_numpy(future_target.T).float(),
                #     'past_time_feat': torch.from_numpy(past_time_feat.T).float(),
                #     'future_time_feat': torch.from_numpy(future_time_feat.T).float(),
                #     'past_observed_values': torch.from_numpy(past_observed.T).float(),
                #     'future_observed_values': torch.from_numpy(future_observed.T).float(),
                #     'feat_static_cat': torch.tensor(feat_static_cat, dtype=torch.long),
                #     'feat_static_real': torch.tensor(feat_static_real, dtype=torch.float),
                # }
                yield (
                    past_target.T.float(),
                    future_target.T.float(),
                    past_time_feat.T.float(),
                    future_time_feat.T.float(),
                    past_observed.T.float(),
                    future_observed.T.float(),
                    feat_static_cat.long(),
                    feat_static_real.float(),
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
        
        # data_iterator = self.data
        # addr_iterator = zip(iter(self._addr_target), iter(self._addr_feat_dynamic_real), iter(self._addr_feat_static_cat))
        
        # addr = self._addr.numpy()
        # addr = {
        #     k: addr[i, :] for i, k in enumerate(self._data_keys)
        # }
        
        # self.samples = []
        # for entry in data_iterator:
        # for ds_idx in range(self.n_datasets):
            
        #     if ds_idx == 0:
        #         # start_addr_item_id = 
        #         start_addr_start = start_addr_target = start_addr_fdr = start_addr_fsc = 0
        #     else:
        #         # start_addr_item_id = addr["item_id"][ds_idx - 1]
        #         start_addr_start = addr["start"][ds_idx - 1]
        #         start_addr_target = addr["target"][ds_idx - 1]
        #         start_addr_fdr = addr["feat_dynamic_real"][ds_idx - 1]
        #         start_addr_fsc = addr["feat_static_cat"][ds_idx - 1]

        #     # end_addr_item_id = addr["item_id"][ds_idx]
        #     end_addr_start = addr["start"][ds_idx]
        #     end_addr_target = addr["target"][ds_idx]
        #     end_addr_fdr = addr["feat_dynamic_real"][ds_idx]
        #     end_addr_fsc = addr["feat_static_cat"][ds_idx]
            
        #     # item_id = pickle.loads(memoryview(self._data_item_id[start_addr_item_id:end_addr_item_id].numpy()))
        #     start_period = pickle.loads(memoryview(self._data_start[start_addr_start:end_addr_start].numpy()))
        #     target = pickle.loads(memoryview(self._data_target[start_addr_target:end_addr_target].numpy())).reshape((self._dim_target, -1))
        #     feat_static_cat = pickle.loads(memoryview(self._data_feat_static_cat[start_addr_fsc:end_addr_fsc].numpy()))
        #     feat_dynamic_real = pickle.loads(memoryview(self._data_feat_dynamic_real[start_addr_fdr:end_addr_fdr].numpy())).reshape((self._dim_feat_dynamic_real, -1))
        #     feat_static_real = [0.0]
            
        #     # Same processing as parent class but with fixed time point
        #     ts_length = target.shape[1]
        
        #     # Find all valid time points
        #     min_time = self.context_length
        #     max_time = ts_length - self.prediction_length
            
        #     # Fill time indices
        #     sample_indices = np.arange(min_time, max_time + 1, self.skip_indices)
            
        #     if len(sample_indices) == 0:
        #         continue
            
        #     for idx in sample_indices:
        #         # Split into past and future windows at fixed time point t
        #         past_target = target[:, idx - self.context_length:idx]
        #         future_target = target[:, idx:idx + self.prediction_length]
                
        #         # Rest of processing is identical to parent class
        #         # ... (same as parent __getitem__ but without random sampling)
                
        #         # Create time features
        #         _, ts_length = target.shape
        #         time_index = pd.period_range(
        #             start=start_period,
        #             periods=ts_length,
        #             freq=start_period.freq
        #         )
                
        #         past_time_feat = self._create_time_features(
        #             time_index[idx - self.context_length:idx]
        #         )
        #         future_time_feat = self._create_time_features(
        #             time_index[idx:idx + self.prediction_length]
        #         )
                
        #         # Create observed values indicator
        #         past_observed = ~np.isnan(past_target)
        #         future_observed = ~np.isnan(future_target)
                
        #         # Handle NaN values
        #         past_target = np.nan_to_num(past_target, 0.0)
        #         future_target = np.nan_to_num(future_target, 0.0)
                
        #         # Get static features
        #         # feat_static_cat = entry.get('feat_static_cat', [0])
        #         # feat_static_real = entry.get('feat_static_real', [0.0])
                
        #         # Get dynamic features if available
        #         # if 'feat_dynamic_real' in entry:
        #             # feat_dynamic_real = entry['feat_dynamic_real']
        #         past_dynamic = feat_dynamic_real[:, idx - self.context_length:idx]
        #         future_dynamic = feat_dynamic_real[:, idx:idx + self.prediction_length]
                
        #         # Stack with time features
        #         past_time_feat = np.vstack([past_time_feat, past_dynamic])
        #         future_time_feat = np.vstack([future_time_feat, future_dynamic])
                
        #         # Convert to tensors
        #         self.samples.append((
        #             torch.from_numpy(past_target.T).float(),
        #             torch.from_numpy(future_target.T).float(),
        #             torch.from_numpy(past_time_feat.T).float(),
        #             torch.from_numpy(future_time_feat.T).float(),
        #             torch.from_numpy(past_observed.T).float(),
        #             torch.from_numpy(future_observed.T).float(),
        #             torch.tensor(feat_static_cat, dtype=torch.long),
        #             torch.tensor(feat_static_real, dtype=torch.float)
        #         ))
                
        # del self.data  # Free memory after loading
        
    def _base_iter(self):
        """Get a specific window for inference."""
        # for s in self.samples:
            # yield s, np.array('past_target', 'future_target', 'past_time_feat', 
            #                  'future_time_feat', 'past_observed_values', 'future_observed_values',
            #                  'feat_static_cat', 'feat_static_real').astype(np.string_)
        
        # start_addr = 0
        for ds_idx, (start_addr, end_addr) in zip(self.ds_addr, pairwise(self.ds_addr)):
            
            # if ds_idx == 0:
            #     # start_addr_item_id = 
            #     start_addr_start = start_addr_target = start_addr_fdr = start_addr_fsc = 0
            # else:
            #     # start_addr_item_id = addr["item_id"][ds_idx - 1]
            #     start_addr_start = addr["start"][ds_idx - 1]
            #     start_addr_target = addr["target"][ds_idx - 1]
            #     start_addr_fdr = addr["feat_dynamic_real"][ds_idx - 1]
            #     start_addr_fsc = addr["feat_static_cat"][ds_idx - 1]

            # # end_addr_item_id = addr["item_id"][ds_idx]
            # end_addr_start = addr["start"][ds_idx]
            # end_addr_target = addr["target"][ds_idx]
            # end_addr_fdr = addr["feat_dynamic_real"][ds_idx]
            # end_addr_fsc = addr["feat_static_cat"][ds_idx]
            
            # item_id = pickle.loads(memoryview(self._data_item_id[start_addr_item_id:end_addr_item_id].numpy()))
            # start_period = pickle.loads(memoryview(self._data_start[start_addr_start:end_addr_start].numpy()))
            # target = pickle.loads(memoryview(self._data_target[start_addr_target:end_addr_target].numpy())).reshape((self._dim_target, -1))
            # feat_static_cat = pickle.loads(memoryview(self._data_feat_static_cat[start_addr_fsc:end_addr_fsc].numpy()))
            # feat_dynamic_real = pickle.loads(memoryview(self._data_feat_dynamic_real[start_addr_fdr:end_addr_fdr].numpy())).reshape((self._dim_feat_dynamic_real, -1))
            # feat_static_real = [0.0]
            # time = pd.to_datetime(ds.select("time").to_numpy().squeeze())
            # target = ds.select(cs.starts_with("target_")).to_numpy().T
            # feat_dynamic_real = ds.select(cs.starts_with("feat_dynamic_real_")).to_numpy().T
            # feat_static_cat = ds.select(pl.col("feat_static_cat")).head(1).to_numpy()[0,0]
            # feat_static_real = np.array([0])
            
            time = self.data_time[:, start_addr:end_addr]
            target = self.data_target[:, start_addr:end_addr]
            feat_dynamic_real = self.data_fdr[:, start_addr:end_addr]
            feat_static_cat = self.data_fsc[ds_idx, :]
            feat_static_real = torch.tensor([0])
            
            # start_addr = end_addr
            
            # Same processing as parent class but with fixed time point
            ts_length = target.shape[1]
        
            # Find all valid time points
            min_time = self.context_length
            max_time = ts_length - self.prediction_length
            
            # Fill time indices
            sample_indices = np.arange(min_time, max_time + 1, self.skip_indices)
            
            if len(sample_indices) == 0:
                continue
            
            for idx in sample_indices:
                # Split into past and future windows at fixed time point t
                past_target = target[:, idx - self.context_length:idx]
                future_target = target[:, idx:idx + self.prediction_length]
                
                # Rest of processing is identical to parent class
                # ... (same as parent __getitem__ but without random sampling)
                
                # Create time features
                # _, ts_length = target.shape
                # time_index = pd.period_range(
                #     start=start_period,
                #     periods=ts_length,
                #     freq=start_period.freq
                # )
                # time_index = time
                
                # past_time_feat = self._create_time_features(
                #     time[idx - self.context_length:idx]
                # )
                # future_time_feat = self._create_time_features(
                #     time[idx:idx + self.prediction_length]
                # )
                past_time_feat = time[:, idx - self.context_length:idx]
                future_time_feat = time[:, idx:idx + self.prediction_length]
                
                # Create observed values indicator
                past_observed = ~torch.isnan(past_target)
                future_observed = ~torch.isnan(future_target)
                
                # Handle NaN values
                past_target = torch.nan_to_num(past_target, 0.0)
                future_target = torch.nan_to_num(future_target, 0.0)
                
                # Get static features
                # feat_static_cat = entry.get('feat_static_cat', [0])
                # feat_static_real = entry.get('feat_static_real', [0.0])
                
                # Get dynamic features if available
                # if 'feat_dynamic_real' in entry:
                    # feat_dynamic_real = entry['feat_dynamic_real']
                past_dynamic = feat_dynamic_real[:, idx - self.context_length:idx]
                future_dynamic = feat_dynamic_real[:, idx:idx + self.prediction_length]
                
                # Stack with time features
                past_time_feat = torch.vstack([past_time_feat, past_dynamic])
                future_time_feat = torch.vstack([future_time_feat, future_dynamic])
                
                # Convert to tensors
                yield (
                    past_target.T.float(),
                    future_target.T.float(),
                    past_time_feat.T.float(),
                    future_time_feat.T.float(),
                    past_observed.T.float(),
                    future_observed.T.float(),
                    feat_static_cat.long(),
                    feat_static_real.float()
                )
                # np.array(["past_target", "future_target", "past_time_feat", 
                #              "future_time_feat", "past_observed_values", "future_observed_values",
                #              "feat_static_cat", "feat_static_real"]).astype(np.string_)