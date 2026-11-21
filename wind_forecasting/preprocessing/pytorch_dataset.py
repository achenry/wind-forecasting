"""
PyTorch Dataset for wind forecasting data that enables distributed training.

This module provides a PyTorch Dataset that loads pre-split data from pickle files
and applies necessary transformations for model training.
"""

import logging
import pickle
# from memory_profiler import profile
# from line_profiler import profile
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

def _persist_future_feat_dynamic_real(past_time_feat, future_time_feat, n_time_features):
    """Replace the future values of the dynamic real features (e.g. nacelle direction
    nd_cos/nd_sin) with their last observed value.

    The `time` tensor is [calendar time features | feat_dynamic_real]. Calendar features
    are known in advance, but the true future nacelle direction is not: the turbine's yaw
    controller tracks the future wind, so feeding the realised future yaw leaks the target.
    At inference time (whoc MLForecast._generate_test_data) the last value is persisted, so
    training/validation must do the same.
    """
    if future_time_feat.shape[1] <= n_time_features or past_time_feat.shape[0] == 0:
        return future_time_feat
    future_time_feat = future_time_feat.clone()
    future_time_feat[:, n_time_features:] = past_time_feat[-1:, n_time_features:]
    return future_time_feat


class _EpochAwareDataLoader(DataLoader):
    """DataLoader that advances the dataset's epoch counter each time a new epoch iterator is
    created, so a shuffled IterableDataset gets a new permutation every epoch even when the
    worker processes are not persistent (Lightning only calls set_epoch on samplers, which
    IterableDatasets do not have)."""
    _epoch = 0

    def __iter__(self):
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(self._epoch)
        self._epoch += 1
        return super().__iter__()


class WindForecastingDatamodule(L.LightningDataModule):
    def __init__(self, train_data_path, val_data_path, train_sampler,
                 context_length, prediction_length, time_features, val_sampler=None,
                 train_repeat=False, val_repeat=False,
                 batch_size=32, num_workers=4, pin_memory=True,
                 shuffle_train=True, shuffle_buffer_length=10_000, shuffle_seed=0,
                 persist_future_feat_dynamic_real=True):
        super().__init__()
        self.shuffle_train = shuffle_train
        self.shuffle_buffer_length = shuffle_buffer_length
        self.shuffle_seed = shuffle_seed
        self.persist_future_feat_dynamic_real = persist_future_feat_dynamic_real
        self.n_time_features = len(time_features) if time_features else 0
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
                rank=self.rank,
                shuffle=self.shuffle_train,
                shuffle_buffer_length=self.shuffle_buffer_length,
                shuffle_seed=self.shuffle_seed,
                n_time_features=self.n_time_features,
                persist_future_feat_dynamic_real=self.persist_future_feat_dynamic_real)
        
        logging.info(f"Returning DataLoader in WindForecastingDatamodule.train_dataloader() on rank {self.rank} for world_size {self.world_size}")
        
        return _EpochAwareDataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Will be overridden by DistributedSampler in DDP
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers>0,
            prefetch_factor=2 if (self.num_workers > 0) else None,
            drop_last=self.world_size > 1  # Required for DDP to avoid uneven batch sizes
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
                rank=self.rank,
                n_time_features=self.n_time_features,
                persist_future_feat_dynamic_real=self.persist_future_feat_dynamic_real)
        
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
            prefetch_factor=2 if (self.num_workers > 0) else None,
            drop_last=self.world_size > 1  # Required for DDP
        )

    @property
    def test_dataset(self):
        """Test dataset property - returns None as test data loading is not yet implemented.

        To enable test data loading, extend __init__ to accept test_data_path
        and setup() to process test data similar to train/val.
        """
        return getattr(self, '_test_dataset', None)

    def test_dataloader(self):
        # The same pattern applies for testing.
        if self.test_dataset is None:
            return None

        return DataLoader(
            self.test_dataset,  # Fixed: was self.train_dataset (copy-paste bug)
            batch_size=self.batch_size,
            shuffle=False,  # Never shuffle test data
            # worker_init_fn=self.__class__._worker_init_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers>0,
            prefetch_factor=2 if (self.num_workers > 0) else None
            # drop_last=False,  # Keep all test samples
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
        world_size: int = 1,
        shuffle: bool = False,
        shuffle_buffer_length: int = 10_000,
        shuffle_seed: int = 0,
        n_time_features: int = 0,
        persist_future_feat_dynamic_real: bool = True,
    ):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.sampler = sampler
        self.repeat = repeat
        self.shuffle = shuffle
        self.shuffle_buffer_length = shuffle_buffer_length
        self.shuffle_seed = shuffle_seed
        self.n_time_features = n_time_features
        self.persist_future_feat_dynamic_real = persist_future_feat_dynamic_real
        # incremented on every __iter__ call so each epoch gets a new permutation
        self._n_iters = 0
        
        self.skip_indices = skip_indices
        
        # Store world_size and rank passed from the DataModule
        self.world_size = world_size
        self.rank = rank
        
        self.data_time = data["time"]
        self.data_target = data["target"]
        # data[sorted([col for col in data.columns if col.startswith("target")], key=lambda col: int(col.split("_")[-1]))]
        # self.data_fdr = data[sorted([col for col in data.columns if col.startswith("feat_dynamic_real")], key=lambda col: int(col.split("_")[-1]))]
        self.data_fsc = data["feat_static_cat"]
        self.time_addr = data["time_addr"]
        self.ds_addr = data["ds_addr"]
        self.data_observed = data["observed"]
        self.data_feat_static_real = data["feat_static_real"]
        
        logging.info(f"Instantiating data attributes in WindForecastingDataset.__init__ with rank = {self.rank} and world_size = {self.world_size}")
    
    # @profile
    def __iter__(self):
        
        # if self.world_size > 1:
        #     logger.info(f"Using distributed training with rank={self.rank}, world_size={self.world_size}")
        # else:
        #     logger.info(f"Using single-rank training with rank={self.rank}, world_size={self.world_size}")
                
        worker_info = torch.utils.data.get_worker_info()

        # Every worker (and every DDP rank) must generate the SAME shuffled stream, because
        # each one keeps every n-th sample of it via islice. The RNG therefore depends only on
        # (shuffle_seed, epoch counter): the counter advances once per __iter__ call (i.e. per
        # epoch) in the main process and, with persistent workers, in each worker in lockstep.
        # Without persistent workers, the dataset is re-pickled into the workers each epoch
        # from the main-process copy, whose counter is advanced by set_epoch() below.
        self._n_iters += 1
        rng = np.random.default_rng([self.shuffle_seed, self._n_iters]) if self.shuffle else None

        if worker_info is None: # Main process, num_workers=0 case
            # logger.info(f"training worker_info is None, on main process fetching islice {self.rank}:None:{self.world_size}")
            yield from islice(self._base_iter(rng), self.rank, None, self.world_size)
        else: # In a worker process
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

            global_num_workers = num_workers * self.world_size
            global_worker_id = self.rank * num_workers + worker_id
            # logger.info(f"training worker {worker_info.id} of {num_workers}, fetching islice {global_worker_id}:None:{global_num_workers}")

            yield from islice(self._base_iter(rng), global_worker_id, None, global_num_workers)

    def set_epoch(self, epoch: int):
        """Set the epoch counter explicitly (used by the datamodule so non-persistent workers
        also get a fresh permutation each epoch)."""
        self._n_iters = epoch

    def _shuffled(self, it, rng):
        """Approximate global shuffle with a fixed-size shuffle buffer."""
        buffer = []
        for item in it:
            if len(buffer) < self.shuffle_buffer_length:
                buffer.append(item)
                continue
            j = rng.integers(len(buffer))
            yield buffer[j]
            buffer[j] = item
        rng.shuffle(buffer)
        yield from buffer

    def _make_sample(self, idx, target, time, observed, feat_static_cat):
        context_slice, pred_slice = slice(idx - self.context_length, idx), slice(idx, idx + self.prediction_length)
        past_time_feat, future_time_feat = time[context_slice, :], time[pred_slice, :]
        if self.persist_future_feat_dynamic_real:
            future_time_feat = _persist_future_feat_dynamic_real(
                past_time_feat, future_time_feat, self.n_time_features)
        return (
            target[context_slice, :],
            target[pred_slice, :],
            past_time_feat,
            future_time_feat,
            observed[context_slice, :],
            observed[pred_slice, :],
            feat_static_cat,
            self.data_feat_static_real,
        )
    
    def _iter_pass(self, rng=None):
        """One pass over all continuity groups (items). With an rng, both the group order and
        the window order within each group are permuted."""
        n_groups = len(self.time_addr) - 1
        group_order = rng.permutation(n_groups) if rng is not None else np.arange(n_groups)
        for ds_idx in group_order:
            start_addr, end_addr = self.time_addr[ds_idx], self.time_addr[ds_idx + 1]
            time = self.data_time[start_addr:end_addr, :]
            target = self.data_target[start_addr:end_addr, :]
            feat_static_cat = self.data_fsc[ds_idx, :]
            observed = self.data_observed[start_addr:end_addr, :]

            if target.shape[0] - self.prediction_length < self.context_length:
                continue

            # Get sampled indices from the sampler
            if self.sampler is not None:
                sampled_indices = self.sampler(target.T)
                if self.skip_indices > 1:
                    sampled_indices = sampled_indices[::self.skip_indices]
            else:
                # Default to all valid indices
                sampled_indices = np.arange(self.context_length, target.shape[0] - self.prediction_length + 1, self.skip_indices)

            if rng is not None:
                sampled_indices = rng.permutation(sampled_indices)

            for idx in sampled_indices:
                yield self._make_sample(int(idx), target, time, observed, feat_static_cat)

    # @profile
    def _base_iter(self, rng=None):
        """
        Get an iterable of samples with transformations applied.
        
        Returns
        -------
        Tuple of:
            - past_target: (context_length, num_series)
            - future_target: (prediction_length, num_series)
            - past_time_feat: (context_length, num_time_features)
            - future_time_feat: (prediction_length, num_time_features)
            - past_observed_values: (context_length, num_series)
            - future_observed_values: (prediction_length, num_series)
            - feat_static_cat: (num_static_cat,)
            - feat_static_real: (num_static_real,)

        NOTE: previously, repeat=True used zip(cycle(ds_addr), pairwise(cycle(time_addr))),
        which has N+1 address pairs per cycle vs N item ids, so feat_static_cat drifted out of
        alignment with the series after the first pass. Passes are now explicit.
        """
        def passes():
            n_pass = 0
            while True:
                yield from self._iter_pass(rng)
                n_pass += 1
                if not self.repeat:
                    return

        it = passes()
        if rng is not None and self.shuffle_buffer_length and self.shuffle_buffer_length > 1:
            it = self._shuffled(it, rng)
        yield from it


class WindForecastingInferenceDataset(WindForecastingDataset):
    """
    Dataset for inference that creates windows at all valid time points.
    
    Unlike the training dataset which samples random windows, this creates
    all possible windows for complete evaluation.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    # @profile
    def _base_iter(self, rng=None):
        """Get every window (stride skip_indices) for inference/validation, in order."""
        for ds_idx in range(len(self.time_addr) - 1):
            start_addr, end_addr = self.time_addr[ds_idx], self.time_addr[ds_idx + 1]
            time = self.data_time[start_addr:end_addr, :]
            target = self.data_target[start_addr:end_addr, :]
            observed = self.data_observed[start_addr:end_addr, :]
            feat_static_cat = self.data_fsc[ds_idx, :]

            sample_indices = np.arange(self.context_length, target.shape[0] - self.prediction_length + 1, self.skip_indices)
            for idx in sample_indices:
                yield self._make_sample(int(idx), target, time, observed, feat_static_cat)
