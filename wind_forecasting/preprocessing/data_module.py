from dataclasses import dataclass
from typing import List, Type, Optional
import os
import re
import logging
import glob
import torch
import glob
import torch.distributed as dist
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from torch.utils.data import DataLoader
# from gluonts.dataset.split import split, slice_data_entry
from gluonts.dataset.pandas import PolarsDataset, PandasDataset, IterableLazyFrame
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
# from gluonts.dataset.common import TrainDatasets, MetaData, BasicFeatureInfo, CategoricalFeatureInfo, ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.util import to_pandas
import pickle
# from gluonts.dataset.common import FileDataset
# from gluonts.dataset import Dataset

import polars as pl
import polars.selectors as cs
import pandas as pd
# import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colormaps
import seaborn as sns

from memory_profiler import profile

@dataclass
class DataModule():
    """_summary_
    # DataModule should use a polars LazyFrame and sink it into a parquet,
    # and store the indices in the full dataset to use for each cg, split_idx, and training/test/validation split
    """
    data_path: str
    n_splits: int
    continuity_groups: List[int] | None
    train_split: float
    val_split: float
    test_split: float
    prediction_length: int
    context_length: int
    target_prefixes: List[str]
    target_suffixes: List[str] | None # ie turbine ids after the last underscore eg wt001 in  ws_horz_wt001
    feat_dynamic_real_prefixes: List[str]
    freq: str
    per_turbine_target: bool # if True, feed multiple datasets to trainer, where each one corresponds to the outputs of a single turbine
    dtype: Type = pl.Float32
    as_lazyframe: bool = False
    verbose: bool = False
    normalized: bool = True
    normalization_consts_path: Optional[str] = None
    batch_size: int = 128
    workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    
    def __post_init__(self):
            
        # convert context and prediction length from seconds to time stesp based on freq
        self.context_length = int(pd.Timedelta(self.context_length, unit="s") / pd.Timedelta(self.freq))
        self.prediction_length = int(pd.Timedelta(self.prediction_length, unit="s") / pd.Timedelta(self.freq))
        assert self.context_length > 0, "context_length must be provided in seconds, and must be greaterthan resample_freq."
        assert self.prediction_length > 0, "prediction_length must be provided in seconds, and must be greaterthan resample_freq."
        
        self.set_train_ready_path()
    
    def set_train_ready_path(self):
        sfx = f"ctx{self.context_length}_pred{self.prediction_length}"
        if self.normalized:
            self.train_ready_data_path = self.data_path.replace(
                ".parquet", f"_train_ready_{self.freq}_{'per_turbine' if self.per_turbine_target else 'all_turbine'}_{sfx}.parquet")
        else:
            self.train_ready_data_path = self.data_path.replace(
                ".parquet", f"_train_ready_{self.freq}_{'per_turbine' if self.per_turbine_target else 'all_turbine'}_{sfx}_denormalize.parquet")
     
    def get_split_file_path(self, split):
        """Generate split file path that includes context_length and prediction_length to ensure cache uniqueness."""
        # Extract base name without .parquet extension
        if self.train_ready_data_path.endswith("_denormalize.parquet"):
            base_path = self.train_ready_data_path.replace("_denormalize.parquet", "")
            suffix = "_denormalize"
        else:
            base_path = self.train_ready_data_path.replace(".parquet", "")
            suffix = ""
        
        # Include context_length in the filename to make cache files distinct
        return f"{base_path}_{split}{suffix}.pkl"
    
    def _validate_loaded_splits(self, splits, rank):
        """Validate that loaded splits are compatible with current context_length requirements."""
        min_required_length = self.context_length + self.prediction_length
        
        for split in splits:
            dataset = getattr(self, f"{split}_dataset")
            if not dataset:
                logging.warning(f"Rank {rank}: {split} dataset is empty! This may cause validation issues.")
                continue
                
            # Check if dataset has enough samples for the current context_length requirements
            if isinstance(dataset, list) and len(dataset) > 0:
                # Check a representative sample from the dataset
                sample = dataset[0] if dataset else None
                if sample and "target" in sample:
                    target_length = sample["target"].shape[1] if hasattr(sample["target"], "shape") else len(sample["target"][0])
                    if target_length < min_required_length:
                        logging.error(f"Rank {rank}: {split} dataset samples are too short ({target_length}) for context_length={self.context_length} + prediction_length={self.prediction_length} = {min_required_length}")
                        raise ValueError(f"Loaded {split} dataset is incompatible with current context_length={self.context_length}")
                        
        logging.info(f"Rank {rank}: Validation passed for loaded splits with context_length={self.context_length}, prediction_length={self.prediction_length}")
    
    def compute_scaler_params(self):
        norm_consts = pd.read_csv(self.normalization_consts_path, index_col=None)
        norm_min_cols = [col for col in norm_consts if "_min" in col]
        norm_max_cols = [col for col in norm_consts if "_max" in col]
        data_min = norm_consts[norm_min_cols].values.flatten()
        data_max = norm_consts[norm_max_cols].values.flatten()
        norm_min_cols = [col.replace("_min", "") for col in norm_min_cols]
        norm_max_cols = [col.replace("_max", "") for col in norm_max_cols]
        feature_range = (-1, 1)
        self.norm_scale = ((feature_range[1] - feature_range[0]) / (data_max - data_min))
        self.norm_min = feature_range[0] - (data_min * self.norm_scale)
        return {"min_": dict(zip(norm_min_cols, self.norm_min)), "scale_": dict(zip(norm_min_cols, self.norm_scale))}
     
    def generate_datasets(self):
        
        dataset = IterableLazyFrame(data_path=self.data_path, dtype=self.dtype)
        
        # add warning if upsampling
        dataset_dt = dataset.select(pl.col("time").diff()).slice(1, 1).collect().item()
        if dataset_dt.total_seconds() > int(re.search("\\d+", self.freq).group()):
            logging.warning(f"Downsampling dataset with frequency of {dataset_dt} seconds to {self.freq}.")
        dataset = dataset.with_columns(time=pl.col("time").dt.round(self.freq))\
                    .group_by("time").agg(cs.numeric().mean())\
                    .sort(["continuity_group", "time"])
                    
        if not self.normalized:
            scaler_params = self.compute_scaler_params()
            feat_types = list(scaler_params["min_"])
            dataset = dataset.with_columns([(cs.starts_with(feat_type) - scaler_params["min_"][feat_type]) 
                                                        / scaler_params["scale_"][feat_type] 
                                                        for feat_type in feat_types])
                    
        # TODO if resampling requieres upsampling: historic_measurements.upsample(time_column="time", every=self.data_module.freq).fill_null(strategy="forward")
        # dataset = IterableLazyFrame(data_path=self.train_ready_data_path, dtype=self.dtype) # data stored in RAM
        # gc.collect()
        
        # fetch a subset of continuity groups and turbine data
        if self.verbose:
            logging.info("Getting continuity groups.")
        if self.continuity_groups is None:
            if "continuity_group" in dataset.collect_schema().names():
                self.continuity_groups = dataset.select(pl.col("continuity_group").unique()).collect().to_numpy().flatten()
                if self.target_suffixes is not None:
                    dataset = dataset.select(pl.col("time"), pl.col("continuity_group"), *[cs.ends_with(sfx) for sfx in self.target_suffixes])
                
            else:
                self.continuity_groups = [0]
                # dataset = dataset[[col for col in dataset.columns if any(col.__contains__(tid) for tid in self.target_suffixes)]]
                # dataset.loc[:, "continuity_group"] = 0
                if self.target_suffixes is not None:
                    dataset = dataset.select(pl.col("time"), *[cs.ends_with(sfx) for sfx in self.target_suffixes])
                dataset = dataset.with_columns(continuity_group=pl.lit(0))
        else:
            dataset = dataset.filter(pl.col("continuity_group").is_in(self.continuity_groups))\
                             .select(pl.col("time"), pl.col("continuity_group"), *[cs.ends_with(f"_{sfx}") for sfx in self.target_suffixes])
        if self.verbose:
            logging.info(f"Found continuity groups {self.continuity_groups}") 
        # dataset.target_cols = self.target_cols 
        
        if self.verbose:
            logging.info(f"Writing resampled/sorted parquet to {self.train_ready_data_path}.") 
        dataset.collect().write_parquet(self.train_ready_data_path, statistics=False)
        if self.verbose:
            logging.info(f"Saved resampled/sorted parquet to {self.train_ready_data_path}.")
        
        self.get_dataset_info(dataset)
        # dataset = IterableLazyFrame(data_path=train_ready_data_path)
        # univariate=ListDataset of multiple dictionaires each corresponding to measurements from a single turbine, to implicitly capture correlations
        # or multivariate=multivariate dictionary for all measurements, to explicity capture all correlations
        # or debug= to use electricity dataset
    
    # @profile
    def get_dataset_info(self, dataset=None):
        # print(f"Number of nan/null vars = {dataset.select(pl.sum_horizontal((cs.numeric().is_null() | cs.numeric().is_nan()).sum())).collect().item()}") 
        if dataset is None:
            assert os.path.exists(self.train_ready_data_path), "train_ready_data_path doesn't exist! Should be generated for training."
            dataset = IterableLazyFrame(data_path=self.train_ready_data_path, dtype=self.dtype) 
        
        if self.verbose:
            logging.info("Getting continuity groups.") 
            
        if self.continuity_groups is None:
            if "continuity_group" in dataset.collect_schema().names():
                # TODO this is giving floats??
                self.continuity_groups = dataset.select(pl.col("continuity_group").unique()).collect().to_numpy().flatten()
            else:
                self.continuity_groups = [0]
                
        if self.verbose:
            logging.info(f"Found continuity groups {self.continuity_groups}")
            
            logging.info(f"Getting column names.") 
        if self.target_suffixes is None:
            self.target_cols = dataset.select(*[cs.starts_with(pfx) for pfx in self.target_prefixes]).collect_schema().names()
            self.target_suffixes = sorted(list(set(col.split("_")[-1] for col in self.target_cols)), key=lambda col: int(re.search("\\d+", col).group()))
        else:
            self.target_cols = [col for col in dataset.collect_schema().names() if any(prefix in col for prefix in self.target_prefixes)]
        self.feat_dynamic_real_cols = [col for col in dataset.collect_schema().names() if any(prefix in col for prefix in self.feat_dynamic_real_prefixes)]
        
        if self.verbose:
            logging.info(f"Found column names target_cols={self.target_cols}, feat_dynamic_real_cols={self.feat_dynamic_real_cols}.") 
        
        if self.per_turbine_target:
            self.num_target_vars = len(self.target_prefixes)
            self.num_feat_dynamic_real = int(len(self.feat_dynamic_real_cols) / len(self.target_suffixes))
            self.num_feat_static_cat = 1
            self.num_feat_static_real = 0
            # self.target_cols = self.target_prefixes
            
            # self.cardinality = list(self.static_features.select_dtypes("category").nunique().values)
            self.static_features = None # allow to generate itself
            self.cardinality = (len(self.target_suffixes),)
            
        else:
            self.num_feat_dynamic_real = len(self.feat_dynamic_real_cols)
            self.num_target_vars = len(self.target_cols)
            self.num_feat_static_cat = 0
            self.num_feat_static_real = 0
            
            self.static_features = pd.DataFrame()
            self.cardinality = None 
    
    # @profile # prints memory usage
    def generate_splits(self, splits=None, save=False, reload=True, verbose=None, rank=None):
        if verbose is None:
            verbose = self.verbose
            
        if splits is None:
            splits = ["train", "val", "test"]
        assert all(split in ["train", "val", "test"] for split in splits)
        assert os.path.exists(self.train_ready_data_path), f"Must run generate_datasets before generate_splits to produce {self.train_ready_data_path}."

        # If rank is explicitly provided (e.g., from WORKER_RANK in tuning mode), use it
        # Otherwise, check if PyTorch distributed is initialized (e.g., during DDP training)
        if rank is None:
            rank = 0
            world_size = 1
            is_distributed = dist.is_available() and dist.is_initialized()
            if is_distributed:
                # This path is only taken when PyTorch Lightning has initialized DDP
                rank = dist.get_rank()
                world_size = dist.get_world_size()
                logging.info(f"Rank {rank}/{world_size}: Detected PyTorch distributed mode.")
            else:
                # Check for WORKER_RANK environment variable as fallback
                # This handles tuning mode where workers are independent
                worker_rank = os.environ.get('WORKER_RANK', None)
                logging.info(f"WORKER_RANK environment variable: {worker_rank}")
                logging.info(f"Process ID: {os.getpid()}")
                
                if worker_rank is not None:
                    try:
                        rank = int(worker_rank)
                        logging.info(f"Rank {rank}: Using WORKER_RANK={worker_rank} from environment (independent worker mode).")
                    except ValueError:
                        logging.warning(f"Invalid WORKER_RANK value: '{worker_rank}', defaulting to rank 0")
                        rank = 0
                else:
                    rank = 0
                    logging.info("Rank 0: No WORKER_RANK found, no distributed mode detected, assuming single process.")
        else:
            # Rank was explicitly provided
            logging.info(f"Rank {rank}: Using explicitly provided rank value.")

        logging.info(f"Rank {rank}: Scanning dataset {self.train_ready_data_path}.")
        dataset = IterableLazyFrame(data_path=self.train_ready_data_path, dtype=self.dtype)
        logging.info(f"Rank {rank}: Finished scanning dataset {self.train_ready_data_path}.")

        self.get_dataset_info(dataset)
        split_files_exist = all(os.path.exists(self.get_split_file_path(split)) for split in splits)
        logging.info(f"Rank {rank}: Split file check - context_length={self.context_length}, prediction_length={self.prediction_length}, split_files_exist={split_files_exist}")
        for split in splits:
            split_path = self.get_split_file_path(split)
            exists = os.path.exists(split_path)
            logging.info(f"Rank {rank}: Split file {split}: {split_path} (exists: {exists})")

        if rank == 0 and (reload or not split_files_exist):
            logging.info(f"Rank 0: Generating splits (reload={reload}, files_exist={split_files_exist}).")
            if self.per_turbine_target:
                if self.verbose:
                    logging.info(f"Rank 0: Splitting datasets for per turbine case into train/val/test={self.train_split}/{self.val_split}/{self.test_split}.")
                
                cg_counts = dataset.select("continuity_group").collect().to_series().value_counts().sort("continuity_group").select("count").to_numpy().flatten()
                
                self.rows_per_split = [
                    int(n_rows / self.n_splits) 
                    for turbine_id in self.target_suffixes 
                    for n_rows in cg_counts] # each element corresponds to each continuity group
                
                del cg_counts
                
                self.continuity_groups = dataset.select(pl.col("continuity_group").unique()).collect().to_numpy().flatten()
                
                self.train_dataset, self.val_dataset, self.test_dataset = \
                    self.split_dataset([dataset.filter(pl.col("continuity_group") == cg) for cg in self.continuity_groups]) 
                    
                for d, ds in enumerate(self.train_dataset):
                    fp = self.train_ready_data_path.replace(".parquet", f"_train_{d}_tmp.parquet")
                    self.train_dataset[d].collect().write_parquet(fp)
                    self.train_dataset[d] = pl.scan_parquet(fp)
                    
                for d, ds in enumerate(self.val_dataset):
                    fp = self.train_ready_data_path.replace(".parquet", f"_val_{d}_tmp.parquet")
                    self.val_dataset[d].collect().write_parquet(fp)
                    self.val_dataset[d] = pl.scan_parquet(fp)
                    
                for d, ds in enumerate(self.test_dataset):
                    fp = self.train_ready_data_path.replace(".parquet", f"_test_{d}_tmp.parquet")
                    self.test_dataset[d].collect().write_parquet(fp)
                    self.test_dataset[d] = pl.scan_parquet(fp)
                    
                for split in splits:
                    ds = getattr(self, f"{split}_dataset")
                    if self.verbose:
                        logging.info(f"Setting {split}_dataset attribute.")
                    
                    temp_ds = {}
                    for cg in range(len(ds)):
                        # df = ds[cg].collect(_eager=True)
                        for turbine_id in self.target_suffixes:
                            if self.verbose:
                                logging.info(f"Getting dataset for turbine_id={turbine_id}, cg={cg} of {len(ds)}.")
                            temp_ds[f"TURBINE{turbine_id}_SPLIT{cg}"] = self.get_df_by_turbine(ds[cg], turbine_id)#.lazy()
                    
                    setattr(self, f"{split}_dataset", temp_ds)
                
                if self.as_lazyframe:
                    static_index = [f"TURBINE{turbine_id}_SPLIT{split}" for turbine_id in self.target_suffixes for cg in range(len(self.train_dataset))]
                    self.static_features = pd.DataFrame(
                        {
                            "turbine_id": pd.Categorical(turbine_id for turbine_id in self.target_suffixes for cg in range(len(self.train_dataset)))
                        },
                        index=static_index
                    )
                    
                    for split in splits:
                        ds = getattr(self, f"{split}_dataset")
                        setattr(self, f"{split}_dataset", 
                                PolarsDataset(ds, 
                                        target=self.target_prefixes, timestamp="time", freq=self.freq, 
                                        feat_dynamic_real=self.feat_dynamic_real_prefixes, static_features=self.static_features, 
                                        assume_sorted=True, assume_resampled=True, unchecked=True))
                else:
                    
                    # convert dictionary of item_id: lazyframe datasets into list of dictionaries with numpy arrays for data
                    for split in splits:
                        datasets = []
                        item_ids = list(getattr(self, f"{split}_dataset").keys())
                        for item_id in item_ids:
                            if verbose:
                                logging.info(f"Transforming {split} dataset {item_id} into numpy form.")
                            ds = getattr(self, f"{split}_dataset")[item_id]
                            start_time = pd.Period(ds.select(pl.col("time").first()).collect().item(), freq=self.freq)
                            ds = ds.select(self.feat_dynamic_real_prefixes + self.target_prefixes).collect().to_numpy().T
                            datasets.append({
                                "target": ds[-len(self.target_prefixes):, :],
                                 "item_id": item_id,
                                 "start": start_time,
                                 "feat_static_cat": [self.target_suffixes.index(re.search("(?<=TURBINE)\\w+(?=_SPLIT)", item_id).group(0))],
                                 "feat_dynamic_real": ds[:-len(self.target_prefixes), :]
                            })
                            del getattr(self, f"{split}_dataset")[item_id]
                        setattr(self, f"{split}_dataset", datasets)

                if verbose:
                    logging.info(f"Finished splitting datasets for per turbine case.") 

            else:
                if verbose:
                    logging.info(f"Splitting datasets for all turbine case into train/val/test={self.train_split}/{self.val_split}/{self.test_split}.") 
                
                cg_counts = dataset.select("continuity_group").collect().to_series().value_counts().sort("continuity_group").select("count").to_numpy().flatten()
                self.rows_per_split = [int(n_rows / self.n_splits) for n_rows in cg_counts] # each element corresponds to each continuity group
                del cg_counts
                self.continuity_groups = dataset.select(pl.col("continuity_group").unique()).collect().to_numpy().flatten()
                # generate an iterablelazy frame for each continuity group and split within it
                self.train_dataset, self.val_dataset, self.test_dataset = \
                    self.split_dataset([dataset.filter(pl.col("continuity_group") == cg) for cg in self.continuity_groups])
                
                for d, ds in enumerate(self.train_dataset):
                    fp = self.train_ready_data_path.replace(".parquet", f"_train_{d}_tmp.parquet")
                    self.train_dataset[d].collect().write_parquet(fp)
                    self.train_dataset[d] = pl.scan_parquet(fp)
                    
                for d, ds in enumerate(self.val_dataset):
                    fp = self.train_ready_data_path.replace(".parquet", f"_val_{d}_tmp.parquet")
                    self.val_dataset[d].collect().write_parquet(fp)
                    self.val_dataset[d] = pl.scan_parquet(fp)
                    
                for d, ds in enumerate(self.test_dataset):
                    fp = self.train_ready_data_path.replace(".parquet", f"_test_{d}_tmp.parquet")
                    self.test_dataset[d].collect().write_parquet(fp)
                    self.test_dataset[d] = pl.scan_parquet(fp)
                    
                # train_grouper = MultivariateGrouper(
                #     max_target_dim=self.num_target_vars,
                #     split_on="continuity_group" if len(self.continuity_groups) > 1 else None
                # )
                # transform list into dictionary if item_id, reduced dataset pairs
                for split in splits:
                    ds = getattr(self, f"{split}_dataset")
                    setattr(self, f"{split}_dataset", 
                            {f"SPLIT{split}": 
                            ds[split].select([pl.col("time")] + self.feat_dynamic_real_cols + self.target_cols) 
                            for split in range(len(ds))})

                if self.as_lazyframe:
                    for split in splits:
                        setattr(self, f"{split}_dataset", 
                                PolarsDataset(
                                    getattr(self, f"{split}_dataset"), 
                                    timestamp="time", freq=self.freq, 
                                    target=self.target_cols, feat_dynamic_real=self.feat_dynamic_real_cols, static_features=self.static_features, 
                                    assume_sorted=True, assume_resampled=True, unchecked=True
                            ))
                        
                else:
                    
                    # convert dictionary of item_id: lazyframe datasets into list of dictionaries with numpy arrays for data
                    for split in splits:
                        datasets = []
                        item_ids = list(getattr(self, f"{split}_dataset").keys())
                        for item_id in item_ids:
                            if self.verbose:
                                logging.info(f"Transforming {split} dataset {item_id} into numpy form.")
                            ds = getattr(self, f"{split}_dataset")[item_id]
                            start_time = pd.Period(ds.select(pl.col("time").first()).collect().item(), freq=self.freq)
                            ds = ds.select(self.feat_dynamic_real_cols + self.target_cols).collect().to_numpy().T
                            datasets.append({
                                "target": ds[-len(self.target_cols):, :],
                                 "item_id": item_id,
                                 "start": start_time,
                                 "feat_dynamic_real": ds[:-len(self.target_cols), :]
                            })
                            del getattr(self, f"{split}_dataset")[item_id]
                        setattr(self, f"{split}_dataset", datasets)
                if verbose:
                    logging.info(f"Finished splitting datasets for all turbine case.")
            
            # Log generated dataset sizes
            for split in splits:
                dataset = getattr(self, f"{split}_dataset")
                dataset_size = len(dataset) if dataset else 0
                logging.info(f"Rank 0: Generated {split} dataset size: {dataset_size} samples (context_length={self.context_length}, prediction_length={self.prediction_length})")
            
            if save:
                logging.info(f"Rank 0: Saving generated splits.")
                for split in splits:
                    if self.as_lazyframe:
                        raise NotImplementedError("Saving LazyFrame splits not implemented.")
                    else:
                        final_path = self.get_split_file_path(split)
                        # Use process ID in temp filename to avoid collisions between workers
                        temp_path = final_path + f".tmp.{os.getpid()}"
                        logging.info(f"Rank 0: Saving {split} data to {temp_path}")
                        try:
                            # Write to temp file
                            with open(temp_path, 'wb') as fp:
                                pickle.dump(getattr(self, f"{split}_dataset"), fp)
                            
                            # Atomic rename - if file exists, this will overwrite it atomically
                            logging.info(f"Rank 0: Atomically moving {temp_path} to {final_path}")
                            os.replace(temp_path, final_path)  # os.replace is atomic on POSIX
                            logging.info(f"Rank 0: Successfully saved {split} data to {final_path}")
                        except Exception as e:
                            logging.error(f"Rank 0: Error saving {split} data: {e}")
                            # Clean up temp file if it still exists
                            if os.path.exists(temp_path):
                                try:
                                    os.remove(temp_path)
                                    logging.info(f"Rank 0: Cleaned up temp file {temp_path}")
                                except OSError as cleanup_error:
                                    logging.error(f"Rank 0: Error removing temp file {temp_path}: {cleanup_error}")
                                    
            for temp_path in glob.glob(self.train_ready_data_path.replace(".parquet", f"_{split}_*_tmp.parquet")):
                os.remove(temp_path)
                    

        # Only use barrier if PyTorch distributed is actually initialized
        # In tuning mode with independent workers, we don't need/want a barrier
        is_distributed = dist.is_available() and dist.is_initialized()
        if is_distributed:
            logging.info(f"Rank {rank}: Waiting at distributed barrier before loading splits.")
            dist.barrier()
            logging.info(f"Rank {rank}: Passed distributed barrier.")
        elif rank != 0:
            # For independent workers (e.g., tuning mode), use file-based synchronization
            # Wait for rank 0 to finish by checking if all split files exist
            import time
            max_wait_time = 300  # 5 minutes timeout
            check_interval = 2   # Check every 2 seconds
            start_time = time.time()
            
            while not all(os.path.exists(self.get_split_file_path(split)) for split in splits):
                if time.time() - start_time > max_wait_time:
                    raise TimeoutError(f"Rank {rank}: Timeout waiting for split files from rank 0")
                logging.info(f"Rank {rank}: Waiting for rank 0 to generate split files...")
                time.sleep(check_interval)
            
            logging.info(f"Rank {rank}: All split files detected, proceeding to load.")
        
        if rank != 0 or (not reload and split_files_exist):
            logging.info(f"Rank {rank}: Loading saved split datasets.")
            for split in splits:
                split_path = self.get_split_file_path(split)
                if not os.path.exists(split_path):
                     # This should ideally not happen after the barrier if rank 0 succeeded
                     logging.error(f"Rank {rank}: ERROR - Split file {split_path} not found after barrier!")
                     raise FileNotFoundError(f"Rank {rank}: Split file {split_path} not found after barrier!")
                try:
                    with open(split_path, 'rb') as fp:
                        data = pickle.load(fp)
                        setattr(self, f"{split}_dataset", data)
                    logging.info(f"Rank {rank}: Successfully loaded {split} dataset from {split_path}.")
                except EOFError as e:
                     logging.error(f"Rank {rank}: EOFError loading {split_path}. File might be corrupted. Error: {e}")
                     raise
                except Exception as e:
                     logging.error(f"Rank {rank}: Error loading {split_path}: {e}")
                     raise

            # Log dataset sizes and validate loaded splits for compatibility with current context_length
            for split in splits:
                dataset = getattr(self, f"{split}_dataset")
                dataset_size = len(dataset) if dataset else 0
                logging.info(f"Rank {rank}: Loaded {split} dataset size: {dataset_size} samples")
            
            self._validate_loaded_splits(splits, rank)

        # for split, datasets in [("train", self.train_dataset), ("val", self.val_dataset), ("test", self.test_dataset)]:
        #     for ds in iter(datasets):
        #         for key in ["target", "feat_dynamic_real"]:
        #             print(f"{split} {key} {ds['item_id']} dataset - num nan/nulls = {ds[key].select(pl.sum_horizontal((cs.numeric().is_null() | cs.numeric().is_nan()).sum())).collect().item()}")
          
        # return dataset
        
    def get_df_by_turbine(self, dataset, turbine_id):
        return dataset.select(pl.col("time"), *[col for col in (self.feat_dynamic_real_cols + self.target_cols) if turbine_id in col])\
                        .rename(mapping={**{f"{tgt_col}_{turbine_id}": tgt_col for tgt_col in self.target_prefixes},
                                        **{f"{feat_col}_{turbine_id}": feat_col for feat_col in self.feat_dynamic_real_prefixes}})

    def split_datasets_by_turbine(self, datasets):
         
        return [
                ds.select([pl.col("time")] 
                                       + [pl.col(f"{tgt_col}_{turbine_id}") for tgt_col in self.target_prefixes]
                                       + [pl.col(f"{feat_col}_{turbine_id}") for feat_col in self.feat_dynamic_real_prefixes]
                ) for ds in datasets for turbine_id in self.target_suffixes
        ]

    # def denormalize(self, split):
    #     assert split in ["train", "test", "val"]
    #     ds = getattr(self, f"{split}_dataset")
        
    def split_dataset(self, dataset):
        train_datasets = []
        test_datasets = []
        val_datasets = []
        
        # TODO total past length may also have to supercede context_len + max(self.lags_seq)
        for cg, ds in enumerate(dataset):
            # TODO in this case should just add to training data anyway? 
            if round(min(self.train_split, self.val_split, self.test_split) * self.rows_per_split[cg] * self.n_splits) < self.context_length + self.prediction_length:
                logging.info(f"Can't split dataset corresponding to continuity group {cg} into training, validation, testing, the full dataset only has data points {round(self.rows_per_split[cg] * self.n_splits)}")
                
                if self.train_split * self.rows_per_split[cg] * self.n_splits >= self.context_length + self.prediction_length:
                    logging.info(f"Adding dataset corresponding to continuity group {cg} to training data, since it can't be split")
                    train_datasets += [ds] 
                
                continue
            
            # splitting each continuity group into subsections, a training/val/test dataset will then be generated from each subsection. 
            # We do this to get a coherent mix of training, val, test data that is more independent of trends over time
            datasets = [] 
            for split_idx in range(self.n_splits):
                slc = slice(split_idx * self.rows_per_split[cg], (split_idx + 1) * self.rows_per_split[cg])
                # check that each split is at least context_len + target_len long, otherwise don't split it
                # if slc.stop - slc.start >= self.context_length + self.prediction_length:
                
                if round(min(self.train_split, self.val_split, self.test_split) * (slc.stop - slc.start)) >= self.context_length + self.prediction_length: 
                    # split_dataset.append(slice_data_entry(ds, slice_=slc))
                    # logging.info(f"full dataset cg {cg} split {split_idx} 
                    #                 start time = {split_dataset[-1]['start']}, 
                    #                 end time = {split_dataset[-1]['start'] + split_dataset[-1]['target'].shape[1]}, 
                    #                 duration = {split_dataset[-1]['target'].shape[1] * pd.Timedelta(split_dataset[-1]['start'].freq)}")

                    datasets.append(ds.select(pl.exclude("continuity_group")).slice(slc.start, slc.stop - slc.start))
                    start_time = datasets[-1].select(pl.col("time").first()).collect().item()
                    end_time = datasets[-1].select(pl.col("time").last()).collect().item()
                    duration = end_time - start_time
                    logging.info(f"full dataset cg {cg} split {split_idx}, start time = {start_time}, end time = {end_time}, duration = {duration}")
                else:
                    logging.info(f"Can't split dataset {cg} into {self.n_splits} , not enough data points, returning whole.")
                    # split_dataset = [ds]
                    self.rows_per_split[cg] *= self.n_splits
                    datasets.append(ds.select(pl.exclude("continuity_group")))
                    break
            
            train_offset = round(self.train_split * self.rows_per_split[cg])
            val_offset = round(self.val_split * self.rows_per_split[cg])
            test_offset = round(self.test_split * self.rows_per_split[cg])

            # TODO shouldn't test data include history, and just the labels be unseen by training data?
            # train_datasets.append(ds.slice(0, train_offset))
            # val_datasets.append(ds.slice(train_offset, val_offset))
            # test_datasets.append(ds.slice(train_offset + val_offset, test_offset))
            train_datasets += [d.slice(0, train_offset) for d in datasets]
            val_datasets += [d.slice(train_offset, val_offset) for d in datasets]  # val_offset is the length of validation data
            test_datasets += [d.slice(train_offset + val_offset, test_offset) for d in datasets]  # test_offset is the length of test data
            
            # TODO doesn't work with iterable lazy frame
            # if self.verbose:
            #     for t, train_entry in enumerate(iter(train_datasets)):
            #         logging.info(f"training dataset cg {cg}, split {t} start time = {train_entry['start']}, end time = {train_entry['start'] + train_entry['target'].shape[1]}, duration = {train_entry['target'].shape[1] * pd.Timedelta(train_entry['start'].freq)}\n")

            #     for v, val_entry in enumerate(iter(val_datasets)):
            #         logging.info(f"validation dataset cg {cg}, split {v} start time = {val_entry['start']}, end time = {val_entry['start'] + val_entry['target'].shape[1]}, duration = {val_entry['target'].shape[1] * pd.Timedelta(val_entry['start'].freq)}\n")

            #     for t, test_entry in enumerate(iter(test_datasets)):
            #         logging.info(f"test dataset cg {cg}, split {t} start time = {test_entry['start']}, end time = {test_entry['start'] + test_entry['target'].shape[1]}, duration = {test_entry['target'].shape[1] * pd.Timedelta(test_entry['start'].freq)}\n")
            
            # n_test_windows = int((self.test_split * self.rows_per_split[cg]) / self.prediction_length)
            # test_dataset = test_gen.generate_instances(prediction_length=self.prediction_length, windows=n_test_windows)
        
        if self.verbose:
            logging.info("Returning train/val/test datasets.")
        return train_datasets, val_datasets, test_datasets

    def highlight_entry(self, entry, color, ax, vlines=None):
        start = entry["start"].to_timestamp()
        # end = entry["start"] + (entry["target"].shape[1] * entry["start"].freq.delta)
        end = (entry["start"] + entry["target"].shape[1]).to_timestamp()
        # print(f"start time = {start}, end time = {end}")
        if vlines is not None:
            ax.axvline(x=start, ymin=vlines[0], ymax=vlines[1], color=color)
            ax.axvline(x=end, ymin=vlines[0], ymax=vlines[1], color=color, linestyle="--")
        else:
            ax.axvspan(start, end, facecolor=color, alpha=0.2)
        
    def plot_dataset_splitting(self):
        colors = colormaps["Pastel1"].colors
        fig, axs = plt.subplots(self.num_target_vars, 1, sharex=True)
        for d, ds in enumerate([self.train_dataset, self.val_dataset, self.test_dataset]): 
            for entry in ds:
                # df = to_pandas(entry, is_multivariate=True).reset_index(names="time")
                # pd.DataFrame(
                #     data=entry[FieldName.TARGET].T,
                #     index=period_index(entry, freq=freq),
                # )
                df = pd.DataFrame(
                    data=entry[FieldName.TARGET].collect().to_numpy(),
                    index=pd.period_range(
                        start=entry[FieldName.START],
                        periods=entry[FieldName.TARGET].select(pl.len()).collect().item(),
                        freq=entry[FieldName.START].freq,
                    )
                ).reset_index(names="time")
                df["time"] = df["time"].dt.to_timestamp()
                df = df.rename(columns={col: f"{tgt}_{entry['item_id']}" for col, tgt in zip(df.columns[1:], self.target_cols)})
                
                for a in range(self.num_target_vars):
                    sns.lineplot(data=df, ax=axs[a], x="time", y=df.columns[a + 1])
                    self.highlight_entry(entry, colors[d], ax=axs[a], vlines=(0.0, 0.5))

        # for t, (test_input, test_label) in enumerate(self.test_dataset):
        #     for a in range(self.num_target_vars):
        #         # self.highlight_entry(test_input, colors[2], axs[a])
        #         self.highlight_entry(test_label, colors[3], axs[a])

            # plt.legend(["sub dataset", "test input", "test label"], loc="upper left")
        
        fig.show()
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=False
        )