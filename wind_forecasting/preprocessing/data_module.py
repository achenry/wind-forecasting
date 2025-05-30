from dataclasses import dataclass
from typing import List, Type, Optional
import os
import re
import logging
import torch
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
# from gluonts.dataset = Dataset

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
    _base_raw_data_path: str = "" # To store the original data_path
    
    def __post_init__(self):
        # Store the initial data_path provided at instantiation. This will be immutable.
        self._base_raw_data_path = self.data_path
        self.set_train_ready_path() # Still call this, but it will use _base_raw_data_path
            
        # convert context and prediction length from seconds to time steps based on freq
        self.context_length = int(pd.Timedelta(self.context_length, unit="s") / pd.Timedelta(self.freq))
        self.prediction_length = int(pd.Timedelta(self.prediction_length, unit="s") / pd.Timedelta(self.freq))
        assert self.context_length > 0, "context_length must be provided in seconds, and must be greater than resample_freq."
        assert self.prediction_length > 0, "prediction_length must be provided in seconds, and must be greater than resample_freq."
    
    def set_train_ready_path(self):
        # Use _base_raw_data_path to ensure the path derivation is always based on the original raw data location
        base_path = self._base_raw_data_path # Use the immutable original path
        
        if self.normalized:
            self.train_ready_data_path = base_path.replace(
                ".parquet", f"_train_ready_{self.freq}_{'per_turbine' if self.per_turbine_target else 'all_turbine'}.parquet")
        else:
            self.train_ready_data_path = base_path.replace(
                ".parquet", f"_train_ready_{self.freq}_{'per_turbine' if self.per_turbine_target else 'all_turbine'}_denormalize.parquet")
     
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
    
    # @profile # prints memory usage
    def get_dataset_info(self, dataset=None):
        # print(f"Number of nan/null vars = {dataset.select(pl.sum_horizontal((cs.numeric().is_null() | cs.numeric().is_nan()).sum())).collect().item()}") 
        if dataset is None:
            temp_dataset_lazy = IterableLazyFrame(data_path=self.train_ready_data_path, dtype=self.dtype)
            dataset_for_info = temp_dataset_lazy.collect() # Ensure it's eager for consistent processing below
        else:
            dataset_for_info = dataset # Already an eager DataFrame from generate_splits
        
        if self.verbose:
            logging.info("Getting continuity groups.") 
            
        if self.continuity_groups is None:
            if "continuity_group" in dataset_for_info.columns: # Check columns for eager DataFrame
                self.continuity_groups = dataset_for_info.select(pl.col("continuity_group").unique()).to_numpy().flatten() # No .collect()
            else:
                self.continuity_groups = [0]
                
        if self.verbose:
            logging.info(f"Found continuity groups {self.continuity_groups}")
            
            logging.info(f"Getting column names.") 
        if self.target_suffixes is None:
            self.target_cols = dataset_for_info.select(*[cs.starts_with(pfx) for pfx in self.target_prefixes]).columns # Use .columns for eager DF
            self.target_suffixes = sorted(list(set(col.split("_")[-1] for col in self.target_cols)), key=lambda col: int(re.search("\\d+", col).group()))
        else:
            self.target_cols = [col for col in dataset_for_info.columns if any(prefix in col for prefix in self.target_prefixes)] # Use .columns for eager DF
        self.feat_dynamic_real_cols = [col for col in dataset_for_info.columns if any(prefix in col for prefix in self.feat_dynamic_real_prefixes)] # Use .columns for eager DF
        
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
    def generate_splits(self, splits=None, save=False, reload=True, verbose=None):
        if verbose is None:
            verbose = self.verbose
            
        if splits is None:
            splits = ["train", "val", "test"]
        assert all(split in ["train", "val", "test"] for split in splits)
        assert os.path.exists(self.train_ready_data_path), f"Must run generate_datasets before generate_splits to produce {self.train_ready_data_path}."

        rank = 0
        world_size = 1
        is_distributed = dist.is_available() and dist.is_initialized()
        if is_distributed:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            logging.info(f"Rank {rank}/{world_size}: Entering generate_splits.")

        logging.info(f"Rank {rank}: Scanning dataset {self.train_ready_data_path}.")
        dataset_lazy = IterableLazyFrame(data_path=self.train_ready_data_path, dtype=self.dtype)
        logging.info(f"Rank {rank}: Finished scanning dataset {self.train_ready_data_path}.")

        # Materialize the entire dataset to an eager DataFrame here for efficient slicing
        logging.info(f"Rank {rank}: Collecting full dataset from {self.train_ready_data_path} to eager DataFrame.")
        dataset_eager = dataset_lazy.collect()
        logging.info(f"Rank {rank}: Finished collecting full dataset.")

        self.get_dataset_info(dataset_eager) # Pass eager dataset for info extraction
        split_files_exist = all(os.path.exists(self.train_ready_data_path.replace(".parquet", f"_{split}.pkl")) for split in splits)

        if rank == 0 and (reload or not split_files_exist):
            logging.info(f"Rank 0: Generating splits (reload={reload}, files_exist={split_files_exist}).")
            if self.per_turbine_target:
                if self.verbose:
                    logging.info(f"Rank 0: Splitting datasets for per turbine case.")
                
                # Operate on dataset_eager directly since it's already collected.
                cg_counts = dataset_eager.select("continuity_group").to_series().value_counts().sort("continuity_group").select("count").to_numpy().flatten()
                self.rows_per_split = [
                    int(n_rows / self.n_splits)
                    for turbine_id in self.target_suffixes
                    for n_rows in cg_counts] # each element corresponds to each continuity group
                del cg_counts
                self.continuity_groups = dataset_eager.select(pl.col("continuity_group").unique()).to_numpy().flatten()
                
                # Crucial change: Explicitly pass eager DataFrames to split_dataset. No more .collect() here.
                self.train_dataset, self.val_dataset, self.test_dataset = \
                    self.split_dataset([dataset_eager.filter(pl.col("continuity_group") == cg) for cg in self.continuity_groups])
                    
                for split in splits:
                    ds = getattr(self, f"{split}_dataset")
                    setattr(self, f"{split}_dataset", 
                            {f"TURBINE{turbine_id}_SPLIT{cg}": 
                            self.get_df_by_turbine(ds[cg], turbine_id) 
                            for turbine_id in self.target_suffixes for cg in range(len(ds))})
                
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
                                        assume_sorted=True, assume_resampled=True,unchecked=True))
                else:
                    
                    # convert dictionary of item_id: lazyframe datasets into list of dictionaries with numpy arrays for data
                    for split in splits:
                        datasets = []
                        item_ids = list(getattr(self, f"{split}_dataset").keys())
                        for item_id in item_ids:
                            if verbose:
                                logging.info(f"Transforming {split} dataset {item_id} into numpy form.")
                            ds = getattr(self, f"{split}_dataset")[item_id]
                            # Start time, end time and ds conversion should happen on eager DataFrames.
                            # Removed .collect() as ds is already an eager DataFrame from get_df_by_turbine.
                            dataset_item_eager = ds 
                            start_time = pd.Period(dataset_item_eager.select(pl.col("time").first()).item(), freq=self.freq)
                            dataset_item_eager = dataset_item_eager.select(self.feat_dynamic_real_cols + self.target_prefixes).to_numpy().T
                            datasets.append({
                                "target": dataset_item_eager[-len(self.target_prefixes):, :],
                                 "item_id": item_id,
                                 "start": start_time,
                                 "feat_static_cat": [self.target_suffixes.index(re.search("(?<=TURBINE)\\w+(?=_SPLIT)", item_id).group(0))],
                                 "feat_dynamic_real": dataset_item_eager[:-len(self.target_prefixes), :]
                            })
                            del getattr(self, f"{split}_dataset")[item_id]
                        setattr(self, f"{split}_dataset", datasets)
                if verbose:
                    logging.info(f"Finished splitting datasets for per turbine case.") 
                
            else:
                if verbose:
                    logging.info(f"Splitting datasets for all turbine case.") 
                
                # Operate on dataset_eager directly since it's already collected.
                cg_counts = dataset_eager.select("continuity_group").to_series().value_counts().sort("continuity_group").select("count").to_numpy().flatten()
                self.rows_per_split = [int(n_rows / self.n_splits) for n_rows in cg_counts] # each element corresponds to each continuity group
                del cg_counts
                self.continuity_groups = dataset_eager.select(pl.col("continuity_group").unique()).to_numpy().flatten()
                
                # Crucial change: Explicitly pass eager DataFrames to split_dataset. No more .collect() here.
                self.train_dataset, self.val_dataset, self.test_dataset = \
                    self.split_dataset([dataset_eager.filter(pl.col("continuity_group") == cg) for cg in self.continuity_groups]) # Removed .collect()
                
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
                                PolarsDataset(ds, 
                                        target=self.target_prefixes, timestamp="time", freq=self.freq, 
                                        feat_dynamic_real=self.feat_dynamic_real_prefixes, static_features=self.static_features, 
                                        assume_sorted=True, assume_resampled=True,unchecked=True))
                        
                else:
                    
                    # convert dictionary of item_id: lazyframe datasets into list of dictionaries with numpy arrays for data
                    for split in splits:
                        datasets = []
                        item_ids = list(getattr(self, f"{split}_dataset").keys())
                        for item_id in item_ids:
                            if self.verbose:
                                logging.info(f"Transforming {split} dataset {item_id} into numpy form.")
                            ds = getattr(self, f"{split}_dataset")[item_id]
                            # Start time, end time and ds conversion should happen on eager DataFrames.
                            # Removed .collect() as ds is already an eager DataFrame from the split_dataset output.
                            dataset_item_eager = ds 
                            start_time = pd.Period(dataset_item_eager.select(pl.col("time").first()).item(), freq=self.freq)
                            dataset_item_eager = dataset_item_eager.select(self.feat_dynamic_real_cols + self.target_cols).to_numpy().T
                            datasets.append({
                                "target": dataset_item_eager[-len(self.target_cols):, :],
                                 "item_id": item_id,
                                 "start": start_time,
                                 "feat_dynamic_real": dataset_item_eager[:-len(self.target_cols), :]
                            })
                            del getattr(self, f"{split}_dataset")[item_id]
                        setattr(self, f"{split}_dataset", datasets)
                if verbose:
                    logging.info(f"Finished splitting datasets for all turbine case.")
            
            if save:
                logging.info(f"Rank 0: Saving generated splits.")
                for split in splits:
                    if self.as_lazyframe:
                        raise NotImplementedError("Saving LazyFrame splits not implemented.")
                    else:
                        final_path = self.train_ready_data_path.replace(".parquet", f"_{split}.pkl")
                        temp_path = final_path + ".tmp"
                        logging.info(f"Rank 0: Saving {split} data to {temp_path}")
                        try:
                            with open(temp_path, 'wb') as fp:
                                pickle.dump(getattr(self, f"{split}_dataset"), fp)
                            logging.info(f"Rank 0: Automically moving {temp_path} to {final_path}")
                            os.rename(temp_path, final_path)
                        except Exception as e:
                            logging.error(f"Rank 0: Error saving {split} data: {e}")
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                            raise
                        finally:
                            if os.path.exists(temp_path):
                                try:
                                    os.remove(temp_path)
                                except OSError as e:
                                     logging.error(f"Rank 0: Error removing temp file {temp_path}: {e}")
 
        if is_distributed:
            logging.info(f"Rank {rank}: Waiting at barrier before loading splits.")
            dist.barrier()
            logging.info(f"Rank {rank}: Passed barrier.")
        
        if rank != 0 or (not reload and split_files_exist):
            logging.info(f"Rank {rank}: Loading saved split datasets.")
            for split in splits:
                split_path = self.train_ready_data_path.replace(".parquet", f"_{split}.pkl")
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
        min_observation_length = self.context_length + self.prediction_length

        for cg, df_cg in enumerate(dataset): # df_cg is an eager DataFrame per continuity group now as per generate_splits changes
            
            # TODO total past length may also have to supercede context_len + max(self.lags_seq)
            # The previous check was a bit complex due to round() and self.rows_per_split
            # Replaced with a direct check on the collected DataFrame's length
            if len(df_cg) < min_observation_length:
                logging.info(f"Can't split dataset corresponding to continuity group {cg} into training, validation, testing, the full dataset only has data points {len(df_cg)} (min needed: {min_observation_length}).")
                
                # If it's too short for splitting, add whole thing to train if long enough for an observation
                if len(df_cg) >= min_observation_length:
                    logging.info(f"Adding dataset corresponding to continuity group {cg} to training data, since it's too short to be split proportionally but can form observations.")
                    train_datasets.append(df_cg.select(pl.exclude("continuity_group"))) # Exclude cg column here
                else:
                    logging.info(f"Continuity group {cg} (length {len(df_cg)}) is too short for any observation ({min_observation_length} needed). Skipping entirely.")
                continue
            
            # Splitting each continuity group into subsections, a training/val/test dataset will then be generated from each subsection.
            # We do this to get a coherent mix of training, val, test data that is more independent of trends over time
            segments = [] # Renamed from 'datasets' to avoid confusion with overall train/val/test_datasets
            num_rows_in_cg = len(df_cg) # Length of the current entire continuity group DataFrame
            
            # Use self.n_splits to segment the current continuity group DataFrame
            rows_per_split_segment = num_rows_in_cg // self.n_splits # Integer division for segment size

            for split_idx in range(self.n_splits):
                start_idx = split_idx * rows_per_split_segment
                # For the last segment, take all remaining rows to avoid truncation
                end_idx = (split_idx + 1) * rows_per_split_segment if split_idx < self.n_splits - 1 else num_rows_in_cg

                current_segment = df_cg.slice(start_idx, end_idx - start_idx).select(pl.exclude("continuity_group"))
                
                # Only add if the segment itself is long enough for at least one observation
                if len(current_segment) >= min_observation_length:
                    segments.append(current_segment)
                    # Log info about the segment
                    start_time = current_segment.select(pl.col("time").first()).item()
                    end_time = current_segment.select(pl.col("time").last()).item()
                    duration = end_time - start_time
                    logging.info(f"full dataset cg {cg} split {split_idx}, start time = {start_time}, end time = {end_time}, duration = {duration}")
                else:
                    logging.warning(f"Segment from CG {cg} split {split_idx} (length {len(current_segment)}) too short for observation. Skipping this segment.")
            
            if not segments:
                logging.warning(f"No valid segments generated for continuity group {cg}. Skipping this group for proportional splitting.")
                continue # Skip this continuity group if no valid segments could be formed

            # Process each sub-segment (which are now eager DataFrames)
            for s in segments: # 's' is now an eager DataFrame segment
                current_sub_segment_len = len(s) # Direct length of DataFrame
                
                # Calculate specific split points for this sub-segment based on its actual length
                train_slice_len = int(current_sub_segment_len * self.train_split)
                val_slice_len = int(current_sub_segment_len * self.val_split)
                
                # Ensure minimum length for slices
                # Important: Slices must be at least min_observation_length to be useful for models
                # The remaining part implicitly becomes the test slice
                
                temp_train_segment = s.slice(0, train_slice_len)
                temp_val_segment = s.slice(train_slice_len, val_slice_len)
                temp_test_segment = s.slice(train_slice_len + val_slice_len) # Slice from here to end
                
                # Add to respective lists only if the segment is long enough
                if len(temp_train_segment) >= min_observation_length:
                    train_datasets.append(temp_train_segment)
                else:
                    logging.warning(f"Train part of sub-segment from CG {cg} (len {len(temp_train_segment)}) too short, skipping from train_datasets.")
                
                if len(temp_val_segment) >= min_observation_length:
                    val_datasets.append(temp_val_segment)
                else:
                    logging.warning(f"Validation part of sub-segment from CG {cg} (len {len(temp_val_segment)}) too short, skipping from val_datasets. This may lead to no validation samples.")
                
                if len(temp_test_segment) >= min_observation_length:
                    test_datasets.append(temp_test_segment)
                else:
                    logging.warning(f"Test part of sub-segment from CG {cg} (len {len(temp_test_segment)}) too short, skipping from test_datasets.")
                
            logging.info(f"CG {cg} Sub-segment split details (Min {min_observation_length} needed): "
                         f"Train: {len(temp_train_segment)}, " # Use final length of appended segment
                         f"Val: {len(temp_val_segment)}, "
                         f"Test: {len(temp_test_segment)}")
            
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
                #     index=pd.period_range(
                #         start=entry[FieldName.START],
                #         periods=entry[FieldName.TARGET].select(pl.len()).collect().item(),
                #         freq=entry[FieldName.START].freq,
                #     )
                # ).reset_index(names="time")
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
        #         # self.highlight_entry
        #         # self.highlight_entry(test_label, colors[3], axs[a])
        
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