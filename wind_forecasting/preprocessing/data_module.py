from dataclasses import dataclass
from typing import List, Type, Optional
import os
import shutil
import re
import logging
import time
import torch.distributed as dist

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

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
class DataModule:
    """_summary_
    # DataModule should use a polars LazyFrame and sink it into a parquet,
    # and store the indices in the full dataset to use for each cg, split_idx, and training/test/validation split
    """

    normalized_data_path: str
    n_splits: int
    continuity_groups: List[int] | None
    train_split: float
    val_split: float
    test_split: float
    prediction_length: int
    context_length: int
    target_prefixes: List[str]
    target_suffixes: (
        List[str] | None
    )  # ie turbine ids after the last underscore eg wt001 in  ws_horz_wt001
    feat_dynamic_real_prefixes: List[str]
    freq: str
    per_turbine_target: bool  # if True, feed multiple datasets to trainer, where each one corresponds to the outputs of a single turbine
    dtype: Type = pl.Float32
    as_lazyframe: bool = False
    verbose: bool = False
    use_normalization: bool = True
    normalization_consts_path: Optional[str] = None
    batch_size: int = 128
    workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True

    def __post_init__(self):

        assert "normalized" in self.normalized_data_path, (
            "normalized_data_path must point to a file with normalized data"
        )
        # convert context and prediction length from seconds to time stesp based on freq
        self.context_length = int(
            pd.Timedelta(self.context_length, unit="s") / pd.Timedelta(self.freq)
        )
        self.prediction_length = int(
            pd.Timedelta(self.prediction_length, unit="s") / pd.Timedelta(self.freq)
        )
        assert self.context_length > 0, (
            "context_length must be provided in seconds, and must be greaterthan resample_freq."
        )
        assert self.prediction_length > 0, (
            "prediction_length must be provided in seconds, and must be greaterthan resample_freq."
        )

        self.set_train_ready_path()

    @property
    def train_dataset(self):
        """Return the training dataset from the datasets dictionary.

        Provides backward-compatible attribute access for code that expects
        data_module.train_dataset instead of data_module.datasets["train"].
        """
        if not hasattr(self, "datasets") or self.datasets is None:
            return None
        return self.datasets.get("train")

    @property
    def val_dataset(self):
        """Return the validation dataset from the datasets dictionary.

        Provides backward-compatible attribute access for code that expects
        data_module.val_dataset instead of data_module.datasets["val"].
        """
        if not hasattr(self, "datasets") or self.datasets is None:
            return None
        return self.datasets.get("val")

    @property
    def test_dataset(self):
        """Return the test dataset from the datasets dictionary.

        Provides backward-compatible attribute access for code that expects
        data_module.test_dataset instead of data_module.datasets["test"].
        """
        if not hasattr(self, "datasets") or self.datasets is None:
            return None
        return self.datasets.get("test")

    def set_train_ready_path(self):
        sfx = f"ctx{self.context_length}_pred{self.prediction_length}"
        if self.use_normalization:
            self.train_ready_data_path = self.normalized_data_path.replace(
                ".parquet",
                f"_train_ready_{self.freq}_{'per_turbine' if self.per_turbine_target else 'all_turbine'}_{sfx}.parquet",
            )
        else:
            self.train_ready_data_path = self.normalized_data_path.replace(
                ".parquet",
                f"_train_ready_{self.freq}_{'per_turbine' if self.per_turbine_target else 'all_turbine'}_{sfx}_denormalize.parquet",
            )

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
        if self.as_lazyframe:
            return f"{base_path}_{split}{suffix}.parquet"
        else:
            return f"{base_path}_{split}{suffix}.pkl"

    def _validate_loaded_splits(self, splits, rank):
        """Validate that loaded splits are compatible with current context_length requirements."""
        min_required_length = self.context_length + self.prediction_length

        for split in splits:
            dataset = self.datasets[split]
            if (
                self.as_lazyframe and dataset.select(pl.len()).collect().item() == 0
            ) or (
                not self.as_lazyframe
                and sum(ds["target"].shape[1] for ds in dataset) == 0
            ):
                logging.warning(
                    f"Rank {rank}: {split} dataset is empty! This may cause validation issues."
                )
                continue

            # Check if dataset has enough samples for the current context_length requirements
            if isinstance(dataset, list) and len(dataset) > 0:
                # Check a representative sample from the dataset
                sample = dataset[0] if len(dataset) else None
                if sample and "target" in sample:
                    target_length = (
                        sample["target"].shape[1]
                        if hasattr(sample["target"], "shape")
                        else len(sample["target"][0])
                    )
                    if target_length < min_required_length:
                        logging.error(
                            f"Rank {rank}: {split} dataset samples are too short ({target_length}) for context_length={self.context_length} + prediction_length={self.prediction_length} = {min_required_length}"
                        )
                        raise ValueError(
                            f"Loaded {split} dataset is incompatible with current context_length={self.context_length}"
                        )
            elif isinstance(dataset, pl.LazyFrame):
                # For Polars DataFrame, use a single group_by to get all item lengths at once (O(1) vs O(n))
                item_lengths = (
                    dataset.group_by("item_id")
                    .agg(pl.len().alias("_item_len"))
                    .collect()
                )
                total_items = item_lengths.height
                short_items_df = item_lengths.filter(
                    pl.col("_item_len") < min_required_length
                ).sort("_item_len")

                if short_items_df.height > 0:
                    shortest_len = short_items_df["_item_len"].min()

                    logging.warning(
                        f"Rank {rank}: {split} dataset has {short_items_df.height}/{total_items} items shorter than "
                        f"min_required_length={min_required_length} (context_length={self.context_length} + "
                        f"prediction_length={self.prediction_length}). Shortest item has {shortest_len} rows."
                    )

                    # Show up to 10 shortest items for debugging
                    for row in short_items_df.head(10).iter_rows():
                        logging.warning(
                            f"  - item_id='{row[0]}' has only {row[1]} rows (need >= {min_required_length})"
                        )
                    if short_items_df.height > 10:
                        logging.warning(
                            f"  ... and {short_items_df.height - 10} more short items"
                        )

                    # Filter out short items instead of failing
                    valid_items = item_lengths.filter(
                        pl.col("_item_len") >= min_required_length
                    )["item_id"]
                    if valid_items.len() == 0:
                        raise ValueError(
                            f"ALL {total_items} items in {split} dataset are shorter than "
                            f"context_length={self.context_length} + prediction_length={self.prediction_length} = {min_required_length}. "
                            f"Cannot proceed — reduce context_length or use longer data."
                        )

                    self.datasets[split] = dataset.filter(
                        pl.col("item_id").is_in(valid_items)
                    )
                    logging.warning(
                        f"Rank {rank}: Filtered {split} dataset from {total_items} to {valid_items.len()} items "
                        f"(removed {short_items_df.height} short items)"
                    )

        logging.info(
            f"Rank {rank}: Validation passed for loaded splits with context_length={self.context_length}, prediction_length={self.prediction_length}"
        )

    def compute_scaler_params(self):
        # TODO add option for different normalizers and include in yaml
        norm_consts = pd.read_csv(self.normalization_consts_path, index_col=None)

        norm_mean_cols = [col for col in norm_consts if col.endswith("_mean")]
        norm_scale_cols = [col for col in norm_consts if col.endswith("_std")]

        if len(norm_mean_cols) > 0 and len(norm_scale_cols) > 0:
            # norm_mean_cols = [col.replace("_mean", "") for col in norm_mean_cols]
            # norm_scale_cols = [col.replace("_std", "") for col in norm_scale_cols]
            return {
                "offset_": {
                    k.replace("_mean", ""): v
                    for k, v in norm_consts[norm_mean_cols].iloc[0].to_dict().items()
                },
                "scale_": {
                    k.replace("_std", ""): v
                    for k, v in norm_consts[norm_scale_cols].iloc[0].to_dict().items()
                },
            }

        norm_min_cols = [col for col in norm_consts if col.endswith("_min")]
        norm_max_cols = [col for col in norm_consts if col.endswith("_max")]
        data_min = norm_consts[norm_min_cols].values.flatten()
        data_max = norm_consts[norm_max_cols].values.flatten()

        if len(norm_min_cols) > 0 and len(norm_max_cols) > 0:
            norm_min_cols = [col.replace("_min", "") for col in norm_min_cols]
            norm_max_cols = [col.replace("_max", "") for col in norm_max_cols]
            feature_range = (-1, 1)
            norm_scale = (feature_range[1] - feature_range[0]) / (data_max - data_min)
            norm_min = feature_range[0] - (data_min * norm_scale)
            return {
                "offset_": dict(zip(norm_min_cols, norm_min)),
                "scale_": dict(zip(norm_min_cols, norm_scale)),
            }

    def generate_datasets(self):

        dataset = IterableLazyFrame(
            data_path=self.normalized_data_path, dtype=self.dtype
        )
        # dataset = dataset.filter(pl.col("continuity_group").is_in([507, 1249,  388,  400,  791]))
        # dataset = dataset.head(1000000)

        # add warning if upsampling
        dataset_dt = dataset.select(pl.col("time").diff()).slice(1, 1).collect().item()

        if dataset_dt.total_seconds() > int(re.search("\\d+", self.freq).group()):
            logging.warning(
                f"Downsampling dataset with frequency of {dataset_dt} seconds to {self.freq}."
            )

        dataset = (
            dataset.with_columns(time=pl.col("time").dt.round(self.freq))
            .group_by("time")
            .agg(cs.numeric().mean())
            .sort(["continuity_group", "time"])
        )

        if False:
            suffixes = ["_wt005", "_wt074", "_wt075"]
            for cg in (
                dataset.select(pl.col("continuity_group").unique())
                .collect()
                .to_numpy()
                .flatten()
            ):
                fig, ax = plt.subplots(len(self.target_prefixes), 1, sharex=True)
                ds = dataset.filter(pl.col("continuity_group") == cg)
                time = ds.select("time").collect().to_numpy()
                for ax_idx, feat_type in enumerate(self.target_prefixes):
                    ax[ax_idx].plot(
                        time,
                        ds.select([f"{feat_type}{sfx}" for sfx in suffixes])
                        .collect()
                        .to_numpy(),
                        linestyle="-",
                        label="original",
                    )
                    ax[ax_idx].set(title=feat_type)  # , xlim=(1.295*1e7,1.345*1e7))
            fig.show()

        if not self.use_normalization:
            # if we don't want the normalized data generated in preprocessing_main, we need to denormalize/inverse transform it here

            scaler_params = self.compute_scaler_params()
            # feat_types = list(scaler_params["min_"])
            # dataset = dataset.with_columns([(cs.starts_with(feat_type) - scaler_params["min_"][feat_type])
            #                                             / scaler_params["scale_"][feat_type]
            #                                             for feat_type in feat_types])
            features = list(scaler_params["offset_"])
            dataset = dataset.with_columns(
                [
                    (pl.col(feat) * scaler_params["scale_"][feat])
                    + scaler_params["offset_"][feat]
                    for feat in features
                ]
            )

        # TODO if resampling requires upsampling: historic_measurements.upsample(time_column="time", every=self.data_module.freq).fill_null(strategy="forward")
        # dataset = IterableLazyFrame(data_path=self.train_ready_data_path, dtype=self.dtype) # data stored in RAM
        # gc.collect()

        # fetch a subset of continuity groups and turbine data
        if self.verbose:
            logging.info("Getting continuity groups.")
        if self.continuity_groups is None:
            if "continuity_group" in dataset.collect_schema().names():
                self.continuity_groups = (
                    dataset.select(pl.col("continuity_group").unique())
                    .collect()
                    .to_numpy()
                    .flatten()
                )
                if self.target_suffixes is not None:
                    dataset = dataset.select(
                        pl.col("time"),
                        pl.col("continuity_group"),
                        *[cs.ends_with(sfx) for sfx in self.target_suffixes],
                    )

            else:
                self.continuity_groups = [0]
                # dataset = dataset[[col for col in dataset.columns if any(col.__contains__(tid) for tid in self.target_suffixes)]]
                # dataset.loc[:, "continuity_group"] = 0
                if self.target_suffixes is not None:
                    dataset = dataset.select(
                        pl.col("time"),
                        *[cs.ends_with(sfx) for sfx in self.target_suffixes],
                    )
                dataset = dataset.with_columns(continuity_group=pl.lit(0))
        else:
            dataset = dataset.filter(
                pl.col("continuity_group").is_in(self.continuity_groups)
            ).select(
                pl.col("time"),
                pl.col("continuity_group"),
                *[cs.ends_with(f"_{sfx}") for sfx in self.target_suffixes],
            )
        if self.verbose:
            logging.info(f"Found continuity groups {self.continuity_groups}")
        # dataset.target_cols = self.target_cols

        temp_fp = f"{self.train_ready_data_path}.done"

        if self.verbose:
            logging.info(f"Writing resampled/sorted parquet to {temp_fp}")

        # TODO HIGH there may be a racing condition here if multiple workers are writing to the same file e.g. for tuning
        # confirm that this only occurs on rank 0

        dataset.collect().write_parquet(temp_fp)

        # if os.path.exists(self.train_ready_data_path):
        #     os.remove(self.train_ready_data_path)
        # os.rename(temp_fp, self.train_ready_data_path)
        # Atomic rename - if file exists, this will overwrite it atomically
        logging.info(
            f"Rank 0: Atomically moving {temp_fp} to {self.train_ready_data_path}"
        )
        os.replace(temp_fp, self.train_ready_data_path)  # os.replace is atomic on POSIX

        if self.verbose:
            logging.info(
                f"Saved resampled/sorted parquet to {self.train_ready_data_path}."
            )

        self.get_dataset_info(dataset)
        # dataset = IterableLazyFrame(data_path=train_ready_data_path)
        # univariate=ListDataset of multiple dictionaires each corresponding to measurements from a single turbine, to implicitly capture correlations
        # or multivariate=multivariate dictionary for all measurements, to explicity capture all correlations
        # or debug= to use electricity dataset

    # @profile
    def get_dataset_info(self, dataset=None):
        # print(f"Number of nan/null vars = {dataset.select(pl.sum_horizontal((cs.numeric().is_null() | cs.numeric().is_nan()).sum())).collect().item()}")
        if dataset is None:
            assert os.path.exists(self.train_ready_data_path), (
                f"train_ready_data_path, {self.train_ready_data_path}, doesn't exist! Should be generated for training."
            )
            dataset = IterableLazyFrame(
                data_path=self.train_ready_data_path, dtype=self.dtype
            )

        if self.verbose:
            logging.info("Getting continuity groups.")

        if self.continuity_groups is None:
            if "continuity_group" in dataset.collect_schema().names():
                # TODO this is giving floats??
                self.continuity_groups = (
                    dataset.select(pl.col("continuity_group").unique())
                    .collect()
                    .to_numpy()
                    .flatten()
                )
            else:
                self.continuity_groups = [0]

        if self.verbose:
            logging.info(f"Found continuity groups {self.continuity_groups}")

            logging.info(f"Getting column names.")
        if self.target_suffixes is None:
            self.target_cols = (
                dataset.select(*[cs.starts_with(pfx) for pfx in self.target_prefixes])
                .collect_schema()
                .names()
            )
            self.target_suffixes = sorted(
                list(set(col.split("_")[-1] for col in self.target_cols)),
                key=lambda col: int(re.search("\\d+", col).group()),
            )
        else:
            self.target_cols = [
                col
                for col in dataset.collect_schema().names()
                if any(prefix in col for prefix in self.target_prefixes)
            ]
        self.feat_dynamic_real_cols = [
            col
            for col in dataset.collect_schema().names()
            if any(prefix in col for prefix in self.feat_dynamic_real_prefixes)
        ]

        if self.verbose:
            logging.info(
                f"Found column names target_cols={self.target_cols}, feat_dynamic_real_cols={self.feat_dynamic_real_cols}."
            )

        if self.per_turbine_target:
            self.num_target_vars = len(self.target_prefixes)
            self.num_feat_dynamic_real = int(
                len(self.feat_dynamic_real_cols) / len(self.target_suffixes)
            )
            self.num_feat_static_cat = 1
            self.num_feat_static_real = 0
            # self.target_cols = self.target_prefixes

            # self.cardinality = list(self.static_features.select_dtypes("category").nunique().values)
            self.static_features = None  # allow to generate itself
            self.cardinality = (len(self.target_suffixes),)

        else:
            self.num_feat_dynamic_real = len(self.feat_dynamic_real_cols)
            self.num_target_vars = len(self.target_cols)
            self.num_feat_static_cat = 0
            self.num_feat_static_real = 0

            self.static_features = pd.DataFrame()
            self.cardinality = None

    # @profile # prints memory usage
    def generate_splits(
        self, splits=None, save=False, reload=True, verbose=None, rank=None
    ):
        if verbose is None:
            verbose = self.verbose

        if splits is None:
            splits = ["train", "val", "test"]
        assert all(split in ["train", "val", "test"] for split in splits)
        assert os.path.exists(self.train_ready_data_path), (
            f"Must run generate_datasets before generate_splits to produce {self.train_ready_data_path}."
        )

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
                logging.info(
                    f"Rank {rank}/{world_size}: Detected PyTorch distributed mode."
                )
            else:
                # Check for WORKER_RANK environment variable as fallback
                # This handles tuning mode where workers are independent
                worker_rank = os.environ.get("WORKER_RANK", None)
                logging.info(f"WORKER_RANK environment variable: {worker_rank}")
                logging.info(f"Process ID: {os.getpid()}")

                if worker_rank is not None:
                    try:
                        rank = int(worker_rank)
                        logging.info(
                            f"Rank {rank}: Using WORKER_RANK={worker_rank} from environment (independent worker mode)."
                        )
                    except ValueError:
                        logging.warning(
                            f"Invalid WORKER_RANK value: '{worker_rank}', defaulting to rank 0"
                        )
                        rank = 0
                else:
                    rank = 0
                    logging.info(
                        "Rank 0: No WORKER_RANK found, no distributed mode detected, assuming single process."
                    )
        else:
            # Rank was explicitly provided
            logging.info(f"Rank {rank}: Using explicitly provided rank value.")

        split_files_exist = all(
            os.path.exists(self.get_split_file_path(split)) for split in splits
        )
        logging.info(
            f"Rank {rank}: Split file check - context_length={self.context_length}, prediction_length={self.prediction_length}, split_files_exist={split_files_exist}"
        )
        for split in splits:
            split_path = self.get_split_file_path(split)
            exists = os.path.exists(split_path)
            logging.info(
                f"Rank {rank}: Split file {split}: {split_path} (exists: {exists})"
            )

        if self.as_lazyframe:
            temp_dir = os.path.join(
                os.path.dirname(self.train_ready_data_path), "splits_temp"
            )
            os.makedirs(temp_dir, exist_ok=True)

        if rank == 0 and (reload or not split_files_exist):
            logging.info(f"Rank {rank}: Scanning dataset {self.train_ready_data_path}.")
            dataset = IterableLazyFrame(
                data_path=self.train_ready_data_path, dtype=self.dtype
            )
            logging.info(
                f"Rank {rank}: Finished scanning dataset {self.train_ready_data_path}."
            )

            # ds = dataset._df.collect().partition_by("continuity_group")
            # sub_ds = ds[9]
            # fig, axs = plt.subplots(4, 1, figsize=(10, 8))
            # axs[0].plot(sub_ds.select(pl.col("time")).to_numpy().flatten(),
            #             sub_ds.select(cs.starts_with("ws_horz")).to_numpy())
            # axs[1].plot(sub_ds.select(pl.col("time")).to_numpy().flatten(),
            #             sub_ds.select(cs.starts_with("ws_vert")).to_numpy())
            # axs[2].plot(sub_ds.select(pl.col("time")).to_numpy().flatten(),
            #             sub_ds.select(cs.starts_with("nd_cos")).to_numpy())
            # axs[3].plot(sub_ds.select(pl.col("time")).to_numpy().flatten(),
            #             sub_ds.select(cs.starts_with("nd_sin")).to_numpy())
            # for sub_ds in ds:

            # sets self.continuity_groups, self.target_cols, self.target_suffixes, self.feat_dynamic_real_cols, self.num_target_vars,
            #       self.num_feat_dynamic_real, self.num_feat_static_cat, self.num_feat_static_real, self.static_features, self.cardinality
            self.get_dataset_info(dataset)

            logging.info(
                f"Rank 0: Generating splits (reload={reload}, files_exist={split_files_exist})."
            )
            if self.per_turbine_target:
                if self.verbose:
                    logging.info(
                        f"Rank 0: Splitting datasets for per turbine case into train/val/test={self.train_split}/{self.val_split}/{self.test_split}."
                    )

                cg_counts = (
                    dataset.select("continuity_group")
                    .collect()
                    .to_series()
                    .value_counts()
                    .sort("continuity_group")
                    .select("count")
                    .to_numpy()
                    .flatten()
                )

                self.rows_per_split = [
                    int(n_rows / self.n_splits)
                    for turbine_id in self.target_suffixes
                    for n_rows in cg_counts
                ]  # each element corresponds to each continuity group

                del cg_counts

                self.continuity_groups = (
                    dataset.select(pl.col("continuity_group").unique())
                    .collect()
                    .to_numpy()
                    .flatten()
                )

                # self.continuity_groups = [0,1,2]
                datasets = self.split_dataset(
                    [
                        dataset.filter(pl.col("continuity_group") == cg)
                        for cg in self.continuity_groups
                    ],
                    splits,
                )

                if self.as_lazyframe:
                    for split in splits:
                        # setattr(self, f"{split}_dataset", pl.collect_all(getattr(self, f"{split}_dataset")))
                        split_ds = datasets[split]

                        for d, ds in enumerate(split_ds):
                            for turbine_id in self.target_suffixes:
                                item_id = f"TURBINE{turbine_id}_SPLIT{d}"
                                temp_path = os.path.join(
                                    temp_dir,
                                    os.path.basename(self.normalized_data_path).replace(
                                        ".parquet", f"_{split}_{item_id}.parquet"
                                    ),
                                )

                                if reload or not os.path.exists(temp_path):
                                    if verbose:
                                        logging.info(
                                            f"Transforming {split} dataset {item_id} into polars form."
                                        )

                                    ds_len = ds.select(pl.len()).collect().item()
                                    start_time = pd.Period(
                                        ds.select(pl.col("time").first())
                                        .collect()
                                        .item(),
                                        freq=self.freq,
                                    )
                                    dt = pd.Timedelta(start_time.freq)

                                    self.get_df_by_turbine(ds, turbine_id).select(
                                        item_id=pl.lit(item_id),
                                        time=pl.datetime_range(
                                            start=start_time.start_time,
                                            end=start_time.start_time + (dt * ds_len),
                                            interval=dt,
                                            time_unit="ns",
                                            closed="left",
                                        ),
                                        feat_static_cat=pl.lit(
                                            [
                                                self.target_suffixes.index(
                                                    re.search(
                                                        "(?<=TURBINE)\\w+(?=_SPLIT)",
                                                        item_id,
                                                    ).group(0)
                                                )
                                            ]
                                        ),
                                        *[
                                            pl.col(col).alias(f"target_{i}")
                                            for i, col in enumerate(
                                                self.target_prefixes
                                            )
                                        ],
                                        *[
                                            pl.col(col).alias(f"feat_dynamic_real_{i}")
                                            for i, col in enumerate(
                                                self.feat_dynamic_real_prefixes
                                            )
                                        ],
                                    ).sink_parquet(temp_path, maintain_order=True)
                                elif verbose:
                                    logging.info(
                                        f"Loading existing cached file for {split} dataset {item_id}."
                                    )

                        # datasets[split] = pl.concat([
                        #     pl.scan_parquet(os.path.join(
                        #             temp_dir,
                        #             os.path.basename(self.data_path).replace(".parquet", f"_{split}_TURBINE{turbine_id}_SPLIT{d}.parquet")
                        #         )) for d in range(len(split_ds)) for turbine_id in self.target_suffixes],
                        #     how="vertical")

                        datasets[split] = pl.scan_parquet(
                            os.path.join(
                                temp_dir,
                                os.path.basename(self.normalized_data_path).replace(
                                    ".parquet", f"_{split}_TURBINE*_SPLIT*.parquet"
                                ),
                            ),
                            glob=True,
                        ).sort("item_id", maintain_order=True)

                else:
                    # convert dictionary of item_id: lazyframe datasets into list of dictionaries with numpy arrays for data
                    for split in splits:
                        transformed_datasets = []
                        # item_ids = list(getattr(self, f"{split}_dataset").keys())
                        # setattr(self, f"{split}_dataset", pl.collect_all(getattr(self, f"{split}_dataset")))
                        split_ds = datasets[split]
                        for d, ds in enumerate(split_ds):
                            for turbine_id in self.target_suffixes:
                                item_id = f"TURBINE{turbine_id}_SPLIT{d}"
                                if verbose:
                                    logging.info(
                                        f"Transforming {split} dataset {item_id} into numpy form."
                                    )
                                # ds = getattr(self, f"{split}_dataset")[item_id]
                                start_time = pd.Period(
                                    ds.select(pl.col("time").first()).collect().item(),
                                    freq=self.freq,
                                )
                                # self.get_df_by_turbine(split_ds[d], turbine_id)
                                # ds = split_ds[d].select(self.feat_dynamic_real_prefixes + self.target_prefixes)
                                ds = (
                                    self.get_df_by_turbine(ds, turbine_id)
                                    .collect()
                                    .to_numpy()
                                    .T
                                )
                                transformed_datasets.append(
                                    {
                                        "target": ds[-len(self.target_prefixes) :, :],
                                        "item_id": item_id,
                                        "start": start_time,
                                        "feat_static_cat": [
                                            self.target_suffixes.index(
                                                re.search(
                                                    "(?<=TURBINE)\\w+(?=_SPLIT)",
                                                    item_id,
                                                ).group(0)
                                            )
                                        ],
                                        "feat_dynamic_real": ds[
                                            1 : -len(self.target_prefixes), :
                                        ],
                                    }
                                )

                            # del getattr(self, f"{split}_dataset")[item_id]
                        datasets[split] = transformed_datasets

                if verbose:
                    logging.info(f"Finished splitting datasets for per turbine case.")

            else:
                if verbose:
                    logging.info(
                        f"Splitting datasets for all turbine case into train/val/test={self.train_split}/{self.val_split}/{self.test_split}."
                    )

                cg_counts = (
                    dataset.select("continuity_group")
                    .collect()
                    .to_series()
                    .value_counts()
                    .sort("continuity_group")
                    .select("count")
                    .to_numpy()
                    .flatten()
                )
                self.rows_per_split = [
                    int(n_rows / self.n_splits) for n_rows in cg_counts
                ]  # each element corresponds to each continuity group
                del cg_counts

                self.continuity_groups = (
                    dataset.select(pl.col("continuity_group").unique())
                    .collect()
                    .to_numpy()
                    .flatten()
                )

                # generate an iterablelazy frame for each continuity group and split within it
                datasets = self.split_dataset(
                    [
                        dataset.filter(pl.col("continuity_group") == cg)
                        for cg in self.continuity_groups
                    ],
                    splits,
                )

                # Persist the per-split lists on self so downstream getattr(self, "{split}_dataset") works.
                # The property accessors (train_dataset / val_dataset / test_dataset, lines 72-103) read from
                # self.datasets, so we assign there rather than setattr (the properties are read-only).
                if not hasattr(self, "datasets") or self.datasets is None:
                    self.datasets = {}
                for split in splits:
                    self.datasets[split] = datasets[split]

                if False:
                    split = "test"
                    suffixes = ["wt005", "wt074", "wt075"]
                    for cg, ds in enumerate(datasets[split]):
                        fig, ax = plt.subplots(
                            len(self.target_prefixes), 1, sharex=True
                        )
                        time = ds.select("time").collect().to_numpy()
                        for ax_idx, feat_type in enumerate(self.target_prefixes):
                            ax[ax_idx].plot(
                                time[:-400, :],
                                ds.select([f"{feat_type}_{sfx}" for sfx in suffixes])
                                .collect()
                                .to_numpy()[:-400, :],
                                linestyle="-",
                                label="original",
                            )
                            ax[ax_idx].set(
                                title=feat_type
                            )  # , xlim=(1.295*1e7,1.345*1e7))
                    fig.show()

                if self.as_lazyframe:
                    for split in splits:
                        split_ds = datasets[split]
                        for d, ds in enumerate(split_ds):
                            item_id = f"SPLIT{d}"

                            temp_path = os.path.join(
                                temp_dir,
                                os.path.basename(self.normalized_data_path).replace(
                                    ".parquet", f"_{split}_{item_id}.parquet"
                                ),
                            )

                            if reload or not os.path.exists(temp_path):
                                if verbose:
                                    logging.info(
                                        f"Transforming {split} dataset {item_id} into polars form."
                                    )

                                ds_len = ds.select(pl.len()).collect().item()
                                start_time = pd.Period(
                                    ds.select(pl.col("time").first()).collect().item(),
                                    freq=self.freq,
                                )
                                dt = pd.Timedelta(start_time.freq)
                                # When per_turbine_target=False, item_id is just "SPLIT{d}" with no TURBINE prefix,
                                # so the (?<=TURBINE)\w+(?=_SPLIT) regex returns None. Fall back to category 0
                                # in the all-turbine case (no single turbine to associate with).
                                _turbine_match = re.search(
                                    r"(?<=TURBINE)\w+(?=_SPLIT)", item_id
                                )
                                _feat_static_cat_value = (
                                    self.target_suffixes.index(_turbine_match.group(0))
                                    if _turbine_match is not None
                                    else 0
                                )
                                ds.select(
                                    [pl.col("time")]
                                    + self.feat_dynamic_real_cols
                                    + self.target_cols
                                ).select(
                                    item_id=pl.lit(item_id),
                                    time=pl.datetime_range(
                                        start=start_time.start_time,
                                        end=start_time.start_time + (dt * ds_len),
                                        interval=dt,
                                        time_unit="ns",
                                        closed="left",
                                    ),
                                    feat_static_cat=pl.lit([_feat_static_cat_value]),
                                    *[
                                        pl.col(col).alias(f"target_{i}")
                                        for i, col in enumerate(self.target_cols)
                                    ],
                                    *[
                                        pl.col(col).alias(f"feat_dynamic_real_{i}")
                                        for i, col in enumerate(
                                            self.feat_dynamic_real_cols
                                        )
                                    ],
                                ).sink_parquet(temp_path, maintain_order=True)
                                logging.info(
                                    f"Saved split {split} dataset {item_id} to {temp_path}"
                                )
                            elif verbose:
                                logging.info(
                                    f"Loading existing cached file for {split} dataset {item_id}."
                                )

                        # datasets[split] = pl.concat([
                        #     pl.scan_parquet(os.path.join(
                        #             temp_dir,
                        #             os.path.basename(self.data_path).replace(".parquet", f"_{split}_SPLIT{d}.parquet")
                        #         )) for d in range(len(split_ds))],
                        #     how="vertical")
                        logging.info(
                            f"Scanning and sorting parquet files for {split} split from {temp_dir} with pattern {os.path.basename(self.normalized_data_path).replace('.parquet', f'_{split}_SPLIT*.parquet')}."
                        )
                        datasets[split] = pl.scan_parquet(
                            os.path.join(
                                temp_dir,
                                os.path.basename(self.normalized_data_path).replace(
                                    ".parquet", f"_{split}_SPLIT*.parquet"
                                ),
                            ),
                            glob=True,
                        ).sort("item_id", maintain_order=True)

                        if False:
                            for cg, ds in enumerate(
                                [
                                    datasets[split].filter(pl.col("item_id") == item_id)
                                    for item_id in datasets[split]
                                    .select(pl.col("item_id").unique())
                                    .collect()
                                    .to_numpy()
                                    .flatten()
                                ]
                            ):
                                fig, ax = plt.subplots(2, 1, sharex=True)
                                for ax_idx, pfx in enumerate(self.target_prefixes):
                                    target_cols = [
                                        f"{pfx}_{sfx}"
                                        for sfx in ["wt005", "wt074", "wt075"]
                                    ]
                                    target_cols = [
                                        f"target_{self.target_cols.index(sfx)}"
                                        for sfx in target_cols
                                    ]
                                    ax[ax_idx].plot(
                                        ds.select("time")
                                        .collect()
                                        .to_numpy()[:-400, :],
                                        ds.select(target_cols)
                                        .collect()
                                        .to_numpy()[:-400, :],
                                        linestyle="-",
                                    )
                            fig.show()
                        # datasets[split] = pl.concat(transformed_datasets, how="vertical")

                else:
                    # convert dictionary of item_id: lazyframe datasets into list of dictionaries with numpy arrays for data
                    for split in splits:
                        transformed_datasets = []

                        # setattr(self, f"{split}_dataset", pl.collect_all(getattr(self, f"{split}_dataset")))
                        split_ds = datasets[split]
                        # item_ids = list(getattr(self, f"{split}_dataset").keys())
                        for d, ds in enumerate(split_ds):
                            item_id = f"SPLIT{d}"
                            if self.verbose:
                                logging.info(
                                    f"Transforming {split} dataset {item_id} into numpy form."
                                )
                            # ds = getattr(self, f"{split}_dataset")[item_id]
                            start_time = pd.Period(
                                ds.select(pl.col("time").first()).collect().item(),
                                freq=self.freq,
                            )
                            ds = (
                                ds.select(
                                    self.feat_dynamic_real_cols + self.target_cols
                                )
                                .collect()
                                .to_numpy()
                                .T
                            )
                            transformed_datasets.append(
                                {
                                    "target": ds[-len(self.target_cols) :, :],
                                    "item_id": item_id,
                                    "start": start_time,
                                    "feat_dynamic_real": ds[
                                        : -len(self.target_cols), :
                                    ],
                                }
                            )
                            # del getattr(self, f"{split}_dataset")[item_id]
                        datasets[split] = transformed_datasets
                if verbose:
                    logging.info(f"Finished splitting datasets for all turbine case.")

            # Log generated dataset sizes
            for split in splits:
                if self.as_lazyframe:
                    dataset_size = (
                        datasets[split].select(pl.len()).collect().item()
                        if datasets[split] is not None
                        else 0
                    )
                else:
                    dataset_size = (
                        sum(ds["target"].shape[1] for ds in datasets[split])
                        if datasets[split] is not None
                        else 0
                    )
                logging.info(
                    f"Rank 0: Generated {split} dataset size: {dataset_size} samples (context_length={self.context_length}, prediction_length={self.prediction_length})"
                )

            if save:
                logging.info(f"Rank 0: Saving generated splits.")
                for split in splits:
                    final_path = self.get_split_file_path(split)
                    # Use process ID in temp filename to avoid collisions between workers
                    temp_path = final_path + f".tmp.{os.getpid()}"
                    logging.info(f"Rank 0: Saving {split} data to {temp_path}")
                    try:
                        # Write to temp file
                        if self.as_lazyframe:
                            datasets[split].collect().write_parquet(temp_path)
                        else:
                            with open(temp_path, "wb") as fp:
                                pickle.dump(datasets[split], fp)

                        # Atomic rename - if file exists, this will overwrite it atomically
                        logging.info(
                            f"Rank 0: Atomically moving {temp_path} to {final_path}"
                        )
                        os.replace(
                            temp_path, final_path
                        )  # os.replace is atomic on POSIX
                        if self.as_lazyframe:
                            datasets[split] = pl.scan_parquet(
                                final_path
                            )  # reload as lazyframe
                        logging.info(
                            f"Rank 0: Successfully saved {split} data to {final_path}"
                        )
                    except Exception as e:
                        logging.error(f"Rank 0: Error saving {split} data: {e}")
                        # Clean up temp file if it still exists
                        if os.path.exists(temp_path):
                            try:
                                os.remove(temp_path)
                                logging.info(
                                    f"Rank 0: Cleaned up temp file {temp_path}"
                                )
                            except OSError as cleanup_error:
                                logging.error(
                                    f"Rank 0: Error removing temp file {temp_path}: {cleanup_error}"
                                )

                # NOTE: do NOT rmtree splits_temp — datasets[split] above is a lazy scan_parquet
                # over splits_temp/_SPLIT*.parquet (line ~556). Removing the dir invalidates the
                # LazyFrame; the validation harness then crashes with FileNotFoundError when it
                # later .collect()s. The temp files are small (per-split slices); leave them.
                # if self.as_lazyframe:
                #     shutil.rmtree(os.path.join(os.path.dirname(self.train_ready_data_path), "splits_temp"))

        # Only use barrier if PyTorch distributed is actually initialized
        # In tuning mode with independent workers, we don't need/want a barrier
        is_distributed = dist.is_available() and dist.is_initialized()
        if is_distributed:
            logging.info(
                f"Rank {rank}: Waiting at distributed barrier before loading splits."
            )
            dist.barrier()
            logging.info(f"Rank {rank}: Passed distributed barrier.")
        elif rank != 0:
            # For independent workers (e.g., tuning mode), use file-based synchronization
            # Wait for rank 0 to finish by checking if all split files exist

            max_wait_time = 300  # 5 minutes timeout
            check_interval = 2  # Check every 2 seconds
            start_time = time.time()

            while not all(
                os.path.exists(self.get_split_file_path(split)) for split in splits
            ):
                if time.time() - start_time > max_wait_time:
                    raise TimeoutError(
                        f"Rank {rank}: Timeout waiting for split files from rank 0"
                    )
                logging.info(
                    f"Rank {rank}: Waiting for rank 0 to generate split files..."
                )

                time.sleep(check_interval)

            logging.info(f"Rank {rank}: All split files detected, proceeding to load.")

        if rank != 0 or (not reload and split_files_exist):
            logging.info(f"Rank {rank}: Scanning dataset {self.train_ready_data_path}.")
            dataset = IterableLazyFrame(
                data_path=self.train_ready_data_path, dtype=self.dtype
            )
            logging.info(
                f"Rank {rank}: Finished scanning dataset {self.train_ready_data_path}."
            )

            # sets self.continuity_groups, self.target_cols, self.target_suffixes, self.feat_dynamic_real_cols, self.num_target_vars,
            #       self.num_feat_dynamic_real, self.num_feat_static_cat, self.num_feat_static_real, self.static_features, self.cardinality
            self.get_dataset_info(dataset)
            logging.info(f"Rank {rank}: Loading saved split datasets.")
            datasets = {}
            for split in splits:
                split_path = self.get_split_file_path(split)
                if not os.path.exists(split_path):
                    # This should ideally not happen after the barrier if rank 0 succeeded
                    logging.error(
                        f"Rank {rank}: ERROR - Split file {split_path} not found after barrier!"
                    )
                    raise FileNotFoundError(
                        f"Rank {rank}: Split file {split_path} not found after barrier!"
                    )
                try:
                    if self.as_lazyframe:
                        datasets[split] = pl.scan_parquet(split_path)
                    else:
                        with open(split_path, "rb") as fp:
                            datasets[split] = pickle.load(fp)
                    logging.info(
                        f"Rank {rank}: Successfully loaded {split} dataset from {split_path}."
                    )
                except EOFError as e:
                    logging.error(
                        f"Rank {rank}: EOFError loading {split_path}. File might be corrupted. Error: {e}"
                    )
                    raise
                except Exception as e:
                    logging.error(f"Rank {rank}: Error loading {split_path}: {e}")
                    raise

            # Log dataset sizes and validate loaded splits for compatibility with current context_length
            for split in splits:
                if self.as_lazyframe:
                    dataset_size = (
                        datasets[split].select(pl.len()).collect().item()
                        if datasets[split] is not None
                        else 0
                    )
                else:
                    dataset_size = (
                        sum(ds["target"].shape[1] for ds in datasets[split])
                        if datasets[split] is not None
                        else 0
                    )
                logging.info(
                    f"Rank {rank}: Loaded {split} dataset size: {dataset_size} samples"
                )

            # Filter out items that are too short for the current context_length + prediction_length
            if self.as_lazyframe:
                min_required = self.context_length + self.prediction_length
                for split in splits:
                    if datasets[split] is not None:
                        # Get item lengths and filter to keep only valid items
                        item_lengths = (
                            datasets[split]
                            .group_by("item_id")
                            .agg(pl.len().alias("_item_len"))
                            .collect()
                        )
                        valid_items = item_lengths.filter(
                            pl.col("_item_len") >= min_required
                        ).select("item_id")
                        short_items = item_lengths.filter(
                            pl.col("_item_len") < min_required
                        )

                        if short_items.height > 0:
                            original_count = item_lengths.height
                            datasets[split] = datasets[split].filter(
                                pl.col("item_id").is_in(valid_items["item_id"])
                            )
                            logging.warning(
                                f"Rank {rank}: Filtered out {short_items.height}/{original_count} items from {split} dataset "
                                f"that were shorter than min_required={min_required} (context_length={self.context_length} + "
                                f"prediction_length={self.prediction_length})"
                            )

        self.datasets = datasets

        if rank != 0 or (not reload and split_files_exist):
            self._validate_loaded_splits(splits, rank)

    def get_df_by_turbine(self, dataset, turbine_id):
        return dataset.select(
            pl.col("time"),
            *[
                col
                for col in (self.feat_dynamic_real_cols + self.target_cols)
                if col.endswith(f"_{turbine_id}")
            ],
        ).rename(
            mapping={
                **{
                    f"{tgt_col}_{turbine_id}": tgt_col
                    for tgt_col in self.target_prefixes
                },
                **{
                    f"{feat_col}_{turbine_id}": feat_col
                    for feat_col in self.feat_dynamic_real_prefixes
                },
            }
        )

    def split_datasets_by_turbine(self, datasets):

        return [
            ds.select(
                [pl.col("time")]
                + [
                    pl.col(f"{tgt_col}_{turbine_id}")
                    for tgt_col in self.target_prefixes
                ]
                + [
                    pl.col(f"{feat_col}_{turbine_id}")
                    for feat_col in self.feat_dynamic_real_prefixes
                ]
            )
            for ds in datasets
            for turbine_id in self.target_suffixes
        ]

    def split_dataset(self, dataset, splits):
        train_datasets = []
        test_datasets = []
        val_datasets = []

        # TODO total past length may also have to supercede context_len + max(self.lags_seq)
        for cg, ds in enumerate(dataset):
            logging.info(f"Splitting {cg}th dataset of {len(dataset)}.")
            # TODO in this case should just add to training data anyway?
            if (
                round(
                    min(self.train_split, self.val_split, self.test_split)
                    * self.rows_per_split[cg]
                    * self.n_splits
                )
                < self.context_length + self.prediction_length
            ):
                logging.info(
                    f"Can't split dataset corresponding to continuity group {cg} into training, validation, testing, the full dataset only has data points {round(self.rows_per_split[cg] * self.n_splits)}"
                )

                if (
                    "train" in splits
                    and self.train_split * self.rows_per_split[cg] * self.n_splits
                    >= self.context_length + self.prediction_length
                ):
                    logging.info(
                        f"Adding dataset corresponding to continuity group {cg} to training data, since it can't be split"
                    )
                    train_datasets += [ds]

                continue

            # splitting each continuity group into subsections, a training/val/test dataset will then be generated from each subsection.
            # We do this to get a coherent mix of training, val, test data that is more independent of trends over time
            datasets = []
            for split_idx in range(self.n_splits):
                slc = slice(
                    split_idx * self.rows_per_split[cg],
                    (split_idx + 1) * self.rows_per_split[cg],
                )
                # check that each split is at least context_len + target_len long, otherwise don't split it
                # if slc.stop - slc.start >= self.context_length + self.prediction_length:

                if (
                    round(
                        min(self.train_split, self.val_split, self.test_split)
                        * (slc.stop - slc.start)
                    )
                    >= self.context_length + self.prediction_length
                ):
                    datasets.append(
                        ds.select(pl.exclude("continuity_group")).slice(
                            slc.start, slc.stop - slc.start
                        )
                    )
                    start_time = (
                        datasets[-1].select(pl.col("time").first()).collect().item()
                    )
                    end_time = (
                        datasets[-1].select(pl.col("time").last()).collect().item()
                    )
                    duration = end_time - start_time
                    logging.info(
                        f"full dataset cg {cg} split {split_idx}, start time = {start_time}, end time = {end_time}, duration = {duration}"
                    )
                else:
                    logging.info(
                        f"Can't split dataset {cg} into {self.n_splits} , not enough data points, returning whole."
                    )

                    self.rows_per_split[cg] *= self.n_splits
                    datasets.append(ds.select(pl.exclude("continuity_group")))
                    break

            train_offset = round(self.train_split * self.rows_per_split[cg])
            val_offset = round(self.val_split * self.rows_per_split[cg])
            test_offset = round(self.test_split * self.rows_per_split[cg])

            if "train" in splits:
                logging.info(f"Creating {cg}th of {len(dataset)} train_datasets list.")
                train_datasets += [
                    ds.slice(0, train_offset).with_columns(continuity_group=pl.lit(d))
                    for d, ds in enumerate(datasets)
                ]

            if "val" in splits:
                logging.info(f"Creating {cg}th of {len(dataset)} val_datasets list.")
                val_datasets += [
                    ds.slice(train_offset, val_offset).with_columns(
                        continuity_group=pl.lit(d)
                    )
                    for d, ds in enumerate(datasets)
                ]  # val_offset is the length of validation data

            if "test" in splits:
                logging.info(f"Creating {cg}th of {len(dataset)} test_datasets list.")
                test_datasets += [
                    ds.slice(train_offset + val_offset, test_offset).with_columns(
                        continuity_group=pl.lit(d)
                    )
                    for d, ds in enumerate(datasets)
                ]  # test_offset is the length of test data

        if self.verbose:
            logging.info("Returning train/val/test datasets.")

        # return pl.concat(train_datasets, how="vertical"), pl.concat(val_datasets, how="vertical"), pl.concat(test_datasets, how="vertical")
        datasets = {}
        if "train" in splits:
            datasets["train"] = train_datasets
        if "val" in splits:
            datasets["val"] = val_datasets
        if "test" in splits:
            datasets["test"] = test_datasets

        return datasets

    def highlight_entry(self, entry, color, ax, vlines=None):
        start = entry["start"].to_timestamp()
        # end = entry["start"] + (entry["target"].shape[1] * entry["start"].freq.delta)
        end = (entry["start"] + entry["target"].shape[1]).to_timestamp()
        # print(f"start time = {start}, end time = {end}")
        if vlines is not None:
            ax.axvline(x=start, ymin=vlines[0], ymax=vlines[1], color=color)
            ax.axvline(
                x=end, ymin=vlines[0], ymax=vlines[1], color=color, linestyle="--"
            )
        else:
            ax.axvspan(start, end, facecolor=color, alpha=0.2)

    def plot_dataset_splitting(self):
        colors = colormaps["Pastel1"].colors
        fig, axs = plt.subplots(self.num_target_vars, 1, sharex=True)
        for d, ds in enumerate(
            [self.train_dataset, self.val_dataset, self.test_dataset]
        ):
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
                        periods=entry[FieldName.TARGET]
                        .select(pl.len())
                        .collect()
                        .item(),
                        freq=entry[FieldName.START].freq,
                    ),
                ).reset_index(names="time")
                df["time"] = df["time"].dt.to_timestamp()
                df = df.rename(
                    columns={
                        col: f"{tgt}_{entry['item_id']}"
                        for col, tgt in zip(df.columns[1:], self.target_cols)
                    }
                )

                for a in range(self.num_target_vars):
                    sns.lineplot(data=df, ax=axs[a], x="time", y=df.columns[a + 1])
                    self.highlight_entry(entry, colors[d], ax=axs[a], vlines=(0.0, 0.5))

        # for t, (test_input, test_label) in enumerate(self.test_dataset):
        #     for a in range(self.num_target_vars):
        #         # self.highlight_entry(test_input, colors[2], axs[a])
        #         self.highlight_entry(test_label, colors[3], axs[a])

        # plt.legend(["sub dataset", "test input", "test label"], loc="upper left")

        fig.show()
