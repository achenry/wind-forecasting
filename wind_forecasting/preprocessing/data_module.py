from dataclasses import dataclass
from typing import List, Type
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from gluonts.dataset.split import split, slice_data_entry
from gluonts.dataset.pandas import PolarsDataset, PandasDataset, IterableLazyFrame
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.common import TrainDatasets, MetaData, BasicFeatureInfo, CategoricalFeatureInfo, ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.util import to_pandas
# from gluonts.dataset.common import FileDataset
from gluonts.dataset import Dataset

import polars as pl
import polars.selectors as cs
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colormaps
import seaborn as sns

from memory_profiler import profile


# from torch.utils.data import IterableDataset, DataLoader, get_worker_info
# from gluonts.dataset.jsonl import JsonLinesWriter, JsonLinesFile

# matplotlib.use('TkAgg')

# TODO where does the line-by-line parsing of target for each time step happen and can I stream this? eg with IterableDataset
# class LazyFrameStreamingDataset(IterableDataset):
# @dataclass
# class LazyFrameStreamingDataset(Dataset):
#     """_summary_
#     # LazyFrameStreamingDataset should iterate over this lazyframe, selecting subsets based on the indices in DataModule using LazyFrame.slice

#     Args:
#         IterableDataset (_type_): _description_
#     """
#     datasets: Iterable[pl.DataFrame]
#     target_cols: List[str]
#     feat_dynamic_real_cols: List[str] 
#     freq: str
#     static_features: pd.DataFrame
#     dtype: Type = np.float32

#     def __post_init__(self):
#         self.lengths = [ds.select(pl.len()).collect().item() for ds in self.datasets]
#         self.total_length = sum(self.lengths)
#         self._static_cats = (
#                     self.static_features.select_dtypes("category")
#                     .apply(lambda col: col.cat.codes)
#                     .astype(self.dtype).T
#         ).values
#         self.start_times = [pd.Period(ds.select(pl.col("time").first()).collect().item(), freq=self.freq) for ds in self.datasets]
#         available_cols = [ds.collect_schema().names() for ds in self.datasets]
#         self.target_cols = [set(cols).intersection(self.target_cols) for cols in available_cols]
#         self.feat_dynamic_real_cols = [set(cols).intersection(self.feat_dynamic_real_cols) for cols in available_cols]

#     def __iter__(self):
#         for d, ds in enumerate(self.datasets):
#             entry = {
#                 FieldName.TARGET: ds.select(self.target_cols[d]).collect().to_numpy(),
#                 FieldName.FEAT_DYNAMIC_REAL: ds.select(self.feat_dynamic_real_cols[d]).collect().to_numpy(),
#                 FieldName.START: self.start_times[d]
#             }
#             if len(self.static_features.index):
#                 entry[FieldName.FEAT_STATIC_CAT] = self._static_cats[:, d] 
#             yield entry
    
#     def __len__(self):
#         return self.total_length

# class IterableLazyFrameMetaClass(type):
#     def __instancecheck__(self, instance):
#         return isinstance(instance, (self.__class__, pl.LazyFrame))

#     def __subclasscheck__(self, subclass):
#         return issubclass(subclass, (self.__class__, pl.LazyFrame))


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
    verbose: bool = False
    
    def __post_init__(self):
        self.train_ready_data_path = self.data_path.replace(".parquet", "_train_ready.parquet")
     
    @profile
    def generate_datasets(self):
        
        dataset = IterableLazyFrame(data_path=self.data_path, dtype=self.dtype)\
                    .with_columns(time=pl.col("time").dt.round(self.freq))\
                    .group_by("time").agg(cs.numeric().mean())\
                    .sort(["continuity_group", "time"])
                    # .collect().write_parquet(self.train_ready_data_path, statistics=False)
        
        # dataset = IterableLazyFrame(data_path=self.train_ready_data_path, dtype=self.dtype) # data stored in RAM
        # gc.collect()
        
        # fetch a subset of continuity groups and turbine data
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
                dataset = dataset.select(pl.col("time"), *[cs.ends_with(sfx) for sfx in self.target_suffixes])
                dataset = dataset.with_columns(continuity_group=pl.lit(0))
        else:
            dataset = dataset.filter(pl.col("continuity_group").is_in(self.continuity_groups))\
                             .select(pl.col("time"), pl.col("continuity_group"), *[cs.ends_with(sfx) for sfx in self.target_suffixes])
        logging.info(f"Found continuity groups {self.continuity_groups}") 
        # dataset.target_cols = self.target_cols 
        
        logging.info(f"Writing resampled/sorted parquet to {self.train_ready_data_path}.") 
        dataset.collect().write_parquet(self.train_ready_data_path, statistics=False)
        logging.info(f"Saved resampled/sorted parquet to {self.train_ready_data_path}.")
        # dataset = IterableLazyFrame(data_path=train_ready_data_path)
        # univariate=ListDataset of multiple dictionaires each corresponding to measurements from a single turbine, to implicitly capture correlations
        # or multivariate=multivariate dictionary for all measurements, to explicity capture all correlations
        # or debug= to use electricity dataset
        
    def generate_splits(self):
        assert os.path.exists(self.train_ready_data_path), "Must run generate_datasets before generate_splits."
        
        logging.info(f"Scanning dataset {self.train_ready_data_path}.") 
        dataset = IterableLazyFrame(data_path=self.train_ready_data_path, dtype=self.dtype)
        logging.info(f"Finished scanning dataset {self.train_ready_data_path}.")
        
        # print(f"Number of nan/null vars = {dataset.select(pl.sum_horizontal((cs.numeric().is_null() | cs.numeric().is_nan()).sum())).collect().item()}") 
        
        logging.info("Getting continuity groups.") 
        if self.continuity_groups is None:
            if "continuity_group" in dataset.collect_schema().names():
                self.continuity_groups = dataset.select(pl.col("continuity_group").unique()).collect().to_numpy().flatten()
            else:
                self.continuity_groups = [0]
        logging.info(f"Found continuity groups {self.continuity_groups}")
         
        logging.info(f"Getting column names.") 
        if self.target_suffixes is None:
            self.target_cols = dataset.select(*[cs.starts_with(pfx) for pfx in self.target_prefixes]).collect_schema().names()
            self.target_suffixes = sorted(list(set(col.split("_")[-1] for col in self.target_cols)))
        else:
            # float64_cols = list(dataset.select_dtypes(include="float64"))
            # dataset[float64_cols] = dataset[float64_cols].astype("float32")
            # dataset.filter(pl.col("continuity_group") == 0).to_pandas().to_csv("/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data/sample_data.csv", index=False) 
            self.target_cols = [col for col in dataset.collect_schema().names() if any(prefix in col for prefix in self.target_prefixes)]
        self.feat_dynamic_real_cols = [col for col in dataset.collect_schema().names() if any(prefix in col for prefix in self.feat_dynamic_real_prefixes)]
        logging.info(f"Found column names target_cols={self.target_cols}, feat_dynamic_real_cols={self.feat_dynamic_real_cols}.") 
        
        if self.per_turbine_target:
            logging.info(f"Splitting datasets for per turbine case.") 
            
            self.num_target_vars = len(self.target_prefixes)
            self.num_feat_dynamic_real = int(len(self.feat_dynamic_real_cols) / len(self.target_suffixes))

            cg_counts = dataset.select("continuity_group").collect(streaming=True).to_series().value_counts().sort("continuity_group").select("count").to_numpy().flatten()
            self.rows_per_split = [
                int(n_rows / self.n_splits) 
                for turbine_id in self.target_suffixes 
                for n_rows in cg_counts] # each element corresponds to each continuity group
            del cg_counts
            
            self.train_dataset, self.val_dataset, self.test_dataset = \
                self.split_dataset([dataset.filter(pl.col("continuity_group") == cg) for cg in self.continuity_groups]) 
            
            static_index = [f"TURBINE{turbine_id}_SPLIT{split}" for turbine_id in self.target_suffixes for split in range(len(self.train_dataset))]
            self.static_features = pd.DataFrame(
                {
                    "turbine_id": pd.Categorical(turbine_id for turbine_id in self.target_suffixes for split in range(len(self.train_dataset)))
                },
                index=static_index
            )

            self.cardinality = list(self.static_features.select_dtypes("category").nunique().values)
            
            self.train_dataset = {f"TURBINE{turbine_id}_SPLIT{split}": 
                self.get_df_by_turbine(self.train_dataset[split], turbine_id) 
                for turbine_id in self.target_suffixes for split in range(len(self.train_dataset))}
            self.val_dataset = {f"TURBINE{turbine_id}_SPLIT{split}": 
                self.get_df_by_turbine(self.val_dataset[split], turbine_id) 
                for turbine_id in self.target_suffixes for split in range(len(self.val_dataset))}
            self.test_dataset = {f"TURBINE{turbine_id}_SPLIT{split}": 
                self.get_df_by_turbine(self.test_dataset[split], turbine_id) 
                for turbine_id in self.target_suffixes for split in range(len(self.test_dataset))}

            self.train_dataset = PolarsDataset(self.train_dataset, 
                            target=self.target_prefixes, timestamp="time", freq=self.freq, 
                            feat_dynamic_real=self.feat_dynamic_real_prefixes, static_features=self.static_features, 
                            assume_sorted=True, assume_resampled=True, unchecked=True)
            
            self.val_dataset = PolarsDataset(self.val_dataset, 
                            target=self.target_prefixes, timestamp="time", freq=self.freq, 
                            feat_dynamic_real=self.feat_dynamic_real_prefixes, static_features=self.static_features, 
                            assume_sorted=True, assume_resampled=True, unchecked=True)
            
            self.test_dataset = PolarsDataset(self.test_dataset, 
                            target=self.target_prefixes, timestamp="time", freq=self.freq, 
                            feat_dynamic_real=self.feat_dynamic_real_prefixes, static_features=self.static_features, 
                            assume_sorted=True, assume_resampled=True, unchecked=True)

            self.num_feat_static_cat = 1
            self.num_feat_static_real = 0
            self.target_cols = self.target_prefixes
            
            logging.info(f"Finished splitting datasets for per turbine case.") 

        else:
            logging.info(f"Splitting datasets for all turbine case.") 
            self.num_feat_dynamic_real = len(self.feat_dynamic_real_cols)
            self.num_target_vars = len(self.target_cols)
            cg_counts = dataset.select("continuity_group").collect().to_series().value_counts().sort("continuity_group").select("count").to_numpy().flatten()
            self.rows_per_split = [int(n_rows / self.n_splits) for n_rows in cg_counts] # each element corresponds to each continuity group
            del cg_counts
            
            self.train_dataset, self.val_dataset, self.test_dataset = \
                self.split_dataset([dataset.filter(pl.col("continuity_group") == cg) for cg in self.continuity_groups])

            self.static_features = pd.DataFrame()
            self.cardinality = None 
            self.num_feat_static_cat = 0
            self.num_feat_static_real = 0

            # train_grouper = MultivariateGrouper(
            #     max_target_dim=self.num_target_vars,
            #     split_on="continuity_group" if len(self.continuity_groups) > 1 else None
            # )

            self.train_dataset = {f"SPLIT{split}": 
                self.train_dataset[split].select([pl.col("time")] + self.feat_dynamic_real_cols + self.target_cols) 
                for split in range(len(self.train_dataset))}
            
            self.val_dataset = {f"SPLIT{split}": 
                self.val_dataset[split].select([pl.col("time")] + self.feat_dynamic_real_cols + self.target_cols) 
                for split in range(len(self.val_dataset))}
            
            self.test_dataset = {f"SPLIT{split}": self.test_dataset[split].select([pl.col("time")] + self.feat_dynamic_real_cols + self.target_cols) 
                                 for split in range(len(self.test_dataset))}

            self.train_dataset = PolarsDataset(
                self.train_dataset, 
                timestamp="time", freq=self.freq, 
                target=self.target_cols, feat_dynamic_real=self.feat_dynamic_real_cols, static_features=self.static_features, 
                assume_sorted=True, assume_resampled=True, unchecked=True
                )
            
            self.val_dataset = PolarsDataset(
                self.val_dataset, 
                timestamp="time", freq=self.freq, 
                target=self.target_cols, feat_dynamic_real=self.feat_dynamic_real_cols, static_features=self.static_features, 
                assume_sorted=True, assume_resampled=True, unchecked=True)
            
            self.test_dataset = PolarsDataset(
                self.test_dataset, 
                timestamp="time", freq=self.freq, 
                target=self.target_cols, feat_dynamic_real=self.feat_dynamic_real_cols, static_features=self.static_features, 
                assume_sorted=True, assume_resampled=True, unchecked=True)
            
            logging.info(f"Finished splitting datasets for all turbine case.") 
        
        # for split, datasets in [("train", self.train_dataset), ("val", self.val_dataset), ("test", self.test_dataset)]:
        #     for ds in iter(datasets):
        #         for key in ["target", "feat_dynamic_real"]:
        #             print(f"{split} {key} {ds['item_id']} dataset - num nan/nulls = {ds[key].select(pl.sum_horizontal((cs.numeric().is_null() | cs.numeric().is_nan()).sum())).collect().item()}")
         
        return None
        
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
            train_datasets.append(ds.slice(0, train_offset))
            val_datasets.append(ds.slice(train_offset, val_offset))
            test_datasets.append(ds.slice(train_offset + val_offset, test_offset))
            
            if self.verbose:
                for t, train_entry in enumerate(iter(train_datasets[-1])):
                    logging.info(f"training dataset cg {cg}, split {t} start time = {train_entry['start']}, end time = {train_entry['start'] + train_entry['target'].shape[1]}, duration = {train_entry['target'].shape[1] * pd.Timedelta(train_entry['start'].freq)}\n")

                for v, val_entry in enumerate(iter(val_datasets[-1])):
                    logging.info(f"validation dataset cg {cg}, split {v} start time = {val_entry['start']}, end time = {val_entry['start'] + val_entry['target'].shape[1]}, duration = {val_entry['target'].shape[1] * pd.Timedelta(val_entry['start'].freq)}\n")

                for t, test_entry in enumerate(iter(test_datasets[-1])):
                    logging.info(f"test dataset cg {cg}, split {t} start time = {test_entry['start']}, end time = {test_entry['start'] + test_entry['target'].shape[1]}, duration = {test_entry['target'].shape[1] * pd.Timedelta(test_entry['start'].freq)}\n")
            
            # n_test_windows = int((self.test_split * self.rows_per_split[cg]) / self.prediction_length)
            # test_dataset = test_gen.generate_instances(prediction_length=self.prediction_length, windows=n_test_windows)
            
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
                df = to_pandas(entry, is_multivariate=True).reset_index(names="time")
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