from dataclasses import dataclass
from collections import defaultdict
from typing import List, Iterable, Type
from itertools import chain
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from gluonts.dataset.split import split, slice_data_entry
from gluonts.dataset.pandas import PolarsDataset, PandasDataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.common import TrainDatasets, MetaData, BasicFeatureInfo, CategoricalFeatureInfo, ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.util import to_pandas
# from gluonts.dataset.common import FileDataset
# from gluonts.dataset.jsonl import JsonLinesWriter, JsonLinesFile
from gluonts.dataset import Dataset

import polars as pl
import polars.selectors as cs
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colormaps
import seaborn as sns

# from torch.utils.data import IterableDataset, DataLoader, get_worker_info

# matplotlib.use('TkAgg')

# TODO where does the line-by-line parsing of target for each time step happen and can I stream this? eg with IterableDataset
# class LazyFrameStreamingDataset(IterableDataset):
@dataclass
class LazyFrameStreamingDataset(Dataset):
    """_summary_
    # LazyFrameStreamingDataset should iterate over this lazyframe, selecting subsets based on the indices in DataModule using LazyFrame.slice

    Args:
        IterableDataset (_type_): _description_
    """
    datasets: Iterable[pl.DataFrame]
    target_cols: List[str]
    feat_dynamic_real_cols: List[str] 
    freq: str
    static_features: pd.DataFrame
    dtype: Type = np.float32

    def __post_init__(self):
        self.lengths = [ds.select(pl.len()).collect().item() for ds in self.datasets]
        self.total_length = sum(self.lengths)
        self._static_cats = (
                    self.static_features.select_dtypes("category")
                    .apply(lambda col: col.cat.codes)
                    .astype(self.dtype).T
        ).values
        self.start_times = [pd.Period(ds.select(pl.col("time").first()).collect().item(), freq=self.freq) for ds in self.datasets]
        available_cols = [ds.collect_schema().names() for ds in self.datasets]
        self.target_cols = [set(cols).intersection(self.target_cols) for cols in available_cols]
        self.feat_dynamic_real_cols = [set(cols).intersection(self.feat_dynamic_real_cols) for cols in available_cols]

    def __iter__(self):
        for d, ds in enumerate(self.datasets):
            entry = {
                FieldName.TARGET: ds.select(self.target_cols[d]).collect().to_numpy(),
                FieldName.FEAT_DYNAMIC_REAL: ds.select(self.feat_dynamic_real_cols[d]).collect().to_numpy(),
                FieldName.START: self.start_times[d]
            }
            if len(self.static_features.index):
                entry[FieldName.FEAT_STATIC_CAT] = self._static_cats[:, d] 
            yield entry
    
    def __len__(self):
        return self.total_length


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

    def __post_init__(self):
        # TODO make this work for a large dataset
        dataset = pl.scan_parquet(self.data_path).with_columns(time=pl.col("time").dt.round(self.freq)).collect(streaming=True).group_by("time").agg(cs.numeric().mean()).sort(["continuity_group", "time"]).lazy()

        # fetch a subset of continuity groups and turbine data
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
        
        if self.target_suffixes is None:
            self.target_cols = dataset.select(*[cs.starts_with(pfx) for pfx in self.target_prefixes]).collect_schema().names()
            self.target_suffixes = sorted(list(set(col.split("_")[-1] for col in self.target_cols)))
        else:
            # float64_cols = list(dataset.select_dtypes(include="float64"))
            # dataset[float64_cols] = dataset[float64_cols].astype("float32")
            # dataset.filter(pl.col("continuity_group") == 0).to_pandas().to_csv("/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data/sample_data.csv", index=False) 
            self.target_cols = [col for col in dataset.collect_schema().names() if any(prefix in col for prefix in self.target_prefixes)]
        
        self.feat_dynamic_real_cols = [col for col in dataset.collect_schema().names() if any(prefix in col for prefix in self.feat_dynamic_real_prefixes)]
        self.num_feat_dynamic_real = len(self.feat_dynamic_real_cols)
        
        # univariate=ListDataset of multiple dictionaires each corresponding to measurements from a single turbine, to implicitly capture correlations
        # or multivariate=multivariate dictionary for all measurements, to explicity capture all correlations
        # or debug= to use electricity dataset
        if self.per_turbine_target:
            self.num_target_vars = len(self.target_prefixes)
            self.rows_per_split = [int(n_rows / self.n_splits) 
                                    for turbine_id in self.target_suffixes 
                                        for n_rows in dataset.select("continuity_group").collect().to_series().value_counts().sort("continuity_group").select("count").to_numpy().flatten()] # each element corresponds to each continuity group
            self.train_dataset, self.val_dataset, self.test_dataset = self.split_dataset([dataset.filter(pl.col("continuity_group") == cg) for cg in self.continuity_groups]) 
            

            # self.train_dataset = self.split_datasets_by_turbine(self.train_dataset)
            # self.val_dataset = self.split_datasets_by_turbine(self.val_dataset)
            # self.test_dataset = self.split_datasets_by_turbine(self.test_dataset)

            # self.static_features = pd.DataFrame(
            # {
            #     "turbine_id": pd.Categorical(turbine_id for turbine_id in self.target_suffixes for cg in self.continuity_groups),
            #     "continuity_group": pd.Categorical(int(cg) for turbine_id in self.target_suffixes for cg in self.continuity_groups)
            # },
            # index=static_index
            # )
            
            static_index = [f"T{turbine_id}" for turbine_id in self.target_suffixes for cg in self.continuity_groups]
            self.static_features = pd.DataFrame(
            {
                "turbine_id": pd.Categorical(turbine_id for turbine_id in self.target_suffixes for cg in self.continuity_groups)
            },
            index=static_index
            )
            
            self.train_dataset = {f"T{turbine_id}": self.get_df_by_turbine(self.train_dataset, turbine_id) for turbine_id in self.target_suffixes}
            self.val_dataset = {f"T{turbine_id}": self.get_df_by_turbine(self.train_dataset, turbine_id) for turbine_id in self.target_suffixes}
            self.test_dataset = {f"T{turbine_id}": self.get_df_by_turbine(self.train_dataset, turbine_id) for turbine_id in self.target_suffixes}

            self.train_dataset = PolarsDataset(self.train_dataset, 
                            target=self.target_prefixes, timestamp="time", freq=self.freq, 
                            feat_dynamic_real=self.feat_dynamic_real_prefixes, static_features=self.static_features, assume_sorted=True)
            self.val_dataset = PolarsDataset(self.val_dataset, 
                            target=self.target_prefixes, timestamp="time", freq=self.freq, 
                            feat_dynamic_real=self.feat_dynamic_real_prefixes, static_features=self.static_features, assume_sorted=True)
            self.test_dataset = PolarsDataset(self.test_dataset, 
                            target=self.target_prefixes, timestamp="time", freq=self.freq, 
                            feat_dynamic_real=self.feat_dynamic_real_prefixes, static_features=self.static_features, assume_sorted=True)

            self.num_feat_static_cat = 1
            self.num_feat_static_real = 0

            # self.train_dataset = LazyFrameStreamingDataset(train_datasets, target_cols=self.target_cols, feat_dynamic_real_cols=self.feat_dynamic_real_cols, 
            #                                             freq=self.freq, static_features=self.static_features)
            # self.val_dataset = LazyFrameStreamingDataset(val_datasets, target_cols=self.target_cols, feat_dynamic_real_cols=self.feat_dynamic_real_cols, 
            #                                             freq=self.freq, static_features=self.static_features)
            # self.test_dataset = LazyFrameStreamingDataset(test_datasets, target_cols=self.target_cols, feat_dynamic_real_cols=self.feat_dynamic_real_cols, 
            #                                             freq=self.freq, static_features=self.static_features)

        else:
            self.num_target_vars = len(self.target_cols)
            self.rows_per_split = [int(n_rows / self.n_splits) for n_rows in dataset.select("continuity_group").collect().to_series().value_counts().sort("continuity_group").select("count").to_numpy().flatten()] # each element corresponds to each continuity group
            
            # static_index = [f"{tgt_col}_{int(cg)}" for tgt_col in self.target_cols for cg in self.continuity_groups]
            # self.static_features = pd.DataFrame({
            #     "turbine_id": pd.Categorical(col.split("_")[-1] for col in self.target_cols for cg in self.continuity_groups),
            #     "output_category": pd.Categorical("_".join(col.split("_")[:-1]) for col in self.target_cols for cg in self.continuity_groups),
            #     "continuity_group": pd.Categorical(int(cg) for col in self.target_cols for cg in self.continuity_groups)
            # }, index=static_index)
            
            self.static_features = pd.DataFrame()
            
            self.num_feat_static_cat = 0
            self.num_feat_static_real = 0

            train_grouper = MultivariateGrouper(
                max_target_dim=self.num_target_vars,
                split_on="continuity_group" if len(self.continuity_groups) > 1 else None
            )
            self.full_dataset = PolarsDataset(
                    {f"{tgt_col}_{int(cg)}": dataset.filter(pl.col("continuity_group") == cg)\
                                                    .select([pl.col("time")] + self.feat_dynamic_real_cols + [tgt_col])\
                                                    .rename(mapping={tgt_col: "target"}) for tgt_col in self.target_cols for cg in self.continuity_groups}, 
                                    timestamp="time", freq=self.freq, feat_dynamic_real=self.feat_dynamic_real_cols, static_features=self.static_features, assume_sorted=True)
            self.full_dataset = train_grouper(self.full_dataset)

            self.train_dataset, self.val_dataset, self.test_dataset = self.split_dataset([dataset.filter(pl.col("continuity_group") == cg) for cg in self.continuity_groups])

            # self.train_dataset = LazyFrameStreamingDataset(train_datasets, target_cols=self.target_cols, feat_dynamic_real_cols=self.feat_dynamic_real_cols, 
            #                                             freq=self.freq, static_features=self.static_features)
            # self.val_dataset = LazyFrameStreamingDataset(val_datasets, target_cols=self.target_cols, feat_dynamic_real_cols=self.feat_dynamic_real_cols, 
            #                                             freq=self.freq, static_features=self.static_features)
            # self.test_dataset = LazyFrameStreamingDataset(test_datasets, target_cols=self.target_cols, feat_dynamic_real_cols=self.feat_dynamic_real_cols, 
            #                                             freq=self.freq, static_features=self.static_features)

        # self.full_dataset.sink_parquet(self.data_path.replace(".parquet", "_final.parquet"))
        # self.full_dataset.sink_ipc(self.data_path.replace(".parquet", "_final.arrow"))

        self.train_dataset = ListDataset(self.train_dataset, self.freq, one_dim_target=False)
        self.val_dataset = ListDataset(self.val_dataset, self.freq, one_dim_target=False)
        self.test_dataset = ListDataset(self.test_dataset, self.freq, one_dim_target=False)

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

        for cg, ds in enumerate(dataset):
            # TODO in this case should just add to training data anyway? 
            if round(min(self.train_split, self.val_split, self.test_split) * self.rows_per_split[cg] * self.n_splits) < self.context_length + self.prediction_length:
                logging.info(f"Can't split dataset {cg} into training, validation, testing, not enough data points.")
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

                    datasets.append(ds.slice(slc.start, slc.stop - slc.start).select(pl.exclude("continuity_group")))
                    start_time = datasets[-1].select(pl.col("time").first()).collect().item()
                    end_time = datasets[-1].select(pl.col("time").last()).collect().item()
                    duration = end_time - start_time
                    logging.info(f"full dataset cg {cg} split {split_idx}, start time = {start_time}, end time = {end_time}, duration = {duration}")
                else:
                    logging.info(f"Can't split dataset {cg} into {self.n_splits} , not enough data points, returning whole.")
                    # split_dataset = [ds]
                    self.rows_per_split[cg] *= self.n_splits
                    datasets.append(ds)
                    break
            
            train_offset = round(self.train_split * self.rows_per_split[cg])
            val_offset = round(self.val_split * self.rows_per_split[cg])
            test_offset = round(self.test_split * self.rows_per_split[cg])

            # TODO shouldn't test data include history, and just the labels be unseen by training data?
            train_datasets += [ds.slice(0, train_offset) for ds in datasets]
            val_datasets += [ds.slice(train_offset, val_offset) for ds in datasets]
            test_datasets += [ds.slice(train_offset + val_offset, test_offset) for ds in datasets]
            # train_dataset = [slice_data_entry(ds, slice_=slice(0, train_offset)) for ds in iter(split_dataset)]
            # val_dataset = [slice_data_entry(ds, slice_=slice(train_offset, train_offset + val_offset)) for ds in iter(split_dataset)]
            # test_dataset = [slice_data_entry(ds, slice_=slice(train_offset + val_offset, train_offset + val_offset + test_offset)) for ds in iter(split_dataset)]

            # val_dataset = [slice_data_entry(ds, slice_=slice(train_offset - self.context_length, train_offset + val_offset)) for ds in iter(split_dataset)]
            # test_dataset = [slice_data_entry(ds, slice_=slice(train_offset + val_offset - self.context_length, train_offset + val_offset + test_offset)) for ds in iter(split_dataset)]

            # for t, train_entry in enumerate(iter(train_dataset)):
            #     print(f"training dataset cg {cg}, split {t} start time = {train_entry['start']}, end time = {train_entry['start'] + train_entry['target'].shape[1]}, duration = {train_entry['target'].shape[1] * pd.Timedelta(train_entry['start'].freq)}\n")

            # for v, val_entry in enumerate(iter(val_dataset)):
            #     print(f"validation dataset cg {cg}, split {v} start time = {val_entry['start']}, end time = {val_entry['start'] + val_entry['target'].shape[1]}, duration = {val_entry['target'].shape[1] * pd.Timedelta(val_entry['start'].freq)}\n")

            # for t, test_entry in enumerate(iter(test_dataset)):
            #     print(f"test dataset cg {cg}, split {t} start time = {test_entry['start']}, end time = {test_entry['start'] + test_entry['target'].shape[1]}, duration = {test_entry['target'].shape[1] * pd.Timedelta(test_entry['start'].freq)}\n")
             
            # n_test_windows = int((self.test_split * self.rows_per_split[cg]) / self.prediction_length)
            # test_dataset = test_gen.generate_instances(prediction_length=self.prediction_length, windows=n_test_windows)

        # inferred_freq = sub_dataset.loc[sub_dataset["continuity_group"] == 0].index.inferred_freq  

        # self.train_dataset = ListDataset(chain(*train_datasets), self.freq, one_dim_target=False)
        # self.val_dataset = ListDataset(chain(*val_datasets), self.freq, one_dim_target=False)
        # self.test_dataset = ListDataset(chain(*test_datasets), self.freq, one_dim_target=False)
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
        for o, full_entry in enumerate(self.full_dataset):
            dataset = to_pandas(full_entry, is_multivariate=True).reset_index(names="time")
            dataset["time"] = dataset["time"].dt.to_timestamp()

            if self.per_turbine_target:
                dataset = dataset.rename(columns={col: f"{tgt}_{full_entry['item_id']}_split{o}" for col, tgt in zip(dataset.columns[1:], self.target_prefixes)})
            else:
                dataset = dataset.rename(columns={col: f"{tgt}_split{o}" for col, tgt in zip(dataset.columns[1:], self.target_cols)})

            for a in range(self.num_target_vars):
                sns.lineplot(data=dataset, ax=axs[a], x="time", y=dataset.columns[a + 1])
        
        for train_entry in self.train_dataset:
            for a in range(self.num_target_vars):
                self.highlight_entry(train_entry, colors[0], ax=axs[a], vlines=(0.0, 0.5))
        
        # axs[0].legend(["sub dataset", "training dataset"], loc="upper left")

        for val_entry in self.val_dataset:
            for a in range(self.num_target_vars):
                self.highlight_entry(val_entry, colors[1], axs[a], vlines=(0.5, 1.0))

        for test_entry in self.test_dataset:
            for a in range(self.num_target_vars):
                self.highlight_entry(test_entry, colors[2], axs[a])

        # for t, (test_input, test_label) in enumerate(self.test_dataset):
        #     for a in range(self.num_target_vars):
        #         # self.highlight_entry(test_input, colors[2], axs[a])
        #         self.highlight_entry(test_label, colors[3], axs[a])

            # plt.legend(["sub dataset", "test input", "test label"], loc="upper left")
        
        plt.show()