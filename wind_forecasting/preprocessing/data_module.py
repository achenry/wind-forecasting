from dataclasses import dataclass
from typing import List
from itertools import chain
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from gluonts.dataset.split import split, slice_data_entry
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.common import TrainDatasets, MetaData, BasicFeatureInfo, CategoricalFeatureInfo, ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.util import to_pandas

import polars as pl
import polars.selectors as cs
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colormaps
import seaborn as sns
# matplotlib.use('TkAgg')

@dataclass
class DataModule():
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
        # dataset = pl.read_parquet(DATA_PATH).to_pandas().groupby("continuity_group").resample(config["dataset"]["resample"], on="time").mean().reset_index(level=0, drop=True)
        # dataset = pl.read_parquet(self.data_path).group_by("continuity_group").group_by_dynamic("time", every=self.freq).agg(cs.numeric().mean()).sort_by("time").collect().to_pandas().set_index("time")
        # dataset.index.rename("timestamp")
        dataset = pl.read_parquet(self.data_path).with_columns(time=pl.col("time").dt.round(self.freq)).group_by("time").agg(cs.numeric().mean()).sort(["continuity_group", "time"])
        # .group_by_dynamic("time", every=self.freq).agg(cs.numeric().mean()).sort_by("time")

        # fetch a subset of continuity groups and turbine data
        if self.continuity_groups is None:
            if "continuity_group" in dataset.collect_schema().names():
                # continuity_groups = list(dataset["continuity_group"].value_counts().index[1:5])
                # self.continuity_groups = sorted(np.unique(dataset["continuity_group"].values.astype(int)))
                self.continuity_groups = dataset.select(pl.col("continuity_group").unique()).to_numpy().flatten()
                # sub_dataset = dataset.loc[dataset["continuity_group"].isin(continuity_groups),
                #         [col for col in dataset.columns if any(col.__contains__(tid) for tid in turbine_ids)] + ["continuity_group"]]
                # dataset = dataset[[col for col in dataset.columns if any(col.__contains__(tid) for tid in self.target_suffixes)] + ["continuity_group"]]
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
            self.target_suffixes = sorted(list(set(col.split("_")[-1] for col in dataset.select(*[cs.starts_with(pfx) for pfx in self.target_prefixes]).columns)))

        # float64_cols = list(dataset.select_dtypes(include="float64"))
        # dataset[float64_cols] = dataset[float64_cols].astype("float32")
        # dataset.filter(pl.col("continuity_group") == 0).to_pandas().to_csv("/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data/sample_data.csv", index=False) 
        self.target_cols = [col for col in dataset.columns if any(prefix in col for prefix in self.target_prefixes)]
        self.feat_dynamic_real_cols = [col for col in dataset.columns if any(prefix in col for prefix in self.feat_dynamic_real_prefixes)]
        self.num_feat_dynamic_real = len(self.feat_dynamic_real_cols)

        # univariate=ListDataset of multiple dictionaires each corresponding to measurements from a single turbine, to implicitly capture correlations
        # or multivariate=multivariate dictionary for all measurements, to explicity capture all correlations
        # or debug= to use electricity dataset
        if self.per_turbine_target:
            self.num_target_vars = len(self.target_prefixes)
            self.rows_per_split = [int(n_rows / self.n_splits) for turbine_id in self.target_suffixes for n_rows in dataset.select("continuity_group").to_series().value_counts().sort("continuity_group").select("count").to_numpy().flatten()] # each element corresponds to each continuity group

            self.static_features = pd.DataFrame(
            {
                "turbine_id": pd.Categorical(turbine_id for turbine_id in self.target_suffixes for cg in self.continuity_groups),
                "continuity_group": pd.Categorical(int(cg) for turbine_id in self.target_suffixes for cg in self.continuity_groups)
            },
            index=[f"{turbine_id}_cg{int(cg)}" for turbine_id in self.target_suffixes for cg in self.continuity_groups]
            )
            static_index = [f"{turbine_id}_cg{int(cg)}" for turbine_id in self.target_suffixes for cg in self.continuity_groups]
                
            self.static_features = pd.DataFrame(
                self.static_features,
                index=static_index
            )

            self.full_dataset = PandasDataset(
                    {f"{turbine_id}_cg{int(cg)}": dataset.filter(pl.col("continuity_group") == cg)\
                                                       .select(pl.col("time"), *[col for col in (self.feat_dynamic_real_cols + self.target_cols) if turbine_id in col])\
                                                       .to_pandas()\
                                                       .rename(columns={**{f"{tgt_col}_{turbine_id}": tgt_col for tgt_col in self.target_prefixes},
                                                                        **{f"{feat_col}_{turbine_id}": feat_col for feat_col in self.feat_dynamic_real_prefixes}})
                                                    for turbine_id in self.target_suffixes for cg in self.continuity_groups}, 
                                                target=self.target_prefixes, timestamp="time", freq=self.freq, feat_dynamic_real=self.feat_dynamic_real_prefixes, static_features=self.static_features, assume_sorted=True)
            
            self.num_feat_static_cat = 2
            self.num_feat_static_real = 0

        else:
            self.num_target_vars = len(self.target_cols)
            self.rows_per_split = [int(n_rows / self.n_splits) for n_rows in dataset.select("continuity_group").to_series().value_counts().sort("continuity_group").select("count").to_numpy().flatten()] # each element corresponds to each continuity group
        
            self.static_features = {
                "turbine_id": pd.Categorical(col.split("_")[-1] for col in self.target_cols for cg in self.continuity_groups),
                "output_category": pd.Categorical("_".join(col.split("_")[:-1]) for col in self.target_cols for cg in self.continuity_groups),
                "continuity_group": pd.Categorical(int(cg) for col in self.target_cols for cg in self.continuity_groups)
            }
            static_index = [f"{tgt_col}_{int(cg)}" for tgt_col in self.target_cols for cg in self.continuity_groups]
                
            self.static_features = pd.DataFrame(
                self.static_features,
                # index=target_cols
                index=static_index
            )
            
            self.num_feat_static_cat = 0 
            self.num_feat_static_real = 0
            
            # n_turbines = static_features["turbine_id"].dtype.categories.shape[0]
            # n_output_categories = static_features["output_category"].dtype.categories.shape[0]

            train_grouper = MultivariateGrouper(
                max_target_dim=self.num_target_vars,
                split_on="continuity_group" if len(self.continuity_groups) > 1 else None
            )

            self.full_dataset = train_grouper(PandasDataset(
                    {f"{tgt_col}_{int(cg)}": dataset.filter(pl.col("continuity_group") == cg).select([pl.col("time")] + self.feat_dynamic_real_cols + [tgt_col]).to_pandas()\
                                    .rename(columns={tgt_col: "target"}) for tgt_col in self.target_cols for cg in self.continuity_groups}, 
                                    timestamp="time", freq=self.freq, feat_dynamic_real=self.feat_dynamic_real_cols, static_features=self.static_features, assume_sorted=True))
        
        self.split_dataset(self.full_dataset)

    def split_dataset(self, dataset):
        train_datasets = []
        test_datasets = []
        val_datasets = []

        for cg, ds in enumerate(dataset):
            # TODO in this case should just add to training data anyway? 
            if round(min(self.train_split, self.val_split, self.test_split) * self.rows_per_split[cg] * self.n_splits) < self.context_length + self.prediction_length:
                logging.info(f"Can't split dataset {cg} into training, validation, testing, not enough data points.")
                continue 

            split_dataset = []
            for split_idx in range(self.n_splits):
                slc = slice(split_idx * self.rows_per_split[cg], (split_idx + 1) * self.rows_per_split[cg])
                # check that each split is at least context_len + target_len long, otherwise don't split it
                # if slc.stop - slc.start >= self.context_length + self.prediction_length:
                if round(min(self.train_split, self.val_split, self.test_split) * (slc.stop - slc.start)) >= self.context_length + self.prediction_length: 
                    split_dataset.append(slice_data_entry(ds, slice_=slc))
                    logging.info(f"full dataset cg {cg} split {split_idx} start time = {split_dataset[-1]['start']}, end time = {split_dataset[-1]['start'] + split_dataset[-1]['target'].shape[1]}, duration = {split_dataset[-1]['target'].shape[1] * pd.Timedelta(split_dataset[-1]['start'].freq)}")
                else:
                    logging.info(f"Can't split dataset {cg} into {self.n_splits} , not enough data points, returning whole.")
                    split_dataset = [ds]
                    self.rows_per_split[cg] *= self.n_splits
                    break
            
            train_offset = round(self.train_split * self.rows_per_split[cg])
            val_offset = round(self.val_split * self.rows_per_split[cg])
            test_offset = round(self.test_split * self.rows_per_split[cg])
            # train_val_offset = train_offset + val_offset
            # train_val_dataset, test_gen = split(split_dataset, offset=train_val_offset)

            # train_dataset = [slice_data_entry(ds, slice_=slice(0, train_offset)) for ds in iter(train_val_dataset)]
            # val_dataset = [slice_data_entry(ds, slice_=slice(train_offset, train_offset + val_offset)) for ds in iter(train_val_dataset)]

            # TODO shouldn't test data include history, and just the labels be unseen by training data?
            train_dataset = [slice_data_entry(ds, slice_=slice(0, train_offset)) for ds in iter(split_dataset)]
            # val_dataset = [slice_data_entry(ds, slice_=slice(train_offset - self.context_length, train_offset + val_offset)) for ds in iter(split_dataset)]
            val_dataset = [slice_data_entry(ds, slice_=slice(train_offset, train_offset + val_offset)) for ds in iter(split_dataset)]
            # test_dataset = [slice_data_entry(ds, slice_=slice(train_offset + val_offset - self.context_length, train_offset + val_offset + test_offset)) for ds in iter(split_dataset)]
            test_dataset = [slice_data_entry(ds, slice_=slice(train_offset + val_offset, train_offset + val_offset + test_offset)) for ds in iter(split_dataset)]

            for t, train_entry in enumerate(iter(train_dataset)):
                print(f"training dataset cg {cg}, split {t} start time = {train_entry['start']}, end time = {train_entry['start'] + train_entry['target'].shape[1]}, duration = {train_entry['target'].shape[1] * pd.Timedelta(train_entry['start'].freq)}\n")

            for v, val_entry in enumerate(iter(val_dataset)):
                print(f"validation dataset cg {cg}, split {v} start time = {val_entry['start']}, end time = {val_entry['start'] + val_entry['target'].shape[1]}, duration = {val_entry['target'].shape[1] * pd.Timedelta(val_entry['start'].freq)}\n")

            for t, test_entry in enumerate(iter(test_dataset)):
                print(f"test dataset cg {cg}, split {t} start time = {test_entry['start']}, end time = {test_entry['start'] + test_entry['target'].shape[1]}, duration = {test_entry['target'].shape[1] * pd.Timedelta(test_entry['start'].freq)}\n")
             
            # n_test_windows = int((self.test_split * self.rows_per_split[cg]) / self.prediction_length)
            # test_dataset = test_gen.generate_instances(prediction_length=self.prediction_length, windows=n_test_windows)

            train_datasets.append(train_dataset)
            test_datasets.append(test_dataset)
            val_datasets.append(val_dataset)

        # inferred_freq = sub_dataset.loc[sub_dataset["continuity_group"] == 0].index.inferred_freq  
        self.train_dataset = ListDataset(chain(*train_datasets), self.freq, one_dim_target=False)
        self.val_dataset = ListDataset(chain(*val_datasets), self.freq, one_dim_target=False)
        self.test_dataset = ListDataset(chain(*test_datasets), self.freq, one_dim_target=False)
        # self.test_dataset = list(chain(*test_datasets))

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