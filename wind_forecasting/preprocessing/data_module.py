from dataclasses import dataclass
from typing import List
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from gluonts.dataset.split import split, slice_data_entry
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.common import TrainDatasets, MetaData, BasicFeatureInfo, CategoricalFeatureInfo, ListDataset
from gluonts.dataset.field_names import FieldName

import polars as pl
import numpy as np

@dataclass
class DataModule():
    data_path: str
    n_splits: int
    train_split: float
    val_split: float
    test_split: float
    prediction_length: int
    context_length: int
    target_prefixes: List[str]
    target_suffixes: List[str] # ie turbine ids after the last underscore eg wt001 in  ws_horz_wt001
    feat_dynamic_real_prefixes: List[str]
    freq: str
    one_dim_target: str # could be univariate or multivariate

    def __post_init__(self):
        # df = pl.read_parquet(DATA_PATH).to_pandas().groupby("continuity_group").resample(config["dataset"]["resample"], on="time").mean().reset_index(level=0, drop=True)
        dataset = pl.read_parquet(self.data_path).group_by("continuity_group").group_by_dynamic("time", every=self.freq).agg(cs.numeric().mean()).sort_by("time").collect().to_pandas().set_index("time")
        dataset.index.rename("timestamp")

        # fetch a subset of continuity groups and turbine data
        if "continuity_group" in df.columns:
            # continuity_groups = list(df["continuity_group"].value_counts().index[1:5])
            self.continuity_groups = sorted(np.unique(df["continuity_group"].values.astype(int)))
            # sub_df = df.loc[df["continuity_group"].isin(continuity_groups),
            #         [col for col in df.columns if any(col.__contains__(tid) for tid in turbine_ids)] + ["continuity_group"]]
            dataset = dataset[[col for col in df.columns if any(col.__contains__(tid) for tid in self.target_suffixes)] + ["continuity_group"]]
        else:
            self.continuity_groups = [0]
            dataset = dataset[[col for col in dataset.columns if any(col.__contains__(tid) for tid in self.target_suffixes)]]
            dataset.loc[:, "continuity_group"] = 0
        
        float64_cols = list(df.select_dtypes(include="float64"))
        dataset[float64_cols] = dataset[float64_cols].astype("float32")
        self.rows_per_split = [int(n_rows / self.n_splits) for n_rows in df.value_counts("continuity_group")[self.continuity_groups]] # each element corresponds to each continuity group
        
        self.target_cols = [col for col in df.columns if any(prefix in col for prefix in self.target_prefixes)]
        self.num_target_vars = len(self.target_cols)
        self.feat_dynamic_real_cols = [col for col in df.columns if any(prefix in col for prefix in self.feat_dynamic_real_prefixes)]
        self.num_feat_dynamic_real = len(self.feat_dynamic_real_cols)

        # univariate=ListDataset of multiple dictionaires each corresponding to measurements from a single turbine, to implicitly capture correlations
        # or multivariate=multivariate dictionary for all measurements, to explicity capture all correlations
        # or debug= to use electricity dataset
        if self.one_dim_target:
            static_features = pd.DataFrame(
            {
                "turbine_id": pd.Categorical(turbine_id for turbine_id in turbine_ids for cg in continuity_groups),
                "continuity_group": pd.Categorical(int(cg) for turbine_id in turbine_ids for cg in continuity_groups)
            },
            index=[f"{turbine_id}_{int(cg)}" for turbine_id in turbine_ids for cg in continuity_groups]
            )
            static_index = [f"{turbine_id}_{int(cg)}" for turbine_id in turbine_ids for cg in continuity_groups]
                
            static_features = pd.DataFrame(
                static_features,
                index=static_index
            )

            dataset = PandasDataset(
                    {f"{turbine_id}_{int(cg)}": dataset.loc[dataset["continuity_group"] == cg, [col for col in (self.feat_dynamic_real_cols + self.target_cols) if turbine_id in col]]\
                                                    .rename(columns={**{f"{tgt_col}_{turbine_id}": tgt_col for tgt_col in self.target_prefixes},
                                                                    **{f"{feat_col}_{turbine_id}": feat_col for feat_col in self.feat_dynamic_real_prefixes}})
                                                    for turbine_id in self.target_suffixes for cg in self.continuity_groups}, 
                                                target=self.target_prefixes, feat_dynamic_real=self.feat_dynamic_real_prefixes, static_features=self.static_features, assume_sorted=True)
            
            self.num_feat_static_cat = 2
            self.num_feat_static_real = 0

        else:
            static_features = {
                "turbine_id": pd.Categorical(col.split("_")[-1] for col in self.target_cols for cg in self.continuity_groups),
                "output_category": pd.Categorical("_".join(col.split("_")[:-1]) for col in self.target_cols for cg in self.continuity_groups),
                "continuity_group": pd.Categorical(int(cg) for col in self.target_cols for cg in self.continuity_groups)
            }
            static_index = [f"{tgt_col}_{int(cg)}" for tgt_col in self.target_cols for cg in self.continuity_groups]
                
            static_features = pd.DataFrame(
                static_features,
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

            dataset = train_grouper(PandasDataset(
                    {f"{tgt_col}_{int(cg)}": dataset.loc[dataset["continuity_group"] == cg, self.feat_dynamic_real_cols + [tgt_col]]\
                                    .rename(columns={tgt_col: "target"}) for tgt_col in self.target_cols for cg in self.continuity_groups}, 
                                    feat_dynamic_real=self.feat_dynamic_real_cols, static_features=self.static_features, assume_sorted=True))
            
        
        self.dataset = self.split_dataset(dataset)

    def split_dataset(self, dataset):
        
        for cg, ds in enumerate(dataset):
            split_dataset = []
            for split_idx in range(self.n_splits):
                slc = slice(split_idx * self.rows_per_split[cg], (split_idx + 1) * self.rows_per_split[cg])
                # check that each split is at least context_len + target_len long, otherwise don't split it
                if slc.stop - slc.start >= self.context_length + self.prediction_length:
                    split_dataset.append(slice_data_entry(ds, slice_=slc, prediction_length=self.prediction_length))
                else:
                    logging.info(f"Can't split dataset {cg} into {self.n_splits} , not enough data points, returning whole.")
                    split_dataset = [ds]
                    self.rows_per_split[cg] *= self.n_splits
                    break
                # split_dataset.append({})
                # for k, v in ds.items():
                #     if isinstance(v, np.ndarray) and v.ndim == 2:
                #         split_dataset[-1][k] = v[:, slc]
                #     elif k == "start":
                #         inferred_freq = sub_df.loc[sub_df["continuity_group"] == 0].index.inferred_freq 
                #         split_dataset[-1][k] = ds["start"] \
                #                 + (np.timedelta64(inferred_freq[:-1], inferred_freq[-1]) 
                #                    * rows_per_split[cg] * split_idx)
                #         print(f"full dataset cg {cg} split {split_idx} start time = {split_dataset[-1][k]}, end time = {split_dataset[-1][k] + split_dataset[-1]['target'].shape[1]}, duration = {split_dataset[-1]['target'].shape[1] * split_dataset[-1][k].freq.delta}")
                #     else:
                #         split_dataset[-1][k] = v
            
            train_offset = int(self.train_split * self.rows_per_split[cg])
            val_offset = int(self.val_split * self.rows_per_split[cg])
            train_val_offset = train_offset + val_offset
            train_val_dataset, test_gen = split(split_dataset, offset=train_val_offset)
            train_dataset = []
            val_dataset = []
            for t, train_entry in enumerate(iter(train_val_dataset)):
                print(f"training dataset cg {cg}, split {t} start time = {train_entry['start']}, end time = {train_entry['start'] + train_entry['target'].shape[1]}, duration = {train_entry['target'].shape[1] * train_entry['start'].freq.delta}")

            train_dataset = [slice_data_entry(ds, slice_=slice(0, train_offset), prediction_length=self.prediction_length) for ds in iter(train_val_dataset)]
            val_dataset = [slice_data_entry(ds, slice_=slice(train_offset, val_offset), prediction_length=self.prediction_length) for ds in iter(train_val_dataset)]
            
            n_test_windows = int((self.test_split * self.rows_per_split[cg]) / self.prediction_length)
            test_dataset = test_gen.generate_instances(prediction_length=self.prediction_length, windows=n_test_windows)

            train_datasets.append(train_dataset)
            test_datasets.append(test_dataset)
            val_datasets.append(val_dataset)

        # inferred_freq = sub_df.loc[sub_df["continuity_group"] == 0].index.inferred_freq  
        train_dataset = ListDataset(chain(*train_datasets), self.freq, one_dim_target=self.one_dim_target)
        val_dataset = ListDataset(chain(*val_datasets), self.freq, one_dim_target=self.one_dim_target)
        test_dataset = list(chain(*test_datasets))
        return {"train": train_dataset, "val": val_dataset, "test": test_dataset}

    def highlight_entry(self, entry, color, ax, end_line=False):
        start = entry["start"].to_timestamp()
        # end = entry["start"] + (entry["target"].shape[1] * entry["start"].freq.delta)
        end = (entry["start"] + entry["target"].shape[1]).to_timestamp()
        # print(f"start time = {start}, end time = {end}")
        if end_line:
            ax.axvline(x=end)
        else:
            ax.axvspan(start, end, facecolor=color, alpha=0.2)
        
    def plot_dataset_splitting(self, original_dataset, train_dataset, test_pairs, val_pairs):
        # num_target_vars = next(original_dataset)["target"].shape[0]
        
        fig, axs = plt.subplots(self.num_target_vars, 1, sharex=True)
        for o, original_entry in enumerate(original_dataset):
            df = to_pandas(original_entry, is_multivariate=True).reset_index(names="time")
            df["time"] = df["time"].dt.to_timestamp()
            for a in range(self.num_target_vars):
                sns.lineplot(data=df, ax=axs[a], x="time", y=df.columns[a + 1])
        
        for train_entry in train_dataset:
            for a in range(self.num_target_vars):
                highlight_entry(train_entry, "red", ax=axs[a], end_line=True)
        # plt.show()
        axs[0].legend(["sub dataset", "training dataset"], loc="upper left")

        colors = colormaps["Pastel1"].colors
        for t, (test_input, test_label) in enumerate(test_pairs):
            for a in range(self.num_target_vars):
                highlight_entry(test_input, colors[0], axs[a])
                highlight_entry(test_label, colors[1], axs[a])
        for v, (val_input, val_label) in enumerate(val_pairs):
            for a in range(self.num_target_vars):
                highlight_entry(val_input, colors[2], axs[a])
                highlight_entry(val_label, colors[3], axs[a])
            plt.legend(["sub dataset", "test input", "test label"], loc="upper left")
        
        plt.show()