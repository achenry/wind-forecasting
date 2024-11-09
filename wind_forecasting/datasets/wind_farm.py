import numpy as np
import polars as pl
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import torch
from wind_forecasting.models import spacetimeformer as stf
import random
from sklearn.preprocessing import MinMaxScaler


# TODO make a base class to enforce implementing certain methods...
class KPWindFarm(Dataset):
	def __init__(self, *, data_path, context_len, target_len, normalize, test_split, val_split, **kwargs):
		# TODO input validation etc
		self.context_len = context_len
		self.target_len = target_len
		self.normalize = normalize
		self.time_col_name = "time"
		self.time_features = ["year", "month", "day", "hour", "minute", "second"]
		
		dfs = [df.to_pandas() for df in pl.scan_parquet(source=data_path).collect(streaming=True).partition_by("continuity_group")]
		
		horz_ws_cols = sorted([col for col in dfs[0].columns if "horizontal_ws" in col])
		vert_ws_cols = sorted([col for col in dfs[0].columns if "vertical_ws" in col])
		nd_sin = sorted([col for col in dfs[0].columns if "nd_sin" in col])
		nd_cos = sorted([col for col in dfs[0].columns if "nd_cos" in col])

		self.input_cols = horz_ws_cols + vert_ws_cols + nd_sin + nd_cos
		self.target_cols = [col for col in (horz_ws_cols + vert_ws_cols) if any(tid in col for tid in kwargs["target_turbine_ids"])]

		self._x_dim = len(self.input_cols)
		self._yc_dim = self._yt_dim = len(self.target_cols)

		dfs = [stf.data.timefeatures.time_features(
				pd.to_datetime(df[self.time_col_name], format="%Y-%m-%d %H:%M:%S"),
				df,
				time_col_name=self.time_col_name,
				use_features=self.time_features,
			) for df in dfs]
		
		self.time_cols = [col for col in dfs[0].columns if col.lower() in self.time_features]

		# TODO split training/validation/testing as percentages of each dataset? OR shuffle and approximately select 70/20/10
		random.shuffle(dfs)
		total_rows = sum(len(df) for df in dfs)
		n_train_rows = round(total_rows * (1.0 - test_split - val_split))
		n_test_rows = round(total_rows * test_split)
		n_val_rows = round(total_rows * val_split)

		train_indices = fill_split(dfs=dfs, n_split_rows=n_train_rows, taken_split_indices=[])
		val_indices = fill_split(dfs=dfs, n_split_rows=n_val_rows, taken_split_indices=train_indices)
		test_indices = fill_split(dfs=dfs, n_split_rows=n_test_rows, taken_split_indices=train_indices+val_indices)

		self._train_data = [dfs[i] for i in train_indices]
		self._val_data = [dfs[i] for i in val_indices]
		
		if ["test_split"] == 0.0:
			print("`test_split` set to 0. Using Val set as Test set.")
			# self._test_data = df[val_mask]
			self._test_data = [dfs[i] for i in val_indices]
		else:
			self._test_data = [dfs[i] for i in test_indices]

		self._scaler = MinMaxScaler()
		
		not_exo_cols = self.time_cols + self.target_cols
		self.exo_cols = dfs[0].columns.difference(not_exo_cols).tolist()
		self.exo_cols.remove(self.time_col_name)
		
	@property
	def x_dim(self):
		return self._x_dim

	@property
	def yc_dim(self):
		return self._yc_dim

	@property
	def yt_dim(self):
		return self._yt_dim
		
	@property
	def train_data(self):
		return self._train_data

	@property
	def val_data(self):
		return self._val_data

	@property
	def test_data(self):
		return self._test_data
	
	def length(self, split, dataset_index):
		return len(getattr(self, f"{split}_data")[dataset_index])

	def __len__(self):
		return (self.len - self.seq_len)-1

	def __getitem__(self, idx):
		x = np.transpose(self.X[idx:idx + self.context_len])
		y = self.y[idx + self.context_len:idx + self.context_len + self.target_len]

		return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)
	
	def get_slice(self, split, dataset_index, start, stop, skip):
		assert split in ["train", "val", "test"]
		if split == "train":
			return self.train_data[dataset_index].iloc[start:stop:skip]
		elif split == "val":
			return self.val_data[dataset_index].iloc[start:stop:skip]
		else:
			return self.test_data[dataset_index].iloc[start:stop:skip]
		
	def apply_scaling(self, array):
		if not self.normalize:
			return array
		dim = array.shape[-1]
		return (array - self._scaler.min_[:dim]) / (self._scaler.max_[:dim] - self._scaler.min_[:dim])
	
	def reverse_scaling(self, array):
		if not self.normalize:
			return array
		# self._scaler is fit for target_cols + exo_cols
		# if the array dim is less than this length we start
		# slicing from the target cols
		dim = array.shape[-1]
		return (array * (self._scaler.max_[:dim] - self._scaler.min_[:dim])) + self._scaler.min_[:dim]

def fill_split(dfs, n_split_rows, taken_split_indices):
	split_indices = []
	for i in range(len(dfs)):
		if i in taken_split_indices:
			continue

		split_len = sum(len(dfs[k]) for k in split_indices)
		if split_len >= n_split_rows:
			return split_indices
		elif split_len + len(dfs[i]) > n_split_rows:
			lowest_diff = abs(n_split_rows - split_len)
			best_last_df_idx = i
			for j in range(i + 1, len(dfs)):
				if abs(n_split_rows - (split_len + len(dfs[j]))) < lowest_diff:
					lowest_diff = n_split_rows - (split_len + len(dfs[j]))
					best_last_df_idx = j
			split_indices.append(best_last_df_idx)
		else:
			split_indices.append(i)

	return split_indices