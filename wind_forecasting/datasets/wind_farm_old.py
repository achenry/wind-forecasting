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
	def __init__(self, *, data_path, context_len, target_len, normalize, normalization_consts, **kwargs):
		# TODO input validation etc
		self.context_len = context_len
		self.target_len = target_len
		self.normalize = normalize
		self.time_col_name = "time"
		self.time_features = ["year", "month", "day", "hour", "minute", "second"]
		
		dfs = [df.to_pandas() for df in pl.scan_parquet(source=data_path).collect(streaming=True).partition_by("continuity_group")]

		to_remove = []
		for d in range(len(dfs)):
			if len(dfs[d]) < ((self.target_len + self.context_len)):
				print(f"{d}-th dataframe is too short with length of {len(dfs[d])}. Removing from list.")
				to_remove.append(d)
		dfs = [df for d, df in enumerate(dfs) if d not in to_remove]

		horz_ws_cols = sorted([col for col in dfs[0].columns if "ws_horz" in col])
		vert_ws_cols = sorted([col for col in dfs[0].columns if "ws_vert" in col])
		nd_sin = sorted([col for col in dfs[0].columns if "nd_sin" in col])
		nd_cos = sorted([col for col in dfs[0].columns if "nd_cos" in col])

		# self.input_cols = horz_ws_cols + vert_ws_cols + nd_sin + nd_cos
		self.input_cols = [col for col in (horz_ws_cols + vert_ws_cols + nd_sin + nd_cos) if any(tid in col for tid in kwargs["target_turbine_ids"])]
		self.target_cols = [col for col in (horz_ws_cols + vert_ws_cols) if any(tid in col for tid in kwargs["target_turbine_ids"])]

		dfs = [df[list(set([self.time_col_name] + self.input_cols + self.target_cols))] for df in dfs]

		dfs = [stf.data.timefeatures.time_features(
				pd.to_datetime(df[self.time_col_name], format="%Y-%m-%d %H:%M:%S"),
				df,
				time_col_name=self.time_col_name,
				use_features=self.time_features,
			) for df in dfs]
		
		self.time_cols = [col for col in dfs[0].columns if col.lower() in self.time_features]
		self.all_continuous_dfs = dfs

		not_exo_cols = self.time_cols + self.target_cols
		self.exo_cols = dfs[0].columns.difference(not_exo_cols).tolist()
		self.exo_cols.remove(self.time_col_name)

		self._x_dim = len(self.time_features) # dimension of time feature
		self._yc_dim = len(self.target_cols + self.exo_cols)
		self._yt_dim = len(self.target_cols)

		# scalar value are defined for target values
		if normalize:
			self._scaler = MinMaxScaler(feature_range=(-1, 1))
			self._scaler = self._scaler.fit(
                self._train_data[self.target_cols + self.exo_cols].values
            )
		elif normalization_consts is not None:
			self._scaler = MinMaxScaler(feature_range=(-1, 1))
			self._scaler.min_ = [normalization_consts[f"{'_'.join(col.split('_')[:-1])}_min"].iloc[0] for col in self.target_cols + self.exo_cols]
			self._scaler.max_ = [normalization_consts[f"{'_'.join(col.split('_')[:-1])}_max"].iloc[0] for col in self.target_cols + self.exo_cols]
		
	@property
	def x_dim(self):
		return self._x_dim

	@property
	def yc_dim(self):
		return self._yc_dim

	@property
	def yt_dim(self):
		return self._yt_dim
		
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