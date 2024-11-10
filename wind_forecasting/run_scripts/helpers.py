from torch.utils.data import Dataset, DataLoader
import warnings

import lightning as L
import torch
from lightning.pytorch.utilities import CombinedLoader

class TorchDataModule(L.LightningDataModule):
	def __init__(
		self,
		dataset,
		**config
	):
		# TODO add input validation etc
		super().__init__()
		self.dataset = dataset
		self.batch_size = config["batch_size"]
		self.workers = config["workers"]
		self.collate_fn = config["collate_fn"]
		self.context_len = config["context_len"]
		self.target_len = config["target_len"]
		if config["overfit"]:
			warnings.warn("Overriding val and test dataloaders to use train set!")
		self.overfit = config["overfit"]

	def train_dataloader(self, shuffle=True):
		return self._make_dloader("train", shuffle=shuffle)

	def val_dataloader(self, shuffle=False):
		return self._make_dloader("val", shuffle=shuffle)

	def test_dataloader(self, shuffle=False):
		return self._make_dloader("test", shuffle=shuffle)

	def _make_dloader(self, split, shuffle=False):
		if self.overfit:
			split = "train"
			shuffle = True
		# datasets = getattr(self.dataset, f"{split}_data")
		n_datasets = len(getattr(self.dataset, f"{split}_data"))
		combined_loader = []
		for d in range(n_datasets):
			combined_loader.append(DataLoader(
				TorchDataset(dataset=self.dataset,
				dataset_index=d, split=split,
				context_len=self.dataset.context_len,
				target_len=self.dataset.target_len),
				shuffle=shuffle,
				batch_size=self.batch_size,
				num_workers=self.workers,
				collate_fn=self.collate_fn,
				persistent_workers=True
			))
		return CombinedLoader(combined_loader, mode="max_size")

class TorchDataset(Dataset):
	def __init__(
		self,
		dataset: Dataset,
		dataset_index: int = 0,
		split: str = "train",
		context_len: int = None,
		target_len: int = None
	):
		self.series = dataset
		self.dataset_index = dataset_index
		self.split = split
		self.context_len = context_len
		self.target_len = target_len
		# TODO this will be empty if target points + context points is too long`
		self._slice_start_points = [
			i
			for i in range(
				0,
				self.series.length(self.split, self.dataset_index)
				+ (-self.target_len - self.context_len)
				+ 1,
			)
		]

	def __len__(self):
		return len(self._slice_start_points)

	def _torch(self, *dfs):
		return tuple(torch.from_numpy(x.values).float() for x in dfs)

	def __getitem__(self, i):
		start = self._slice_start_points[i]
		stop = start + (self.context_len + self.target_len)
		# series_slice = self.series.iloc[start:stop]
		series_slice = self.series.get_slice(
			self.split,
			self.dataset_index,
			start=start,
			stop=stop,
			skip=1,
		)

		series_slice = series_slice.drop(columns=[self.series.time_col_name])
		ctxt_slice, trgt_slice = (
			series_slice.iloc[: self.context_len],
			series_slice.iloc[self.context_len :],
		)

		ctxt_x = ctxt_slice[self.series.time_cols]
		trgt_x = trgt_slice[self.series.time_cols]

		ctxt_y = ctxt_slice[self.series.target_cols + self.series.exo_cols]
		# ctxt_y = ctxt_y.drop(columns=self.series.remove_target_from_context_cols)

		trgt_y = trgt_slice[self.series.target_cols]

		return self._torch(ctxt_x, ctxt_y, trgt_x, trgt_y)