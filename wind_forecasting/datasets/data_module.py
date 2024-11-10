## TODO MERGE

import os

import numpy as np
import lightning as L
from pytorch_lightning.utilities import CombinedLoader
import torch
from torch.utils.data import Dataset, DataLoader
from ..utils.colors import Colors

from torch.utils.data import Dataset, DataLoader
import warnings

# INFO: Data module for STTRE using PyTorch Lightning TODO JUAN make this general to all modules
class STTREDataModule(L.LightningDataModule):
    def __init__(self, dataset_class, data_path, batch_size, test_split=0.2, val_split=0.1):
        super().__init__()
        self.dataset_class = dataset_class
        self.data_path = data_path
        self.batch_size = batch_size
        self.test_split = test_split
        self.val_split = val_split
        self.num_workers = min(32, os.cpu_count())  # Dynamically set workers
        self.persistent_workers = True # Keep workers alive between epochs
        self.pin_memory = True # Use pinned memory for faster memory transfers
        
    def setup(self, stage=None):
        if not hasattr(self, 'train_dataset'):  # Only setup once
            try:
                # Create full dataset
                full_dataset = self.dataset_class(self.data_path)
                
                # Validate dataset size
                if len(full_dataset) < self.batch_size:
                    raise ValueError(f"Dataset size ({len(full_dataset)}) must be greater than batch size ({self.batch_size})")
                
                # Calculate lengths
                dataset_size = len(full_dataset)
                
                # Calculate split sizes
                test_size = int(self.test_split * dataset_size)
                val_size = int(self.val_split * dataset_size)
                train_size = dataset_size - test_size - val_size
                
                # Validate split sizes before creating splits
                min_required_size = self.batch_size * 2  # Ensure at least 2 batches per split
                if train_size < min_required_size or val_size < min_required_size or test_size < min_required_size:
                    # Adjust batch size automatically
                    suggested_batch_size = min(train_size // 2, val_size // 2, test_size // 2)
                    raise ValueError(
                        f"Dataset too small for current batch size ({self.batch_size}). "
                        f"Split sizes - train: {train_size}, val: {val_size}, test: {test_size}. "
                        f"Suggested batch size: {suggested_batch_size}"
                    )
                
                # Create splits # TODO JUAN will these be sequential?
                generator = torch.Generator().manual_seed(42)
                self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
                    full_dataset, 
                    [train_size, val_size, test_size],
                    generator=generator
                )
                
                # Only print once from rank 0
                if self.trainer and self.trainer.is_global_zero:
                    print(f"{Colors.BOLD_BLUE}Dataset splits:{Colors.ENDC}")
                    print(f"{Colors.CYAN}Training samples: {train_size}{Colors.ENDC}")
                    print(f"{Colors.YELLOW}Validation samples: {val_size}{Colors.ENDC}")
                    print(f"{Colors.GREEN}Test samples: {test_size}{Colors.ENDC}")
                
            except Exception as e:
                raise ValueError(f"Error preparing data: {str(e)}")

    # Training dataloader
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=4,  # change prefetching
            drop_last=True  # Drop incomplete batches
        )

    # Validation dataloader
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )

    # Test dataloader
    def test_dataloader(self):
        print("Creating test dataloader...")
        try:
            loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers
            )
            print(f"Test dataloader created with {len(loader)} batches")
            return loader
        except Exception as e:
            print(f"Error creating test dataloader: {str(e)}")
            raise

class DataModule(L.LightningDataModule):
    """_summary_
    The DataModule Encapsulates the entire data pipeline, including:
    Training, validation, and testing datasets.
    Data loading with DataLoader.
    Data transformations (e.g., resizing, normalization).
    Data downloading or preprocessing.
    Args:
        L (_type_): _description_
    """
    def __init__(
        self,
        dataset_class,
        config
    ):
        # TODO add input validation etc
        super().__init__()
        self.dataset = dataset_class(data_path=config["dataset"]["data_path"], 
                                      context_len=config["dataset"]["context_len"], 
                                      target_len=config["dataset"]["target_len"], 
                                      normalize=config["dataset"]["normalize"], 
                                      test_split=config["dataset"]["test_split"], 
                                      val_split=config["dataset"]["val_split"],
                                      **config["dataset"]["dataset_kwargs"])
        self.batch_size = config["dataset"]["batch_size"]
        self.workers = config["dataset"]["workers"]
        self.collate_fn = config["dataset"]["collate_fn"]
        self.context_len = config["dataset"]["context_len"]
        self.target_len = config["dataset"]["target_len"]
        if config["dataset"]["overfit"]:
            warnings.warn("Overriding val and test dataloaders to use train set!")
        self.overfit = config["dataset"]["overfit"]

    def train_dataloader(self, shuffle=True):
        return self._make_dloader("train", shuffle=shuffle)

    def val_dataloader(self, shuffle=False):
        return self._make_dloader("val", shuffle=shuffle)

    def test_dataloader(self, shuffle=False):
        return self._make_dloader("test", shuffle=shuffle)

    def _make_dloader(self, split, shuffle=False):
        """_summary_
        The DataLoader wraps a Dataset to provide an iterable over the data; 
        Enables efficient batching, shuffling, and parallel data loading;
        Handles the logic for loading data in batches and shuffling the order of samples;

        Args:
            split (_type_): _description_
            shuffle (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        if self.overfit:
            split = "train"
            shuffle = True
        # datasets = getattr(self.dataset, f"{split}_data")
        n_datasets = len(getattr(self.dataset, f"{split}_data"))
        combined_loader = []
        for d in range(n_datasets):
            combined_loader.append(DataLoader(
                ContinuousDataset(dataset=self.dataset,
                dataset_index=d, split=split),
                shuffle=shuffle,
                batch_size=self.batch_size,
                num_workers=self.workers,
                collate_fn=self.collate_fn,
                persistent_workers=True
            ))
        return CombinedLoader(combined_loader, mode="max_size")

class ContinuousDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        dataset_index: int = 0,
        split: str = "train"
    ):
        """_summary_
        this class represents a particular split from a particular individual continuous dataset to feed to the combined DataLoader.
        The Dataset represents a collection of data samples and must implement __len__ and __getitem__
        Args:
            dataset (Dataset): _description_
            dataset_index (int, optional): _description_. Defaults to 0.
            split (str, optional): _description_. Defaults to "train".
            context_len (int, optional): _description_. Defaults to None.
            target_len (int, optional): _description_. Defaults to None.
        """
        self.series = dataset
        self.dataset_index = dataset_index
        self.split = split
        self.context_len = dataset.context_len
        self.target_len = dataset.target_len
        # TODO this will be empty if target points + context points is too long`
        self._slice_start_points = list(range(
                0,
                self.series.length(self.split, self.dataset_index)
                + (-self.target_len - self.context_len)
                + 1,
            ))

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
     
class BaseDataset(Dataset):
    def __init__(self, dir, seq_len, columns):
        self.seq_len = seq_len
        data = []
        with open(dir, 'r') as file:
            next(file)  # Skip header
            for line in file:
                try:
                    row = [float(line.split(',')[col]) for col in columns]
                    data.append(row)
                except ValueError:
                    continue  # Skip non-numeric rows
        data = np.array(data)
        
        # Check if we have enough data points
        if len(data) <= seq_len + 1:
            raise ValueError(f"Dataset length ({len(data)}) must be greater than sequence length + 1 ({seq_len + 1})")
            
        self.X = self.normalize(data[:, :-1])
        self.y = data[:, [-1]]
        self.len = len(self.y)

    def __len__(self):
        valid_length = self.len - self.seq_len - 1
        if valid_length <= 0:
            raise ValueError(f"No valid sequences can be generated with sequence length {self.seq_len}")
        return valid_length

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Index out of bounds")
        x = np.transpose(self.X[idx:idx+self.seq_len])
        label = self.y[idx+self.seq_len+1]
        return torch.tensor(x, dtype=torch.float), torch.tensor(label, dtype=torch.float)

    def normalize(self, X):
        X = np.transpose(X)
        X_norm = [(x - np.min(x)) / (np.max(x) - np.min(x)) for x in X]
        return np.transpose(X_norm)