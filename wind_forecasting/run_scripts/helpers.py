# /home/boujuan/wind-forecasting/wind_forecasting/run_scripts/helpers.py
import lightning as L
from torch.utils.data import DataLoader

class TorchDataModule(L.LightningDataModule):
    def __init__(self, dataset, batch_size, workers, **kwargs):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = workers
        
    def train_dataloader(self):
        return DataLoader(
            self.dataset.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )