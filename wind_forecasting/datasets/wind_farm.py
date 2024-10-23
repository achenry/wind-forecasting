import numpy as np
import polars as pl
from torch.utils.data import Dataset
import numpy as np
import torch

class KPWindFarm(Dataset):
    def __init__(self, data_path, is_normalized, target_turbine_ids, context_len, target_len):
        self.context_len = context_len
        self.target_len = target_len
        
        data = pl.scan_parquet(source=data_path).to_pandas()
        
        horz_ws_cols = sorted([col for col in data.columns if "horizontal_ws" in col])
        vert_ws_cols = sorted([col for col in data.columns if "vertical_ws" in col])
        nd_sin = sorted([col for col in data.columns if "nd_sin" in col])
        nd_cos = sorted([col for col in data.columns if "nd_cos" in col])

        input_cols = horz_ws_cols + vert_ws_cols + nd_sin + nd_cos
        output_cols = [col for col in (horz_ws_cols + vert_ws_cols) if any(tid in col for tid in target_turbine_ids)]

        if is_normalized:
            self.X = data.loc[:, input_cols].to_numpy()
        else:
            self.X = self.normalize(data.loc[:, input_cols].to_numpy())

        self.y = data.loc[:, output_cols].to_numpy()
        self.len = len(self.y)

        self.target_cols = output_cols
        self.time_col_name = "time" 
        self.time_features = ["year", "month", "day", "hour", "minute", "second"]

    def __len__(self):
        return (self.len - self.seq_len)-1

    def __getitem__(self, idx):
        x = np.transpose(self.X[idx:idx + self.context_len])
        y = self.y[idx + self.context_len:idx + self.context_len + self.target_len]

        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)

    def normalize(self, X):
        X = np.transpose(X)
        X_norm = []
        for x in X:
            x = (x-np.min(x)) / (np.max(x)-np.min(x))
            X_norm.append(x)
        return np.transpose(X_norm)