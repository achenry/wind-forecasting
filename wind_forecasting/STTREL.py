#!/usr/bin/env python
# coding: utf-8

# TODO:
# - Turn into pytorch lightning for easier parallelization ✅
# - Add parallelization (DistributedDataParallel) ✅
# - Use better logging (wandb/neptune/comet/clearml/mlflow) ✅
# - Add learning rate scheduler ✅
# - Check and reasure the number of heads necessary for the model. Per attention heads? ✅
# - Add multiple nodes parallelization capabilities
# - Add my dataset
# - Integrate preprocessing codebase with model codebase with my dataset.
# - Add automatic hyperparameter tuning (Population Based Training)
# - Add decoder
# - Add dataloader for multiple datasets
# - Reformat entire codebase into separate files (_init_.py, data.py, dataset.py, model.py, train.py, encoder.py, decoder.py, main.py...)
# - Enable colorful/rich terminal output + Emojis

# Current learning rate scheduler: (not implemented yet)
# 1. Starts with a lower learning rate (lr/10)
# 2. Warms up linearly to the peak learning rate over 10% of training
# 3. Gradually decreases following a cosine curve
# 4. Ends at a very small learning rate (lr/10000)

import warnings
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import polars as pl
import seaborn as sns # Used for Plotter class

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchmetrics import MeanSquaredError, MeanAbsolutePercentageError, MeanAbsoluteError

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

import wandb

warnings.filterwarnings('ignore', category=UserWarning, module='pandas.core.computation.expressions')
torch.set_float32_matmul_precision('medium')

# INFO: Custom model checkpoint callback to stop training when validation loss is below a threshold (50.0)
class ThresholdModelCheckpoint(ModelCheckpoint):
    def __init__(self, loss_threshold=50.0, **kwargs):
        super().__init__(**kwargs)
        self.loss_threshold = loss_threshold

    def _should_save_on_train_epoch_end(self, trainer):
        return self._save_if_below_threshold(trainer)
        
    def _save_if_below_threshold(self, trainer):
        """Only save if validation loss is below threshold"""
        current_loss = trainer.callback_metrics.get('val/loss')
        if current_loss is not None and current_loss < self.loss_threshold:
            return True
        return False

class Colors:
    # Regular colors
    BLUE = '\033[94m'      # Light/Bright Blue
    RED = '\033[91m'       # Light/Bright Red
    GREEN = '\033[92m'     # Light/Bright Green
    YELLOW = '\033[93m'    # Light/Bright Yellow
    CYAN = '\033[96m'      # Light/Bright Cyan
    MAGENTA = '\033[95m'   # Light/Bright Magenta
    
    # Bold colors
    BOLD_BLUE = '\033[1;34m'
    BOLD_GREEN = '\033[1;32m'
    BOLD_RED = '\033[1;31m'
    
    # Text style
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    # End color
    ENDC = '\033[0m'
    
    # Emojis
    ROCKET = '🚀'
    HOURGLASS = '⌛'
    CHECK = '✅'
    CROSS = '❌'
    FIRE = '🔥'
    CHART = '📊'
    WARNING = '⚠️'
    BRAIN = '🧠'
    SAVE = '💾'
    STAR = '⭐'
    
    @classmethod
    def disable_colors(cls):
        for attr in dir(cls):
            if not attr.startswith('__') and isinstance(getattr(cls, attr), str):
                setattr(cls, attr, '')

    @staticmethod
    def supports_color():
        """Check if term supports colors"""
        return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

if not Colors.supports_color():
    Colors.disable_colors()

class Config:
    HOME = str(Path.home())
    SAVE_DIR = os.path.join(HOME, 'STTRE_outputs')
    MODEL_DIR = os.path.join(SAVE_DIR, 'models')
    PLOT_DIR = os.path.join(SAVE_DIR, 'plots')
    DATA_DIR = os.path.join(SAVE_DIR, 'data')

    @classmethod
    def create_directories(cls):
        for directory in [cls.SAVE_DIR, cls.MODEL_DIR, cls.PLOT_DIR, cls.DATA_DIR]:
            os.makedirs(directory, exist_ok=True)

# INFO: Currently not used
class ProgressBar:
    def __init__(self, initial_error, target_error=0, width=50):
        self.initial_error = initial_error
        self.target_error = target_error
        self.width = width
        self.best_error = initial_error
        
    def update(self, current_error):
        self.best_error = min(self.best_error, current_error)
        # Calculate progress (0 to 1) where 1 means error reduced to target
        progress = 1 - (self.best_error - self.target_error) / (self.initial_error - self.target_error)
        progress = max(0, min(1, progress))  # Clamp between 0 and 1
        
        # Create the progress bar
        filled_length = int(self.width * progress)
        bar = '█' * filled_length + '░' * (self.width - filled_length)
        
        # Calculate percentage
        percent = progress * 100
        
        return f'{Colors.BOLD_BLUE}Progress: |{Colors.GREEN}{bar}{Colors.BOLD_BLUE}| {percent:6.2f}% {Colors.ENDC}'

# INFO: Currently not used
class Plotter:
    @staticmethod
    def plot_metrics(train_metrics, val_metrics, metric_names, dataset):
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.5)
        
        all_data = []
        for metric_name in metric_names:
            # Ensure train and val metrics have the same length
            min_len = min(len(train_metrics[metric_name]), len(val_metrics[metric_name]))
            epochs = list(range(1, min_len + 1))
            
            # Truncate metrics to the same length
            train_values = train_metrics[metric_name][:min_len]
            val_values = val_metrics[metric_name][:min_len]
            
            train_df = pl.DataFrame({
                'Epoch': epochs,
                'Value': train_values,
                'Type': ['Train'] * min_len,
                'Metric': [metric_name] * min_len
            })
            
            val_df = pl.DataFrame({
                'Epoch': epochs,
                'Value': val_values,
                'Type': ['Validation'] * min_len,
                'Metric': [metric_name] * min_len
            })
            
            all_data.extend([train_df, val_df])
        
        try:
            df = pl.concat(all_data)
            metric_data_pd = df.to_pandas()
            
            # Create FacetGrid
            g = sns.FacetGrid(metric_data_pd, col='Metric', col_wrap=len(metric_names), 
                            height=7, aspect=1.5)
            
            # Draw the lines
            g.map_dataframe(sns.lineplot, x='Epoch', y='Value', hue='Type',
                          palette=['#2ecc71', '#e74c3c'], linewidth=2.5)
            
            # Customize titles and labels
            g.set_titles(col_template="{col_name}", size=16, weight='bold', pad=20)
            g.set_axis_labels("Epoch", "Value", size=12)
            g.add_legend(title=None, fontsize=10)
            
            # Save the plot
            g.savefig(os.path.join(Config.PLOT_DIR, f'{dataset}_metrics_latest.png'),
                     bbox_inches='tight',
                     facecolor='white',
                     edgecolor='none',
                     dpi=300)
            
        except Exception as e:
            print(f"Error in plotting: {str(e)}")
            # Don't let plotting errors stop the training process
            pass

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

class Uber(BaseDataset):
    def __init__(self, dir, seq_len=60):
        super().__init__(dir, seq_len, columns=[1, 2, 3, 4, 5])

class IstanbulStock(BaseDataset):
    def __init__(self, dir, seq_len=40):
        super().__init__(dir, seq_len, columns=range(8))

class AirQuality(BaseDataset):
    def __init__(self, dir=None, seq_len=24):
        try:
            from ucimlrepo import fetch_ucirepo
            
            beijing_pm2_5 = fetch_ucirepo(id=381)
            
            # Get features DataFrame and convert to polars
            features_df = pl.from_pandas(beijing_pm2_5.data.features)
            
            # Get unique wind directions and create dummy columns
            wind_dummies = features_df.get_column('cbwd').unique()
            for direction in wind_dummies:
                features_df = features_df.with_columns(
                    pl.when(pl.col('cbwd') == direction)
                    .then(1)
                    .otherwise(0)
                    .alias(f'wind_{direction}')
                )
            
            # Drop original column
            features_df = features_df.drop('cbwd')
            
            # Convert to numpy arrays
            X = features_df.to_numpy().astype(np.float32)
            y = beijing_pm2_5.data.targets.to_numpy().astype(np.float32)
            
            # Combine features and target
            data = np.column_stack((X, y))
            
            # Remove rows with missing values
            data = data[~np.isnan(data).any(axis=1)]
            
            self.seq_len = seq_len
            self.X = self.normalize(data[:, :-1])
            self.y = data[:, [-1]]
            self.len = len(self.y)
            
        except Exception as e:
            print(f"Error fetching UCI data: {str(e)}")
            raise

    def __len__(self):
        return self.len - self.seq_len - 1

    def __getitem__(self, idx):
        x = np.transpose(self.X[idx:idx+self.seq_len])
        label = self.y[idx+self.seq_len+1]
        return torch.tensor(x, dtype=torch.float), torch.tensor(label, dtype=torch.float)

    # Normalise data prem
    def normalize(self, X):
        X = np.transpose(X)
        X_norm = [(x - np.min(x)) / (np.max(x) - np.min(x)) for x in X]
        return np.transpose(X_norm)

class Traffic(BaseDataset):
    def __init__(self, dir, seq_len=24):
        super().__init__(dir, seq_len, columns=range(8))

class AppliancesEnergy1(BaseDataset):
    def __init__(self, dir, seq_len=144):
        super().__init__(dir, seq_len, columns=range(26))

class AppliancesEnergy2(BaseDataset):
    def __init__(self, dir, seq_len=144):
        super().__init__(dir, seq_len, columns=range(26))

# INFO: Self-attention module for STTRE
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads, seq_len, module, rel_emb, device):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.seq_len = seq_len
        self.module = module
        self.rel_emb = rel_emb
        self.device = device
        modules = ['spatial', 'temporal', 'spatiotemporal', 'output']
        assert (module in modules), "Invalid module"

        if module in ['spatial', 'temporal']:
            self.head_dim = seq_len
            self.values = nn.Linear(self.embed_size, self.embed_size, dtype=torch.float32)
            self.keys = nn.Linear(self.embed_size, self.embed_size, dtype=torch.float32)
            self.queries = nn.Linear(self.embed_size, self.embed_size, dtype=torch.float32)

            if rel_emb:
                self.E = nn.Parameter(torch.randn([self.heads, self.head_dim, self.embed_size]).to(self.device))
        else:
            self.head_dim = embed_size // heads
            assert (self.head_dim * heads == embed_size), "Embed size not div by heads"
            self.values = nn.Linear(self.head_dim, self.head_dim, dtype=torch.float32)
            self.keys = nn.Linear(self.head_dim, self.head_dim, dtype=torch.float32)
            self.queries = nn.Linear(self.head_dim, self.head_dim, dtype=torch.float32)

            if rel_emb:
                self.E = nn.Parameter(torch.randn([1, self.seq_len, self.head_dim]).to(self.device))

        self.fc_out = nn.Linear(self.embed_size, self.embed_size)

    # Forward pass for self-attention
    def forward(self, x):
        N, _, _ = x.shape

        if self.module in ['spatial', 'temporal']:
            values = self.values(x)
            keys = self.keys(x)
            queries = self.queries(x)
            values = values.reshape(N, self.seq_len, self.heads, self.embed_size)
            keys = keys.reshape(N, self.seq_len, self.heads, self.embed_size)
            queries = queries.reshape(N, self.seq_len, self.heads, self.embed_size)
        else:
            values, keys, queries = x, x, x
            values = values.reshape(N, self.seq_len, self.heads, self.head_dim)
            keys = keys.reshape(N, self.seq_len, self.heads, self.head_dim)
            queries = queries.reshape(N, self.seq_len, self.heads, self.head_dim)
            values = self.values(values)
            keys = self.keys(keys)
            queries = self.queries(queries)

        if self.rel_emb:
            QE = torch.matmul(queries.transpose(1, 2), self.E.transpose(1,2))
            QE = self._mask_positions(QE)
            S = self._skew(QE).contiguous().view(N, self.heads, self.seq_len, self.seq_len)
            qk = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
            # Create mask on the same device as input tensor using triu (upper triangular matrix)
            mask = torch.triu(torch.ones(1, self.seq_len, self.seq_len, device=x.device), 1)
            if mask is not None:
                qk = qk.masked_fill(mask == 0, float("-1e20"))
            attention = torch.softmax(qk / (self.embed_size ** (1/2)), dim=3) + S
        else:
            qk = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
            # Create mask on the same device as input tensor using triu (upper triangular matrix)
            mask = torch.triu(torch.ones(1, self.seq_len, self.seq_len, device=x.device), 1)
            if mask is not None:
                qk = qk.masked_fill(mask == 0, float("-1e20"))
            attention = torch.softmax(qk / (self.embed_size ** (1/2)), dim=3)

        if self.module in ['spatial', 'temporal']:
            # Reshape attention and values to match the expected output shape
            z = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, self.seq_len*self.heads, self.embed_size)
        else:
            z = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, self.seq_len, self.heads*self.head_dim)

        z = self.fc_out(z)
        return z

    # Mask positions in the attention matrix
    def _mask_positions(self, qe):
        L = qe.shape[-1]
        # Create mask on the same device as input tensor
        mask = torch.triu(torch.ones(L, L, device=qe.device), 1).flip(1)
        return qe.masked_fill((mask == 1), 0)

    # Skew the attention matrix 
    def _skew(self, qe):
        padded_qe = F.pad(qe, [1,0])
        s = padded_qe.shape
        padded_qe = padded_qe.view(s[0], s[1], s[3], s[2])
        return padded_qe[:,:,1:,:]

# INFO: Transformer block for STTRE
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, seq_len, module, forward_expansion, rel_emb, device):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads, seq_len, module, rel_emb=rel_emb, device=device)

        # Batch normalisation for spatial and temporal modules
        if module in ['spatial', 'temporal']:
            self.norm1 = nn.BatchNorm1d(seq_len*heads)
            self.norm2 = nn.BatchNorm1d(seq_len*heads)
        else:
            self.norm1 = nn.BatchNorm1d(seq_len)
            self.norm2 = nn.BatchNorm1d(seq_len)

        # Feed-forward layer
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.LeakyReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )

    def forward(self, x):
        attention = self.attention(x)
        x = self.norm1(attention + x)
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        return out

# INFO: Encoder for STTRE
class Encoder(nn.Module):
    def __init__(self, seq_len, embed_size, num_layers, heads, device, forward_expansion, module, output_size=1, rel_emb=True):
        super(Encoder, self).__init__()
        self.module = module
        self.embed_size = embed_size
        self.device = device
        self.rel_emb = rel_emb
        self.fc_out = nn.Linear(embed_size, embed_size)

        # List of transformer blocks
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, seq_len, module, forward_expansion=forward_expansion, rel_emb=rel_emb, device=device)
             for _ in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
        out = self.fc_out(out)
        return out

# INFO: Main model class for STTRE using PyTorch Lightning
class LitSTTRE(L.LightningModule):
    def __init__(self, input_shape, output_size, model_params, train_params):
        super().__init__()
        self.automatic_optimization = True  # Enable automatic optimization
        self.save_hyperparameters()
        self.batch_size, self.num_var, self.seq_len = input_shape
        self.num_elements = self.seq_len * self.num_var
        self.embed_size = model_params['embed_size']
        self.train_params = train_params
        
        # Model components
        self.element_embedding = nn.Linear(self.seq_len, model_params['embed_size']*self.seq_len)
        self.pos_embedding = nn.Embedding(self.seq_len, model_params['embed_size'])
        self.variable_embedding = nn.Embedding(self.num_var, model_params['embed_size'])
        
        # Encoder components
        self.temporal = Encoder(
            seq_len=self.seq_len,
            embed_size=model_params['embed_size'],
            num_layers=model_params['num_layers'],
            heads=self.num_var,
            device=self.device,
            forward_expansion=model_params['forward_expansion'],
            module='temporal',
            rel_emb=True
        )
        
        # Spatial encoder
        self.spatial = Encoder(
            seq_len=self.num_var,
            embed_size=model_params['embed_size'],
            num_layers=model_params['num_layers'],
            heads=self.seq_len,
            device=self.device,
            forward_expansion=model_params['forward_expansion'],
            module='spatial',
            rel_emb=True
        )
        
        # Spatiotemporal encoder
        self.spatiotemporal = Encoder(
            seq_len=self.seq_len*self.num_var,
            embed_size=model_params['embed_size'],
            num_layers=model_params['num_layers'],
            heads=model_params['heads'],
            device=self.device,
            forward_expansion=model_params['forward_expansion'],
            module='spatiotemporal',
            rel_emb=True
        )
        
        # Output layers
        self.fc_out1 = nn.Linear(model_params['embed_size'], model_params['embed_size']//2)
        self.fc_out2 = nn.Linear(model_params['embed_size']//2, 1)
        self.out = nn.Linear((self.num_elements*3), output_size)
        
        # Initialize metrics for training
        metrics = ['mse', 'mae', 'mape']
        for split in ['train', 'val']:
            for metric in metrics:
                metric_class = {
                    'mse': MeanSquaredError,
                    'mae': MeanAbsoluteError,
                    'mape': MeanAbsolutePercentageError
                }[metric]
                setattr(self, f'{split}_{metric}', metric_class())
        
        # Initialize test metrics
        self.test_mse = None
        self.test_mae = None
        self.test_mape = None
        
        # Initialize training history
        self.training_history = {
            'train': {'MSE': [], 'MAE': [], 'MAPE': []},
            'val': {'MSE': [], 'MAE': [], 'MAPE': []}
        }
        
        # Track whether metrics have been updated
        self.metrics_updated = False

        # Enable gradient checkpointing
        self.gradient_checkpointing = True

    def forward(self, x):
        batch_size = len(x)
        
        # Temporal embedding
        positions = torch.arange(0, self.seq_len).expand(batch_size, self.num_var, self.seq_len).reshape(batch_size, self.num_var * self.seq_len).to(self.device)
        x_temporal = self.element_embedding(x).reshape(batch_size, self.num_elements, self.embed_size)
        x_temporal = F.dropout(self.pos_embedding(positions) + x_temporal, self.train_params['dropout'] if self.training else 0)
        
        # Spatial embedding
        x_spatial = torch.transpose(x, 1, 2).reshape(batch_size, self.num_var, self.seq_len)
        vars = torch.arange(0, self.num_var).expand(batch_size, self.seq_len, self.num_var).reshape(batch_size, self.num_var * self.seq_len).to(self.device)
        x_spatial = self.element_embedding(x_spatial).reshape(batch_size, self.num_elements, self.embed_size)
        x_spatial = F.dropout(self.variable_embedding(vars) + x_spatial, self.train_params['dropout'] if self.training else 0)
        
        # Spatiotemporal embedding
        x_spatio_temporal = self.element_embedding(x).reshape(batch_size, self.seq_len* self.num_var, self.embed_size)
        x_spatio_temporal = F.dropout(self.pos_embedding(positions) + x_spatio_temporal, self.train_params['dropout'] if self.training else 0)
        
        # Process through encoders
        out1 = self.temporal(x_temporal)
        out2 = self.spatial(x_spatial)
        out3 = self.spatiotemporal(x_spatio_temporal)
        
        # Final processing
        out = torch.cat((out1, out2, out3), 1)
        out = F.leaky_relu(self.fc_out1(out))
        out = F.leaky_relu(self.fc_out2(out))
        out = torch.flatten(out, 1)
        out = self.out(out)
        
        return out

    # Training step for STTRE
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        
        # Update metrics
        self.train_mse(y_hat, y)
        self.train_mae(y_hat, y)
        self.train_mape(y_hat, y)
        self.metrics_updated = True
        
        # Log metrics to wandb logger
        self.log_dict({
            'train/loss': loss,
            'train/mse': self.train_mse,
            'train/mae': self.train_mae,
            'train/mape': self.train_mape,
        }, prog_bar=True, sync_dist=True)
        
        return loss

    # Validation step for STTRE
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.mse_loss(y_hat, y)
        
        # Update metrics
        self.val_mse(y_hat, y)
        self.val_mae(y_hat, y)
        self.val_mape(y_hat, y)
        
        # Log metrics to wandb
        self.log_dict({
            'val/loss': val_loss,
            'val/mse': self.val_mse,
            'val/mae': self.val_mae,
            'val/mape': self.val_mape,
        }, prog_bar=True, sync_dist=True)
        
        return val_loss

    def test_step(self, batch, batch_idx):
        if not hasattr(self, 'test_metrics_initialized'):
            self.test_mse = MeanSquaredError().to(self.device)
            self.test_mae = MeanAbsoluteError().to(self.device)
            self.test_mape = MeanAbsolutePercentageError().to(self.device)
            self.test_metrics_initialized = True
        print(f"Processing test batch {batch_idx}")  # DEBUG
        
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        
        # Add sync_dist=True to ensure proper metric synchronization across devices
        self.log('test_loss', loss, on_step=True, on_epoch=True)
        self.log('test_mse', self.test_mse(y_hat, y), sync_dist=True)
        self.log('test_mae', self.test_mae(y_hat, y), sync_dist=True)
        self.log('test_mape', self.test_mape(y_hat, y), sync_dist=True)
        
        return loss

    # Called at the end of each training epoch
    def on_train_epoch_end(self):
        # Only compute and log metrics if they've been updated
        if not self.metrics_updated:
            return
        
        try:
            # Update history with current metrics
            metrics = {
                'train': {
                    'MSE': self.train_mse,
                    'MAE': self.train_mae,
                    'MAPE': self.train_mape
                },
                'val': {
                    'MSE': self.val_mse,
                    'MAE': self.val_mae,
                    'MAPE': self.val_mape
                }
            }
            
            # Update history
            for split in ['train', 'val']:
                for metric_name, metric in metrics[split].items():
                    if metric.update_called:  # Only compute if metric has been updated
                        value = float(metric.compute())
                        self.training_history[split][metric_name].append(value)
                        metric.reset()
            
            self.metrics_updated = False
            
        except Exception as e:
            print(f"Error in on_train_epoch_end: {str(e)}")

    # Called at the start of each training epoch
    def on_train_epoch_start(self):
        self.train_mse.reset()
        self.train_mae.reset()
        self.train_mape.reset()
        self.val_mse.reset()
        self.val_mae.reset()
        self.val_mape.reset()
        self.metrics_updated = False

    def configure_optimizers(self):
        # Choose Adam optimizer and learning rate scheduler for training
        optimizer = torch.optim.Adam(self.parameters(), lr=self.train_params['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=20
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1
            }
        }

    # Called when training starts
    def on_train_start(self):
        # Initialize metrics to zero
        self.train_mse.reset()
        self.train_mae.reset()
        self.train_mape.reset()
        self.val_mse.reset()
        self.val_mae.reset()
        self.val_mape.reset()
        self.metrics_updated = False
        
        # Initialize training history
        self.training_history = {
            'train': {'MSE': [], 'MAE': [], 'MAPE': []},
            'val': {'MSE': [], 'MAE': [], 'MAPE': []}
        }

    # INFO: Not used
    def _verify_test_data(self, data_module):
        print("Verifying test dataset...")
        
        try:
            test_loader = data_module.test_dataloader()
            if test_loader is None:
                print(f"{Colors.RED}Test dataloader is None{Colors.ENDC}")
                return False
                
            # Try to get a batch
            test_batch = next(iter(test_loader))
            print(f"Test batch shapes: input={test_batch[0].shape}, target={test_batch[1].shape}")
            
            return True
        except Exception as e:
            print(f"{Colors.RED}Error verifying test data: {str(e)}{Colors.ENDC}")
            return False

# INFO: Data module for STTRE using PyTorch Lightning
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
                
                # Create splits
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

# INFO: Base class for all transformer models to inherit from
class BaseTransformer:
    def __init__(self):
        self.model = None
        self.trainer = None
        self.data_module = None
        
    def cleanup_memory(self):
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
    def cleanup_old_checkpoints(self, model_dir, dataset_name, keep_top_k=3):
        """Clean up old checkpoints, keeping only the top k best models."""
        try:
            print(f"{Colors.BLUE}CC start...{Colors.ENDC}")  # DEBUG
            
            checkpoints = [f for f in os.listdir(model_dir) 
                          if f.startswith(f'{self.model_prefix}-{dataset_name.lower()}-') 
                          and f.endswith('.ckpt')]
            
            print(f"Found {len(checkpoints)} checkpoints")  # Debug line
            
            if len(checkpoints) > keep_top_k:
                # Sort checkpoints by validation loss (extracted from filename)
                checkpoints.sort(key=lambda x: float(x.split('-loss')[1].split('.ckpt')[0]))
                
                # Remove all but the top k checkpoints
                for checkpoint in checkpoints[keep_top_k:]:
                    checkpoint_path = os.path.join(model_dir, checkpoint)
                    try:
                        print(f"clean {checkpoint}")  # DEBUG
                        os.remove(checkpoint_path)
                        print(f"{Colors.YELLOW}Removed old checkpoint: {checkpoint}{Colors.ENDC}")
                    except OSError as e:
                        print(f"{Colors.RED}Error removing checkpoint {checkpoint}: {str(e)}{Colors.ENDC}")
                
                print(f"{Colors.GREEN}Kept top {keep_top_k} checkpoints for {dataset_name}{Colors.ENDC}")
            
            print(f"{Colors.BLUE}CCC{Colors.ENDC}")  # DEBUG
            
        except Exception as e:
            print(f"{Colors.RED}Error during checkpoint cleanup: {str(e)}{Colors.ENDC}")
            raise  # Re-raise the exception to be caught by the outer try-except

    def train(self, dataset_class, data_path, model_params, train_params):
        raise NotImplementedError("Subclasses must implement train method")

    def test(self, dataset_class, data_path, model_params, train_params, checkpoint_path):
        raise NotImplementedError("Subclasses must implement test method")

class STTREModel(BaseTransformer):
    """STTRE specific implementation"""
    def __init__(self):
        super().__init__()
        self.model_prefix = 'sttre'
        
    def _initialize_data_module(self, dataset_class, data_path, train_params):
        self.data_module = STTREDataModule(
            dataset_class=dataset_class,
            data_path=data_path,
            batch_size=train_params['batch_size']
        )
        self.data_module.setup()
        return self.data_module

    def _initialize_model(self, input_shape, model_params, train_params):
        self.model = LitSTTRE(
            input_shape=input_shape,
            output_size=model_params['output_size'],
            model_params=model_params,
            train_params=train_params
        )
        return self.model

    def _setup_trainer(self, run_name, callbacks, train_params, local_rank):
        wandb_logger = None
        if local_rank == 0:
            wandb_logger = WandbLogger(
                project="STTRE",
                name=run_name,
                log_model=True,
                save_dir=Config.SAVE_DIR,
                config={
                    "model_params": model_params,
                    "train_params": train_params,
                    "dataset": dataset_class.__name__,
                }
            )
            
        self.trainer = L.Trainer(
            max_epochs=train_params['epochs'],
            accelerator='auto',
            devices='auto',
            strategy='ddp_find_unused_parameters_true',
            logger=wandb_logger if local_rank == 0 else False,
            callbacks=callbacks,
            gradient_clip_val=train_params.get('gradient_clip', 1.0),
            precision=train_params.get('precision', 32),
            accumulate_grad_batches=train_params.get('accumulate_grad_batches', 1),
            log_every_n_steps=1,
            enable_progress_bar=(local_rank == 0),
            detect_anomaly=False,
            benchmark=True,
            deterministic=False,
        )
        return self.trainer, wandb_logger

    def _setup_callbacks(self, dataset_class, local_rank):
        callbacks = []
        
        # Only add RichProgressBar for rank 0
        if local_rank == 0:
            callbacks.append(
                RichProgressBar(
                    theme=RichProgressBarTheme(
                        description="white",
                        progress_bar="#6206E0",
                        progress_bar_finished="#6206E0",
                        progress_bar_pulse="#6206E0",
                        batch_progress="white",
                        time="grey54",
                        processing_speed="grey70",
                        metrics="white"
                    ),
                    leave=True
                )
            )
        
        # Add other callbacks for all ranks
        early_stopping = EarlyStopping(
            monitor='val/loss',
            patience=20,
            mode='min',
            min_delta=0.001,
            check_finite=True,
            check_on_train_epoch_end=False,
            verbose=True  # Add verbose output
        )
        
        checkpoint_callback = ThresholdModelCheckpoint(
            loss_threshold=50.0,
            monitor='val/loss',
            dirpath=Config.MODEL_DIR,
            filename=f'sttre-{dataset_class.__name__.lower()}-' + 'epoch{epoch:03d}-loss{val/loss:.4f}',
            save_top_k=3,
            mode='min',
            save_last=True,  # Save last model
            auto_insert_metric_name=False,
            verbose=True  # Add verbose output
        )
        
        callbacks.extend([checkpoint_callback, early_stopping])
        
        return callbacks

    # INFO: Train the STTRE model using the specified dataset
    def train(self, dataset_class, data_path, model_params, train_params):
        """Train the STTRE model using the specified dataset."""
        wandb_logger = None
        try:
            self.cleanup_memory()
            Config.create_directories()
            
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            if local_rank == 0:
                print(f"\n{Colors.BOLD_GREEN}Starting training for {dataset_class.__name__}{Colors.ENDC}")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{dataset_class.__name__}_" \
                      f"e{model_params['embed_size']}_" \
                      f"l{model_params['num_layers']}_" \
                      f"h{model_params['heads']}_" \
                      f"b{train_params['batch_size']}_" \
                      f"lr{train_params['lr']}_" \
                      f"{timestamp}"
            
            # Initialize components
            data_module = self._initialize_data_module(dataset_class, data_path, train_params)
            sample_batch = next(iter(data_module.train_dataloader()))
            model = self._initialize_model(sample_batch[0].shape, model_params, train_params)
            
            # Initialize wandb logger
            if local_rank == 0:
                wandb_logger = WandbLogger(
                    project="STTRE",
                    name=run_name,
                    log_model=True,
                    save_dir=Config.SAVE_DIR,
                    config={
                        "model_params": model_params,
                        "train_params": train_params,
                        "dataset": dataset_class.__name__,
                    }
                )
            
            # Setup callbacks and trainer
            callbacks = self._setup_callbacks(dataset_class, local_rank)
            trainer = L.Trainer(
                max_epochs=train_params['epochs'],
                accelerator='auto',
                devices='auto',
                strategy='ddp_find_unused_parameters_true',
                logger=wandb_logger if local_rank == 0 else False,
                callbacks=callbacks,
                gradient_clip_val=train_params.get('gradient_clip', 1.0),
                precision=train_params.get('precision', 32),
                accumulate_grad_batches=train_params.get('accumulate_grad_batches', 1),
                log_every_n_steps=1,
                enable_progress_bar=(local_rank == 0),
                detect_anomaly=False,
                benchmark=True,
                deterministic=False,
            )
            
            # Train and test
            print(f"{Colors.BLUE}Starting model training...{Colors.ENDC}")
            trainer.fit(model, data_module)
            print(f"{Colors.GREEN}Model training completed!{Colors.ENDC}")
            
            if local_rank == 0:
                if trainer.callback_metrics.get('val/loss') is not None:
                    print(f"\n{Colors.BOLD_GREEN}Final validation loss: {trainer.callback_metrics['val/loss']:.4f}{Colors.ENDC}")
                
                print(f"\n{Colors.BOLD_BLUE}Post-training phase:{Colors.ENDC}")
                
                try:
                    print("Cleaning up checkpoints...")
                    self.cleanup_old_checkpoints(Config.MODEL_DIR, dataset_class.__name__)
                    print("Checkpoint cleanup completed successfully")
                    
                    print(f"\n{Colors.BOLD_BLUE}Starting testing...{Colors.ENDC}")
                    
                    # Create a new trainer specifically for testing
                    test_trainer = L.Trainer(
                        accelerator='gpu',
                        devices=[0],  # Use single GPU
                        num_nodes=1,  # Single node
                        logger=wandb_logger,
                        enable_progress_bar=True,
                        strategy='auto'  # Use default strategy for single device
                    )
                    
                    print("Test trainer created")
                    print(f"Testing with model on device: {next(model.parameters()).device}")
                    print(f"DataModule state: {data_module.test_dataloader() is not None}")
                    
                    try:
                        test_results = test_trainer.test(model, datamodule=data_module)
                        print(f"{Colors.BOLD_GREEN}Testing completed!{Colors.ENDC}")
                        print(f"Test results: {test_results}")
                    except Exception as e:
                        print(f"{Colors.RED}Error during test phase: {str(e)}{Colors.ENDC}")
                        print(f"Model state: {model.training}")
                        print(f"Device info: {torch.cuda.get_device_properties(0)}")
                        raise
                    
                except Exception as e:
                    print(f"{Colors.RED}Error in post-training phase: {str(e)}{Colors.ENDC}")
                    raise
            else:
                test_results = None
                
            return model, trainer, test_results
            
        except Exception as e:
            if local_rank == 0:
                print(f"{Colors.RED}Error during training: {str(e)} {Colors.CROSS}{Colors.ENDC}")
                print(f"Current trainer state: {trainer.state if trainer else 'No trainer'}")
            raise
        finally:
            self.cleanup_memory()
            if local_rank == 0 and wandb_logger:
                wandb.finish()

    def test(self, dataset_class, data_path, model_params, train_params, checkpoint_path):
        """Test the STTRE model using a saved checkpoint."""
        Config.create_directories()
        
        data_module = self._initialize_data_module(dataset_class, data_path, train_params)
        sample_batch = next(iter(data_module.train_dataloader()))
        
        model = LitSTTRE.load_from_checkpoint(
            checkpoint_path,
            input_shape=sample_batch[0].shape,
            output_size=model_params['output_size'],
            model_params=model_params,
            train_params=train_params
        )
        
        wandb_logger = WandbLogger(
            project="STTRE",
            name=f"test_{dataset_class.__name__}",
            log_model=False,
            save_dir=Config.SAVE_DIR
        )
        
        test_trainer = L.Trainer(
            accelerator='gpu',
            devices=[0],
            num_nodes=1,
            logger=wandb_logger,
            enable_progress_bar=True,
            strategy='auto'
        )
        
        model = model.cuda()
        test_results = test_trainer.test(model, datamodule=data_module, verbose=True)
        print(f"{Colors.BOLD_GREEN}Testing completed! {Colors.CHECK}{Colors.ENDC}")
        
        return model, test_results

# TODO: SpaceTimeFormer model
class SpaceTimeFormer(BaseTransformer):
    def __init__(self):
        super().__init__()
        # ...
        
    def train(self, dataset_class, data_path, model_params, train_params):
        # ...
        pass
        
    def test(self, dataset_class, data_path, model_params, train_params, checkpoint_path):
        # ...
        pass

##################################################################################################
#########################################[ MAIN ]#################################################
##################################################################################################

if __name__ == "__main__":
    # Initialize wandb only on rank 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        wandb.login() # Login to wandb website
        
    # Change to checkpoint path to test and validate for pre-trained model
    checkpoint_path = os.path.join(Config.MODEL_DIR, 'sttre-uber-epoch=519-val_loss=6.46.ckpt')
    
    # Initialize model
    sttre = STTREModel()
    
    # INFO: MAIN MODEL PARAMETERS
    model_params = {
        'embed_size': 32, # Determines dimension of the embedding space
        'num_layers': 3, # Number of transformer blocks stacked
        'heads': 4, # Number of heads for spatio-temporal attention
        'forward_expansion': 4, # Multiplier for feedforward network size
        'output_size': 1 # Number of output variables
    }

    # INFO: MAIN TRAINING PARAMETERS
    train_params = {
        'batch_size': 32, # larger = more stable gradients
        'epochs': 2000, # Maximum number of epochs to train
        'lr': 0.0001, # Step size
        'dropout': 0.1, # Regularization parameter (prevents overfitting)
        'patience': 50, # Number of epochs to wait before early stopping
        'gradient_clip': 0.5, # Prevents gradient explosion
        'precision': '32-true', # 16-mixed enables mixed precision training, 32-true is full precision
        'accumulate_grad_batches': 2, # Simulates a larger batch size
        'test_split': 0.5, # Fraction of data to use for testing
        'val_split': 0.2 # Fraction of data to use for validation
    }

    # INFO: DATASET CHOICE AND PATHS
    datasets = {
        # 'AirQuality': (AirQuality, None),
        'Uber': (Uber, os.path.join(Config.DATA_DIR, 'uber_stock.csv')),
        # 'IstanbulStock': (IstanbulStock, os.path.join(Config.DATA_DIR, 'istanbul_stock.csv')),
        # 'Traffic': (Traffic, os.path.join(Config.DATA_DIR, 'traffic.csv')),
        # 'AppliancesEnergy1': (AppliancesEnergy1, os.path.join(Config.DATA_DIR, 'appliances_energy1.csv')),
        # 'AppliancesEnergy2': (AppliancesEnergy2, os.path.join(Config.DATA_DIR, 'appliances_energy2.csv'))
    }

    trainer = None
    for dataset_name, (dataset_class, data_path) in datasets.items():
        try:
            if local_rank == 0:
                print(f"\n{Colors.BOLD_BLUE}Processing {dataset_name} dataset...{Colors.ENDC}")
            
            model, trainer, test_results = sttre.train(
                dataset_class, 
                data_path, 
                model_params, 
                train_params
            )
            
            if local_rank == 0:
                
                # model, test_results = test_sttre(
                #     dataset_class, 
                #     data_path, 
                #     model_params, 
                #     train_params,
                #     checkpoint_path
                # )
                # print(f"{Colors.BOLD_GREEN}Completed testing {dataset_name} dataset {Colors.CHECK}{Colors.ENDC}")
            
                print(f"\n{Colors.BOLD_GREEN}Completed {dataset_name} dataset{Colors.CHECK}{Colors.ENDC}")
                if test_results:
                    print(f"Test results: {test_results}")
            
            # Cleanup after training
            del model, trainer
            sttre.cleanup_memory()
            
        except Exception as e:
            if local_rank == 0:
                print(f"{Colors.RED}Error processing {dataset_name}: {str(e)} {Colors.CROSS}{Colors.ENDC}")
            sttre.cleanup_memory()
            continue

    if local_rank == 0:
        print(f"\n{Colors.BOLD_GREEN}All experiments completed! {Colors.CHECK}{Colors.ENDC}")