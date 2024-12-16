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
import uuid
from datetime import datetime

import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics import MeanSquaredError, MeanAbsolutePercentageError, MeanAbsoluteError

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

from ..utils.checkpoints import ThresholdModelCheckpoint
from ..utils.colors import Colors
from ..utils.config import Config
from ..datasets.data_module import STTREDataModule

from .spacetimeformer.spacetimeformer.spacetimeformer_model.nn.extra_layers import (
    Normalization,
    ConvBlock,
    Flatten,
    Localize,
    ReverseLocalize,
    WindowTime,
    ReverseWindowTime,
    MakeSelfMaskFromSeq,
    MakeCrossMaskFromSeq,
)

import wandb

warnings.filterwarnings('ignore', category=UserWarning, module='pandas.core.computation.expressions')
torch.set_float32_matmul_precision('medium')

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
    
# INFO: Decoder class for STTRE [TODO: Implement]
class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, emb_dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm_layer = norm_layer
        self.emb_dropout = nn.Dropout(emb_dropout)

    def forward(
        self,
        val_time_emb,
        space_emb,
        cross,
        self_mask_seq=None,
        cross_mask_seq=None,
        output_cross_attn=False,
    ):
        x = self.emb_dropout(val_time_emb) + self.emb_dropout(space_emb)

        attns = []
        for i, layer in enumerate(self.layers):
            x, attn = layer(
                x,
                cross,
                self_mask_seq=self_mask_seq,
                cross_mask_seq=cross_mask_seq,
                output_cross_attn=output_cross_attn,
            )
            attns.append(attn)

        if self.norm_layer is not None:
            x = self.norm_layer(x)

        return x, attns
    
# INFO: Decoder layer for STTRE [TODO: Implement]
class DecoderLayer(nn.Module):
    def __init__(
        self,
        global_self_attention,
        local_self_attention,
        global_cross_attention,
        local_cross_attention,
        d_model,
        d_yt,
        d_yc,
        time_windows=1,
        time_window_offset=0,
        d_ff=None,
        dropout_ff=0.1,
        dropout_attn_out=0.0,
        activation="relu",
        norm="layer",
    ):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.local_self_attention = local_self_attention
        self.global_self_attention = global_self_attention
        self.global_cross_attention = global_cross_attention
        if local_cross_attention is not None and d_yc != d_yt:
            assert d_yt < d_yc
            warnings.warn(
                "The implementation of Local Cross Attn with exogenous variables \n\
                makes an unintuitive assumption about variable order. Please see \n\
                spacetimeformer_model.nn.decoder.DecoderLayer source code and comments"
            )
            """
            The unintuitive part is that if there are N variables in the context
            sequence (the encoder input) and K (K < N) variables in the target sequence
            (the decoder input), then this implementation of Local Cross Attn
            assumes that the first K variables in the context correspond to the
            first K in the target. This means that if the context sequence is shape 
            (batch, length, N), then context[:, :, :K] gets you the context of the
            K target variables (target[..., i] is the same variable
            as context[..., i]). If this isn't true the model will still train but
            you will be connecting variables by cross attention in a very arbitrary
            way. Note that the built-in CSVDataset *does* account for this and always
            puts the target variables in the same order in the lowest indices of both
            sequences. ** If your target variables do not appear in the context sequence
            Local Cross Attention should almost definitely be turned off
            (--local_cross_attn none) **.
            """

        self.local_cross_attention = local_cross_attention

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        self.norm1 = Normalization(method=norm, d_model=d_model)
        self.norm2 = Normalization(method=norm, d_model=d_model)
        self.norm3 = Normalization(method=norm, d_model=d_model)
        self.norm4 = Normalization(method=norm, d_model=d_model)
        self.norm5 = Normalization(method=norm, d_model=d_model)

        self.dropout_ff = nn.Dropout(dropout_ff)
        self.dropout_attn_out = nn.Dropout(dropout_attn_out)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.time_windows = time_windows
        self.time_window_offset = time_window_offset
        self.d_yt = d_yt
        self.d_yc = d_yc

    def forward(
        self, x, cross, self_mask_seq=None, cross_mask_seq=None, output_cross_attn=False
    ):
        # pre-norm Transformer architecture
        attn = None
        if self.local_self_attention:
            # self attention on each variable in target sequence ind.
            assert self_mask_seq is None
            x1 = self.norm1(x)
            x1 = Localize(x1, self.d_yt)
            x1, _ = self.local_self_attention(x1, x1, x1, attn_mask=self_mask_seq)
            x1 = ReverseLocalize(x1, self.d_yt)
            x = x + self.dropout_attn_out(x1)

        if self.global_self_attention:
            x1 = self.norm2(x)
            x1 = WindowTime(
                x1,
                dy=self.d_yt,
                windows=self.time_windows,
                window_offset=self.time_window_offset,
            )
            self_mask_seq = WindowTime(
                self_mask_seq,
                dy=self.d_yt,
                windows=self.time_windows,
                window_offset=self.time_window_offset,
            )
            x1, _ = self.global_self_attention(
                x1,
                x1,
                x1,
                attn_mask=MakeSelfMaskFromSeq(self_mask_seq),
            )
            x1 = ReverseWindowTime(
                x1,
                dy=self.d_yt,
                windows=self.time_windows,
                window_offset=self.time_window_offset,
            )
            self_mask_seq = ReverseWindowTime(
                self_mask_seq,
                dy=self.d_yt,
                windows=self.time_windows,
                window_offset=self.time_window_offset,
            )
            x = x + self.dropout_attn_out(x1)

        if self.local_cross_attention:
            # cross attention between target/context on each variable ind.
            assert cross_mask_seq is None
            x1 = self.norm3(x)
            bs, *_ = x1.shape
            x1 = Localize(x1, self.d_yt)
            # see above warnings and explanations about a potential
            # silent bug here.
            cross_local = Localize(cross, self.d_yc)[: self.d_yt * bs]
            x1, _ = self.local_cross_attention(
                x1,
                cross_local,
                cross_local,
                attn_mask=cross_mask_seq,
            )
            x1 = ReverseLocalize(x1, self.d_yt)
            x = x + self.dropout_attn_out(x1)

        if self.global_cross_attention:
            x1 = self.norm4(x)
            x1 = WindowTime(
                x1,
                dy=self.d_yt,
                windows=self.time_windows,
                window_offset=self.time_window_offset,
            )
            cross = WindowTime(
                cross,
                dy=self.d_yc,
                windows=self.time_windows,
                window_offset=self.time_window_offset,
            )
            cross_mask_seq = WindowTime(
                cross_mask_seq,
                dy=self.d_yc,
                windows=self.time_windows,
                window_offset=self.time_window_offset,
            )
            x1, attn = self.global_cross_attention(
                x1,
                cross,
                cross,
                attn_mask=MakeCrossMaskFromSeq(self_mask_seq, cross_mask_seq),
                output_attn=output_cross_attn,
            )
            cross = ReverseWindowTime(
                cross,
                dy=self.d_yc,
                windows=self.time_windows,
                window_offset=self.time_window_offset,
            )
            cross_mask_seq = ReverseWindowTime(
                cross_mask_seq,
                dy=self.d_yc,
                windows=self.time_windows,
                window_offset=self.time_window_offset,
            )
            x1 = ReverseWindowTime(
                x1,
                dy=self.d_yt,
                windows=self.time_windows,
                window_offset=self.time_window_offset,
            )
            x = x + self.dropout_attn_out(x1)

        x1 = self.norm5(x)
        # feedforward layers as 1x1 convs
        x1 = self.dropout_ff(self.activation(self.conv1(x1.transpose(-1, 1))))
        x1 = self.dropout_ff(self.conv2(x1).transpose(-1, 1))
        output = x + x1

        return output, attn

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
        
        # Decoder embedding components [TODO: Implement]
        self.decoder_embedding = nn.Linear()
        
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
        
        # Decoder components [TODO: Implement]
        self.decoder = Decoder(
            layers=[DecoderLayer(**decoder_layer_params) for decoder_layer_params in model_params['decoder_layers']],
            norm_layer=model_params['norm_layer'],
            emb_dropout=model_params['emb_dropout']
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

class STTREModel(BaseModel):
    """STTRE model implementation"""
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
    
    def _setup_trainer(self, dataset, experiment_params, callbacks, model_params, train_params, local_rank):
        wandb_logger = None
        if local_rank == 0:
            wandb_logger = WandbLogger(
                project=self.__class__.__name__,
                name=experiment_params["run_name"],
                log_model=True,
                save_dir=Config.SAVE_DIR,
                config={
                    "model_params": model_params,
                    "train_params": train_params,
                    "dataset": dataset.__class__.__name__,
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

    def _setup_callbacks(self, dataset_class, local_rank, config):
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

        filename = f"{config.experiment.run_name}_" + str(uuid.uuid1()).split("-")[0]
        model_ckpt_dir = os.path.join(config.log_dir, filename)
        config.experiment.model_ckpt_dir = model_ckpt_dir 
        checkpoint_callback = ThresholdModelCheckpoint(
            loss_threshold=50.0,
            monitor='val/loss',
            dirpath=model_ckpt_dir,
            filename=f"{config.experiment.run_name}" + "{epoch:02d}", # f'sttre-{dataset_class.__name__.lower()}-' + 'epoch{epoch:03d}-loss{val/loss:.4f}',
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
    model = STTREModel()
    
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
        # 'Uber': (Uber, os.path.join(Config.DATA_DIR, 'uber_stock.csv')),
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
            
            model, trainer, test_results = model.train(
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
            model.cleanup_memory()
            
        except Exception as e:
            if local_rank == 0:
                print(f"{Colors.RED}Error processing {dataset_name}: {str(e)} {Colors.CROSS}{Colors.ENDC}")
            model.cleanup_memory()
            continue

    if local_rank == 0:
        print(f"\n{Colors.BOLD_GREEN}All experiments completed! {Colors.CHECK}{Colors.ENDC}")