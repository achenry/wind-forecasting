import logging
import torch
import torch.nn as nn
import lightning.pytorch as pl
from typing import Dict, List

try:
    from pytorch_transformer_ts.tactis_2.attentional_copula import AttentionalCopula
except ImportError:
    AttentionalCopula = type('AttentionalCopula', (), {})
    logging.warning("Could not import AttentionalCopula.")

logger = logging.getLogger(__name__)


class DeadNeuronMonitor(pl.Callback):
    """
    A PyTorch Lightning callback that monitors the percentage of dead neurons
    (neurons with zero activations) in the feed-forward networks of the AttentionalCopula
    component in TACTiS-2 models.
    """
    
    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        # Initialize storage for activation outputs and hook handles
        self.activation_outputs = {}
        self.hook_handles = []
        
        tactis_module = getattr(pl_module.model, 'tactis', None)
        decoder = getattr(tactis_module, 'decoder', None)
        copula = getattr(decoder, 'copula', None)
        
        # Check if we found an AttentionalCopula
        if isinstance(copula, AttentionalCopula):
            logger.debug("Initializing DeadNeuronMonitor for AttentionalCopula.")
            
            # Iterate through feed-forward blocks in the copula
            for i, ff_block in enumerate(copula.feed_forwards):
                for layer_idx, layer in enumerate(ff_block):
                    if isinstance(layer, (nn.ReLU, nn.GELU, nn.SiLU, nn.Mish, nn.LeakyReLU)):
                        layer_key = f"copula_ff_{i}_activation"
                        # Initialize storage for activations from this layer
                        self.activation_outputs[layer_key] = []
                        
                        # Register forward hook to capture activations
                        handle = layer.register_forward_hook(self._create_hook(layer_key))
                        self.hook_handles.append(handle)
                        
                        logger.debug(f"Registered hook for {layer_key} (layer type: {type(layer).__name__})")
                        # Only monitor the first activation function in each block
                        break
        else:
            logger.debug("DeadNeuronMonitor: AttentionalCopula not found or not of correct type. "
                         "Skipping activation monitoring for this model.")
    
    def _create_hook(self, layer_key: str):
        def hook(module, input, output):
            # Detach and move to CPU
            self.activation_outputs[layer_key].append(output.detach().cpu())
        return hook
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Calculate and log the percentage of dead neurons at the end of each validation epoch.
        
        Args:
            trainer: The PyTorch Lightning trainer
            pl_module: The Lightning module being monitored
        """
        if not hasattr(self, 'activation_outputs') or not self.activation_outputs:
            return
            
        # Process each layer's activations
        for layer_key, outputs_list in self.activation_outputs.items():
            if outputs_list:
                try:
                    all_outputs = torch.cat(outputs_list, dim=0)
                    
                    # Calculate percentage of dead neurons (those that never activated)
                    dead_percentage = (all_outputs == 0).float().mean().item() * 100
                    
                    pl_module.log(
                        f"debug/dead_neurons/{layer_key}_percent", 
                        dead_percentage, 
                        sync_dist=True
                    )
                    
                    logger.debug(f"{layer_key} dead neurons: {dead_percentage:.2f}%")
                except Exception as e:
                    logger.warning(f"Error processing activations for {layer_key}: {e}")
                
                self.activation_outputs[layer_key] = []
    
    def teardown(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        # Remove all hooks
        if hasattr(self, 'hook_handles'):
            for handle in self.hook_handles:
                handle.remove()
            self.hook_handles = []
            logger.debug("DeadNeuronMonitor: Removed all hooks")