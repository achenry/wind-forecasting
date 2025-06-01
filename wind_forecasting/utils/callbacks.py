import logging
import torch
import torch.nn as nn
import lightning.pytorch as pl
from typing import Dict, List, Optional, Tuple, Any

try:
    from pytorch_transformer_ts.tactis_2.attentional_copula import AttentionalCopula
except ImportError:
    AttentionalCopula = type('AttentionalCopula', (), {})
    logging.warning("Could not import AttentionalCopula.")

logger = logging.getLogger(__name__)


ACTIVATION_TYPES: Tuple[type, ...] = (nn.ReLU, nn.GELU, nn.SiLU, nn.Mish, nn.LeakyReLU)

class DeadNeuronMonitor(pl.Callback):
    """
    A PyTorch Lightning callback that monitors the percentage of dead neurons
    (neurons with zero activations) in the feed-forward networks of the AttentionalCopula
    component in TACTiS-2 models.
    """
    
    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        self.activation_outputs: Dict[str, List[torch.Tensor]] = {}
        self.hook_handles: List[Any] = []
        
        tactis_module = getattr(pl_module.model, 'tactis', None)
        if tactis_module is None:
            logger.debug("DeadNeuronMonitor: TACTiS model not found at pl_module.model.tactis. Skipping.")
            return

        logger.debug("Initializing DeadNeuronMonitor...")

        # --- Helper function to register hooks ---
        def register_hook_if_activation(module: nn.Module, key_prefix: str, index: Optional[int] = None):
            """Registers hook if module is an activation layer and key doesn't exist."""
            if isinstance(module, ACTIVATION_TYPES):
                layer_key = f"{key_prefix}_activation"
                if index is not None:
                    layer_key += f"_{index}"
                
                if layer_key in self.activation_outputs: # Avoid duplicate keys
                    # Log only once per key to avoid spamming logs if structure is complex
                    if not hasattr(self, '_logged_duplicates'):
                        self._logged_duplicates = set()
                    if layer_key not in self._logged_duplicates:
                        logger.warning(f"Duplicate layer key detected: {layer_key}. Skipping registration.")
                        self._logged_duplicates.add(layer_key)
                    return False

                self.activation_outputs[layer_key] = []
                handle = module.register_forward_hook(self._create_hook(layer_key))
                self.hook_handles.append(handle)
                logger.debug(f"Registered hook for {layer_key} (type: {type(module).__name__})")
                return True # Indicate that a hook was registered
            return False

        # Reset logged duplicates tracker at each setup
        if hasattr(self, '_logged_duplicates'):
             del self._logged_duplicates

        # --- Monitor Flow Input Encoder ---
        flow_input_encoder = getattr(tactis_module, 'flow_input_encoder', None)
        if isinstance(flow_input_encoder, nn.Sequential):
            logger.debug("Monitoring Flow Input Encoder activations...")
            for i, layer in enumerate(flow_input_encoder):
                register_hook_if_activation(layer, "flow_input_encoder", i)
        elif flow_input_encoder is not None:
             logger.debug(f"Flow Input Encoder is not nn.Sequential (type: {type(flow_input_encoder).__name__}). Skipping detailed monitoring.")


        # --- Monitor Flow Encoder ---
        flow_encoder = getattr(tactis_module, 'flow_encoder', None)
        if isinstance(flow_encoder, nn.TransformerEncoder):
            logger.debug("Monitoring Flow Encoder activations...")
            for i, encoder_layer in enumerate(flow_encoder.layers):
                 ffn_activation_found = False
                 # Try common attribute names first (more reliable if present)
                 potential_activations = [getattr(encoder_layer, name, None) for name in ['activation', 'relu', 'gelu']]
                 for act_layer in potential_activations:
                     if act_layer is not None and register_hook_if_activation(act_layer, f"flow_encoder_layer_{i}_ffn"):
                         ffn_activation_found = True
                         break
                 
                 # If not found by common names, iterate through modules after norm2 (common pattern)
                 if not ffn_activation_found:
                     norm2 = getattr(encoder_layer, 'norm2', None)
                     if norm2:
                         try:
                             # Find modules after norm2 within the encoder_layer's direct children
                             children = list(encoder_layer.children())
                             norm2_idx = children.index(norm2)
                             modules_after_norm2 = children[norm2_idx + 1:]
                             
                             # Look for the first activation layer in the FFN part
                             for module in modules_after_norm2:
                                 if isinstance(module, ACTIVATION_TYPES):
                                     if register_hook_if_activation(module, f"flow_encoder_layer_{i}_ffn"):
                                         ffn_activation_found = True
                                         break # Register only the first activation in FFN
                                 # Stop searching if we encounter the next major block (like another norm or attention)
                                 # Or if we hit the final linear layer of the FFN
                                 elif isinstance(module, (nn.LayerNorm, nn.MultiheadAttention, nn.Linear)):
                                     # Check if the previous module was the activation's linear layer
                                     # This is heuristic, assumes Linear -> Activation -> Dropout -> Linear
                                     if len(modules_after_norm2) > 1 and module == modules_after_norm2[-1] and isinstance(modules_after_norm2[-2], nn.Dropout):
                                         pass # Allow searching past dropout if it's before the last linear
                                     else:
                                         break
                             
                         except (ValueError, AttributeError) as e:
                             # This can happen if norm2 is not a direct child or structure differs
                             logger.debug(f"Could not reliably find modules after norm2 for flow_encoder layer {i}: {e}. Trying iteration.")
                             # Fallback: Iterate all modules in the layer (less precise)
                             for module in encoder_layer.modules():
                                 if module is not encoder_layer and register_hook_if_activation(module, f"flow_encoder_layer_{i}_ffn"):
                                     ffn_activation_found = True
                                     logger.debug("Found activation via fallback iteration.")
                                     break # Take the first one found this way

                 if not ffn_activation_found:
                     logger.debug(f"Could not find activation in FFN for flow_encoder layer {i}. Skipping.")

        elif flow_encoder is not None:
            logger.debug(f"Flow Encoder is not nn.TransformerEncoder (type: {type(flow_encoder).__name__}). Skipping detailed monitoring.")


        # --- Monitor Marginal Conditioner ---
        decoder = getattr(tactis_module, 'decoder', None)
        marginal = getattr(decoder, 'marginal', None)
        marginal_conditioner = getattr(marginal, 'marginal_conditioner', None)
        if isinstance(marginal_conditioner, nn.Sequential):
            logger.debug("Monitoring Marginal Conditioner activations...")
            for i, layer in enumerate(marginal_conditioner):
                register_hook_if_activation(layer, "marginal_conditioner", i)
        elif marginal_conditioner is not None:
             logger.debug(f"Marginal Conditioner is not nn.Sequential (type: {type(marginal_conditioner).__name__}). Skipping detailed monitoring.")


        # --- Monitor Copula Input Encoder (Optional) ---
        copula_input_encoder = getattr(tactis_module, 'copula_input_encoder', None)
        if isinstance(copula_input_encoder, nn.Sequential):
            logger.debug("Monitoring Copula Input Encoder activations...")
            for i, layer in enumerate(copula_input_encoder):
                register_hook_if_activation(layer, "copula_input_encoder", i)
        elif copula_input_encoder is not None:
             logger.debug(f"Copula Input Encoder is not nn.Sequential (type: {type(copula_input_encoder).__name__}). Skipping detailed monitoring.")


        # --- Monitor Copula Encoder (Optional) ---
        copula_encoder = getattr(tactis_module, 'copula_encoder', None)
        if isinstance(copula_encoder, nn.TransformerEncoder):
            logger.debug("Monitoring Copula Encoder activations...")
            for i, encoder_layer in enumerate(copula_encoder.layers):
                 ffn_activation_found = False
                 potential_activations = [getattr(encoder_layer, name, None) for name in ['activation', 'relu', 'gelu']]
                 for act_layer in potential_activations:
                     if act_layer is not None and register_hook_if_activation(act_layer, f"copula_encoder_layer_{i}_ffn"):
                         ffn_activation_found = True
                         break
                 
                 if not ffn_activation_found:
                     norm2 = getattr(encoder_layer, 'norm2', None)
                     if norm2:
                         try:
                             children = list(encoder_layer.children())
                             norm2_idx = children.index(norm2)
                             modules_after_norm2 = children[norm2_idx + 1:]
                             for module in modules_after_norm2:
                                 if isinstance(module, ACTIVATION_TYPES):
                                     if register_hook_if_activation(module, f"copula_encoder_layer_{i}_ffn"):
                                         ffn_activation_found = True
                                         break
                                 elif isinstance(module, (nn.LayerNorm, nn.MultiheadAttention, nn.Linear)):
                                     if len(modules_after_norm2) > 1 and module == modules_after_norm2[-1] and isinstance(modules_after_norm2[-2], nn.Dropout):
                                         pass
                                     else:
                                         break
                         except (ValueError, AttributeError) as e:
                             logger.debug(f"Could not reliably find modules after norm2 for copula_encoder layer {i}: {e}. Trying iteration.")
                             for module in encoder_layer.modules():
                                 if module is not encoder_layer and register_hook_if_activation(module, f"copula_encoder_layer_{i}_ffn"):
                                     ffn_activation_found = True
                                     logger.debug("Found activation via fallback iteration.")
                                     break
                                 
                 if not ffn_activation_found:
                     logger.debug(f"Could not find activation in FFN for copula_encoder layer {i}. Skipping.")
                     
        elif copula_encoder is not None:
            logger.debug(f"Copula Encoder is not nn.TransformerEncoder (type: {type(copula_encoder).__name__}). Skipping detailed monitoring.")


        # --- Monitor Attentional Copula (Existing logic adapted) ---
        # Ensure decoder is fetched if not already done (might be None if marginal wasn't found)
        if decoder is None:
            decoder = getattr(tactis_module, 'decoder', None)
            
        copula = getattr(decoder, 'copula', None)
        # Check base class if specific import failed
        if AttentionalCopula.__name__ != 'AttentionalCopula' and not isinstance(copula, AttentionalCopula):
             # If the placeholder was used, we can't be sure, so skip.
             if AttentionalCopula.__name__ == 'AttentionalCopula':
                 logger.debug("AttentionalCopula type check skipped as import failed.")
             else: # Only log if the type is definitely wrong and not the placeholder
                 logger.debug(f"Decoder Copula is not AttentionalCopula (type: {type(copula).__name__}). Skipping AttentionalCopula monitoring.")
        elif isinstance(copula, AttentionalCopula):
             logger.debug("Monitoring AttentionalCopula activations...")
             # Ensure feed_forwards exists and is iterable
             feed_forwards = getattr(copula, 'feed_forwards', None)
             if isinstance(feed_forwards, nn.ModuleList):
                 for i, ff_block in enumerate(feed_forwards):
                     activation_registered_in_block = False
                     # Check if ff_block is iterable (like nn.Sequential)
                     if hasattr(ff_block, '__iter__') and not isinstance(ff_block, torch.Tensor): # Check it's not a tensor
                         for layer in ff_block:
                             if register_hook_if_activation(layer, f"attentional_copula_ff_{i}"):
                                 activation_registered_in_block = True
                                 break # Only monitor the first activation function in each block
                     elif isinstance(ff_block, nn.Module): # Handle case where ff_block might be a single layer/module
                          if register_hook_if_activation(ff_block, f"attentional_copula_ff_{i}"):
                              activation_registered_in_block = True

                     if not activation_registered_in_block:
                          logger.debug(f"No activation layer found or registered in AttentionalCopula feed_forward block {i}.")
             else:
                 logger.debug("AttentionalCopula found, but 'feed_forwards' attribute is missing or not a ModuleList/iterable.")
        # No need for an else here, covers cases where copula is None or not AttentionalCopula (and import succeeded)

        if not self.hook_handles:
             logger.warning("DeadNeuronMonitor: No activation layers found to monitor in the specified components.")
    
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