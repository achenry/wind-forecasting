#!/usr/bin/env python3
"""
Comprehensive analysis of TACTiS checkpoint to understand training stage and probabilistic capabilities.
"""

import torch
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def analyze_tactis_checkpoint(checkpoint_path: str):
    """
    Analyze a TACTiS checkpoint to understand its training stage and copula capabilities.
    
    Args:
        checkpoint_path: Path to the checkpoint file
    """
    print("="*80)
    print("TACTIS CHECKPOINT ANALYSIS")
    print("="*80)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✓ Successfully loaded checkpoint: {checkpoint_path}")
    except Exception as e:
        print(f"✗ Failed to load checkpoint: {e}")
        return
    
    # Basic checkpoint info
    print(f"\n📋 BASIC CHECKPOINT INFO:")
    print(f"   Current epoch: {checkpoint['epoch']}")
    print(f"   Global step: {checkpoint['global_step']}")
    print(f"   Lightning version: {checkpoint.get('pytorch-lightning_version', 'Unknown')}")
    
    # Analyze hyperparameters
    hyper_params = checkpoint.get('hyper_parameters', {})
    print(f"\n🎛️  TRAINING CONFIGURATION:")
    print(f"   Stage 2 start epoch: {hyper_params.get('stage2_start_epoch', 'Not found')}")
    print(f"   Recorded stage: {hyper_params.get('stage', 'Not found')}")
    print(f"   Should be in Stage 2: {checkpoint['epoch'] >= hyper_params.get('stage2_start_epoch', float('inf'))}")
    print(f"   Skip copula flag: {hyper_params.get('skip_copula', 'NOT FOUND')}")
    
    # Model configuration
    model_config = hyper_params.get('model_config', {})
    print(f"\n🏗️  MODEL CONFIGURATION:")
    copula_keys = [k for k in model_config.keys() if 'copula' in k.lower()]
    for key in copula_keys:
        print(f"   {key}: {model_config[key]}")
    
    # Analyze state dict
    state_dict = checkpoint['state_dict']
    print(f"\n🧠 STATE DICT ANALYSIS:")
    print(f"   Total parameters: {len(state_dict)}")
    
    # Count parameter types
    flow_keys = [k for k in state_dict.keys() if 'flow' in k.lower()]
    copula_keys = [k for k in state_dict.keys() if 'copula' in k.lower()]
    marginal_keys = [k for k in state_dict.keys() if 'marginal' in k.lower()]
    
    print(f"   Flow parameters: {len(flow_keys)}")
    print(f"   Copula parameters: {len(copula_keys)}")
    print(f"   Marginal parameters: {len(marginal_keys)}")
    
    # Analyze key copula parameters
    print(f"\n🔍 COPULA PARAMETER ANALYSIS:")
    key_copula_params = [
        'model.tactis.decoder.copula.dist_extractors.weight',
        'model.tactis.decoder.copula.dist_extractors.bias',
        'model.tactis.decoder.copula.dimension_shifting_layer.weight',
        'model.tactis.copula_input_encoder.0.weight'
    ]
    
    for param_name in key_copula_params:
        if param_name in state_dict:
            param = state_dict[param_name]
            mean_val = param.mean().item()
            std_val = param.std().item()
            min_val = param.min().item()
            max_val = param.max().item()
            
            print(f"   {param_name.split('.')[-2]}.{param_name.split('.')[-1]}:")
            print(f"     Shape: {param.shape}")
            print(f"     Stats: mean={mean_val:.6f}, std={std_val:.6f}")
            print(f"     Range: [{min_val:.6f}, {max_val:.6f}]")
            
            # Check training status
            if torch.allclose(param, torch.zeros_like(param), atol=1e-6):
                status = "❌ All zeros - likely not trained"
            elif std_val < 1e-4:
                status = "⚠️  Very small variance - possibly not well trained"
            else:
                status = "✅ Non-zero with reasonable variance - likely trained"
            print(f"     Status: {status}")
        else:
            print(f"   {param_name}: NOT FOUND")
    
    # Analyze optimizer states
    print(f"\n⚙️  OPTIMIZER ANALYSIS:")
    opt_states = checkpoint.get('optimizer_states', [])
    if opt_states:
        first_opt = opt_states[0]
        opt_state = first_opt.get('state', {})
        print(f"   Number of optimized parameters: {len(opt_state)}")
        
        # Estimate parameter distribution
        param_sizes = []
        for param_state in opt_state.values():
            if 'exp_avg' in param_state:
                param_sizes.append(param_state['exp_avg'].numel())
        
        if param_sizes:
            print(f"   Parameter size distribution:")
            print(f"     Small params (<1000): {sum(1 for s in param_sizes if s < 1000)}")
            print(f"     Medium params (1000-10000): {sum(1 for s in param_sizes if 1000 <= s < 10000)}")
            print(f"     Large params (>=10000): {sum(1 for s in param_sizes if s >= 10000)}")
    
    # Check LR schedulers
    print(f"\n📈 LEARNING RATE SCHEDULER:")
    lr_schedulers = checkpoint.get('lr_schedulers', [])
    if lr_schedulers:
        scheduler = lr_schedulers[0]
        print(f"   Last epoch: {scheduler.get('last_epoch', 'Unknown')}")
        print(f"   Step count: {scheduler.get('_step_count', 'Unknown')}")
        print(f"   Last LR: {scheduler.get('_last_lr', 'Unknown')}")
    
    # Training progress analysis
    print(f"\n🏃 TRAINING PROGRESS:")
    loops = checkpoint.get('loops', {})
    fit_loop = loops.get('fit_loop', {})
    epoch_progress = fit_loop.get('epoch_progress', {})
    batch_progress = fit_loop.get('epoch_loop.batch_progress', {})
    
    if epoch_progress:
        print(f"   Epoch progress: {epoch_progress}")
    if batch_progress:
        print(f"   Batch progress: {batch_progress}")
    
    # Final diagnosis
    print(f"\n🩺 DIAGNOSIS:")
    epoch = checkpoint['epoch']
    stage2_start = hyper_params.get('stage2_start_epoch', 0)
    recorded_stage = hyper_params.get('stage', 1)
    skip_copula_in_hparams = 'skip_copula' in hyper_params
    copula_params_exist = len(copula_keys) > 0
    
    if epoch >= stage2_start and recorded_stage == 1:
        print(f"   ⚠️  ISSUE: Model should be in Stage 2 (epoch {epoch} >= {stage2_start}) but recorded stage is {recorded_stage}")
    
    if not skip_copula_in_hparams:
        print(f"   ⚠️  ISSUE: 'skip_copula' flag not found in hyperparameters")
        print(f"       This suggests the model may not have proper stage transition logic")
    
    if copula_params_exist:
        print(f"   ✅ Copula parameters exist and appear to be trained")
    else:
        print(f"   ❌ No copula parameters found in checkpoint")
    
    # Key recommendation
    print(f"\n💡 RECOMMENDATION:")
    if epoch >= stage2_start and copula_params_exist:
        print(f"   The model appears to have trained copula parameters despite the recorded stage being 1.")
        print(f"   This suggests the stage transition occurred during training but wasn't properly recorded.")
        print(f"   The probabilistic uncertainty issue may be due to:")
        print(f"   1. Model still using skip_copula=True during inference")
        print(f"   2. Copula parameters not being used even though they're trained")
        print(f"   3. Need to explicitly set model.tactis.skip_copula=False for inference")
    else:
        print(f"   Model may need proper stage 2 training to enable probabilistic capabilities.")
    
    print("="*80)

if __name__ == "__main__":
    checkpoint_path = "/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/tune_tactis_flasc_3_local_tactis/20250722_210704_0_0/epoch=72-step=716422-val_loss=-41.45.ckpt"
    analyze_tactis_checkpoint(checkpoint_path)