#!/usr/bin/env python3
"""
Test script to load TACTiS checkpoint and verify its inference behavior and skip_copula state.
"""

import torch
import numpy as np
import sys
import os
from pathlib import Path

# Add the necessary paths to sys.path
sys.path.append('/fs/dss/home/taed7566/Forecasting/pytorch-transformer-ts')
sys.path.append('/fs/dss/home/taed7566/Forecasting/wind-forecasting')

# Import TACTiS components
try:
    from pytorch_transformer_ts.tactis_2.lightning_module import TACTiS2LightningModule
    from pytorch_transformer_ts.tactis_2.module import TACTiS2Model
    print("✓ Successfully imported TACTiS modules")
except ImportError as e:
    print(f"✗ Failed to import TACTiS modules: {e}")
    sys.exit(1)

def test_tactis_model_state(checkpoint_path: str):
    """
    Load TACTiS model from checkpoint and test its inference state.
    """
    print("="*80)
    print("TACTIS MODEL INFERENCE STATE TEST")
    print("="*80)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✓ Loaded checkpoint from: {checkpoint_path}")
    except Exception as e:
        print(f"✗ Failed to load checkpoint: {e}")
        return
    
    # Load the Lightning module
    try:
        # Extract model configuration from hyperparameters
        hyper_params = checkpoint['hyper_parameters']
        
        # Create Lightning module
        lightning_module = TACTiS2LightningModule.load_from_checkpoint(
            checkpoint_path,
            map_location='cpu',
            strict=False
        )
        print("✓ Successfully loaded Lightning module")
        
        # Access the underlying model
        model = lightning_module.model
        tactis = model.tactis
        
        print(f"\n🔍 MODEL STATE INSPECTION:")
        print(f"   Model type: {type(model).__name__}")
        print(f"   TACTiS type: {type(tactis).__name__}")
        print(f"   Model stage: {getattr(tactis, 'stage', 'Not found')}")
        print(f"   Skip copula: {getattr(tactis, 'skip_copula', 'Not found')}")
        
        # Check if copula components exist
        has_copula_encoder = hasattr(tactis, 'copula_series_encoder')
        has_copula_decoder = hasattr(tactis.decoder, 'copula') if hasattr(tactis, 'decoder') else False
        
        print(f"   Has copula encoder: {has_copula_encoder}")
        print(f"   Has copula decoder: {has_copula_decoder}")
        
        if has_copula_encoder:
            print(f"   Copula encoder device: {next(tactis.copula_series_encoder.parameters()).device}")
        
        # Test inference behavior
        print(f"\n🧪 INFERENCE BEHAVIOR TEST:")
        
        # Create dummy input data
        batch_size = 2
        num_series = 2
        context_length = 72
        prediction_length = 12
        
        # Create dummy historical data
        hist_time = torch.arange(context_length).float().unsqueeze(0).repeat(batch_size, 1)
        hist_value = torch.randn(batch_size, num_series, context_length)
        pred_time = torch.arange(context_length, context_length + prediction_length).float().unsqueeze(0).repeat(batch_size, 1)
        
        print(f"   Input shapes:")
        print(f"     hist_time: {hist_time.shape}")
        print(f"     hist_value: {hist_value.shape}")
        print(f"     pred_time: {pred_time.shape}")
        
        # Set model to eval mode
        model.eval()
        
        with torch.no_grad():
            try:
                # Test multiple samples
                num_samples = 200
                samples = []
                
                for i in range(num_samples):
                    # Single sample inference
                    output, _ = tactis(hist_time, hist_value, pred_time)
                    samples.append(output.cpu().numpy())
                
                samples = np.array(samples)  # [num_samples, batch, series, pred_len]
                
                print(f"\n📊 SAMPLE STATISTICS:")
                print(f"   Sample shape: {samples.shape}")
                
                # Calculate statistics across samples
                sample_mean = np.mean(samples, axis=0)
                sample_std = np.std(samples, axis=0)
                
                print(f"   Mean across samples - min: {sample_mean.min():.6f}, max: {sample_mean.max():.6f}")
                print(f"   Std across samples - min: {sample_std.min():.6f}, max: {sample_std.max():.6f}")
                
                # Check if samples are actually different
                sample_diff = np.max(samples, axis=0) - np.min(samples, axis=0)
                print(f"   Sample range (max-min) - min: {sample_diff.min():.6f}, max: {sample_diff.max():.6f}")
                
                if sample_diff.max() < 1e-6:
                    print(f"   ❌ ISSUE: All samples are nearly identical! Probabilistic sampling not working.")
                    print(f"   This confirms the skip_copula=True behavior.")
                else:
                    print(f"   ✅ Samples show variation - probabilistic sampling appears to work.")
                
                # Test with explicitly setting skip_copula=False
                print(f"\n🔧 TESTING WITH skip_copula=False:")
                original_skip_copula = tactis.skip_copula
                original_stage = tactis.stage
                
                # Force enable copula
                tactis.skip_copula = False
                tactis.stage = 2
                if hasattr(tactis.decoder, 'skip_copula'):
                    tactis.decoder.skip_copula = False
                    
                print(f"   Set skip_copula={tactis.skip_copula}, stage={tactis.stage}")
                
                # Test again with copula enabled
                samples_copula = []
                for i in range(10):  # Fewer samples for quick test
                    output, _ = tactis(hist_time, hist_value, pred_time)
                    samples_copula.append(output.cpu().numpy())
                
                samples_copula = np.array(samples_copula)
                sample_diff_copula = np.max(samples_copula, axis=0) - np.min(samples_copula, axis=0)
                
                print(f"   With copula enabled - sample range: {sample_diff_copula.min():.6f} to {sample_diff_copula.max():.6f}")
                
                if sample_diff_copula.max() > sample_diff.max():
                    print(f"   ✅ Copula enabled shows more variation!")
                else:
                    print(f"   ⚠️  No significant improvement with copula enabled")
                
                # Restore original settings
                tactis.skip_copula = original_skip_copula
                tactis.stage = original_stage
                if hasattr(tactis.decoder, 'skip_copula'):
                    tactis.decoder.skip_copula = original_skip_copula
                    
            except Exception as e:
                print(f"   ✗ Inference test failed: {e}")
                import traceback
                traceback.print_exc()
        
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
    
    print("="*80)

if __name__ == "__main__":
    checkpoint_path = "/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/tune_tactis_flasc_3_local_tactis/20250722_210704_0_0/epoch=72-step=716422-val_loss=-41.45.ckpt"
    test_tactis_model_state(checkpoint_path)