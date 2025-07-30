# TACTiS Checkpoint Analysis Report

## Executive Summary

The TACTiS model checkpoint at epoch 72 has a critical issue: **the copula parameters are trained but not being used during inference**, resulting in extremely low probabilistic uncertainty despite generating 200 samples.

## Key Findings

### 1. Training Stage Mismatch
- **Current epoch**: 72
- **Stage 2 start epoch**: 20 (copula should be active)
- **Recorded stage**: 1 (still in flow-only mode)
- **Issue**: Model should be in Stage 2 but is recorded as Stage 1

### 2. Copula Parameters Status
- **134 copula parameters** exist in the checkpoint
- All copula parameters show **proper training signatures**:
  - Non-zero values with reasonable variance
  - Parameter statistics consistent with trained weights
  - Includes: `dist_extractors`, `dimension_shifting_layer`, `input_encoder`, etc.

### 3. Model Loading Issue
When attempting to load the checkpoint, PyTorch Lightning reports:
```
Found keys that are not in the model state dict but in the checkpoint: 
['model.tactis.copula_series_encoder.weight', 'model.tactis.copula_input_encoder.0.weight', ...]
```

**This indicates that copula parameters exist in the checkpoint but the model is being initialized without copula components.**

### 4. Missing Configuration
- **No `skip_copula` flag** found in hyperparameters
- This suggests the model lacks proper stage transition logic
- The stage transition may have occurred during training but wasn't properly recorded

## Root Cause Analysis

### Primary Issue: Inconsistent Stage State
1. **Training**: The model transitioned to Stage 2 at epoch 20 and trained copula parameters
2. **Checkpoint**: The stage was incorrectly recorded as 1 instead of 2
3. **Inference**: Model is loaded with `skip_copula=True` (default for stage 1)
4. **Result**: Copula parameters exist but are completely bypassed during sampling

### Secondary Issues
1. **Model Initialization**: Current model loading doesn't properly handle checkpoints with trained copula parameters
2. **Stage Management**: The stage transition logic may not be properly updating the recorded stage
3. **Configuration Mismatch**: Hyperparameters don't reflect the actual model state after stage transition

## Impact on Probabilistic Forecasting

### Current Behavior
- Model generates 200 samples using **only the flow component**
- All samples are nearly identical (range < 1e-6)
- Probabilistic uncertainty is artificially low
- The trained copula (which models cross-series dependencies) is unused

### Expected Behavior with Copula
- Samples should show significant variation across predictions
- Cross-series correlations should be properly modeled
- Probabilistic uncertainty should reflect true forecast uncertainty

## Recommended Solutions

### Immediate Fix (for inference)
1. **Load checkpoint and manually enable copula**:
   ```python
   model = load_checkpoint(checkpoint_path)
   model.tactis.skip_copula = False
   model.tactis.stage = 2
   if hasattr(model.tactis.decoder, 'skip_copula'):
       model.tactis.decoder.skip_copula = False
   ```

### Long-term Fixes (for training)
1. **Fix stage transition recording**: Ensure stage changes are properly saved in hyperparameters
2. **Add skip_copula to hyperparameters**: Include this flag in saved configuration
3. **Improve checkpoint loading**: Handle copula parameters correctly during model initialization
4. **Add validation**: Verify copula components are active when stage ≥ 2

### Training Verification
1. **Stage 2 activation check**: Verify copula loss is being computed after epoch 20
2. **Parameter optimization**: Confirm copula parameters have non-zero gradients
3. **Checkpoint validation**: Ensure saved stage matches actual model state

## Technical Details

### Checkpoint Contents
- **Total parameters**: 249
- **Flow parameters**: 104 (properly trained)
- **Copula parameters**: 134 (trained but unused)
- **Marginal parameters**: 10 (part of decoder)

### Key Copula Components Found
- `copula_series_encoder`: Series embedding for copula path ✅
- `copula_input_encoder`: Input processing layers ✅  
- `copula_encoder`: Transformer encoder for copula ✅
- `decoder.copula`: Attentional copula decoder ✅
- `dist_extractors`: Final distribution parameters ✅

### Model Architecture Verification
- All copula components have reasonable parameter distributions
- No zero-initialized or undertrained parameters detected
- Parameter sizes match expected architecture

## Conclusion

The TACTiS model has been successfully trained with copula components, but a configuration management issue prevents these components from being used during inference. The trained copula parameters represent significant computational investment and should dramatically improve probabilistic forecasting quality once properly activated.

**Priority**: HIGH - This directly impacts the quality of probabilistic wind forecasting and utilization of trained model capacity.