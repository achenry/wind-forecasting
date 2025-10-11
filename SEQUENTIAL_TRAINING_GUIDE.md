# Sequential Training Guide: TACTiS Stage 1→2

## Overview

This guide explains the new sequential training setup that combines both Stage 1 (marginals/flow/decoder) and Stage 2 (copula) training in a single job with automatic transition and parameter freezing.

## Best Hyperparameters from Tuning

### Stage 1 (Trial 157, val_loss: -193.864)
- **Common/Architectural:**
  - `context_length_factor: 25`
  - `encoder_type: temporal`
  - `batch_size: 64`
  - `dropout_rate: 0.008`

- **Marginal/Flow:**
  - `marginal_embedding_dim_per_head: 16`
  - `marginal_num_heads: 6`
  - `marginal_num_layers: 3`
  - `flow_input_encoder_layers: 4`
  - `flow_series_embedding_dim: 16`

- **Decoder:**
  - `decoder_dsf_num_layers: 3`
  - `decoder_dsf_hidden_dim: 512`
  - `decoder_mlp_num_layers: 5`
  - `decoder_mlp_hidden_dim: 48`
  - `decoder_transformer_num_layers: 3`
  - `decoder_transformer_embedding_dim_per_head: 32`
  - `decoder_transformer_num_heads: 3`
  - `decoder_num_bins: 300`

- **Optimizer:**
  - `lr_stage1: 0.000381`
  - `weight_decay_stage1: 0.0`
  - `gradient_clip_val_stage1: 0`
  - `eta_min_fraction_s1: 0.00395`

### Stage 2 (Trial 146, val_loss: -193.546)
- **Copula Architecture:**
  - `copula_embedding_dim_per_head: 16`
  - `copula_num_heads: 3`
  - `copula_num_layers: 1`
  - `copula_input_encoder_layers: 4`
  - `copula_series_embedding_dim: 32`
  - `ac_mlp_num_layers: 4`
  - `ac_mlp_dim: 256`

- **Optimizer:**
  - `lr_stage2: 0.000386`
  - `weight_decay_stage2: 2e-05`
  - `gradient_clip_val_stage2: 0`
  - `eta_min_fraction_s2: 0.00819`

## How Sequential Training Works

### Automatic Stage Transition Architecture

**★ Key Insight**: The model uses two mechanisms to ensure proper Stage 1→2 transition:

1. **Forward Pass Control**: Copula is only used when `skip_copula=false AND stage >= 2`
2. **Parameter Freezing**: At epoch 50, marginal/flow params are frozen with `requires_grad=False`

### Training Timeline (100 epochs total)

#### Epochs 0-49: Stage 1
- **Model State:**
  - `skip_copula: false` (copula initialized but NOT used in forward pass)
  - `stage: 1` (copula skipped due to `stage < 2` condition)
  - `lock_skip_copula: false` (allows transition)

- **What's Trained:**
  - Marginal encoder parameters
  - Flow encoder parameters
  - Decoder parameters
  - Copula parameters NOT used (no gradients)

- **Optimizer:**
  - LR: `0.000381` (lr_stage1)
  - Weight decay: `0.0`
  - All parameters in optimizer, but only marginal/flow/decoder receive gradients

#### Epoch 50: Automatic Transition
1. **`on_train_epoch_start()` callback triggers:**
   - Detects `current_epoch >= stage2_start_epoch`
   - Calls `model.set_stage(2)`

2. **`set_stage(2)` executes:**
   - Updates `self.stage = 2`
   - Since `lock_skip_copula=false`, sets `skip_copula = (stage == 1) = false`
   - Calls `_initialize_copula_components()` (already initialized, no-op)
   - Updates `decoder.skip_copula = false`

3. **Lightning module freezes parameters:**
   ```python
   for name, param in self.model.tactis.named_parameters():
       if name.startswith("flow_") or name.startswith("marginal"):
           param.requires_grad = False  # FREEZE
       elif name.startswith("copula_"):
           param.requires_grad = True   # UNFREEZE
   ```

4. **Optimizer reconfiguration:**
   - Updates all param groups:
     - `lr = lr_stage2` (0.000386)
     - `weight_decay = weight_decay_stage2` (2e-05)
     - `initial_lr = lr_stage2` (for scheduler reference)

5. **Scheduler reconfiguration:**
   - Creates new warmup + cosine annealing scheduler for Stage 2
   - Uses Stage 2 hyperparameters

#### Epochs 50-99: Stage 2
- **Model State:**
  - `skip_copula: false`
  - `stage: 2` (copula NOW used in forward pass)

- **What's Trained:**
  - Copula encoder parameters (unfrozen)
  - Attentional copula MLP parameters (unfrozen)
  - Marginal/flow/decoder parameters (FROZEN)

- **Optimizer:**
  - LR: `0.000386` (lr_stage2)
  - Weight decay: `2e-05`

## Configuration Files

### Config: `config/training/training_inputs_juan_awaken_train_storm_pred60_full.yaml`

**Critical Settings for Sequential Training:**
```yaml
model:
  tactis:
    skip_copula: false        # ✓ Allow both stages (copula initialized from start)
    lock_skip_copula: false   # ✓ Allow automatic stage transition
    initial_stage: 1          # ✓ Start in Stage 1
    stage2_start_epoch: 50    # ✓ Transition at epoch 50

trainer:
  max_epochs: 100             # ✓ 50 epochs per stage
  limit_train_batches: null   # ✓ Use FULL dataset (no subsampling)

dataset:
  sampler: "sequential"       # ✓ Deterministic training order
  batch_size: 64              # ✓ From tuning
```

**Key Differences from Tuning Configs:**
| Setting | Stage 1 Tuning | Stage 2 Tuning | Sequential Training |
|---------|----------------|----------------|---------------------|
| `skip_copula` | `true` | `false` | `false` |
| `lock_skip_copula` | `true` | `true` | **`false`** |
| `initial_stage` | `1` | `2` | `1` |
| `stage2_start_epoch` | `999` | `0` | `50` |
| `max_epochs` | `30` | `30` | `100` |
| `limit_train_batches` | `5000` | `5000` | **`null`** |
| `sampler` | `random` | `random` | **`sequential`** |

### Script: `run_scripts/train_scripts/train_awaken_storm_60_full_sequential.sh`

## How to Run

### 1. Review Configuration
```bash
# Verify config is correct
python /tmp/verify_sequential_config.py
```

Expected output: "✅ ALL CHECKS PASSED"

### 2. Make Script Executable
```bash
chmod +x wind_forecasting/run_scripts/train_scripts/train_awaken_storm_60_full_sequential.sh
```

### 3. Submit Job
```bash
sbatch wind_forecasting/run_scripts/train_scripts/train_awaken_storm_60_full_sequential.sh
```

### 4. Monitor Training

#### Check Job Status
```bash
squeue -u $USER
```

#### Monitor GPU Usage
```bash
# Get the node name from squeue output
ssh -L 8088:localhost:8088 $USER@<node_name>

# Then on the node:
mamba activate wf_env_storm
gpustat -P --no-processes --watch 0.5
```

#### Check Logs
```bash
# Main output log
tail -f /dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/awaken_train_tactis_60_seq_full_*.out

# Error log
tail -f /dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/awaken_train_tactis_60_seq_full_*.err
```

#### WandB Dashboard
- Project: `train_awaken_storm_full_pred60_15s_sequential`
- Look for:
  - Learning rate changes at epoch 50
  - Validation loss curve
  - Stage transition markers

### 5. Verify Stage Transition

Search logs for transition markers:
```bash
grep -E "Entering Stage 2 transition|Froze.*marginal|Updated optimizer lr" <log_file>
```

Expected output around epoch 50:
```
Epoch 50: Entering Stage 2 transition.
Freezing flow/marginal parameters and unfreezing copula parameters...
Froze 123 flow/marginal parameters. Ensured 45 copula parameters are trainable.
Epoch 50: Switched to Stage 2. Updated optimizer lr=0.000386, weight_decay=2e-05.
```

## Expected Checkpoints

Checkpoints will be saved in:
```
/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/checkpoints/*_60_sequential_full/
```

### Key Checkpoints to Examine:

1. **`seq_train_epoch=49-step=*-val_loss=*.ckpt`**
   - End of Stage 1 (best marginal training)
   - Use this to verify Stage 1 converged

2. **`seq_train_epoch=50-step=*-val_loss=*.ckpt`**
   - First epoch of Stage 2
   - Use to verify transition occurred

3. **`seq_train_epoch=99-step=*-val_loss=*.ckpt`**
   - Final trained model
   - Contains both optimized marginals and copula

4. **`last.ckpt`**
   - Latest checkpoint (useful for resuming)

## Verification Checklist

Before submitting, verify:

- [✓] Config file has `skip_copula: false`
- [✓] Config file has `lock_skip_copula: false`
- [✓] Config file has `initial_stage: 1`
- [✓] Config file has `stage2_start_epoch: 50`
- [✓] Config file has `max_epochs: 100`
- [✓] Config file has `limit_train_batches: null`
- [✓] Config file has `sampler: "sequential"`
- [✓] All hyperparameters from tuning are present
- [✓] Script is executable
- [✓] Output directories exist

## Troubleshooting

### Issue: Copula trained during Stage 1
**Symptom**: Copula parameters changing in epochs 0-49

**Diagnosis**: Check forward pass logic - copula should NOT be used when `stage < 2`

**Solution**: Verify config has `initial_stage: 1` and check model logs for stage value

### Issue: Marginal parameters change after epoch 50
**Symptom**: Marginal params have `requires_grad=True` in Stage 2

**Diagnosis**: Freezing logic failed

**Solution**: Check transition logs for "Froze X flow/marginal parameters" message

### Issue: Learning rate doesn't change at epoch 50
**Symptom**: LR stays at `0.000381` throughout

**Diagnosis**: Optimizer not reconfigured

**Solution**: Check logs for "Updated optimizer lr=0.000386" at epoch 50

### Issue: Stage never transitions
**Symptom**: Training runs all 100 epochs in Stage 1

**Diagnosis**: `lock_skip_copula` might be `true`

**Solution**: Verify config has `lock_skip_copula: false`

## Comparison with Independent Stage Training

### Independent Stages (How Tuning Worked)
```
Job 1: Stage 1 Only (30 epochs)
  - skip_copula: true
  - lock_skip_copula: true
  - Train marginals/flow/decoder
  - Save checkpoint

Job 2: Stage 2 Only (30 epochs)
  - skip_copula: false
  - lock_skip_copula: true
  - initial_stage: 2
  - Load Stage 1 checkpoint
  - Train copula with marginals frozen
```

### Sequential Training (New Approach)
```
Single Job (100 epochs)
  - skip_copula: false (from start)
  - lock_skip_copula: false (allow transition)
  - initial_stage: 1
  - stage2_start_epoch: 50

  Epochs 0-49: Stage 1 (copula exists but unused)
  Epoch 50: Automatic transition + freezing
  Epochs 50-99: Stage 2 (marginals frozen)
```

**Advantages:**
1. Single job submission
2. No checkpoint management between stages
3. Automatic parameter freezing
4. Continuous training timeline
5. Easier to track in WandB

## Questions About Sequential Training Design

### Q: Why is copula initialized in Stage 1 if it's not used?
**A**: With `skip_copula=false`, copula components are initialized at the start but NOT used in forward pass due to `stage < 2` condition. This allows seamless transition at epoch 50 without reinitializing components.

### Q: How does the model know not to train copula in Stage 1?
**A**: The forward pass checks `if not self.skip_copula and self.stage >= 2:` before using copula. In Stage 1, `stage=1`, so condition fails and copula is skipped. No gradients → no training.

### Q: What prevents marginal parameters from training in Stage 2?
**A**: At epoch 50, the transition callback explicitly sets `param.requires_grad = False` for all parameters starting with "flow_" or "marginal". PyTorch won't compute gradients for these parameters.

### Q: Why 100 epochs (50 per stage)?
**A**: Tuning used 30 epochs per stage with `limit_train_batches=5000` (subset). With full dataset, more epochs are beneficial. 50 per stage provides sufficient convergence without excessive training time.

### Q: Can I use different epoch splits?
**A**: Yes! Just change `stage2_start_epoch` and `max_epochs`. For example:
- 40/60 split: `stage2_start_epoch: 40`, `max_epochs: 100`
- 60/40 split: `stage2_start_epoch: 60`, `max_epochs: 100`

### Q: What if I want to resume training?
**A**: Load the checkpoint and continue. The model preserves stage info. If resuming from epoch >= 50, it will automatically be in Stage 2.

## Additional Notes

### Epoch Counts
- **Tuning**: 30 epochs × 5000 batches = 150,000 steps per stage
- **Full Training**: 50 epochs × ~15,000 batches = ~750,000 steps per stage (5x more data!)

### Memory Requirements
- Full dataset requires more GPU memory than tuning
- Mixed precision enabled (`precision: "16-mixed"`) to help
- 4 GPUs with DDP distributes batch across GPUs

### Validation Strategy
- `val_check_interval: 1.0` = validate after each full epoch
- More frequent than tuning (which used batch intervals)
- Better tracking of stage transition impact

---

**Created**: 2025-10-09
**Author**: Claude Code (Anthropic)
**Purpose**: Guide for sequential TACTiS Stage 1→2 training with best hyperparameters from tuning trials 157 (Stage 1) and 146 (Stage 2)
