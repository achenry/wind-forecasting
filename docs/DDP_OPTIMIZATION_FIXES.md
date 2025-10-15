# DDP Training Optimization Fixes

**Date**: 2025-10-14
**Status**: ✅ IMPLEMENTED - READY FOR TESTING
**Branch**: feature/fix-ddp-partitioning

## Summary

Fixed four critical issues with DDP training on 3× H100 GPUs that were causing inefficiency and lack of observability.

---

## Issues Fixed

### 🔴 Issue #1: Inefficient Data Partitioning (CRITICAL)
**Problem**: All 21 workers (3 ranks × 7 workers/rank) loaded and processed the same 22,880 time series, then `islice` filtered 20/21 of generated samples. This caused **21× redundant computation**.

**Solution**: Partition time series at initialization so each worker only loads and processes 1/21 of data (~1,090 series each).

**Files Modified**:
- `/wind_forecasting/preprocessing/pytorch_dataset.py`
  - Changed `self.data` to `self.full_data_list` (keep as list for partitioning)
  - Removed `islice` filtering from `__iter__`
  - Added data partitioning logic: `worker_time_series = [entry for idx, entry in enumerate(self.full_data_list) if idx % global_num_workers == global_worker_id]`
  - Applied same fix to `WindForecastingInferenceDataset`

**Expected Impact**:
- **21× reduction** in redundant computation per worker
- **21× reduction** in memory usage per worker
- **>30% speed increase** (from ~21 it/s to >30 it/s)
- **Increased GPU utilization** (data loading no longer bottleneck)

---

### 🟡 Issue #2: Missing Multi-GPU Monitoring (MEDIUM)
**Problem**: WandB only logged `system/gpu.0.*` metrics. GPUs 1 and 2 were invisible in monitoring.

**Solution**: Created `MultiGPUMonitor` callback that queries `nvidia-smi` from rank 0 to log metrics from all GPUs.

**Files Created**:
- `/wind_forecasting/callbacks/__init__.py`
- `/wind_forecasting/callbacks/multi_gpu_monitor.py`

**Files Modified**:
- `/wind_forecasting/run_scripts/run_model.py` (lines 807-827)
  - Auto-detects multi-GPU mode (SLURM_NTASKS > 1 or WORLD_SIZE > 1)
  - Instantiates and appends `MultiGPUMonitor(log_interval=10)`

**Expected Impact**:
- WandB now logs `system/gpu.{0,1,2}.*` metrics
- Visibility into all GPU utilization, memory, power, temperature
- Easier debugging of GPU imbalance issues

---

### 🟠 Issue #3: Performance Degradation (HIGH)
**Problem**: Training slowed from ~30 it/s to ~21 it/s (30% slower).

**Root Cause**: Consequence of Issue #1 (inefficient partitioning).

**Solution**: Automatically resolved by Fix #1.

**Expected Impact**: Training speed **>30 it/s** after partitioning fix.

---

### 🟢 Issue #4: Progress Bar Shows "?" (LOW)
**Problem**: `IterableDataset` doesn't support `__len__()`, so Lightning can't display total batch count.

**Solution**: Implemented `__len__()` and `_calculate_total_samples()` methods.

**Files Modified**:
- `/wind_forecasting/preprocessing/pytorch_dataset.py`
  - Added `_calculate_total_samples()` method
  - Implemented `__len__()` to return total samples

**Expected Impact**: Progress bar shows `954368/618396` instead of `954368/?`.

---

## Testing & Validation

### Unit Tests ✅
All 9 unit tests pass:

```bash
cd /user/taed7566/Forecasting/wind-forecasting
python -m pytest tests/test_data_partitioning.py -v
```

**Tests verify**:
- ✅ Single worker gets all data
- ✅ DDP ranks get non-overlapping time series
- ✅ Multi-worker partitioning is correct
- ✅ No duplicate series across workers
- ✅ All series are assigned (no gaps)
- ✅ Partitioning is deterministic
- ✅ Correct with different world sizes
- ✅ `__len__()` returns correct count
- ✅ `__len__()` handles `skip_indices`

### Next Steps for Validation

1. **Short SLURM Test Job** (30 minutes):
   ```bash
   sbatch test_ddp_fixes_short.sh
   ```
   - Monitor logs for: `"Worker X/21 assigned Y/22880 time series"`
   - Verify: Sum of assigned series across workers = 22,880
   - Check: Training speed >30 it/s
   - Confirm: WandB shows all 3 GPUs

2. **Full Training Job** (7 days):
   - Only submit after short test passes
   - Monitor first 2 hours closely
   - Verify checkpoint creation works correctly

---

## Technical Details

### Data Partitioning Logic

**Old Approach** (Inefficient):
```python
# All workers iterate all 22,880 series
for entry in self.data:
    sampled_indices = self.sampler(dummy_target)  # Generate ALL windows
    for idx in sampled_indices:
        yield sample  # Generate ALL samples

# Then islice filters 20/21 of them
return islice(self._base_iter(), global_worker_id, None, global_num_workers)
```

**New Approach** (Efficient):
```python
# Partition series at initialization
worker_time_series = [
    entry for idx, entry in enumerate(self.full_data_list)
    if idx % global_num_workers == global_worker_id
]

# Each worker only processes ITS series
for entry in worker_time_series:
    sampled_indices = self.sampler(dummy_target)  # Generate windows for THIS series
    for idx in sampled_indices:
        yield sample  # Yield ALL samples from THIS series

# No filtering needed!
```

**Partitioning Example** (88 series, 21 workers):
- Worker 0: series [0, 21, 42, 63, 84] (5 series)
- Worker 1: series [1, 22, 43, 64, 85] (5 series)
- ...
- Worker 20: series [20, 41, 62, 83] (4 series)

Total: 88 series, each assigned to exactly one worker.

---

## Performance Expectations

### Before Fixes:
- **Training speed**: ~21 it/s
- **Computation efficiency**: 1/21 = 4.76% (95.24% wasted)
- **GPU utilization**: Low (data loading bottleneck)
- **Monitoring**: Only GPU 0 visible

### After Fixes:
- **Training speed**: >30 it/s (43% faster)
- **Computation efficiency**: 100% (no redundant work)
- **GPU utilization**: High (efficient data loading)
- **Monitoring**: All 3 GPUs visible

---

## Rollback Plan

If issues arise:

```bash
# Current branch
git branch  # feature/fix-ddp-partitioning

# Revert to main
git checkout main
git reset --hard origin/main

# Cancel running job
scancel <JOB_ID>

# Resubmit with old code
sbatch train_awaken_storm_60_full_sequential.sh
```

**Checkpoint compatibility**: All fixes preserve checkpoint format. Can resume from old checkpoints with new code.

---

## Files Modified Summary

### Core Changes:
1. `/wind_forecasting/preprocessing/pytorch_dataset.py` - Data partitioning + `__len__()`
2. `/wind_forecasting/callbacks/multi_gpu_monitor.py` - New GPU monitoring callback
3. `/wind_forecasting/callbacks/__init__.py` - Package init
4. `/wind_forecasting/run_scripts/run_model.py` - Integrate GPU monitoring

### Testing:
5. `/tests/test_data_partitioning.py` - Comprehensive unit tests

### Documentation:
6. `/docs/DDP_OPTIMIZATION_FIXES.md` - This document

---

## Monitoring Checklist

After submitting test job, verify:

- [ ] Workers log: `"Worker X/21 assigned Y/22880 time series"`
- [ ] Sum of assigned series = 22,880 (all series covered)
- [ ] No duplicate assignments across workers
- [ ] Training speed >25 it/s (preferably >30 it/s)
- [ ] WandB logs `system/gpu.0.*`, `system/gpu.1.*`, `system/gpu.2.*`
- [ ] Progress bar shows `X/618396` instead of `X/?`
- [ ] GPU utilization increases (visible in `nvidia-smi` or WandB)
- [ ] No errors in SLURM logs

---

## Additional Notes

### Why 7 workers per GPU?
With 8 CPUs allocated per GPU:
- 1 CPU for main GPU training process
- 7 CPUs for DataLoader workers (parallel data loading)

This maximizes CPU utilization for data loading while keeping GPUs fed.

### Why `islice` was inefficient?
`islice` only filters the OUTPUT of the iterator, not the WORK done by the iterator. All workers still:
1. Loaded all 22,880 pickle entries
2. Generated all window indices via sampler
3. Constructed all window samples
4. Then threw away 20/21 of them

It's like 21 chefs all cooking the same 22,880 meals, then each chef throws away 20/21 of their food and only serves 1/21. Wasteful!

### New approach efficiency:
Each chef gets assigned 1/21 of the recipes upfront, cooks only their recipes, serves all their meals. No waste!

---

## Contact

For questions about these fixes:
- Check SLURM logs: `/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/`
- Check WandB: https://wandb.ai/jmb0507-cu-boulder/train_train_awaken_storm_full_pred60_15s_sequential_tactis/
- Review unit tests: `pytest tests/test_data_partitioning.py -v`
