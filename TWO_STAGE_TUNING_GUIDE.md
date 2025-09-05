# Two-Stage TACTiS Tuning Guide for AWAKEN Data

## Overview
This guide explains how to run the two-stage tuning process for TACTiS-2 model on AWAKEN data.
- **Stage 1**: Optimizes marginal/flow parameters only (skip_copula=true)
- **Stage 2**: Loads best Stage 1 checkpoint and optimizes copula parameters only

## Configuration
- **Context Length**: 600 seconds
- **Prediction Length**: 60 seconds  
- **Dataset**: AWAKEN (normalized)
- **Database**: Oldenburg University PostgreSQL (pg.optuna.uni-oldenburg.de)
- **GPUs**: 3 parallel workers per stage

## Files Created/Modified

### Configuration Files
- `config/training/training_inputs_juan_awaken_tune_storm_pred60_stage1.yaml` - Stage 1 config
- `config/training/training_inputs_juan_awaken_tune_storm_pred60_stage2.yaml` - Stage 2 config

### SLURM Scripts  
- `wind_forecasting/run_scripts/tune_scripts/tune_model_storm_awaken_p60_stage1.sh` - Stage 1 launcher
- `wind_forecasting/run_scripts/tune_scripts/tune_model_storm_awaken_p60_stage2.sh` - Stage 2 launcher

### Code Modifications
- `pytorch-transformer-ts/pytorch_transformer_ts/tactis_2/tactis.py` - Added `lock_skip_copula` parameter
- `pytorch-transformer-ts/pytorch_transformer_ts/tactis_2/estimator.py` - Pass through `lock_skip_copula`
- `wind_forecasting/run_scripts/run_model.py` - Added `--stage1_study` argument
- `wind_forecasting/tuning/core.py` - Pass stage1_study to MLTuningObjective
- `wind_forecasting/tuning/objective.py` - Load Stage 1 checkpoint for Stage 2

## Pre-Launch Validation

Always run validation before launching:
```bash
cd /fs/dss/home/taed7566/Forecasting/wind-forecasting
./validate_stage1_launch.sh
```

## Stage 1: Training Marginals Only

### Launch Command
```bash
cd /fs/dss/home/taed7566/Forecasting/wind-forecasting
sbatch wind_forecasting/run_scripts/tune_scripts/tune_model_storm_awaken_p60_stage1.sh
```

### Stage 1 Configuration
- **skip_copula**: true (trains marginals only)
- **lock_skip_copula**: true (prevents automatic switching)
- **initial_stage**: 1
- **stage2_start_epoch**: 999 (never reaches Stage 2)
- **max_epochs**: 25
- **Trials per worker**: 50
- **Total trials**: 100

### Monitor Progress
```bash
# Get job ID from sbatch output
JOB_ID=<your_job_id>

# Monitor all workers
tail -f /dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/${JOB_ID}/stage1_worker_*.log

# Check GPU usage (from login node)
ssh -L 8088:localhost:8088 ${USER}@<node_name>
# Then in new session:
mamba activate wf_env_storm
gpustat -P --no-processes --watch 0.5
```

### Finding the Study Name
After Stage 1 completes, look for the study name in the logs:
```bash
grep "Study name:" /dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/${JOB_ID}/stage1_worker_0_${JOB_ID}.log
```

The study name will look like: `tune_awaken_tactis_pred60_stage1_marginals_2025_01_05_143022`

## Stage 2: Training Copula Only

### Set Stage 1 Study Name
```bash
# CRITICAL: Set this to the actual study name from Stage 1
export STAGE1_STUDY_NAME="tune_awaken_tactis_pred60_stage1_marginals_2025_01_05_143022"
```

### Launch Command  
```bash
cd /fs/dss/home/taed7566/Forecasting/wind-forecasting
sbatch wind_forecasting/run_scripts/tune_scripts/tune_model_storm_awaken_p60_stage2.sh
```

### Stage 2 Configuration
- **skip_copula**: false (trains copula)
- **lock_skip_copula**: true (keeps it false)
- **initial_stage**: 2
- **stage2_start_epoch**: 0 (immediately Stage 2)
- **max_epochs**: 30
- **Loads**: Best checkpoint from Stage 1 study
- **Frozen**: All marginal/flow parameters from Stage 1

### Monitor Progress
Similar to Stage 1, but check stage2_worker_*.log files.

## Database Management

### View Studies
```bash
# Connect to database
psql -h pg.optuna.uni-oldenburg.de -p 5432 -U optuna02 -d optuna

# List all studies
\dt optuna.*;

# View study details
SELECT study_name, direction FROM optuna.studies;
```

### Optuna Dashboard (if needed)
```bash
# For PostgreSQL backend
optuna-dashboard postgresql://optuna02:<password>@pg.optuna.uni-oldenburg.de:5432/optuna --port 8088
```

## Training with Best Hyperparameters

After both stages complete, use the Stage 2 study for final training:

```bash
python wind_forecasting/run_scripts/run_model.py \
  --config config/training/training_inputs_juan_awaken_tune_storm_pred60_stage2.yaml \
  --model tactis \
  --mode train \
  --use_tuned_parameters \
  --seed 42
```

## Important Notes

1. **Stage Separation**: Each stage has its own Optuna study but shares the same PostgreSQL database
2. **Checkpoint Transfer**: Stage 2 automatically loads the best Stage 1 checkpoint 
3. **Parameter Freezing**: Marginal parameters are frozen in Stage 2, only copula is optimized
4. **Fresh Optimizer**: Stage 2 uses a fresh optimizer (no optimizer state from Stage 1)
5. **Metric**: Both stages use `val_loss` (NLL) as the optimization metric
6. **Database Credentials**: Stored in `/user/taed7566/Forecasting/Docs/db_login`

## Troubleshooting

### Database Connection Issues
- Check `/user/taed7566/Forecasting/Docs/db_login` exists and has correct format
- Ensure you're on compute node or have network access to pg.optuna.uni-oldenburg.de

### Stage 1 Study Not Found
- Check exact study name in Stage 1 logs
- Ensure database contains the study (use psql to verify)

### Out of Memory
- Reduce batch_size in config files
- Check GPU memory usage with gpustat

### Workers Failing
- Check individual worker logs in `/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/<job_id>/`
- Look for Python tracebacks or CUDA errors

## Summary of Key Parameters

| Parameter | Stage 1 | Stage 2 |
|-----------|---------|---------|
| skip_copula | true | false |
| lock_skip_copula | true | true |
| initial_stage | 1 | 2 |
| stage2_start_epoch | 999 | 0 |
| max_epochs | 25 | 30 |
| Learning focus | Marginals only | Copula only |
| Loads from | Fresh start | Stage 1 best checkpoint |

## Next Steps

After successful two-stage tuning:
1. Analyze results in WandB
2. Compare Stage 1 vs Stage 2 performance
3. Use best hyperparameters for final training
4. Validate on test set
5. Deploy model for inference