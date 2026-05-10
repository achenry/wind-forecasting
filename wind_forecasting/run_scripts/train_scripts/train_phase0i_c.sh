#!/bin/bash

#SBATCH --partition=cfdg.p          # cfdg.p preferred (7-day cap fits 36h budget)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # SINGLE-GPU (avoids NCCL DDP hangs per Phase 0h pattern)
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8016
#SBATCH --gres=gpu:H100:1
#SBATCH --exclude=cfdg001
#SBATCH --time=36:00:00              # 36h budget (full 100 ep)
#SBATCH --job-name=tactis_phase0i_c
#SBATCH --output=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/tactis_phase0i_c_%j.out
#SBATCH --error=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/tactis_phase0i_c_%j.err
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block
#SBATCH --gres-flags=enforce-binding

# =============================================================================
# TACTiS-2 PHASE 0i-C — production retrain with Sa (a_floor) fix from scratch
# =============================================================================
# Single root cause confirmed by 2026-05-08 D1-D14 diagnostics:
#   52.7% of layer-0 DSF `a` values < 0.1 → flat CDF regions → F1 + F2.
# Sa adds a soft-hinge L2 penalty: relu(a_floor_threshold - a)^2 with
#   a_floor_threshold=0.5, lambda_a_floor=1.0 (CENTER recommendation).
# REVISIT after pilot if a different config is selected.
#
# Submit AFTER mini-pilot (train_phase0i_c_pilot.sh) has identified a winner:
#   eval "$(grep '^export LOCAL_PG_PASSWORD=' ~/.zshrc)"
#   sbatch --export=ALL,LOCAL_PG_PASSWORD train_phase0i_c.sh
#
# Pass criterion (probe at end of Stage 1 / ep29):
#   - layer0_frac_a_lt_0p1 < 0.10 (was 0.527 in 0i-B)
#   - F^-1 spread (z-space) > 2.0 (was ~0.26 in 0i-B ep74)
#   - persistence-skill positive on >70% turbines after validation harness
# =============================================================================

export MODEL_NAME="tactis"
export CONFIG_FILE="/user/taed7566/Forecasting/wind-forecasting/config/training/training_inputs_storm_awaken_unsmoothed_pred60_tactis_phase0i_c.yaml"

BASE_DIR="/user/taed7566/Forecasting/wind-forecasting"
OUTPUT_DIR="/dss/work/taed7566/Forecasting_Outputs/wind-forecasting"
export WORK_DIR="${BASE_DIR}/wind_forecasting"
export LOG_DIR="${OUTPUT_DIR}/logs"
export NUMEXPR_MAX_THREADS=128

mkdir -p ${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}
mkdir -p ${LOG_DIR}/checkpoints

cd ${WORK_DIR} || exit 1

export PYTHONPATH=${WORK_DIR}:/fs/dss/home/taed7566/Forecasting/pytorch-transformer-ts:${PYTHONPATH}
export WANDB_DIR=${LOG_DIR}

echo "================================================================"
echo "TACTiS-2 PHASE 0i-C — production retrain with Sa (a_floor) fix"
echo "================================================================"
echo "JOB ID:      ${SLURM_JOB_ID}"
echo "PARTITION:   ${SLURM_JOB_PARTITION}"
echo "NODE:        ${SLURM_JOB_NODELIST}"
echo "CONFIG:      ${CONFIG_FILE}"
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | uniq)
echo "GPU TYPE:    ${GPU_TYPE}"
echo "TIME LIMIT:  36h"
echo "Stage 1: epochs 0-29 (a_floor active) | Stage 2: epochs 30-99 (flow frozen)"
echo "================================================================"
echo "Branch states:"
echo "  pytorch-transformer-ts: $(cd /fs/dss/home/taed7566/Forecasting/pytorch-transformer-ts && git rev-parse --abbrev-ref HEAD 2>/dev/null) @ $(cd /fs/dss/home/taed7566/Forecasting/pytorch-transformer-ts && git rev-parse --short HEAD 2>/dev/null)"
echo "  wind-forecasting:       $(cd ${BASE_DIR} && git rev-parse --abbrev-ref HEAD 2>/dev/null) @ $(cd ${BASE_DIR} && git rev-parse --short HEAD 2>/dev/null)"
echo "================================================================"

module purge
module load slurm/hpc-2023/23.02.7
module load hpc-env/13.1
module load mpi4py/3.1.4-gompi-2023a
module load Mamba/24.3.0-0
module load CUDA/12.4.0
module load git
eval "$(conda shell.bash hook)"
conda activate wf_env_storm

if [ -z "${LOCAL_PG_PASSWORD:-}" ]; then
    echo "ERROR: LOCAL_PG_PASSWORD env var not set." >&2
    exit 1
fi
export PGPASSWORD="${LOCAL_PG_PASSWORD}"

date +"%Y-%m-%d %H:%M:%S"
echo "=== STARTING SINGLE-GPU TRAINING (phase0i_c) ==="

python ${WORK_DIR}/run_scripts/run_model.py \
    --config ${CONFIG_FILE} \
    --model ${MODEL_NAME} \
    --mode train \
    --seed 42 \
    --single_gpu

EXIT_CODE=$?

echo "=== TRAINING FINISHED (exit ${EXIT_CODE}) ==="
date +"%Y-%m-%d %H:%M:%S"

# Move main slurm logs into job-id subdir
for f in ${LOG_DIR}/slurm_logs/*_${SLURM_JOB_ID}.out; do
    [ -e "$f" ] && mv "$f" "${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/"
done
for f in ${LOG_DIR}/slurm_logs/*_${SLURM_JOB_ID}.err; do
    [ -e "$f" ] && mv "$f" "${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/"
done

echo ""
echo "================================================================"
echo "PHASE 0i-C COMPLETE. Verification steps:"
echo "  1. Probe end-of-Stage-1 (manual_save_epoch29) — layer0_frac_a_lt_0p1 < 0.10"
echo "  2. Probe F^-1 spread on real contexts — std-space spread > 2.0"
echo "  3. Run validation harness (run_validate_phase0i_c.sh) — CRPS skill > 0%"
echo "  4. Compare to Phase 0i-B baseline using compare_metrics.py"
echo "================================================================"

exit ${EXIT_CODE}
