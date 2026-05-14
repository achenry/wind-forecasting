#!/bin/bash

#SBATCH --partition=cfdg.p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8016
#SBATCH --gres=gpu:H100:1
#SBATCH --exclude=cfdg001
#SBATCH --time=48:00:00
#SBATCH --job-name=tactis_phase0i_e
#SBATCH --output=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/tactis_phase0i_e_%j.out
#SBATCH --error=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/tactis_phase0i_e_%j.err
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block
#SBATCH --gres-flags=enforce-binding

# =============================================================================
# TACTiS-2 PHASE 0i-E PRODUCTION — NSF marginal flow, full retrain
# =============================================================================
# Submit AFTER F-pilot identifies a winner. Update phase0i_e.yaml to carry
# the winning (num_bins, tail_bound, num_flow_layers) AND optional ES if the
# pilot's median F^-1 width was in [0.5, 1.0] (per decision tree in plan).
#
# Submit:
#   eval "$(grep '^export LOCAL_PG_PASSWORD=' ~/.zshrc)"
#   sbatch --export=ALL,LOCAL_PG_PASSWORD train_phase0i_e.sh
#
# Pass criterion (validation):
#   - median F^-1 width > 1.0
#   - sample-axis std > 0.3 m/s
#   - >80% turbines beat persistence (vs 0i-B's 74%)
#   - PIT-extreme < 50% (vs 0i-B's 98%)
# =============================================================================

export MODEL_NAME="tactis"
export CONFIG_FILE="/user/taed7566/Forecasting/wind-forecasting/config/training/training_inputs_storm_awaken_unsmoothed_pred60_tactis_phase0i_e.yaml"

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
echo "TACTiS-2 PHASE 0i-E PRODUCTION — NSF marginal full retrain"
echo "================================================================"
echo "JOB ID:      ${SLURM_JOB_ID}"
echo "PARTITION:   ${SLURM_JOB_PARTITION}"
echo "NODE:        ${SLURM_JOB_NODELIST}"
echo "CONFIG:      ${CONFIG_FILE}"
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | uniq)
echo "GPU TYPE:    ${GPU_TYPE}"
echo "TIME LIMIT:  48h"
echo "Stage 1: epochs 0-29 (NSF marginal trains) | Stage 2: 30-99 (flow frozen)"
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
echo "=== STARTING SINGLE-GPU TRAINING (phase0i_e production) ==="

python ${WORK_DIR}/run_scripts/run_model.py \
    --config ${CONFIG_FILE} \
    --model ${MODEL_NAME} \
    --mode train \
    --seed 42 \
    --single_gpu

EXIT_CODE=$?

echo "=== TRAINING FINISHED (exit ${EXIT_CODE}) ==="
date +"%Y-%m-%d %H:%M:%S"

for f in ${LOG_DIR}/slurm_logs/*_${SLURM_JOB_ID}.out; do
    [ -e "$f" ] && mv "$f" "${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/"
done
for f in ${LOG_DIR}/slurm_logs/*_${SLURM_JOB_ID}.err; do
    [ -e "$f" ] && mv "$f" "${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/"
done

echo ""
echo "================================================================"
echo "PHASE 0i-E PRODUCTION COMPLETE. Verification:"
echo "  1. Probe end-of-Stage-1 ckpt — median F^-1 width > 1.0"
echo "  2. Run validation harness; compare_metrics.py vs Phase 0i-B"
echo "  3. PIT-extreme < 50%, beat persistence on >80%"
echo "================================================================"

exit ${EXIT_CODE}
