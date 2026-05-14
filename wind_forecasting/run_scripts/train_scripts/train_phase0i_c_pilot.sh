#!/bin/bash

#SBATCH --partition=cfdg.p          # cfdg.p preferred (7-day); override to all_gpu.p with --partition=all_gpu.p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # SINGLE-GPU
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8016
#SBATCH --gres=gpu:H100:1
#SBATCH --exclude=cfdg001            # cfdg001 is A100
#SBATCH --time=08:00:00              # 8h per trial (Stage 1 only, 25 ep x 2000 batches)
#SBATCH --job-name=tactis_phase0i_c_pilot
#SBATCH --output=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/tactis_phase0i_c_pilot_%j.out
#SBATCH --error=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/tactis_phase0i_c_pilot_%j.err
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block
#SBATCH --gres-flags=enforce-binding

# =============================================================================
# TACTiS-2 PHASE 0i-C MINI-PILOT — sweep (a_floor_threshold, lambda_a_floor)
# =============================================================================
# 5-trial sweep on Stage 1 only (max_epochs=25 with stage2_start_epoch=999)
# to evaluate a_floor regularizer's effect on layer-0 frac(a<0.1).
#
# Pilot variants:
#   C0: control (lambda_a_floor=0.0)               — baseline confirmation
#   C1: a_floor_threshold=0.3, lambda_a_floor=1.0  — mild
#   C2: a_floor_threshold=0.5, lambda_a_floor=1.0  — center recommendation
#   C3: a_floor_threshold=0.5, lambda_a_floor=5.0  — strong lambda
#   C4: a_floor_threshold=1.0, lambda_a_floor=2.0  — aggressive floor
#
# Submit ALL 5 trials in parallel:
#   eval "$(grep '^export LOCAL_PG_PASSWORD=' ~/.zshrc)"
#   for V in C0 C1 C2 C3 C4; do
#       PILOT_VARIANT=$V sbatch --export=ALL,LOCAL_PG_PASSWORD,PILOT_VARIANT \
#           train_phase0i_c_pilot.sh
#   done
#
# Pass criterion (per WandB after ~25 epochs):
#   - layer0_frac_a_lt_0p1 < 0.10 (baseline 0i-B was 0.527)
#   - val_loss not regressed by >5% vs C0 control
# =============================================================================

export MODEL_NAME="tactis"

if [ -z "${PILOT_VARIANT:-}" ]; then
    echo "ERROR: PILOT_VARIANT env var required (one of C0, C1, C2, C3, C4)" >&2
    exit 1
fi

case "${PILOT_VARIANT}" in
    C0) DESC="control: lambda_a_floor=0.0 (no floor)" ;;
    C1) DESC="mild: a_floor_threshold=0.3 lambda_a_floor=1.0" ;;
    C2) DESC="center: a_floor_threshold=0.5 lambda_a_floor=1.0" ;;
    C3) DESC="strong: a_floor_threshold=0.5 lambda_a_floor=5.0" ;;
    C4) DESC="aggressive: a_floor_threshold=1.0 lambda_a_floor=2.0" ;;
    *) echo "ERROR: unknown PILOT_VARIANT=${PILOT_VARIANT}" >&2; exit 1 ;;
esac

BASE_DIR="/user/taed7566/Forecasting/wind-forecasting"
OUTPUT_DIR="/dss/work/taed7566/Forecasting_Outputs/wind-forecasting"
export WORK_DIR="${BASE_DIR}/wind_forecasting"
export LOG_DIR="${OUTPUT_DIR}/logs"
export NUMEXPR_MAX_THREADS=128
export CONFIG_FILE="${BASE_DIR}/config/training/training_inputs_storm_awaken_unsmoothed_pred60_tactis_phase0i_c_pilot_${PILOT_VARIANT}.yaml"

mkdir -p ${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}
mkdir -p ${LOG_DIR}/checkpoints

cd ${WORK_DIR} || exit 1

export PYTHONPATH=${WORK_DIR}:/fs/dss/home/taed7566/Forecasting/pytorch-transformer-ts:${PYTHONPATH}
export WANDB_DIR=${LOG_DIR}

echo "================================================================"
echo "TACTiS-2 PHASE 0i-C MINI-PILOT (variant ${PILOT_VARIANT})"
echo "================================================================"
echo "JOB ID:      ${SLURM_JOB_ID}"
echo "PARTITION:   ${SLURM_JOB_PARTITION}"
echo "NODE:        ${SLURM_JOB_NODELIST}"
echo "CONFIG:      ${CONFIG_FILE}"
echo "DESC:        ${DESC}"
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | uniq)
echo "GPU TYPE:    ${GPU_TYPE}"
echo "TIME LIMIT:  8h"
echo "Stage 1 only: 25 epochs (stage2_start_epoch=999)"
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
echo "=== STARTING SINGLE-GPU TRAINING (${PILOT_VARIANT}) ==="

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
echo "PHASE 0i-C PILOT ${PILOT_VARIANT} COMPLETE"
echo "Verify on WandB:"
echo "  1. a_floor_reg/layer0_frac_a_lt_0p1 should be < 0.10 (baseline 0.527)"
echo "  2. val_loss should be within 5% of C0 control"
echo "  3. a_floor_reg/median_a should be > 0.3"
echo "================================================================"

exit ${EXIT_CODE}
