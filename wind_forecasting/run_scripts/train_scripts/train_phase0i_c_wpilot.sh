#!/bin/bash

#SBATCH --partition=cfdg.p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8016
#SBATCH --gres=gpu:H100:1
#SBATCH --exclude=cfdg001
#SBATCH --time=08:00:00
#SBATCH --job-name=tactis_phase0i_c_wpilot
#SBATCH --output=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/tactis_phase0i_c_wpilot_%j.out
#SBATCH --error=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/tactis_phase0i_c_wpilot_%j.err
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block
#SBATCH --gres-flags=enforce-binding

# =============================================================================
# TACTiS-2 PHASE 0i-C W-PILOT — sweep w-entropy regularizer (Fix Sw)
# =============================================================================
# 2026-05-09 diagnostics revealed eff_dim ≈ 1.0 / 64 — the marginal_conditioner
# emits one-hot softmax weights, which is the actual root cause of F1+F2.
# Sa (C3 settings: a_floor_threshold=0.5, lambda_a_floor=5.0) is kept fixed;
# Sw is the NEW lever being swept here.
#
# Pilot variants (Sw is on TOP of C3's Sa settings):
#   W0: lambda_w_entropy=0.0                 — control (Sa-only baseline)
#   W1: w_entropy_min=1.39  lambda=1.0       — mild (eff_dim ≥ 4)
#   W2: w_entropy_min=2.08  lambda=1.0       — center (eff_dim ≥ 8)
#   W3: w_entropy_min=2.08  lambda=5.0       — strong lambda
#   W4: w_entropy_min=2.77  lambda=2.0       — aggressive (eff_dim ≥ 16)
#
# Submit ALL 5 in parallel:
#   eval "$(grep '^export LOCAL_PG_PASSWORD=' ~/.zshrc)"
#   for V in W0 W1 W2 W3 W4; do
#       PILOT_VARIANT=$V sbatch --export=ALL,LOCAL_PG_PASSWORD,PILOT_VARIANT \
#           train_phase0i_c_wpilot.sh
#   done
#
# Pass criterion (per WandB after ~25 epochs):
#   - w_entropy_reg/median_eff_dim > 4 (baseline ≈ 1.02)
#   - F^-1(0.95)-F^-1(0.05) std-space MEDIAN > 0.5 (baseline 0.27 — flat)
#   - val_loss not regressed by >5% vs W0
# =============================================================================

export MODEL_NAME="tactis"

if [ -z "${PILOT_VARIANT:-}" ]; then
    echo "ERROR: PILOT_VARIANT env var required (one of W0, W1, W2, W3, W4)" >&2
    exit 1
fi

case "${PILOT_VARIANT}" in
    W0) DESC="Sa-only control: lambda_w_entropy=0" ;;
    W1) DESC="mild: w_entropy_min=1.39 lambda=1.0" ;;
    W2) DESC="center: w_entropy_min=2.08 lambda=1.0" ;;
    W3) DESC="strong: w_entropy_min=2.08 lambda=5.0" ;;
    W4) DESC="aggressive: w_entropy_min=2.77 lambda=2.0" ;;
    *) echo "ERROR: unknown PILOT_VARIANT=${PILOT_VARIANT}" >&2; exit 1 ;;
esac

BASE_DIR="/user/taed7566/Forecasting/wind-forecasting"
OUTPUT_DIR="/dss/work/taed7566/Forecasting_Outputs/wind-forecasting"
export WORK_DIR="${BASE_DIR}/wind_forecasting"
export LOG_DIR="${OUTPUT_DIR}/logs"
export NUMEXPR_MAX_THREADS=128
export CONFIG_FILE="${BASE_DIR}/config/training/training_inputs_storm_awaken_unsmoothed_pred60_tactis_phase0i_c_wpilot_${PILOT_VARIANT}.yaml"

mkdir -p ${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}
mkdir -p ${LOG_DIR}/checkpoints

cd ${WORK_DIR} || exit 1

export PYTHONPATH=${WORK_DIR}:/fs/dss/home/taed7566/Forecasting/pytorch-transformer-ts:${PYTHONPATH}
export WANDB_DIR=${LOG_DIR}

echo "================================================================"
echo "TACTiS-2 PHASE 0i-C W-PILOT (variant ${PILOT_VARIANT})"
echo "================================================================"
echo "JOB ID:      ${SLURM_JOB_ID}"
echo "PARTITION:   ${SLURM_JOB_PARTITION}"
echo "NODE:        ${SLURM_JOB_NODELIST}"
echo "CONFIG:      ${CONFIG_FILE}"
echo "DESC:        ${DESC}"
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | uniq)
echo "GPU TYPE:    ${GPU_TYPE}"
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
echo "=== STARTING SINGLE-GPU TRAINING (W-pilot ${PILOT_VARIANT}) ==="

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
echo "PHASE 0i-C W-PILOT ${PILOT_VARIANT} COMPLETE"
echo "Verify on WandB:"
echo "  1. w_entropy_reg/median_eff_dim should be > 4 (baseline ~1.02)"
echo "  2. F^-1(0.95)-F^-1(0.05) std-space > 0.5 (baseline 0.27)"
echo "  3. val_loss within 5% of W0 control"
echo "================================================================"

exit ${EXIT_CODE}
