#!/bin/bash

#SBATCH --partition=cfdg.p          # Default — override to all_gpu.p with --partition=all_gpu.p if cfdg busy
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # SINGLE-GPU — same pattern as Phase 5 v1 (avoids NCCL DDP hangs)
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8016
#SBATCH --gres=gpu:H100:1
#SBATCH --exclude=cfdg001            # cfdg001 is A100
#SBATCH --time=04:00:00              # 4h budget — ~3h actual training (30 ep × 5000 batches × bs=64)
#SBATCH --job-name=tactis_phase0h
#SBATCH --output=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/tactis_phase0h_%j.out
#SBATCH --error=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/tactis_phase0h_%j.err
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block
#SBATCH --gres-flags=enforce-binding

# =============================================================================
# TACTiS-2 PHASE 0h — full v2 retrain (Stage 1 + Stage 2) decision run
# =============================================================================
# Single end-to-end retrain with v2 deeper-fix defenses + winning a_max from
# Phase 0g'. If F⁻¹(U) std > 0.5 at end of Stage 1 AND sample-axis std > 0.05
# at end of Stage 2, we're DONE — no Phase 1 pilot or Phase 3 full retrain
# needed.
#
# Why this design:
#   - v1 collapse fully sets in by epoch 24 → 20 ep Stage 1 sufficient
#   - val_copula_loss=inf bug now fixed → can actually see Stage 2 progress
#   - Trial #65 architecture preserved (no architecture re-tuning needed)
#   - log_density_max=0 fixed (Phase 0g winner)
#   - a_max sed-replaced from $A_MAX env var (set by caller based on 0g' result)
#
# Submit:
#   eval "$(grep '^export LOCAL_PG_PASSWORD=' ~/.zshrc)"
#   A_MAX=5 sbatch --export=ALL,LOCAL_PG_PASSWORD,A_MAX train_phase0h_v2.sh
# (replace 5 with the winning a_max value from Phase 0g')
# =============================================================================

export MODEL_NAME="tactis"

if [ -z "${A_MAX:-}" ]; then
    echo "WARNING: A_MAX env var not set — defaulting to 5.0 (most aggressive in 0g' sweep)" >&2
    export A_MAX=5.0
fi

BASE_DIR="/user/taed7566/Forecasting/wind-forecasting"
OUTPUT_DIR="/dss/work/taed7566/Forecasting_Outputs/wind-forecasting"
export WORK_DIR="${BASE_DIR}/wind_forecasting"
export LOG_DIR="${OUTPUT_DIR}/logs"
export BASE_CONFIG="${BASE_DIR}/config/training/training_inputs_storm_awaken_smoothed_pred60_tactis_phase0h.yaml"

# Generate the per-run config with A_MAX substituted
GENERATED_CONFIG_DIR="${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}"
mkdir -p "${GENERATED_CONFIG_DIR}"
LABEL="$(echo "${A_MAX}" | sed 's/\./p/g')"
export CONFIG_FILE="${GENERATED_CONFIG_DIR}/phase0h_amax${LABEL}.yaml"
sed "s|__A_MAX__|${A_MAX}|g" "${BASE_CONFIG}" > "${CONFIG_FILE}"
echo "Generated config: ${CONFIG_FILE} (A_MAX=${A_MAX})"

export NUMEXPR_MAX_THREADS=128

mkdir -p ${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}
mkdir -p ${LOG_DIR}/checkpoints

cd ${WORK_DIR} || exit 1

export PYTHONPATH=${WORK_DIR}:/fs/dss/home/taed7566/Forecasting/pytorch-transformer-ts:${PYTHONPATH}
export WANDB_DIR=${LOG_DIR}

echo "================================================================"
echo "TACTiS-2 PHASE 0h — full v2 retrain (Stage 1 + Stage 2)"
echo "================================================================"
echo "JOB ID: ${SLURM_JOB_ID}"
echo "PARTITION: ${SLURM_JOB_PARTITION}"
echo "NODE: ${SLURM_JOB_NODELIST}"
echo "CONFIG: ${CONFIG_FILE}"
echo "A_MAX: ${A_MAX}"
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader | uniq)
echo "GPU TYPE: ${GPU_TYPE}"
echo "TIME LIMIT: 4h"
echo "Stage 1 epochs 0-19  (20 epochs — collapse evaluation window)"
echo "Stage 2 epochs 20-29 (10 epochs — copula bootstrap with val_copula_loss fix)"
echo "================================================================"
echo "Branch states:"
echo "  pytorch-transformer-ts: $(cd /fs/dss/home/taed7566/Forecasting/pytorch-transformer-ts && git rev-parse --abbrev-ref HEAD) @ $(cd /fs/dss/home/taed7566/Forecasting/pytorch-transformer-ts && git rev-parse --short HEAD)"
echo "  wind-forecasting:       $(cd ${BASE_DIR} && git rev-parse --abbrev-ref HEAD) @ $(cd ${BASE_DIR} && git rev-parse --short HEAD)"
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

echo "=== STARTING SINGLE-GPU TRAINING (1× H100, single_gpu) ==="
date +"%Y-%m-%d %H:%M:%S"

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
echo "PHASE 0h COMPLETE. Verification steps:"
echo "  Run dir: ${LOG_DIR}/train_phase0h_v2_full_decision_tactis/<run_id>/"
echo "  1. Probe end-of-Stage-1 (manual_save_epoch19): F⁻¹(U) std > 0.5"
echo "  2. Probe end-of-Stage-2 (manual_save_epoch29 + best_copula_*):"
echo "     val_copula_loss FINITE (bug fix verified) AND sample-axis std > 0.05"
echo "  3. If 1+2 pass: Phase 0h IS the full retrain — no Phase 3 needed"
echo "  4. If 1 fails:  v2 still insufficient — escalate to Phase 1 narrow pilot"
echo "  5. If only 2 fails: marginal OK but copula bootstrap problem — Stage 2 lr/wd retune"
echo "================================================================"

exit ${EXIT_CODE}
