#!/bin/bash

#SBATCH --partition=cfdg.p              # H100 nodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1              # SINGLE-GPU — sidesteps NCCL DDP hangs (same as v1 phase5)
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8016
#SBATCH --gres=gpu:H100:1                # 1× H100
#SBATCH --exclude=cfdg001                # cfdg001 is A100; force H100 landing
#SBATCH --time=00:30:00                  # 30 min — smoke test only (3 epochs × 200 batches)
#SBATCH --job-name=tactis_smoke_v2
#SBATCH --output=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/tactis_smoke_v2_%j.out
#SBATCH --error=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/tactis_smoke_v2_%j.err
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block
#SBATCH --gres-flags=enforce-binding

# =============================================================================
# TACTiS-2 PHASE 0d SMOKE TEST — deeper-fix v2 verification
# =============================================================================
# Quick 30-minute single-GPU run to confirm the v2 changes integrate cleanly:
#   * Fix A — per-datapoint log-density regularizer fires + logs to WandB
#   * Fix B — smooth a-cap caps max-a without zeroing gradient
#   * Fix C — schedule_mult=5.0 for epochs 0-4 (we'll see all 3 epochs)
#   * Fix D — existing lambda_a_reg still active
#
# Pass criteria (verify in WandB after job completes):
#   1. 3 epochs run to completion, EXIT_CODE=0
#   2. log_density_reg/loss_term, log_density_reg/max_log_density,
#      log_density_reg/mean_log_density, log_density_reg/schedule_mult all log
#   3. a_reg/max_a peaks LOWER than v1 baseline (~720 was broken; ~21 was Trial #65)
#      Expect to see max-a peak < 25 with a_max=20 cap active
#   4. No NaN/Inf in train_loss
#   5. Step time comparable to v1 phase5 (penalty adds <10% per-step overhead)
#
# Submit:
#   eval "$(grep '^export LOCAL_PG_PASSWORD=' ~/.zshrc)"
#   sbatch --export=ALL,LOCAL_PG_PASSWORD train_smoke_test_deeper_fix_v2.sh
# =============================================================================

export MODEL_NAME="tactis"

BASE_DIR="/user/taed7566/Forecasting/wind-forecasting"
OUTPUT_DIR="/dss/work/taed7566/Forecasting_Outputs/wind-forecasting"
export WORK_DIR="${BASE_DIR}/wind_forecasting"
export LOG_DIR="${OUTPUT_DIR}/logs"
export CONFIG_FILE="${BASE_DIR}/config/training/training_inputs_storm_awaken_smoothed_pred60_tactis_smoke_v2.yaml"

export NUMEXPR_MAX_THREADS=128

mkdir -p ${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}
mkdir -p ${LOG_DIR}/checkpoints

cd ${WORK_DIR} || exit 1

export PYTHONPATH=${WORK_DIR}:/fs/dss/home/taed7566/Forecasting/pytorch-transformer-ts:${PYTHONPATH}
export WANDB_DIR=${LOG_DIR}

echo "================================================================"
echo "TACTiS-2 PHASE 0d SMOKE TEST — deeper-fix v2"
echo "================================================================"
echo "JOB ID: ${SLURM_JOB_ID}"
echo "NODE: ${SLURM_JOB_NODELIST}"
echo "CONFIG: ${CONFIG_FILE}"
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader | uniq)
echo "GPU TYPE: ${GPU_TYPE}"
echo "TIME LIMIT: 30 min"
echo "BUDGET: 3 epochs × 200 batches × bs=64  (smoke test only)"
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
    echo "ERROR: LOCAL_PG_PASSWORD not set. Submit with:"
    echo "  eval \"\$(grep '^export LOCAL_PG_PASSWORD=' ~/.zshrc)\""
    echo "  sbatch --export=ALL,LOCAL_PG_PASSWORD $0"
    exit 1
fi
export PGPASSWORD="${LOCAL_PG_PASSWORD}"

echo "=== STARTING SMOKE TEST (1× H100, single_gpu) ==="
date +"%Y-%m-%d %H:%M:%S"

python ${WORK_DIR}/run_scripts/run_model.py \
    --config ${CONFIG_FILE} \
    --model ${MODEL_NAME} \
    --mode train \
    --seed 42 \
    --single_gpu

EXIT_CODE=$?

echo "=== SMOKE TEST FINISHED (exit ${EXIT_CODE}) ==="
date +"%Y-%m-%d %H:%M:%S"

for f in ${LOG_DIR}/slurm_logs/*_${SLURM_JOB_ID}.out; do
    [ -e "$f" ] && mv "$f" "${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/"
done
for f in ${LOG_DIR}/slurm_logs/*_${SLURM_JOB_ID}.err; do
    [ -e "$f" ] && mv "$f" "${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/"
done

echo ""
echo "================================================================"
echo "SMOKE TEST COMPLETE. Verification steps:"
echo "  1. Check WandB run for project 'smoke_test_v2_tactis_phase0d'"
echo "     Verify metrics present: log_density_reg/{loss_term, max_log_density,"
echo "       mean_log_density, schedule_mult}"
echo "  2. Check a_reg/max_a peak < 25 (smooth a-cap with a_max=20 working)"
echo "  3. No NaN/Inf in train_loss across 3 epochs"
echo "  4. EXIT_CODE=${EXIT_CODE} (must be 0)"
echo ""
echo "If all 4 pass → proceed to Phase 1 pilot v2."
echo "If any fail   → DO NOT advance. Debug locally, fix, re-smoke."
echo "================================================================"

exit ${EXIT_CODE}
