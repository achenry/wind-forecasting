#!/bin/bash

#SBATCH --partition=cfdg.p              # H100 nodes — user-requested 7-day cluster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3              # 3 H100 DDP — cfdg002 has 3 free; queues if all busy
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8016
#SBATCH --gres=gpu:H100:3                # H100 only; excludes A100 nodes
#SBATCH --exclude=cfdg001                # cfdg001 is A100; force cfdg002 H100 landing
#SBATCH --time=7-00:00                   # 7 days — cfdg.p max
#SBATCH --job-name=tactis_phase5_full
#SBATCH --output=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/tactis_phase5_full_%j.out
#SBATCH --error=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/tactis_phase5_full_%j.err
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block
#SBATCH --gres-flags=enforce-binding

# =============================================================================
# TACTiS-2 PHASE 5: FULL RETRAIN FROM SCRATCH with Trial #65 hparams
# =============================================================================
# Sequential Stage 1 + Stage 2 training — replaces the broken epoch=122 ckpt.
# Stage 1 epochs 0-29: regularized marginal training (lambda_a_reg=0.46)
# Stage 2 epochs 30-99: copula training (marginal frozen)
# DDP across 3 H100 GPUs on cfdg002 (1 H100 in use by another user blocks 4-GPU).
#
# Submit:
#   eval "$(grep '^export LOCAL_PG_PASSWORD=' ~/.zshrc)"
#   sbatch --export=ALL,LOCAL_PG_PASSWORD train_tactis_phase5_full_storm_smoothed_p60_reg.sh
# =============================================================================

export MODEL_NAME="tactis"

# --- Base Directories ---
BASE_DIR="/user/taed7566/Forecasting/wind-forecasting"
OUTPUT_DIR="/dss/work/taed7566/Forecasting_Outputs/wind-forecasting"
export WORK_DIR="${BASE_DIR}/wind_forecasting"
export LOG_DIR="${OUTPUT_DIR}/logs"
export CONFIG_FILE="${BASE_DIR}/config/training/training_inputs_storm_awaken_smoothed_pred60_tactis_phase5_full.yaml"

export NUMEXPR_MAX_THREADS=128

mkdir -p ${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}
mkdir -p ${LOG_DIR}/checkpoints

cd ${WORK_DIR} || exit 1

# pytorch-transformer-ts on PYTHONPATH for callbacks resolution
export PYTHONPATH=${WORK_DIR}:/fs/dss/home/taed7566/Forecasting/pytorch-transformer-ts:${PYTHONPATH}
export WANDB_DIR=${LOG_DIR}

echo "================================================================"
echo "TACTiS-2 PHASE 5 — FULL RETRAIN (Trial #65 hparams)"
echo "================================================================"
echo "JOB ID: ${SLURM_JOB_ID}"
echo "NODE: ${SLURM_JOB_NODELIST}"
echo "CONFIG: ${CONFIG_FILE}"
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader | uniq)
echo "GPU TYPE: ${GPU_TYPE}"
echo "TIME LIMIT: 7 days"
echo "STAGE 1 epochs 0-29  (regularized marginal)"
echo "STAGE 2 epochs 30-99 (copula on top of frozen marginal)"
echo "================================================================"
echo "Branch states:"
echo "  pytorch-transformer-ts: $(cd /fs/dss/home/taed7566/Forecasting/pytorch-transformer-ts && git rev-parse --abbrev-ref HEAD) @ $(cd /fs/dss/home/taed7566/Forecasting/pytorch-transformer-ts && git rev-parse --short HEAD)"
echo "  wind-forecasting:       $(cd ${BASE_DIR} && git rev-parse --abbrev-ref HEAD) @ $(cd ${BASE_DIR} && git rev-parse --short HEAD)"
echo "================================================================"

# --- Setup ---
module purge
module load slurm/hpc-2023/23.02.7
module load hpc-env/13.1
module load mpi4py/3.1.4-gompi-2023a
module load Mamba/24.3.0-0
module load CUDA/12.4.0
module load git
eval "$(conda shell.bash hook)"
conda activate wf_env_storm

# Verify LOCAL_PG_PASSWORD inherited
if [ -z "${LOCAL_PG_PASSWORD:-}" ]; then
    echo "ERROR: LOCAL_PG_PASSWORD not set. Submit with:"
    echo "  eval \"\$(grep '^export LOCAL_PG_PASSWORD=' ~/.zshrc)\""
    echo "  sbatch --export=ALL,LOCAL_PG_PASSWORD $0"
    exit 1
fi
export PGPASSWORD="${LOCAL_PG_PASSWORD}"

echo "=== STARTING DDP TRAINING (3× H100 srun) ==="
date +"%Y-%m-%d %H:%M:%S"

# srun launches 3 tasks (one per GPU); Lightning auto-detects DDP world_size from SLURM
srun python ${WORK_DIR}/run_scripts/run_model.py \
    --config ${CONFIG_FILE} \
    --model ${MODEL_NAME} \
    --mode train \
    --seed 42

EXIT_CODE=$?

echo "=== TRAINING FINISHED (exit ${EXIT_CODE}) ==="
date +"%Y-%m-%d %H:%M:%S"

# Move main SLURM logs into the job-ID subdirectory
for f in ${LOG_DIR}/slurm_logs/*_${SLURM_JOB_ID}.out; do
    [ -e "$f" ] && mv "$f" "${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/"
done
for f in ${LOG_DIR}/slurm_logs/*_${SLURM_JOB_ID}.err; do
    [ -e "$f" ] && mv "$f" "${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/"
done

# Reminder for verification
echo ""
echo "================================================================"
echo "PHASE 5 COMPLETE. Verification steps:"
echo "  Checkpoint dir: ${OUTPUT_DIR}/checkpoints/tactis_smoothed_60_phase5_full_trial65_v1/"
echo "  1. Identify the best_copula_*.ckpt with the lowest val_copula_loss"
echo "  2. Run probe_real_context.py:"
echo "     cd /fs/dss/home/taed7566/Forecasting/pytorch-transformer-ts"
echo "     python tools/probe_real_context.py --checkpoint <best_copula.ckpt> \\\\"
echo "       --dump_path /dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/aoife_validation/dumps/forecast_00000.pt"
echo "     PASS: F^-1(U) std > 0.5"
echo "  3. Run Aoife's inference verification (run_aoife_validation_hpc.sh-style)"
echo "     PASS: sample-axis std > 0.05; samples NOT collapsed onto loc"
echo "================================================================"

exit ${EXIT_CODE}
