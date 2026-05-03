#!/bin/bash

#SBATCH --partition=all_gpu.p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8016
#SBATCH --gres=gpu:H100:4
#SBATCH --exclude=cfdg001,cfdg002
#SBATCH --time=23:59:00
#SBATCH --job-name=tactis_p4_reg
#SBATCH --output=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/tactis_p4_reg_%j.out
#SBATCH --error=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/tactis_p4_reg_%j.err
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block
#SBATCH --gres-flags=enforce-binding

# =============================================================================
# TACTiS-2 PHASE 4: SHORT TRAINING WITH PILOT-WINNER HPARAMS (Trial #43)
# =============================================================================
# Single training run (no Optuna), DDP across 4 H100 GPUs on one mpcg node,
# 50 epochs total (30 Stage 1 + 20 Stage 2). Validates that the regularizer
# winning hparams produce a sustained-healthy model BEFORE committing to a
# full retrain. Pass criteria for Phase 5 in the YAML header.
# Submit:
#   eval "$(grep '^export LOCAL_PG_PASSWORD=' ~/.zshrc)"
#   sbatch --export=ALL,LOCAL_PG_PASSWORD train_tactis_phase4_storm_smoothed_p60_reg.sh
# =============================================================================

export MODEL_NAME="tactis"

# --- Base Directories ---
BASE_DIR="/user/taed7566/Forecasting/wind-forecasting"
OUTPUT_DIR="/dss/work/taed7566/Forecasting_Outputs/wind-forecasting"
export WORK_DIR="${BASE_DIR}/wind_forecasting"
export LOG_DIR="${OUTPUT_DIR}/logs"
export CONFIG_FILE="${BASE_DIR}/config/training/training_inputs_storm_awaken_smoothed_pred60_tactis_phase4_reg.yaml"

export NUMEXPR_MAX_THREADS=128

mkdir -p ${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}
mkdir -p ${LOG_DIR}/checkpoints

cd ${WORK_DIR} || exit 1

# pytorch-transformer-ts on PYTHONPATH (matches pilot SBATCH)
export PYTHONPATH=${WORK_DIR}:/fs/dss/home/taed7566/Forecasting/pytorch-transformer-ts:${PYTHONPATH}
export WANDB_DIR=${LOG_DIR}

echo "================================================================"
echo "TACTiS-2 PHASE 4 — SHORT TRAINING (PILOT-WINNER Trial #43)"
echo "================================================================"
echo "JOB ID: ${SLURM_JOB_ID}"
echo "NODE: ${SLURM_JOB_NODELIST}"
echo "CONFIG: ${CONFIG_FILE}"
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader | uniq)
echo "GPU TYPE: ${GPU_TYPE}"
echo "================================================================"

# --- Setup ---
module purge
module load slurm/hpc-2023/23.02.7
module load hpc-env/13.1
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

echo "=== STARTING DDP TRAINING (4 GPUs) ==="
date +"%Y-%m-%d %H:%M:%S"

# Single python invocation; Lightning DDP handles the 4 ranks internally
python ${WORK_DIR}/run_scripts/run_model.py \
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

# Reminder for Phase 4 verification
echo ""
echo "================================================================"
echo "PHASE 4 COMPLETE. Verification steps:"
echo "  1. Find the saved best_copula_*.ckpt under \\\$CHECKPOINT_DIR"
echo "  2. Run probe_real_context.py on it:"
echo "     cd /fs/dss/home/taed7566/Forecasting/pytorch-transformer-ts"
echo "     python tools/probe_real_context.py --checkpoint <best_copula.ckpt> \\\\"
echo "       --dump_path /dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/aoife_validation/dumps/forecast_00000.pt"
echo "     PASS: F^-1(U) std > 0.5 on real pred_encoded"
echo "  3. Re-run Aoife's SBATCH inference (max_steps=1000) on this checkpoint"
echo "     PASS: sample-axis std > 0.05; samples NOT collapsed onto loc"
echo "  4. If both pass → submit full retrain (Phase 5)"
echo "================================================================"

exit ${EXIT_CODE}
