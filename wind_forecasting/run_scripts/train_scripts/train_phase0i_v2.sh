#!/bin/bash

#SBATCH --partition=cfdg.p          # cfdg.p preferred (7-day); override to all_gpu.p with --partition=all_gpu.p (24h cap, fits 18h budget)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # SINGLE-GPU — Phase 0h pattern (avoids NCCL DDP hangs)
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8016
#SBATCH --gres=gpu:H100:1            # H100 only
#SBATCH --exclude=cfdg001            # cfdg001 is A100
#SBATCH --time=18:00:00              # 18h budget (~16h actual based on Phase 0h cadence)
#SBATCH --job-name=tactis_phase0i
#SBATCH --output=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/tactis_phase0i_%j.out
#SBATCH --error=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/tactis_phase0i_%j.err
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block
#SBATCH --gres-flags=enforce-binding

# =============================================================================
# TACTiS-2 PHASE 0i — calibration ablation (looser cap | unsmoothed data)
# =============================================================================
# Wraps either Phase 0i-A (smoothed + a_max=10 + λ_log=1.5) or Phase 0i-B
# (unsmoothed + a_max=3) — selected by the CONFIG_NAME env var.
#
# Submit (e.g.):
#   eval "$(grep '^export LOCAL_PG_PASSWORD=' ~/.zshrc)"
#   CONFIG_NAME=phase0i_a sbatch --export=ALL,LOCAL_PG_PASSWORD,CONFIG_NAME train_phase0i_v2.sh
#   CONFIG_NAME=phase0i_b sbatch --export=ALL,LOCAL_PG_PASSWORD,CONFIG_NAME train_phase0i_v2.sh
# =============================================================================

export MODEL_NAME="tactis"

if [ -z "${CONFIG_NAME:-}" ]; then
    echo "ERROR: CONFIG_NAME env var required (e.g. phase0i_a or phase0i_b)" >&2
    exit 1
fi

case "${CONFIG_NAME}" in
    phase0i_a)
        export CONFIG_FILE="/user/taed7566/Forecasting/wind-forecasting/config/training/training_inputs_storm_awaken_smoothed_pred60_tactis_phase0i_a.yaml"
        export RUN_DESC="recommendation #1: smoothed + a_max=10 + λ_log_density=1.5"
        ;;
    phase0i_b)
        export CONFIG_FILE="/user/taed7566/Forecasting/wind-forecasting/config/training/training_inputs_storm_awaken_unsmoothed_pred60_tactis_phase0i_b.yaml"
        export RUN_DESC="recommendation #2: unsmoothed data + a_max=3 (Phase 0h reg)"
        ;;
    *)
        echo "ERROR: unknown CONFIG_NAME=${CONFIG_NAME}; expected phase0i_a or phase0i_b" >&2
        exit 1
        ;;
esac

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
echo "TACTiS-2 PHASE 0i — calibration ablation"
echo "================================================================"
echo "JOB ID:      ${SLURM_JOB_ID}"
echo "PARTITION:   ${SLURM_JOB_PARTITION}"
echo "NODE:        ${SLURM_JOB_NODELIST}"
echo "CONFIG:      ${CONFIG_FILE}"
echo "DESC:        ${RUN_DESC}"
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | uniq)
echo "GPU TYPE:    ${GPU_TYPE}"
echo "TIME LIMIT:  18h"
echo "Stage 1: epochs 0-29 | Stage 2: epochs 30-99"
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

date +"%Y-%m-%d %H:%M:%S"
echo "=== STARTING SINGLE-GPU TRAINING (${CONFIG_NAME}) ==="

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
echo "PHASE 0i ${CONFIG_NAME} COMPLETE. Verification steps:"
echo "  1. Probe end-of-Stage-1 (manual_save_epoch29) — F⁻¹(U) median should be > 0.05"
echo "  2. Probe end-of-Stage-2 (manual_save_epoch99) — Aoife sa_std mean > 0.05"
echo "  3. Run validation harness — CRPS skill vs persistence > 0%"
echo "     For 0i-B (unsmoothed), persistence baseline is naturally weaker → easier to beat"
echo "  4. Compare to Phase 0h baseline — both ablations should improve calibration"
echo "================================================================"

exit ${EXIT_CODE}
