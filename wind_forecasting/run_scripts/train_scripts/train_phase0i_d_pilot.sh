#!/bin/bash

#SBATCH --partition=cfdg.p,all_gpu.p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8016
#SBATCH --gres=gpu:H100:1
#SBATCH --exclude=cfdg001
#SBATCH --time=08:00:00
#SBATCH --job-name=tactis_phase0i_d_pilot
#SBATCH --output=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/tactis_phase0i_d_pilot_%j.out
#SBATCH --error=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/tactis_phase0i_d_pilot_%j.err
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block
#SBATCH --gres-flags=enforce-binding

# =============================================================================
# TACTiS-2 PHASE 0i-D E-PILOT — proper-scoring-rule loss sweep
# =============================================================================
# After Sa pilot (C0-C4) and Sw pilot (W0-W4) BOTH mechanically fixed their
# parameter pathologies but did NOT widen F^-1 spread, this pilot tests the
# loss-objective family: Energy Score (multivariate, sample-based) ± Variogram
# Score, on top of the C3 winner Sa baseline.
#
# Variants (all carry C3's Sa: lambda_a_floor=5.0, a_floor_threshold=0.5):
#   E0: control = C3 re-run, no Sd                     (Sa-only baseline)
#   E1: ES light  (lambda=0.1, N=8)                    (cheap path)
#   E2: ES heavy  (lambda=1.0, N=16)                   (expected single-shot best)
#   E3: ES + VS   (lambda_es=1.0, N=16, lambda_vs=0.5) (correlation-aware)
#   E4: ES + traj noise (lambda_es=1.0, N=16, sigma=0.05) (free addition)
#
# Submit ALL 5 in parallel:
#   eval "$(grep '^export LOCAL_PG_PASSWORD=' ~/.zshrc)"
#   for V in E0 E1 E2 E3 E4; do
#       PILOT_VARIANT=$V sbatch --export=ALL,LOCAL_PG_PASSWORD,PILOT_VARIANT \
#           train_phase0i_d_pilot.sh
#   done
#
# Pass criterion (per WandB after ~25 epochs):
#   - probe_sample_diversity: median F^-1 width > 0.5 (was 0.27 baseline)
#   - val_loss within 5% of E0 control (no NLL collapse)
#   - sample-axis std (probe) > 0.15 std-units
# =============================================================================

export MODEL_NAME="tactis"

if [ -z "${PILOT_VARIANT:-}" ]; then
    echo "ERROR: PILOT_VARIANT env var required (one of E0, E1, E2, E3, E4)" >&2
    exit 1
fi

case "${PILOT_VARIANT}" in
    E0) DESC="Sa-only control: lambda_energy_score=0" ;;
    E1) DESC="ES light: lambda_es=0.1 N=8" ;;
    E2) DESC="ES heavy: lambda_es=1.0 N=16 (expected best)" ;;
    E3) DESC="ES + VS: lambda_es=1.0 lambda_vs=0.5 N=16" ;;
    E4) DESC="ES + traj-noise: lambda_es=1.0 N=16 sigma=0.05" ;;
    *) echo "ERROR: unknown PILOT_VARIANT=${PILOT_VARIANT}" >&2; exit 1 ;;
esac

BASE_DIR="/user/taed7566/Forecasting/wind-forecasting"
OUTPUT_DIR="/dss/work/taed7566/Forecasting_Outputs/wind-forecasting"
export WORK_DIR="${BASE_DIR}/wind_forecasting"
export LOG_DIR="${OUTPUT_DIR}/logs"
export NUMEXPR_MAX_THREADS=128
export CONFIG_FILE="${BASE_DIR}/config/training/training_inputs_storm_awaken_unsmoothed_pred60_tactis_phase0i_d_pilot_${PILOT_VARIANT}.yaml"

mkdir -p ${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}
mkdir -p ${LOG_DIR}/checkpoints

cd ${WORK_DIR} || exit 1

export PYTHONPATH=${WORK_DIR}:/fs/dss/home/taed7566/Forecasting/pytorch-transformer-ts:${PYTHONPATH}
export WANDB_DIR=${LOG_DIR}

echo "================================================================"
echo "TACTiS-2 PHASE 0i-D E-PILOT (variant ${PILOT_VARIANT})"
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
echo "=== STARTING SINGLE-GPU TRAINING (E-pilot ${PILOT_VARIANT}) ==="

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
echo "PHASE 0i-D E-PILOT ${PILOT_VARIANT} COMPLETE"
echo "Verify on WandB:"
echo "  1. energy_score/loss_term should be present (if E1-E4)"
echo "  2. probe_sample_diversity median F^-1 width > 0.5 (was 0.27)"
echo "  3. val_loss within 5% of E0 control"
echo "================================================================"

exit ${EXIT_CODE}
