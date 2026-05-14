#!/bin/bash

#SBATCH --partition=cfdg.p,all_gpu.p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8016
#SBATCH --gres=gpu:H100:1
#SBATCH --exclude=cfdg001
#SBATCH --time=06:00:00
#SBATCH --job-name=tactis_phase0i_e_pilot
#SBATCH --output=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/tactis_phase0i_e_pilot_%j.out
#SBATCH --error=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/tactis_phase0i_e_pilot_%j.err
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block
#SBATCH --gres-flags=enforce-binding

# =============================================================================
# TACTiS-2 PHASE 0i-E F-PILOT — NSF marginal flow (replaces DSF)
# =============================================================================
# After Phase 0i-D ES/VS pilot ALL 5 trials failed to widen F^-1 spread, the
# diagnosis localized the failure to DSF's parametrization itself. NSF's
# rational-quadratic splines with min_derivative > 0 structurally prevent
# the flat-CDF pattern that traps DSF.
#
# Variants (all NSF + plain NLL only — single-knob orthogonal sweep):
#   F0: balanced default (num_bins=32, tail_bound=4.0, num_flow_layers=2)
#   F1: lower expressivity (num_bins=16)
#   F2: higher expressivity (num_bins=64)
#   F3: wider tails (tail_bound=6.0)
#   F4: deeper composition (num_flow_layers=3)
#
# Submit ALL 5 in parallel:
#   eval "$(grep '^export LOCAL_PG_PASSWORD=' ~/.zshrc)"
#   for V in F0 F1 F2 F3 F4; do
#       PILOT_VARIANT=$V sbatch --export=ALL,LOCAL_PG_PASSWORD,PILOT_VARIANT \
#           train_phase0i_e_pilot.sh
#   done
#
# Pass criterion (per WandB after ~25 epochs):
#   - probe_sample_diversity: median F^-1 width > 1.0 (vs 0.27 broken baseline)
#   - val_loss within 5% of F0 (no NLL collapse)
#   - sample-axis std (probe) > 0.3 std-units
#
# Decision tree (post-pilot):
#   W_max > 1.0      → SELECT winner; production with NSF + NLL only (~46h)
#   0.5 ≤ W_max ≤ 1.0 → SELECT winner; production with NSF + ES (~70h)
#   W_max < 0.5      → STOP. Reassess (tail_bound? min_derivative?)
# =============================================================================

export MODEL_NAME="tactis"

if [ -z "${PILOT_VARIANT:-}" ]; then
    echo "ERROR: PILOT_VARIANT env var required (one of F0, F1, F2, F3, F4)" >&2
    exit 1
fi

case "${PILOT_VARIANT}" in
    F0) DESC="balanced default: num_bins=32, tail_bound=4.0, num_layers=2" ;;
    F1) DESC="lower expressivity: num_bins=16" ;;
    F2) DESC="higher expressivity: num_bins=64" ;;
    F3) DESC="wider tails: tail_bound=6.0" ;;
    F4) DESC="deeper composition: num_flow_layers=3" ;;
    *) echo "ERROR: unknown PILOT_VARIANT=${PILOT_VARIANT}" >&2; exit 1 ;;
esac

BASE_DIR="/user/taed7566/Forecasting/wind-forecasting"
OUTPUT_DIR="/dss/work/taed7566/Forecasting_Outputs/wind-forecasting"
export WORK_DIR="${BASE_DIR}/wind_forecasting"
export LOG_DIR="${OUTPUT_DIR}/logs"
export NUMEXPR_MAX_THREADS=128
export CONFIG_FILE="${BASE_DIR}/config/training/training_inputs_storm_awaken_unsmoothed_pred60_tactis_phase0i_e_pilot_${PILOT_VARIANT}.yaml"

mkdir -p ${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}
mkdir -p ${LOG_DIR}/checkpoints

cd ${WORK_DIR} || exit 1

export PYTHONPATH=${WORK_DIR}:/fs/dss/home/taed7566/Forecasting/pytorch-transformer-ts:${PYTHONPATH}
export WANDB_DIR=${LOG_DIR}

echo "================================================================"
echo "TACTiS-2 PHASE 0i-E F-PILOT (variant ${PILOT_VARIANT})"
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
echo "=== STARTING SINGLE-GPU TRAINING (F-pilot ${PILOT_VARIANT}) ==="

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
echo "PHASE 0i-E F-PILOT ${PILOT_VARIANT} COMPLETE"
echo "Verify:"
echo "  1. probe_sample_diversity median F^-1 width > 1.0 (was 0.27)"
echo "  2. val_loss within 5% of F0 control"
echo "  3. sample-axis std > 0.3 std-units"
echo "================================================================"

exit ${EXIT_CODE}
