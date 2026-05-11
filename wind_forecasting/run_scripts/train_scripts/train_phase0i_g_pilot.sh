#!/bin/bash

#SBATCH --partition=cfdg.p,all_gpu.p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8016
#SBATCH --gres=gpu:H100:1
#SBATCH --exclude=cfdg001
#SBATCH --time=06:00:00
#SBATCH --job-name=tactis_phase0i_g_pilot
#SBATCH --output=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/tactis_phase0i_g_pilot_%j.out
#SBATCH --error=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/tactis_phase0i_g_pilot_%j.err
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block
#SBATCH --gres-flags=enforce-binding

# =============================================================================
# TACTiS-2 PHASE 0i-G G-PILOT — Quantile-head marginal + pinball loss (Path 2)
# =============================================================================
# After Phase 0i-E NSF parametrization swap made collapse WORSE not better, the
# diagnosis pinned the failure on NLL × per-context-single-realization data.
# Pinball loss has K independent quantile terms that CANNOT all be minimised
# at a delta function — the model is structurally incapable of collapsing.
#
# Variants (single-knob orthogonal sweep):
#   G0: baseline           K=11, monotonic_delta, pinball only
#   G1: coarser            K=7  (tests whether fewer constraints are easier to optimise)
#   G2: finer              K=21 (tests whether richer marginal helps)
#   G3: post-hoc sort      K=11, sort instead of monotonic_delta (kills gradient through sort)
#   G4: defence-in-depth   K=11 + Energy Score λ=0.5, N=8 (in case pinball alone undershoots)
#
# Submit ALL 5 in parallel:
#   eval "$(grep '^export LOCAL_PG_PASSWORD=' ~/.zshrc)"
#   for V in G0 G1 G2 G3 G4; do
#       PILOT_VARIANT=$V sbatch --export=ALL,LOCAL_PG_PASSWORD,PILOT_VARIANT \
#           train_phase0i_g_pilot.sh
#   done
#
# Pass criterion (per WandB after ~25 epochs):
#   - probe_sample_diversity: median F^-1(0.95)-F^-1(0.05) > 1.0 std units (vs 0.27 broken baseline)
#   - val_pinball within 5% of G0 control (no objective collapse)
#   - sample-axis std on validation batch > 0.3 std units
#
# Decision tree (post-pilot):
#   W_max > 1.0       → SELECT winner; production with pinball only (~46h)
#   0.5 ≤ W_max ≤ 1.0 → SELECT G4 (or winner + ES); production +24h
#   W_max < 0.5       → STOP; ladder up to Path 3 (5s + 2h context)
# =============================================================================

export MODEL_NAME="tactis"

if [ -z "${PILOT_VARIANT:-}" ]; then
    echo "ERROR: PILOT_VARIANT env var required (one of G0, G1, G2, G3, G4)" >&2
    exit 1
fi

case "${PILOT_VARIANT}" in
    G0) DESC="baseline: K=11, monotonic_delta, pinball only" ;;
    G1) DESC="coarser: K=7, monotonic_delta" ;;
    G2) DESC="finer: K=21, monotonic_delta" ;;
    G3) DESC="K=11, post_hoc_sort crossing fix (no gradient through sort)" ;;
    G4) DESC="K=11 + Energy Score λ=0.5 N=8 (defence-in-depth)" ;;
    *) echo "ERROR: unknown PILOT_VARIANT=${PILOT_VARIANT}" >&2; exit 1 ;;
esac

BASE_DIR="/user/taed7566/Forecasting/wind-forecasting"
OUTPUT_DIR="/dss/work/taed7566/Forecasting_Outputs/wind-forecasting"
export WORK_DIR="${BASE_DIR}/wind_forecasting"
export LOG_DIR="${OUTPUT_DIR}/logs"
export NUMEXPR_MAX_THREADS=128
export CONFIG_FILE="${BASE_DIR}/config/training/training_inputs_storm_awaken_unsmoothed_pred60_tactis_phase0i_g_pilot_${PILOT_VARIANT}.yaml"

mkdir -p ${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}
mkdir -p ${LOG_DIR}/checkpoints

cd ${WORK_DIR} || exit 1

export PYTHONPATH=${WORK_DIR}:/fs/dss/home/taed7566/Forecasting/pytorch-transformer-ts:${PYTHONPATH}
export WANDB_DIR=${LOG_DIR}

echo "================================================================"
echo "TACTiS-2 PHASE 0i-G G-PILOT (variant ${PILOT_VARIANT})"
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
echo "=== STARTING SINGLE-GPU TRAINING (G-pilot ${PILOT_VARIANT}) ==="

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
echo "PHASE 0i-G G-PILOT ${PILOT_VARIANT} COMPLETE"
echo "Verify:"
echo "  1. probe_sample_diversity median F^-1 width > 1.0 (was 0.27 baseline)"
echo "  2. val_pinball within 5% of G0 control"
echo "  3. sample-axis std > 0.3 std units"
echo "================================================================"

exit ${EXIT_CODE}
