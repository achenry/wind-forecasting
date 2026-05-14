#!/bin/bash

#SBATCH --partition=cfdg.p          # cfdg.p preferred (7-day limit). Override to all_gpu.p if cfdg busy (24h cap, fits 18h budget)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # SINGLE-GPU — same pattern as Phase 5 v1 (avoids NCCL DDP hangs)
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8016
#SBATCH --gres=gpu:H100:1            # H100 only — 16h training requires the H100 throughput
#SBATCH --exclude=cfdg001            # cfdg001 is A100 — exclude
#SBATCH --time=18:00:00              # 18h budget — ~16h actual training (matches v1's 15h53m)
#SBATCH --job-name=tactis_phase0h
#SBATCH --output=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/tactis_phase0h_%j.out
#SBATCH --error=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/tactis_phase0h_%j.err
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block
#SBATCH --gres-flags=enforce-binding

# =============================================================================
# TACTiS-2 PHASE 0h — full v2 retrain (100 epochs, matches v1 compute exactly)
# =============================================================================
# Production-grade retrain with v2 deeper-fix defenses + Phase 0g' winning
# a_max=3. If F⁻¹(U) median > 0.05 at end of Stage 1 AND sample-axis std > 0.05
# at end of Stage 2 (Aoife's metric), this IS the deployable model.
#
# Why 100 epochs (not the earlier 30-epoch decision-only design):
#   - v1 spent 30 ep Stage 1 + 70 ep Stage 2; we match for clean A/B comparison
#   - Trial #65's lr schedule is calibrated for 100 epochs — shorter runs decay
#     lr too fast and the model can't fully fit
#   - With v2's tight a_max=3 cap, the model needs more time to learn DIFFERENT
#     ways to fit the data (encoder must produce wider context outputs)
#
# Bug fixes baked in (vs v1):
#   - val_copula_loss=inf → reads live skip_copula flag (Stage 2 now observable)
#   - smooth a-cap (a_max=3) — mechanical bound, breaks the collapse mechanism
#   - per-(b,v) log-density penalty (lambda_log_density=0.5, log_density_max=0)
#   - lambda warmup (5× during ep 0-4, decay to 1× by ep 14)
#   - keep lambda_a_reg=0.46 (second line of defense)
#
# Intermediate-checkpoint probing during training (no interruption):
#   manual_save_epoch{4,9,14,19,24,29,34,...,99}.ckpt land in run dir.
#   Probe with: python tools/probe_compare_runs.py
#                  --dump <forecast_*.pt>
#                  --label phase0h_ep{N} --ckpt <run_dir>/manual_save_epoch{N}.ckpt
#   Watch for F⁻¹ median > 0.05 by ep14 (early stopping signal if not).
#
# Submit (default A_MAX=3.0 — Phase 0g' winner):
#   eval "$(grep '^export LOCAL_PG_PASSWORD=' ~/.zshrc)"
#   sbatch --export=ALL,LOCAL_PG_PASSWORD train_phase0h_v2.sh
# Or override:
#   A_MAX=5 sbatch --export=ALL,LOCAL_PG_PASSWORD,A_MAX train_phase0h_v2.sh
# =============================================================================

export MODEL_NAME="tactis"

if [ -z "${A_MAX:-}" ]; then
    echo "INFO: A_MAX env var not set — defaulting to 3.0 (Phase 0g' clear winner: F⁻¹ median 0.094 vs 0.03-0.04 for larger caps; sa_std mean 0.078 crosses Aoife threshold)" >&2
    export A_MAX=3.0
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
echo "TIME LIMIT: 18h"
echo "Stage 1 epochs 0-29  (30 epochs — full marginal training, matches v1)"
echo "Stage 2 epochs 30-99 (70 epochs — full copula bootstrap, val_copula_loss fix observable)"
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
echo "PHASE 0h COMPLETE (100 epochs). Verification steps:"
echo "  Run dir: ${LOG_DIR}/train_phase0h_v2_full_decision_tactis/<run_id>/"
echo "  1. Probe end-of-Stage-1 (manual_save_epoch29): F⁻¹(U) median > 0.05"
echo "     (target same as Aoife threshold — v1 ep29 was 0.019)"
echo "  2. Probe end-of-Stage-2 (manual_save_epoch99 + best_copula_*):"
echo "     val_copula_loss FINITE throughout (bug fix verified)"
echo "     AND Aoife sample-axis std mean > 0.05"
echo "  3. If 1+2 pass: Phase 0h IS the deployable model — no Phase 1 needed"
echo "  4. If 1 fails:  v2 still insufficient — escalate to lower a_max or different approach"
echo "  5. If only 2 fails: marginal OK but copula bootstrap problem — Stage 2 lr/wd retune"
echo ""
echo "Intermediate probing during training (do NOT wait for completion):"
echo "  cd /fs/dss/home/taed7566/Forecasting/pytorch-transformer-ts"
echo "  python tools/probe_compare_runs.py \\"
echo "    --dump /dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/aoife_validation/dumps/forecast_00000.pt \\"
echo "    --label v1_ep99 --ckpt <v1_path>/manual_save_epoch99.ckpt \\"
echo "    --label phase0h_ep4 --ckpt <run_dir>/manual_save_epoch4.ckpt \\"
echo "    --label phase0h_ep9 --ckpt <run_dir>/manual_save_epoch9.ckpt"
echo "  Early-warning thresholds:"
echo "    - F⁻¹ median at ep4 should be > 0.07 (v1 was 0.068 already collapsed)"
echo "    - F⁻¹ median at ep14 should be > 0.10 (improving)"
echo "    - F⁻¹ median at ep29 should be > 0.05 (Aoife threshold)"
echo "================================================================"

exit ${EXIT_CODE}
