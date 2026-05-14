#!/bin/bash

#SBATCH --partition=cfdg.p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4         # 4 H100 workers in parallel
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8016
#SBATCH --gres=gpu:H100:4
#SBATCH --exclude=cfdg001            # cfdg001 is A100; force H100 (cfdg002)
#SBATCH --time=04:00:00              # FOCUSED: 4h max — actual ~2h expected (12 trials × 5 epochs × 500 batches)
#SBATCH --job-name=tactis_logd_focused
#SBATCH --output=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/tactis_logd_focused_%j.out
#SBATCH --error=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/tactis_logd_focused_%j.err
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block
#SBATCH --gres-flags=enforce-binding

# =============================================================================
# TACTiS-2 PHASE 1 FOCUSED TUNING — sweep log_density_max only
# =============================================================================
# Short Optuna study (12 trials × 5 epochs × 500 batches × bs=64) on 4 H100 in
# parallel. Lambdas held at known-good values (Trial #65 + smoke test); ONLY
# log_density_max varies across [-2, 0, 1, 3].
#
# Goal: learn at which threshold the per-datapoint penalty actually fires for
# *covert* sharpening, not just catastrophic collapse. Smoke test showed
# healthy training keeps max_log_density ≈ -5 nats. With +3 threshold, penalty
# silent until +8 nats above healthy. Lower thresholds catch sharpening earlier.
#
# After this, examine each trial's WandB metrics:
#   - log_density_reg/max_log_density (peak)
#   - log_density_reg/loss_term (did it actually fire?)
#   - a_reg/max_a (did sharpness reduce?)
#   - val_marginal_logdet (proxy for collapse — high = bad)
#
# Submit:
#   eval "$(grep '^export LOCAL_PG_PASSWORD=' ~/.zshrc)"
#   sbatch --export=ALL,LOCAL_PG_PASSWORD tune_log_density_max_focused.sh --restart_tuning
# =============================================================================

export MODEL_NAME="tactis"

export RESTART_TUNING_FLAG=""
for arg in "$@"; do
    if [ "$arg" == "--restart_tuning" ]; then
        export RESTART_TUNING_FLAG="--restart_tuning"
        echo "Restart tuning flag enabled - will create new study"
    fi
done

BASE_DIR="/user/taed7566/Forecasting/wind-forecasting"
OUTPUT_DIR="/dss/work/taed7566/Forecasting_Outputs/wind-forecasting"
export WORK_DIR="${BASE_DIR}/wind_forecasting"
export LOG_DIR="${OUTPUT_DIR}/logs"
export CONFIG_FILE="${BASE_DIR}/config/training/training_inputs_storm_awaken_smoothed_pred60_tactis_log_density_max_focused.yaml"

export AUTO_EXIT_WHEN_DONE="true"
export NUMEXPR_MAX_THREADS=128

mkdir -p ${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}
mkdir -p ${LOG_DIR}/checkpoints

cd ${WORK_DIR} || exit 1

export PYTHONPATH=${WORK_DIR}:/fs/dss/home/taed7566/Forecasting/pytorch-transformer-ts:${PYTHONPATH}
export WANDB_DIR=${LOG_DIR}

echo "=============================================="
echo "TACTiS-2 FOCUSED TUNING — log_density_max sweep"
echo "=============================================="
echo "JOB ID: ${SLURM_JOB_ID}"
echo "PARTITION: ${SLURM_JOB_PARTITION}"
echo "NODE: ${SLURM_JOB_NODELIST}"
echo "NUM GPUS: ${SLURM_NTASKS_PER_NODE}"
echo "----------------------------------------------"
echo "MODEL: ${MODEL_NAME}"
echo "TUNING PHASE: 1 (Stage 1 marginal-flow only)"
echo "CONFIG: ${CONFIG_FILE}"
echo "RESTART FLAG: ${RESTART_TUNING_FLAG}"
echo "----------------------------------------------"
echo "Sweep: log_density_max ∈ [-2.0, 0.0, 1.0, 3.0]"
echo "Fixed: lambda_a_reg=0.46, lambda_log_density=0.5, a_max=20"
echo "Budget: 12 trials × 5 epochs × 500 batches"
echo "=============================================="
echo "Branch states:"
echo "  pytorch-transformer-ts: $(cd /fs/dss/home/taed7566/Forecasting/pytorch-transformer-ts && git rev-parse --abbrev-ref HEAD) @ $(cd /fs/dss/home/taed7566/Forecasting/pytorch-transformer-ts && git rev-parse --short HEAD)"
echo "  wind-forecasting:       $(cd ${BASE_DIR} && git rev-parse --abbrev-ref HEAD) @ $(cd ${BASE_DIR} && git rev-parse --short HEAD)"
echo "=============================================="

module purge
module load slurm/hpc-2023/23.02.7
module load hpc-env/13.1
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

echo "=== STARTING PARALLEL FOCUSED TUNING ==="
date +"%Y-%m-%d %H:%M:%S"

NUM_GPUS=${SLURM_NTASKS_PER_NODE}
export WORLD_SIZE=${NUM_GPUS}
declare -a WORKER_PIDS=()

for i in $(seq 0 $((${NUM_GPUS}-1))); do
    CURRENT_WORKER_SEED=$((42 + i*100))

    echo "Starting worker ${i} on GPU ${i} with seed ${CURRENT_WORKER_SEED}"

    WORKER_RANK=${i} CUDA_VISIBLE_DEVICES=${i} nohup bash -c "
        module purge
        module load slurm/hpc-2023/23.02.7
        module load hpc-env/13.1
        module load mpi4py/3.1.4-gompi-2023a
        module load Mamba/24.3.0-0
        module load CUDA/12.4.0
        module load git

        eval \"\$(conda shell.bash hook)\"
        conda activate wf_env_storm

        export PGPASSWORD=\"\${LOCAL_PG_PASSWORD}\"

        echo \"Worker ${i}: Running FOCUSED tuning (log_density_max sweep)...\"

        python ${WORK_DIR}/run_scripts/run_model.py \\
          --config ${CONFIG_FILE} \\
          --model ${MODEL_NAME} \\
          --mode tune \\
          --seed ${CURRENT_WORKER_SEED} \\
          ${RESTART_TUNING_FLAG} \\
          --tuning_phase 1 \\
          --single_gpu

        status=\$?
        if [ \$status -ne 0 ]; then
            echo \"Worker ${i} FAILED with status \$status\"
        else
            echo \"Worker ${i} COMPLETED successfully\"
        fi
        exit \$status
    " > "${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/worker_${i}_${SLURM_JOB_ID}.log" 2>&1 &

    WORKER_PIDS+=($!)
    sleep 2
done

while true; do
    ALIVE_WORKERS=0
    for pid in ${WORKER_PIDS[@]}; do
        if kill -0 $pid 2>/dev/null; then
            ((ALIVE_WORKERS++))
        fi
    done

    echo "$(date +"%Y-%m-%d %H:%M:%S") - Workers still running: ${ALIVE_WORKERS}/${#WORKER_PIDS[@]}"

    if [ $ALIVE_WORKERS -eq 0 ]; then
        echo "All workers have finished."
        break
    fi
    sleep 60
done

echo "--- Worker Completion Status ---"
FAILED_WORKERS=0
SUCCESSFUL_WORKERS=0
for i in $(seq 0 $((${NUM_GPUS}-1))); do
    WORKER_LOG="${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/worker_${i}_${SLURM_JOB_ID}.log"
    if [ -f "$WORKER_LOG" ]; then
        if grep -q "COMPLETED successfully" "$WORKER_LOG"; then
            echo "Worker ${i}: SUCCESS"
            ((SUCCESSFUL_WORKERS++))
        elif grep -q "FAILED with status" "$WORKER_LOG"; then
            echo "Worker ${i}: FAILED"
            ((FAILED_WORKERS++))
        else
            echo "Worker ${i}: UNKNOWN status"
            ((FAILED_WORKERS++))
        fi
    else
        echo "Worker ${i}: FAILED (log file not found)"
        ((FAILED_WORKERS++))
    fi
done

echo "=============================================="
echo "FOCUSED TUNING COMPLETE"
echo "Successful workers: ${SUCCESSFUL_WORKERS}/${NUM_GPUS}"
echo "=============================================="
date +"%Y-%m-%d %H:%M:%S"

for f in ${LOG_DIR}/slurm_logs/*_${SLURM_JOB_ID}.out; do
    [ -e "$f" ] && mv "$f" "${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/" 2>/dev/null
done
for f in ${LOG_DIR}/slurm_logs/*_${SLURM_JOB_ID}.err; do
    [ -e "$f" ] && mv "$f" "${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/" 2>/dev/null
done

exit $FAILED_WORKERS
