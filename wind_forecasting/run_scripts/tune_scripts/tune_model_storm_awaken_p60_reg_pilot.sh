#!/bin/bash

#SBATCH --partition=cfdg.p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4         # 4 H100 GPUs, one worker per GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8016
#SBATCH --gres=gpu:4
#SBATCH --time=1-12:00              # 36h ceiling for pilot (40 trials × 20 epochs / 4 workers ≈ 24h expected)
#SBATCH --job-name=p60reg_pilot_tactis
#SBATCH --output=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/p60reg_pilot_tactis_%j.out
#SBATCH --error=/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/slurm_logs/p60reg_pilot_tactis_%j.err
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block
#SBATCH --gres-flags=enforce-binding

# =============================================================================
# TACTiS-2 STAGE 1 PILOT TUNING WITH `a` REGULARIZATION
# =============================================================================
# Purpose: validate that the soft-hinge L2 penalty on DSF a_pre prevents the
# unbounded steepness explosion observed in prior training runs.
#
# Pilot scope:
#   - 40 trials total across 4 workers (~10 trials/worker)
#   - 20 epochs/trial (vs 100 for full retrain)
#   - tuning_phase=1 only (Stage 1 / marginal flow)
#   - MarginalHealthMonitor callback enabled — prunes trials whose `a` still
#     explodes despite the regularizer
#
# Pass criteria for moving to full study:
#   1. ≥ 50% trials survive (don't get pruned for max_a > 50)
#   2. Best trial: val_total_nll < -100 (sane density estimate)
#   3. Best trial: max_a in [3, 30] (regularizer working but not under-fitting)
#
# After pilot completes, run tools/probe_real_context.py against the best
# trial's checkpoint to verify F^-1(U) std > 0.1 on real pred_encoded contexts
# (the empirical "marginal not collapsed" criterion).
# =============================================================================

# --- Base Directories ---
BASE_DIR="/user/taed7566/Forecasting/wind-forecasting"
OUTPUT_DIR="/dss/work/taed7566/Forecasting_Outputs/wind-forecasting"
export WORK_DIR="${BASE_DIR}/wind_forecasting"
export LOG_DIR="${OUTPUT_DIR}/logs"
export CONFIG_FILE="${BASE_DIR}/config/training/training_inputs_juan_awaken_tune_storm_pred60_reg_pilot.yaml"
export MODEL_NAME="tactis"
export TUNING_PHASE="1"           # Stage 1 only (where the flow collapse lives)
export RESTART_TUNING_FLAG="--restart_tuning"  # Fresh study for pilot
export AUTO_EXIT_WHEN_DONE="true"
export NUMEXPR_MAX_THREADS=128

# --- Create Logging Directories ---
mkdir -p ${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}
mkdir -p ${LOG_DIR}/checkpoints

cd ${WORK_DIR} || exit 1

export PYTHONPATH=${WORK_DIR}:${PYTHONPATH}
# Make sure pytorch-transformer-ts is importable so the MarginalHealthMonitor callback resolves
export PYTHONPATH=/fs/dss/home/taed7566/Forecasting/pytorch-transformer-ts:${PYTHONPATH}
export WANDB_DIR=${LOG_DIR}

# --- Print Job Info ---
echo "================================================================"
echo "TACTiS-2 STAGE 1 PILOT TUNING WITH a-REGULARIZATION"
echo "================================================================"
echo "JOB ID: ${SLURM_JOB_ID}"
echo "JOB NAME: ${SLURM_JOB_NAME}"
echo "PARTITION: ${SLURM_JOB_PARTITION}"
echo "NODE LIST: ${SLURM_JOB_NODELIST}"
echo "NUM GPUS: ${SLURM_NTASKS_PER_NODE}"
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader | uniq)
echo "GPU TYPE: ${GPU_TYPE}"
echo "----------------------------------------------------------------"
echo "CONFIG_FILE: ${CONFIG_FILE}"
echo "MODEL_NAME: ${MODEL_NAME}"
echo "TUNING_PHASE: ${TUNING_PHASE} (Stage 1 only)"
echo "RESTART_TUNING_FLAG: '${RESTART_TUNING_FLAG}'"
echo "----------------------------------------------------------------"
echo "Branch states (record for reproducibility):"
echo "  pytorch-transformer-ts: $(cd /fs/dss/home/taed7566/Forecasting/pytorch-transformer-ts && git rev-parse --abbrev-ref HEAD) @ $(cd /fs/dss/home/taed7566/Forecasting/pytorch-transformer-ts && git rev-parse --short HEAD)"
echo "  wind-forecasting:       $(cd ${BASE_DIR} && git rev-parse --abbrev-ref HEAD) @ $(cd ${BASE_DIR} && git rev-parse --short HEAD)"
echo "================================================================"

# --- Setup Main Environment ---
echo "Setting up main environment..."
module purge
module load slurm/hpc-2023/23.02.7
module load hpc-env/13.1
module load Mamba/24.3.0-0
module load CUDA/12.4.0
module load git
echo "Modules loaded."

eval "$(conda shell.bash hook)"
conda activate wf_env_storm
echo "Conda environment 'wf_env_storm' activated."
export CAPTURED_LD_LIBRARY_PATH=$LD_LIBRARY_PATH

# Find PostgreSQL binary directory after loading the module
PG_INITDB_PATH=$(which initdb)
if [[ -z "$PG_INITDB_PATH" ]]; then
  echo "FATAL: Could not find 'initdb' after loading PostgreSQL module. Check module system." >&2
  exit 1
fi
export POSTGRES_BIN_DIR=$(dirname "$PG_INITDB_PATH")
echo "Found PostgreSQL bin directory: ${POSTGRES_BIN_DIR}"
export PGPASSWORD="${AIVEN_PG_PASSWORD}"

echo "=== STARTING PARALLEL OPTUNA TUNING WORKERS ==="
date +"%Y-%m-%d %H:%M:%S"

# --- Parallel Worker Launch using nohup ---
NUM_GPUS=${SLURM_NTASKS_PER_NODE}
export WORLD_SIZE=${NUM_GPUS}
declare -a WORKER_PIDS=()

echo "Launching ${NUM_GPUS} tuning workers..."

for i in $(seq 0 $((${NUM_GPUS}-1))); do
    CURRENT_WORKER_SEED=$((12 + i*100))
    echo "Starting worker ${i} on GPU ${i} with seed ${CURRENT_WORKER_SEED}"

    WORKER_RANK=${i} CUDA_VISIBLE_DEVICES=${i} nohup bash -c "
        echo \"Worker ${i} starting environment setup...\"
        module purge
        module load slurm/hpc-2023/23.02.7
        module load hpc-env/13.1
        module load mpi4py/3.1.4-gompi-2023a
        module load Mamba/24.3.0-0
        module load CUDA/12.4.0
        module load git

        eval \"\$(conda shell.bash hook)\"
        conda activate wf_env_storm
        echo \"Worker ${i}: Conda environment activated.\"

        echo \"Worker ${i}: Running pilot tuning with WORKER_RANK=\${WORKER_RANK}, tuning_phase=${TUNING_PHASE}...\"
        python ${WORK_DIR}/run_scripts/run_model.py \\
          --config ${CONFIG_FILE} \\
          --model ${MODEL_NAME} \\
          --mode tune \\
          --seed ${CURRENT_WORKER_SEED} \\
          --tuning_phase ${TUNING_PHASE} \\
          ${RESTART_TUNING_FLAG} \\
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

echo "--- Worker Processes Launched ---"
echo "Number of workers: ${#WORKER_PIDS[@]}"
echo "Process IDs: ${WORKER_PIDS[@]}"
echo "Check worker logs in ${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/"
echo "-------------------------------"

# --- Periodic monitoring (every 10 minutes) ---
trap "echo '--- Stopping monitor ---'; kill \$MONITOR_PID 2>/dev/null" EXIT
(
    eval "$(conda shell.bash hook)"
    conda activate wf_env_storm
    while true; do
        echo "====== STATUS: $(date +"%Y-%m-%d %H:%M:%S") ======"
        echo "Memory (GiB): $(free -g | grep Mem | awk '{print "Used:", $3, "Free:", $4}')"
        echo "GPU Status:"
        gpustat --no-header
        ALIVE_WORKERS=0
        for pid in ${WORKER_PIDS[@]}; do
            if kill -0 $pid 2>/dev/null; then
                ((ALIVE_WORKERS++))
            fi
        done
        echo "Workers: ${ALIVE_WORKERS}/${#WORKER_PIDS[@]}"
        sleep 600
    done
) &
MONITOR_PID=$!

# --- Wait for workers ---
if [ "${AUTO_EXIT_WHEN_DONE}" = "true" ]; then
    while true; do
        ALIVE_WORKERS=0
        for pid in ${WORKER_PIDS[@]}; do
            if kill -0 $pid 2>/dev/null; then
                ((ALIVE_WORKERS++))
            fi
        done
        echo "$(date +"%Y-%m-%d %H:%M:%S") - Workers running: ${ALIVE_WORKERS}/${#WORKER_PIDS[@]}"
        if [ $ALIVE_WORKERS -eq 0 ]; then
            echo "All workers finished."
            break
        fi
        sleep 60
    done
else
    wait
fi

# --- Final Status Check ---
echo "--- Worker Completion Status ---"
FAILED_WORKERS=0
SUCCESSFUL_WORKERS=0
for i in $(seq 0 $((${NUM_GPUS}-1))); do
    WORKER_LOG="${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/worker_${i}_${SLURM_JOB_ID}.log"
    if [ -f "$WORKER_LOG" ]; then
        if grep -q "COMPLETED successfully" "$WORKER_LOG"; then
            ((SUCCESSFUL_WORKERS++))
            echo "Worker ${i}: SUCCESS"
        elif grep -q "FAILED with status" "$WORKER_LOG"; then
            ((FAILED_WORKERS++))
            echo "Worker ${i}: FAILED"
        else
            ((FAILED_WORKERS++))
            echo "Worker ${i}: UNKNOWN status"
        fi
    else
        ((FAILED_WORKERS++))
        echo "Worker ${i}: log missing"
    fi
done

if [ $FAILED_WORKERS -gt 0 ]; then
    echo "SUMMARY: ${FAILED_WORKERS}/${NUM_GPUS} worker(s) failed."
    FINAL_EXIT_CODE=1
else
    echo "SUMMARY: all ${NUM_GPUS} workers succeeded."
    FINAL_EXIT_CODE=0
fi

echo "=== TUNING SCRIPT COMPLETED ==="
date +"%Y-%m-%d %H:%M:%S"

# --- Move main SLURM logs into the job-ID subdirectory ---
for f in ${LOG_DIR}/slurm_logs/*_${SLURM_JOB_ID}.out; do
    [ -e "$f" ] && mv "$f" "${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/"
done
for f in ${LOG_DIR}/slurm_logs/*_${SLURM_JOB_ID}.err; do
    [ -e "$f" ] && mv "$f" "${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/"
done

# Reminder: post-pilot verification commands
echo "================================================================"
echo "PILOT COMPLETE. Next steps:"
echo "  1. Inspect WandB study: project '${SLURM_JOB_NAME}', filter by run prefix"
echo "  2. Pull best trial hparams via wind_forecasting.utils.optuna_param_utils.get_tuned_params"
echo "  3. Run real-context probe on best-trial checkpoint:"
echo "     cd /fs/dss/home/taed7566/Forecasting/pytorch-transformer-ts"
echo "     python tools/probe_real_context.py --checkpoint <best_trial.ckpt> \\"
echo "       --dump_path /dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/aoife_validation/dumps/forecast_00000.pt"
echo "  4. Pass criteria:"
echo "       - ≥ 50% trials survived (didn't hit explode_threshold prune)"
echo "       - Best val_total_nll < -100 (not catastrophically negative like -1197)"
echo "       - Best max_a in [3, 30] (healthy regularization regime)"
echo "       - F^-1(U) std > 0.1 on real pred_encoded contexts"
echo "  5. If all pass → submit Phase 4 short training run"
echo "  6. If NOT → adjust a_reg_threshold or lambda_a_reg upper bound, re-run pilot"
echo "================================================================"

exit $FINAL_EXIT_CODE
