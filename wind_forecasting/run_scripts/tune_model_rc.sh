#!/bin/bash

#SBATCH --partition=aa100             # NVIDIA A100 nodes on Alpine
#SBATCH --qos=normal                  # Default QoS: up to 1 day
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3           # One task per GPU
#SBATCH --cpus-per-task=20            # ~20 CPUs per task (3×20=60 cores)
#SBATCH --gres=gpu:3                  # Request 3 A100 GPUs
#SBATCH --mem-per-cpu=4000M           # ~4 GB per CPU
#SBATCH --time=1-00:00:00             # 1 day
#SBATCH --job-name=tactis_tune_flasc_sql_rc
#SBATCH --output=${HOME}/logging/slurm_logs/tactis_tune_flasc_sql_rc_%j.out
#SBATCH --error=${HOME}/logging/slurm_logs/tactis_tune_flasc_sql_rc_%j.err
#SBATCH --hint=nomultithread          # Disable SMT/hyperthreading
#SBATCH --distribution=block:block    # GPU↔CPU affinity

# --- Configuration flags ---
RESTART_TUNING_FLAG="--restart_tuning"   # set to "" to continue existing study

# --- Base directories (edit as needed) ---
BASE_DIR="${HOME}/wind-forecasting"
export WORK_DIR="${BASE_DIR}/wind_forecasting"
export LOG_DIR="${HOME}/logging"
export CONFIG_FILE="${BASE_DIR}/examples/inputs/training_inputs_juan_flasc.yaml"
export MODEL_NAME="tactis"

# --- Prepare logging directories ---
mkdir -p ${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}
mkdir -p ${LOG_DIR}/optuna
mkdir -p ${LOG_DIR}/checkpoints

# --- Enter work directory ---
cd ${WORK_DIR} || exit 1

# --- Export common env vars ---
export PYTHONPATH=${WORK_DIR}:${PYTHONPATH}
export WANDB_DIR=${LOG_DIR}

echo "--- SLURM JOB INFO ---"
echo "JobID:    ${SLURM_JOB_ID}"
echo "Partition:${SLURM_JOB_PARTITION}"
echo "Nodes:    ${SLURM_JOB_NUM_NODES}"
echo "GPUs:     ${SLURM_NTASKS_PER_NODE}"
echo "CPUs/task:${SLURM_CPUS_PER_TASK}"
echo "Time Lim: ${SLURM_TIMELIMIT}"
echo "BASE_DIR: ${BASE_DIR}"
echo "LOG_DIR:  ${LOG_DIR}"
echo "Config:   ${CONFIG_FILE}"
echo "Model:    ${MODEL_NAME}"
echo "Restart:  '${RESTART_TUNING_FLAG}'"
echo "------------------------"

# --- Load modules ---
module purge
module load slurm/alpine              # Alpine Slurm instance
module load miniforge                 # Conda/Mamba support
module load cuda/12.1.1               # NVIDIA CUDA toolkit
module load git                       # Git client
echo "Modules loaded."

# --- Activate Conda env ---
eval "$(conda shell.bash hook)"
conda activate wf_env_2               # replace with your env name
echo "Conda env 'wf_env_2' activated."

# --- Set up WandB ---
export WANDB_API_KEY="<your‑token>"
export PYTHON_EXECUTABLE=$(which python)
echo "WandB API key set."
echo "Python executable: ${PYTHON_EXECUTABLE}"
echo "CUDA devices available: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "CUDA devices requested: ${SLURM_GPUS_ON_NODE}"
echo "CUDA devices assigned: ${CUDA_VISIBLE_DEVICES}"

echo "=== STARTING ${SLURM_NTASKS_PER_NODE} PARALLEL OPTUNA WORKERS ==="
date +"%Y-%m-%d %H:%M:%S"

NUM_WORKERS=${SLURM_NTASKS_PER_NODE}
declare -a PIDS=()

for i in $(seq 0 $((${NUM_WORKERS}-1))); do
    SEED=$((12 + i*100))
    echo "Launching worker $i on GPU $i with seed $SEED"

    nohup bash -c "
        export CUDA_VISIBLE_DEVICES=${i}
        export WORKER_RANK=${i}
        echo \"Worker $i: running tuning script...\"
        ${PYTHON_EXECUTABLE} ${WORK_DIR}/run_scripts/run_model.py \\
          --config ${CONFIG_FILE} \\
          --model ${MODEL_NAME} \\
          --mode tune \\
          --seed ${SEED} \\
          ${RESTART_TUNING_FLAG} \\
          --single_gpu
        EXIT_STATUS=\$?
        if [ \$EXIT_STATUS -ne 0 ]; then
            echo \"Worker $i FAILED (status \$EXIT_STATUS)\"
        else
            echo \"Worker $i COMPLETED successfully\"
        fi
        exit \$EXIT_STATUS
    " > "${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/worker_${i}.log" 2>&1 &

    PIDS+=($!)
    sleep 1
done

echo "Launched pids: ${PIDS[*]}"
wait

# --- Check worker outcomes ---
FAILED=0
for i in $(seq 0 $((${NUM_WORKERS}-1))); do
    LOGF="${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/worker_${i}.log"
    if grep -q "COMPLETED successfully" "$LOGF"; then
        echo "Worker $i: SUCCESS"
    else
        echo "Worker $i: FAILURE or UNKNOWN"
        ((FAILED++))
    fi
done

if [ $FAILED -gt 0 ]; then
    echo "SUMMARY: $FAILED/$NUM_WORKERS worker(s) failed"
    exit 1
else
    echo "SUMMARY: All $NUM_WORKERS workers succeeded"
    exit 0
fi
