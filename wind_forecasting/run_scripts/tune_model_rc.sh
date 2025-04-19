#!/bin/bash

#SBATCH --partition=aa100              # NVIDIA A100 partition
#SBATCH --qos=normal                   # up to 1 day
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3            # one task per GPU
#SBATCH --cpus-per-task=20             # ~20 cores/task (3×20=60 cores)
#SBATCH --gres=gpu:3                   # 3×A100 GPUs
#SBATCH --mem-per-cpu=4000M            # ~4 GB per CPU
#SBATCH --time=1-00:00:00              # 1 day
#SBATCH --job-name=tactis_tune_flasc_sql
#SBATCH --output=${HOME}/logging/slurm_logs/tactis_tune_flasc_sql_%j.out
#SBATCH --error=${HOME}/logging/slurm_logs/tactis_tune_flasc_sql_%j.err
#SBATCH --hint=nomultithread           # disable hyperthreading
#SBATCH --distribution=block:block     # GPU–CPU affinity

# — User flags —
RESTART_TUNING_FLAG="--restart_tuning"  # set to "" to continue

# — Directories (edit as needed) —
BASE_DIR="${HOME}/wind-forecasting"
WORK_DIR="${BASE_DIR}/wind_forecasting"
LOG_DIR="${HOME}/logging"
CONFIG_FILE="${BASE_DIR}/examples/inputs/training_inputs_juan_flasc.yaml"
MODEL_NAME="tactis"

# — Prepare logging —
mkdir -p ${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}
mkdir -p ${LOG_DIR}/optuna ${LOG_DIR}/checkpoints

# — Enter work dir —
cd ${WORK_DIR} || exit 1

# — Common env vars —
export PYTHONPATH=${WORK_DIR}:${PYTHONPATH}

echo "=== SLURM JOB INFO ==="
echo " JobID:    ${SLURM_JOB_ID}"
echo " Partition:${SLURM_JOB_PARTITION}"
echo " GPUs/task:${SLURM_NTASKS_PER_NODE}"
echo " CPUs/task:${SLURM_CPUS_PER_TASK}"
echo " Mem/CPU:  ${SLURM_MEM_PER_CPU}M"
echo " TimeLim:  ${SLURM_TIMELIMIT}"
echo " BASE_DIR: ${BASE_DIR}"
echo "======================="

API_FILE="${HOME}/.wandb_api_key"
if [ -f "${API_FILE}" ]; then
  source "${API_FILE}"
else
  echo "ERROR: WANDB API‑key file not found at ${API_FILE}" >&2
  exit 1
fi

# — Load modules once —
module purge
module load slurm/alpine
module load miniforge               # lightweight conda + mamba
module load cuda/12.1.1
module load git
echo "Modules loaded."


if [ -z "${WANDB_API_KEY}" ]; then
  echo "WARNING: WANDB_API_KEY is not set. Please ensure it is configured."
fi

# — Activate conda/env & exports —
eval "$(conda shell.bash hook)"
conda activate wf_env_2              # your env name
echo "Activated conda env 'wf_env_2'."

export PYTHON_EXECUTABLE=$(which python)
echo "Using Python at: $PYTHON_EXECUTABLE"

echo "=== STARTING ${SLURM_NTASKS_PER_NODE} WORKERS via srun ==="
date +"%Y-%m-%d %H:%M:%S"

NUM_WORKERS=${SLURM_NTASKS_PER_NODE}
declare -a PIDS=()

for i in $(seq 0 $((${NUM_WORKERS}-1))); do
    SEED=$((12 + i*100))
    LOGF="${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/worker_${i}.log"

    echo "→ Worker $i | GPU $i | seed $SEED | log: $LOGF"

    srun --exclusive \
         -N1 -n1 \
         -c ${SLURM_CPUS_PER_TASK} \
         --gres=gpu:1 \
         --mem-per-cpu=${SLURM_MEM_PER_CPU} \
         --cpu-bind=cores \
         --job-name=${SLURM_JOB_NAME}_w${i} \
         --output=${LOGF} \
      bash -lc "
        export CUDA_VISIBLE_DEVICES=${i}
        export WORKER_RANK=${i}
        echo \"[Worker ${i}] Starting tuning (seed=${SEED})...\"
        $PYTHON_EXECUTABLE ${WORK_DIR}/run_scripts/run_model.py \\
          --config ${CONFIG_FILE} \\
          --model ${MODEL_NAME} \\
          --mode tune \\
          --seed ${SEED} \\
          ${RESTART_TUNING_FLAG} \\
          --single_gpu
      " &

    PIDS+=($!)
done

# wait for all srun steps
wait

echo "=== WORKERS COMPLETED ==="
FAILED=0
for i in $(seq 0 $((${NUM_WORKERS}-1))); do
    LOGF="${LOG_DIR}/slurm_logs/${SLURM_JOB_ID}/worker_${i}.log"
    if grep -q "Starting tuning" "$LOGF"; then
        echo "Worker $i: ran (check $LOGF for status)"
    else
        echo "Worker $i: did NOT start correctly"
        ((FAILED++))
    fi
done

if [ $FAILED -gt 0 ]; then
    echo "SUMMARY: $FAILED/$NUM_WORKERS workers failed to launch"
    exit 1
else
    echo "SUMMARY: All $NUM_WORKERS workers launched"
    exit 0
fi
