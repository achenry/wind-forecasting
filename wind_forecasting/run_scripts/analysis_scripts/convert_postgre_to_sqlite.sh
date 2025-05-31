#!/bin/bash

#SBATCH --partition=all_cpu.p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:05:00
#SBATCH --job-name=pg_convert
#SBATCH --output=/user/taed7566/Forecasting/wind-forecasting/logs/slurm_logs/pg_convert_%j.out
#SBATCH --error=/user/taed7566/Forecasting/wind-forecasting/logs/slurm_logs/pg_convert_%j.err

export BASE_DIR="/user/taed7566/Forecasting/wind-forecasting"
export CONFIG_FILE="${BASE_DIR}/config/training/training_inputs_juan_flasc.yaml"
export OUTPUT_DIR="${BASE_DIR}/optuna/SQL"
export MODEL_NAME="tactis"
export NUMEXPR_MAX_THREADS=128

echo "--- JOB INFO ---"
echo "JOB ID: ${SLURM_JOB_ID}"
echo "JOB NAME: ${SLURM_JOB_NAME}"
echo "NODE: ${SLURM_JOB_NODELIST}"
echo "----------------"
echo "BASE_DIR: ${BASE_DIR}"
echo "CONFIG_FILE: ${CONFIG_FILE}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "MODEL_NAME: ${MODEL_NAME}"
echo "----------------"

echo "Setting up environment..."
module purge
module load slurm/hpc-2023/23.02.7
module load hpc-env/13.1
module load Mamba/24.3.0-0
echo "Modules loaded."

# Find PostgreSQL binary directory after loading the module
PG_INITDB_PATH=$(which initdb)
if [[ -z "$PG_INITDB_PATH" ]]; then
  echo "FATAL: Could not find 'initdb' after loading PostgreSQL module. Check module system." >&2
  exit 1
fi
# Extract the directory path (e.g., /path/to/postgres/bin)
export POSTGRES_BIN_DIR=$(dirname "$PG_INITDB_PATH")
echo "Found PostgreSQL bin directory: ${POSTGRES_BIN_DIR}"

eval "$(conda shell.bash hook)"
conda activate wf_env_storm
if [ $? -ne 0 ]; then
  echo "ERROR: Failed to activate conda environment 'wf_env_storm'" >&2
  exit 1
fi
echo "Conda environment 'wf_env_storm' activated."
export CAPTURED_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
echo "PATH: $PATH"
echo "which pg_ctl: $(which pg_ctl)"

cd "${BASE_DIR}" || { echo "ERROR: Failed to change directory to ${BASE_DIR}" >&2; exit 1; }
echo "Changed directory to ${BASE_DIR}"

echo "Starting PostgreSQL to SQLite conversion..."

# Start PostgreSQL Server
PGDATA_DIR="${BASE_DIR}/optuna/pgdata_flasc_tactis"
SOCKET_DIR="${BASE_DIR}/optuna/sockets/pg_socket_flasc_tactis"
LOGFILE="${PGDATA_DIR}/logfile.log"

# Ensure socket directory exists
mkdir -p "${SOCKET_DIR}"

echo "Attempting to start PostgreSQL server..."
echo "  PGDATA: ${PGDATA_DIR}"
echo "  Socket Dir: ${SOCKET_DIR}"
echo "  Log File: ${LOGFILE}"

# Start command using pg_ctl from the detected bin directory
"${POSTGRES_BIN_DIR}/pg_ctl" start -w -D "${PGDATA_DIR}" -l "${LOGFILE}" -o "-c unix_socket_directories='${SOCKET_DIR}'"

if [ $? -ne 0 ]; then
  echo "ERROR: Failed to start PostgreSQL server. Check server logs: ${LOGFILE}" >&2
  # Optionally try to stop cleanly in case of partial start
  "${POSTGRES_BIN_DIR}/pg_ctl" stop -w -D "${PGDATA_DIR}" -m fast || true
  exit 1
fi
echo "PostgreSQL server started successfully (or was already running)."
date +"%Y-%m-%d %H:%M:%S"

OUTPUT_ARG=""
if [ -n "${SQLITE_OUTPUT_DIR}" ]; then
  OUTPUT_ARG="--output_dir ${SQLITE_OUTPUT_DIR}"
fi

python wind_forecasting/run_scripts/analysis_scripts/convert_postgre_to_sqlite.py \
  --config "${CONFIG_FILE}" \
  --model_name "${MODEL_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  ${OUTPUT_ARG}

EXIT_CODE=$?

date +"%Y-%m-%d %H:%M:%S"
if [ ${EXIT_CODE} -eq 0 ]; then
  echo "Conversion script completed successfully."
else
  echo "ERROR: Conversion script failed with exit code ${EXIT_CODE}." >&2
fi
echo "---------------------------------------"

# Stop PostgreSQL Server
echo "Attempting to stop PostgreSQL server..."
"${POSTGRES_BIN_DIR}/pg_ctl" stop -w -D "${PGDATA_DIR}" -m fast
if [ $? -ne 0 ]; then
  echo "WARNING: pg_ctl stop command failed. Server might have already stopped or encountered an issue." >&2
else
  echo "PostgreSQL server stopped."
  # Clean up socket directory if stop was successful
  if [ -d "${SOCKET_DIR}" ]; then
      echo "Removing socket directory: ${SOCKET_DIR}"
      rm -rf "${SOCKET_DIR}"
  fi
fi
exit ${EXIT_CODE}