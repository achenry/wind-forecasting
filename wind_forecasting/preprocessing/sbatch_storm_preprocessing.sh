#!/bin/bash
#SBATCH -p cfds.p
#SBATCH -N 16
#SBATCH -n 16
#SBATCH -c 128
#SBATCH --mem=32G
#SBATCH -o preprocessing_output.txt
#SBATCH -e preprocessing_errors.txt
#SBATCH --time=2:00:00

# Load modules
module load hpc-env/13.1
module load CUDA/12.4.0
module load Mamba/24.3.0-0

eval "$(conda shell.bash hook)"
conda activate wind_forecasting_cuda

echo "Running preprocessing..."
echo "Current working directory: $(pwd)"
echo "Current environment: $(conda env list)"
echo "Current path: $(echo $PATH)"
echo "Number of CPUs: $(nproc)"
echo "Number of nodes: $SLURM_NNODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"

python wind_forecasting/preprocessing/preprocessing_main.py --config examples/inputs/preprocessing_inputs_flasc.yaml -pd -p -m mpi

echo "Preprocessing complete."

# sbatch sbatch_storm_preprocessing.sh
# srun -p cfds.p -N 16 -n 16 -c 128 --mem=32G --time=2:00:00 python /fs/dss/home/taed7566/wind-forecasting/wind_forecasting/preprocessing/preprocessing_main.py --config /fs/dss/home/taed7566/wind-forecasting/examples/inputs/preprocessing_inputs_flasc.yaml -pd -p -m mpi
# srun -p cfds.p -N 1 -n 1 -c 128 --mem=32G --time=2:00:00 --x11 --pty bash
# python wind_forecasting/preprocessing/preprocessing_main.py --config examples/inputs/preprocessing_inputs_flasc.yaml -pd -p -m mpi

# python wind_forecasting/preprocessing/preprocessing_main.py --config examples/inputs/preprocessing_inputs_flasc.yaml -pd -ld -rf -p
