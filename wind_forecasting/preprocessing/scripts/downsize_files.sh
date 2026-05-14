#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --mem=0
#SBATCH --account=awaken
#SBATCH --time=02:00:00
#SBATCH --partition=bigmem
#SBATCH --output=load_data-%j.out
#SBATCH --exclusive

module purge
ml mamba
mamba activate wind_forecasting_env
echo $SLURM_NTASKS

python downsize_files.py