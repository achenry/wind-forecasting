#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=104
##SBATCH --mem=0
#SBATCH --account=ssc
#SBATCH --time=01:00:00
#SBATCH --partition=debug
#SBATCH --output=sample_sink.out
#SBATCH --tmp=1T

ml mamba
mamba activate wind_forecasting
python sample_sink.py

mv /tmp/scratch/$SLURM_JOB_ID/loaded_data.parquet /home/ahenry/toolboxes/wind_forecasting_env/wind-forecasting/wind_forecasting/run_scripts/ 
