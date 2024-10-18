# Wind Forecasting

By @achenry and @boujuan

... Work in Progress ... [10/18/2024]

## Project Overview

This project focuses on wind forecasting using various deep learning models. It includes data preprocessing, model implementation, and training scripts for different architectures such as STTRE and Spacetimeformer. [add references]

## Setup for Jupyter Notebook Collaboration

To ensure consistent handling of Jupyter notebooks in this repository, please follow these steps after cloning:

1. Install `nbstripout` using Mamba:

   ```bash
   mamba install nbstripout
   ```

2. Set up `nbstripout` for this repository:

   ```bash
   nbstripout --install --attributes .gitattributes
   ```

This setup will help maintain clean version control for Jupyter notebooks by stripping out execution counts and other output metadata before committing changes.

## Setup

### Data

The `examples` folder contains scripts for downloading and processing input data:

```python:
examples/download_flasc_data.py
```

This script downloads the SMARTEOLE wake steering experiment data and extracts it to the `inputs` folder.

### Environment Setup

The `install_rc` folder contains scripts and YAML files for setting up Python environments for both CUDA and ROCm:

- `install.sh`: Script to create and activate the environments
- `wind_forecasting_cuda.yaml`: Conda environment for CUDA-based setups
- `wind_forecasting_rocm.yaml`: Conda environment for ROCm-based setups
- `export_environments.sh`: Script to export environment configurations

To set up the environments, run:

```bash
install_rc/install.sh
```

## Models

The `wind_forecasting/models` directory contains implementations of various forecasting models:

- STTRE
- Spacetimeformer
- Autoformer
- Informer2020

Each model has its own subdirectory with specific implementation details and training scripts.

## Preprocessing

The `wind_forecasting/preprocessing` folder contains scripts for data preprocessing:

- `preprocessing_main.ipynb`: Main script for loading and preprocessing data
- `load_data.sh`: Script for loading the data in the HPC
- `data_loader.py`: Contains methods for loading data
- `data_reader.py`: Contains methods for reading and plotting
- `data_inspector.py`: Methods for plotting and analysing data
- `data_filter.py`: Methods for filtering and arranging data

## Running Jobs

The `rc_jobs` folder contains SLURM scripts for running jobs on HPC environments:

- `job.slurm`: General job script for NVIDIA GPUs
- `job_amd.slurm`: Job script for AMD GPUs
- `job_preprocess.slurm`: Job script for preprocessing data

To submit a job from the HPC to the cluster, use:

```bash
sbatch rc_jobs/job.slurm
```

## Training models

...

## Project Structure

``` markdown
wind-forecasting/
├── examples/
│   ├── download_flasc_data.py
│   ├── SCADA_SMARTEOLE_Overview.ipynb
│   └── inputs/
│       ├── awaken_data
│       └── SMARTEOLE-WFC-open-dataset
├── lut/
├── rc_jobs/
│   ├── estimate_job_start_time.sh
│   ├── job.slurm
│   ├── job_amd.slurm
│   └── job_preprocess.slurm
├── install_rc/
│   ├── export_environments.sh
│   ├── install.sh
│   ├── setup.py
│   ├── wind_forecasting_cuda.yaml
│   └── wind_forecasting_rocm.yaml
├── wind_forecasting/
│   ├── models/
│   │   ├── Autoformer/
│   │   ├── Informer2020/
│   │   └── spacetimeformer/
│   ├── preprocessing/
│   │   ├── preprocessing_main.ipynb
│   │   ├── data_loader.py
│   │   ├── data_reader.py
│   │   ├── data_inspector.py
│   │   ├── data_filter.py
│   │   └── load_data.sh
│   ├── postprocessing/
│   └── run_scripts/
│       ├── train_informer.py
│       └── train_spacetimeformer.py
├── .gitignore
├── .gitattributes
├── .gitmodules
├── STTRE.ipynb
├── STTRE.py
└── README.md
```

## Usage

1. Clone the repository and set up the Jupyter notebook collaboration as described in the setup section.
2. Download the required data using the script in `examples` or use own data.
3. Set up the appropriate environment (CUDA or ROCm) using the scripts in the `install_rc` folder.
4. Preprocess the data using the script in the `wind_forecasting/preprocessing` folder.
5. Train and evaluate models using the scripts in the `wind_forecasting/models` directory.
6. For running jobs on HPC environments, use the SLURM scripts provided in the `rc_jobs` folder.
