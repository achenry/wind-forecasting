# 🌪️ Wind Forecasting Project

<div align="center">

![Project Status](https://img.shields.io/badge/status-work%20in%20progress-yellow)
![Last Updated](https://img.shields.io/badge/last%20updated-October%2018%2C%202024-blue)
![Contributors](https://img.shields.io/badge/contributors-@achenry%20%7C%20@boujuan-orange)

</div>

## 🚀 Project Overview

This project focuses on wind forecasting using various deep learning models. It includes data preprocessing, model implementation, and training scripts for different architectures such as STTRE and Spacetimeformer.

### 📜 Full Project Title

**Ultra-Short-Term Probabilistic Spatio-Temporal Modeling of Wind Farm Dynamics and Disturbances for Wake Steering Control**

This open-source framework aims to predict wind speed and direction in the short term, specifically tailored for wake steering control applications in wind farms.

<details>
<summary>📚 Table of Contents</summary>

- [🌪️ Wind Forecasting Project](#️-wind-forecasting-project)
  - [🚀 Project Overview](#-project-overview)
    - [📜 Full Project Title](#-full-project-title)
  - [🛠 Setup](#-setup)
    - [Jupyter Notebook Collaboration](#jupyter-notebook-collaboration)
    - [Data](#data)
    - [Environment Setup](#environment-setup)
  - [🧠 Models](#-models)
  - [🔄 Preprocessing](#-preprocessing)
  - [🖥️ Running Jobs](#️-running-jobs)
  - [📂 Project Structure](#-project-structure)
  - [📋 Usage](#-usage)

</details>

## 🛠 Setup

### Jupyter Notebook Collaboration

To ensure consistent handling of Jupyter notebooks in this repository, please follow these steps after cloning:

1. Install `nbstripout` using Mamba:

   ```bash
   mamba install nbstripout
   ```

2. Set up `nbstripout` for this repository:

   ```bash
   nbstripout --install --attributes .gitattributes
   ```

### Data

The `examples` folder contains scripts for downloading and processing input data:

```python
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

## 🧠 Models

The `wind_forecasting/models` directory contains implementations of various forecasting models:

- TACTIS
- Spacetimeformer
- Autoformer
- Informer2020

Each model has its own subdirectory with specific implementation details and training scripts.

## 🔄 Preprocessing

The `wind_forecasting/preprocessing` folder contains scripts for data preprocessing:

- `preprocessing_main.ipynb`: Main script for loading and preprocessing data
- `load_data.sh`: Script for loading the data in the HPC
- `data_loader.py`: Contains methods for loading data
- `data_inspector.py`: Methods for plotting and analysing data
- `data_filter.py`: Methods for filtering and arranging data

## 🖥️ Running Jobs

The `rc_jobs` folder contains SLURM scripts for running jobs on HPC environments:

- `job.slurm`: General job script for NVIDIA GPUs
- `job_amd.slurm`: Job script for AMD GPUs
- `job_preprocess.slurm`: Job script for preprocessing data

To submit a job from the HPC to the cluster, use:

```bash
sbatch rc_jobs/job.slurm
```

## 📂 Project Structure

<details>
<summary>Click to expand</summary>

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

</details>

## 📋 Usage
1. Clone the repository and set up the Jupyter notebook collaboration as described in the setup section.
2. Download the required data using the script in `examples` or use your own data.
3. Set up the appropriate environment (CUDA or ROCm) using the scripts in the `install_rc` folder.
4. Preprocess the data using the script in the `wind_forecasting/preprocessing` folder.
5. Train and evaluate models using the scripts in the `wind_forecasting/models` directory.
6. For running jobs on HPC environments, use the SLURM scripts provided in the `rc_jobs` folder.

### Configuration Files
- Data Preprocessing Configuration YAML
- ML-Model Configuration YAML
- WHOC Configuration YAML
- Command Line Arguments for `wind-forecasting/wind_forecasting/preprocessing/preprocessing_main.py`, `wind-forecasting/wind_forecasting/run_scripts/load_data.py`, `wind-forecasting/wind_forecasting/run_scripts/run_model.py`, `wind-hybrid-open-controller/whoc/wind_forecast/tuning.py`, and `wind-hybrid-open-controller/whoc/case_studies/run_case_studies.py`.
- WHOC Case Study Suite in the `case_studies` dictionary defined at the top of `wind-hybrid-open-controller/whoc/case_studies/initialize_case_studies.py`.

### Preprocessing
1. Write a preprocessing configuration file similar to `wind-forecasting/examples/inputs/preprocessing_inputs_flasc.yaml`
2. Run preprocessing on a local machine with `python preprocessing_main.py --config /Users/ahenry/Documents/toolboxes/wind-forecasting/examples/inputs/preprocessing_inputs_flasc.yaml --reload_data --preprocess_data --regenerate_filters --multiprocessor cf --verbose` or on a HPC by running `wind-forecasting/wind_forecasting/preprocessing/load_data.sh`, followed by `wind-forecasting/wind_forecasting/preprocessing/preprocess_data.sh`.
3. Write a training configuration file similar to `wind-forecasting/examples/inputs/training_inputs_kestrel_flasc.yaml`.
4. Run `python wind-forecasting/wind_forecasting/run_scripts/load_data.py --config wind-forecasting/examples/inputs/training_inputs_aoifemac_flasc.yaml --reload`, or on a HPC by running `wind-forecasting/wind_forecasting/run_scripts/load_data_kestrel.sh`, to resample the data as needed, caterogize the variables, and generate train/test/val splits.

### Tuning a ML Model
1. Tune a ML model on a local machine with `python wind-forecasting/wind_forecasting/run_scripts/run_model.py --config wind-forecasting/examples/inputs/training_inputs_aoifemac_flasc.yaml --mode tune --model informer`, or on a HPC by running `wind-forecasting/wind_forecasting/run_scripts/tune_model.sh`. 

### Tuning & Training the Benchmark Models
1. Tune a statistical model on a local machine with `python wind-hybrid-open-controller/whoc/wind_forecast/tuning.py --model_config wind_forecasting/examples/inputs/training_inputs_aoifemac_flasc.yaml --data_config wind_forecasting/examples/inputs/preprocessing_inputs_flasc.yaml --model svr --study_name svr_tuning --restart_tuning`, or on a HPC by running `wind-hybrid-open-controller/whoc/wind_forecast/run_tuning_kestrel.sh [model] [model_config]`.

### Training a ML Model
1. Train a ML model on a local machine with `python wind-forecasting/wind_forecasting/run_scripts/run_model.py --config wind-forecasting/examples/inputs/training_inputs_aoifemac_flasc.yaml --mode train --model informer --use_tuned_parameters`, or on a HPC by running `wind-forecasting/wind_forecasting/run_scripts/train_model_kestrel.sh`.  

### Testing a ML Model
Method a) Test a ML model on a local machine with `python wind-hybrid-open-controller/whoc/wind_forecast/WindForecast.py --model_config wind_forecasting/examples/inputs/training_inputs_aoifemac_flasc.yaml --data_config wind_forecasting/examples/inputs/preprocessing_inputs_flasc.yaml --model kf sf svr persistence --multiprocessor cf --simulation_timestep 1 --prediction_type distribution --prediction_interval 60 300 --checkpoint /path/to/ml_chekpoint.chk`, or on a HPC by running `wind-hybrid-open-controller/whoc/wind_forecast/run_wind_forecasting.sh`.

Method b) Test a ML model on a local machine with `python wind-forecasting/wind_forecasting/run_scripts/run_model.py --config wind-forecasting/examples/inputs/training_inputs_aoifemac_flasc.yaml --mode test --model informer --checkpoint latest`, or on a HPC by running `wind-forecasting/wind_forecasting/run_scripts/test_model.sh`. 

### Testing a WindForecaster class on Wind Farm Data
1. Make predictions at a given controller sampling time intervals, for a given SCADA dataset, and a set of prediction time intervals, compute the RMSE, MAE, PINAW, CWC, CRPS, PICP scores and plot the results with `python wind-hybrid-open-controller/whoc/wind_forecast/WindForecast.py --model_config wind_forecasting/examples/inputs/training_inputs_aoifemac_flasc.yaml --data_config wind_forecasting/examples/inputs/preprocessing_inputs_flasc.yaml --model kf sf svr persistence --multiprocessor cf --simulation_timestep 1 --prediction_type distribution --prediction_interval 60 300`, or on a HPC by running `wind-hybrid-open-controller/whoc/wind_forecast/run_wind_forecasting.sh`.

### Combining a Model with a Wind Farm Controller
1. Write a WHOC configuration file similar to `wind-hybrid-open-controller/examples/hercules_input_001.yaml`. 
2. Run a case study of a yaw controller with a trained model with `python wind-hybrid-open-controller/whoc/case_studies/run_case_studies.py 15 -rs -rrs --verbose -ps -rps -ras -st auto -ns 3 -m cf -sd wind-hybrid-open-controller/examples/floris_case_studies -mcnf wind_forecasting/examples/inputs/training_inputs_aoifemac_flasc.yaml -dcnf wind_forecasting/examples/inputs/preprocessing_inputs_flasc.yaml -wcnf wind-hybrid-open-controller/examples/hercules_input_001.yaml -wf scada`, where you can fine tune parameters for a suite of cases by editing the dictionary `case_studies["baseline_controllers_preview_flasc"]` in `wind-hybrid-open-controller/whoc/case_studies/initialize_case_studies.py` and you can edit the common default parameters in the WHOC configuration file.

TODO add HPC version
---

<!-- <div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Made with ❤️ by achenry and boujuan](https://img.shields.io/badge/Made%20with%20%E2%9D%A4%EF%B8%8F%20by-achenry%20and%20boujuan-red)](https://github.com/achenry/wind-forecasting)

</div> -->