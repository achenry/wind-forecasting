# 🌪️ Wind Forecasting Framework

<div align="center">

![Project Status](https://img.shields.io/badge/status-active%20development-green)
![Last Updated](https://img.shields.io/badge/last%20updated-June%202025-blue)
![Contributors](https://img.shields.io/badge/contributors-@achenry%20%7C%20@boujuan-orange)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>

## 🚀 Project Overview

This project provides a framework to develop, train, tune, and evaluate various **deep learning models for probabilistic, multivariate wind forecasting**. It is designed to work with diverse wind farm operational datasets and facilitate integration with control systems like the [`wind-hybrid-open-controller`](https://github.com/achenry/wind-hybrid-open-controller).

The framework supports multiple forecasting architectures and is built for execution on High-Performance Computing (HPC) clusters, leveraging **Slurm** for job management and **[Optuna](https://optuna.org/)** for distributed hyperparameter optimization. While the examples use PostgreSQL, Optuna supports various backends (SQLite, MySQL, Journal Storage) via configuration.

### 🎯 Goal

To provide a flexible and scalable platform for experimenting with and deploying state-of-the-art wind forecasting models, particularly for ultra-short-term predictions relevant to wind farm control.

<details>
<summary>📚 Table of Contents</summary>

- [🌪️ Wind Forecasting Framework](#️-wind-forecasting-framework)
  - [🚀 Project Overview](#-project-overview)
    - [🎯 Goal](#-goal)
  - [🛠️ Core Technologies](#️-core-technologies)
  - [📂 Project Structure](#-project-structure)
  - [🧠 Integrated Models](#-integrated-models)
  - [⚙️ Setup](#️-setup)
    - [Environment Setup](#environment-setup)
    - [Dependencies](#dependencies)
    - [Example Data](#example-data)
  - [🔄 Workflow](#-workflow)
    - [1. Data Preprocessing](#1-data-preprocessing)
    - [2. Hyperparameter Tuning](#2-hyperparameter-tuning-hpcslurm--optuna)
    - [3. Model Training](#3-model-training)
    - [4. Model Testing](#4-model-testing)
  - [🔧 Configuration](#-configuration)
  - [💻 Usage Examples](#-usage-examples)
    - [Preprocessing](#preprocessing)
    - [Tuning (HPC)](#tuning-hpc)
    - [Training](#training)
    - [Testing](#testing)
  - [🤝 Contributing](#-contributing)
  <!-- - [📄 License](#-license) -->
  - [🙏 Acknowledgements](#-acknowledgements)
  - [📚 References](#-references)

</details>

## 🛠️ Core Technologies

This framework utilizes a modern stack for deep learning and time series analysis with a modular, domain-driven architecture:

*   **🐍 Programming Language:** Python (v3.12+)
*   **🧠 Deep Learning:**
    *   [PyTorch](https://pytorch.org/docs/stable/index.html): Primary tensor computation library.
    *   [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/): Framework for structuring training, validation, testing, checkpointing, logging, multi-GPU/distributed training (`DDP`), and callbacks.
*   **🕰️ Time Series Modeling:**
    *   [GluonTS (Fork)](https://github.com/achenry/gluonts): Provides foundational components (`PyTorchLightningEstimator`, data structures, transformations). **Note:** This project uses a specific fork.
*   **📊 Hyperparameter Optimization:**
    *   [Optuna](https://optuna.org/): Used for distributed hyperparameter tuning via configurable storage backends (PostgreSQL, SQLite, etc.), including pruning mechanisms.
*   **☁️ Distributed Computing & Scheduling:**
    *   Slurm: HPC workload manager for resource allocation and job execution via batch scripts (`.sh`).
*   **📈 Experiment Tracking & Logging:**
    *   [WandB (Weights & Biases)](https://wandb.ai/site): Used for logging metrics, parameters, and configurations.
    *   Python `logging`: Standard library for application messages.
*   **📦 Environment Management:**
    *   Conda / Mamba: Recommended for managing the Python environment.
*   **💾 Data Handling:**
    *   Polars / Pandas: Efficient data manipulation.
    *   Parquet: Recommended file format for storing processed time series data.

### 🏗️ Architecture Highlights

*   **Modular Design:** Clean separation between core functionality, tuning-specific utilities, and cross-mode components.
*   **Domain-Driven Organization:** Hyperparameter tuning is encapsulated in the `wind_forecasting.tuning` subpackage with clear APIs.
*   **Flexible Configuration:** YAML-based configuration system supporting multiple modes (tune/train/test) with shared utilities.
*   **Scalable Infrastructure:** Supports both local development and distributed HPC execution with minimal configuration changes.

## 📂 Project Structure (`wind-forecasting/`)

```
wind-forecasting/
├── 📁 config/             # YAML configurations (training, preprocessing)
│   └── training/
├── 📁 wind_forecasting/   # Core application source code
│   ├── 📁 preprocessing/  # Data loading, processing, splitting (DataModule)
│   ├── 📁 run_scripts/    # Main execution scripts (run_model.py, testing.py, etc.)
│   │   └── tune_scripts/  # Example Slurm scripts for tuning
│   ├── 📁 tuning/         # Hyperparameter optimization subpackage
│   │   ├── core.py        # Main tune_model orchestration
│   │   ├── objective.py   # MLTuningObjective class
│   │   ├── scripts/       # Standalone tuning scripts
│   │   └── utils/         # Tuning-specific utilities
│   └── 📁 utils/          # General & cross-mode utilities
│       ├── optuna_*.py    # Optuna utilities (storage, config, params) used across modes
│       └── callbacks.py   # General PyTorch Lightning callbacks
├── 📁 logs/               # Default directory for runtime outputs (Slurm, WandB, Checkpoints)
├── 📁 optuna/             # Default directory for Optuna storage artifacts (DB data, sockets)
├── 📁 examples/           # Example scripts (data download) & input configurations
│   └── inputs/           # Example configuration files & data directory
├── 📁 install_rc/         # Environment setup scripts & YAML files
├── 📄 .gitignore
├── 📄 .gitattributes
└── 📄 README.md           # This file
```

## 🧠 Integrated Models

This framework is designed to be model-agnostic. Forecasting models are implemented externally in the [`pytorch-transformer-ts`](https://github.com/boujuan/pytorch-transformer-ts) repository and integrated here. Currently supported models include:

*   **Informer**
*   **Autoformer**
*   **Spacetimeformer**
*   **TACTiS-2**

Refer to the [`pytorch-transformer-ts`](https://github.com/boujuan/pytorch-transformer-ts) repository for detailed model implementations and architectures. New models following the GluonTS/PyTorch Lightning `Estimator` pattern can be added and configured via YAML.

## ⚙️ Setup

### Environment Setup

The `install_rc/` directory provides scripts to help create the necessary Python environment using Conda or Mamba.

1.  **Navigate to the directory:**
    ```bash
    cd install_rc
    ```
2.  **Run the installation script:**
    ```bash
    ./install.sh
    ```
    This script uses the provided `.yaml` files (e.g., `wind_forecasting_cuda.yaml`) to create a Conda environment with the required dependencies.

*Note: On HPC environments, necessary system modules (CUDA, compilers, etc.) should be loaded *before* activating the Conda environment, typically within the Slurm job script.*

### Dependencies

A detailed list of dependencies can be found in the environment YAML files within `install_rc/`. Key requirements include:

*   Python 3.12+
*   PyTorch 2.x
*   PyTorch Lightning 2.x
*   Optuna
*   GluonTS (from the specified fork)
*   WandB
*   Polars
*   NumPy, Pandas
*   PyYAML
*   *... (**TODO** Add other dependencies)*

### Example Data

To test the framework, you can download and prepare the public [SMARTEOLE dataset](https://ieawindtask44.tudelft.nl/datasets/smarteole) from NREL's [FLASC repository](https://github.com/NREL/flasc).

1.  **Run the download script:**
    ```bash
    python examples/download_flasc_data.py
    ```
    This downloads the data into `examples/inputs/SMARTEOLE-WFC-open-dataset/`.
2.  Use this data path in your preprocessing configuration.

## 🔄 Workflow

The typical workflow involves these stages:

### 1. Data Preprocessing

1. Write a preprocessing configuration file similar to `wind-forecasting/examples/inputs/preprocessing_inputs_flasc.yaml`
2. Run preprocessing on a local machine with `python preprocessing_main.py --config /Users/ahenry/Documents/toolboxes/wind-forecasting/examples/inputs/preprocessing_inputs_flasc.yaml --reload_data --preprocess_data --regenerate_filters --multiprocessor cf --verbose` or on a HPC by running `wind-forecasting/wind_forecasting/preprocessing/load_data.sh`, followed by `wind-forecasting/wind_forecasting/preprocessing/preprocess_data.sh`.
3. Write a training configuration file similar to `wind-forecasting/examples/inputs/training_inputs_kestrel_flasc.yaml`.
4. Run `python wind-forecasting/wind_forecasting/preprocessing/load_data.py --config wind-forecasting/examples/inputs/training_inputs_aoifemac_flasc.yaml --reload --multiprocessor cf`, or on a HPC by running `wind-forecasting/wind_forecasting/preprocessing/scripts/load_data.sh`, to select features and resample, sort, forward fill the data.
5. Filter, smooth, normalize the data by running 

### 2. Hyperparameter Tuning (ML Models)

The framework includes a comprehensive hyperparameter tuning system using Optuna for distributed optimization. The tuning functionality is organized in the `wind_forecasting/tuning/` subpackage for maintainability and modularity.

### 2.1 Tuning the Prediction Length
+ Find the optimal prediction horizon for a set of test data by navigating to wind-hybrid-open-controller/whoc/case_studies ([wind-hybrid-open-controller/whoc/case_studies](https://github.com/achenry/wind-hybrid-open-controller/tree/feature/wind_preview)) and running `run_case_studies.py 22 --multiprocessor mpi -rs -rrs --ram_limit 65 --wf_source scada -st auto -ns 30 -sd /projects/ssc/ahenry/whoc/floris_case_studies/ -wcnf $HOME/toolboxes/wind_forecasting_env/wind-hybrid-open-controller/examples/hercules_input_001.yaml -dcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/preprocessing/preprocessing_inputs_kestrel_awaken_new.yaml -mcnf $HOME/toolboxes/wind_forecasting_env/wind-forecasting/config/training/training_inputs_kestrel_awaken_predGreedy.yaml`, or on a HPC by running `wind-hybrid-open-controller/whoc/case_studies/bash_script_kestrel_cpu.sh` with the python call containing the case index `22` uncommented. You can also use this script to tune various configuration parameters e.g. see cases `baseline_controllers_svr_forecaster_test_awaken`, `baseline_controllers_baseline_perfect0_forecasters_awaken`, and `baseline_controllers_informer_forecaster_test_awaken` in `wind-hybrid-open-controller/whoc/case_studies/initialize_case_studies.py` and use corresponding case indices (index of that keyword in the `case_families` list defined at the end of the file, passsed as the first argument to the call to `python run_case_studies.py [case index]`).

### 2.2 Tuning a ML Model
+ Update the YAML files in `wind-forecasting/config/training/` with the prediction lengths found in 2.1. Tune a ML model on a local machine with `python wind-forecasting/wind_forecasting/run_scripts/run_model.py --config wind-forecasting/examples/inputs/training_inputs_aoifemac_flasc.yaml --mode tune --model informer`, or on a HPC by running `wind-forecasting/wind_forecasting/run_scripts/tune_scripts/tune_all_kestrel.sh`, making suire that the correct YAML files are used in the argument passed to `--config`.

### 2.3 Tuning a Statistical Model
+ Tune a statistical model on a local machine with `python wind-hybrid-open-controller/whoc/wind_forecast/tuning.py --model_config wind_forecasting/examples/inputs/training_inputs_aoifemac_flasc.yaml --data_config wind_forecasting/examples/inputs/preprocessing_inputs_flasc.yaml --model svr --study_name svr_tuning --restart_tuning`, or on a HPC by running `wind-hybrid-open-controller/whoc/wind_forecast/run_scripts/run_all_tuning_kestrel.sh`, making suire that the correct YAML files are used in the argument passed to `--model_config`.

### 3. Training a ML Model
+ Train a ML model on a local machine with `python wind-forecasting/wind_forecasting/run_scripts/run_model.py --config wind-forecasting/examples/inputs/training_inputs_aoifemac_flasc.yaml --mode train --model informer --use_tuned_parameters`, or on a HPC by running `wind-forecasting/wind_forecasting/run_scripts/train_model_kestrel.sh`.  

<!-- ### 4. Testing a ML Model
+ Test a ML model on a local machine with `python wind-forecasting/wind_forecasting/run_scripts/run_model.py --config wind-forecasting/examples/inputs/training_inputs_aoifemac_flasc.yaml --mode test --model informer --checkpoint latest`, or on a HPC by running `wind-forecasting/wind_forecasting/run_scripts/test_model.sh`.  -->

### 4. Testing a WindForecaster class on Wind Farm Data
+ First, prepare the simulation datasets by running `cd wind-hybrid-open-controller/whoc/wind_forecast/run_scripts; sbatch generate_datasets.sh`. Then, make predictions at a given controller sampling time intervals, for a given SCADA dataset, and a given prediction time interval, compute the accuracy score and plot the results with `python wind-hybrid-open-controller/whoc/wind_forecast/run_forecaster_validation.py --model_config wind_forecasting/examples/inputs/training_inputs_aoifemac_flasc.yaml --data_config wind_forecasting/examples/inputs/preprocessing_inputs_flasc.yaml --model informer`, or on a HPC by running `cd wind-hybrid-open-controller/whoc/wind_forecast/run_scripts; sbatch run_wind_forecasting_cpu.sh` on `kestrel` for Persistence, Spatial Filtering, SVR, Kalman Filtering models and `cd wind-hybrid-open-controller/whoc/wind_forecast/run_scripts; ./run_all_wind_forecasting_gpu.sh` on `kestrel-gpu` for Transformer models. Pull the results from the `validation_results` folder in the directory passed to the `--save_dir`/`-sd` argument in the line `python ../run_forecaster_validation.py ...` called by `run_wind_forecasting_cpu.sh` or by `run_wind_forecasting_gpu.sh` in `run_all_wind_forecasting_gpu.sh`.

### 5. Combining a Statistical or ML Model with a Wind Farm Controller
+ Write a WHOC configuration file similar to `wind-hybrid-open-controller/examples/hercules_input_001.yaml`. Run a case study of a yaw controller with a trained model with `cd wind-hybrid-open-controller/whoc/case_studies; sbatch bash_script_kestrel_cpu.sh` on `kestrel` for Persistence, Spatial Filtering, SVR, Kalman Filtering models and `cd wind-hybrid-open-controller/whoc/case_studies; ./run_all_gpu.sh` on `kestrel-gpu` for Transformer models. You can fine tune parameters for a suite of cases by editing the dictionary `case_studies["baseline_controllers_preview_flasc"]` in `wind-hybrid-open-controller/whoc/case_studies/initialize_case_studies.py` and you can edit the common default parameters in the WHOC configuration file. Pull the results from the directory passed to the `--save_dir`/`-sd` argument in the line `srun python run_case_studies.py ...` in `bash_script_kestrel_cpu.sh` or `bash_script_kestrel_gpu.sh` called by `run_all_gpu.sh`. Generate plos and results with `cd wind-hybrid-open-controller/whoc/case_studies; python run_case_studies.py 15 16 17 18 19 20 22 -ps -rps -st auto -ns 30 -sd /Users/ahenry/Documents/toolboxes/wind-hybrid-open-controller/examples/floris_case_studies -mcnf /Users/ahenry/Documents/toolboxes/wind_forecasting/config/training/training_inputs_aoifemac_awaken_predLUT.yaml -dcnf /Users/ahenry/Documents/toolboxes/wind_forecasting/config/preprocessing/preprocessing_inputs_aoifemac_awaken_new.yaml -wcnf /Users/ahenry/Documents/toolboxes/wind-hybrid-open-controller/examples/hercules_input_001.yaml -wf scada -m cf`.

## 🔧 Configuration

Primary configuration is via YAML files in `config/training/`.

*   **Example:** `config/training/training_inputs_juan_flasc.yaml`
*   **Sections:** `experiment`, `logging`, `optuna`, `dataset`, `model` (with nested `<model_name>` keys), `callbacks`, `trainer`.
*   Supports basic variable substitution (e.g., `${logging.optuna_dir}`).

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

1.  **Configure:** Create/edit preprocessing YAML (e.g., `examples/inputs/preprocessing_inputs_flasc.yaml`).
2.  **Run:** Execute `wind_forecasting/preprocessing/preprocessing_main.py` with appropriate flags or use HPC scripts.

**Local Machine:**
```bash
python preprocessing_main.py --config examples/inputs/preprocessing_inputs_flasc.yaml --reload_data --preprocess_data --regenerate_filters --multiprocessor cf --verbose
```

**HPC System:**
```bash
# First load the data
./wind_forecasting/preprocessing/load_data.sh

# Then preprocess the data
./wind_forecasting/preprocessing/preprocess_data.sh
```

3. **Data Loading:** After preprocessing, load and prepare the data for model training:

```bash
python wind_forecasting/preprocessing/load_data.py --config ../../config/preprocessing/preprocessing_inputs_kestrel_awaken_new.yaml --reload_data --multiprocessor cf
```

**HPC System:**
```bash
./wind_forecasting/preprocessing/scripts/load_data.sh
```

### Tuning (HPC)

The framework's modular tuning system supports distributed hyperparameter optimization with PostgreSQL backend and comprehensive monitoring.

1.  **Configure:** Edit training YAML (`config/training/`) with Optuna settings.
2.  **Submit Job:** Modify and submit Slurm script (e.g., `tune_model_storm.sh`), ensuring the correct `--model <model_name>` is targeted.
    ```bash
    sbatch wind_forecasting/run_scripts/tune_scripts/tune_model_storm.sh
    ```
3.  **Monitor:** Use `squeue`, Slurm logs, WandB, and Optuna dashboard.

**Local Machine:**
```bash
python wind_forecasting/run_scripts/run_model.py --config examples/inputs/training_inputs_flasc.yaml --mode tune --model informer
```

**HPC System:**
```bash
# Use the provided tuning script
./wind_forecasting/run_scripts/tune_model.sh
```

### Training

1.  **Configure:** Edit training YAML. Set `use_tuned_parameters: true` (optional), high `limit_train_batches`, `max_epochs`.
2.  **Run:**
    ```bash
    python wind_forecasting/run_scripts/run_model.py \
      --config config/training/training_inputs_*.yaml \
      --mode train \
      --model <model_name> \
      [--use_tuned_parameters] \
      [--checkpoint <path | 'best' | 'latest'>] # To resume
    ```
    (Or use an HPC script)

**Local Machine:**
```bash
python wind_forecasting/run_scripts/run_model.py --config examples/inputs/training_inputs_flasc.yaml --mode train --model informer --use_tuned_parameters
```

**HPC System:**
```bash
# Use the provided training script
./wind_forecasting/run_scripts/train_model_kestrel.sh
```

### Testing

1.  **Configure:** Ensure training YAML points to the correct dataset config.
2.  **Run:**
    ```bash
    python wind_forecasting/run_scripts/run_model.py \
      --config config/training/training_inputs_*.yaml \
      --mode test \
      --model <model_name> \
      --checkpoint <path | 'best' | 'latest'>
    ```
    (Or use an HPC script)

**Local Machine:**
```bash
python wind_forecasting/run_scripts/run_model.py --config examples/inputs/training_inputs_flasc.yaml --mode test --model informer --checkpoint latest
```

**HPC System:**
```bash
# Use the provided testing script
./wind_forecasting/run_scripts/test_model.sh
```

## 🤝 Contributing

### Tuning & Training the Benchmark Models
1. Tune a statistical model on a local machine with `python wind-hybrid-open-controller/whoc/wind_forecast/tuning.py --model_config wind_forecasting/examples/inputs/training_inputs_aoifemac_flasc.yaml --data_config wind_forecasting/examples/inputs/preprocessing_inputs_flasc.yaml --model svr --study_name svr_tuning --restart_tuning`, or on a HPC by running `wind-hybrid-open-controller/whoc/wind_forecast/run_tuning_kestrel.sh [model] [model_config]`.

Contributions are welcome! Please follow standard Git practices (fork, branch, pull request).


<!--
## 📄 License

### Testing a ML Model
Method a) Test a ML model on a local machine with `python wind-hybrid-open-controller/whoc/wind_forecast/WindForecast.py --model_config wind_forecasting/examples/inputs/training_inputs_aoifemac_flasc.yaml --data_config wind_forecasting/examples/inputs/preprocessing_inputs_flasc.yaml --model kf sf svr persistence --multiprocessor cf --simulation_timestep 1 --prediction_type distribution --prediction_interval 60 300 --checkpoint /path/to/ml_chekpoint.chk`, or on a HPC by running `wind-hybrid-open-controller/whoc/wind_forecast/run_wind_forecasting.sh`.

Method b) Test a ML model on a local machine with `python wind-forecasting/wind_forecasting/run_scripts/run_model.py --config wind-forecasting/examples/inputs/training_inputs_aoifemac_flasc.yaml --mode test --model informer --checkpoint latest`, or on a HPC by running `wind-forecasting/wind_forecasting/run_scripts/test_model.sh`. 

### Testing a WindForecaster class on Wind Farm Data
1. Make predictions at a given controller sampling time intervals, for a given SCADA dataset, and a set of prediction time intervals, compute the RMSE, MAE, PINAW, CWC, CRPS, PICP scores and plot the results with `python wind-hybrid-open-controller/whoc/wind_forecast/WindForecast.py --model_config wind_forecasting/examples/inputs/training_inputs_aoifemac_flasc.yaml --data_config wind_forecasting/examples/inputs/preprocessing_inputs_flasc.yaml --model kf sf svr persistence --multiprocessor cf --simulation_timestep 1 --prediction_type distribution --prediction_interval 60 300`, or on a HPC by running `wind-hybrid-open-controller/whoc/wind_forecast/run_wind_forecasting.sh`.

### Combining a Model with a Wind Farm Controller
1. Write a WHOC configuration file similar to `wind-hybrid-open-controller/examples/hercules_input_001.yaml`. 
2. Run a case study of a yaw controller with a trained model with `python wind-hybrid-open-controller/whoc/case_studies/run_case_studies.py 15 -rs -rrs --verbose -ps -rps -ras -st auto -ns 3 -m cf -sd wind-hybrid-open-controller/examples/floris_case_studies -mcnf wind_forecasting/examples/inputs/training_inputs_aoifemac_flasc.yaml -dcnf wind_forecasting/examples/inputs/preprocessing_inputs_flasc.yaml -wcnf wind-hybrid-open-controller/examples/hercules_input_001.yaml -wf scada`, where you can fine tune parameters for a suite of cases by editing the dictionary `case_studies["baseline_controllers_preview_flasc"]` in `wind-hybrid-open-controller/whoc/case_studies/initialize_case_studies.py` and you can edit the common default parameters in the WHOC configuration file.
=======
This project is licensed under the MIT License - see the LICENSE file for details.
-->

## 🙏 Acknowledgements

*   Authors and developers of the integrated forecasting models and underlying libraries (PyTorch, Lightning, GluonTS, Optuna, WandB, etc.).
*   Compute resources provided by the [University of Oldenburg HPC group](https://uol.de/en/school5/sc/high-perfomance-computing/hpc-facilities/storm-mouse), [University of Colorado Boulder](https://www.colorado.edu/), and [NREL](https://www.nrel.gov/).


## 📚 References

*   **TACTiS:** Drouin, A., Marcotte, É., & Chapados, N. (2022). TACTiS: Transformer-Attentional Copulas for Time Series. *ICML*. ([Link](https://proceedings.mlr.press/v162/drouin22a.html))
*   **TACTiS-2:** Ashok, A., Marcotte, É., Zantedeschi, V., Chapados, N., & Drouin, A. (2024). TACTIS-2: Better, Faster, Simpler Attentional Copulas for Multivariate Time Series. *ICLR*. ([arXiv](https://arxiv.org/abs/2310.01327))
*   **Informer:** Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021). Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. *AAAI*. ([arXiv](https://arxiv.org/abs/2012.07436))
*   **Autoformer:** Wu, H., Xu, J., Wang, J., & Long, M. (2021). Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting. *NeurIPS*. ([arXiv](https://arxiv.org/abs/2106.13008))
*   **Spacetimeformer:** Shinde, A., et al. (2021). Spacetimeformer: Spatio-Temporal Transformer for Time Series Forecasting. ([arXiv](https://arxiv.org/abs/2109.12218))
*   **GluonTS:** Alexandrov, A., et al. (2020). GluonTS: Probabilistic Time Series Modeling in Python. *JMLR*. ([Link](http://jmlr.org/papers/v21/19-820.html))
*   **PyTorch Lightning:** ([Link](https://lightning.ai/))
*   **Optuna:** Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. *KDD*. ([Link](https://dl.acm.org/doi/10.1145/3292500.3330701))
*   **WandB:** ([Link](https://wandb.ai/))
*   **Related Repositories:**
    *   [`pytorch-transformer-ts`](https://github.com/boujuan/pytorch-transformer-ts) (Model Implementations)
    *   [`gluonts` (Fork)](https://github.com/achenry/gluonts)
    *   [`wind-hybrid-open-controller`](https://github.com/achenry/wind-hybrid-open-controller) (Downstream Application)


[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Made with ❤️ by achenry and boujuan](https://img.shields.io/badge/Made%20with%20%E2%9D%A4%EF%B8%8F%20by-achenry%20and%20boujuan-red)](https://github.com/achenry/wind-forecasting)
