# 🌪️ Wind Forecasting Framework

<div align="center">

![Project Status](https://img.shields.io/badge/status-active%20development-green)
![Last Updated](https://img.shields.io/badge/last%20updated-April%202025-blue)
![Contributors](https://img.shields.io/badge/contributors-@achenry%20%7C%20@boujuan-orange)
<!-- [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) -->

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
    - [2. Hyperparameter Tuning (HPC/Slurm + Optuna)](#2-hyperparameter-tuning-hpcslurm--optuna)
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

This framework utilizes a modern stack for deep learning and time series analysis:

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

## 📂 Project Structure (`wind-forecasting/`)

```
wind-forecasting/
├── 📁 config/             # YAML configurations (training, preprocessing)
│   └── training/
├── 📁 wind_forecasting/   # Core application source code
│   ├── 📁 preprocessing/  # Data loading, processing, splitting (DataModule)
│   ├── 📁 run_scripts/    # Main execution scripts (run_model.py, tuning.py, etc.)
│   │   └── tune_scripts/ # Example Slurm scripts for tuning
│   └── 📁 utils/          # Utility functions (Optuna DB, trial handling, etc.)
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
4. Run `python wind-forecasting/wind_forecasting/run_scripts/load_data.py --config wind-forecasting/examples/inputs/training_inputs_aoifemac_flasc.yaml --reload`, or on a HPC by running `wind-forecasting/wind_forecasting/run_scripts/load_data_kestrel.sh`, to resample the data as needed, caterogize the variables, and generate train/test/val splits.

### 2.1 Hyperparameter Tuning

1. Tune a ML model on a local machine with `python wind-forecasting/wind_forecasting/run_scripts/run_model.py --config wind-forecasting/examples/inputs/training_inputs_aoifemac_flasc.yaml --mode tune --model informer`, or on a HPC by running `wind-forecasting/wind_forecasting/run_scripts/tune_model.sh`. 

### 2.2 Tuning a Statistical Model
1. Tune a statistical model on a local machine with `python wind-hybrid-open-controller/whoc/wind_forecast/tuning.py --model_config wind_forecasting/examples/inputs/training_inputs_aoifemac_flasc.yaml --data_config wind_forecasting/examples/inputs/preprocessing_inputs_flasc.yaml --model svr --study_name svr_tuning --restart_tuning`, or on a HPC by running `wind-hybrid-open-controller/whoc/wind_forecast/run_tuning.sh [model] [number of models to tune]`.

### Training a ML Model
1. Train a ML model on a local machine with `python wind-forecasting/wind_forecasting/run_scripts/run_model.py --config wind-forecasting/examples/inputs/training_inputs_aoifemac_flasc.yaml --mode train --model informer --use_tuned_parameters`, or on a HPC by running `wind-forecasting/wind_forecasting/run_scripts/train_model_kestrel.sh`.  

### Testing a ML Model
1. Test a ML model on a local machine with `python wind-forecasting/wind_forecasting/run_scripts/run_model.py --config wind-forecasting/examples/inputs/training_inputs_aoifemac_flasc.yaml --mode test --model informer --checkpoint latest`, or on a HPC by running `wind-forecasting/wind_forecasting/run_scripts/test_model.sh`. 

### Testing a WindForecaster class on Wind Farm Data
1. Make predictions at a given controller sampling time intervals, for a given SCADA dataset, and a given prediction time interval, compute the accuracy score and plot the results with `python wind-hybrid-open-controller/whoc/wind_forecast/WindForecast.py --model_config wind_forecasting/examples/inputs/training_inputs_aoifemac_flasc.yaml --data_config wind_forecasting/examples/inputs/preprocessing_inputs_flasc.yaml --model informer`.

### Combining a Statistical or ML Model with a Wind Farm Controller
1. Write a WHOC configuration file similar to `wind-hybrid-open-controller/examples/hercules_input_001.yaml`. Run a case study of a yaw controller with a trained model with `python wind-hybrid-open-controller/whoc/case_studies/run_case_studies.py 15 -rs -rrs --verbose -ps -rps -ras -st auto -ns 3 -m cf -sd wind-hybrid-open-controller/examples/floris_case_studies -mcnf wind_forecasting/examples/inputs/training_inputs_aoifemac_flasc.yaml -dcnf wind_forecasting/examples/inputs/preprocessing_inputs_flasc.yaml -wcnf wind-hybrid-open-controller/examples/hercules_input_001.yaml -wf scada`, where you can fine tune parameters for a suite of cases by editing the dictionary `case_studies["baseline_controllers_preview_flasc"]` in `wind-hybrid-open-controller/whoc/case_studies/initialize_case_studies.py` and you can edit the common default parameters in the WHOC configuration file.

## 🔧 Configuration

Primary configuration is via YAML files in `config/training/`.

*   **Example:** `config/training/training_inputs_juan_flasc.yaml`
*   **Sections:** `experiment`, `logging`, `optuna`, `dataset`, `model` (with nested `<model_name>` keys), `callbacks`, `trainer`.
*   Supports basic variable substitution (e.g., `${logging.optuna_dir}`).

## 💻 Usage Examples

*(Refer to specific scripts and YAML files for detailed arguments)*

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
python wind_forecasting/run_scripts/load_data.py --config examples/inputs/training_inputs_flasc.yaml --reload
```

**HPC System:**
```bash
./wind_forecasting/run_scripts/load_data_kestrel.sh
```

### Tuning (HPC)

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

Contributions are welcome! Please follow standard Git practices (fork, branch, pull request).

<!--
## 📄 License

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