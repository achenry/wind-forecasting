# Wind Forecasting Tuning Subpackage

This subpackage contains tuning-specific functionality for hyperparameter optimization using Optuna. It follows Domain-Driven Design principles with a clean separation between core functionality and utilities.

## Architecture Overview

```
wind_forecasting/tuning/
├── __init__.py              # Public API exports
├── README.md               # This documentation
├── core.py                 # Main tune_model function
├── objective.py            # MLTuningObjective class
├── scripts/                # Standalone scripts
│   └── delete_optuna_trials.py
└── utils/                  # Tuning-specific utilities
    ├── callbacks.py        # SafePruningCallback
    ├── checkpoint_utils.py # Checkpoint loading for trials
    ├── db_utils.py         # Database management utilities
    ├── helpers.py          # Trial and callback helpers
    ├── metrics_utils.py    # Metrics evaluation for trials
    ├── optuna_utils.py     # Optuna visualizations and utilities
    ├── path_utils.py       # Path resolution utilities
    └── trial_utils.py      # Trial protection utilities

wind_forecasting/utils/      # Cross-mode utilities
├── optuna_config_utils.py  # Optuna configuration (used in tune/train/test)
├── optuna_param_utils.py   # Parameter retrieval (used in train/test)
└── optuna_storage.py       # Storage setup (used in tune/train/test)
```

## Design Philosophy

### Core vs Utilities Separation

The tuning subpackage is organized with a clear distinction:

1. **Core modules** (in root directory):
   - `objective.py`: MLTuningObjective class - the heart of trial execution
   - `core.py`: tune_model function - orchestrates the optimization process
   - `__init__.py`: Public API maintaining backward compatibility

2. **Utility modules** (in utils/ subdirectory):
   - Tuning-specific helpers and utilities
   - Internal implementation details
   - Trial management and evaluation functions

### Cross-Mode Utilities

Some utilities are used across multiple modes (tune/train/test) and are placed in the main `utils/` directory:

- **`optuna_config_utils.py`**: Database configuration used when setting up storage
- **`optuna_param_utils.py`**: Parameter retrieval used when loading tuned hyperparameters
- **`optuna_storage.py`**: Storage backend setup used for accessing Optuna studies

## Public API

The main public interface is exposed through `__init__.py`:

```python
from wind_forecasting.tuning import MLTuningObjective, tune_model, get_tuned_params
```

## Core Components

### 1. MLTuningObjective (`objective.py`)
- Main Optuna objective function class
- Handles trial execution, model training, evaluation
- Integrates with PyTorch Lightning and WandB logging
- GPU monitoring and memory management

### 2. tune_model (`core.py`) 
- Orchestrates the entire hyperparameter optimization process
- Manages distributed workers and study creation/loading
- Handles pruning strategies and visualization generation
- Integrates with PostgreSQL/SQLite backends

### 3. get_tuned_params (via `utils/optuna_param_utils.py`)
- Utility to retrieve best parameters from completed studies
- Used by train/test modes to load optimized hyperparameters
- Exposed through the public API for backward compatibility

## Tuning-Specific Utilities

Located in `tuning/utils/`:

### Trial Management
- `helpers.py`: Trial-specific utilities (seeds, data module updates, callbacks)
- `trial_utils.py`: Trial protection and OOM handling
- `callbacks.py`: SafePruningCallback for distributed tuning

### Model & Evaluation
- `checkpoint_utils.py`: Checkpoint loading with trial-specific logic
- `metrics_utils.py`: Evaluation metrics computation for trials
- `path_utils.py`: Path resolution for trial configurations

### Infrastructure
- `db_utils.py`: PostgreSQL instance management
- `optuna_utils.py`: Visualization generation and WandB integration

## Usage Patterns

### Tuning Mode
```python
from wind_forecasting.tuning import tune_model
from wind_forecasting.utils.optuna_storage import setup_optuna_storage
from wind_forecasting.utils.optuna_config_utils import generate_db_setup_params

# Setup storage and run tuning
db_params = generate_db_setup_params(model, config)
storage, _ = setup_optuna_storage(db_params, restart_tuning=False, rank=0)
best_params = tune_model(model, config, study_name, storage, ...)
```

### Training/Testing with Tuned Parameters
```python
from wind_forecasting.tuning import get_tuned_params
from wind_forecasting.utils.optuna_storage import setup_optuna_storage

# Connect to storage and load best parameters
storage, _ = setup_optuna_storage(db_params, restart_tuning=False, rank=0)
tuned_params = get_tuned_params(storage, study_name)
# Apply to model configuration
config['model'][model].update(tuned_params)
```

## Design Principles

### 1. Clean Architecture
- Core functionality separated from utilities
- Clear module boundaries and responsibilities
- Minimal coupling between components

### 2. Domain-Driven Design
- Tuning-specific logic encapsulated in the subpackage
- Cross-mode utilities properly separated
- Clear interfaces and contracts

### 3. Backward Compatibility
- Public API maintains compatibility with existing code
- Internal refactoring is transparent to users
- Import paths remain stable through `__init__.py`

## Dependencies

### Internal Dependencies
- `wind_forecasting.preprocessing.data_module`: DataModule for dataset management
- `pytorch_transformer_ts.*`: Model estimators and lightning modules
- `wind_forecasting.utils.callbacks`: General callback utilities
- `wind_forecasting.utils.optuna_*`: Cross-mode Optuna utilities

### External Dependencies
- `optuna`: Hyperparameter optimization framework
- `wandb`: Experiment tracking and logging
- `pytorch-lightning`: Training framework
- `psycopg2`: PostgreSQL driver (for distributed tuning)

## Testing

Tests are located in `/tests/test_tuning_refactor.py` and validate:
- Individual utility functions
- Integration with Optuna
- Backward compatibility
- Error handling

## Scripts

### delete_optuna_trials.py
Standalone script for cleaning up Optuna trials from databases.

Usage:
```bash
python wind_forecasting/tuning/scripts/delete_optuna_trials.py --study-name <name>
```

## Migration Notes

This subpackage was created by refactoring a monolithic 1800+ line `tuning.py` file. The refactoring:

1. **Improved Organization**: Core modules at root, utilities in subdirectory
2. **Better Separation**: Cross-mode utilities moved to main utils/
3. **Enhanced Clarity**: Clear distinction between tuning-specific and general utilities
4. **Maintained Compatibility**: All original functionality preserved

All original functionality is preserved with identical behavior.