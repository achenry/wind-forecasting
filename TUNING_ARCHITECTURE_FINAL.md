# Wind Forecasting Tuning Architecture - Final Structure

## Overview

The tuning subpackage has been reorganized with a clean separation between:
1. **Core modules** in the root directory (main functionality)
2. **Utilities** in a utils/ subdirectory (supporting functions)
3. **Cross-mode utilities** in the main utils/ directory (used by tune/train/test modes)

## Final Directory Structure

```
wind_forecasting/
├── tuning/                          # Tuning-specific functionality
│   ├── __init__.py                 # Public API exports
│   ├── README.md                   # Documentation
│   ├── core.py                     # Main tune_model function (637 lines)
│   ├── objective.py                # MLTuningObjective class (531 lines)
│   ├── scripts/                    # Standalone scripts
│   │   └── delete_optuna_trials.py # Database cleanup script
│   └── utils/                      # Tuning-specific utilities
│       ├── callbacks.py            # SafePruningCallback
│       ├── checkpoint_utils.py     # Trial checkpoint handling
│       ├── db_utils.py             # PostgreSQL management
│       ├── helpers.py              # Trial setup helpers
│       ├── metrics_utils.py        # Trial evaluation metrics
│       ├── optuna_utils.py         # Optuna visualizations
│       ├── path_utils.py           # Path resolution
│       └── trial_utils.py          # Trial protection/OOM handling
│
└── utils/                          # General utilities
    ├── callbacks.py                # General PyTorch Lightning callbacks
    ├── optuna_config_utils.py      # Optuna DB configuration (tune/train/test)
    ├── optuna_param_utils.py       # Parameter retrieval (train/test)
    ├── optuna_storage.py           # Storage setup (tune/train/test)
    └── ...                         # Other general utilities
```

## Module Organization Rationale

### Core Modules (tuning/)
Only the essential modules remain at the root:
- **`objective.py`**: The MLTuningObjective class - core trial execution logic
- **`core.py`**: The tune_model function - main orchestration
- **`__init__.py`**: Public API for backward compatibility

### Tuning Utilities (tuning/utils/)
All supporting utilities specific to tuning:
- Trial management (helpers, callbacks, trial_utils)
- Evaluation infrastructure (checkpoint_utils, metrics_utils)
- Tuning infrastructure (db_utils, optuna_utils, path_utils)

### Cross-Mode Utilities (utils/)
Utilities used across multiple modes with descriptive "optuna_" prefix:
- **`optuna_config_utils.py`**: Database setup parameters (all modes with --use_tuned_parameters)
- **`optuna_param_utils.py`**: Best parameter retrieval (train/test modes)
- **`optuna_storage.py`**: Storage backend connection (all modes with --use_tuned_parameters)

## Usage Patterns by Mode

### Tuning Mode (`--mode tune`)
```python
# Uses tuning subpackage heavily
from wind_forecasting.tuning import tune_model
from wind_forecasting.tuning.utils.trial_utils import handle_trial_with_oom_protection
from wind_forecasting.utils.optuna_storage import setup_optuna_storage
from wind_forecasting.utils.optuna_config_utils import generate_db_setup_params
```

### Training Mode (`--mode train --use_tuned_parameters`)
```python
# Uses cross-mode utilities to load tuned parameters
from wind_forecasting.tuning import get_tuned_params
from wind_forecasting.utils.optuna_storage import setup_optuna_storage
from wind_forecasting.utils.optuna_config_utils import generate_db_setup_params
```

### Testing Mode (`--mode test --use_tuned_parameters`)
```python
# Same as training - loads tuned parameters
from wind_forecasting.tuning import get_tuned_params
from wind_forecasting.utils.optuna_storage import setup_optuna_storage
from wind_forecasting.utils.optuna_config_utils import generate_db_setup_params
```

## Benefits of This Structure

1. **Clear Organization**: 
   - Core functionality is immediately visible
   - Utilities are organized in subdirectory
   - Cross-mode utilities are clearly marked with "optuna_" prefix

2. **Proper Encapsulation**:
   - Tuning-specific logic stays in tuning/
   - Shared functionality is in appropriate location
   - No confusion about module purpose

3. **Easy Navigation**:
   - Developers can quickly find core modules
   - Supporting utilities are grouped together
   - Cross-mode dependencies are explicit

4. **Maintainability**:
   - Clear boundaries between domains
   - Easier to modify without side effects
   - Obvious where new functionality should go

## Public API

The public API remains unchanged for backward compatibility:

```python
from wind_forecasting.tuning import (
    MLTuningObjective,  # Main objective class
    tune_model,         # Orchestration function
    get_tuned_params    # Parameter retrieval (delegates to utils/)
)
```

## Key Design Decisions

1. **No utils.py in utils/**: Renamed to `param_utils.py` to avoid confusion
2. **Descriptive naming**: All Optuna-related utilities in main utils/ have "optuna_" prefix
3. **Minimal root directory**: Only core modules at package root
4. **Clear separation**: Tuning-specific vs cross-mode utilities

This architecture provides a clean, organized structure that clearly separates concerns while maintaining all original functionality.