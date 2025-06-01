"""
Wind Forecasting Tuning Subpackage

This subpackage provides comprehensive hyperparameter tuning functionality for wind forecasting models.
It includes Optuna-based optimization, distributed training support, comprehensive model evaluation 
capabilities, and all related utilities organized by domain.

Public API:
    - MLTuningObjective: Optuna objective function for model tuning
    - tune_model: Main function for hyperparameter optimization
    - get_tuned_params: Utility to retrieve best parameters from completed studies

Internal Modules:
    - core: Main tuning orchestration logic
    - objective: Trial execution and evaluation
    - utils: General tuning utilities
    - helpers: Trial setup and data management
    - config_utils: Configuration and database setup
    - optuna_utils: Optuna-specific utilities (visualization, persistence, etc.)
    - trial_utils: Trial error handling and protection
    - callbacks: Tuning-specific PyTorch Lightning callbacks
"""

from wind_forecasting.tuning.objective import MLTuningObjective
from wind_forecasting.tuning.core import tune_model
from wind_forecasting.utils.optuna_param_utils import get_tuned_params

# Re-export for backward compatibility and easy access
__all__ = [
    'MLTuningObjective',
    'tune_model', 
    'get_tuned_params'
]