"""
Helper utilities for trial setup and data module configuration during hyperparameter tuning.
"""
import os
import logging
import torch
import random
import numpy as np
import pandas as pd
import inspect
import importlib
from typing import Dict, Any, List, Optional, Tuple
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from wind_forecasting.utils.callbacks import SafePruningCallback, DeadNeuronMonitor


def set_trial_seeds(trial_number: int, base_seed: int) -> int:
    """
    Set random seeds for reproducibility within each trial.
    
    Args:
        trial_number: Current trial number
        base_seed: Base seed value
        
    Returns:
        The trial-specific seed used
    """
    trial_seed = base_seed + trial_number
    torch.manual_seed(trial_seed)
    torch.cuda.manual_seed_all(trial_seed)
    random.seed(trial_seed)
    np.random.seed(trial_seed)
    logging.info(f"Set random seed for trial {trial_number} to {trial_seed}")
    return trial_seed


def update_data_module_params(
    data_module: Any,
    params: Dict[str, Any],
    config: Dict[str, Any],
    trial_number: int
) -> bool:
    """
    Update DataModule parameters based on trial tuned parameters.
    
    Args:
        data_module: DataModule instance to update
        params: Trial parameters
        config: Configuration dictionary
        trial_number: Trial number for logging
        
    Returns:
        Boolean indicating if splits need to be regenerated
    """
    needs_split_regeneration = False
    
    # Determine the frequency for the current trial
    current_freq_val = params.get('resample_freq', int(data_module.freq[:-1]))
    current_freq_str = f"{current_freq_val}s"
    
    # Get original prediction length in seconds from config (immutable)
    original_prediction_len_seconds = config["dataset"]["prediction_length"]
    
    # Calculate current prediction_length in timesteps
    current_prediction_len_timesteps = int(
        pd.Timedelta(original_prediction_len_seconds, unit="s") / pd.Timedelta(current_freq_str)
    )
    
    # Get tuned context_length_factor for this trial
    context_length_factor_trial = params.get(
        'context_length_factor', 
        config["dataset"].get("context_length_factor", 2)
    )
    
    # Calculate the tuned model context_length in timesteps
    tuned_model_context_len_timesteps = int(context_length_factor_trial * current_prediction_len_timesteps)
    
    # Update frequency if changed
    if data_module.freq != current_freq_str:
        logging.info(f"Trial {trial_number}: Updating DataModule.freq from {data_module.freq} to {current_freq_str}")
        data_module.freq = current_freq_str
        needs_split_regeneration = True
    
    # Update prediction length if changed
    if data_module.prediction_length != current_prediction_len_timesteps:
        logging.info(f"Trial {trial_number}: Updating DataModule.prediction_length (timesteps) from "
                    f"{data_module.prediction_length} to {current_prediction_len_timesteps}")
        data_module.prediction_length = current_prediction_len_timesteps
    
    # Update context length if changed
    if data_module.context_length != tuned_model_context_len_timesteps:
        logging.info(f"Trial {trial_number}: Updating DataModule.context_length (timesteps) from "
                    f"{data_module.context_length} to {tuned_model_context_len_timesteps} "
                    f"(factor: {context_length_factor_trial})")
        data_module.context_length = tuned_model_context_len_timesteps
        needs_split_regeneration = True
    
    # Update per_turbine_target if tuned
    if "per_turbine" in params and data_module.per_turbine_target != params["per_turbine"]:
        logging.info(f"Trial {trial_number}: Updating DataModule.per_turbine_target to {params['per_turbine']}")
        data_module.per_turbine_target = params["per_turbine"]
        needs_split_regeneration = True
    
    # Update batch size if present
    if "batch_size" in params and data_module.batch_size != params["batch_size"]:
        logging.info(f"Trial {trial_number}: Updating DataModule batch_size from "
                    f"{data_module.batch_size} to {params['batch_size']}")
        data_module.batch_size = params['batch_size']
    
    return needs_split_regeneration


def regenerate_data_splits(data_module: Any, trial_number: int) -> None:
    """
    Regenerate data splits if needed.
    
    Args:
        data_module: DataModule instance
        trial_number: Trial number for logging
        
    Raises:
        FileNotFoundError: If base parquet file doesn't exist
    """
    logging.info(f"Trial {trial_number}: DataModule parameters changed. Loading or generating splits.")
    data_module.set_train_ready_path()
    
    if not os.path.exists(data_module.train_ready_data_path):
        logging.error(f"Trial {trial_number}: ERROR: Base resampled Parquet file "
                     f"{data_module.train_ready_data_path} does not exist. Pre-computation issue.")
        raise FileNotFoundError(f"Missing pre-computed base Parquet file for freq {data_module.freq} "
                              f"and per_turbine {data_module.per_turbine_target}")
    
    logging.info(f"Trial {trial_number}: Calling generate_splits with reload=False. "
                f"Path: {data_module.train_ready_data_path}")
    try:
        data_module.generate_splits(save=True, reload=False, splits=["train", "val"])
    except Exception as e:
        logging.error(f"Trial {trial_number}: Error during data split generation: {e}", exc_info=True)
        raise


def prepare_feedforward_params(
    params: Dict[str, Any],
    estimator_class: type,
    config: Dict[str, Any],
    model_name: str
) -> None:
    """
    Prepare dim_feedforward parameter based on d_model.
    
    Args:
        params: Trial parameters (modified in-place)
        estimator_class: Estimator class
        config: Configuration dictionary
        model_name: Model name
    """
    estimator_sig = inspect.signature(estimator_class.__init__)
    estimator_params = [param.name for param in estimator_sig.parameters.values()]
    
    if "dim_feedforward" not in params and "d_model" in params:
        # set dim_feedforward to 4x the d_model found in this trial
        params["dim_feedforward"] = params["d_model"] * 4
    elif "d_model" in estimator_params and estimator_sig.parameters["d_model"].default is not inspect.Parameter.empty:
        # if d_model is not contained in the trial but is a parameter, get the default
        params["dim_feedforward"] = estimator_sig.parameters["d_model"].default * 4


def calculate_dynamic_limit_train_batches(
    params: Dict[str, Any],
    config: Dict[str, Any],
    base_limit_train_batches: Optional[int],
    base_batch_size: Optional[int]
) -> Optional[int]:
    """
    Calculate dynamic limit_train_batches to maintain constant total data per epoch.
    
    Args:
        params: Trial parameters
        config: Configuration dictionary
        base_limit_train_batches: Base limit_train_batches value
        base_batch_size: Base batch size
        
    Returns:
        Calculated dynamic limit_train_batches or None if not applicable
    """
    current_batch_size = params.get('batch_size', config["dataset"].get("batch_size", 128))
    
    if (base_limit_train_batches is not None and 
        base_batch_size is not None and 
        base_batch_size > 0):
        
        dynamic_limit_train_batches = max(1, round(
            base_limit_train_batches * base_batch_size / current_batch_size
        ))
        
        logging.info(f"Dynamic limit_train_batches calculation: base_limit={base_limit_train_batches}, "
                    f"base_batch_size={base_batch_size}, current_batch_size={current_batch_size}, "
                    f"calculated_limit={dynamic_limit_train_batches}")
        
        return dynamic_limit_train_batches
    
    return None


def create_trial_checkpoint_callback(
    trial_number: int,
    config: Dict[str, Any],
    model_name: str
) -> ModelCheckpoint:
    """
    Create a trial-specific ModelCheckpoint callback.
    
    Args:
        trial_number: Trial number
        config: Configuration dictionary
        model_name: Model name
        
    Returns:
        ModelCheckpoint instance
    """
    monitor_metric = config.get("trainer", {}).get("monitor_metric", "val_loss")
    checkpoint_mode = "min" if config.get("optuna", {}).get("direction", "minimize") == "minimize" else "max"
    
    # Create a unique directory for this trial's checkpoints
    chkp_dir_suffix = config.get("logging", {}).get("chkp_dir_suffix", "")
    
    checkpoint_dir = os.path.join(
        config.get("logging", {}).get("checkpoint_dir", "checkpoints"),
        f"{model_name}{chkp_dir_suffix}",
        f"trial_{trial_number}"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create a trial-specific ModelCheckpoint instance
    trial_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"trial_{trial_number}_{{epoch}}-{{step}}-{{{monitor_metric}:.2f}}",
        monitor=monitor_metric,
        mode=checkpoint_mode,
        save_top_k=1,
        save_last=True,
        verbose=True
    )
    
    logging.info(f"Trial {trial_number}: Created trial-specific ModelCheckpoint monitoring "
                f"'{monitor_metric}' (mode: {checkpoint_mode}) saving to {checkpoint_dir}")
    
    return trial_checkpoint_callback


def setup_trial_callbacks(
    trial: Any,
    config: Dict[str, Any],
    model_name: str,
    pruning_enabled: bool,
    trial_checkpoint_callback: ModelCheckpoint
) -> List[pl.Callback]:
    """
    Setup all callbacks for a trial.
    
    Args:
        trial: Optuna trial object
        config: Configuration dictionary
        model_name: Model name
        pruning_enabled: Whether pruning is enabled
        trial_checkpoint_callback: Trial-specific checkpoint callback
        
    Returns:
        List of callback instances
    """
    current_callbacks = []
    
    # Add pruning callback if enabled
    if pruning_enabled:
        pruning_monitor_metric = config.get("trainer", {}).get("monitor_metric", "val_loss")
        pruning_callback = SafePruningCallback(trial, monitor=pruning_monitor_metric)
        current_callbacks.append(pruning_callback)
        logging.info(f"Added pruning callback for trial {trial.number}, monitoring '{pruning_monitor_metric}'")
    
    # Add trial-specific ModelCheckpoint
    current_callbacks.append(trial_checkpoint_callback)
    
    # Process callbacks from configuration
    callback_configurations_dict = config.get('callbacks', {})
    general_instantiated_callbacks = []
    early_stopping_config_args = None
    
    if isinstance(callback_configurations_dict, dict):
        logging.info(f"Trial {trial.number}: Processing callback configurations from YAML: "
                    f"{list(callback_configurations_dict.keys())}")
        
        for cb_name, cb_setting in callback_configurations_dict.items():
            # Skip disabled callbacks
            if isinstance(cb_setting, dict) and cb_setting.get('enabled', True) is False:
                logging.info(f"Trial {trial.number}: Skipping disabled callback from config: {cb_name}")
                continue
            
            # Handle specific callbacks
            if cb_name == 'model_checkpoint':
                logging.debug(f"Trial {trial.number}: Skipping '{cb_name}' config, handled by trial-specific ModelCheckpoint.")
                continue
            
            if cb_name == 'early_stopping':
                if isinstance(cb_setting, dict) and early_stopping_config_args is None:
                    early_stopping_config_args = cb_setting.get('init_args', {}).copy()
                    early_stopping_config_args.setdefault('monitor', config.get("trainer", {}).get("monitor_metric", "val_loss"))
                    early_stopping_config_args.setdefault('mode', "min" if config.get("optuna", {}).get("direction", "minimize") == "minimize" else "max")
                    logging.info(f"Trial {trial.number}: Captured EarlyStopping args from '{cb_name}' config: {early_stopping_config_args}")
                continue
            
            if cb_name == 'dead_neuron_monitor':
                logging.debug(f"Trial {trial.number}: Skipping '{cb_name}' config here, will be handled separately.")
                continue
            
            # Instantiate general callbacks
            if isinstance(cb_setting, dict) and 'class_path' in cb_setting:
                try:
                    callback_instance = instantiate_callback_from_config(cb_setting, config, trial.number)
                    general_instantiated_callbacks.append(callback_instance)
                    logging.info(f"Trial {trial.number}: Instantiated general callback from config (name: {cb_name}).")
                except Exception as e:
                    logging.error(f"Trial {trial.number}: Error instantiating general callback '{cb_name}' from config: {e}", 
                                exc_info=True)
            
            # Handle simple boolean flags
            elif isinstance(cb_setting, bool) and cb_setting is True:
                callback_instance = instantiate_simple_callback(cb_name, callback_configurations_dict, trial.number)
                if callback_instance:
                    general_instantiated_callbacks.append(callback_instance)
    
    # Add EarlyStopping if configured
    if early_stopping_config_args:
        trial_early_stopping_callback = pl.callbacks.EarlyStopping(**early_stopping_config_args)
        current_callbacks.append(trial_early_stopping_callback)
        logging.info(f"Trial {trial.number}: Created trial-specific EarlyStopping from captured config.")
    
    # Add DeadNeuronMonitor if configured
    dead_neuron_monitor_setting = callback_configurations_dict.get('dead_neuron_monitor', {})
    if isinstance(dead_neuron_monitor_setting, dict) and dead_neuron_monitor_setting.get('enabled', False):
        dnm_init_args = dead_neuron_monitor_setting.get('init_args', {})
        dead_neuron_callback = DeadNeuronMonitor(**dnm_init_args)
        current_callbacks.append(dead_neuron_callback)
        logging.info(f"Trial {trial.number}: Added DeadNeuronMonitor callback from config.")
    elif isinstance(dead_neuron_monitor_setting, bool) and dead_neuron_monitor_setting is True:
        dead_neuron_callback = DeadNeuronMonitor()
        current_callbacks.append(dead_neuron_callback)
        logging.info(f"Trial {trial.number}: Added DeadNeuronMonitor callback (enabled by boolean flag).")
    
    return general_instantiated_callbacks + current_callbacks


def instantiate_callback_from_config(
    cb_setting: Dict[str, Any],
    config: Dict[str, Any],
    trial_number: int
) -> pl.Callback:
    """
    Instantiate a callback from configuration dictionary.
    
    Args:
        cb_setting: Callback configuration
        config: Full configuration dictionary
        trial_number: Trial number for logging
        
    Returns:
        Instantiated callback
    """
    module_path, class_name = cb_setting['class_path'].rsplit('.', 1)
    CallbackClass = getattr(importlib.import_module(module_path), class_name)
    init_args = cb_setting.get('init_args', {})
    
    # Resolve paths if needed
    if 'dirpath' in init_args and isinstance(init_args['dirpath'], str):
        if '${logging.checkpoint_dir}' in init_args['dirpath']:
            base_chkpt_dir = config.get("logging", {}).get("checkpoint_dir", "checkpoints")
            init_args['dirpath'] = init_args['dirpath'].replace('${logging.checkpoint_dir}', base_chkpt_dir)
        if not os.path.isabs(init_args['dirpath']):
            project_root = config.get('experiment', {}).get('project_root', '.')
            init_args['dirpath'] = os.path.abspath(os.path.join(project_root, init_args['dirpath']))
        os.makedirs(init_args['dirpath'], exist_ok=True)
    
    return CallbackClass(**init_args)


def instantiate_simple_callback(
    cb_name: str,
    callback_configurations_dict: Dict[str, Any],
    trial_number: int
) -> Optional[pl.Callback]:
    """
    Instantiate simple callbacks based on boolean flags.
    
    Args:
        cb_name: Callback name
        callback_configurations_dict: All callback configurations
        trial_number: Trial number for logging
        
    Returns:
        Instantiated callback or None
    """
    if cb_name == "progress_bar":
        from lightning.pytorch.callbacks import RichProgressBar
        logging.info(f"Trial {trial_number}: Instantiated RichProgressBar from config (name: {cb_name}).")
        return RichProgressBar()
    elif cb_name == "lr_monitor":
        from lightning.pytorch.callbacks import LearningRateMonitor
        lr_mon_init_args = callback_configurations_dict.get(f"{cb_name}_config", {}).get('init_args', {})
        logging.info(f"Trial {trial_number}: Instantiated LearningRateMonitor from config (name: {cb_name}).")
        return LearningRateMonitor(**lr_mon_init_args)
    
    return None