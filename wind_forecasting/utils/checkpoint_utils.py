"""
Checkpoint handling utilities for model loading and validation.
"""
import os
import re
import logging
import torch
import inspect
from typing import Dict, Optional, Tuple, Any, List


def load_checkpoint(checkpoint_path: str, trial_number: int) -> Dict[str, Any]:
    """
    Load a checkpoint from disk with validation.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        trial_number: Trial number for logging
        
    Returns:
        Loaded checkpoint dictionary
        
    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        RuntimeError: If checkpoint loading fails
    """
    if not os.path.exists(checkpoint_path) or not checkpoint_path:
        error_msg = f"Checkpoint path does not exist or is empty: '{checkpoint_path}'. Cannot load model for metric retrieval."
        logging.error(f"Trial {trial_number} - {error_msg}")
        raise FileNotFoundError(error_msg)

    logging.info(f"Trial {trial_number} - Loading checkpoint from: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        return checkpoint
    except Exception as e:
        error_msg = f"Failed to load checkpoint from {checkpoint_path}: {str(e)}"
        logging.error(f"Trial {trial_number} - {error_msg}")
        raise RuntimeError(error_msg) from e


def parse_epoch_from_checkpoint_path(checkpoint_path: str, trial_number: int) -> int:
    """
    Extract epoch number from checkpoint filename.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        trial_number: Trial number for logging
        
    Returns:
        Epoch number
        
    Raises:
        ValueError: If epoch number cannot be parsed
    """
    try:
        match = re.search(r"epoch=(\d+)", checkpoint_path)
        if match:
            epoch_number = int(match.group(1))
            logging.info(f"Trial {trial_number} - Extracted epoch {epoch_number} from checkpoint path: {checkpoint_path}")
            return epoch_number
        else:
            error_msg = f"Failed to parse epoch number from best checkpoint path: {checkpoint_path}"
            logging.error(f"Trial {trial_number} - {error_msg}")
            raise ValueError(f"Trial {trial_number}: {error_msg}")
    except (ValueError, TypeError) as parse_error:
        error_msg = f"Error parsing epoch from checkpoint path '{checkpoint_path}': {parse_error}"
        logging.error(f"Trial {trial_number} - {error_msg}")
        raise ValueError(f"Trial {trial_number}: {error_msg}") from parse_error


def determine_tactis_stage(
    epoch_number: int, 
    stage2_start_epoch: Optional[int], 
    trial_number: int
) -> int:
    """
    Determine the correct stage for TACTiS model based on epoch number.
    
    Args:
        epoch_number: Current epoch number
        stage2_start_epoch: Epoch at which stage 2 starts
        trial_number: Trial number for logging
        
    Returns:
        Stage number (1 or 2)
        
    Raises:
        KeyError: If stage2_start_epoch is None
        RuntimeError: If stage determination fails
    """
    if stage2_start_epoch is None:
        error_msg = "'stage2_start_epoch' parameter missing in trial configuration for TACTiS model. Cannot determine stage."
        logging.error(f"Trial {trial_number} - {error_msg}")
        raise KeyError(f"Trial {trial_number}: {error_msg}")

    try:
        stage2_start = int(stage2_start_epoch)
        logging.info(f"Trial {trial_number} - Retrieved stage2_start_epoch: {stage2_start}")

        if epoch_number >= stage2_start:
            correct_stage = 2
        else:
            correct_stage = 1
            
        logging.info(f"Trial {trial_number} - Determined correct stage for TACTiS re-instantiation: "
                    f"{correct_stage} (Epoch: {epoch_number}, Stage 2 Start: {stage2_start})")
        
        return correct_stage
        
    except (ValueError, TypeError) as stage_error:
        error_msg = f"Error processing stage information for TACTiS (epoch={epoch_number}, stage2_start_epoch={stage2_start_epoch}): {stage_error}"
        logging.error(f"Trial {trial_number} - {error_msg}")
        raise RuntimeError(f"Trial {trial_number}: {error_msg}") from stage_error


def extract_hyperparameters(
    checkpoint: Dict[str, Any], 
    checkpoint_path: str, 
    trial_number: int
) -> Dict[str, Any]:
    """
    Extract hyperparameters from checkpoint.
    
    Args:
        checkpoint: Loaded checkpoint dictionary
        checkpoint_path: Path to checkpoint (for error messages)
        trial_number: Trial number for logging
        
    Returns:
        Dictionary containing hyperparameters and model_config
        
    Raises:
        KeyError: If required hyperparameters are missing
    """
    hparams = checkpoint.get('hyper_parameters', checkpoint.get('hparams'))
    if hparams is None:
        error_msg = f"Hyperparameters not found in checkpoint: {checkpoint_path}. Cannot re-instantiate model."
        logging.error(f"Trial {trial_number} - {error_msg}")
        raise KeyError(f"Trial {trial_number}: {error_msg}")

    logging.debug(f"Loaded hparams from checkpoint: {hparams}")

    model_config = hparams.get('model_config')
    if model_config is None:
        error_msg = f"Critical: 'model_config' dictionary not found within loaded hyperparameters in {checkpoint_path}. Check saving logic."
        logging.error(f"Trial {trial_number} - {error_msg}")
        raise KeyError(f"Trial {trial_number}: {error_msg}")

    return hparams


def prepare_model_init_args(
    hparams: Dict[str, Any],
    lightning_module_class: type,
    config: Dict[str, Any],
    model_name: str,
    trial_number: int
) -> Dict[str, Any]:
    """
    Prepare initialization arguments for model re-instantiation.
    
    Args:
        hparams: Hyperparameters from checkpoint
        lightning_module_class: Lightning module class to instantiate
        config: Configuration dictionary
        model_name: Name of the model
        trial_number: Trial number for logging
        
    Returns:
        Dictionary of initialization arguments
        
    Raises:
        KeyError: If required parameters are missing
    """
    module_sig = inspect.signature(lightning_module_class.__init__)
    module_params = [param.name for param in module_sig.parameters.values()]
    module_params.remove("self")
    
    model_config = hparams.get('model_config')
    
    init_args = {
        'model_config': model_config,
        **{k: hparams.get(k, config["model"][model_name].get(k))
           for k in module_params if k not in ['model_config']}
    }
    
    # Log warnings for missing parameters
    for key, val in init_args.items():
        if (key not in ['model_config', 'initial_stage']) and (key not in hparams) and (key in module_params):
            logging.warning(f"Hyperparameter '{key}' not found in checkpoint, using default value from config: {val}")

    # Check for missing required arguments
    missing_args = [k for k, v in init_args.items()
                   if v is None and k not in ['model_config', 'initial_stage']]

    if missing_args:
        error_msg = f"Missing required hyperparameters in checkpoint even after checking defaults: {missing_args}"
        logging.error(f"Trial {trial_number} - {error_msg}")
        raise KeyError(f"Trial {trial_number}: {error_msg}")
    
    return init_args


def load_model_state(
    model: torch.nn.Module, 
    checkpoint: Dict[str, Any], 
    trial_number: int
) -> None:
    """
    Load state dict into model.
    
    Args:
        model: Model instance to load state into
        checkpoint: Checkpoint containing state_dict
        trial_number: Trial number for logging
        
    Raises:
        RuntimeError: If state dict loading fails
    """
    try:
        logging.info(f"Loading state_dict into re-instantiated model...")
        model.load_state_dict(checkpoint['state_dict'])
        logging.info("State_dict loaded successfully.")
    except Exception as e:
        logging.error(f"Trial {trial_number} - Error loading state_dict: {str(e)}", exc_info=True)
        raise RuntimeError(f"Error loading state_dict in trial {trial_number}: {str(e)}") from e


def set_tactis_stage(
    model: Any, 
    correct_stage: int, 
    trial_number: int
) -> None:
    """
    Set the stage for TACTiS model after instantiation.
    
    Args:
        model: TACTiS model instance
        correct_stage: Stage to set (1 or 2)
        trial_number: Trial number for logging
    """
    try:
        logging.info(f"Trial {trial_number} - Setting TACTiS model stage to {correct_stage} before loading checkpoint.")
        model.stage = correct_stage
        
        if hasattr(model.model, 'tactis') and hasattr(model.model.tactis, 'set_stage'):
            model.model.tactis.set_stage(correct_stage)
            logging.info(f"Trial {trial_number} - Successfully called model.model.tactis.set_stage({correct_stage}).")
        else:
            logging.warning(f"Trial {trial_number} - Could not find model.model.tactis.set_stage() method. "
                          "Stage setting might not be fully applied.")
    except Exception as stage_set_error:
        logging.error(f"Trial {trial_number} - Error setting TACTiS stage post-instantiation: {stage_set_error}", 
                     exc_info=True)
        logging.warning(f"Trial {trial_number} - Proceeding with state_dict loading despite stage setting error.")