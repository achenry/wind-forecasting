"""
Utility to retrieve parameters from a specific Optuna trial.

This script allows extracting hyperparameters from a specific trial number
rather than just the best trial, which is useful for reproducing specific
experimental results or exploring alternative hyperparameter configurations.
"""

import logging
import re
from typing import Dict, Any, Optional


def get_specific_trial_params(storage, study_name: str, trial_number: int) -> Dict[str, Any]:
    """
    Get hyperparameters from a specific trial in an Optuna study.
    
    Args:
        storage: Optuna storage backend (SQLite or PostgreSQL)
        study_name: Name of the study (can be a prefix)
        trial_number: The trial number to retrieve (0-indexed)
        
    Returns:
        Dictionary containing the hyperparameters from the specified trial
        
    Raises:
        FileNotFoundError: If the study is not found in storage
        ValueError: If the trial number doesn't exist
    """
    logging.info(f"Getting storage for Optuna prefix study name {study_name}.")
    
    available_studies = [study.study_name for study in storage.get_all_studies()]
    logging.info(f"Found studies {available_studies}. Looking for those containing {study_name}.")
    
    if study_name in available_studies:
        full_study_name = study_name
    else:
        # Find the most recent study matching the prefix
        matching_studies = [s for s in available_studies if study_name in s]
        if not matching_studies:
            raise FileNotFoundError(f"No studies found matching prefix {study_name}. Available studies: {available_studies}")
        full_study_name = sorted(matching_studies)[-1]
    
    try:
        study_id = storage.get_study_id_from_name(full_study_name)
        logging.info(f"Found storage for Optuna full study name {full_study_name}.")
    except Exception:
        raise FileNotFoundError(f"Optuna study {full_study_name} not found. Available studies are: {available_studies}.")
    
    # Get all trials from the study
    trials = storage.get_all_trials(study_id)
    
    if trial_number >= len(trials):
        raise ValueError(f"Trial number {trial_number} not found. Study has {len(trials)} trials (0-{len(trials)-1}).")
    
    trial = trials[trial_number]
    
    logging.info(f"Retrieved trial {trial_number} from study {full_study_name}")
    logging.info(f"Trial status: {trial.state}")
    logging.info(f"Trial value: {trial.value}")
    
    return trial.params


def print_trial_params(storage, study_name: str, trial_number: int) -> None:
    """
    Print formatted parameters from a specific trial.
    
    Args:
        storage: Optuna storage backend
        study_name: Name of the study (can be a prefix)
        trial_number: The trial number to retrieve (0-indexed)
    """
    params = get_specific_trial_params(storage, study_name, trial_number)
    
    print(f"\n=== Trial {trial_number} Parameters ===")
    print(f"Total parameters: {len(params)}")
    print("\nParameters:")
    
    # Group parameters by prefix for better readability
    grouped_params = {}
    for key, value in sorted(params.items()):
        prefix = key.split('_')[0] if '_' in key else 'other'
        if prefix not in grouped_params:
            grouped_params[prefix] = []
        grouped_params[prefix].append((key, value))
    
    for prefix, param_list in sorted(grouped_params.items()):
        print(f"\n  {prefix.upper()} Parameters:")
        for key, value in param_list:
            # Format float values to scientific notation if very small
            if isinstance(value, float) and abs(value) < 1e-3:
                print(f"    {key}: {value:.2e}")
            else:
                print(f"    {key}: {value}")


if __name__ == "__main__":
    # Example usage for testing
    import argparse
    import optuna
    
    parser = argparse.ArgumentParser(description="Get parameters from a specific Optuna trial")
    parser.add_argument("--storage_url", required=True, help="Optuna storage URL")
    parser.add_argument("--study_name", required=True, help="Study name or prefix")
    parser.add_argument("--trial_number", type=int, required=True, help="Trial number (0-indexed)")
    
    args = parser.parse_args()
    
    storage = optuna.storages.RDBStorage(url=args.storage_url)
    print_trial_params(storage, args.study_name, args.trial_number)