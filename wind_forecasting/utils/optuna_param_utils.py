"""
Optuna parameter retrieval utilities.

These utilities are used across multiple modes (train/test) for:
- Loading best hyperparameters from completed Optuna studies
- Finding and accessing study results from storage
"""

import logging
import re
from typing import Dict, Any


def get_tuned_params(storage, study_name: str) -> Dict[str, Any]:
    """
    Get the best hyperparameters from a completed Optuna study.
    
    Args:
        storage: Optuna storage backend (SQLite or PostgreSQL)
        study_name: Name of the study (can be a prefix)
        
    Returns:
        Dictionary containing the best hyperparameters from the study
        
    Raises:
        FileNotFoundError: If the study is not found in storage
    """
    logging.info(f"Getting storage for Optuna prefix study name {study_name}.")
    
    available_studies = [study.study_name for study in storage.get_all_studies()]
    logging.info(f"Found studies {available_studies}. Looking for those containing {study_name}.")
    if study_name in available_studies:
        full_study_name = study_name
    else:
        full_study_name = sorted(storage.get_all_studies(), key=lambda study: int(re.search(f"(?<={study_name}_)(\\d+)", study.study_name).group()))[-1].study_name
    
    try:
        study_id = storage.get_study_id_from_name(full_study_name)
        logging.info(f"Found storage for Optuna full study name {full_study_name}.")
    except Exception:
        raise FileNotFoundError(f"Optuna study {full_study_name} not found. Please run tune_hyperparameters_multi for all outputs first. Available studies are: {available_studies}.")
    # self.model[output].set_params(**storage.get_best_trial(study_id).params)
    # storage.get_all_studies()[0]._study_id
    # estimato = self.create_model(**storage.get_best_trial(study_id).params)
    return storage.get_best_trial(study_id).params