"""
Metrics extraction and processing utilities for model evaluation.
"""
import logging
import torch
import numpy as np
from typing import Dict, Any, Optional, Union
from gluonts.evaluation import MultivariateEvaluator, make_evaluation_predictions
from gluonts.model.forecast_generator import DistributionForecastGenerator, SampleForecastGenerator


def extract_metric_value(
    agg_metrics: Dict[str, Any], 
    metric_name: str, 
    trial_number: int
) -> float:
    """
    Extract and convert a metric value from aggregated metrics dictionary.
    
    Args:
        agg_metrics: Dictionary of aggregated metrics
        metric_name: Name of the metric to extract
        trial_number: Trial number for logging
        
    Returns:
        Float value of the metric
        
    Raises:
        KeyError: If metric not found
        ValueError: If metric cannot be converted to float
    """
    metric_value = agg_metrics.get(metric_name)
    
    if metric_value is None:
        error_msg = f"Metric key '{metric_name}' not found in calculated agg_metrics: {list(agg_metrics.keys())}"
        logging.error(f"Trial {trial_number} - {error_msg}")
        raise KeyError(f"Trial {trial_number} failed: {error_msg}")
    
    try:
        # Handle tensor values
        if hasattr(metric_value, 'item'):
            metric_value = metric_value.item()
        elif isinstance(metric_value, (np.ndarray, torch.Tensor)) and metric_value.size == 1:
            metric_value = metric_value.item()
        
        metric_value = float(metric_value)
        
        logging.info(f"Trial {trial_number} - Returning metric '{metric_name}' to Optuna: {metric_value}")
        return metric_value
        
    except (TypeError, ValueError) as e:
        error_msg = f"Error converting metric '{metric_name}' (value: {metric_value}) to float: {e}"
        logging.error(f"Trial {trial_number} - {error_msg}", exc_info=True)
        logging.error(f"Available metrics: {list(agg_metrics.keys())}")
        raise ValueError(f"Trial {trial_number} failed: {error_msg}") from e


def compute_evaluation_metrics(
    predictor: Any,
    val_dataset: Any,
    model_name: str,
    evaluator: MultivariateEvaluator,
    num_target_vars: int,
    trial_number: int
) -> Dict[str, Any]:
    """
    Compute evaluation metrics for a model.
    
    Args:
        predictor: Model predictor instance
        val_dataset: Validation dataset
        model_name: Name of the model (e.g., 'tactis')
        evaluator: MultivariateEvaluator instance
        num_target_vars: Number of target variables
        trial_number: Trial number for logging
        
    Returns:
        Dictionary of aggregated metrics
        
    Raises:
        RuntimeError: If evaluation fails
    """
    try:
        eval_kwargs = {
            "dataset": val_dataset,
            "predictor": predictor,
        }
        
        if model_name == 'tactis':
            logging.info(f"Trial {trial_number}: Evaluating TACTiS using SampleForecast.")
        else:
            eval_kwargs["output_distr_params"] = {
                "loc": "mean", 
                "cov_factor": "cov_factor", 
                "cov_diag": "cov_diag"
            }
            logging.info(f"Trial {trial_number}: Evaluating {model_name} using DistributionForecast "
                        f"with params: {eval_kwargs['output_distr_params']}")

        forecast_it, ts_it = make_evaluation_predictions(**eval_kwargs)
        agg_metrics, _ = evaluator(ts_it, forecast_it, num_series=num_target_vars)
        
        return agg_metrics
        
    except Exception as e:
        logging.error(f"Trial {trial_number} - Error making evaluation predictions: {str(e)}", exc_info=True)
        raise RuntimeError(f"Error making evaluation predictions in trial {trial_number}: {str(e)}") from e


def update_metrics_with_checkpoint_score(
    agg_metrics: Dict[str, Any],
    checkpoint_callback: Any,
    trial_number: int
) -> None:
    """
    Update aggregated metrics with the best score from checkpoint callback.
    
    Args:
        agg_metrics: Dictionary to update with metrics
        checkpoint_callback: Checkpoint callback containing best score
        trial_number: Trial number for logging
    """
    monitor_metric = checkpoint_callback.monitor
    best_score = checkpoint_callback.best_model_score
    
    if best_score is not None:
        if hasattr(best_score, 'item'):
            best_score = best_score.item()
        agg_metrics[monitor_metric] = best_score
        logging.info(f"Trial {trial_number} - Setting {monitor_metric} to {best_score} "
                    "from trial-specific checkpoint callback")
    else:
        logging.warning(f"Trial {trial_number} - No best_model_score available in the "
                       "trial-specific checkpoint callback")


def validate_metrics_for_return(
    agg_metrics: Optional[Dict[str, Any]],
    metric_to_return: str,
    trial_number: int
) -> float:
    """
    Validate and extract the metric to return to Optuna.
    
    Args:
        agg_metrics: Aggregated metrics dictionary (can be None)
        metric_to_return: Name of metric to return
        trial_number: Trial number for logging
        
    Returns:
        Float value of the metric
        
    Raises:
        RuntimeError: If agg_metrics is None
        KeyError: If metric not found
        ValueError: If metric cannot be converted to float
    """
    if agg_metrics is None:
        error_msg = (f"Trial {trial_number} - 'agg_metrics' is None, indicating an error "
                    "occurred before metrics could be computed.")
        logging.error(error_msg)
        raise RuntimeError(f"Trial {trial_number} failed: validation metrics were not "
                         "computed due to an earlier error.")
    
    return extract_metric_value(agg_metrics, metric_to_return, trial_number)