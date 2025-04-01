import logging
import torch
import gc

def handle_trial_with_oom_protection(tuning_objective, trial):
    """
    A wrapper function to handle Out-of-Memory (OOM) and other errors during Optuna trials.
    """
    try:
        # Log GPU memory at the start of each trial
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            logging.info(f"Trial {trial.number} - Starting GPU Memory: {torch.cuda.memory_allocated(device)/1e9:.2f}GB / {torch.cuda.get_device_properties(device).total_memory/1e9:.2f}GB")
        
        result = tuning_objective(trial)
        
        # Log GPU memory after trial completes
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            logging.info(f"Trial {trial.number} - Ending GPU Memory: {torch.cuda.memory_allocated(device)/1e9:.2f}GB / {torch.cuda.get_device_properties(device).total_memory/1e9:.2f}GB")
            
        # Add explicit garbage collection after each trial
        gc.collect()
        return result
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logging.warning(f"Trial {trial.number} failed with CUDA OOM error")
            if torch.cuda.is_available():
                logging.warning(f"OOM at memory usage: {torch.cuda.memory_allocated()/1e9:.2f}GB")
            # Force garbage collection
            torch.cuda.empty_cache()
            gc.collect()
            # Return a very poor score
            return float('inf')  # Assuming minimization; for maximization tasks this should be float('-inf')
        raise e
    except Exception as e:
        # Catch GPU configuration errors and other exceptions
        if "MisconfigurationException" in str(type(e)) and "gpu" in str(e):
            logging.error(f"Trial {trial.number} failed with GPU configuration error: {str(e)}")
            logging.error("This is likely due to a mismatch between requested GPUs and available GPUs.")
            logging.error("Please check the --single_gpu flag and CUDA_VISIBLE_DEVICES setting.")
            
            # Return a poor score to allow the study to continue
            return float('inf')  # Assuming minimization
        elif "MisconfigurationException" in str(type(e)):
            logging.error(f"Trial {trial.number} failed with configuration error: {str(e)}")
            return float('inf')  # Assuming minimization
        
        # For any other unexpected errors, log details and re-raise
        logging.error(f"Trial {trial.number} failed with unexpected error: {type(e).__name__}: {str(e)}")
        raise