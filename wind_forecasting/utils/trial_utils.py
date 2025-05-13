import logging
import torch
import gc
import sys
import wandb
from optuna.exceptions import TrialPruned

def handle_trial_with_oom_protection(tuning_objective, trial):
    """
    A wrapper function to handle Out-of-Memory (OOM), Pruning, and other errors during Optuna trials
    """
    result = None
    try:
        # Log GPU memory at the start of each trial
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            logging.info(f"Trial {trial.number} - Starting GPU Memory (trial_utils): {torch.cuda.memory_allocated(device)/1e9:.2f}GB / {torch.cuda.get_device_properties(device).total_memory/1e9:.2f}GB")
        
        result = tuning_objective(trial)
        
        # Log GPU memory after trial completes
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            logging.info(f"Trial {trial.number} - Ending GPU Memory (trial_utils after objective success): {torch.cuda.memory_allocated(device)/1e9:.2f}GB")
            
        return result
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logging.warning(f"Trial {trial.number} failed with CUDA OOM error (caught in trial_utils)")
            if torch.cuda.is_available():
                logging.warning(f"OOM at memory usage (trial_utils): {torch.cuda.memory_allocated()/1e9:.2f}GB")
            # Mark the wandb run as failed if it exists and is active
            if 'wandb' in sys.modules and wandb.run is not None and wandb.run.id is not None:
                logging.info(f"Marking WandB run as failed for trial {trial.number} due to OOM (trial_utils)")
                wandb.finish(exit_code=1)
            # No explicit cleanup here, finally block will handle it.
            raise # Re-raise to mark trial as FAILED in Optuna
        logging.error(f"Trial {trial.number} failed with other RuntimeError (trial_utils): {str(e)}", exc_info=True)
        if 'wandb' in sys.modules and wandb.run is not None and wandb.run.id is not None:
             wandb.finish(exit_code=1)
        raise
    except TrialPruned as e:
        logging.info(f"Trial {trial.number} was pruned (caught in trial_utils): {str(e)}")
        raise
    except Exception as e:
        if "MisconfigurationException" in str(type(e)):
            if "gpu" in str(e).lower():
                logging.error(f"Trial {trial.number} failed with GPU configuration error (trial_utils): {str(e)}")
                logging.error("This is likely due to a mismatch between requested GPUs and available GPUs.")
                logging.error("Please check the --single_gpu flag and CUDA_VISIBLE_DEVICES setting.")
            else:
                logging.error(f"Trial {trial.number} failed with configuration error (trial_utils): {str(e)}")
        else:
            logging.error(f"Trial {trial.number} failed with unexpected error (trial_utils): {type(e).__name__}: {str(e)}", exc_info=True)
        
        if 'wandb' in sys.modules and wandb.run is not None and wandb.run.id is not None:
            logging.info(f"Marking WandB run as failed for trial {trial.number} due to {type(e).__name__} (trial_utils)")
            wandb.finish(exit_code=1)
        raise
    finally:
        logging.info(f"Trial {trial.number} - Executing cleanup in trial_utils.py finally block.")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info(f"Trial {trial.number} - GPU Memory after cleanup in trial_utils.py finally: {torch.cuda.memory_allocated()/1e9:.2f}GB")