import logging
import wandb
import optuna
from optuna.trial import TrialState

def log_detailed_trials_table_to_wandb(study: optuna.Study, wandb_run):
    """
    Generates a detailed WandB table summarizing all Optuna trials and logs it.

    Args:
        study: The Optuna study object.
        wandb_run: The active wandb.sdk.wandb_run.Run object to log to.
    """
    logging.info("Creating and logging detailed Optuna trials table to W&B...")
    # Get all trials, including potentially running/waiting ones for a complete picture
    all_trials = study.get_trials(deepcopy=False, states=None) # Get all states

    if not all_trials:
        logging.warning("No trials found in the study to log to the detailed summary table.")
        return

    best_trial = None
    try:
        # best_trial only considers COMPLETED trials
        best_trial = study.best_trial
    except ValueError:
        logging.warning("Could not determine best trial (likely none completed successfully yet).")
    except Exception as e:
        logging.error(f"Unexpected error getting best trial: {e}", exc_info=True)


    # Collect all unique hyperparameter keys across all trials
    all_param_keys = set()
    for trial in all_trials:
        all_param_keys.update(trial.params.keys())
    sorted_param_keys = sorted(list(all_param_keys))

    # Define table columns
    columns = ["Trial Number", "State", "Value", "Is Best", "Is Pruned", "Is Failed"] + sorted_param_keys
    detailed_trial_table = wandb.Table(columns=columns)

    # Populate table rows
    for trial in all_trials:
        # Determine 'Is Best' based on the best *completed* trial
        is_best = best_trial is not None and trial.number == best_trial.number and trial.state == TrialState.COMPLETE
        is_pruned = trial.state == TrialState.PRUNED
        is_failed = trial.state == TrialState.FAIL

        row_data = [
            trial.number,
            trial.state.name,
            trial.value, # Will be None if not COMPLETE
            is_best,
            is_pruned,
            is_failed
        ]
        # Add parameter values, using None if a trial didn't have a specific param
        for key in sorted_param_keys:
            row_data.append(trial.params.get(key, None))

        # Use * to unpack the list into arguments for add_data
        detailed_trial_table.add_data(*row_data)

    # Log the table to the provided W&B run
    wandb_run.log({"optuna_trials_detailed_summary": detailed_trial_table})
    logging.info(f"Logged detailed Optuna trials table ({len(all_trials)} trials) to W&B run {wandb_run.id}.")