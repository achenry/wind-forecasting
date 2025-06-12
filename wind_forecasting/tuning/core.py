"""
Core tuning functionality for wind forecasting models.

This module contains the main tune_model function that orchestrates
the entire hyperparameter optimization process using Optuna.
"""

import os
import logging
import time
import subprocess
from datetime import datetime
import pandas as pd

from optuna import create_study, load_study
from optuna.study import MaxTrialsCallback
from optuna.pruners import HyperbandPruner, PercentilePruner, PatientPruner, SuccessiveHalvingPruner, NopPruner
from optuna.trial import TrialState

from wind_forecasting.tuning.objective import MLTuningObjective
from wind_forecasting.tuning.utils.optuna_utils import (
    OptunaSamplerPrunerPersistence
)
from wind_forecasting.utils.optuna_config_utils import generate_db_setup_params, generate_optuna_dashboard_command


def tune_model(model, config, study_name, optuna_storage, lightning_module_class, estimator_class,
               distr_output_class, data_module,
               trial_protection_callback=None, seed=42, tuning_phase=0, restart_tuning=False, optimize_callbacks=None,):

    # Log safely without credentials if they were included (they aren't for socket trust)
    if hasattr(optuna_storage, "url"):
        log_storage_url = optuna_storage.url.split('@')[0] + '@...' if '@' in optuna_storage.url else optuna_storage.url
        logging.info(f"Using Optuna storage URL: {log_storage_url}")

    # NOTE: Restarting the study is now handled in the Slurm script by deleting the PGDATA directory

    # Configure pruner based on settings
    pruner = None
    if "pruning" in config["optuna"] and config["optuna"]["pruning"].get("enabled", False):
        pruning_type = config["optuna"]["pruning"].get("type", "hyperband").lower()
        logging.info(f"Configuring pruner: type={pruning_type}")

        if pruning_type == "patient":
            patience = config["optuna"]["pruning"].get("patience", 0)
            min_delta = config["optuna"]["pruning"].get("min_delta", 0.0)

            # Configure wrapped pruner if specified
            wrapped_config = config["optuna"]["pruning"].get("wrapped_pruner")
            wrapped_pruner_instance = None

            if wrapped_config and isinstance(wrapped_config, dict):
                wrapped_type = wrapped_config.get("type", "").lower()
                logging.info(f"Configuring wrapped pruner of type: {wrapped_type}")

                if wrapped_type == "percentile":
                    percentile = wrapped_config.get("percentile", 50.0)
                    n_startup_trials = wrapped_config.get("n_startup_trials", 4)
                    n_warmup_steps = wrapped_config.get("n_warmup_steps", 12)
                    interval_steps = wrapped_config.get("interval_steps", 1)
                    n_min_trials = wrapped_config.get("n_min_trials", 1)

                    wrapped_pruner_instance = PercentilePruner(
                        percentile=percentile,
                        n_startup_trials=n_startup_trials,
                        n_warmup_steps=n_warmup_steps,
                        interval_steps=interval_steps,
                        n_min_trials=n_min_trials
                    )
                    logging.info(f"Created wrapped PercentilePruner with percentile={percentile}, n_startup_trials={n_startup_trials}, n_warmup_steps={n_warmup_steps}")
                    
                elif wrapped_type == "successivehalving":
                    min_resource = wrapped_config.get("min_resource", 2)
                    reduction_factor = wrapped_config.get("reduction_factor", 2)
                    min_early_stopping_rate = wrapped_config.get("min_early_stopping_rate", 0)
                    bootstrap_count = wrapped_config.get("bootstrap_count", 0)

                    wrapped_pruner_instance = SuccessiveHalvingPruner(
                        min_resource=min_resource,
                        reduction_factor=reduction_factor,
                        min_early_stopping_rate=min_early_stopping_rate,
                        bootstrap_count=bootstrap_count
                    )
                    logging.info(f"Created wrapped SuccessiveHalvingPruner with min_resource={min_resource}, reduction_factor={reduction_factor}, min_early_stopping_rate={min_early_stopping_rate}, bootstrap_count={bootstrap_count}, bootstrap_count={bootstrap_count}")
                
                else:
                    logging.warning(f"Unknown wrapped pruner type: {wrapped_type}. Defaulting to NopPruner.")
                    wrapped_pruner_instance = NopPruner()
            else:
                logging.warning("No wrapped pruner configuration found. Defaulting to NopPruner.")
                wrapped_pruner_instance = NopPruner()
            
            # If no valid wrapped pruner is configured, use NopPruner
            if wrapped_pruner_instance is None:
                logging.warning("No valid wrapped pruner configuration found. PatientPruner will wrap NopPruner.")
                wrapped_pruner_instance = NopPruner()

            # Create PatientPruner wrapping the configured pruner
            pruner = PatientPruner(
                wrapped_pruner=wrapped_pruner_instance,
                patience=patience,
                min_delta=min_delta
            )
            logging.info(f"Created PatientPruner with patience={patience}, min_delta={min_delta} wrapping {type(wrapped_pruner_instance).__name__}")

        elif pruning_type == "hyperband":
            min_resource = config["optuna"]["pruning"].get("min_resource", 2)
            max_resource = config["optuna"]["pruning"].get("max_resource", max_epochs)
            reduction_factor = config["optuna"]["pruning"].get("reduction_factor", 2)
            bootstrap_count = config["optuna"]["pruning"].get("bootstrap_count", 0)
            
            pruner = HyperbandPruner(
                min_resource=min_resource,
                max_resource=max_resource,
                reduction_factor=reduction_factor,
                bootstrap_count=bootstrap_count
            )
            logging.info(f"Created HyperbandPruner with min_resource={min_resource}, max_resource={max_resource}, reduction_factor={reduction_factor}, bootstrap_count={bootstrap_count}")

        elif pruning_type == "successivehalving":
            min_resource = config["optuna"]["pruning"].get("min_resource", 2)
            reduction_factor = config["optuna"]["pruning"].get("reduction_factor", 2)
            min_early_stopping_rate = config["optuna"]["pruning"].get("min_early_stopping_rate", 0)
            bootstrap_count = config["optuna"]["pruning"].get("bootstrap_count", 0)

            pruner = SuccessiveHalvingPruner(
                min_resource=min_resource,
                reduction_factor=reduction_factor,
                min_early_stopping_rate=min_early_stopping_rate,
                bootstrap_count=bootstrap_count
            )
            logging.info(f"Created SuccessiveHalvingPruner with min_resource={min_resource}, reduction_factor={reduction_factor}, min_early_stopping_rate={min_early_stopping_rate}, bootstrap_count={bootstrap_count}, bootstrap_count={bootstrap_count}")

        elif pruning_type == "percentile":
            percentile = config["optuna"]["pruning"].get("percentile", 25)
            n_startup_trials = config["optuna"]["pruning"].get("n_startup_trials", 5)
            n_warmup_steps = config["optuna"]["pruning"].get("n_warmup_steps", 2)
            interval_steps = config["optuna"]["pruning"].get("interval_steps", 1)
            n_min_trials = config["optuna"]["pruning"].get("n_min_trials", 1)

            pruner = PercentilePruner(
                percentile=percentile,
                n_startup_trials=n_startup_trials,
                n_warmup_steps=n_warmup_steps,
                interval_steps=interval_steps,
                n_min_trials=n_min_trials
            )
            logging.info(f"Created PercentilePruner with percentile={percentile}, n_startup_trials={n_startup_trials}, n_warmup_steps={n_warmup_steps}")

        else:
            logging.warning(f"Unknown pruner type: {pruning_type}, using no pruning")
            pruner = NopPruner()
    else:
        logging.info("Pruning is disabled, using NopPruner")
        pruner = NopPruner()

    # Get worker ID for study creation/loading logic
    # Use WORKER_RANK consistent with run_model.py. Default to '0' if not set.
    worker_id = os.environ.get('WORKER_RANK', '0')

    # Generate unique study name based on restart_tuning flag

    base_study_prefix = study_name
    if restart_tuning:
        job_id = os.environ.get('SLURM_JOB_ID')
        if job_id:
            # If running in SLURM, use the job ID
            final_study_name = f"{base_study_prefix}_{job_id}"
        else:
            # Otherwise use a timestamp
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            final_study_name = f"{base_study_prefix}_{timestamp}"
        logging.info(f"Creating a new study with unique name: {final_study_name}")
    else:
        # If not restarting, use the base name to resume existing study
        final_study_name = base_study_prefix
        logging.info(f"Using existing study name to resume: {final_study_name}")

    # Define pickle directory for sampler/pruner persistence
    pickle_dir = os.path.join(config.get("logging", {}).get("optuna_dir", "logging/optuna"), "pickles")
    
    # Instantiate the persistence utility
    sampler_pruner_persistence = OptunaSamplerPrunerPersistence(config, seed)

    # Get sampler and pruner objects using pickling logic
    try:
        sampler, pruner_for_study = sampler_pruner_persistence.get_sampler_pruner_objects(
            worker_id, pruner, restart_tuning, final_study_name, optuna_storage, pickle_dir
        )
    except Exception as e:
        logging.error(f"Worker {worker_id}: Error getting sampler/pruner objects: {str(e)}", exc_info=True)
        raise

    # Create study on rank 0, load on other ranks
    study = None # Initialize study variable
    try:
        if worker_id == '0':
            logging.info(f"Rank 0: Creating/loading Optuna study '{final_study_name}' with pruner: {type(pruner_for_study).__name__}")
            study = create_study(
                study_name=final_study_name,
                storage=optuna_storage,
                direction=config["optuna"].get("direction", "minimize"),
                load_if_exists=not restart_tuning, # Only load if not restarting
                sampler=sampler,
                pruner=pruner_for_study
            )
            logging.info(f"Rank 0: Study '{final_study_name}' created or loaded successfully.")
        else:
            # Non-rank-0 workers MUST load the study created by rank 0
            logging.info(f"Rank {worker_id}: Attempting to load existing Optuna study '{final_study_name}'")
            # Add a small delay and retry mechanism for loading, in case rank 0 is slightly delayed
            max_retries = 6 # Increased retries slightly
            retry_delay = 10 # Increased delay slightly
            for attempt in range(max_retries):
                try:
                    study = load_study(
                        study_name=final_study_name,
                        storage=optuna_storage,
                        sampler=sampler,
                        pruner=pruner_for_study
                    )
                    logging.info(f"Rank {worker_id}: Study '{final_study_name}' loaded successfully on attempt {attempt+1}.")
                    break # Exit loop on success
                except KeyError as e: # Optuna <3.0 raises KeyError if study doesn't exist yet
                     if attempt < max_retries - 1:
                          logging.warning(f"Rank {worker_id}: Study '{final_study_name}' not found yet (attempt {attempt+1}/{max_retries}). Retrying in {retry_delay}s... Error: {e}")
                          time.sleep(retry_delay)
                     else:
                          logging.error(f"Rank {worker_id}: Failed to load study '{final_study_name}' after {max_retries} attempts (KeyError). Aborting.")
                          raise
                except Exception as e: # Catch other potential loading errors (e.g., DB connection issues)
                     logging.error(f"Rank {worker_id}: An unexpected error occurred while loading study '{final_study_name}' on attempt {attempt+1}: {e}", exc_info=True)
                     # Decide whether to retry on other errors or raise immediately
                     if attempt < max_retries - 1:
                          logging.warning(f"Retrying in {retry_delay}s...")
                          time.sleep(retry_delay)
                     else:
                          logging.error(f"Rank {worker_id}: Failed to load study '{final_study_name}' after {max_retries} attempts due to persistent errors. Aborting.")
                          raise # Re-raise other errors after retries

            # Check if study was successfully loaded after the loop
            if study is None:
                 # This condition should ideally be caught by the error handling within the loop, but added for safety.
                 raise RuntimeError(f"Rank {worker_id}: Could not load study '{final_study_name}' after multiple retries.")

    except Exception as e:
        # Log error with rank information
        logging.error(f"Rank {worker_id}: Error creating/loading study '{final_study_name}': {str(e)}", exc_info=True)
        # Log storage URL safely
        if hasattr(optuna_storage, "url"):
            log_storage_url_safe = str(optuna_storage.url).split('@')[0] + '@...' if '@' in str(optuna_storage.url) else str(optuna_storage.url)
            logging.error(f"Error details - Type: {type(e).__name__}, Storage: {log_storage_url_safe}")
        else:
            logging.error(f"Error details - Type: {type(e).__name__}, Storage: Journal")
        raise

    # Define study_config_params for all workers
    max_epochs = config["optuna"].get("max_epochs")
    metric = config["optuna"].get("metric", "val_loss")
    study_config_params = {
        "dataset_per_turbine_target": config["dataset"].get("per_turbine_target"),
        "optuna_sampler": config["optuna"].get("sampler"),
        "optuna_pruner_type": config["optuna"]["pruning"].get("type") if "pruning" in config["optuna"] else None,
        "optuna_max_epochs": max_epochs,
        "optuna_base_limit_train_batches": config["optuna"].get("base_limit_train_batches"),
        "optuna_limit_train_batches": config["optuna"].get("limit_train_batches"),  # Legacy fallback
        "dataset_base_batch_size": config["dataset"].get("base_batch_size"),
        "optuna_metric": metric,
        "dataset_data_path": config["dataset"].get("data_path"),
        "dataset_resample_freq": config["dataset"].get("resample_freq"),
        "dataset_test_split": config["dataset"].get("test_split"),
        "dataset_val_split": config["dataset"].get("val_split")
    }
    
    # Use base_limit_train_batches if available, otherwise fallback to limit_train_batches
    limit_train_batches = config["optuna"].get("base_limit_train_batches") or config["optuna"].get("limit_train_batches")
        
    # Set study user attributes from config (only on rank 0)
    if worker_id == '0':
        for key, value in study_config_params.items():
            if value is not None:
                study.set_user_attr(key, value)

        logging.info(f"Set study user attributes: {list(study_config_params.keys())}")

        # --- Launch Dashboard (Rank 0 only) ---
        # if hasattr(optuna_storage, "url"):
        #     launch_optuna_dashboard(config, optuna_storage.url) # Call imported function
        # --------------------------------------

    # Worker ID already fetched above for study creation/loading
    dynamic_params = None
    
    if tuning_phase == 0 and worker_id == "0":
        resample_freq_choices = config["optuna"].get("resample_freq_choices", [int(data_module.freq[:-1])])
        fixed_per_turbine = config.get("dataset", {}).get("per_turbine_target", False)
        logging.info(f"Rank 0: DataModule 'per_turbine_target' fixed to: {fixed_per_turbine} for pre-computation.")

        original_dm_freq = data_module.freq
        original_dm_per_turbine = data_module.per_turbine_target
        original_dm_pred_len = data_module.prediction_length
        original_dm_ctx_len = data_module.context_length

        logging.info("Rank 0: Starting pre-computation of base resampled Parquet files.")
        for resample_freq_seconds in resample_freq_choices:
            current_freq_str = f"{resample_freq_seconds}s"
            logging.info(f"Rank 0: Checking/generating base resampled Parquet for freq={current_freq_str}, per_turbine={fixed_per_turbine}.")
            
            data_module.freq = current_freq_str
            data_module.per_turbine_target = fixed_per_turbine
            
            original_prediction_len_seconds_config = config["dataset"]["prediction_length"]
            data_module.prediction_length = int(pd.Timedelta(original_prediction_len_seconds_config, unit="s") / pd.Timedelta(data_module.freq))
            
            data_module.set_train_ready_path()

            if not os.path.exists(data_module.train_ready_data_path):
                logging.info(f"Rank 0: Base resampled Parquet {data_module.train_ready_data_path} not found. Calling DataModule.generate_datasets().")
                data_module.generate_datasets()
            else:
                logging.info(f"Rank 0: Base resampled Parquet {data_module.train_ready_data_path} already exists.")

        data_module.freq = original_dm_freq
        data_module.per_turbine_target = original_dm_per_turbine
        data_module.prediction_length = original_dm_pred_len
        data_module.context_length = original_dm_ctx_len
        data_module.set_train_ready_path()

        logging.info("Rank 0: Finished pre-computation of base resampled Parquet files.")
        dynamic_params = {"resample_freq": resample_freq_choices}

    logging.info(f"Worker {worker_id}: Participating in Optuna study {final_study_name}")

    # get from config
    resample_freq_choices = config.get("optuna", {}).get("resample_freq_choices", None)
    if resample_freq_choices is None:
        logging.warning("'optuna.resample_freq_choices' not found in config. Default to 60s.")
        resample_freq_choices = [60]

    tuning_objective = MLTuningObjective(model=model, config=config,
                                        lightning_module_class=lightning_module_class,
                                        estimator_class=estimator_class,
                                        distr_output_class=distr_output_class,
                                        max_epochs=max_epochs,
                                        limit_train_batches=limit_train_batches,
                                        data_module=data_module,
                                        metric=metric,
                                        seed=seed,
                                        tuning_phase=tuning_phase,
                                        dynamic_params=dynamic_params,
                                        study_config_params=study_config_params)

    # Use the trial protection callback if provided
    objective_fn = (lambda trial: trial_protection_callback(tuning_objective, trial)) if trial_protection_callback else tuning_objective

    # WandB integration deprecated
    if optimize_callbacks is None:
        optimize_callbacks = []
    elif not isinstance(optimize_callbacks, list):
        optimize_callbacks = [optimize_callbacks]

    # Add sampler/pruner state checkpointing callback for crash recovery
    # Save after every trial completion to preserve TPESampler algorithmic state
    sampler_checkpoint_callback = sampler_pruner_persistence.create_trial_completion_callback(
        worker_id=worker_id,
        save_frequency=1  # Save after every trial - overhead is minimal compared to trial duration
    )
    optimize_callbacks.append(sampler_checkpoint_callback)
    logging.info(f"Worker {worker_id}: Added sampler state checkpointing after every trial completion")

    try:
        n_trials_per_worker = config["optuna"].get("n_trials_per_worker", 10)
        total_study_trials_config = config["optuna"].get("total_study_trials", 100)
        
        n_trials_setting_for_optimize = None
        
        # Determine number of trials to run
        if isinstance(total_study_trials_config, int) and total_study_trials_config > 0:
            total_study_trials = total_study_trials_config
            study.set_user_attr("total_study_trials", total_study_trials)
            logging.info(f"Set global trial limit to {total_study_trials} trials.")
            n_trials_setting_for_optimize = None
            
            max_trials_cb = MaxTrialsCallback(
                n_trials=total_study_trials,
                states=(TrialState.COMPLETE, TrialState.PRUNED) # INFO: Do not count failed trials
            )
            optimize_callbacks.append(max_trials_cb)
            logging.info(f"MaxTrialsCallback added for {total_study_trials} trials.")
        else:
            # Fall back to per-worker limit if no global limit is set
            n_trials_setting_for_optimize = n_trials_per_worker
            logging.info(f"No valid global trial limit found (value: {total_study_trials_config}). Using per-worker limit of {n_trials_per_worker}.")
            n_trials_setting_for_optimize = n_trials_per_worker
        
        # Let Optuna handle trial distribution - each worker will ask the storage for a trial
        # Show progress bar only on rank 0 to avoid cluttered logs
        study.optimize(
            objective_fn,
            n_trials=n_trials_setting_for_optimize,
            callbacks=optimize_callbacks,
            show_progress_bar=(worker_id=='0')
        )
    except KeyError as e:
        logging.error(f"Configuration key missing: {e}")
    except Exception as e:
        logging.error(f"Worker {worker_id}: Failed during study optimization: {str(e)}", exc_info=True)
        raise

    if worker_id == '0' and study:
        logging.info("Rank 0: Starting W&B summary run creation.")

        # Wait for all expected trials to complete
        num_workers = int(os.environ.get('WORLD_SIZE', 1))
        
        if total_study_trials:
            expected_total_trials = total_study_trials
            logging.info(f"Rank 0: Expecting a maximum of {expected_total_trials} trials (global limit).")
        else:
            expected_total_trials = num_workers * n_trials_per_worker
            logging.info(f"Rank 0: Expecting a total of {expected_total_trials} trials ({num_workers} workers * {n_trials_per_worker} trials/worker).")

        logging.info("Rank 0: Waiting for all expected Optuna trials to reach a terminal state...")
        wait_interval_seconds = 30
        while True:
            # Refresh trials from storage
            all_trials_current = study.get_trials(deepcopy=False)
            finished_trials = [t for t in all_trials_current if t.state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)]
            num_finished = len(finished_trials)
            num_total_in_db = len(all_trials_current) # Current count in DB

            logging.info(f"Rank 0: Trial status check: {num_finished} finished / {num_total_in_db} in DB (expected total: {expected_total_trials}).")

            if num_finished >= expected_total_trials:
                logging.info(f"Rank 0: All {expected_total_trials} expected trials have reached a terminal state.")
                break
            elif num_total_in_db > expected_total_trials and num_finished >= expected_total_trials:
                 logging.warning(f"Rank 0: Found {num_total_in_db} trials in DB (expected {expected_total_trials}), but {num_finished} finished trials meet the expectation.")
                 break

            logging.info(f"Rank 0: Still waiting for trials to finish ({num_finished}/{expected_total_trials}). Sleeping for {wait_interval_seconds} seconds...")
            time.sleep(wait_interval_seconds)

        # Fetch best trial *before* initializing summary run
        best_trial = None
        try:
            best_trial = study.best_trial
            logging.info(f"Rank 0: Fetched best trial: Number={best_trial.number}, Value={best_trial.value}")
        except ValueError:
            logging.warning("Rank 0: Could not retrieve best trial (likely no trials completed successfully).")
        except Exception as e_best_trial:
            logging.error(f"Rank 0: Error fetching best trial: {e_best_trial}", exc_info=True)

        # Fetch Git info directly using subprocess
        remote_url = None
        commit_hash = None
        try:
            # Get remote URL
            remote_url_bytes = subprocess.check_output(['git', 'config', '--get', 'remote.origin.url'], stderr=subprocess.STDOUT).strip()
            remote_url = remote_url_bytes.decode('utf-8')
            # Convert SSH URL to HTTPS URL if necessary
            if remote_url.startswith("git@"):
                remote_url = remote_url.replace(":", "/").replace("git@", "https://")
            # Remove .git suffix AFTER potential conversion
            if remote_url.endswith(".git"):
                remote_url = remote_url[:-4]

            # Get commit hash
            commit_hash_bytes = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT).strip()
            commit_hash = commit_hash_bytes.decode('utf-8')
            logging.info(f"Rank 0: Fetched Git Info - URL: {remote_url}, Commit: {commit_hash}")
        except subprocess.CalledProcessError as e:
            logging.warning(f"Rank 0: Could not get Git info: {e.output.decode('utf-8').strip()}")
        except FileNotFoundError:
            logging.warning("Rank 0: 'git' command not found. Cannot log Git info.")
        except Exception as e_git:
                logging.error(f"Rank 0: An unexpected error occurred while fetching Git info: {e_git}", exc_info=True)
                
    # All workers log their contribution
    logging.info(f"Worker {worker_id} completed optimization")

    # Generate visualizations if enabled (only rank 0 should do this)
    if worker_id == '0' and config.get("optuna", {}).get("visualization", {}).get("enabled", False):
        if study:
            try:
                from wind_forecasting.tuning.utils.optuna_utils import generate_visualizations
                # Import the path resolution helper from db_utils or optuna_db_utils
                # from wind_forecasting.tuning.utils.db_utils import _resolve_path

                vis_config = config["optuna"]["visualization"]

                # Resolve the output directory using the helper function and full config
                default_vis_path = os.path.join(config.get("logging", {}).get("optuna_dir", "logging/optuna"), "visualizations")
                # Pass vis_config as the dict containing 'output_dir', key 'output_dir', and the full 'config'
                # visualization_dir = _resolve_path(vis_config, "output_dir", full_config=config, default=default_vis_path)
                visualization_dir = vis_config.get("output_dir", default_vis_path)
                if not visualization_dir:
                     logging.error("Rank 0: Could not determine visualization output directory. Skipping visualization.")
                else:
                    logging.info(f"Rank 0: Resolved visualization output directory: {visualization_dir}")
                    os.makedirs(visualization_dir, exist_ok=True) # Ensure directory exists

                    # Generate plots
                    logging.info(f"Rank 0: Generating Optuna visualizations in {visualization_dir}")
                    summary_path = generate_visualizations(study, visualization_dir, vis_config) # Pass vis_config

                    if summary_path:
                        logging.info(f"Rank 0: Generated Optuna visualizations - summary available at: {summary_path}")
                    else:
                        logging.warning("Rank 0: No visualizations were generated - study may not have enough completed trials or an error occurred.")

            except ImportError:
                 logging.warning("Rank 0: Could not import visualization modules. Skipping visualization generation.")
            except Exception as e:
                logging.error(f"Rank 0: Failed to generate Optuna visualizations: {e}", exc_info=True)
        else:
             logging.warning("Rank 0: Study object not available, cannot generate visualizations.")

    # Log best trial details (only rank 0)
    if worker_id == '0' and study: # Check if study object exists
        if len(study.trials) > 0:
            logging.info("Number of finished trials: {}".format(len(study.trials)))
            logging.info("Best trial:")
            trial = study.best_trial
            logging.info("  Value: {}".format(trial.value))
            logging.info("  Params: ")
            for key, value in trial.params.items():
                logging.info("    {}: {}".format(key, value))
        else:
            logging.warning("No trials were completed")

        # Generate and print Optuna Dashboard command
        db_setup_params = generate_db_setup_params(model, config)
        dashboard_command_output = generate_optuna_dashboard_command(db_setup_params, final_study_name)
        logging.info(dashboard_command_output)

    return study.best_params