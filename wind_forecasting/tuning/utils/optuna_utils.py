"""
Optuna-specific utilities for hyperparameter optimization.

This module consolidates all Optuna-related functionality including:
- Sampler and pruner persistence across distributed workers
- Dashboard launching and management  
- Visualization generation and W&B logging
- Detailed trial table logging

These utilities are specific to the Optuna hyperparameter optimization framework
and support distributed tuning workflows.
"""

import os
import pickle
import shutil
import tempfile
import logging
import datetime
import subprocess
import atexit
import getpass
import socket
from pathlib import Path

import wandb
import plotly.io as pio
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_slice,
    plot_parallel_coordinate
)

# Global variable for dashboard process management
_dashboard_process = None


class OptunaSamplerPrunerPersistence:
    """
    Manages creation of Optuna samplers and pruners for distributed workers.

    This class creates fresh sampler and pruner instances for each worker, enabling:
    - Proper distributed optimization without pickle state conflicts
    - Diverse hyperparameter suggestions during TPE startup phase
    - Shared learning through database-backed study coordination

    Note: Pickle persistence has been removed to fix identical hyperparameter issues.
    """
    
    def __init__(self, config, seed):
        self.config = config
        self.seed = seed

    def _create_new_sampler_pruner(self, pruner, worker_id=0):
        """
        Create new sampler and pruner instances with job-aware worker-specific seeds.

        This ensures that different SLURM jobs explore different hyperparameter regions
        while maintaining worker diversity within each job.

        Args:
            pruner: Pre-configured pruner instance
            worker_id: Worker ID for generating unique seed

        Returns:
            tuple: (sampler, pruner)
        """
        # Generate job-aware worker-specific seed
        job_id = os.environ.get('SLURM_JOB_ID', '0')
        worker_id_int = int(worker_id) if isinstance(worker_id, str) else worker_id
        job_id_int = int(job_id)

        if self.seed is not None:
            # Generate job-aware seed using hash to stay within numpy seed limits (0 to 2^32-1)
            import hashlib

            # Create a string combining base seed, job ID, and worker ID
            seed_string = f"{self.seed}_{job_id_int}_{worker_id_int}"

            # Hash and convert to valid seed range
            hash_object = hashlib.md5(seed_string.encode())
            # Take first 4 bytes and convert to integer within numpy's seed range
            worker_seed = int.from_bytes(hash_object.digest()[:4], byteorder='big') % (2**32)
        else:
            worker_seed = None

        sampler = TPESampler(
            seed=worker_seed,
            n_startup_trials=self.config["optuna"]["sampler_params"]["tpe"].get("n_startup_trials", 16),
            multivariate=self.config["optuna"]["sampler_params"]["tpe"].get("multivariate", True),
            constant_liar=self.config["optuna"]["sampler_params"]["tpe"].get("constant_liar", True),
            group=self.config["optuna"]["sampler_params"]["tpe"].get("group", False)
        )
        return sampler, pruner

    def _save_sampler_pruner_atomic(self, sampler, pruner, sampler_path, pruner_path):
        """
        Save sampler and pruner to pickle files using atomic operations.
        
        Args:
            sampler: Sampler instance to save
            pruner: Pruner instance to save
            sampler_path: Path to save sampler pickle
            pruner_path: Path to save pruner pickle
        """
        # Ensure pickle directory exists
        pickle_dir = os.path.dirname(sampler_path)
        os.makedirs(pickle_dir, exist_ok=True)
        
        # Save sampler atomically
        with tempfile.NamedTemporaryFile(mode='wb', dir=pickle_dir, delete=False) as temp_file:
            pickle.dump(sampler, temp_file)
            temp_sampler_path = temp_file.name
        shutil.move(temp_sampler_path, sampler_path)
        
        # Save pruner atomically
        with tempfile.NamedTemporaryFile(mode='wb', dir=pickle_dir, delete=False) as temp_file:
            pickle.dump(pruner, temp_file)
            temp_pruner_path = temp_file.name
        shutil.move(temp_pruner_path, pruner_path)
        
        logging.info(f"Saved sampler/pruner to {sampler_path} and {pruner_path}")

    def _load_sampler_pruner_atomic(self, sampler_path, pruner_path):
        """
        Load sampler and pruner from pickle files.
        
        Args:
            sampler_path: Path to sampler pickle file
            pruner_path: Path to pruner pickle file
            
        Returns:
            tuple: (sampler, pruner) or None if loading fails
        """
        try:
            with open(sampler_path, 'rb') as f:
                sampler = pickle.load(f)
            with open(pruner_path, 'rb') as f:
                pruner = pickle.load(f)
            logging.info(f"Loaded sampler/pruner from {sampler_path} and {pruner_path}")
            return sampler, pruner
        except Exception as e:
            logging.warning(f"Failed to load sampler/pruner: {e}")
            return None

    def get_sampler_pruner_objects(self, worker_id, pruner, restart_tuning=None, final_study_name=None, optuna_storage=None, pickle_dir=None):
        """
        Get sampler and pruner objects by creating fresh instances for each worker.

        This approach ensures proper distributed optimization without pickle persistence,
        allowing each worker to create its own TPESampler instance while still benefiting
        from shared study history through the database backend.

        Args:
            worker_id: ID of the current worker
            pruner: Configured pruner instance
            restart_tuning: Whether this is a fresh start (not used anymore)
            final_study_name: Name of the Optuna study
            optuna_storage: Optuna storage backend
            pickle_dir: Directory for pickle files (not used anymore)

        Returns:
            tuple: (sampler, pruner)
        """
        # Calculate seed for logging (matches the calculation in _create_new_sampler_pruner)
        job_id = os.environ.get('SLURM_JOB_ID', '0')
        worker_id_int = int(worker_id) if isinstance(worker_id, str) else worker_id
        job_id_int = int(job_id)

        if self.seed is not None:
            # Calculate seed using same hash method as _create_new_sampler_pruner
            import hashlib

            seed_string = f"{self.seed}_{job_id_int}_{worker_id_int}"
            hash_object = hashlib.md5(seed_string.encode())
            calculated_seed = int.from_bytes(hash_object.digest()[:4], byteorder='big') % (2**32)
        else:
            calculated_seed = None

        logging.info(f"Worker {worker_id} (Job {job_id}): Creating fresh sampler/pruner objects with job-aware seed {calculated_seed}")
        sampler, pruner_for_study = self._create_new_sampler_pruner(pruner, worker_id=worker_id)

        return sampler, pruner_for_study

    def create_trial_completion_callback(self, worker_id=None, save_frequency=None):
        """
        Create a callback for trial completion (no longer saves pickle state).

        With the removal of pickle persistence, this callback no longer saves sampler state.
        Each worker maintains its own fresh TPESampler instance and relies on the shared
        database for coordinated optimization.

        Args:
            worker_id: ID of the current worker
            save_frequency: Not used anymore but kept for API compatibility

        Returns:
            Callback function for study.optimize() (currently a no-op)
        """
        def no_op_callback(study=None, trial=None):
            # No longer saving pickle state - distributed optimization works through database
            pass

        return no_op_callback


def _terminate_optuna_dashboard():
    """Terminate the background optuna-dashboard process if it's running."""
    global _dashboard_process
    if _dashboard_process and _dashboard_process.poll() is None:
        logging.info("Terminating background optuna-dashboard process...")
        _dashboard_process.terminate()
        _dashboard_process.wait()
        logging.info("optuna-dashboard process terminated.")


def launch_optuna_dashboard(config, storage_url):
    """
    Launch the Optuna dashboard in the background.
    
    Args:
        config: Configuration dictionary
        storage_url: Database storage URL for Optuna
    """
    global _dashboard_process
    
    if _dashboard_process and _dashboard_process.poll() is None:
        logging.info("optuna-dashboard is already running.")
        return
    
    try:
        # Register cleanup function
        atexit.register(_terminate_optuna_dashboard)
        
        # Dashboard configuration
        dashboard_host = config.get("optuna", {}).get("dashboard", {}).get("host", "0.0.0.0")
        dashboard_port = config.get("optuna", {}).get("dashboard", {}).get("port", 8081)
        
        # Create log directory
        dashboard_log_dir = Path(config.get("logging", {}).get("optuna_dir", "logging/optuna")) / "dashboard_logs"
        dashboard_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate log filename with timestamp and worker info
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        username = getpass.getuser()
        hostname = socket.gethostname()
        log_filename = f"dashboard_{timestamp}_{username}_{hostname}.log"
        log_path = dashboard_log_dir / log_filename
        
        # Launch dashboard
        cmd = [
            "optuna-dashboard",
            storage_url,
            "--host", dashboard_host,
            "--port", str(dashboard_port)
        ]
        
        with open(log_path, 'w') as log_file:
            _dashboard_process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True
            )
        
        logging.info(f"Launched optuna-dashboard on {dashboard_host}:{dashboard_port}")
        logging.info(f"Dashboard logs: {log_path}")
        
    except Exception as e:
        logging.error(f"Failed to launch optuna-dashboard: {e}")


def generate_visualizations(study, output_dir, vis_config=None):
    """
    Generate comprehensive Optuna visualizations.
    
    Args:
        study: Optuna study object
        output_dir: Directory to save visualizations
        vis_config: Visualization configuration
        
    Returns:
        str: Path to HTML summary file, or None if generation failed
    """
    try:
        completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if len(completed_trials) < 2:
            logging.warning(f"Insufficient completed trials ({len(completed_trials)}) for meaningful visualizations. Need at least 2.")
            return None
        
        logging.info(f"Generating visualizations for {len(completed_trials)} completed trials...")
        
        # Generate individual plots
        plots = {}
        
        # Optimization history
        try:
            plots['optimization_history'] = plot_optimization_history(study)
            plots['optimization_history'].write_html(os.path.join(output_dir, "optimization_history.html"))
        except Exception as e:
            logging.warning(f"Failed to generate optimization history plot: {e}")
        
        # Parameter importances
        try:
            plots['param_importances'] = plot_param_importances(study)
            plots['param_importances'].write_html(os.path.join(output_dir, "param_importances.html"))
        except Exception as e:
            logging.warning(f"Failed to generate parameter importances plot: {e}")
        
        # Parameter relationships
        try:
            plots['parallel_coordinate'] = plot_parallel_coordinate(study)
            plots['parallel_coordinate'].write_html(os.path.join(output_dir, "parallel_coordinate.html"))
        except Exception as e:
            logging.warning(f"Failed to generate parallel coordinate plot: {e}")
        
        # Additional plots if sufficient trials
        if len(completed_trials) >= 3:
            try:
                plots['slice'] = plot_slice(study)
                plots['slice'].write_html(os.path.join(output_dir, "slice.html"))
            except Exception as e:
                logging.warning(f"Failed to generate slice plot: {e}")
        
        # Generate HTML summary
        summary_path = generate_html_summary(study, output_dir, plots)
        logging.info(f"Generated {len(plots)} visualization plots")
        
        return summary_path
        
    except Exception as e:
        logging.error(f"Failed to generate visualizations: {e}", exc_info=True)
        return None


def generate_html_summary(study, output_dir, plots):
    """
    Generate an HTML summary page linking all visualizations.
    
    Args:
        study: Optuna study object
        output_dir: Directory containing visualizations
        plots: Dictionary of generated plots
        
    Returns:
        str: Path to summary HTML file
    """
    summary_path = os.path.join(output_dir, "summary.html")
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Optuna Study: {study.study_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .plot-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 20px; }}
            .plot-card {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
            .plot-card h3 {{ margin-top: 0; }}
            .best-trial {{ background-color: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Optuna Study: {study.study_name}</h1>
            <p><strong>Total Trials:</strong> {len(study.trials)}</p>
            <p><strong>Completed Trials:</strong> {len([t for t in study.trials if t.state == TrialState.COMPLETE])}</p>
            <p><strong>Direction:</strong> {study.direction.name}</p>
        </div>
    """
    
    # Best trial information
    try:
        best_trial = study.best_trial
        html_content += f"""
        <div class="best-trial">
            <h2>Best Trial</h2>
            <p><strong>Trial #{best_trial.number}</strong></p>
            <p><strong>Value:</strong> {best_trial.value:.6f}</p>
            <p><strong>Parameters:</strong></p>
            <ul>
        """
        for key, value in best_trial.params.items():
            html_content += f"<li><strong>{key}:</strong> {value}</li>"
        html_content += "</ul></div>"
    except ValueError:
        html_content += '<div class="best-trial"><h2>Best Trial</h2><p>No completed trials available.</p></div>'
    
    # Plot links
    html_content += '<div class="plot-grid">'
    plot_titles = {
        'optimization_history': 'Optimization History',
        'param_importances': 'Parameter Importances',
        'parallel_coordinate': 'Parallel Coordinates',
        'slice': 'Slice Plots',
        'contour': 'Contour Plots'
    }
    
    for plot_name in plots.keys():
        title = plot_titles.get(plot_name, plot_name.replace('_', ' ').title())
        html_content += f"""
        <div class="plot-card">
            <h3>{title}</h3>
            <p><a href="{plot_name}.html" target="_blank">View Plot</a></p>
        </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(summary_path, 'w') as f:
        f.write(html_content)
    
    return summary_path


def log_optuna_visualizations_to_wandb(study, wandb_run):
    """
    Generate and log Optuna visualizations to Weights & Biases.
    
    Args:
        study: Optuna study object
        wandb_run: Active W&B run object
    """
    logging.info("Generating and logging Optuna visualizations to W&B...")
    
    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if len(completed_trials) < 2:
        logging.warning(f"Insufficient completed trials ({len(completed_trials)}) for visualizations")
        return
    
    try:
        # Optimization history
        fig = plot_optimization_history(study)
        wandb_run.log({"optuna_optimization_history": wandb.Html(pio.to_html(fig, include_plotlyjs='inline'))})
        
        # Parameter importances
        fig = plot_param_importances(study)
        wandb_run.log({"optuna_param_importances": wandb.Html(pio.to_html(fig, include_plotlyjs='inline'))})
        
        # Parallel coordinates
        fig = plot_parallel_coordinate(study)
        wandb_run.log({"optuna_parallel_coordinate": wandb.Html(pio.to_html(fig, include_plotlyjs='inline'))})
        
        # Additional plots for studies with more trials
        if len(completed_trials) >= 3:
            try:
                fig = plot_slice(study)
                wandb_run.log({"optuna_slice": wandb.Html(pio.to_html(fig, include_plotlyjs='inline'))})
            except Exception as e:
                logging.warning(f"Could not generate slice plot: {e}")
        
        logging.info("Successfully logged Optuna visualizations to W&B")
        
    except Exception as e:
        logging.error(f"Failed to log Optuna visualizations to W&B: {e}", exc_info=True)


def log_detailed_trials_table_to_wandb(study, wandb_run):
    """
    Generate a detailed W&B table summarizing all Optuna trials and log it.

    Args:
        study: The Optuna study object
        wandb_run: The active wandb.sdk.wandb_run.Run object to log to
    """
    logging.info("Creating and logging detailed Optuna trials table to W&B...")
    
    all_trials = study.get_trials(deepcopy=False, states=None)
    if not all_trials:
        logging.warning("No trials found in the study to log to the detailed summary table.")
        return

    best_trial = None
    try:
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
        is_best = best_trial is not None and trial.number == best_trial.number and trial.state == TrialState.COMPLETE
        is_pruned = trial.state == TrialState.PRUNED
        is_failed = trial.state == TrialState.FAIL

        row_data = [
            trial.number,
            trial.state.name,
            trial.value,
            is_best,
            is_pruned,
            is_failed
        ]
        
        # Add parameter values
        for key in sorted_param_keys:
            row_data.append(trial.params.get(key, None))

        detailed_trial_table.add_data(*row_data)

    # Log the table to W&B
    wandb_run.log({"optuna_trials_detailed_summary": detailed_trial_table})
    logging.info(f"Logged detailed Optuna trials table ({len(all_trials)} trials) to W&B run {wandb_run.id}.")


# Register cleanup function when module is imported
atexit.register(_terminate_optuna_dashboard)