import os
import logging
import datetime
import optuna
import subprocess # For launching dashboard
import atexit # For cleaning up dashboard process
from pathlib import Path # For resolving log path
import getpass # To get current username
import socket # To get current hostname (compute node)
import wandb # Added for W&B logging
import plotly.io as pio # Added for Plotly HTML conversion
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_slice,
    plot_parallel_coordinate,
    plot_contour,
    plot_intermediate_values
)

# --- Dashboard Auto-Launch Globals ---
_dashboard_process = None

# --- Dashboard Auto-Launch Functions ---

def _terminate_optuna_dashboard():
    """Terminate the background optuna-dashboard process if it's running."""
    global _dashboard_process
    if _dashboard_process and _dashboard_process.poll() is None: # Check if process exists and is running
        logging.info("Terminating Optuna dashboard process...")
        try:
            _dashboard_process.terminate() # Send SIGTERM
            _dashboard_process.wait(timeout=5) # Wait a bit for graceful shutdown
            logging.info("Optuna dashboard process terminated.")
        except subprocess.TimeoutExpired:
            logging.warning("Optuna dashboard process did not terminate gracefully, sending SIGKILL.")
            _dashboard_process.kill() # Force kill
        except Exception as e:
            logging.error(f"Error terminating Optuna dashboard process: {e}")
        _dashboard_process = None

def launch_optuna_dashboard(config, storage_url):
    """
    Launch optuna-dashboard as a background process.
    Should only be called by rank 0.
    """
    global _dashboard_process
    dashboard_config = config.get("optuna", {}).get("dashboard", {})

    if not dashboard_config.get("enabled", False):
        logging.info("Optuna dashboard auto-launch is disabled in the configuration.")
        return

    if _dashboard_process and _dashboard_process.poll() is None:
        logging.warning("Optuna dashboard process seems to be already running.")
        return

    port = dashboard_config.get("port", 8088) # Default to 8088 if not specified
    log_file_path_str = dashboard_config.get("log_file", None)

    # Resolve log file path using similar logic to db_utils._resolve_path
    if log_file_path_str:
        # Basic variable substitution (can be enhanced if more variables are needed)
        if "${logging.optuna_dir}" in log_file_path_str:
            optuna_dir = config.get("logging", {}).get("optuna_dir")
            if not optuna_dir:
                 logging.error("Cannot resolve dashboard log_file path: logging.optuna_dir not defined.")
                 return
            # Ensure optuna_dir is absolute
            if not Path(optuna_dir).is_absolute():
                 project_root_str = config.get("experiment", {}).get("project_root", os.getcwd())
                 optuna_dir = str((Path(project_root_str) / Path(optuna_dir)).resolve())
            log_file_path_str = log_file_path_str.replace("${logging.optuna_dir}", optuna_dir)

        # Resolve relative to project root if not absolute
        log_file_path = Path(log_file_path_str)
        if not log_file_path.is_absolute():
            project_root_str = config.get("experiment", {}).get("project_root", os.getcwd())
            log_file_path = (Path(project_root_str) / log_file_path).resolve()
        else:
            log_file_path = log_file_path.resolve()

        # Ensure log directory exists
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        log_file_path_str = str(log_file_path)
        logging.info(f"Optuna dashboard log file: {log_file_path_str}")
    else:
        logging.warning("Optuna dashboard log_file not specified. Output will not be saved.")
        log_file_path_str = os.devnull # Redirect to null device if no log file specified

    # Construct the command
    # Ensure storage_url is correctly formatted for the command line
    # (especially handling potential special characters if not using sockets)
    cmd = [
        "optuna-dashboard",
        "--port", str(port),
        storage_url # Pass the full storage URL
    ]

    logging.info(f"Launching Optuna dashboard in background: {' '.join(cmd)}")
    logging.info(f"Dashboard will listen on port {port}. Use SSH tunneling to access.")

    try:
        # Open the log file for writing stdout/stderr
        log_handle = open(log_file_path_str, 'a') if log_file_path_str != os.devnull else subprocess.DEVNULL

        # Launch the process in the background

        _dashboard_process = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=log_handle,
            # Close file descriptors in the child process to avoid issues
            close_fds=True,
            # Use current environment, assuming necessary paths (like conda env bin) are set
            env=os.environ.copy(),
            # shell=True # CHANGE TODO shel work on linux?
        )
        logging.info(f"Optuna dashboard process started (PID: {_dashboard_process.pid}).")

        # --- Log SSH Tunnel Instructions ---
        try:
            username = getpass.getuser()
            compute_node_hostname = socket.gethostname() # Get the hostname where this script is running
            local_port = port # Use the same port locally for simplicity
            login_node_placeholder = "YOUR_LOGIN_NODE_ADDRESS" # Placeholder

            logging.info("----------------------------------------------------------------------")
            logging.info("Optuna Dashboard Access Instructions:")
            logging.info(f"Dashboard is running on compute node '{compute_node_hostname}' on port {port}.")
            logging.info("To access it from your local machine, set up a double SSH tunnel:")
            logging.info("")
            logging.info("1. **First Tunnel (Local PC to Login Node):**")
            logging.info("   Open a terminal on your LOCAL machine and run:")
            logging.info(f"   ssh -L {local_port}:localhost:{port} {username}@{login_node_placeholder}")
            logging.info(f"   (Replace {login_node_placeholder} with the actual login node address you use)")
            logging.info("   Keep this terminal open.")
            logging.info("")
            logging.info("2. **Second Tunnel (Login Node to Compute Node):**")
            logging.info("   Open ANOTHER terminal, SSH into the LOGIN node, and then run:")
            logging.info(f"   ssh -L {port}:localhost:{port} {username}@{compute_node_hostname}")
            logging.info("   Keep this terminal open.")
            logging.info("")
            logging.info("3. **Access Dashboard:**")
            logging.info("   Open a web browser on your LOCAL machine and go to:")
            logging.info(f"   http://localhost:{local_port}")
            logging.info("----------------------------------------------------------------------")

        except Exception as log_e:
            logging.warning(f"Could not generate full SSH instructions: {log_e}")
        # -----------------------------------

        # Register cleanup function to terminate the dashboard on exit
        atexit.register(_terminate_optuna_dashboard)

    except FileNotFoundError:
        logging.error("Error: 'optuna-dashboard' command not found. Is it installed and in the PATH?")
        _dashboard_process = None
    except Exception as e:
        logging.error(f"Failed to launch Optuna dashboard: {e}")
        _dashboard_process = None
        if log_file_path_str != os.devnull and 'log_handle' in locals():
            log_handle.close() # Close the handle if opened

# --- Visualization Generation ---
def generate_visualizations(study, output_dir, config=None):
    """
    Generate interactive visualization plots for an Optuna study.

    Args:
        study: Optuna study object
        output_dir: Directory to save visualizations
        config: Visualization configuration dictionary

    Returns:
        Path to the HTML summary file
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = study.study_name

    # Default configuration if none provided
    if not config:
        config = {
            "plots": {
                "optimization_history": True,
                "parameter_importance": True,
                "slice_plot": True
            }
        }

    logging.info(f"Generating interactive Optuna visualizations in {output_dir}")

    # Tracking generated plot files for the index
    generated_plots = []

    # Only proceed if we have completed trials
    if len(study.trials) == 0 or not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        logging.warning("No completed trials found in study. Skipping visualization generation.")
        return None

    # 1. Optimization History
    if config["plots"].get("optimization_history", True):
        try:
            fig = plot_optimization_history(study)
            file_path = os.path.join(output_dir, f"{study_name}_optimization_history_{timestamp}.html")
            fig.write_html(file_path)
            generated_plots.append(("Optimization History", os.path.basename(file_path)))
            logging.info(f"Saved interactive optimization history plot to {file_path}")
        except Exception as e:
            logging.error(f"Error generating optimization history plot: {e}")

    # 2. Parameter Importance
    if config["plots"].get("parameter_importance", True):
        try:
            fig = plot_param_importances(study)
            file_path = os.path.join(output_dir, f"{study_name}_param_importance_{timestamp}.html")
            fig.write_html(file_path)
            generated_plots.append(("Parameter Importance", os.path.basename(file_path)))
            logging.info(f"Saved interactive parameter importance plot to {file_path}")
        except Exception as e:
            logging.error(f"Error generating parameter importance plot: {e}")

    # 3. Slice Plot
    if config["plots"].get("slice_plot", True):
        try:
            fig = plot_slice(study)
            file_path = os.path.join(output_dir, f"{study_name}_slice_plot_{timestamp}.html")
            fig.write_html(file_path)
            generated_plots.append(("Parameter Slice Plot", os.path.basename(file_path)))
            logging.info(f"Saved interactive slice plot to {file_path}")
        except Exception as e:
            logging.error(f"Error generating slice plot: {e}")

    # Generate HTML summary page with links to all interactive plots
    summary_path = generate_html_summary(study, output_dir, timestamp, generated_plots)

    return summary_path

def generate_html_summary(study, output_dir, timestamp, generated_plots):
    """
    Generate an HTML summary page with links to all interactive plots and study details.

    Args:
        study: Optuna study object
        output_dir: Directory to save the summary
        timestamp: Timestamp string for unique filenames
        generated_plots: List of tuples (plot_name, plot_filename)

    Returns:
        Path to the generated HTML summary file
    """
    # Create the summary file
    html_path = os.path.join(output_dir, f"index_{timestamp}.html")

    with open(html_path, 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Optuna Study: {study.study_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #2c3e50; }}
        .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        .plot-container {{ margin-bottom: 30px; }}
        .plot-link {{ display: block; margin-top: 5px; color: #3498db; text-decoration: none; padding: 8px;
                    border: 1px solid #ddd; border-radius: 4px; background-color: #f9f9f9; }}
        .plot-link:hover {{ background-color: #eaeaea; }}
        .best-trial {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin-top: 20px; }}
        .pruned-info {{ color: #e67e22; }}
        .completed-info {{ color: #27ae60; }}
    </style>
</head>
<body>
    <h1>Optuna Study Summary: {study.study_name}</h1>
    <div class="summary">
        <h2>Study Details</h2>
        <table>
            <tr><th>Study Name</th><td>{study.study_name}</td></tr>
            <tr><th>Direction</th><td>{study.direction}</td></tr>
            <tr><th>Number of Trials</th><td>{len(study.trials)}</td></tr>
            <tr><th>Number of Completed Trials</th><td class="completed-info">{len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}</td></tr>
            <tr><th>Number of Pruned Trials</th><td class="pruned-info">{len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}</td></tr>
            <tr><th>Best Value</th><td>{study.best_value if study.best_trial else "N/A"}</td></tr>
            <tr><th>Generated</th><td>{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
        </table>
""")

        # Add best parameters if available
        if study.best_trial:
            f.write(f"""
        <div class="best-trial">
            <h2>Best Trial Parameters (Trial #{study.best_trial.number})</h2>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
""")

            # Add best parameters
            for param, value in study.best_params.items():
                f.write(f"                <tr><td>{param}</td><td>{value}</td></tr>\n")

            f.write("""            </table>
        </div>
""")

        f.write("""    </div>

    <h2>Interactive Visualizations</h2>
""")

        # Add links to interactive plots
        if generated_plots:
            for plot_name, plot_filename in generated_plots:
                f.write(f"""    <div class="plot-container">
        <h3>{plot_name}</h3>
        <a class="plot-link" href="{plot_filename}" target="_blank">
            Open Interactive {plot_name} Plot
        </a>
    </div>
""")
        else:
            f.write("""    <p>No plots were generated.</p>""")

        f.write("""</body>
</html>""")

    # Create a simple redirect to the latest index
    latest_index_path = os.path.join(output_dir, "index.html")
    with open(latest_index_path, 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="refresh" content="0; url={os.path.basename(html_path)}" />
</head>
<body>
    <p>Redirecting to <a href="{os.path.basename(html_path)}">latest visualization</a>...</p>
</body>
</html>""")

    logging.info(f"Generated HTML summary at {html_path}")
    logging.info(f"Created redirection at {latest_index_path}")

    return html_path

def log_optuna_visualizations_to_wandb(study: optuna.Study, wandb_run):
    """
    Generates Optuna visualization plots and logs them to an active W&B run.

    Args:
        study: The Optuna study object.
        wandb_run: The active wandb.sdk.wandb_run.Run object.
    """
    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        logging.warning("No completed trials found in study. Skipping W&B visualization logging.")
        return

    plot_functions = {
        "optuna_optimization_history": plot_optimization_history,
        "optuna_param_importances": plot_param_importances,
        "optuna_parallel_coordinate": plot_parallel_coordinate,
        "optuna_contour": plot_contour,
        "optuna_slice": plot_slice,
        "optuna_intermediate_values": plot_intermediate_values,
    }

    logging.info(f"Logging Optuna visualizations to W&B run: {wandb_run.id}")

    for plot_key, plot_func in plot_functions.items():
        try:
            fig = plot_func(study)
            html_string = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
            wandb_run.log({plot_key: wandb.Html(html_string)})
            logging.debug(f"Successfully logged Optuna plot '{plot_key}' to W&B.")
        except ValueError as ve:
            # Optuna often raises ValueError for plots that cannot be generated
            # (e.g., param_importances with no completed trials, contour/slice with too few params)
            logging.warning(f"Could not generate/log Optuna plot '{plot_key}' (likely due to study state): {ve}")
        except Exception as e:
            # Catch other potential errors during plot generation or logging
            logging.warning(f"Could not generate/log Optuna plot '{plot_key}': {e}")