import os
import logging
import datetime
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_slice
)

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