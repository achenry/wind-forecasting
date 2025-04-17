import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_metric_comparison_bar(metrics_df: pd.DataFrame, output_dir: str, filename: str = "metric_comparison_bar.png"):
    """
    Generates a grouped bar chart comparing models based on specified metrics.

    Args:
        metrics_df (pd.DataFrame): DataFrame with columns like ['model_name', 'metric', 'score', 'turbine_id', 'feature'].
                                   Assumes scores are averaged appropriately before passing.
        output_dir (str): Directory to save the plot.
        filename (str): Name for the output plot file.
    """
    plt.figure(figsize=(12, 7))
    # Example: Bar plot comparing average score per metric across models
    # You might need to aggregate or pivot metrics_df depending on its exact structure
    # This example assumes metrics_df has 'model_name', 'metric', and 'score' (averaged score)
    
    # Ensure 'score' is numeric
    metrics_df['score'] = pd.to_numeric(metrics_df['score'], errors='coerce')
    metrics_df.dropna(subset=['score'], inplace=True)

    # Aggregate if necessary (example: mean score per model per metric)
    agg_metrics = metrics_df.groupby(['model_name', 'metric'])['score'].mean().reset_index()

    sns.barplot(data=agg_metrics, x='metric', y='score', hue='model_name')
    
    plt.title('Model Comparison based on Deterministic Metrics')
    plt.ylabel('Average Score')
    plt.xlabel('Metric')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Model')
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.close()
    print(f"Metric comparison bar chart saved to {output_path}")

def plot_metric_comparison_table(metrics_df: pd.DataFrame, output_dir: str, filename: str = "metric_comparison_table.tex"):
    """
    Generates a LaTeX table comparing models based on specified metrics.

    Args:
        metrics_df (pd.DataFrame): DataFrame with columns like ['model_name', 'metric', 'score', ...].
        output_dir (str): Directory to save the table.
        filename (str): Name for the output LaTeX file.
    """
    # Example: Pivot table for LaTeX output
    # This assumes metrics_df has 'model_name', 'metric', and 'score' (averaged score)
    
    # Ensure 'score' is numeric
    metrics_df['score'] = pd.to_numeric(metrics_df['score'], errors='coerce')
    metrics_df.dropna(subset=['score'], inplace=True)
    
    # Aggregate if necessary (example: mean score per model per metric)
    agg_metrics = metrics_df.groupby(['model_name', 'metric'])['score'].mean().reset_index()

    pivot_df = agg_metrics.pivot(index='model_name', columns='metric', values='score')
    
    # Format for better readability in LaTeX
    pivot_df = pivot_df.round(3) 
    
    latex_string = pivot_df.to_latex(
        caption='Comparison of Models based on Deterministic Metrics',
        label='tab:metric_comparison',
        escape=False, # Allows LaTeX commands in headers/index if needed
        na_rep='-'   # Representation for missing values
    )
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w') as f:
        f.write(latex_string)
    print(f"Metric comparison LaTeX table saved to {output_path}")


def plot_ts_comparison(true_df: pd.DataFrame, 
                       predictions_df: pd.DataFrame, 
                       model_names: list, 
                       output_dir: str, 
                       turbine_id: str, 
                       feature: str, 
                       time_window: tuple = None, 
                       filename_suffix: str = ""):
    """
    Generates time series plots comparing true vs. predicted values for a specific
    turbine and feature across different models.

    Args:
        true_df (pd.DataFrame): DataFrame with true values (columns: time, turbine_id, feature, value).
        predictions_df (pd.DataFrame): DataFrame with predictions from all models 
                                        (columns: time, turbine_id, feature, value, model_name).
        model_names (list): List of model names to plot.
        output_dir (str): Directory to save the plot.
        turbine_id (str): The specific turbine ID to plot.
        feature (str): The specific feature (e.g., 'ws_horz') to plot.
        time_window (tuple, optional): (start_time, end_time) to limit the plot. Defaults to None.
        filename_suffix (str, optional): Suffix to append to the output filename.
    """
    
    # Ensure time column is datetime
    true_df['time'] = pd.to_datetime(true_df['time'])
    predictions_df['time'] = pd.to_datetime(predictions_df['time'])

    # Filter data for the specific turbine and feature
    true_filt = true_df[(true_df['turbine_id'] == turbine_id) & (true_df['feature'] == feature)].copy()
    pred_filt = predictions_df[(predictions_df['turbine_id'] == turbine_id) & 
                               (predictions_df['feature'] == feature) & 
                               (predictions_df['model_name'].isin(model_names))].copy()

    if not true_filt.empty:
        # Apply time window if specified
        if time_window:
            start_time, end_time = pd.to_datetime(time_window[0]), pd.to_datetime(time_window[1])
            true_filt = true_filt[(true_filt['time'] >= start_time) & (true_filt['time'] <= end_time)]
            pred_filt = pred_filt[(pred_filt['time'] >= start_time) & (pred_filt['time'] <= end_time)]

        if true_filt.empty and pred_filt.empty:
             print(f"Warning: No data found for turbine {turbine_id}, feature {feature} within the specified time window. Skipping plot.")
             return
             
        plt.figure(figsize=(15, 6))
        
        # Plot true data
        sns.lineplot(data=true_filt, x='time', y='value', label='True', color='black', linestyle='--')
        
        # Plot predictions for each model
        sns.lineplot(data=pred_filt, x='time', y='value', hue='model_name', style='model_name')
        
        plt.title(f'Time Series Comparison for Turbine {turbine_id} - Feature {feature}')
        plt.xlabel('Time')
        plt.ylabel(feature)
        plt.legend()
        plt.tight_layout()

        # Construct filename
        base_filename = f"ts_comparison_{turbine_id}_{feature}"
        if filename_suffix:
            base_filename += f"_{filename_suffix}"
        filename = f"{base_filename}.png"
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path)
        plt.close()
        print(f"Time series comparison plot saved to {output_path}")
    else:
        print(f"Warning: No true data found for turbine {turbine_id}, feature {feature}. Skipping plot.") 