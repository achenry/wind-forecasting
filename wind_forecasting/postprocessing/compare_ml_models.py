import argparse
import yaml
import os
import glob
import pandas as pd
import importlib
import traceback

# Dynamically import plotting functions
from .plotting.comparison_plots import plot_metric_comparison_bar, plot_metric_comparison_table, plot_ts_comparison

# Attempt to dynamically import metrics from the sibling directory
try:
    # Assuming deterministic_metrics.py is in the same directory as __init__.py
    # Or adjust the path accordingly if it's elsewhere
    metrics_module = importlib.import_module(".deterministic_metrics", package="wind_forecasting.postprocessing") 
except ImportError:
    print("Error: Could not import deterministic_metrics. Ensure it exists in the postprocessing directory and contains the required metric functions.")
    metrics_module = None

def load_predictions(base_dir, model_key, file_pattern):
    """Loads prediction data for a given model."""
    search_path = os.path.join(base_dir, model_key, file_pattern)
    prediction_files = glob.glob(search_path)
    
    if not prediction_files:
        print(f"Warning: No prediction files found for model '{model_key}' using pattern '{search_path}'")
        return None
        
    # Assuming one prediction file per model for simplicity, load the first found
    # Adapt if multiple files need aggregation
    file_path = prediction_files[0]
    print(f"Loading predictions for model '{model_key}' from {file_path}...")
    if file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        print(f"Warning: Unsupported file format for {file_path}. Only .parquet and .csv are supported.")
        return None
        
    # --- Data Validation and Formatting ---
    # Ensure required columns exist (adjust names if necessary)
    required_cols = ['time', 'turbine_id', 'feature', 'value']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Prediction file {file_path} is missing required columns. Expected: {required_cols}")
        print(f"Actual columns: {df.columns.tolist()}")
        # Attempt to reshape if common wide format is detected (example)
        # This part needs customization based on your actual wide format
        if 'time' in df.columns and not all(col in df.columns for col in ['turbine_id', 'feature', 'value']):
             try:
                 print("Attempting to melt wide format...")
                 # Example melt operation - ADJUST ID_VARS AND VALUE_VARS
                 df = df.melt(id_vars=['time'], var_name='feature_turbine', value_name='value')
                 # Example splitting 'ws_horz_T01' into 'ws_horz' and 'T01'
                 df[['feature', 'turbine_id']] = df['feature_turbine'].str.extract(r'(\w+)_(\w+)')
                 df = df.drop(columns=['feature_turbine'])
                 df = df[required_cols] # Reorder
                 print("Melt successful.")
                 if not all(col in df.columns for col in required_cols):
                      raise ValueError("Melt failed to produce required columns.")
             except Exception as e:
                 print(f"Error: Automatic melt failed. Please ensure prediction file is in long format. {e}")
                 return None
        else:
             return None

    # Ensure correct dtypes
    df['time'] = pd.to_datetime(df['time'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df['turbine_id'] = df['turbine_id'].astype(str)
    df['feature'] = df['feature'].astype(str)
    df.dropna(subset=['value'], inplace=True)
    
    return df

def load_true_data(file_path):
    """Loads ground truth data."""
    print(f"Loading true data from {file_path}...")
    if not os.path.exists(file_path):
        print(f"Error: True data file not found at {file_path}")
        return None
        
    if file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        print(f"Warning: Unsupported file format for {file_path}. Only .parquet and .csv are supported.")
        return None
        
    # --- Data Validation and Formatting ---
    required_cols = ['time', 'turbine_id', 'feature', 'value']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: True data file {file_path} is missing required columns. Expected: {required_cols}")
        # Add melt logic here if true data might be wide, similar to load_predictions
        return None

    df['time'] = pd.to_datetime(df['time'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df['turbine_id'] = df['turbine_id'].astype(str)
    df['feature'] = df['feature'].astype(str)
    df.dropna(subset=['value'], inplace=True)

    return df

def calculate_metrics(true_df, pred_df, metric_names, model_name):
    """Calculates specified metrics for a given model's predictions."""
    if metrics_module is None:
        print("Skipping metric calculation as metrics module failed to import.")
        return pd.DataFrame()
        
    all_metrics_data = []
    
    # Ensure alignment on time, turbine, feature
    merged_df = pd.merge(true_df, pred_df, on=['time', 'turbine_id', 'feature'], suffixes=('_true', '_pred'))
    
    if merged_df.empty:
        print(f"Warning: No matching time/turbine/feature found between true data and predictions for model {model_name}. Cannot calculate metrics.")
        return pd.DataFrame()

    for metric_name in metric_names:
        try:
            metric_func = getattr(metrics_module, metric_name)
        except AttributeError:
            print(f"Warning: Metric function '{metric_name}' not found in deterministic_metrics. Skipping.")
            continue
            
        # Calculate metric per turbine/feature
        grouped = merged_df.groupby(['turbine_id', 'feature'])
        
        for (turbine_id, feature), group in grouped:
            y_true = group['value_true'].values
            y_pred = group['value_pred'].values
            
            if len(y_true) == 0:
                continue

            try:
                score = metric_func(y_true, y_pred)
                all_metrics_data.append({
                    'model_name': model_name,
                    'turbine_id': turbine_id,
                    'feature': feature,
                    'metric': metric_name,
                    'score': score
                })
            except Exception as e:
                print(f"Error calculating metric '{metric_name}' for {turbine_id}/{feature} of model {model_name}: {e}")
                # traceback.print_exc() # Uncomment for detailed error trace
                
    return pd.DataFrame(all_metrics_data)

def main(config_path):
    """Main execution function."""
    print(f"Loading configuration from {config_path}...")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        return

    # --- Load Data --- 
    true_data = load_true_data(config['true_data_path'])
    if true_data is None:
        return
        
    all_predictions = []
    all_metrics = []
    model_display_names = []

    for model_info in config.get('models_to_compare', []):
        model_key = model_info['key']
        model_name = model_info['name']
        model_display_names.append(model_name)
        
        pred_df = load_predictions(
            config['prediction_base_dir'], 
            model_key, 
            config['prediction_file_pattern']
        )
        
        if pred_df is not None:
            pred_df['model_name'] = model_name # Add model name for aggregation
            all_predictions.append(pred_df)
            
            # Calculate metrics for this model
            metrics_df = calculate_metrics(
                true_data, 
                pred_df, 
                config.get('metrics', []), 
                model_name
            )
            if not metrics_df.empty:
                all_metrics.append(metrics_df)

    if not all_predictions:
        print("Error: No prediction data could be loaded for any model. Exiting.")
        return
        
    # Combine predictions and metrics from all models
    combined_predictions = pd.concat(all_predictions, ignore_index=True)
    combined_metrics = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()

    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Generate Plots & Tables --- 
    if not combined_metrics.empty:
        print("\n--- Generating Metric Comparisons ---")
        plot_metric_comparison_bar(combined_metrics, output_dir)
        plot_metric_comparison_table(combined_metrics, output_dir)
    else:
        print("\n--- Skipping Metric Comparisons (no metrics calculated) ---")

    print("\n--- Generating Time Series Comparisons ---")
    for ts_plot_config in config.get('timeseries_plots', []):
        turbine_id = ts_plot_config['turbine_id']
        feature = ts_plot_config['feature']
        time_window = ts_plot_config.get('time_window', None)
        
        plot_ts_comparison(
            true_data, 
            combined_predictions, 
            model_display_names, 
            output_dir, 
            turbine_id, 
            feature, 
            time_window=time_window
        )
        
    print("\n--- Regarding 'predictions made every controller_dt' plot ---")
    print("The generation of plots showing predictions made at each controller timestep requires specific data logging during model testing or live model inference within a simulated loop.")
    print("Standard testing procedures usually output predictions for the entire test set at once.")
    print("If your prediction files contain information about *when* each forecast sequence was generated (e.g., a 'forecast_generation_time' column), this plot could be adapted.")
    print("Otherwise, consider modifying your model testing script (`run_model.py`) to save predictions in such a manner or simulating the prediction calls step-by-step.")

    print("\nComparison script finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare ML Wind Forecasting Models")
    parser.add_argument("--config", type=str, required=True, 
                        help="Path to the YAML configuration file (e.g., compare_config.yaml)")
    args = parser.parse_args()
    main(args.config) 