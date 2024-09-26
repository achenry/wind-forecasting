# INFO: MIC used for feature selection from minepy package. Only available for Python <3.10
# Can use sklearn.feature_selection.mutual_info_regression instead?
import time
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# from minepy import MINE
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
from numba import jit, prange
from tqdm.auto import tqdm
import multiprocessing
# from joblib import Parallel, delayed
from functools import partial

SECONDS_PER_HOUR = np.float64(3600)
SECONDS_PER_DAY = 86400
SECONDS_PER_YEAR = 31536000  # non-leap year, 365 days

def calculate_wind_direction(u, v):
    return np.mod(180 + np.rad2deg(np.arctan2(u, v)), 360)
    
def calculate_mi_for_chunk(args):
    X, y, y_direction, chunk = args
    mi_scores_u = np.zeros(X.shape[2])
    mi_scores_v = np.zeros(X.shape[2])
    mi_scores_dir = np.zeros(X.shape[2])
    for i, j in chunk:
        mi_scores_u += mutual_info_regression(X[:, i, :], y[:, j, 0])
        mi_scores_v += mutual_info_regression(X[:, i, :], y[:, j, 1])
        mi_scores_dir += mutual_info_regression(X[:, i, :], y_direction[:, j])
    return mi_scores_u, mi_scores_v, mi_scores_dir

def calculate_and_display_mutual_info_scores(X, y, feature_names, sequence_length, prediction_horizon):
    start_time = time.time()
    n_features = X.shape[2]
    
    # Calculate wind direction for the entire prediction horizon
    y_direction = calculate_wind_direction(y[:, :, 0], y[:, :, 1])
    
    # Create chunks of work
    chunks = [(i, j) for i in range(sequence_length) for j in range(prediction_horizon)]
    chunk_size = min(1000, len(chunks) // (multiprocessing.cpu_count() * 2))
    chunks = [chunks[i:i + chunk_size] for i in range(0, len(chunks), chunk_size)]
    
    # Prepare the progress bar
    pbar = tqdm(total=len(chunks), desc="Calculating MI scores")
    
    # Use multiprocessing Pool
    with multiprocessing.Pool(processes=max(1, multiprocessing.cpu_count() - 1)) as pool:
        args_list = [(X, y, y_direction, chunk) for chunk in chunks]
        results = []
        for result in pool.imap_unordered(calculate_mi_for_chunk, args_list):
            results.append(result)
            pbar.update()
    
    pbar.close()
    
    # Aggregate results
    mi_scores_u = np.zeros(n_features)
    mi_scores_v = np.zeros(n_features)
    mi_scores_dir = np.zeros(n_features)
    
    for result in tqdm(results, desc="Aggregating results"):
        mi_scores_u += result[0]
        mi_scores_v += result[1]
        mi_scores_dir += result[2]
        
    total_steps = sequence_length * prediction_horizon
    mi_scores_u /= total_steps
    mi_scores_v /= total_steps
    mi_scores_dir /= total_steps
    
    mi_df = pd.DataFrame({
        'Feature': feature_names,
        'MI Score (u)': mi_scores_u,
        'MI Score (v)': mi_scores_v,
        'MI Score (direction)': mi_scores_dir,
        'MI Score (avg)': (mi_scores_u + mi_scores_v + mi_scores_dir) / 3
    })
    mi_df = mi_df.sort_values('MI Score (avg)', ascending=False)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"\nMutual Information calculation completed in {execution_time:.2f} seconds")
    print("\nMutual Information Scores (sorted by average importance):")
    print(mi_df.to_string(index=False))
    
    plt.figure(figsize=(12, 6))
    plt.bar(mi_df['Feature'], mi_df['MI Score (avg)'])
    plt.xticks(rotation=90)
    plt.title('Average Mutual Information Scores for Each Feature')
    plt.tight_layout()
    plt.savefig('mi_scores_avg.png')
    plt.close()
    
    plt.figure(figsize=(12, 6))
    plt.bar(mi_df['Feature'], mi_df['MI Score (u)'], alpha=0.3, label='u component')
    plt.bar(mi_df['Feature'], mi_df['MI Score (v)'], alpha=0.3, label='v component')
    plt.bar(mi_df['Feature'], mi_df['MI Score (direction)'], alpha=0.3, label='direction')
    plt.xticks(rotation=90)
    plt.title('Mutual Information Scores for Each Feature (u, v, and direction)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('mi_scores_uvdir.png')
    plt.close()

###########################################################################################
# INFO: MIC Scores from minepy package (only available for Python <3.10)
###########################################################################################
# def calculate_and_display_mic_scores(X, y, feature_names):
    
#     mine = MINE(alpha=0.6, c=15)  # Default parameters
#     mic_scores = []
    
#     # Combine u and v components into a single target
#     y_combined = np.sqrt(y[:, 0]**2 + y[:, 1]**2)  # Calculate magnitude
    
#     for i in range(X.shape[1]):
#         mine.compute_score(X[:, i], y_combined)
#         mic_scores.append(mine.mic())
    
#     # Create a DataFrame with feature names and MIC scores
#     mic_df = pd.DataFrame({'Feature': feature_names, 'MIC Score': mic_scores})
#     mic_df = mic_df.sort_values('MIC Score', ascending=False)
    
#     print("\nMIC Scores (sorted by importance):")
#     print(mic_df.to_string(index=False))
    
#     # Optionally, you can create a bar plot of MIC scores
#     import matplotlib.pyplot as plt
#     plt.figure(figsize=(12, 6))
#     plt.bar(mic_df['Feature'], mic_df['MIC Score'])
#     plt.xticks(rotation=90)
#     plt.title('MIC Scores for Each Feature')
#     plt.tight_layout()
#     plt.savefig('mic_scores.png')
#     plt.close()
###########################################################################################

def process_wind_vectors(df: pd.DataFrame, wind_columns: List[str]):
    """
    Convert wind speeds and directions to u, v components.
    Args:
        df (pd.DataFrame): DataFrame containing the wind data.
        wind_columns (List[str]): List of column name pairs (speed, direction) to be converted.
    Returns:
        DataFrame with u, v components of the wind vectors.
    """
    for speed_col, dir_col in wind_columns:
        speed = df[speed_col]
        direction = np.deg2rad(270 - df[dir_col])  # Convert to radians, adjust for North 0 degrees convention
        df[f'{speed_col}_u'] = speed * np.cos(direction)
        df[f'{speed_col}_v'] = speed * np.sin(direction)
    return df.drop(columns=[col for pair in wind_columns for col in pair])

def create_sequences(df: pd.DataFrame, sequence_length: int):
    """
    Create sequences of data for training.
    Args:
        df (pd.DataFrame): DataFrame containing the input data.
        sequence_length (int): Length of the sequences for the sliding window used for training.
    Returns:
        np.ndarray: Array of sequences.
    """
    data = df.values
    X = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
    return np.array(X)

def load_and_preprocess_data(file_path: str, sequence_length, prediction_horizon, target_turbine):
    """
    Load data from a CSV file and preprocess it for training.
    Args:
        file_path (str): Path to the CSV file containing the data.
        sequence_length (int): Length of the sequences for the sliding window used for training.
    Returns:
        Tuple of features (X) and target (y).
    """
    df = pd.read_csv(file_path)
    
    relevant_columns = ['Time', 'FreestreamWindMag', 'FreestreamWindDir', 
                        'TurbineWindMag_0', 'TurbineWindDir_0',
                        'TurbineWindMag_1', 'TurbineWindDir_1',
                        'TurbineWindMag_2', 'TurbineWindDir_2']
    df = df[relevant_columns]
    
    # Convert Time to float64 for accurate division
    df['Time'] = df['Time'].astype(np.float64)
    
    # Create time features (Time column in seconds)
    df['hour'] = (df['Time'] % SECONDS_PER_DAY) / SECONDS_PER_HOUR
    df['day'] = ((df['Time'] // SECONDS_PER_DAY) % 365).astype(int)
    df['year'] = (df['Time'] // SECONDS_PER_YEAR).astype(int)

    # Normalize time features using sin/cos for capturing cyclic patterns
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365)
    df['year_sin'] = np.sin(2 * np.pi * df['year'])
    df['year_cos'] = np.cos(2 * np.pi * df['year'])
    
    # Convert wind speeds and directions to u, v components
    wind_columns = [('FreestreamWindMag', 'FreestreamWindDir'),
                    ('TurbineWindMag_0', 'TurbineWindDir_0'),
                    ('TurbineWindMag_1', 'TurbineWindDir_1'),
                    ('TurbineWindMag_2', 'TurbineWindDir_2')]
    df = process_wind_vectors(df, wind_columns)
    
    # Normalize non-time features
    features_to_normalize = [col for col in df.columns if col not in ['Time', 'hour', 'day', 'year']]
    df[features_to_normalize] = MinMaxScaler().fit_transform(df[features_to_normalize])
    
    # Drop original time columns
    df = df.drop(columns=['Time', 'hour', 'day', 'year'])
    
    # INFO: Create sequences for training, drop the target turbines to avoid data leakage
    X = create_sequences(df.drop(columns=[f'TurbineWindMag_{target_turbine}_u', f'TurbineWindMag_{target_turbine}_v']), sequence_length)
    y = create_sequences(df[[f'TurbineWindMag_{target_turbine}_u', f'TurbineWindMag_{target_turbine}_v']], prediction_horizon)
    
    # Align X and y
    X = X[:-prediction_horizon]
    y = y[sequence_length:]
    
    # # INFO: Calculate MIC scores
    # X_flattened = X.reshape(-1, X.shape[2])  # Flatten the sequences
    # y_flattened = y  # Keep y as 2D array
    
    # # Ensure X_flattened and y_flattened have the same number of samples
    # min_samples = min(X_flattened.shape[0], y_flattened.shape[0])
    # X_flattened = X_flattened[:min_samples]
    # y_flattened = y_flattened[:min_samples]
    
    feature_names = list(df.drop(columns=[f'TurbineWindMag_{target_turbine}_u', f'TurbineWindMag_{target_turbine}_v']).columns)
    
    return X, y, feature_names # X [num_samples, sequence_length, num_features], y [num_samples, num_features(u, v)]

def main():
    ###########################################################################################
    file_path = 'lut/time_series_results_case_LUT_seed_0.csv'
    sequence_length = 600 # 600 time steps of 0.5 seconds = 5 minutes
    prediction_horizon = 240 # 240 time steps of 0.5 seconds = 2 minutes
    target_turbine = 0 # Choose the turbine to predict
    ###########################################################################################
    
    X, y, feature_names = load_and_preprocess_data(file_path, sequence_length=sequence_length, prediction_horizon=prediction_horizon, target_turbine=target_turbine)
    
    print("\n" + "="*50)
    print("Wind Forecasting Data Preprocessing Summary")
    print("="*50)
    
    print(f"\nInput file: {file_path}")
    print(f"Sequence length: {sequence_length} time steps")
    print(f"Prediction horizon: {prediction_horizon} time steps")
    print(f"Target turbine: {target_turbine}")
    print("\nFeature matrix (X):")
    print(f"  Shape: {X.shape}")
    print(f"  Number of sequences: {X.shape[0]}")
    print(f"  Time steps per sequence: {X.shape[1]}")
    print(f"  Number of features: {X.shape[2]}")
    
    processed_df = pd.DataFrame(X[0, :5], columns=feature_names)
    print("\nProcessed Data Sample (first sequence, first 5 time steps):")
    print(processed_df.to_string(index=False, float_format='{:.8f}'.format))
    
    print("\nTarget matrix (y):")
    print(f"  Shape: {y.shape}")
    print(f"  Number of target vectors: {y.shape[0]}")
    print(f"  Components per vector: {y.shape[1]} (u, v)")
    
    # Display target data sample showing first time step of first 5 sequences
    target_df = pd.DataFrame(y[:5, 0, :], columns=['u', 'v'])
    print("\nTarget Sample (first time step of first 5 sequences):")
    print(target_df.to_string(index=False, float_format='{:.8f}'.format))
    
    print("\nFeatures list:")
    for i, feature in enumerate(feature_names):
        print(f"  {i+1}. {feature}")
    print("\n")
    
    ###########################################################################################    
    # INFO: Calculate and display MIC scores
    # X_flattened = X.reshape(-1, X.shape[2])  # Flatten the sequences
    # y_flattened = y  # Keep y as 2D array
    
    # Ensure X_flattened and y_flattened have the same number of samples
    # min_samples = min(X_flattened.shape[0], y_flattened.shape[0])
    # X_flattened = X_flattened[:min_samples]
    # y_flattened = y_flattened[:min_samples]
    
    # calculate_and_display_mic_scores(X_flattened, y_flattened, feature_names)    
    ###########################################################################################
    
    calculate_and_display_mutual_info_scores(X, y, feature_names, sequence_length, prediction_horizon)

if __name__ == "__main__":
    main()