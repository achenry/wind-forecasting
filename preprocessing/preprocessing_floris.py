# NOTE: FilteredFreestreamWindDir is not being used. It's a low-pass filtered version of FreestreamWindDir. Shall I use it?
# It could lose high frequency components, including turbulence. But it's cleaner.
# So it could lose short-term variations, and induce some lag in the data.
# Overfitting risk?
# INFO: MIC used for feature selection from minepy package. Only available for Python <3.10
# Can use sklearn.feature_selection.mutual_info_regression instead?

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple
from minepy import MINE # Maximal Information-based Nonparametric Exploration.

SECONDS_PER_HOUR = np.float64(3600)
SECONDS_PER_DAY = 86400
SECONDS_PER_YEAR = 31536000  # non-leap year, 365 days

def calculate_mic_scores(X, y):
    """
    Calculate MIC scores between each feature in X and the target y.
    
    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features)
        y (np.ndarray): Target vector of shape (n_samples, 2) for u and v components
    
    Returns:
        dict: Dictionary of feature names and their MIC scores
    """
    mine = MINE(alpha=0.6, c=15)  # Default parameters
    mic_scores = {}
    
    # Combine u and v components into a single target
    y_combined = np.sqrt(y[:, 0]**2 + y[:, 1]**2)  # Calculate magnitude
    
    for i in range(X.shape[1]):
        mine.compute_score(X[:, i], y_combined)
        mic_scores[f'feature_{i}'] = mine.mic()
    
    return mic_scores

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

def load_and_preprocess_data(file_path: str, sequence_length: int = 24):
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
    
    # print('Hour: ', df['hour'][0:5])
    # print('Day: ', df['day'][0:5])
    # print('Year: ', df['year'][0:5])

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
    
    X = create_sequences(df, sequence_length)
    y = df[['FreestreamWindMag_u', 'FreestreamWindMag_v']].values[sequence_length:]
    
    # Calculate MIC scores
    X_flattened = X.reshape(-1, X.shape[2])  # Flatten the sequences
    y_flattened = y  # Keep y as 2D array
    
    # Ensure X_flattened and y_flattened have the same number of samples
    min_samples = min(X_flattened.shape[0], y_flattened.shape[0])
    X_flattened = X_flattened[:min_samples]
    y_flattened = y_flattened[:min_samples]
    
    mic_scores = calculate_mic_scores(X_flattened, y_flattened)
    
    return X, y, mic_scores # X [num_samples, sequence_length, num_features], y [num_samples, num_features(u, v)]

def main():
    file_path = 'lut/time_series_results_case_LUT_seed_0.csv'
    
    X, y, mic_scores = load_and_preprocess_data(file_path, sequence_length=1200) # 1200 time steps of 0.5 seconds = 5 minutes
    
    print("\n" + "="*50)
    print("Wind Forecasting Data Preprocessing Summary")
    print("="*50)
    
    print(f"\nInput file: {file_path}")
    print(f"Sequence length: 24 time steps")
    
    print("\nFeature matrix (X):")
    print(f"  Shape: {X.shape}")
    print(f"  Number of sequences: {X.shape[0]}")
    print(f"  Time steps per sequence: {X.shape[1]}")
    print(f"  Number of features: {X.shape[2]}")
    
    feature_names = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'year_sin', 'year_cos',
                 'FreestreamWindMag_u', 'FreestreamWindMag_v',
                 'TurbineWindMag_0_u', 'TurbineWindMag_0_v',
                 'TurbineWindMag_1_u', 'TurbineWindMag_1_v',
                 'TurbineWindMag_2_u', 'TurbineWindMag_2_v']
    
    processed_df = pd.DataFrame(X[0, :5], columns=feature_names)
    print("\nProcessed Data Sample (first sequence, first 5 time steps):")
    print(processed_df.to_string(index=False, float_format='{:.8f}'.format))
    
    print("\nTarget matrix (y):")
    print(f"  Shape: {y.shape}")
    print(f"  Number of target vectors: {y.shape[0]}")
    print(f"  Components per vector: {y.shape[1]} (u, v)")
    
    target_df = pd.DataFrame(y[:5], columns=['u', 'v'])
    print("\nTarget Sample (first 5 rows):")
    print(target_df.to_string(index=False, float_format='{:.8f}'.format))
    
    print("\nFeatures list:")
    for i, feature in enumerate(feature_names):
        print(f"  {i+1}. {feature}")
        
    print("\nMIC Scores:")
    for feature, score in mic_scores.items():
        print(f"  {feature}: {score:.4f}")
        
    # Sort features by MIC score in descending order
    sorted_mic_scores = sorted(mic_scores.items(), key=lambda x: x[1], reverse=True)
    print("\nFeatures sorted by MIC score:")
    for feature, score in sorted_mic_scores:
        print(f"  {feature}: {score:.4f}")
    
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()