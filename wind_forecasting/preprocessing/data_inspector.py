### This file contains class and methods to: 
### - compute statistical summary of the data, 
### - plot the data distributions: power curve distribution, v vs u distribution, yaw angle distribution

import seaborn as sns
import matplotlib.pyplot as plt
# from matplotlib.patches import Arrow
from windrose import WindroseAxes
import numpy as np
from floris import FlorisModel
from floris.flow_visualization import visualize_cut_plane
import floris.layout_visualization as layoutviz
import scipy.stats as stats
import polars as pl
#INFO: @Juan 10/02/24
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor
from sklearn.feature_selection import mutual_info_regression
from tqdm.auto import tqdm
import time
import logging
import os

#INFO: TO use MPI, need to run the script with the following command:
# mpiexec -n <number_of_processes> python your_script.py

class DataInspector:
    """_summary_
    - compute statistical summary of the data,
    - plot the data distributions: 
    -   power curve distribution, 
    -   v vs u distribution, 
    -   yaw angle distribution 
    """
    #INFO: @Juan 10/02/24 Added extra parameters to constructor
    def __init__(self, df: pl.DataFrame, X: np.ndarray, y: np.ndarray, 
                 feature_names: list[str], sequence_length: int, prediction_horizon: int,
                 turbine_input_filepath: str, farm_input_filepath: str):
        self._validate_input_data(df, X, y, feature_names, sequence_length, prediction_horizon, turbine_input_filepath, farm_input_filepath)
        self.df = df
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.turbine_input_filepath = turbine_input_filepath
        self.farm_input_filepath = farm_input_filepath

    #INFO: @Juan 10/02/24 Added method to validate input data
    def _validate_input_data(self, df, X, y, feature_names, sequence_length, prediction_horizon,
                             turbine_input_filepath, farm_input_filepath):
        if not isinstance(df, pl.DataFrame):
            raise TypeError("df must be a polars DataFrame")
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X and y must be numpy arrays")
        if not isinstance(feature_names, list) or not all(isinstance(f, str) for f in feature_names):
            raise TypeError("feature_names must be a list of strings")
        if not isinstance(turbine_input_filepath, str) or not isinstance(farm_input_filepath, str):
            raise TypeError("turbine_input_filepath and farm_input_filepath must be strings")
        if X.shape[2] != len(feature_names):
            raise ValueError("Number of features in X does not match length of feature_names")
        if y.shape[1] != prediction_horizon:
            raise ValueError("Second dimension of y does not match prediction_horizon")
        if not os.path.exists(turbine_input_filepath):
            raise FileNotFoundError(f"Turbine input file not found: {turbine_input_filepath}")
        if not os.path.exists(farm_input_filepath):
            raise FileNotFoundError(f"Farm input file not found: {farm_input_filepath}")
        
    def plot_time_series(self, turbine_ids: list[str]) -> None:
        if isinstance(turbine_ids, str):
            turbine_ids = [turbine_ids]  # Convert single ID to list
        
        available_turbines = self.df['turbine_id'].unique()
        valid_turbines = [tid for tid in turbine_ids if tid in available_turbines]
        
        if not valid_turbines:
            raise ValueError(f"No valid turbine IDs. Available turbine IDs: {available_turbines}")
        
        sns.set_style("whitegrid")
        sns.set_palette("deep")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        for turbine_id in valid_turbines:
            # turbine_data = self.df.loc[turbine_id]
            # BUG: Should be using .to_pandas() here???
            turbine_data = self.df.filter(pl.col("turbine_id") == turbine_id).drop_nulls().to_pandas()
            # plt.plot(turbine_data["time"], turbine_data["wind_speed"])
            sns.lineplot(data=turbine_data, x='time', y='wind_speed', ax=ax1, label=f'{turbine_id} Wind Speed')
            sns.lineplot(data=turbine_data, x='time', y='power_output', ax=ax2, label=f'{turbine_id} Power Output')
        
        ax1.set_ylabel('Wind Speed (m/s)')
        ax2.set_ylabel('Power Output (kW)')
        ax2.set_xlabel('Time')
        
        ax1.set_title('Wind Speed Over Time', fontsize=14)
        ax2.set_title('Power Output Over Time', fontsize=14)
        
        fig.suptitle(f'Wind Speed and Power Output for Turbines: {", ".join(valid_turbines)}', fontsize=16)
        
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        plt.show()

    def plot_scatter_wind_speed_power(self, turbine_ids: list[str]) -> None: #NOTE: @Juan 10/02/24 Improved plotting of scatter plot
        """_summary_

        """
        if isinstance(turbine_ids, str):
            turbine_ids = [turbine_ids]  # Convert single ID to list
        
        available_turbines = self.df['turbine_id'].unique()
        valid_turbines = [tid for tid in turbine_ids if tid in available_turbines]
        
        if not valid_turbines:
            logging.error(f"No valid turbine IDs. Available turbine IDs: {available_turbines}")
            return
        
        # _, ax = plt.subplots(1, 1, figsize=(12, 6))
        plt.figure(figsize=(12, 6))

        for turbine_id in valid_turbines:
            # turbine_data = self.df.loc[turbine_id]
            turbine_data = self.df.filter(pl.col("turbine_id") == turbine_id).drop_nulls().to_pandas()
            sns.scatterplot(data=turbine_data, x='wind_speed', y='power_output', label=turbine_id, alpha=0.5)

        plt.xlabel('Wind Speed [m/s]')
        plt.ylabel('Power Output [kW]')
        plt.title('Scatter Plot of Wind Speed vs Power Output')
        plt.legend(title='Turbine ID', loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True, alpha=0.3)
        sns.despine()
        plt.tight_layout()
        plt.show()

    def plot_wind_rose(self) -> None:
        """_summary_

        Args:
            wind_direction (float): _description_
            wind_speed (float): _description_
        """
        plt.figure(figsize=(10, 10))
        ax = WindroseAxes.from_ax()
        wind_direction = self.df["wind_direction"]
        wind_speed = self.df["wind_speed"]
        ax.bar(wind_direction, wind_speed, normed=True, opening=0.8, edgecolor='white')
        ax.set_legend()
        plt.title('Wind Rose')
        plt.show()

    def plot_temperature_distribution(self) -> None:
        """_summary_

        """
        temp_columns = [
            'generator_bearing_de_temp',
            'generator_bearing_nde_temp',
            'generator_inlet_temp',
            'generator_stator_temp_1',
            'generator_stator_temp_2',
            'nacelle_temperature',
            'ambient_temperature'
        ]
        
        _, ax = plt.subplots(1, 1, figsize=(15, 10))
        for col in temp_columns:
            sns.histplot(self.df[col], bins=20, kde=True, label=col)
        
        ax.set(title='Temperature Distributions', xlabel="Temperature", ylabel="Frequency")
        plt.legend()
        plt.show()

    def plot_heatmap_correlation(self) -> None: #NOTE: @Juan 10/02/24 Improved plotting of heatmap
        """_summary_
        """
        corr_matrix = self.df.select(self.feature_names).corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.show()

    def plot_boxplot_wind_speed(self, turbine_id: str) -> None:
        """_summary_

        Args:
            turbine_id (str): _description_
        """
        # Select data for the specified turbine
        # turbine_data = self.df.loc[turbine_id]
        turbine_data = self.df.filter(pl.col("turbine_id") == turbine_id)
        
        # Extract hour from the time index
        # turbine_data = turbine_data.reset_index()
        turbine_data['hour'] = turbine_data['time'].dt.hour
        
        _, ax = plt.subplots(1, 1, figsize=(12, 6))
        sns.boxplot(data=turbine_data, x='hour', y='wind_speed', ax=ax)
        ax.set(title=f'Wind Speed Distribution by Hour for Turbine {turbine_id}', xlabel="Hour of Day", ylabel="Wind Speed [m/s]")
        plt.show() 

    def plot_wind_speed_weibull(self) -> None: #NOTE: @Juan 10/02/24
        """_summary_
            Plot the weibull distribution of the wind speed data
        """
        # Extract wind speed data
        wind_speeds = self.df['wind_speed'].drop_nulls().to_numpy() #NOTE: @Juan 10/02/24 Changed to numpy array

        # Fit Weibull distribution
        shape, loc, scale = stats.weibull_min.fit(wind_speeds, floc=0)

        # Create a range of wind speeds for the fitted distribution
        x = np.linspace(0, wind_speeds.max(), 100)
        y = stats.weibull_min.pdf(x, shape, loc, scale)

        # Plot
        plt.figure(figsize=(12, 6))
        sns.histplot(wind_speeds, stat='density', kde=True, color='skyblue', label='Observed')
        plt.plot(x, y, 'r-', lw=2, label=f'Weibull (k={shape:.2f}, λ={scale:.2f})')
        
        plt.title('Wind Speed Distribution with Fitted Weibull', fontsize=16)
        plt.xlabel('Wind Speed (m/s)', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        sns.despine()
        plt.show()

        print(f"Weibull shape parameter (k): {shape:.2f}")
        print(f"Weibull scale parameter (λ): {scale:.2f}")

    def plot_king_plains_wind_farm(self) -> None:
        """_summary_

        Returns:
            _type_: _description_
        """
        # Ensure the paths are absolute
        
        # Initialize the FLORIS model
        try:
            fmodel = FlorisModel(self.farm_input_filepath)
        except FileNotFoundError:
            raise FileNotFoundError(f"Farm input file not found: {self.farm_input_filepath}")
        
        # Load the turbine data
        try:
            fmodel.set_turbine_type(self.turbine_input_filepath)
        except FileNotFoundError:
            logging.warning(f"Turbine file not found: {self.turbine_input_filepath}. Using default turbine type.")
        
        # Set initial wind conditions
        fmodel.set(wind_directions=[270.0], wind_speeds=[8.0], turbulence_intensities=[0.08])
        
        # Create the plot
        _, ax = plt.subplots(figsize=(15, 15))
        
        # Plot the turbine layout
        layoutviz.plot_turbine_layout(fmodel, ax=ax)
        
        # Add turbine labels
        turbine_names = [f"T{i+1}" for i in range(fmodel.n_turbines)]
        layoutviz.plot_turbine_labels(
            fmodel, ax=ax, turbine_names=turbine_names, show_bbox=True, bbox_dict={"facecolor": "white", "alpha": 0.5}
        )
        
        # Calculate and visualize the flow field
        horizontal_plane = fmodel.calculate_horizontal_plane(height=90.0)
        visualize_cut_plane(horizontal_plane, ax=ax, min_speed=4, max_speed=10, color_bar=True)
        
        # Plot turbine rotors
        layoutviz.plot_turbine_rotors(fmodel, ax=ax)
        
        # Set plot title and labels
        plt.title('King Plains Wind Farm Layout', fontsize=16)
        plt.xlabel('X coordinate (m)', fontsize=12)
        plt.ylabel('Y coordinate (m)', fontsize=12)
        
        # Adjust layout and display the plot
        plt.tight_layout()
        plt.show()
        
        return fmodel

    #INFO: @Juan 10/02/24 Added method to calculate wind direction
    @staticmethod
    def calculate_wind_direction(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.mod(180 + np.rad2deg(np.arctan2(u, v)), 360)
    
    #INFO: @Juan 10/02/24 Added method to calculate mutual information for chunks of data
    @staticmethod
    def calculate_mi_for_chunk(args: tuple) -> tuple:
        X, y, y_direction, chunk = args
        chunk_size = len(chunk)
        mi_scores_u = np.zeros(X.shape[2])
        mi_scores_v = np.zeros(X.shape[2])
        mi_scores_dir = np.zeros(X.shape[2])
        
        # Preallocate arrays for chunks
        X_chunk = np.empty((X.shape[0], chunk_size, X.shape[2]))
        y_u_chunk = np.empty((y.shape[0], chunk_size))
        y_v_chunk = np.empty((y.shape[0], chunk_size))
        y_dir_chunk = np.empty((y.shape[0], chunk_size))
        
        for idx, (i, j) in enumerate(chunk):
            X_chunk[:, idx, :] = X[:, i, :]
            y_u_chunk[:, idx] = y[:, j, 0]
            y_v_chunk[:, idx] = y[:, j, 1]
            y_dir_chunk[:, idx] = y_direction[:, j]
            
        # Flatten the chunks for mutual_info_regression
        X_chunk_flat = X_chunk.reshape(-1, X.shape[2])
        y_u_chunk_flat = y_u_chunk.flatten()
        y_v_chunk_flat = y_v_chunk.flatten()
        y_dir_chunk_flat = y_dir_chunk.flatten()
        
        # Calculate mutual information
        mi_scores_u += np.sum(mutual_info_regression(X_chunk_flat, y_u_chunk_flat).reshape(chunk_size, -1), axis=0)
        mi_scores_v += np.sum(mutual_info_regression(X_chunk_flat, y_v_chunk_flat).reshape(chunk_size, -1), axis=0)
        mi_scores_dir += np.sum(mutual_info_regression(X_chunk_flat, y_dir_chunk_flat).reshape(chunk_size, -1), axis=0)
        
        return mi_scores_u, mi_scores_v, mi_scores_dir

    #INFO: @Juan 10/02/24 Added method to calculate MI scores
    def calculate_and_display_mutual_info_scores(self, X: np.ndarray, y: np.ndarray, feature_names: list[str], sequence_length: int, prediction_horizon: int) -> None:
        start_time = time.time()
        
        # Calculate wind direction for the entire prediction horizon
        y_direction = self.calculate_wind_direction(y[:, :, 0], y[:, :, 1])
        
        # Create chunks of work
        # BUG: @Juan Make sure that numpy array works with this, otherwise revert to list of tuples
        chunks = np.array([(i, j) for i in range(sequence_length) for j in range(prediction_horizon)])
        chunk_size = min(1000, len(chunks) // (MPI.COMM_WORLD.Get_size() * 2)) #NOTE: @Juan 10/02/24 Added MPI
        chunks = [chunks[i:i + chunk_size] for i in range(0, len(chunks), chunk_size)]
        
        # INFO: @Juan 10/02/24 Use MPI for parallel processing
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        # Use multiprocessing Pool with tqdm progress bar
        with MPICommExecutor(comm, root=0) as executor:
            if executor is not None:  # This is true for the root process
                args_list = [(X, y, y_direction, chunk) for chunk in chunks]
                results = list(tqdm(executor.map(self.calculate_mi_for_chunk, args_list), 
                                    total=len(chunks), desc="Calculating MI scores"))
                
                # Aggregate results
                mi_scores_u = np.sum([result[0] for result in results], axis=0)
                mi_scores_v = np.sum([result[1] for result in results], axis=0)
                mi_scores_dir = np.sum([result[2] for result in results], axis=0)
                
                total_steps = sequence_length * prediction_horizon
                mi_scores_u /= total_steps
                mi_scores_v /= total_steps
                mi_scores_dir /= total_steps
                
                mi_df = pl.DataFrame({
                    'Feature': feature_names,
                    'MI Score (u)': mi_scores_u,
                    'MI Score (v)': mi_scores_v,
                    'MI Score (direction)': mi_scores_dir,
                    'MI Score (avg)': (mi_scores_u + mi_scores_v + mi_scores_dir) / 3
                }).sort('MI Score (avg)', descending=True)
                
                logging.info(f"\nMutual Information calculation completed in {time.time() - start_time:.2f} seconds")
                logging.info("\nMutual Information Scores (sorted by average importance):")
                logging.info(mi_df)
                
                self._plot_mi_scores(mi_df)
            else:
                # Non-root processes will enter here and participate in the computation
                pass #NOTE: @Juan 10/02/24 Added separated static method to plot MI scores
    
    @staticmethod
    def _plot_mi_scores(mi_df: pl.DataFrame) -> None:
        """Plot mutual information scores."""
        plt.figure(figsize=(12, 6))
        plt.bar(mi_df['Feature'], mi_df['MI Score (avg)'])
        plt.xticks(rotation=90)
        plt.title('Average Mutual Information Scores for Each Feature')
        plt.tight_layout()
        plt.savefig('./preprocessing/outputs/mi_scores_avg.png')
        plt.close()
        
        plt.figure(figsize=(12, 6))
        plt.bar(mi_df['Feature'], mi_df['MI Score (u)'], alpha=0.3, label='u component')
        plt.bar(mi_df['Feature'], mi_df['MI Score (v)'], alpha=0.3, label='v component')
        plt.bar(mi_df['Feature'], mi_df['MI Score (direction)'], alpha=0.3, label='direction')
        plt.xticks(rotation=90)
        plt.title('Mutual Information Scores for Each Feature (u, v, and direction)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('./preprocessing/outputs/mi_scores_uvdir.png')
        plt.close()

    #INFO: @Juan 10/02/24 Added method to calculate and display mutual information scores for the target turbine
    #NOTE: Future work: Accept more than one turbine ID as input, Accept feature_names as input
    def calculate_mi_scores(self, target_turbine: str) -> None:
        # Remove the target turbine data in Y from the feature set X
        # 1. Create bool mask to filter out (~) data of target turbine. This works for both u and v components
        feature_mask = ~np.char.startswith(self.feature_names, f'TurbineWindMag_{target_turbine}_')
        
        # 2. Apply the mask to filter X and feature_names
        X_filtered = self.X[:, :, feature_mask]
        feature_names_filtered = np.array(self.feature_names)[feature_mask]
        
        # 3. Calculate and display mutual information scores
        logging.info(f"Calculating Mutual Information scores for target turbine: {target_turbine}")
        self.calculate_and_display_mutual_info_scores(X_filtered, self.y, feature_names_filtered, self.sequence_length, self.prediction_horizon)
    