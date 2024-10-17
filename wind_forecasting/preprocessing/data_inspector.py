### This file contains class and methods to: 
### - compute statistical summary of the data, 
### - plot the data distributions: power curve distribution, v vs u distribution, yaw angle distribution
import os
import time
import logging

import seaborn as sns
import matplotlib.pyplot as plt
from windrose import WindroseAxes
import numpy as np
from floris import FlorisModel
from floris.flow_visualization import visualize_cut_plane
import floris.layout_visualization as layoutviz
import scipy.stats as stats
import polars as pl
import polars.selectors as cs
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor
from sklearn.feature_selection import mutual_info_regression
from tqdm.auto import tqdm
import re

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
    
    def __init__(self, turbine_input_filepath: str, farm_input_filepath: str):
        self._validate_input_data(turbine_input_filepath=turbine_input_filepath, farm_input_filepath=farm_input_filepath)
        self.turbine_input_filepath = turbine_input_filepath
        self.farm_input_filepath = farm_input_filepath

    def _get_valid_turbine_ids(self, df, turbine_ids: list[str]) -> list[str]:
        if isinstance(turbine_ids, str):
            turbine_ids = [turbine_ids]  # Convert single ID to list
        
        # valid_turbines = df.select("turbine_id").unique().filter(pl.col("turbine_id").is_in(turbine_ids)).collect(streaming=True).to_numpy()[:, 0]
        cols = df.collect_schema().names()
        available_turbines = np.unique([re.findall(f"(?<=wind_direction_)(.*)", col)[0] for col in cols if "wind_direction" in col])
        valid_turbines = [tid for tid in available_turbines if tid in turbine_ids]

        if len(valid_turbines) == 0:
            print(f"Error: No valid turbine IDs")
            print("Available turbine IDs:", available_turbines)
            return []
        
        return valid_turbines
    
    def _validate_input_data(self, *, X=None, y=None, features=None, sequence_length=None, prediction_horizon=None,
                             turbine_input_filepath=None, farm_input_filepath=None):
        if (X is not None and not isinstance(X, np.ndarray)) or (y is not None and not isinstance(y, np.ndarray)):
            raise TypeError("X and y must be numpy arrays")
        if (features is not None) and (not isinstance(features, list) or not all(isinstance(f, str) for f in features)):
            raise TypeError("feature_names must be a list of strings")
        if (turbine_input_filepath is not None and not isinstance(turbine_input_filepath, str)) or (farm_input_filepath is not None and not isinstance(farm_input_filepath, str)):
            raise TypeError("turbine_input_filepath and farm_input_filepath must be strings")
        if X is not None and features is not None and X.shape[2] != len(features):
            raise ValueError("Number of features in X does not match length of feature_names")
        if y is not None and prediction_horizon is not None and y.shape[1] != prediction_horizon:
            raise ValueError("Second dimension of y does not match prediction_horizon")
        if turbine_input_filepath is not None and not os.path.exists(turbine_input_filepath):
            raise FileNotFoundError(f"Turbine input file not found: {turbine_input_filepath}")
        if farm_input_filepath is not None and not os.path.exists(farm_input_filepath):
            raise FileNotFoundError(f"Farm input file not found: {farm_input_filepath}")

    def plot_time_series(self, df, turbine_ids: list[str]) -> None:
        if isinstance(turbine_ids, str):
            turbine_ids = [turbine_ids]  # Convert single ID to list
        
        valid_turbines = self._get_valid_turbine_ids(df, turbine_ids=turbine_ids)
        
        if len(valid_turbines) == 0:
            return
        
        sns.set_style("whitegrid")
        sns.set_palette("deep")

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        for turbine_id in valid_turbines:
            turbine_data = df.select([pl.col("time"), cs.ends_with(f"{turbine_id}")]).drop_nulls()
            # plt.plot(turbine_data["time"], turbine_data["wind_speed"])
            sns.lineplot(data=turbine_data.collect(streaming=True).to_pandas(),
                         x='time', y=f'wind_speed_{turbine_id}', ax=ax1, label=f'{turbine_id} Wind Speed')
            sns.lineplot(data=turbine_data.collect(streaming=True).to_pandas(),
                         x='time', y=f'wind_direction_{turbine_id}', ax=ax2, label=f'{turbine_id} Wind Direction')
            sns.lineplot(data=turbine_data.collect(streaming=True).to_pandas(),
                         x='time', y=f'power_output_{turbine_id}', ax=ax3, label=f'{turbine_id} Power Output')
        
        ax1.set_ylabel('Wind Speed (m/s)')
        ax2.set_ylabel('Wind Direction (kW)')
        ax3.set_ylabel('Power Output (kW)')
        ax2.set_xlabel('Time')
        
        ax1.set_title('Wind Speed vs. Time', fontsize=14)
        ax2.set_title('Wind Direction vs. Time', fontsize=14)
        ax2.set_title('Power Output vs. Time', fontsize=14)
        
        fig.suptitle(f'Wind Speed, Wind Direction, and Power Output for Turbines: {", ".join(valid_turbines)}', fontsize=16)
        
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax3.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        plt.show()

    def plot_wind_speed_power(self, df, turbine_ids: list[str]) -> None:
        """_summary_

        """
        valid_turbines = self._get_valid_turbine_ids(df, turbine_ids=turbine_ids)
        
        if len(valid_turbines) == 0:
             return
        
        for turbine_id in valid_turbines:
            _, ax = plt.subplots(1, 1, figsize=(12, 6))
            # TODO does seaborn plot null/nan values??
            turbine_data = df.select(pl.col(f"wind_speed_{turbine_id}"), pl.col(f"power_output_{turbine_id}"))\
                .filter(pl.all_horizontal(pl.col(f"wind_speed_{turbine_id}", f"power_output_{turbine_id}").is_not_null()))\
                            .collect(streaming=True).to_pandas()
            sns.scatterplot(data=turbine_data, ax=ax, x=f"wind_speed_{turbine_id}", y=f"power_output_{turbine_id}", label=turbine_id, alpha=0.5)

            plt.xlabel('Wind Speed [m/s]')
            plt.ylabel('Power Output [kW]')
            plt.title('Scatter Plot of Wind Speed vs Power Output')
            plt.legend(title='Turbine ID', loc='upper left', bbox_to_anchor=(1, 1))
            plt.grid(True, alpha=0.3)
            sns.despine()
            plt.tight_layout()
            plt.show()

    def plot_wind_rose(self, df, turbine_ids: list[str] | str) -> None:
        """_summary_

        Args:
            wind_direction (float): _description_
            wind_speed (float): _description_
        """
        if turbine_ids == "all":
            plt.figure(figsize=(10, 10))
            ax = WindroseAxes.from_ax()
            ax.bar(df.select(pl.col("wind_direction*")).collect(streaming=True).to_numpy()[:, 0], 
                   df.select(pl.col("wind_speed*")).collect(streaming=True).to_numpy()[:, 0], 
                   normed=True, opening=0.8, edgecolor='white')
            ax.set_legend()
            plt.title('Wind Rose for all Turbines')
            plt.show()
        else:
            valid_turbines = self._get_valid_turbine_ids(df, turbine_ids=turbine_ids)
        
            if len(valid_turbines) == 0:
                return

            for turbine_id in valid_turbines:
                turbine_data = df.select([pl.col(f"wind_speed_{turbine_id}"), pl.col(f"wind_direction_{turbine_id}")])\
                    .filter(pl.all_horizontal(pl.col(f"wind_speed_{turbine_id}").is_not_null(), pl.col(f"wind_direction_{turbine_id}").is_not_null()))
                plt.figure(figsize=(10, 10))
                ax = WindroseAxes.from_ax()
                ax.bar(turbine_data.select(f"wind_direction_{turbine_id}").collect(streaming=True).to_numpy()[:, 0], 
                       turbine_data.select(f"wind_speed_{turbine_id}").collect(streaming=True).to_numpy()[:, 0], normed=True, opening=0.8, edgecolor='white')
                ax.set_legend()
                plt.title(f'Wind Rose for Turbine {turbine_id}')
                plt.show()


    def plot_temperature_distribution(self, df) -> None:
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
            sns.histplot(df[col], bins=20, kde=True, label=col)
        
        ax.set(title='Temperature Distributions', xlabel="Temperature", ylabel="Frequency")
        plt.legend()
        plt.show()

    def plot_correlation(self, df, features) -> None:
        """_summary_
        """
        _, ax = plt.subplots(1, 1, figsize=(12, 10))
        sns.heatmap(df.select(features).collect(streaming=True).to_pandas().corr(), 
                    annot=True, cmap='coolwarm', linewidths=0.5,  vmin=-1, vmax=1, center=0, ax=ax,
                    xticklabels=features, yticklabels=features)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()

    def plot_boxplot_wind_speed_direction(self, df, turbine_ids: list[str]) -> None:
        """_summary_

        Args:
            turbine_id (str): _description_
        """
        valid_turbines = self._get_valid_turbine_ids(df, turbine_ids=turbine_ids)
        
        if len(valid_turbines) == 0:
            return

        for turbine_id in valid_turbines:
            # Select data for the specified turbine
            # turbine_data = df.loc[turbine_id]
            
            turbine_data = df.select([pl.col("time"), pl.col(f"wind_speed_{turbine_id}"), pl.col(f"wind_direction_{turbine_id}")])\
                .filter(pl.any_horizontal(pl.col(f"wind_speed_{turbine_id}", f"wind_direction_{turbine_id}").is_not_null()))\
                .collect(streaming=True).to_pandas()
            
            if "hour" not in turbine_data.columns:
                # Extract hour from the time index
                # turbine_data = turbine_data.reset_index()
                # turbine_data = turbine_data.with_columns((pl.col('time').dt.hour).alias("hour"))
                turbine_data["hour"] = turbine_data["time"].dt.hour
            
            fig, ax = plt.subplots(2, 1, figsize=(12, 6))
            sns.boxplot(data=turbine_data, x='hour', y=f"wind_speed_{turbine_id}", ax=ax[0])
            sns.boxplot(data=turbine_data, x='hour', y=f"wind_direction_{turbine_id}", ax=ax[1])
            ax[0].set_title(f'Wind Speed Distribution by Hour for Turbine {turbine_id}')
            ax[1].set_title(f'Wind Direction Distribution by Hour for Turbine {turbine_id}')
            ax[0].set_xlabel("")
            ax[1].set_xlabel("Hour of Day")
            ax[0].set_ylabel("Wind Speed (m/s)")
            ax[1].set_ylabel("Wind Direction ($^\\circ$)")
            fig.tight_layout()
            plt.show()

    def plot_wind_speed_weibull(self, df, turbine_ids: list[str]) -> None:
        """_summary_

        Args:
            df (_type_): _description_
            turbine_ids (list[str]): _description_
        """
        valid_turbines = self._get_valid_turbine_ids(df, turbine_ids=turbine_ids)
        
        if len(valid_turbines) == 0:
            return
        
        for turbine_id in valid_turbines: 

            # Extract wind speed data
            wind_speeds = df.select(f"wind_speed_{turbine_id}")\
                .filter(pl.col(f"wind_speed_{turbine_id}").is_not_null())\
                .collect(streaming=True).to_pandas()

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

    def plot_wind_farm(self, wind_directions:list[float]=None, wind_speeds:list[float]|None=None, turbulence_intensities:list[float]|None=None) -> None:
        """_summary_

        Returns:
            _type_: _description_
        """
        if wind_directions is None:
            wind_directions = [90.0]
            
        if wind_speeds is None:
            wind_speeds = [10.0]

        if turbulence_intensities is None:
            turbulence_intensities = [0.08]

        # Ensure the paths are absolute
        
        # Initialize the FLORIS model
        try:
            fmodel = FlorisModel(self.farm_input_filepath)
        except FileNotFoundError:
            raise FileNotFoundError(f"Farm input file not found: {self.farm_input_filepath}")
        
        # Load the turbine data
        # try:
        #     fmodel.set_turbine_type(self.turbine_input_filepath)
        # except FileNotFoundError:
        #     print(f"Turbine file not found: {self.turbine_input_filepath}")
        #     print("Using default turbine type.")
        
        # Set initial wind conditions
        fmodel.set(turbine_library_path=os.path.dirname(self.turbine_input_filepath),
                   wind_directions=wind_directions, wind_speeds=wind_speeds, turbulence_intensities=turbulence_intensities)
        
        # Create the plot
        _, ax = plt.subplots(figsize=(15, 15))
        
        # Plot the turbine layout
        layoutviz.plot_turbine_points(fmodel, ax=ax)
        
        # Add turbine labels
        turbine_names = [f"T{i+1}" for i in range(fmodel.n_turbines)]
        layoutviz.plot_turbine_labels(
            fmodel, ax=ax, turbine_names=turbine_names, show_bbox=True, bbox_dict={"facecolor": "white", "alpha": 0.5}
        )
        
        # Calculate and visualize the flow field
        horizontal_plane = fmodel.calculate_horizontal_plane(height=90.0) # TODO get hubheight from turbine type
        visualize_cut_plane(horizontal_plane, ax=ax, min_speed=4, max_speed=10, color_bar=True)
        
        # Plot turbine rotors
        layoutviz.plot_turbine_rotors(fmodel, ax=ax)

        ax.set_xlim((fmodel.core.farm.layout_x.min(), fmodel.core.farm.layout_x.max()))
        ax.set_ylim((fmodel.core.farm.layout_y.min(), fmodel.core.farm.layout_y.max()))
        
        # Set plot title and labels
        plt.title('Wind Farm Layout', fontsize=16)
        plt.xlabel('X coordinate (m)', fontsize=12)
        plt.ylabel('Y coordinate (m)', fontsize=12)
        
        # Adjust layout and display the plot
        plt.tight_layout()
        plt.show()
        
        return fmodel

    def plot_filtered_vs_unfiltered(self, df, mask, features, feature_types, feature_labels):
        # feature_types = np.unique(["_".join(feat.split("_")[:-1]) for feat in features])
        _, ax = plt.subplots(len(feature_types), 1, sharex=True)
        if not hasattr(ax, "__len__"):
            ax = [ax]

        for feature in features:
            for ft, feature_type in enumerate(feature_types):
                if feature_type in feature:
                    ax_idx = ft
                    ax[ax_idx].set_title(feature_labels[ft])
                    break

            tid = feature.split("_")[-1]
            y = df.filter(mask(tid)).select(feature).collect(streaming=True).to_numpy().flatten()
            # ax[0].scatter(x=["operational"] * y.shape[0], y=y, color="red")
            ax[ax_idx].scatter(x=[tid] * y.shape[0], y=y, color="red", label="operational and null measurements")
            y = df.select(feature).collect(streaming=True).to_numpy().flatten()
            # ax[0].scatter(x=["inoperational"] * y.shape[0], y=y, color="blue", label="inoperational measurements")
            ax[ax_idx].scatter(x=[tid] * y.shape[0], y=y, color="blue", label="all measurements")
        
        h, l = ax[-1].get_legend_handles_labels()
        ax[-1].legend(h[:2], l[:2])
        del y

    @staticmethod
    def print_pc_unfiltered_vals(df, features, mask):
        out = []
        for feature in features:
            turbine_id = feature.split("_")[-1]
            pc_unfiltered_vals = 100 * (
                df.filter(mask(turbine_id)).select(feature).collect(streaming=True).shape[0] 
                / df.select(feature).collect(streaming=True).shape[0] 
            )
            print(f"Feature {feature} has {pc_unfiltered_vals} % unfiltered values.")
            out.append((feature, pc_unfiltered_vals))
            #   qa.describe(DataInspector.collect_data(df=df_query, feature_types=feature_type, turbine_ids=[turbine_id], mask=out_of_window[:, t_idx]))
        return out

    @staticmethod
    def get_features(df, feature_types, turbine_ids="all"):
        if feature_types is not None and not isinstance(feature_types, list):
            feature_types = [feature_types]

        if turbine_ids == "all":
            return sorted([col for col in df.collect_schema().names() if any(feat in col for feat in feature_types)])
        elif isinstance(turbine_ids, str):
            return sorted([col for col in df.collect_schema().names() if any((feat in col and turbine_ids in col) or (feat == col) for feat in feature_types)])
        else:
            return sorted([col for col in df.collect_schema().names() if any((feat in col and tid in col) or (feat == col) for feat in feature_types for tid in turbine_ids)])

    @staticmethod
    def collect_data(df, feature_types=None, turbine_ids="all", mask=None, to_pandas=True):
        if feature_types is not None and not isinstance(feature_types, list):
            feature_types = [feature_types]

        if feature_types is not None:
            df = df.select([pl.col(feat) for feat in DataInspector.get_features(df, feature_types, turbine_ids)])
        
        if mask is not None:
            df = df.filter(mask)

        if to_pandas:
            return df.collect(streaming=True).to_pandas()
        else:
            return df.collect(streaming=True)

    @staticmethod
    def unpivot_dataframe(df):
        # Unpivot LazyFrame into Long Form with `turbine_id` Column
        return pl.concat([
            df.select(pl.col("time"), cs.starts_with(feature_type))\
            .unpivot(index="time", value_name=feature_type)\
            .with_columns(pl.col("variable").str.slice(-5).alias("turbine_id"))\
            .drop("variable") for feature_type in ["wind_speed", "wind_direction", "turbine_status", "power_output", "nacelle_direction"]], how="align")\
                .group_by("turbine_id", "time").agg(cs.numeric().drop_nulls().first()).sort("turbine_id", "time")

    @staticmethod
    def pivot_dataframe(df):
        return df.collect(streaming=True).pivot(on="turbine_id", index="time").lazy()

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
    def calculate_and_display_mutual_info_scores(self, X: np.ndarray, y: np.ndarray, features: list[str], sequence_length: int, prediction_horizon: int) -> None:
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
                
                mi_df = pl.LazyFrame({
                    'Feature': features,
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
    def _plot_mi_scores(mi_df: pl.LazyFrame) -> None:
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
    def calculate_mi_scores(self, target_turbine: str, X: np.ndarray, y: np.ndarray, features: int, sequence_length: int, prediction_horizon: int) -> None:
        self._validate_input_data(X=X, y=y, features=features, sequence_length=sequence_length, prediction_horizon=prediction_horizon)
        # Remove the target turbine data in Y from the feature set X
        # 1. Create bool mask to filter out (~) data of target turbine. This works for both u and v components
        feature_mask = ~np.char.startswith(features, f'TurbineWindMag_{target_turbine}_')
        
        # 2. Apply the mask to filter X and feature_names
        X_filtered = X[:, :, feature_mask]
        feature_names_filtered = np.array(features)[feature_mask]
        
        # 3. Calculate and display mutual information scores
        logging.info(f"Calculating Mutual Information scores for target turbine: {target_turbine}")
        self.calculate_and_display_mutual_info_scores(X_filtered, y, feature_names_filtered, sequence_length, prediction_horizon)