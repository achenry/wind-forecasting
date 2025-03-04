### This file contains class and methods to: 
### - compute statistical summary of the data, 
### - plot the data distributions: power curve distribution, v vs u distribution, yaw angle distribution
import os
import time
import logging
from typing import Callable, Optional
from pathlib import Path

from itertools import cycle

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

mpi_exists = False
try:
    from mpi4py import MPI
    from mpi4py.futures import MPICommExecutor
    mpi_exists = True
except:
    print("No MPI available on system.")
    
from sklearn.feature_selection import mutual_info_regression
from tqdm.auto import tqdm
import re
import matplotlib.pyplot as plt
import seaborn as sns

factor = 1.5
# factor = 3.0 # single column
plt.rc('font', size=12*factor)          # controls default text sizes
plt.rc('axes', titlesize=20*factor)     # fontsize of the axes title
plt.rc('axes', labelsize=15*factor)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=12*factor)    # fontsize of the xtick labels
plt.rc('ytick', labelsize=12*factor)    # fontsize of the ytick labels
plt.rc('legend', fontsize=12*factor)    # legend fontsize
plt.rc('legend', title_fontsize=14*factor)  # legend title fontsize

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
    
    # INFO: @Juan 11/17/24 Added feature_mapping to allow for custom feature mapping, which is required for different data sources
    def __init__(self, turbine_input_filepath: Path, farm_input_filepath: Path, turbine_signature: Path, save_dir: Path, data_format='auto'):
        self._validate_input_data(turbine_input_filepath=turbine_input_filepath, farm_input_filepath=farm_input_filepath)
        self.turbine_input_filepath = turbine_input_filepath
        self.farm_input_filepath = farm_input_filepath
        self.turbine_signature = turbine_signature
        self.data_format = data_format
        self.save_dir = save_dir
        
        # Initialize the FLORIS modelw
        try:
            self.fmodel = FlorisModel(self.farm_input_filepath)
        except FileNotFoundError:
            raise FileNotFoundError(f"Farm input file not found: {self.farm_input_filepath}")
        
    # INFO: @Juan 10/18/24 Added method to detect data format automatically (wide or long)
    def detect_data_format(self, df: pl.LazyFrame) -> str:
        if self.data_format == 'auto':
            # Get schema without materializing the data
            column_names = df.collect_schema().names()
            return 'long' if 'turbine_id' in column_names else 'wide'
        return self.data_format

    def _get_valid_turbine_ids(self, df, turbine_ids: list[str]) -> list[str]:
        if isinstance(turbine_ids, str):
            turbine_ids = [turbine_ids]  # Convert single ID to list
        
        # valid_turbines = df.select("turbine_id").unique().filter(pl.col("turbine_id").is_in(turbine_ids)).collect(streaming=True).to_numpy()[:, 0]
        cols = df.collect_schema().names()
        # available_turbines = np.unique([re.findall(f"(?<=wind_direction_)(.*)", col)[0] for col in cols if "wind_direction" in col])
        available_turbines = set([m.group() for col in cols if (m:=re.search(self.turbine_signature, col))])

        if turbine_ids == "all":
            valid_turbines = available_turbines
        else:
            valid_turbines = [tid for tid in available_turbines if tid in turbine_ids]

        if len(valid_turbines) == 0:
            print(f"Error: No valid turbine IDs")
            print("Available turbine IDs:", available_turbines)
            return []
        
        if len(re.findall("\\d+", valid_turbines[0])):
            return sorted(valid_turbines, key=lambda tid: int(re.search("\\d+", tid).group(0)))
        else:
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


    def plot_time_series(self, df_query, turbine_ids: list[str], feature_types:Optional[list] = None, feature_labels:Optional[list] = None, continuity_groups: Optional[list] = None, label="", scatter = False) -> None:
        # Use provided feature mapping or fall back to instance default
        # current_mapping = feature_mapping or self.feature_mapping
        
        if feature_types is None:
            feature_types = ["wind_speed", "wind_direction", "nacelle_direction", "power_output"]
            feature_labels = ["Wind Speed (m/s)", "Wind Direction (deg)", "Nacelle Direction (deg)", "Power Output (kW)"]
        elif feature_labels is None:
           feature_labels = [" ".join(feat.split("_")).title() for feat in feature_types] 
        
        if isinstance(turbine_ids, str):
            turbine_ids = [turbine_ids]  # Convert single ID to list
        
        valid_turbines = self._get_valid_turbine_ids(df_query.select([cs.starts_with(feat_type) for feat_type in feature_types]), turbine_ids=turbine_ids)
         
        sns.set_style("whitegrid")
        sns.set_palette("deep")
        
        # columns = df_query.collect_schema().names()
        
        if continuity_groups is not None:
            for c, cg in enumerate(continuity_groups):
                fig, ax = plt.subplots(len(feature_types), 1, sharex=True)
                if not hasattr(ax, "__len__"):
                    ax = [ax]
                for f, feat in enumerate(feature_types):
                    # map feature name
                    # feat_col =  f"{current_mapping[feat][0]}"
                    # available_cols = ["time"]
                    # if all(f"{feat_col}_{tid}" in columns for tid in valid_turbines):
                    #     available_cols.append(feat_col)
                    # else:
                    #     print(f"Warning: Column {feat_col} not found in data for {valid_turbines}")
                    #     print(f"No valid data columns found for {valid_turbines}")
                    #     continue
                    
                    feature_df = df_query.filter(pl.col("continuity_group") == cg)\
                                 .select(pl.col("time"), cs.starts_with(feat))
                    
                    for tid in valid_turbines:
                        turbine_df = feature_df.select([pl.col("time"), cs.ends_with(tid)]).collect().to_pandas()
                        if scatter:
                            sns.scatterplot(data=turbine_df, x='time', y=f'{feat}_{tid}', ax=ax[f], label=f'{tid}')
                        else:
                            sns.lineplot(data=turbine_df, x='time', y=f'{feat}_{tid}', ax=ax[f], label=f'{tid}')
                        
                    ax[f].set_title(f"{feature_labels[f]} for CG {int(cg)}", fontsize=14)
                    ax[f].set_xlabel("Time (s)")
                    ax[f].set_ylabel(feature_labels[f])
                    ax[f].legend([], [], frameon=False)
                ax[-1].legend(bbox_to_anchor=(0.95, 1), loc="upper left", ncol=2)
        else:
            fig, ax = plt.subplots(len(feature_types), 1, figsize=(10, 6), sharex=True)
            if not hasattr(ax, "__len__"):
                ax = [ax]
            for f, feat in enumerate(feature_types):
                # map feature name
                # feat_col =  f"{current_mapping[feat][0]}"
                # available_cols = ["time"]
                # if all(f"{feat_col}_{tid}" in columns for tid in valid_turbines):
                #     available_cols.append(feat_col)
                # else:
                #     print(f"Warning: Column {feat_col} not found in data for {valid_turbines}")
                #     print(f"No valid data columns found for {valid_turbines}")
                #     continue
                    
                feature_df = df_query.select(pl.col("time"), cs.starts_with(feat))
                             
                for tid in valid_turbines:
                    turbine_df = feature_df.select([pl.col("time"), cs.ends_with(tid)]).collect().to_pandas()
                    if scatter:
                        sns.scatterplot(data=turbine_df, x='time', y=f'{feat}_{tid}', ax=ax[f], label=f'{tid}')
                    else:
                        sns.lineplot(data=turbine_df, x='time', y=f'{feat}_{tid}', ax=ax[f], label=f'{tid}')
                
                # ax[f].set_title(f"{feature_labels[f]}")
                ax[f].set_xlabel("Time (s)")
                ax[f].set_ylabel(feature_labels[f])
                ax[f].legend([], [], frameon=False)
            # ax[-1].legend(bbox_to_anchor=(0.95, 1), loc="upper left", ncol=2)
                
        fig.suptitle(f'Time Series for Turbines: {", ".join(valid_turbines)}', fontsize=16)
        ax[0].legend(loc='upper left', bbox_to_anchor=(0.95, 1), ncol=2)
        
        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(self.save_dir, f'time_series_{label}.png'), dpi=100)
        # plt.close()

    def plot_wind_speed_power(self, df: pl.LazyFrame, turbine_ids: list[str], fig_label) -> None:
        """Plot wind speed vs power output scatter plot for specified turbines.
        
        Args:
            df: Input dataframe
            turbine_ids: List of turbine IDs to plot
        """
        # Convert turbine_ids to list if it's a single string
        if isinstance(turbine_ids, str):
            turbine_ids = [turbine_ids]
        
        valid_turbines = self._get_valid_turbine_ids(df, turbine_ids=turbine_ids)
        
        if len(valid_turbines) == 0:
            return
        
        # file_set_indices = df.select("file_set_idx").unique().collect().to_numpy().flatten()
        
        # for file_set_idx in file_set_indices:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Collect all required data at once for better performance
        
        required_cols = []
        for turbine_id in valid_turbines:
            # Get the actual column names using the mapping
            ws_cols = self.get_features(df, "wind_speed", turbine_id)
            power_cols = self.get_features(df, "power_output", turbine_id)
            if ws_cols and power_cols:
                required_cols.extend([ws_cols[0], power_cols[0]])
            else:
                print(f"Could not find required columns for turbine {turbine_id}")
                print(f"Wind speed columns found: {ws_cols}")
                print(f"Power output columns found: {power_cols}")
        
        if not required_cols:
            print("No valid columns found for plotting")
            return
        
        # Collect all data at once
        all_data = df.select(required_cols)\
            .filter(pl.all_horizontal(pl.col(required_cols).is_not_null()))\
            .collect()\
            .to_pandas()
        
        # Plot for each turbine using the collected data
        for turbine_id in valid_turbines:
            ws_cols = self.get_features(df, "wind_speed", turbine_id)
            power_cols = self.get_features(df, "power_output", turbine_id)
            
            if not ws_cols or not power_cols:
                continue
            
            ws_col = ws_cols[0]
            power_col = power_cols[0]
            
            sns.scatterplot(data=all_data, ax=ax, x=ws_col, y=power_col,
                        label=turbine_id, alpha=0.5)
        
        ax.set_xlabel('Wind Speed (m/s)')
        ax.set_ylabel('Power Output (kW)')
        ax.set_title('Wind Speed vs Power Output')
        ax.legend(title='Turbine ID', loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.3)
        sns.despine()
        plt.tight_layout()
        plt.show()
        fig.savefig(os.path.join(self.save_dir, f'wind_speed_power{fig_label}.png'), dpi=100)
        # plt.close()

    # DEBUG: @Juan 10/18/24 Added method to plot wind rose for both wide and long formats [CHECK]
    def plot_wind_rose(self, df, turbine_ids: list[str] | str, feature_type: str, fig_label: str="") -> None:
        data_format = self.detect_data_format(df)
        # file_set_indices = df.select("file_set_idx").unique().collect().to_numpy().flatten()
        if data_format == 'wide':
            if turbine_ids == "all":
                # Get all wind direction and speed columns
                columns = df.collect_schema().names()
                dir_cols = [col for col in columns if feature_type in col]
                wind_spd_cols = [col for col in columns if "wind_speed" in col]
                
                # Filter and collect data all at once
                filtered_data = df.select([
                    *dir_cols,
                    *wind_spd_cols
                ])\
                .filter(pl.all_horizontal(pl.col(dir_cols + wind_spd_cols).is_not_null()))
                
                # Combine all turbine data
                direc = filtered_data.select(dir_cols).collect().to_numpy().flatten()
                wind_spd = filtered_data.select(wind_spd_cols).collect().to_numpy().flatten()
                
                # Double check arrays have same length and are not empty
                if len(direc) != len(wind_spd) or len(direc) == 0:
                    print(f"Mismatch in data lengths or empty data: dir={len(direc)}, spd={len(wind_spd)}")
                    return
                
                # Create the windrose plot directly without creating a separate figure first
                fig = plt.figure(figsize=(10, 10))
                rect = [0.1, 0.1, 0.8, 0.8]
                ax = WindroseAxes(fig, rect)
                fig.add_axes(ax)
                
                ax.bar(direc, wind_spd, normed=True, opening=0.8, edgecolor='white')
                ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
                # ax.set_xlim((0, 6))
                ax.set_ylim((0, 25))
                ax.set_title("")
                # ax.set_title(f"{' '.join(feature_type.split('_')).capitalize()} Rose all Turbines")
                plt.show()
                fig.savefig(os.path.join(self.save_dir, f'{fig_label}.png'), dpi=100)
            else:
                valid_turbines = self._get_valid_turbine_ids(df, turbine_ids=turbine_ids)

                if len(valid_turbines) == 0:
                    return

                for turbine_id in valid_turbines:
                    # Filter out NaN values for specific turbine
                    turbine_data = df.select([
                        pl.col(f"wind_speed_{turbine_id}"), 
                        pl.col(f"{feature_type}_{turbine_id}")
                    ])\
                    .filter(pl.all_horizontal(
                        pl.col(f"wind_speed_{turbine_id}").is_not_null(), 
                        pl.col(f"{feature_type}_{turbine_id}").is_not_null()
                    ))
                    
                    direc = turbine_data.select(f"{feature_type}_{turbine_id}").collect().to_numpy().flatten()
                    wind_spd = turbine_data.select(f"wind_speed_{turbine_id}").collect().to_numpy().flatten()

                    # Verify data lengths match
                    if len(direc) != len(wind_spd) or len(direc) == 0:
                        print(f"Mismatch in data lengths or empty data for turbine {turbine_id}")
                        continue

                    fig = plt.figure(figsize=(10, 10))
                    ax = WindroseAxes.from_ax()
                    ax.bar(direc, wind_spd, normed=True, opening=0.8, edgecolor='white')
                    ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
                    ax.set_title(f"{' '.join(feature_type.split('_')).capitalize()} Rose for Turbine {turbine_id}")
                    plt.show()
                    fig.savefig(os.path.join(self.save_dir, f'{fig_label}_{turbine_id}.png'), dpi=100)
                    
        else:  # long format
            if turbine_ids == "all":
                fig = plt.figure(figsize=(10, 10))
                ax = WindroseAxes.from_ax()
                ax.bar(df.select(feature_type).collect().to_numpy()[:, 0], 
                        df.select("wind_speed").collect().to_numpy()[:, 0], 
                        normed=True, opening=0.8, edgecolor='white')
                ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
                ax.set_title(f"{' '.join(feature_type.split('_')).capitalize()} Rose for all Turbines")
                plt.show()
                fig.savefig(os.path.join(self.save_dir, f'{fig_label}.png'), dpi=100)
            else:
                valid_turbines = self._get_valid_turbine_ids(df, turbine_ids=turbine_ids)
            
                if len(valid_turbines) == 0:
                    return

                for turbine_id in valid_turbines:
                    turbine_data = df.filter(pl.col("turbine_id") == turbine_id)\
                        .select(["wind_speed", feature_type])\
                        .filter(pl.all_horizontal(pl.col("wind_speed").is_not_null(), pl.col(feature_type).is_not_null()))
                    
                    fig, ax = plt.subplot(1, 1, figsize=(10, 10))
                    ax = WindroseAxes.from_ax()
                    ax.bar(turbine_data.select(feature_type).collect().to_numpy()[:, 0], 
                            turbine_data.select("wind_speed").collect().to_numpy()[:, 0], normed=True, opening=0.8, edgecolor='white')
                    ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
                    ax.set_title(f"{' '.join(feature_type.split('_')).capitalize()} Rose for Turbine {turbine_id}")
                    plt.show()
                    fig.savefig(os.path.join(self.save_dir, f"{fig_label}_{turbine_id}.png"), dpi=100)
        # plt.close()

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
        
        _, ax = plt.subplots(1, 1, figsize=(10, 6))
        for col in temp_columns:
            sns.histplot(df[col], bins=20, kde=True, label=col)
        
        ax.set(title='Temperature Distributions', xlabel="Temperature", ylabel="Frequency")
        plt.legend()
        # plt.show()
        plt.savefig(os.path.join(self.save_dir, 'temperature_distribution.png'), dpi=100)
        plt.close()

    def plot_correlation(self, df, features) -> None:
        """_summary_
        """
        _, ax = plt.subplots(1, 1, figsize=(10, 6))
        sns.heatmap(df.select(features).collect().to_pandas().corr(), 
                    annot=True, cmap='coolwarm', linewidths=0.5,  vmin=-1, vmax=1, center=0, ax=ax,
                    xticklabels=features, yticklabels=features)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(self.save_dir, 'correlation_matrix.png'), dpi=100)
        plt.close()

    def plot_boxplot_wind_speed_direction(self, df, turbine_ids: list[str]) -> None:
        """Plot boxplots of wind speed and direction by hour for specified turbines."""
        valid_turbines = self._get_valid_turbine_ids(df, turbine_ids=turbine_ids)
        
        if len(valid_turbines) == 0:
            return

        for turbine_id in valid_turbines:
            # Select and cast data types in Polars
            turbine_data = df.select([
                pl.col("time"),
                pl.col(f"wind_speed_{turbine_id}"),
                pl.col(f"wind_direction_{turbine_id}")
            ])\
            .filter(
                pl.any_horizontal([
                    pl.col(f"wind_speed_{turbine_id}").is_not_null(),
                    pl.col(f"wind_direction_{turbine_id}").is_not_null()
                ])
            )\
            .with_columns(
                pl.col("time").dt.hour().alias("hour").cast(pl.Int32)
            )\
            .collect()\
            .to_pandas()
            
            # turbine_data['hour'] = turbine_data['hour'].astype('int32')
            # turbine_data[f"wind_speed_{turbine_id}"] = turbine_data[f"wind_speed_{turbine_id}"].astype('float64')
            # turbine_data[f"wind_direction_{turbine_id}"] = turbine_data[f"wind_direction_{turbine_id}"].astype('float64')
            
            # Create plots
            fig, ax = plt.subplots(2, 1, figsize=(10, 6))
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
            plt.savefig(os.path.join(self.save_dir, 'boxplot_wind_speed_direction.png'), dpi=100)
            plt.close()

    def plot_data_distribution(self, df, feature_types, turbine_ids: list[str], distribution: Callable[[np.ndarray], np.ndarray]=stats.weibull_min) -> None:
        """_summary_

        Args:
            df (_type_): _description_
            turbine_ids (list[str]): _description_
        """
        fig, ax = plt.subplots(1, len(feature_types), figsize=(10, 6))
        for ax_idx, feature_type in enumerate(feature_types):
            x = np.linspace(df.select(cs.starts_with(feature_type)).collect().to_numpy().min(), 
                            df.select(cs.starts_with(feature_type)).collect().to_numpy().max(), 1000)
            for turbine_id in turbine_ids:
                # Extract data
                values = df.select(f"{feature_type}_{turbine_id}").collect().to_numpy().flatten()

                # Fit Weibull distribution
                dist_params = distribution.fit(values)

                # Create a range of wind speeds for the fitted distribution
                # x = np.linspace(distribution.ppf(0.01), distribution.ppf(0.99), 100)
                y = distribution.pdf(x, *dist_params)

                # Plot
                ax[ax_idx].plot(x, y, '--', lw=2)
                ax[ax_idx].hist(values, density=True, bins="auto", color=ax[ax_idx].lines[-1]._color, label=turbine_id)
                
            ax[ax_idx].set_title(f'{feature_type} Distribution with Fitted {distribution.__class__.__name__}', fontsize=16)
        plt.axis("tight")
        fig.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        fig.tight_layout()
        # sns.despine()
        # plt.show()
        fig.savefig(os.path.join(self.save_dir, 'data_distribution.png'), dpi=100)
        plt.close()

    def plot_wind_speed_weibull(self, df, turbine_ids: list[str], fig_label: str="") -> None:
        """_summary_

        Args:
            df (_type_): _description_
            turbine_ids (list[str]): _description_
        """
        if turbine_ids == "all":
            # Extract wind speed data
            wind_speeds = df.select(cs.contains("wind_speed")).collect().to_numpy().flatten()
            wind_speeds = wind_speeds[wind_speeds > 0]
            wind_speeds = wind_speeds[np.isfinite(wind_speeds)]  # Remove non-finite values
        
            if len(wind_speeds) == 0:
                print("No valid wind speed data found after filtering")
                return

            # Fit Weibull distribution
            shape, loc, scale = stats.weibull_min.fit(wind_speeds, floc=0)

            # Create a range of wind speeds for the fitted distribution
            x = np.linspace(0, wind_speeds.max(), 100)
            y = stats.weibull_min.pdf(x, shape, loc, scale)

            # Plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            sns.histplot(wind_speeds, stat='density', kde=True, color='skyblue', label='Observed', ax=ax)
            ax.plot(x, y, 'r-', lw=2, label=f'Weibull (k={shape:.2f}, λ={scale:.2f})')
            
            # ax.set_title('Wind Speed Distribution with Fitted Weibull', fontsize=16)
            ax.set_xlabel('Wind Speed (m/s)')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
            sns.despine()
            plt.show()
            plt.tight_layout()
            fig.savefig(os.path.join(self.save_dir, f'wind_speed_weibull{fig_label}.png'), dpi=100)
            # plt.close()

            print(f"Weibull shape parameter (k): {shape:.2f}")
            print(f"Weibull scale parameter (λ): {scale:.2f}")
        else:
            valid_turbines = self._get_valid_turbine_ids(df, turbine_ids=turbine_ids)
            
            if len(valid_turbines) == 0:
                return
            
            for turbine_id in valid_turbines:
                # Extract wind speed data
                wind_speeds = df\
                    .select(f"wind_speed_{turbine_id}")\
                    .filter(pl.col(f"wind_speed_{turbine_id}").is_not_null())\
                    .collect().to_numpy().flatten()
                wind_speeds = wind_speeds[np.isfinite(wind_speeds)]  # Remove non-finite values
                
                if len(wind_speeds) == 0:
                    print(f"No valid wind speed data found for turbine {turbine_id}")
                    continue

                # Fit Weibull distribution
                shape, loc, scale = stats.weibull_min.fit(wind_speeds, floc=0)

                # Create a range of wind speeds for the fitted distribution
                x = np.linspace(0, wind_speeds.max(), 100)
                y = stats.weibull_min.pdf(x, shape, loc, scale)

                # Plot
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                sns.histplot(wind_speeds, stat='density', kde=True, color='skyblue', label='Observed', ax=ax)
                ax.plot(x, y, 'r-', lw=2, label=f'Weibull (k={shape:.2f}, λ={scale:.2f})')
                
                ax.set_title(f'Wind Speed Distribution with Fitted Weibull - Turbine {turbine_id}', fontsize=16)
                ax.set_xlabel('Wind Speed (m/s)', fontsize=12)
                ax.set_ylabel('Density', fontsize=12)
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                sns.despine()
                plt.show()
                fig.savefig(os.path.join(self.save_dir, f'wind_speed_weibull{fig_label}_{turbine_id}.png'), dpi=100)
                # plt.close()

                print(f"Weibull shape parameter (k): {shape:.2f}")
                print(f"Weibull scale parameter (λ): {scale:.2f}")

    def plot_wind_farm(self, wind_directions:list[float]=None, wind_speeds:list[float]|None=None, turbulence_intensities:list[float]|None=None) -> None:
        """_summary_

        Returns:
            _type_: _description_
        """
        if wind_directions is None:
            wind_directions = [190.0]
            
        if wind_speeds is None:
            wind_speeds = [10.0]

        if turbulence_intensities is None:
            turbulence_intensities = [0.08]

        # Ensure the paths are absolute
        
        # Load the turbine data
        # try:
        #     fmodel.set_turbine_type(self.turbine_input_filepath)
        # except FileNotFoundError:
        #     print(f"Turbine file not found: {self.turbine_input_filepath}")
        #     print("Using default turbine type.")
        
        # Set initial wind conditions
        self.fmodel.set(turbine_library_path=os.path.dirname(self.turbine_input_filepath),
                   wind_directions=wind_directions, wind_speeds=wind_speeds, turbulence_intensities=turbulence_intensities)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot the turbine layout
        layoutviz.plot_turbine_points(self.fmodel, ax=ax)
        
        # Add turbine labels
        # turbine_names = [f"T{i+1}" for i in range(self.fmodel.n_turbines)]
        # layoutviz.plot_turbine_labels(
        #     self.fmodel, ax=ax, turbine_names=turbine_names, show_bbox=True, bbox_dict={"facecolor": "white", "alpha": 0.5}
        # )
        
        # Calculate and visualize the flow field
        horizontal_plane = self.fmodel.calculate_horizontal_plane(height=self.fmodel.core.farm.hub_heights[0])
        visualize_cut_plane(horizontal_plane, ax=ax, min_speed=4, max_speed=10, color_bar=True)
        
        # Plot turbine rotors
        # layoutviz.plot_turbine_rotors(self.fmodel, ax=ax)
        
        ax.set_xlim((horizontal_plane.df.x1.min(), horizontal_plane.df.x1.max()))
        ax.set_ylim((horizontal_plane.df.x2.min(), horizontal_plane.df.x2.max()))
        
        # Set plot title and labels
        # ax.set_title('Wind Farm Layout', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_xlabel('X coordinate (m)', fontsize=20)
        ax.set_ylabel('Y coordinate (m)', fontsize=20)
        fig.get_axes()[1].yaxis.label.set_text("Wind Speed (m/s)")
         
        # Adjust layout and display the plot
        plt.tight_layout()
        plt.show()
        fig.savefig(os.path.join(self.save_dir, 'wind_farm.png'), dpi=100)
        # plt.close()
        
        # return self.fmodel

    # @staticmethod
    def plot_nulled_vs_remaining(self, df, mask_func, mask_input_features, output_features, feature_types, feature_labels):
        
        sns.set(style="whitegrid")

        fig, ax = plt.subplots(len(feature_types), 1, sharex=True, figsize=(10, 6))
        if not isinstance(ax, np.ndarray):
            ax = [ax]

        for inp_feat, opt_feat in zip(mask_input_features, output_features):
            mask_array = mask_func(inp_feat)
            if mask_array is None:
                continue

            for ft, feature_type in enumerate(feature_types):
                if feature_type in opt_feat:
                    ax_idx = ft
                    ax[ax_idx].set_title(feature_labels[ft])
                    break

            # Plot all measurements
            y_all = df.select(opt_feat).collect().to_numpy().flatten()
            ax[ax_idx].scatter(x=[inp_feat] * len(y_all), y=y_all, color="blue", label="All Measurements")

            # Plot nulled measurements
            y_nulled = df.filter(mask_array).select(opt_feat).collect().to_numpy().flatten()
            ax[ax_idx].scatter(x=[inp_feat] * len(y_nulled), y=y_nulled, color="red", label="Nulled Measurements")

        ax[-1].set_xlabel("Turbine ID")
        # Avoid duplicate labels
        handles, labels = ax[-1].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax[-1].legend(by_label.values(), by_label.keys(), loc="upper right")
        fig.show()
        fig.savefig(os.path.join(self.save_dir, 'filtered_vs_unfiltered.png'), dpi=100)
        # plt.close()

    @staticmethod
    def print_pc_remaining_vals(df, mask_func, mask_input_features, output_features):
        out = []
        for inp_feat, opt_feat in zip(mask_input_features, output_features):
            # tid = feature.split("_")[-1]
            mask_array = mask_func(inp_feat)
            if mask_array is None:
                logging.info(f"Mask error for feature {inp_feat}: mask is None")
                continue
            try:
                pc_remaining_vals = 100 * (
                    df.filter(~mask_array)
                    .select(pl.len())
                    .collect()
                    .item()
                    / df.select(pl.len()).collect().item()
                )
                print(f"Feature {opt_feat} has {pc_remaining_vals:.2f}% remaining values.")
                out.append((opt_feat, pc_remaining_vals))
            except Exception as e:
                logging.error(f"Error processing feature {opt_feat}: {str(e)}")
        return out

    def get_features(self, df, feature_types, turbine_ids="all"):
        """Get feature columns based on mapping and turbine ID."""
        data_format = self.detect_data_format(df)
        if feature_types is not None and not isinstance(feature_types, list):
            feature_types = [feature_types]
        
        cols = df.collect_schema().names()
        if data_format == 'wide':
            matching_cols = []
            for feature_type in feature_types:
                # Get the possible feature names from mapping
                # mapped_features = self.feature_mapping.get(feature_type, [feature_type])
                
                if turbine_ids == "all":
                    new_cols = [col for col in cols if feature_type in col]
                elif isinstance(turbine_ids, str):
                    new_cols = [col for col in cols if col == f"{feature_type}_{turbine_ids}"]
                else:
                    new_cols = [col for col in cols if any(col == f"{feature_type}_{tid}" for tid in turbine_ids)]
                matching_cols.extend(new_cols)
            
            if len(re.findall("\\d+", matching_cols[0])):
                 return sorted(matching_cols, 
                               key=lambda col: (re.search(".*?(?=\\d+)", col).group(0), int(re.search("\\d+", col).group(0))))
            else:
                return sorted(matching_cols)
        else:  # long format
            return sorted([col for col in cols if col in feature_types])

    def collect_data(self, df, feature_types=None, turbine_ids="all", mask=None, to_pandas=True):
        data_format = self.detect_data_format(df)
        if feature_types is not None and not isinstance(feature_types, list):
            feature_types = [feature_types]

        if data_format == 'wide':
            if feature_types is not None:
                df = df.select([pl.col(feat) for feat in self.get_features(df, feature_types, turbine_ids)])
        else:  # long format
            if feature_types is not None:
                df = df.filter(pl.col('feature_name').is_in(feature_types))
            if turbine_ids != "all":
                df = df.filter(pl.col('turbine_id').is_in(turbine_ids))

        if mask is not None:
            df = df.filter(mask)

        if to_pandas:
            return df.collect().to_pandas()
        else:
            return df.collect(streaming=True)

    @staticmethod
    def unpivot_dataframe(df, value_vars, turbine_signature, data_format="wide"):
        id_vars = df.select(pl.exclude(*[f"^{vv}.*$" for vv in value_vars])).columns
        # value_vars = df.select(pl.exclude(*[f"^{iv}.*$" for iv in id_vars])).columns 
        if data_format == 'wide':
            # Unpivot wide format to long format
            return pl.concat([
                df.select(*[pl.col(id_var) for id_var in id_vars], cs.starts_with(feature_type))\
                .unpivot(index=id_vars, variable_name="feature", value_name=feature_type)\
                .with_columns(pl.col("feature").str.extract(turbine_signature, group_index=0).alias("turbine_id"))\
                .drop("feature") for feature_type in value_vars if len(df.select(cs.starts_with(feature_type)).columns)], how="align")\
                .group_by("turbine_id", *id_vars).agg(cs.numeric().drop_nulls().first()).sort("turbine_id", "time")
        else:
            # Data is already in long format
            return df

    @staticmethod
    def pivot_dataframe(df, data_format="long"):
        # data_format = self.detect_data_format(df)
        if data_format == 'long':
            # Pivot long format to wide format
            if "continuity_group" in df.collect_schema().names():
              return df.collect(streaming=True).pivot(on="turbine_id", index=["time", "continuity_group"]).lazy()
            else:
              return df.collect(streaming=True).pivot(on="turbine_id", index="time").lazy()
        else:
            # Data is already in wide format
            return df
    
    #INFO: @Juan 10/18/24 Adapted and incorporated plotting method for yaw and power time series from old defunct data_reader.py
    def plot_yaw_power_ts(self, df, turbine_ids, save_path=None, include_yaw=True, include_power=True, controller_dt=None):
        df = df.collect().to_pandas()
        colors = sns.color_palette(palette='Paired')

        turbine_wind_direction_cols = self.get_features(df, "wind_direction", turbine_ids)
        turbine_power_cols = self.get_features(df, "power_output", turbine_ids)
        yaw_angle_cols = self.get_features(df, "nacelle_direction", turbine_ids)

        for seed in sorted(np.unique(df["WindSeed"])):
            fig, ax = plt.subplots(int(include_yaw + include_power), 1, sharex=True, figsize=(10, 6))
            ax = np.atleast_1d(ax)

            seed_df = df.loc[df["WindSeed"] == seed].sort_values(by="time")
            
            if include_yaw:
                ax_idx = 0
                ax[ax_idx].plot(seed_df["time"], seed_df["FreestreamWindDir"], label="Freestream wind dir.", color="black")
                ax[ax_idx].plot(seed_df["time"], seed_df["FilteredFreestreamWindDir"], label="Filtered freestream wind dir.", color="black", linestyle="--")
                
            for t, (wind_dir_col, power_col, yaw_col, color) in enumerate(zip(turbine_wind_direction_cols, turbine_power_cols, yaw_angle_cols, cycle(colors))):
                if include_yaw:
                    ax_idx = 0
                    ax[ax_idx].plot(seed_df["time"], seed_df[yaw_col], color=color, label=f"T{t+1} yaw setpoint", linestyle=":")
                    
                    if controller_dt is not None:
                        [ax[ax_idx].axvline(x=_x, linestyle=(0, (1, 10)), linewidth=0.5) for _x in np.arange(0, seed_df["time"].iloc[-1], controller_dt)]

                if include_power:
                    next_ax_idx = (1 if include_yaw else 0)
                    if t == 0:
                        ax[next_ax_idx].fill_between(seed_df["time"], seed_df[power_col] / 1e3, color=color, label=f"T{t+1} power")
                    else:
                        ax[next_ax_idx].fill_between(seed_df["time"], seed_df[turbine_power_cols[:t+1]].sum(axis=1) / 1e3, 
                                        seed_df[turbine_power_cols[:t]].sum(axis=1)  / 1e3,
                            color=color, label=f"T{t+1} power")
            
            if include_power:
                next_ax_idx = (1 if include_yaw else 0)
                ax[next_ax_idx].plot(seed_df["time"], seed_df[turbine_power_cols].sum(axis=1) / 1e3, color="black", label="Farm power")
        
            if include_yaw:
                ax_idx = 0
                ax[ax_idx].set(title="Wind Direction / Nacelle Direction ($^\\circ$)", xlim=(0, int((seed_df["time"].max() + seed_df["time"].diff().iloc[1]) // 1)), ylim=(245, 295))
                ax[ax_idx].legend(ncols=2, loc="lower right")
                if not include_power:
                    ax[ax_idx].set(xlabel="Time (s)", title="Turbine Powers (MW)")
            
            if include_power:
                next_ax_idx = (1 if include_yaw else 0)
                ax[next_ax_idx].set(xlabel="Time (s)", title="Turbine Powers (MW)", ylim=(0, None))
                ax[next_ax_idx].legend(ncols=2, loc="lower right")

            fig.suptitle(f"Yaw and Power Time Series for Seed {seed}")
            
            if save_path:
                fig.savefig(os.path.join(self.save_dir, save_path.replace(".png", f"_seed{seed}.png")), dpi=100)
            else:
                # plt.show()
                plt.savefig(os.path.join(self.save_dir, 'yaw_power_ts.png'), dpi=100)
                plt.close()

        return fig, ax

    def plot_wind_offset(self, full_df, title, turbine_ids):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for turbine_id in turbine_ids:
            # df = full_df.filter(pl.col(f"power_output_{turbine_id}") >= 0).select("time", f"wind_direction_{turbine_id}").collect()
            df = full_df.filter(pl.col(f"power_output_{turbine_id}") >= 0)\
                        .select("time", cs.starts_with("wind_direction"), "wd_median")
                        
            ax.plot(df.select("time").collect().to_numpy().flatten(),
                    df.select(pl.col(f"wind_direction_{turbine_id}") - pl.col("wd_median"))\
                    .select(pl.when(pl.all() > 180.0).then(pl.all() - 360.0).otherwise(pl.all())).collect().to_numpy().flatten(),
                                label=f"{turbine_id}")

        # ax.legend(ncol=8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Wind Direction - Median Wind Direction ($^\\circ$)")

        ax.set_title(title)
        fig.savefig(os.path.join(self.save_dir, f"wind_offset_{title}.png"), dpi=100)
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
        # TODO remove dependency on MPI
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

    @staticmethod
    def print_df_state(df_query, feature_types=None):
        if feature_types is None:
            feature_types = ["wind_speed", "wind_direction"]
        
        df_query = df_query.select(
            [pl.col(col) for col in df_query.collect_schema().names() 
                         if (not df_query.select(pl.col(col).is_null().all()).collect().item() 
                         and any(col.startswith(feat_type) for feat_type in feature_types))])
        
        # TODO not robust way to capture feature... what if one feat_type is a substring of another..
        feature_types = set(feat_type for feat_type in feature_types if any(feat_type in col for col in df_query.collect_schema().names()))
        n_unique_expr =  pl.all().drop_nulls().n_unique()
        print("% unique values", pl.concat([
            df_query.select(cs.starts_with(feat_type))\
                    .select((100 * pl.min_horizontal(n_unique_expr) / pl.len()).alias(f"{feat_type}_min_n_unique"), 
                            (100 * pl.max_horizontal(n_unique_expr) / pl.len()).alias(f"{feat_type}_max_n_unique"))\
                    .collect() for feat_type in feature_types], how="horizontal"), sep="\n")
        
        n_non_null_expr = pl.all().count()
        print("% non-null values", pl.concat([
            df_query.select(cs.starts_with(feat_type))\
                    .select((100 * pl.min_horizontal(n_non_null_expr) / pl.len()).alias(f"{feat_type}_min_non_null"), 
                            (100 * pl.max_horizontal(n_non_null_expr) / pl.len()).alias(f"{feat_type}_max_non_null"))\
                    .collect() for feat_type in feature_types], how="horizontal"), sep="\n")