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

class DataInspector:
    """_summary_
    - compute statistical summary of the data,
    - plot the data distributions: 
    -   power curve distribution, 
    -   v vs u distribution, 
    -   yaw angle distribution 
    """
    def __init__(self, df: pl.DataFrame, turbine_input_filepath: str, farm_input_filepath: str):
        self.df = df
        self.turbine_input_filepath = turbine_input_filepath
        self.farm_input_filepath = farm_input_filepath

    def plot_time_series(self, turbine_ids: list[str]) -> None:
        if isinstance(turbine_ids, str):
            turbine_ids = [turbine_ids]  # Convert single ID to list
        
        available_turbines = self.df['turbine_id'].unique()
        valid_turbines = [tid for tid in turbine_ids if tid in available_turbines]
        
        if not valid_turbines:
            print(f"Error: No valid turbine IDs")
            print("Available turbine IDs:", available_turbines)
            return
        
        sns.set_style("whitegrid")
        sns.set_palette("deep")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        for turbine_id in valid_turbines:
            # turbine_data = self.df.loc[turbine_id]
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

    def plot_scatter_wind_speed_power(self, turbine_ids: list[str]) -> None:
        """_summary_

        """
        if isinstance(turbine_ids, str):
            turbine_ids = [turbine_ids]  # Convert single ID to list
        
        available_turbines = self.df['turbine_id'].unique()
        valid_turbines = [tid for tid in turbine_ids if tid in available_turbines]
        
        if not valid_turbines:
            print(f"Error: No valid turbine IDs")
            print("Available turbine IDs:", available_turbines)
            return
        
        _, ax = plt.subplots(1, 1, figsize=(12, 6))

        for turbine_id in valid_turbines:
            # turbine_data = self.df.loc[turbine_id]
            turbine_data = self.df.filter(pl.col("turbine_id") == turbine_id).drop_nulls().to_pandas()
            sns.scatterplot(data=turbine_data, ax=ax, x='wind_speed', y='power_output', ax=ax)

        ax.set(xlabel='Wind Speed', ylabel='Power Output', title='Scatter Plot of Wind Speed vs Power Output')
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

    def plot_heatmap_correlation(self) -> None:
        """_summary_
        """
        _, ax = plt.subplots(1, 1, figsize=(12, 6))
        sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, ax=ax)
        ax.set(title='Heatmap of Correlation Matrix')
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

    def plot_wind_speed_weibull(self) -> None:
        """_summary_

        """
        # Extract wind speed data
        wind_speeds = self.df['wind_speed'].drop_na()

        # Fit Weibull distribution
        shape, loc, scale = stats.weibull_min.fit(wind_speeds, floc=0)

        # Create a range of wind speeds for the fitted distribution
        x = np.linspace(0, wind_speeds.max(), 100)
        y = stats.weibull_min.pdf(x, shape, loc, scale)

        # Plot
        plt.figure(figsize=(12, 6))
        sns.histplot(wind_speeds, stat='density', kde=True, color='skyblue', label='Observed')
        plt.plot(x, y, 'r-', lw=2, label=f'Weibull (k={shape:.2f}, λ={scale:.2f})')
        
        plt.title(title='Wind Speed Distribution with Fitted Weibull', fontsize=16)
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
        fmodel = FlorisModel(self.farm_input_filepath)
        
        # Load the turbine data
        try:
            fmodel.set_turbine_type(self.turbine_input_filepath)
        except FileNotFoundError:
            print(f"Turbine file not found: {self.turbine_input_filepath}")
            print("Using default turbine type.")
        
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