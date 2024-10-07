### This file contains class and methods to: 
### - compute statistical summary of the data, 
### - plot the data distributions: power curve distribution, v vs u distribution, yaw angle distribution
import os
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

    def _get_valid_turbine_ids(self, turbine_ids: list[str]) -> list[str]:
        if isinstance(turbine_ids, str):
            turbine_ids = [turbine_ids]  # Convert single ID to list
        
        available_turbines = self.df['turbine_id'].unique()
        valid_turbines = [tid for tid in turbine_ids if tid in available_turbines]
        
        if not valid_turbines:
            print(f"Error: No valid turbine IDs")
            print("Available turbine IDs:", available_turbines)
            return []
        
        return valid_turbines

    def plot_time_series(self, turbine_ids: list[str]) -> None:
        """_summary_

        Args:
            turbine_ids (list[str]): _description_
        """
        valid_turbines = self._get_valid_turbine_ids(turbine_ids=turbine_ids)
        
        if not valid_turbines:
            return
        
        sns.set_style("whitegrid")
        sns.set_palette("deep")

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        for turbine_id in valid_turbines:
            # turbine_data = self.df.loc[turbine_id]
            # turbine_data = self.df.filter(pl.col("turbine_id") == turbine_id).drop_nulls().to_pandas()
            turbine_data = self.df.select(["time", "wind_speed", "wind_direction", "power_output", "turbine_id"]).filter(pl.col("turbine_id") == turbine_id)
            # plt.plot(turbine_data["time"], turbine_data["wind_speed"])
            sns.lineplot(data=turbine_data.filter(pl.col("wind_speed").is_not_nan()).to_pandas(),
                         x='time', y='wind_speed', ax=ax1, label=f'{turbine_id} Wind Speed')
            sns.lineplot(data=turbine_data.filter(pl.col("wind_direction").is_not_nan()).to_pandas(),
                         x='time', y='wind_direction', ax=ax2, label=f'{turbine_id} Wind Direction')
            sns.lineplot(data=turbine_data.filter(pl.col("power_output").is_not_nan()).to_pandas(),
                         x='time', y='power_output', ax=ax3, label=f'{turbine_id} Power Output')
        
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

    def plot_wind_speed_power(self, turbine_ids: list[str]) -> None:
        """_summary_

        """
        valid_turbines = self._get_valid_turbine_ids(turbine_ids=turbine_ids)
        
        if not valid_turbines:
            return
        
        _, ax = plt.subplots(1, 1, figsize=(12, 6))

        for turbine_id in valid_turbines:
            # turbine_data = self.df.loc[turbine_id]
            # turbine_data = self.df.filter(pl.col("turbine_id") == turbine_id).filter(~pl.all_horizontal(pl.col("wind_speed").is_null(), pl.col("power_output").is_null())).to_pandas()
            turbine_data = self.df.select(["wind_speed", "power_output", "turbine_id"])\
                .filter(pl.col("turbine_id") == turbine_id, 
                        pl.all_horizontal(pl.col("wind_speed", "power_output").is_not_nan())).to_pandas()
            sns.scatterplot(data=turbine_data, ax=ax, x='wind_speed', y='power_output')

        ax.set_xlabel('Wind Speed')
        ax.set_ylabel('Power Output')
        ax.set_title('Scatter Plot of Wind Speed vs Power Output')
        plt.show()

    def plot_wind_rose(self, turbine_ids: list[str] | str) -> None:
        """_summary_

        Args:
            wind_direction (float): _description_
            wind_speed (float): _description_
        """
        if turbine_ids == "all":
            plt.figure(figsize=(10, 10))
            ax = WindroseAxes.from_ax()
            wind_direction = self.df["wind_direction"]
            wind_speed = self.df["wind_speed"]
            ax.bar(wind_direction, wind_speed, normed=True, opening=0.8, edgecolor='white')
            ax.set_legend()
            plt.title('Wind Rose for all Turbines')
            plt.show()
        else:
            valid_turbines = self._get_valid_turbine_ids(turbine_ids=turbine_ids)
        
            if not valid_turbines:
                return 

            for turbine_id in valid_turbines:
                turbine_data = self.df.select(["turbine_id", "wind_speed", "wind_direction"])\
                    .filter(pl.col("turbine_id") == turbine_id,
                            pl.all_horizontal(pl.col("wind_speed", "wind_direction").is_not_nan())).to_pandas()
                plt.figure(figsize=(10, 10))
                ax = WindroseAxes.from_ax()
                wind_direction = turbine_data["wind_direction"]
                wind_speed = turbine_data["wind_speed"]
                ax.bar(wind_direction, wind_speed, normed=True, opening=0.8, edgecolor='white')
                ax.set_legend()
                plt.title(f'Wind Rose for Turbine {turbine_id}')
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

    def plot_correlation(self, features) -> None:
        """_summary_
        """
        _, ax = plt.subplots(1, 1, figsize=(12, 6))
        sns.heatmap(self.df.select(features).corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, ax=ax,
                    xticklabels=features, yticklabels=features)
        ax.set(title='Heatmap of Correlation Matrix')
        plt.show()

    def plot_boxplot_wind_speed_direction(self, turbine_ids: list[str]) -> None:
        """_summary_

        Args:
            turbine_id (str): _description_
        """
        valid_turbines = self._get_valid_turbine_ids(turbine_ids=turbine_ids)
        
        if not valid_turbines:
            return
        
        cols = ["hour", "wind_speed", "wind_direction", "turbine_id"] if "hour" in self.df.columns else ["time", "wind_speed", "wind_direction", "turbine_id"]

        for turbine_id in valid_turbines:
            # Select data for the specified turbine
            # turbine_data = self.df.loc[turbine_id]
            
            turbine_data = self.df.select(cols)\
                .filter(pl.col("turbine_id") == turbine_id, pl.any_horizontal(pl.col("wind_speed", "wind_direction").is_not_nan()))\
                    .to_pandas()
            
            if "hour" not in turbine_data.columns:
                # Extract hour from the time index
                # turbine_data = turbine_data.reset_index()
                # turbine_data = turbine_data.with_columns((pl.col('time').dt.hour).alias("hour"))
                turbine_data["hour"] = turbine_data["time"].dt.hour
            
            fig, ax = plt.subplots(2, 1, figsize=(12, 6))
            sns.boxplot(data=turbine_data, x='hour', y='wind_speed', ax=ax[0])
            sns.boxplot(data=turbine_data, x='hour', y='wind_direction', ax=ax[1])
            ax[0].set_title(f'Wind Speed Distribution by Hour for Turbine {turbine_id}')
            ax[1].set_title(f'Wind Direction Distribution by Hour for Turbine {turbine_id}')
            ax[0].set_xlabel("")
            ax[1].set_xlabel("Hour of Day")
            ax[0].set_ylabel("Wind Speed (m/s)")
            ax[1].set_ylabel("Wind Direction ($^\\circ$)")
            fig.tight_layout()
            plt.show()

    def plot_wind_speed_weibull(self, turbine_ids: list[str]) -> None:
        """_summary_

        """
        valid_turbines = self._get_valid_turbine_ids(turbine_ids=turbine_ids)
        
        if not valid_turbines:
            return
        
        for turbine_id in valid_turbines: 

            # Extract wind speed data
            wind_speed_data = self.df.select(["turbine_id", "wind_speed"])\
                .filter(pl.col("turbine_id") == turbine_id, pl.col("wind_speed").is_not_nan())\
                .select(["wind_speed"])

            # Fit Weibull distribution
            shape, loc, scale = stats.weibull_min.fit(wind_speed_data, floc=0)

            # Create a range of wind speeds for the fitted distribution
            x = np.linspace(0, wind_speed_data.max().item(), 100)
            y = stats.weibull_min.pdf(x, shape, loc, scale)

            # Plot
            plt.figure(figsize=(12, 6))
            sns.histplot(wind_speed_data, stat='density', kde=True, color='skyblue', label='Observed')
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
        fmodel = FlorisModel(self.farm_input_filepath)
        
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

if __name__ == "__main__":
    from wind_forecasting.preprocessing.data_loader import DataLoader
    from wind_forecasting.preprocessing.data_filter import DataFilter

    DATA_DIR = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data"
    FILE_SIGNATURE = "kp.turbine.z02.b0.20220301.*.wt073.nc"
    MULTIPROCESSOR = None

    data_loader = DataLoader(data_dir=DATA_DIR, file_signature=FILE_SIGNATURE, multiprocessor=MULTIPROCESSOR)
    df = data_loader.read_multi_netcdf()

    data_filter = DataFilter(raw_df=df)
    df = data_filter.filter_turbine_data(status_codes=[1], availability_codes=[100], include_nan=True)
    df = data_filter.resolve_missing_data(features=["wind_speed", "wind_direction", "power_output", "nacelle_direction"])

    TURBINE_INPUT_FILEPATH = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/inputs/ge_282_127.yaml"
    FARM_INPUT_FILEPATH = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/inputs/gch_KP_v4.yaml"
    data_inspector = DataInspector(df=df, turbine_input_filepath=TURBINE_INPUT_FILEPATH, farm_input_filepath=FARM_INPUT_FILEPATH)

    data_inspector.plot_wind_farm()
    data_inspector.plot_time_series(turbine_ids=["wt073"])
    data_inspector.plot_wind_speed_power(turbine_ids=["wt073"])
    data_inspector.plot_wind_speed_weibull(turbine_ids=["wt073"])
    data_inspector.plot_wind_rose(turbine_ids=["wt073"])
    data_inspector.plot_temperature_distribution()
    data_inspector.plot_correlation(["wind_speed", "wind_direction", "nacelle_direction", "power_output"])
    data_inspector.plot_boxplot_wind_speed(turbine_ids=["wt073"])

    print("Unique wind direction values:", df['wind_direction'].unique(), sep="\n")
    print("-"*100  )
    print("Unique turbine status values:", df['turbine_status'].unique(), sep="\n") # 1: running, 3: not running
    print("Unique turbine availability values:", df['turbine_availability'].unique(), sep="\n") # 100, 50 (partially available?)