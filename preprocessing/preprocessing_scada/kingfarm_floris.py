import numpy as np
import pandas as pd
# import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
from floris import FlorisModel
from floris.layout_visualization import plot_turbine_points, plot_turbine_labels, plot_turbine_rotors
from floris.flow_visualization import visualize_cut_plane
from scipy.interpolate import LinearNDInterpolator

def plot_wind_farm(floris_input_file, turbine_yaml_directory, lut_file, save_path): 

    df_lut = pd.read_csv(lut_file, index_col=0)    
    wind_speeds = df_lut.index.tolist()
    wind_directions = df_lut["wd"].tolist()
    turbulence_intensities = df_lut["ti"].tolist()
    
    MIN_WS = min(wind_speeds)
    MAX_WS = max(wind_speeds)

    # Initialize FLORIS Model
    fmodel = FlorisModel(floris_input_file)
    fmodel.set(
        turbine_library_path=turbine_yaml_directory, # Set the custom turbine library path
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        turbulence_intensities=turbulence_intensities
    )
    
    yaw_columns = [f'yaw_{i:03d}' for i in range(fmodel.n_turbines)]  # designation for the 88 turbines (yaw column names)
    yaw_angles = df_lut[yaw_columns].values
    print(yaw_columns)
    
    # interpolator for yaw angles
    points = np.column_stack((df_lut.index, df_lut['wd'], df_lut['ti']))
    interpolator = LinearNDInterpolator(points, yaw_angles)
    
    # INFO: Conditions for visualisation
    vis_ws = 12
    vis_wd = 270
    vis_ti = 0.05
    current_yaw = interpolator([vis_ws, vis_wd, vis_ti])
    print("wind_speeds: ", vis_ws, "wind_directions: ", vis_wd, "turbulence_intensities: ", vis_ti)
    print("current_yaw: ", current_yaw)
    
    # Set the conditions for the FLORIS model and apply the yaw angles
    fmodel.set(wind_speeds=[vis_ws], wind_directions=[vis_wd], turbulence_intensities=[vis_ti])
    fmodel.set_operation(yaw_angles=current_yaw)
    
    control_methods = {        
        'Consensus': [0, 1, 2, 5, 6, 7, 9, 11, 13, 15, 17, 18, 19, 20, 21, 22, 23, 25, 26, 71, 72, 80, 81, 83, 84, 86],
        'Wake Steering': [39, 40, 41, 42, 46, 47, 48, 53, 54, 55, 58, 65, 66, 67, 74],
    }
    control_methods['Reference'] = [i for i in range(fmodel.n_turbines) if i not in control_methods['Consensus'] and i not in control_methods['Wake Steering']]
    # mapping of turbine index to control method
    turbine_control = {}
    for method, turbines in control_methods.items():
        for turbine in turbines:
            turbine_control[turbine] = method  # 0 indexing
    colors = {'Reference': 'blue', 'Consensus': 'red', 'Wake Steering': 'green'}
    
    # Calculate the windflow (static)
    horizontal_plane = fmodel.calculate_horizontal_plane(x_resolution=200, y_resolution=100, height=90.0)

    fig, ax = plt.subplots(figsize=(16, 10))

    # Show the wake effect
    visualize_cut_plane(horizontal_plane, ax=ax, min_speed=MIN_WS, max_speed=MAX_WS, color_bar=False)

    plot_turbine_rotors(fmodel, ax=ax, yaw_angles=current_yaw[0])    
    plot_turbine_points(fmodel, ax=ax)
    plot_turbine_labels(fmodel, ax=ax)
    
    # Plot turbines with different colors based on control method
    for i, (x, y) in enumerate(zip(fmodel.layout_x, fmodel.layout_y)):
        method = turbine_control[i]
        color = colors[method]
        ax.plot(x, y, 'o', color=color, markersize=5, zorder=10)
        
    # Set axis limits to focus on the area of interest
    x_min, x_max = min(fmodel.layout_x), max(fmodel.layout_x)
    y_min, y_max = min(fmodel.layout_y), max(fmodel.layout_y)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_title(f"King Plains Wind Farm Layout (WS: {vis_ws:.1f} m/s, WD: {vis_wd:.1f}°, TI: {vis_ti:.2f})")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=method,
                                  markerfacecolor=color, markersize=10)
                       for method, color in colors.items()]
    ax.legend(handles=legend_elements, loc='upper right')
    
    filename = f"king_plains_layout_ws{vis_ws}_wd{vis_wd}_ti{vis_ti:.2f}.png"
    save_path = f"{save_path}/{filename}"

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

floris_input_file = r"./preprocessing/preprocessing_scada/floris_inputs/gch_KP_v4.yaml"
turbine_yaml_directory = r"./preprocessing/preprocessing_scada/floris_library"
lut_file = r"./preprocessing/preprocessing_scada/floris_inputs/KingPlains_WakeSteering_YawOffset_LookUpTable_v2_TurbineIndices 1.csv"
save_path = r"./preprocessing/preprocessing_scada/results/"

plot_wind_farm(floris_input_file, turbine_yaml_directory, lut_file, save_path)