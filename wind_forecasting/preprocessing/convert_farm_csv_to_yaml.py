import pandas as pd
import yaml

# Load the coordinates CSV from the SMARTEOLE-WFC-open-dataset repository
coords_df = pd.read_csv('examples/inputs/SMARTEOLE-WFC-open-dataset/SMARTEOLE_WakeSteering_Coordinates_staticData.csv')

# Extract the X and Y coordinates
layout_x = coords_df['X_RGF93'].tolist()
layout_y = coords_df['Y_RGF93'].tolist()

# Create the farm dictionary
farm_dict = {
    'name': 'SMARTEOLE_Farm',
    'description': 'Wind farm layout for SMARTEOLE data',
    'floris_version': 'v4',
    'logging': {
        'console': {'enable': True, 'level': 'WARNING'},
        'file': {'enable': False, 'level': 'WARNING'}
    },
    'solver': {
        'type': 'turbine_grid',
        'turbine_grid_points': 3
    },
    'farm': {
        'layout_x': layout_x,
        'layout_y': layout_y,
        'turbine_type': ['smarteole_turbine'] * len(layout_x),
    },
    'wake': {
        'model_strings': {
            'combination_model': 'sosfs',
            'deflection_model': 'gauss',
            'turbulence_model': 'crespo_hernandez',
            'velocity_model': 'gauss',
        },
        # Other wake parameters?
    },
    'flow_field': {
        'air_density': 1.225,
        'reference_wind_height': 90.0,  # Adjust based on turbine specifications
        'wind_shear': 0.12,
        'wind_veer': 0.0,
        'turbulence_intensity': 0.06,
        # Add other flow field parameters if necessary
    }
}

# Save the farm dictionary to a YAML file
with open('examples/inputs/smarteole_farm.yaml', 'w') as outfile:
    yaml.dump(farm_dict, outfile, default_flow_style=False)