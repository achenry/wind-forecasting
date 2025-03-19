import pandas as pd
import yaml

# Load the power curve CSV
power_curve_df = pd.read_csv('examples/inputs/SMARTEOLE-WFC-open-dataset/SMARTEOLE_WakeSteering_GuaranteedPowerCurve_staticData.csv')

# Create the turbine dictionary (MM82 turbine)
turbine_dict = {
    'turbine_type': 'smarteole_turbine',
    'hub_height': 80.0,
    'rotor_diameter': 82.0,
    'TSR': 8.0,
    'pP': 1.88,
    'pT': 1.88,
    'generator_efficiency': 1.0,
    'power_thrust_table': {
        'wind_speed': power_curve_df['V'].tolist(),
        'power': power_curve_df['P'].tolist(),
        'thrust_coefficient': power_curve_df['Ct'].tolist(),
    },
}

# Save the turbine dictionary to a YAML file
with open('examples/inputs/smarteole_turbine.yaml', 'w') as outfile:
    yaml.dump(turbine_dict, outfile, default_flow_style=False)