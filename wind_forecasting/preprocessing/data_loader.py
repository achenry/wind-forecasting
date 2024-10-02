### This file contains class and method to: 
### - load the scada data, 
### - convert timestamps to datetime objects
### - convert circular measurements to sinusoidal measurements
### - normalize data
import glob
import os
import re
import netCDF4 as nc
import polars as pl
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor
from concurrent.futures import ProcessPoolExecutor

SECONDS_PER_MINUTE = np.float64(60)
SECONDS_PER_HOUR = SECONDS_PER_MINUTE * 60
SECONDS_PER_DAY = SECONDS_PER_HOUR * 24
SECONDS_PER_YEAR = SECONDS_PER_DAY * 365  # non-leap year, 365 days

# RUN_ONCE = 

class DataLoader:
    """_summary_
       - load the scada data, 
       - convert timestamps to datetime objects
       - convert circular measurements to sinusoidal measurements
       - normalize data 
    """
    def __init__(self, data_dir: str = r"/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data", file_signature: str = r"kp.turbine.z02.b0.*.wt*.nc", 
                 multiprocessor: str | None=None, features=None):
        
        if features is None:
            features = ["time", "turbine_id", "turbine_status", "turbine_availability", "wind_direction", "wind_speed", "power_output"]

        # Get all the wts in the folder
        self.file_paths = glob.glob(f"{data_dir}/{file_signature}")
        self.multiprocessor = multiprocessor
        self.file_prefix = re.match(r"(.*)(?=\*)", file_signature)[0]
        self.df = None

    def print_netcdf_structure(self, file_path) -> None:
        try:
            with nc.Dataset(file_path, 'r') as dataset:
                print(f"NetCDF File: {os.path.basename(file_path)}")
                print("\nGlobal Attributes:")
                for attr in dataset.ncattrs():
                    print(f"  {attr}: {getattr(dataset, attr)}")

                print("\nDimensions:")
                for dim_name, dim in dataset.dimensions.items():
                    print(f"  {dim_name}: {len(dim)}")

                print("\nVariables:")
                for var_name, var in dataset.variables.items():
                    print(f"  {var_name}:")
                    print(f"    Dimensions: {var.dimensions}")
                    print(f"    Shape: {var.shape}")
                    print(f"    Data type: {var.dtype}")
                    print("    Attributes:")
                    for attr in var.ncattrs():
                        print(f"      {attr}: {getattr(var, attr)}")

        except Exception as e:
            print(f"Error reading NetCDF file: {e}")
    
    def read_multi_netcdf(self): # -> pl.DataFrame | None:
        dfs  = []
        if self.multiprocessor is not None:
            if self.multiprocessor == "mpi":
                comm_size = MPI.COMM_WORLD.Get_size()
                executor = MPICommExecutor(MPI.COMM_WORLD, root=0)
            elif self.multiprocessor == "cf":
                executor = ProcessPoolExecutor()

            with executor as run_simulations_exec:
                if self.multiprocessor == "mpi":
                    run_simulations_exec.max_workers = comm_size
                
                futures = [run_simulations_exec.submit(read_single_netcdf, file_path=file_path) for file_path in self.file_paths]

                dfs = [fut.result() for fut in futures]
        else:
            for file_path in self.file_paths:
                dfs.append(read_single_netcdf(file_path))

        if (self.multiprocessor == "mpi" and (comm_rank := MPI.COMM_WORLD.Get_rank()) == 0) \
            or (self.multiprocessor != "mpi") or (self.multiprocessor is None):
            if dfs:
                combined_df = pl.concat([df for df in dfs if df is not None])
                # combined_df.set_index(['turbine_id', 'time'], inplace=True)
                print(f"Combined DataFrame shape: {combined_df.shape}")
                print(f"Unique turbine IDs: {combined_df['turbine_id'].unique()}")
                self.df = combined_df
                return combined_df
            
            print("No data frames were created.")

    def convert_time_to_sin(self) -> pl.DataFrame:
        """_summary_
            convert timestamp to cosine and sinusoidal components
        Returns:
            pl.DataFrame: _description_
        """
        # Convert Time to float64 for accurate division
        self.df['Time'] = self.df['Time'].astype(np.float64)

        # Create time features (Time column in seconds)
        # self.df['minute'] = (self.df['Time'] % SECONDS_PER_HOUR) / SECONDS_PER_MINUTE
        self.df['hour'] = (self.df['Time'] % SECONDS_PER_DAY) / SECONDS_PER_HOUR
        self.df['day'] = ((self.df['Time'] // SECONDS_PER_DAY) % 365).astype(int)
        self.df['year'] = (self.df['Time'] // SECONDS_PER_YEAR).astype(int)

        # Normalize time features using sin/cos for capturing cyclic patterns
        # self.df['minute_sin'] = np.sin(2 * np.pi * self.df['minute'] / 60)
        # self.df['minute_cos'] = np.cos(2 * np.pi * self.df['minute'] / 60)
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)
        self.df['day_sin'] = np.sin(2 * np.pi * self.df['day'] / 365)
        self.df['day_cos'] = np.cos(2 * np.pi * self.df['day'] / 365)
        self.df['year_sin'] = np.sin(2 * np.pi * self.df['year'])
        self.df['year_cos'] = np.cos(2 * np.pi * self.df['year'])

        return self.df

    def reduce_features(self) -> pl.DataFrame:
        self.df = self.df.select(self.features)
        return self.df

    def normalize_features(self) -> pl.DataFrame:
        """_summary_
            use minmax scaling to normalize non-temporal features
        Returns:
            pl.DataFrame: _description_
        """
        # Normalize non-time features
        features_to_normalize = [col for col in self.df.columns
                                 if all(c not in col for c in ['Time', 'hour', 'day', 'year'])]
        self.df[features_to_normalize] = MinMaxScaler().fit_transform(self.df[features_to_normalize])
        return self.df
    
def read_single_netcdf(file_path):
    """_summary_

    Args:
        file_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    try:
        with nc.Dataset(file_path, 'r') as dataset:
            # Convert time to datetime

            # if "date" in dataset.variables:
            time = dataset.variables['date']
            time = pd.to_datetime(nc.num2date(times=time[:], units=time.units, calendar=time.calendar, only_use_cftime_datetimes=False, only_use_python_datetimes=True))
            # else:
            #     # date = re.findall(f"(?<={file_prefix})(\d{8})(=?.)", os.path.basename(file_path))
            #     start_date = datetime.strptime(re.findall(r"(\d{8})", os.path.basename(file_path))[0], "%Y%m%d")
            #     time = [start_date + datetime.timedelta(seconds=self.dt * i) for i in range(0, dataset.dimensions["index"].size)]

            # time = [datetime.strptime(t.strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S') for t in time]
            # time = datetime.strptime([t.strftime('%Y-%m-%d %H:%M:%S') for t in time])

            data = {
                'time': time,
                'turbine_id': [os.path.basename(file_path).split('.')[-2]] * dataset.variables["date"].shape[0],
                'generator_current_phase_1': dataset.variables['WCNV.GnA1'][:],
                'generator_current_phase_2': dataset.variables['WCNV.GnA2'][:],
                'generator_current_phase_3': dataset.variables['WCNV.GnA3'][:],
                'generator_voltage_phase_1': dataset.variables['WCNV.GnPNV1'][:],
                'generator_voltage_phase_2': dataset.variables['WCNV.GnPNV2'][:],
                'generator_voltage_phase_3': dataset.variables['WCNV.GnPNV3'][:],
                'power_output': dataset.variables['WTUR.W'][:],
                'turbine_availability': dataset.variables['WAVL.TurAvl'][:],
                'generator_bearing_de_temp': dataset.variables['WGEN.BrgDETmp'][:],
                'generator_bearing_nde_temp': dataset.variables['WGEN.BrgNDETmp'][:],
                'generator_inlet_temp': dataset.variables['WGEN.InLetTmp'][:],
                'generator_stator_temp_1': dataset.variables['WGEN.SttTmp1'][:],
                'generator_stator_temp_2': dataset.variables['WGEN.SttTmp2'][:],
                'generator_rotor_speed': dataset.variables['WGEN.RotSpd'][:],
                'nacelle_direction': dataset.variables['WNAC.Dir'][:],
                'nacelle_temperature': dataset.variables['WNAC.Tmp'][:],
                'ambient_temperature': dataset.variables['WMET.EnvTmp'][:],
                'blade_pitch_angle_1': dataset.variables['WROT.BlPthAngVal1'][:],
                'blade_pitch_angle_2': dataset.variables['WROT.BlPthAngVal2'][:],
                'blade_pitch_angle_3': dataset.variables['WROT.BlPthAngVal3'][:],
                'rotor_speed': dataset.variables['WROT.RotSpd'][:],
                'turbine_status': dataset.variables['WTUR.TurSt'][:],
                'wind_direction': dataset.variables['WMET.HorWdDir'][:],
                'wind_speed': dataset.variables['WMET.HorWdSpd'][:]
            }
            
            df = pl.DataFrame(data).sort(by="time")
            # df["time"].str.to_datetime(time_unit="ms") #dataset.variables['date'].units)
            
            # .sort(by="time")
            # 

            # Group by timestamp and aggregate JUAN what did you want to do here?
            # df = df.group_by('time').agg(*[pl.col(c).first() for c in df.columns if c != "time"])
                
            #     {   # Group by timestamp and aggregate
            #     col: 'first' for col in df.columns if col != 'time'
            # }).reset_index()            
            print(f"Processed {file_path}, shape: {df.shape}")
            return df
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    data_dir = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data"
    file_signature = "kp.turbine.z02.b0.202203*.wt073.nc"
    multiprocessor = None

    data_loader = DataLoader(data_dir=data_dir, file_signature=file_signature, multiprocessor=multiprocessor)
    data_loader.print_netcdf_structure(data_loader.file_paths[0])
    df = data_loader.read_multi_netcdf()