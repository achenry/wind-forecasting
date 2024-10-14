### This file contains class and method to: 
### - load the scada data, 
### - convert timestamps to datetime objects
### - convert circular measurements to sinusoidal measurements
### - normalize data

import glob
import os
import logging
from concurrent.futures import ProcessPoolExecutor

import netCDF4 as nc
import polars as pl
import polars.selectors as cs
from pandas import to_datetime as pd_to_datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor

SECONDS_PER_MINUTE = np.float64(60)
SECONDS_PER_HOUR = SECONDS_PER_MINUTE * 60
SECONDS_PER_DAY = SECONDS_PER_HOUR * 24
SECONDS_PER_YEAR = SECONDS_PER_DAY * 365  # non-leap year, 365 days

# INFO: @Juan 10/02/24 Set Looging up
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    """_summary_
       - load the scada data, 
       - convert timestamps to datetime objects
       - convert circular measurements to sinusoidal measurements
       - normalize data 
    """
    def __init__(self, data_dir: str = r"/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data", 
                 file_signature: str = r"kp.turbine.z02.b0.*.wt*.nc", 
                 save_path: str = r"/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data/kp.turbine.zo2.b0.raw.parquet",
                 multiprocessor: str | None = None,
                 features: list[str] = None,
                 dt: int | None = 5):
        
        if features is None:
            self.features = ["time", "turbine_id", "turbine_status", "turbine_availability", "wind_direction", "wind_speed", "power_output"]
        else:
            self.features = features

        # Get all the wts in the folder
        self.file_paths = glob.glob(f"{data_dir}/{file_signature}")
        if not self.file_paths:
            raise FileExistsError(f"File with signature {file_signature} in directory {data_dir} doesn't exist.")

        self.multiprocessor = multiprocessor
        self.dt = dt
        self.save_path = save_path

    def print_netcdf_structure(self, file_path) -> None: #INFO: @Juan 10/02/24 Changed print to logging
        try:
            with nc.Dataset(file_path, 'r') as dataset:
                logging.info(f"NetCDF File: {os.path.basename(file_path)}")
                logging.info("\nGlobal Attributes:")
                for attr in dataset.ncattrs():
                    logging.info(f"  {attr}: {getattr(dataset, attr)}")

                logging.info("\nDimensions:")
                for dim_name, dim in dataset.dimensions.items():
                    logging.info(f"  {dim_name}: {len(dim)}")

                logging.info("\nVariables:")
                for var_name, var in dataset.variables.items():
                    logging.info(f"  {var_name}:")
                    logging.info(f"    Dimensions: {var.dimensions}")
                    logging.info(f"    Shape: {var.shape}")
                    logging.info(f"    Data type: {var.dtype}")
                    logging.info("    Attributes:")
                    for attr in var.ncattrs():
                        logging.info(f"      {attr}: {getattr(var, attr)}")

        except Exception as e:
            logging.error(f"Error reading NetCDF file: {e}")
    
    def read_multi_netcdf(self) -> pl.LazyFrame | None:
        """_summary_

        Returns:
            _type_: _description_
        """
        
        if self.multiprocessor is not None:
            if self.multiprocessor == "mpi":
                comm_size = MPI.COMM_WORLD.Get_size()
                executor = MPICommExecutor(MPI.COMM_WORLD, root=0)
            elif self.multiprocessor == "cf":
                executor = ProcessPoolExecutor()

            with executor as run_simulations_exec:
                if self.multiprocessor == "mpi":
                    run_simulations_exec.max_workers = comm_size
                
                # INFO: @Juan 10/02/24 Turned read_single_netcdf into a private method
                futures = [run_simulations_exec.submit(self._read_single_netcdf, file_path=file_path) for file_path in self.file_paths]
                df_query = [fut.result() for fut in futures]
        else:
            df_query  = []
            for file_path in self.file_paths:
                df_query.append(self._read_single_netcdf(file_path))

        if (self.multiprocessor == "mpi" and (comm_rank := MPI.COMM_WORLD.Get_rank()) == 0) \
            or (self.multiprocessor != "mpi") or (self.multiprocessor is None):
            if [df for df in df_query if df is not None]:
                df_query = pl.concat([df for df in df_query if df is not None]).sort(["turbine_id", "time"])
                df_query.collect(streaming=True).write_parquet(self.save_path)
                # combined_df.set_index(['turbine_id', 'time'], inplace=True)
                # logging.info(f"Combined DataFrame shape: {combined_df.shape}")
                # logging.info(f"Unique turbine IDs: {combined_df['turbine_id'].unique()}")
                return df_query
            
            logging.warning("No data frames were created.")
            return None

    # INFO: @Juan 10/02/24 Revamped this method to use Polars functions consistently, vectorized where possible, and using type casting for consistency and performance enhancements.
    def convert_time_to_sin(self, df) -> pl.LazyFrame:
        """_summary_
            convert timestamp to cosine and sinusoidal components
        Returns:
            pl.LaxyFrame: _description_
        """
        if df is None:
            raise ValueError("Data not loaded > call read_multi_netcdf() first.")
        
        # Convert Time to float64 for accurate division and create time features (hour, day, year) using Polars vectorized operations
        df = df.with_columns([
            pl.col('Time').cast(pl.Float64),
            (pl.col('Time') % SECONDS_PER_DAY / SECONDS_PER_HOUR).alias('hour'),
            ((pl.col('Time') // SECONDS_PER_DAY) % 365).cast(pl.Int32).alias('day'),
            (pl.col('Time') // SECONDS_PER_YEAR).cast(pl.Int32).alias('year'),
        ])

        # Normalize time features using sin/cos for capturing cyclic patterns using Polars vectorized operations
        df = df.with_columns([
            pl.sin(2 * np.pi * pl.col('hour') / 24).alias('hour_sin'),
            pl.cos(2 * np.pi * pl.col('hour') / 24).alias('hour_cos'),
            pl.sin(2 * np.pi * pl.col('day') / 365).alias('day_sin'),
            pl.cos(2 * np.pi * pl.col('day') / 365).alias('day_cos'),
            pl.sin(2 * np.pi * pl.col('year')).alias('year_sin'),
            pl.cos(2 * np.pi * pl.col('year')).alias('year_cos'),
        ])

        return df

    def reduce_features(self, df) -> pl.LazyFrame:
        """_summary_

        Returns:
            pl.LazyFrame: _description_
        """
        return df.select(self.features).filter(pl.any_horizontal(cs.numeric().is_not_null()))

    def resample(self, df) -> pl.LazyFrame:
        return df.with_columns(pl.col("time").dt.round(f"{self.dt}s").alias("time"))\
                 .group_by("turbine_id", "time").agg(cs.numeric().drop_nulls().first()).sort(["turbine_id", "time"])

    def normalize_features(self, df) -> pl.LazyFrame:
        """_summary_
            use minmax scaling to normalize non-temporal features
        Returns:
            pl.LazyFrame: _description_
        """
        if self.df is None:
            raise ValueError("Data not loaded > call read_multi_netcdf() first.")
        
        # Normalize non-time features
        features_to_normalize = [col for col in df.columns
                                 if all(c not in col for c in ['Time', 'hour', 'day', 'year'])]
        
        # INFO: @Juan 10/02/24 Explicitly convert to numpy and back to DF to ensure compatibility with MinMaxScaler
        normalized_data = MinMaxScaler().fit_transform(self.df.select(features_to_normalize).to_numpy())
        # INFO: @Juan 10/02/24 Hstack (grow horizontally) the normalized data df back to the original DF
        return df.drop(features_to_normalize).hstack(pl.DataFrame(normalized_data, schema=features_to_normalize))
    
    def create_sequences(self, df, target_turbine: str, 
                         features: list[str] | None = None, 
                         sequence_length: int = 600, 
                         prediction_horizon: int = 240) -> tuple[np.ndarray, np.ndarray, list[str], int, int]:
        
        features = [col for col in df.columns if col not in [f'TurbineWindMag_{target_turbine}_u', f'TurbineWindMag_{target_turbine}_v']]
        y_columns = [f'TurbineWindMag_{target_turbine}_u', f'TurbineWindMag_{target_turbine}_v']
        
        X_data = df.select(features).collect(streaming=True).to_numpy()
        y_data = df.select(y_columns).collect(streaming=True).to_numpy()
        
        X = np.array([X_data[i:i + sequence_length] for i in range(len(X_data) - sequence_length - prediction_horizon + 1)])
        y = np.array([y_data[i:i + prediction_horizon] for i in range(sequence_length, len(y_data) - prediction_horizon + 1)])
        
        return X, y, features, sequence_length, prediction_horizon
    
    def _read_single_netcdf(self, file_path: str) -> pl.DataFrame:
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
                # INFO: @Juan 10/02/24 Changed pandas to polars for time conversion pl.from_numpy
                # BUG: Time is not being converted correctly, check if this new implementation with polars is working!
                # time = pl.from_numpy(nc.num2date(times=time[:], units=time.units, calendar=time.calendar, only_use_cftime_datetimes=False, only_use_python_datetimes=True))
                time = pd_to_datetime(nc.num2date(times=time[:], units=time.units, calendar=time.calendar, only_use_cftime_datetimes=False, only_use_python_datetimes=True))

                # NOTE: Future work: Have config file to define turbine feature names in data etc
                data = {
                    'turbine_id': [os.path.basename(file_path).split('.')[-2]] * dataset.variables["date"].shape[0],
                    'time': time,
                    'turbine_status': dataset.variables['WTUR.TurSt'][:],
                    # 'turbine_availability': dataset.variables['WAVL.TurAvl'][:],
                    'wind_direction': dataset.variables['WMET.HorWdDir'][:],
                    'wind_speed': dataset.variables['WMET.HorWdSpd'][:],
                    # 'generator_current_phase_1': dataset.variables['WCNV.GnA1'][:],
                    # 'generator_current_phase_2': dataset.variables['WCNV.GnA2'][:],
                    # 'generator_current_phase_3': dataset.variables['WCNV.GnA3'][:],
                    # 'generator_voltage_phase_1': dataset.variables['WCNV.GnPNV1'][:],
                    # 'generator_voltage_phase_2': dataset.variables['WCNV.GnPNV2'][:],
                    # 'generator_voltage_phase_3': dataset.variables['WCNV.GnPNV3'][:],
                    'power_output': dataset.variables['WTUR.W'][:],
                    
                    # 'generator_bearing_de_temp': dataset.variables['WGEN.BrgDETmp'][:],
                    # 'generator_bearing_nde_temp': dataset.variables['WGEN.BrgNDETmp'][:],
                    # 'generator_inlet_temp': dataset.variables['WGEN.InLetTmp'][:],
                    # 'generator_stator_temp_1': dataset.variables['WGEN.SttTmp1'][:],
                    # 'generator_stator_temp_2': dataset.variables['WGEN.SttTmp2'][:],
                    # 'generator_rotor_speed': dataset.variables['WGEN.RotSpd'][:],
                    'nacelle_direction': dataset.variables['WNAC.Dir'][:],
                    # 'nacelle_temperature': dataset.variables['WNAC.Tmp'][:],
                    # 'ambient_temperature': dataset.variables['WMET.EnvTmp'][:],
                    # 'blade_pitch_angle_1': dataset.variables['WROT.BlPthAngVal1'][:],
                    # 'blade_pitch_angle_2': dataset.variables['WROT.BlPthAngVal2'][:],
                    # 'blade_pitch_angle_3': dataset.variables['WROT.BlPthAngVal3'][:],
                    # 'rotor_speed': dataset.variables['WROT.RotSpd'][:],
                }
                
                df_query = pl.LazyFrame(data).group_by("turbine_id", "time").agg(
                    # remove the rows with all nans (corresponding to rows where excluded columns would have had a value)
                    # and bundle all values corresponding to identical time stamps together
                    cs.numeric().drop_nans().first()
                ).fill_nan(None)
                df_query = self.reduce_features(df_query)
                if self.dt is not None:
                    df_query = self.resample(df_query)

                del data
                
                logging.info(f"Processed {file_path}") #, shape: {df.shape}")
                return df_query
            
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            return None

if __name__ == "__main__":
    from sys import platform
    
    if platform == "darwin":
        DATA_DIR = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data"
        PL_SAVE_PATH = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data/kp.turbine.zo2.b0.raw.parquet"
        FILE_SIGNATURE = "kp.turbine.z02.b0.20220301.*.*.nc"
        MULTIPROCESSOR = "cf"
        TURBINE_INPUT_FILEPATH = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/inputs/ge_282_127.yaml"
        FARM_INPUT_FILEPATH = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/inputs/gch_KP_v4.yaml"
    elif platform == "linux":
        DATA_DIR = "/pl/active/paolab/awaken_data/kp.turbine.z02.b0/"
        PL_SAVE_PATH = "/scratch/alpine/aohe7145/awaken_data/kp.turbine.zo2.b0.raw.parquet"
        FILE_SIGNATURE = "kp.turbine.z02.b0.*.*.*.nc"
        MULTIPROCESSOR = "mpi"
        TURBINE_INPUT_FILEPATH = "/projects/aohe7145/toolboxes/wind-forecasting/examples/inputs/ge_282_127.yaml"
        FARM_INPUT_FILEPATH = "/projects/aohe7145/toolboxes/wind-forecasting/examples/inputs/gch_KP_v4.yaml"
    
    DT = 5
    RUN_ONCE = (MULTIPROCESSOR == "mpi" and (comm_rank := MPI.COMM_WORLD.Get_rank()) == 0) or (MULTIPROCESSOR != "mpi") or (MULTIPROCESSOR is None)

    if RUN_ONCE:
        data_loader = DataLoader(data_dir=DATA_DIR, file_signature=FILE_SIGNATURE, save_path=PL_SAVE_PATH,
                                 multiprocessor=MULTIPROCESSOR, dt=DT,
                         features=["time", "turbine_id", "turbine_status", "wind_direction", "wind_speed", "power_output", "nacelle_direction"])
        
        if os.path.exists(data_loader.save_path):
            # Note that the order of the columns in the provided schema must match the order of the columns in the CSV being read.
            schema = pl.Schema({"turbine_id": pl.String(),
                                "time": pl.Datetime(time_unit="ms"),
                                "turbine_status": pl.Float64,
                                "wind_direction": pl.Float64,
                                "wind_speed": pl.Float64,
                                "power_output": pl.Float64,
                                "nacelle_direction": pl.Float64,
                            })
            
            df_query = pl.scan_parquet(source=data_loader.save_path, hive_schema=schema)
        else:
            df_query = data_loader.read_multi_netcdf()