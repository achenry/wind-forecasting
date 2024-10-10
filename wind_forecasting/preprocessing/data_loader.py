### This file contains class and method to: 
### - load the scada data, 
### - convert timestamps to datetime objects
### - convert circular measurements to sinusoidal measurements
### - normalize data
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import netCDF4 as nc
import polars as pl
import polars.selectors as cs
from pandas import to_datetime as pd_to_datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor

from openoa.utils import qa, plot

SECONDS_PER_MINUTE = np.float64(60)
SECONDS_PER_HOUR = SECONDS_PER_MINUTE * 60
SECONDS_PER_DAY = SECONDS_PER_HOUR * 24
SECONDS_PER_YEAR = SECONDS_PER_DAY * 365  # non-leap year, 365 days

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
                 multiprocessor: str | None = None, features: list[str] | None = None, dt: float | None = None):
        
        self.features = features or ["time", "turbine_id", "turbine_status", "wind_direction", "wind_speed", "power_output", "nacelle_direction"]

        # Get all the wts in the folder
        self.file_paths = glob.glob(f"{data_dir}/{file_signature}")
        if not self.file_paths:
            raise FileExistsError(f"File with signature {file_signature} in directory {data_dir} doesn't exist.")
        self.save_path = save_path
        self.multiprocessor = multiprocessor
        self.dt = dt
        # self.file_prefix = re.match(r"(.*)(?=\*)", file_signature)[0]

    def print_netcdf_structure(self, file_path) -> None:
        """_summary_

        Args:
            file_path (_type_): _description_
        """
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

    def read_multi_netcdf(self): # -> pl.LazyFrame | None:
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
                
                futures = [run_simulations_exec.submit(read_single_netcdf, data_loader=self, file_path=file_path) for file_path in self.file_paths]

                df_query = [fut.result() for fut in futures]
        else:
            df_query = []
            for file_path in self.file_paths:
                df_query.append(read_single_netcdf(self, file_path))

        if (self.multiprocessor == "mpi" and (comm_rank := MPI.COMM_WORLD.Get_rank()) == 0) \
            or (self.multiprocessor != "mpi") or (self.multiprocessor is None):
            if [df for df in df_query if df is not None]:
                df_query = pl.concat([df for df in df_query if df is not None]).sort(["turbine_id", "time"])
                df_query.collect(streaming=True).write_parquet(self.save_path)
                # print(f"Combined LazyFrame shape: {dfs.shape}")
                # print(f"Unique turbine IDs: {dfs['turbine_id'].unique()}")
                return df_query
            
            print("No data frames were created.")

    def convert_time_to_sin(self, df) -> pl.LazyFrame:
        """_summary_
            convert timestamp to cosine and sinusoidal components
        Returns:
            pl.LazyFrame: _description_
        """
        # TODO polarize
        # Convert Time to float64 for accurate division
        df['Time'] = df['Time'].astype(np.float64)

        # Create time features (Time column in seconds)
        # df['minute'] = (df['Time'] % SECONDS_PER_HOUR) / SECONDS_PER_MINUTE
        df['hour'] = (df['Time'] % SECONDS_PER_DAY) / SECONDS_PER_HOUR
        df['day'] = ((df['Time'] // SECONDS_PER_DAY) % 365).astype(int)
        df['year'] = (df['Time'] // SECONDS_PER_YEAR).astype(int)

        # Normalize time features using sin/cos for capturing cyclic patterns
        # df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
        # df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365)
        df['year_sin'] = np.sin(2 * np.pi * df['year'])
        df['year_cos'] = np.cos(2 * np.pi * df['year'])

        return df

    def reduce_features(self, df) -> pl.LazyFrame:
        """_summary_

        Returns:
            pl.LazyFrame: _description_
        """
        return df.select(self.features).filter(pl.any_horizontal(cs.numeric().is_not_null()))

    def resample(self, df) -> pl.LazyFrame:
        """_summary_

        Args:
            df (_type_): _description_
            dt (_type_): _description_

        Returns:
            pl.LazyFrame: _description_
        """
        # return df.sort(["turbine_id", "time"])\
        # .group_by_dynamic(index_column="time", group_by="turbine_id", every=f"{dt}s", period=f"{dt}s", closed="right")\
        # .agg(pl.all().first())
        return df.with_columns(pl.col("time").dt.round(f"{self.dt}s").alias("time"))\
                 .group_by("turbine_id", "time").agg(cs.numeric().drop_nulls().first()).sort(["turbine_id", "time"])
    
    def normalize_features(self, df) -> pl.LazyFrame:
        """_summary_
            use minmax scaling to normalize non-temporal features
        Returns:
            pl.LazyFrame: _description_
        """
        # Normalize non-time features
        # TODO polarize
        features_to_normalize = [col for col in df.columns
                                 if all(c not in col for c in ['Time', 'hour', 'day', 'year'])]
        df[features_to_normalize] = MinMaxScaler().fit_transform(df[features_to_normalize])
        return df
    
def read_single_netcdf(data_loader, file_path):
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
            time = pd_to_datetime(nc.num2date(times=time[:], units=time.units, calendar=time.calendar, only_use_cftime_datetimes=False, only_use_python_datetimes=True))
            
            # TODO add column mapping
            data = {
                'turbine_id': [os.path.basename(file_path).split('.')[-2]] * dataset.variables["date"].shape[0],
                'time': time,
                'turbine_status': dataset.variables['WTUR.TurSt'][:],
                'wind_direction': dataset.variables['WMET.HorWdDir'][:],
                'wind_speed': dataset.variables['WMET.HorWdSpd'][:],
                'power_output': dataset.variables['WTUR.W'][:],
                'nacelle_direction': dataset.variables['WNAC.Dir'][:]
            }
            
            df_query = pl.LazyFrame(data)\
                .group_by("turbine_id", "time").agg(
                # remove the rows with all nans (corresponding to rows where excluded columns would have had a value)
                # and bundle all values corresponding to identical time stamps together
                cs.numeric().drop_nans().first()
            )

            df_query = data_loader.reduce_features(df_query)

            if data_loader.dt is not None:
                df_query = data_loader.resample(df_query)

            # print(df_query.explain(streaming=True))

            del data
            # del df_query

            print(f"\nProcessed {file_path}") #, shape: {df.shape}")
            return df_query
    except Exception as e:
        print(f"\nError processing {file_path}: {e}")

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
        # data_loader.print_netcdf_structure(data_loader.file_paths[0])
    
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
        # df_query.sink_parquet(PL_SAVE_PATH)

    # print(df_query.explain(streaming=True))

    # from wind_forecasting.preprocessing.data_inspector import DataInspector
    # data_inspector = DataInspector(turbine_input_filepath=TURBINE_INPUT_FILEPATH, farm_input_filepath=FARM_INPUT_FILEPATH)
    # data_inspector.plot_wind_speed_power(df_query, turbine_ids=["wt073"])
    # data_inspector.plot_wind_speed_weibull(df_query, turbine_ids=["wt073"])
    # data_inspector.plot_wind_rose(df_query, turbine_ids=["wt073"])
    # data_inspector.plot_boxplot_wind_speed_direction(df_query, turbine_ids=["wt073"])
    # data_inspector.plot_time_series(df_query, turbine_ids=["wt073"])