import netCDF4 as nc
import os

def print_netcdf_structure(file_path):
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

            print("\nGroups:")
            def print_group(group, indent="  "):
                for group_name, group_obj in group.groups.items():
                    print(f"{indent}{group_name}:")
                    print(f"{indent}  Variables:")
                    for var_name, var in group_obj.variables.items():
                        print(f"{indent}    {var_name}:")
                        print(f"{indent}      Dimensions: {var.dimensions}")
                        print(f"{indent}      Shape: {var.shape}")
                        print(f"{indent}      Data type: {var.dtype}")
                        print(f"{indent}      Attributes:")
                        for attr in var.ncattrs():
                            print(f"{indent}        {attr}: {getattr(var, attr)}")
                    print_group(group_obj, indent + "  ")

            print_group(dataset)

    except Exception as e:
        print(f"Error reading NetCDF file: {e}")

if __name__ == "__main__":
    file_path = "../../awaken_data/kp.turbine.z01.b0.20201201.000000.wt001.nc"
    print_netcdf_structure(file_path)