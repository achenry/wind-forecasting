import os
import requests
import numpy as np
from zipfile import ZipFile

def download_smarteole_data():
    """Function that downloads 1-minute SCADA data from the SMARTEOLE wake
    steering experiment at the Sole du Moulin Vieux wind plant along with
    static wind plant and turbine data.
    """

    r = requests.get(r"https://zenodo.org/api/records/7342466")
    r_json = r.json()
    filesize = r_json["files"][0]["size"] / (1024 * 1024)
    filename = os.path.join("inputs", r_json["files"][0]["key"])
    result = requests.get(r_json["files"][0]["links"]["self"], stream=True)
    os.makedirs("inputs", exist_ok=True)
    if not os.path.exists(filename):
        print("SMARTEOLE data not found locally. Beginning file download from Zenodo...")
        chunk_number = 0
        with open(filename, "wb") as f:
            for chunk in result.iter_content(chunk_size=1024 * 1024):
                chunk_number = chunk_number + 1
                print(f"{chunk_number} out of {int(np.ceil(filesize))} MB downloaded", end="\r")
                f.write(chunk)
    else:
        print("SMARTEOLE data found locally.")

    if not os.path.exists(filename[:-4]):
        print("Extracting SMARTEOLE zip file")
        with ZipFile(filename) as zipfile:
            zipfile.extractall("inputs")
    else:
        print("SMARTEOLE data already extracted locally.")

    print("\nList of SMARTEOLE files:")
    for f in os.listdir(os.path.join(filename[:-4])):
        print(f)

# download data from Zenodo
download_smarteole_data()