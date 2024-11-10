ssh ahenry@kestrel-gpu.hpc.nrel.gov
# https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent?platform=linux

ml mamba
mamba create -n wind_forecasting_env -y
mamba activate wind_forecasting_env

git clone --recurse-submodules https://github.com/achenry/wind-forecasting.git

conda install notebook jupyterlab nb_conda_kernels cython pyyaml matplotlib numpy seaborn netcdf4 opt_einsum wandb scipy -c conda-forge -y
conda install pytorch torchvision torchaudio torchmetrics lightning cudatoolkit -c pytorch -c nvidia -y
# pytorch-forecasting

#python -m pip install -r ./spacetimeformer/requirements.txt
python wind_forecasting/models/spacetimeformer/setup.py develop
pip install wind_forecasting/preprocessing/OpenOA
# TODO write setup.py scripts for informer, autoformer
#python -m pip install -r ./Informer2020/requirements.txt
#python -m pip install -r ./Autoformer/requirements.txt

# for osx
#conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 -c pytorch -y
# for cuda 12.1 linux



# python -m pip install --no-binary datatable datatable
python -m pip install opencv-python performer-pytorch mpi4py polars floris
# conda install pytorch::pytorch torchvision torchaudio -c pytorch



# git pull --recurse-submodules
#git clone https://github.com/achenry/spacetimeformer.git
#git clone https://github.com/achenry/Autoformer.git
#git clone https://github.com/achenry/Informer2020.git
