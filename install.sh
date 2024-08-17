ssh ahenry@kestrel-gpu.hpc.nrel.gov
# https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent?platform=linux

ml mamba
mamba create -n wind_forecasting_env -y
mamba activate wind_forecasting_env

git clone --recurse-submodules https://github.com/achenry/wind-forecasting.git

cd wind-forecasting/models
conda install notebook jupyterlab nb_conda_kernels cython numpy pyyaml matplotlib numpy=1.26.4 seaborn netcdf4 opt_einsum wandb -c conda-forge -y
conda install pytorch torchvision torchaudio torchmetrics pytorch-forecasting lightning=2.3.3 cudatoolkit=11.7 -c pytorch -c nvidia

#python -m pip install -r ./spacetimeformer/requirements.txt
python ./spacetimeformer/setup.py develop
#python -m pip install -r ./Informer2020/requirements.txt
#python -m pip install -r ./Autoformer/requirements.txt

# for osx
#conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 -c pytorch -y
# for cuda 12.1 linux



# python -m pip install --no-binary datatable datatable
python -m pip install opencv-python performer-pytorch
# conda install pytorch::pytorch torchvision torchaudio -c pytorch



# git pull --recurse-submodules
#git clone https://github.com/achenry/spacetimeformer.git
#git clone https://github.com/achenry/Autoformer.git
#git clone https://github.com/achenry/Informer2020.git
