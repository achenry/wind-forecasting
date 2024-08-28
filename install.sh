# Instructions for installing environment on the CU Boulder Alpine cluster (for jubo7621)

ssh jubo7621@login.rc.colorado.edu
# https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent?platform=linux

acompile # log in the compile node
ml mambaforge # load mamba
mamba create -n wind_forecasting_env -y
mamba activate wind_forecasting_env

cd /projects/$USER/
git clone --recurse-submodules https://github.com/achenry/wind-forecasting.git

cd wind-forecasting/wind-forecasting/models
mamba install notebook jupyterlab nb_conda_kernels cython numpy pyyaml matplotlib numpy=1.26.4 seaborn netcdf4 opt_einsum wandb -c conda-forge -y
mamba install pytorch torchvision torchaudio torchmetrics pytorch-forecasting lightning=2.3.3 cudatoolkit=11.7 -c pytorch -c nvidia

#python -m pip install -r ./spacetimeformer/requirements.txt
python ./spacetimeformer/setup.py develop
#python -m pip install -r ./Informer2020/requirements.txt
#python -m pip install -r ./Autoformer/requirements.txt

# python -m pip install --no-binary datatable datatable
python -m pip install opencv-python performer-pytorch
# mamba install pytorch::pytorch torchvision torchaudio -c pytorch


# git pull --recurse-submodules
#git clone https://github.com/achenry/spacetimeformer.git
#git clone https://github.com/achenry/Autoformer.git
#git clone https://github.com/achenry/Informer2020.git
