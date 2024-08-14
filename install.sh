conda create -n wind_forcasting_env
conda activate wind_forecasting_env

# for osx
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 -c pytorch
# for cuda 12.1 linux
conda install pytorch torchvision torchaudio cpuonly -c pytorch

conda install pytorch-forecasting -c pytorch -c conda-forge
conda install lightning -c conda-forge

pip install --no-binary datatable datatable
conda install pyyaml matplotlib numpy seaborn
# conda install pytorch::pytorch torchvision torchaudio -c pytorch
conda install -c conda-forge notebook jupyterlab nb_conda_kernels torchmetrics # omegaconf
pip install opencv-python performer-pytorch netCDF4 opt_einsum

git clone --recurse-submodules https://github.com/achenry/wind_forecasting.git
cd wind_forecasting/models/spacetimeformer
pip install -r ./spacetimeformer/requirements.txt
pip install -e ./spacetimeformer

cd ../Informer2020
pip install -r ./Informer2020/requirements.txt

cd ../Autoformer
pip install -r ./Autoformer/requirements.txt

# git pull --recurse-submodules
#git clone https://github.com/achenry/spacetimeformer.git
#git clone https://github.com/achenry/Autoformer.git
#git clone https://github.com/achenry/Informer2020.git
