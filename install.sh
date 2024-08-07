conda create -n wind_forcasting_env
conda activate wind_forecasting_env

# for osx
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 -c pytorch
conda install pytorch-forecasting -c pytorch -c conda-forge
conda install lightning -c conda-forge

# for cuda 12.1 linux
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install lightning -c conda-forge

pip install --no-binary datatable datatable
conda install pyyaml matplotlib numpy seaborn
conda install pytorch::pytorch torchvision torchaudio -c pytorch
conda install -c conda-forge notebook jupyterlab nb_conda_kernels torchmetrics # omegaconf
pip install opencv-python performer-pytorch netCDF4 opt_einsum

git clone https://github.com/QData/spacetimeformer.git
pip install -r ./spacetimeformer/requirements.txt
pip install -e ./spacetimeformer

git clone https://github.com/thuml/Autoformer.git
pip install -r ./Autoformer/requirements.txt

git clone https://github.com/zhouhaoyi/Informer2020.git
pip install -r ./Informer2020/requirements.txt