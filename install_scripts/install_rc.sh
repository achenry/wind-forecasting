acompile
ml mambaforge
mamba create --prefix=/projects/aohe7145/software/anaconda/envs/wind_forecasting python=3.12 --y
mamba activate wind_forecasting

# mamba install conda-forge::cuda-version=12.4 nvidia/label/cuda-12.4.0::cuda-toolkit performer-pytorch pytorch torchvision torchaudio torchmetrics pytorch-cuda=12.4 lightning -c pytorch -c nvidia --y
mamba install polars windrose statsmodels scikit-learn jupyterlab nb_conda_kernels pyyaml matplotlib numpy seaborn opt_einsum netcdf4 scipy h5pyd wandb einops --y 
python -m pip install opencv-python floris performer-pytorch psutil memory_profiler optuna mysqlclient mysql-connector-python filterpy

brew install mysql  && brew services start mysql

git clone https://github.com/achenry/wind-forecasting.git
cd wind-forecasting
git checkout feature/spacetimeformer
python setup.py develop
cd ..

# cd wind-forecasting
# python setup.py develop
# cd ../gluonts
# ./dev_setup.sh
# python -m pip install -e .
# cd ../pytorch-transformer-ts
# python -m pip install ujson datasets xformers etsformer-pytorch reformer_pytorch pykeops apex # gluonts[torch]
# python -m pip install git+https://github.com/kashif/hopfield-layers@pytorch-2 git+https://github.com/microsoft/torchscale
# cd ..

git clone https://github.com/boujuan/pytorch-transformer-ts
cd pytorch-transformer-ts
git checkout feature/spacetimeformer
python -m pip install ujson datasets xformers etsformer-pytorch reformer_pytorch pykeops apex # gluonts[torch]
python -m pip install git+https://github.com/kashif/hopfield-layers@pytorch-2 git+https://github.com/microsoft/torchscale
cd ..

mamba install conda-forge::cuda-version=12.4 nvidia/label/cuda-12.4.0::cuda-toolkit performer-pytorch pytorch torchvision torchaudio torchmetrics pytorch-cuda=12.4 lightning -c pytorch -c nvidia --y

git clone https://github.com/achenry/gluonts
cd gluonts
git checkout mv_prob
./dev_setup.sh
python -m pip install -e .
cd ..

git clone https://github.com/achenry/floris.git
cd floris
git checkout feature/mpc
pip install -e .
cd ..

git clone https://github.com/achenry/wind-hybrid-open-controller.git
cd wind-hybrid-open-controller
git checkout feature/wind_preview
pip install -e .
cd ..

# git clone https://github.com/achenry/OpenOA
# cd OpenOA
# pip install -e .
# cd ..
