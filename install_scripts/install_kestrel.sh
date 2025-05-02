# TODO fix clashing versions below e.g numpy scipy 
ssh ahenry@kestrel-gpu.hpc.nrel.gov
# https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent?platform=linux

ml mamba
mamba create --prefix=/projects/ssc/ahenry/conda/envs/wind_forecasting_env python==3.12.9
mamba activate wind_forecasting_env

## FOR PREPROCESSING ONLY
# mkdir /home/ahenry/.conda
# chmod a=rx /home/ahenry/.conda
# cd $LARGE_STORAGE
# mkdir conda_pkgs
# conda config --add pkgs_dirs /srv/data/nfs/ahenry/conda_pkgs 
# mkdir conda_envs
# conda config --append envs_dirs /srv/data/nfs/ahenry/conda_envs
# conda create --prefix=/srv/data/nfs/ahenry/conda_envs/wind_forecasting_preprocessing --y
# conda activate /srv/data/nfs/ahenry/conda_envs/wind_forecasting_preprocessing
# mkdir wind_forecasting_env && cd wind_forecasting_env && mkdir bin

git clone https://github.com/achenry/OpenOA
cd OpenOA
pip install --target $LARGE_STORAGE/ahenry/wind_forecasting_env/bin .
cd ..

git clone https://github.com/achenry/wind-forecasting.git
cd wind-forecasting
git checkout feature/spacetimeformer
python setup.py develop --prefix=$LARGE_STORAGE/ahenry/wind_forecasting_env/bin
cd ..

conda install statsmodels pyyaml matplotlib numpy seaborn netcdf4 --y 
pip install --target $LARGE_STORAGE/ahenry/wind_forecasting_env/bin floris polars windrose psutil

## END FOR PREPROCESSING ONLY

# rm -rf /projects/ssc/ahenry/conda/envs/wind_forecasting
# rm -rf /home/ahenry/.conda-pkgs/cache
# FOR PREPROCESSING AND RUNNING MODEL


module load PrgEnv-intel
mamba install mpi4py 
pip install plotly memory_profiler optuna optuna-integration optuna-dashboard filterpy # openmpi impi_rt opencv-python 
mamba install conda-forge::cuda-version=12.4 nvidia/label/cuda-12.4.0::cuda-toolkit performer-pytorch pytorch torchvision torchaudio torchmetrics pytorch-cuda=12.4 lightning -c pytorch -c nvidia
mamba install mysqlclient mysql-connector-python polars windrose statsmodels scikit-learn nb_conda_kernels pyyaml matplotlib numpy seaborn netcdf4 scipy h5pyd pyarrow wandb einops # opt_einsum

## FOR WANDB ONLY
API_FILE="../wind_forecasting/run_scripts/.wandb_api_key"
if [ -f "${API_FILE}" ]; then
  source "${API_FILE}"
else
  echo "ERROR: WANDB API‑key file not found at ${API_FILE}" >&2
  exit 1
fi
wandb login
## END WANDB

brew install mysql  && brew services start mysql

git clone https://github.com/achenry/wind-forecasting.git
cd wind_forecasting
git checkout feature/spacetimeformer
python setup.py develop
cd ..

git clone https://github.com/boujuan/pytorch-transformer-ts
cd pytorch-transformer-ts
git checkout feature/spacetimeformer
python setup.py develop
# pip install ujson datasets xformers etsformer-pytorch reformer_pytorch pykeops apex # gluonts[torch]
# pip install git+https://github.com/kashif/hopfield-layers@pytorch-2 git+https://github.com/microsoft/torchscale
cd ..

git clone https://github.com/achenry/gluonts
cd gluonts
git checkout mv_prob
pip install -e .
cd ..

git clone https://github.com/achenry/OpenOA
cd OpenOA
pip install .
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


# salloc --account=ssc --time=01:00:00 --mem-per-cpu=64G --gpus=2 --ntasks-per-node=2 --partition=debug

# vim /projects/ssc/ahenry/conda/envs/wind_forecasting/conda-meta/pinned
# performer-pytorch==1.1.4=pyhd8ed1ab_0  # conda-forge
# pytorch==2.5.1=py3.12_cuda12.4_cudnn9.1.0_0 #  pytorch
# pytorch-cuda==12.4=hc786d27_7  # pytorch
# pytorch-lightning==2.4.0=pyhd8ed1ab_0  # conda-forge
# pytorch-mutex==1.0=cuda  # pytorch
# torchaudio==2.5.1=py312_cu124  # pytorch
# torchmetrics==1.5.2=pyhe5570ce_0  # conda-forge
# torchtriton==3.1.0=py312  # pytorch
# torchvision==0.20.1=py312_cu124  # pytorch

# pytorch-forecasting

#python -m pip install -r ./spacetimeformer/requirements.txt
# python wind_forecasting/models/spacetimeformer/setup.py develop
# pip install wind_forecasting/preprocessing/OpenOA
#python -m pip install -r ./Informer2020/requirements.txt
#python -m pip install -r ./Autoformer/requirements.txt

# for osx
#conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 -c pytorch -y
# for cuda 12.1 linux



# python -m pip install --no-binary datatable datatable
# python -m pip install opencv-python performer-pytorch mpi4py polars floris
# conda install pytorch::pytorch torchvision torchaudio -c pytorch



# git pull --recurse-submodules
#git clone https://github.com/achenry/spacetimeformer.git
#git clone https://github.com/achenry/Autoformer.git
#git clone https://github.com/achenry/Informer2020.git
