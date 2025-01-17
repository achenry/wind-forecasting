acompile
ml mambaforge
mamba create --prefix=/projects/aohe7145/software/anaconda/envs/wind_forecasting python=3.12.4 --y
mamba activate wind_forecasting
mamba install conda-forge::cuda-version=12.4 nvidia/label/cuda-12.4.0::cuda-toolkit performer-pytorch pytorch torchvision torchaudio torchmetrics pytorch-cuda=12.4 lightning -c pytorch -c nvidia --y
mamba install polars windrose statsmodels scikit-learn jupyterlab nb_conda_kernels pyyaml matplotlib numpy seaborn opt_einsum netcdf4 scipy h5pyd pyarrow wandb einops cython --y 
pip3 install mpi4py impi_rt opencv-python floris

git clone https://github.com/achenry/wind-forecasting.git
cd wind-forecasting
git checkout feature/spacetimeformer
python setup.py develop
cd ..

git clone https://github.com/achenry/gluonts
cd gluonts
git checkout mv_prob
pip3 install -e .[dev]
cd ..

git clone https://github.com/boujuan/pytorch-transformer-ts
cd pytorch-transformer-ts
git checkout feature/spacetimeformer
pip3 install ujson datasets xformers etsformer-pytorch reformer_pytorch pykeops apex # gluonts[torch]
pip3 install git+https://github.com/kashif/hopfield-layers@pytorch-2 git+https://github.com/microsoft/torchscale
cd ..

# git clone https://github.com/achenry/OpenOA
# cd OpenOA
# pip install -e .
# cd ..
