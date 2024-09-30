#!/usr/bin/env fish

# Instructions for installing environments on the CU Boulder Alpine cluster (for jubo7621)
eval (ssh-agent -c)
ssh jubo7621@login.rc.colorado.edu << 'EOF'

acompile
module purge
ml mambaforge

# Check if mamba is available
if not command -v mamba > /dev/null
    echo "mamba could not be found. Please check if the mambaforge module is loaded correctly."
    exit 1
end

# Create and activate environments
for env_file in wind_forecasting_cuda.yml wind_forecasting_env.yml wind_forecasting_rocm.yml wind_preprocessing.yml
    set env_name (string replace -r '\.yml$' '' $env_file)
    echo "Creating environment: $env_name"
    mamba env create -f "@install_rc/$env_file" -n "$env_name"
    mamba activate "$env_name"
    mamba deactivate
end

# Clone the repository
cd /projects/$USER
if not test -d "/projects/$USER/wind-forecasting"
    echo "Cloning wind-forecasting repository..."
    git clone --recurse-submodules https://github.com/achenry/wind-forecasting.git
else
    echo "wind-forecasting repository already exists. Skipping clone."
end
cd wind-forecasting/wind-forecasting/models

# ***Uncomment these lines if you need to install requirements for specific models***
# python -m pip install -r ./spacetimeformer/requirements.txt
# python ./spacetimeformer/setup.py develop
# python -m pip install -r ./Informer2020/requirements.txt
# python -m pip install -r ./Autoformer/requirements.txt

# ***Uncomment this line if you need to update submodules***
# git pull --recurse-submodules

# ***Uncomment these lines if you need to clone specific repositories***
# git clone https://github.com/achenry/spacetimeformer.git
# git clone https://github.com/achenry/Autoformer.git
# git clone https://github.com/achenry/Informer2020.git

exit
exit
EOF