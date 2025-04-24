#!/bin/bash

# Bash script to create and configure the 'wf_env_storm' Mamba environment
# on the Oldenburg HPC.
# Installs Python -> Pip critical packages -> Mamba packages in batches -> Local editable.
# > bash wind-forecasting/install_scripts/setup_wf_env_storm.sh

set -euo pipefail

# Configuration
DEFAULT_ENV_NAME="wf_env_storm"
PYTHON_VERSION="3.12"
# Get the directory where the script itself is located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
YAML_FILE_PATH="$SCRIPT_DIR/wf_env_storm.yaml"
declare -a LOCAL_PKGS=(
    "gluonts"
    # "openoa"
    "wind-forecasting"
    "pytorch-transformer-ts"
    "wind-hybrid-open-controller"
)
declare -a FAILED_PACKAGES=()
declare -A FAILED_COMMANDS=()

# Helper Functions
color_echo() {
    local color_code="$1"; local message="$2"
    if [ -t 1 ]; then echo -e "\e[${color_code}m${message}\e[0m"; else echo "${message}"; fi
}
log_info() { color_echo "34" "INFO: $1"; }
log_warning() { color_echo "33" "WARNING: $1"; }
log_error() { color_echo "31" "ERROR: $1"; }
log_success() { color_echo "32" "SUCCESS: $1"; }

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "Command '$1' not found. Please ensure it's installed and in your PATH."
        exit 1
    fi
}

prompt_yes_no() {
    local prompt_message="$1"; local default_choice="${2:-n}"; local answer
    while true; do
        read -p "$prompt_message [y/N]: " answer < /dev/tty; answer=${answer:-$default_choice}
        case $answer in [Yy]* ) return 0;; [Nn]* ) return 1;; * ) echo "Please answer yes (y) or no (n).";; esac
    done
}

# Installation Wrapper
install_with_optional_skip() {
    local pkg_name="$1"; local install_cmd_str="$2"; local full_cmd_str="${3:-$2}"
    log_info "Attempting to install '$pkg_name'..."
    local final_full_cmd_str="${full_cmd_str//\$ENV_NAME/$ENV_NAME}"
    final_full_cmd_str="${final_full_cmd_str//\$YAML_FILE_PATH/$YAML_FILE_PATH}"
    local final_install_cmd_str="${install_cmd_str//\$ENV_NAME/$ENV_NAME}"
    final_install_cmd_str="${final_install_cmd_str//\$YAML_FILE_PATH/$YAML_FILE_PATH}"
    local cmd_failed=0
    if [[ "$full_cmd_str" == "$install_cmd_str" ]]; then
         eval set -- "$final_full_cmd_str"; "$@" || cmd_failed=$?
    else
         bash -c "$final_full_cmd_str" || cmd_failed=$?
    fi
    if [ $cmd_failed -ne 0 ]; then
        log_error "Installation of '$pkg_name' failed (Exit Code: $cmd_failed)."
        if prompt_yes_no "Do you want to skip '$pkg_name' and continue?"; then
            log_warning "Skipping '$pkg_name'. You may need to install it manually later."
            FAILED_PACKAGES+=("$pkg_name")
            FAILED_COMMANDS["$pkg_name"]="$install_cmd_str"
        else log_error "Exiting script due to failed installation of '$pkg_name'."; exit 1; fi
    else log_success "'$pkg_name' installed successfully."; fi
}

# Main
log_info "Starting Environment Setup Script"; log_info "User: $(whoami)"; log_info "Hostname: $(hostname)"
CURRENT_SCRIPT_DIR_CALL=$(pwd)

# 1. Check Node Type
log_info "Checking current node type..."
HOSTNAME=$(hostname)
if [[ "$HOSTNAME" =~ ^(mpcg|cfdg)[0-9]+$ ]]; then log_success "Running on a recognized GPU compute node ($HOSTNAME).";
elif [[ "$HOSTNAME" =~ ^hpcl[0-9]+$ ]]; then
    log_warning "You are running this script on a LOGIN node ($HOSTNAME)."; log_warning "This environment requires GPU resources for PyTorch."
    log_warning "Please allocate a GPU node using: srun --pty -p cfdg.p --ntasks=1 --cpus-per-task=4 --mem=16G --gres=gpu:H100 --time=01:00:00 bash"; exit 1
else
    log_warning "Running on an unrecognized node type ($HOSTNAME)."; log_warning "Intended for GPU nodes (mpcg### or cfdg###)."
    if ! prompt_yes_no "Continue anyway (not recommended)?"; then log_info "Exiting script."; exit 0; fi
fi

# 2. Load Modules
log_info "Loading required HPC modules..."
module purge || { log_error "module purge failed"; exit 1; }
module load slurm/current || { log_error "Failed to load slurm module"; exit 1; }
module load hpc-env/13.1 || { log_error "Failed to load hpc-env (MPI base) module"; exit 1; }
module load mpi4py/3.1.4-gompi-2023a || { log_error "Failed to load mpi4py module"; exit 1; }
module load Mamba/24.3.0-0 || { log_error "Failed to load Mamba module"; exit 1; }
module load CUDA/12.4.0 || { log_error "Failed to load CUDA module"; exit 1; }
log_success "Modules loaded successfully."; echo "--- Module List ---"; module list; echo "-------------------"

# 3. Check Tools
log_info "Verifying required tools..."; check_command mamba; log_success "Mamba found."; log_info "Mamba version: $(mamba --version)"

# 4. Handle Environment Name and Existence
ENV_NAME=$DEFAULT_ENV_NAME
if mamba env list | grep -q "^${DEFAULT_ENV_NAME}\s"; then
    log_warning "Mamba environment '$DEFAULT_ENV_NAME' already exists."
    PS3="Choose an action: "; options=("Remove '$DEFAULT_ENV_NAME' and recreate" "Create a new environment with a different name" "Exit")
    select opt in "${options[@]}"; do
        case $REPLY in
            1) log_info "Removing existing environment '$DEFAULT_ENV_NAME'..."; mamba env remove -n "$DEFAULT_ENV_NAME" -y || { log_error "Failed to remove environment '$DEFAULT_ENV_NAME'."; exit 1; }; log_success "Environment '$DEFAULT_ENV_NAME' removed."; break ;;
            2) read -p "Enter the new environment name: " NEW_ENV_NAME < /dev/tty; if [[ -z "$NEW_ENV_NAME" ]]; then log_error "Environment name cannot be empty."; exit 1; fi; if mamba env list | grep -q "^${NEW_ENV_NAME}\s"; then log_error "Environment '$NEW_ENV_NAME' also exists. Exiting."; exit 1; fi; ENV_NAME=$NEW_ENV_NAME; log_info "Will create a new environment named '$ENV_NAME'."; break ;;
            3) log_info "Exiting script."; exit 0 ;;
            *) echo "Invalid option $REPLY";;
        esac
    done
else log_info "Environment '$ENV_NAME' does not exist. Will create it."; fi

# 5. Create Minimal Mamba Environment (Python + Pip)
log_info "Creating minimal Mamba environment '$ENV_NAME' with Python $PYTHON_VERSION and Pip..."
mamba create -n "$ENV_NAME" python="$PYTHON_VERSION" pip -y || { log_error "Minimal Mamba environment creation failed."; exit 1; }
log_success "Minimal Mamba environment '$ENV_NAME' created successfully."

# Installation within the environment

# 6. Install PyTorch
install_with_optional_skip "PyTorch (CUDA 12.4)" \
    "mamba run -n \$ENV_NAME --no-capture-output python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"

# 7. Install Lightning
install_with_optional_skip "PyTorch Lightning" \
    "mamba run -n \$ENV_NAME --no-capture-output python -m pip install lightning"

# 8. Install mpi4py
MPI4PY_INSTALL_CMD="mamba run -n \$ENV_NAME --no-capture-output python -m pip install mpi4py"
install_with_optional_skip "mpi4py" "$MPI4PY_INSTALL_CMD"


# 9. Install Remaining Packages in Batches using Mamba install
log_info "Installing remaining dependencies in batches into '$ENV_NAME'..."

# Define batches
declare -a BATCH1=(numpy pandas polars scipy scikit-learn statsmodels netcdf4 pyarrow)
declare -a BATCH2=(matplotlib seaborn windrose plotly)
declare -a BATCH3=(optuna optuna-dashboard optuna-integration wandb)
declare -a BATCH4=(floris requests pyyaml tqdm psutil mysql-connector-python postgresql gpustat uv nodejs)
declare -a BATCH5_PIP=(memory-profiler)

# Install Batch 1
log_info "Installing Batch 1 (Data/Numerics)..."
install_with_optional_skip "Batch 1 (Data/Numerics)" \
    "mamba install -n \$ENV_NAME -c pytorch -c nvidia -c conda-forge -c defaults -y ${BATCH1[*]}"

# Install Batch 2
log_info "Installing Batch 2 (Plotting)..."
install_with_optional_skip "Batch 2 (Plotting)" \
    "mamba install -n \$ENV_NAME -c conda-forge -c defaults -y ${BATCH2[*]}"

# Install Batch 3
log_info "Installing Batch 3 (ML/Experiment Tools)..."
install_with_optional_skip "Batch 3 (ML/Experiment Tools)" \
    "mamba install -n \$ENV_NAME -c conda-forge -c defaults -y ${BATCH3[*]}"

# Install Batch 4
log_info "Installing Batch 4 (Wind/Utilities/DB)..."
install_with_optional_skip "Batch 4 (Wind/Utilities/DB)" \
    "mamba install -n \$ENV_NAME -c conda-forge -c defaults -y ${BATCH4[*]}"

# Install Batch 5
log_info "Installing Batch 5 (Pip only)..."
if [ ${#BATCH5_PIP[@]} -gt 0 ]; then
    install_with_optional_skip "Batch 5 (Pip only)" \
        "mamba install -n \$ENV_NAME -y pip" \
        "mamba run -n \$ENV_NAME --no-capture-output python -m pip install ${BATCH5_PIP[*]}"
else
    log_info "No pip-only packages in Batch 5."
fi

log_success "Finished installing Mamba/Pip batch packages."


# 10. Install Local Packages
log_info "Installing local packages in editable mode into '$ENV_NAME'..."
declare -a SEARCH_DIRS=("$CURRENT_SCRIPT_DIR_CALL" "$(dirname "$CURRENT_SCRIPT_DIR_CALL")" "$(dirname "$(dirname "$CURRENT_SCRIPT_DIR_CALL")")" "$HOME")
for pkg in "${LOCAL_PKGS[@]}"; do
    log_info "Searching for package '$pkg'..."
    final_pkg_path=""
    for search_base in "${SEARCH_DIRS[@]}"; do
        potential_path="$search_base/$pkg"
        if [ -d "$potential_path" ] && [ -e "$potential_path/setup.py" ]; then
            final_pkg_path=$(realpath "$potential_path"); log_success "Found '$pkg' at: $final_pkg_path"; break
        fi
    done
    if [ -z "$final_pkg_path" ]; then
        log_warning "Could not automatically find '$pkg'."; read -p "Enter full absolute path for '$pkg': " user_path < /dev/tty
        if [ -z "$user_path" ]; then log_error "Path empty. Skipping '$pkg'."; continue; fi
        user_path_abs=$(realpath "$user_path" 2>/dev/null || echo "")
        if [ -d "$user_path_abs" ] && [ -e "$user_path_abs/setup.py" ]; then final_pkg_path="$user_path_abs"; log_info "Using user path: $final_pkg_path";
        else log_error "Invalid path or setup.py missing: '$user_path'. Skipping '$pkg'."; continue; fi
    fi
    log_info "Attempting to install '$pkg' from $final_pkg_path..."
    pushd "$final_pkg_path" > /dev/null || { log_error "Could not cd to $final_pkg_path"; continue; }
    install_with_optional_skip "$pkg (editable)" "mamba run -n \$ENV_NAME --no-capture-output python -m pip install -e ."
    popd > /dev/null
done

# 11. Final Summary
log_info "-------------------- Final Summary --------------------"
log_success "Environment '$ENV_NAME' setup process finished."
log_info "Python version in environment:"
mamba run -n $ENV_NAME python --version || log_warning "Could not get Python version."
log_info "Key packages installed (attempted):"
mamba run -n $ENV_NAME python -m pip show torch lightning mpi4py | grep -E '^Name:|^Version:' || log_warning "Could not show versions for torch/lightning/mpi4py."
log_info "Other key Mamba packages (attempted):"
mamba run -n $ENV_NAME python -m pip show numpy pandas polars scipy scikit-learn | grep -E '^Name:|^Version:' || log_warning "Could not show versions for numpy/pandas/etc."
log_info "Local editable packages (attempted):"
for pkg in "${LOCAL_PKGS[@]}"; do log_info "  - $pkg"; done

if [ ${#FAILED_PACKAGES[@]} -gt 0 ]; then
    log_warning "-------------------- Installation Issues --------------------"
    log_warning "The following packages failed to install automatically:"
    for failed_pkg in "${FAILED_PACKAGES[@]}"; do
        log_warning "  - $failed_pkg"
        manual_cmd="${FAILED_COMMANDS[$failed_pkg]}"
        manual_cmd_display="${manual_cmd//\$ENV_NAME/$ENV_NAME}"
        log_warning "    To install manually:"
        log_warning "      1. Activate the environment: mamba activate $ENV_NAME"
        log_warning "      2. Load required modules: module load ..."
        log_warning "      3. Run: $manual_cmd_display"
        if [[ "$failed_pkg" == "mpi4py" ]]; then log_warning "      (For mpi4py, ensure 'hpc-env/13.1' AND 'mpi4py/...' modules are loaded first)"; fi
        if [[ "$failed_pkg" == *" (editable)"* ]]; then pkg_base_name=$(echo "$failed_pkg" | sed 's/ (editable)//'); log_warning "      (For $pkg_base_name, ensure you 'cd' into its directory before running 'pip install -e .')"; fi
    done
    log_warning "----------------------------------------------------------"
fi

log_info "Activate the environment using: mamba activate $ENV_NAME"
log_info "--------------------------------------------------------"

exit 0
