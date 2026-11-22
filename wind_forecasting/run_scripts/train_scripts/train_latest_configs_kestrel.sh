#!/bin/bash
#SBATCH --account=awaken
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --output=%x-%j.out
#
# Retrain Informer, Autoformer, Spacetimeformer and TACTiS on Kestrel with the fixes on branch
# fix/critical-forecasting-bugs, each using the configuration of its most recent training run:
#
#   informer        training_inputs_storm_awaken_smoothed_pred60_external.yaml   (train_storm_smoothed_awaken_p60_external_h100_resume.sh)
#   autoformer      training_inputs_storm_awaken_smoothed_pred60_external.yaml   (train_autoformer_h100_1gpu.sh)
#   spacetimeformer training_inputs_storm_awaken_smoothed_pred60_external.yaml   (Storm job stf_train 15534293, 2026-02-16, 1 GPU)
#   tactis          training_inputs_storm_awaken_unsmoothed_pred60_tactis_phase0i_g.yaml (train_phase0i_g.sh)
#
# The Storm configs are copied to <WF_ROOT>/config/training/kestrel_fixes/ with the Storm paths
# replaced by the Kestrel ones (logging dirs, data paths, wandb user). Everything else (model,
# trainer and dataset settings) is kept.
# Hyperparameters: instead of --use_tuned_parameters (which needs the Storm Optuna DB), each model
# starts FRESH training with the hyperparameters stored in its reference checkpoint
# (run_model.py --extract_hparams_from_checkpoint for the model config, plus --override for the
# optimizer/scheduler hparams), extracted by extract_ckpt_hparams.py:
#   autoformer      autoformer_epoch=179-step=176438-val_loss=-0.40.ckpt
#   informer        informer_epoch169_val-0.83.ckpt
#   spacetimeformer spacetimeformer_epoch99_val-0.69.ckpt
#   tactis          quantilehead_manual_save_epoch99.ckpt
# found under <log_dir>/wind_forecasting_awaken_pred60_<model>/, where log_dir is experiment.log_dir
# of the Kestrel training config (training_inputs_kestrel_awaken_pred_smoothed.yaml).
# experiment.project_name gets a "_fixes" suffix so new checkpoints never mix with the old
# (leaky) ones.
# Use the generated configs for validation / case studies too.
#
# Usage (from a Kestrel login node, wind_forecasting_env NOT required):
#   bash train_latest_configs_kestrel.sh                  # submit all four models
#   bash train_latest_configs_kestrel.sh informer tactis  # submit a subset
#   DRY_RUN=1 bash train_latest_configs_kestrel.sh        # generate configs, print sbatch commands only
#
# Environment variables:
#   WF_ROOT            wind-forecasting checkout   (default $HOME/toolboxes/wind_forecasting_env/wind-forecasting)
#   CKPT_LOG_DIR       where the reference checkpoint folders live
#                      (default: experiment.log_dir of training_inputs_kestrel_awaken_pred_smoothed.yaml)
#   EXTRACT_SCRIPT     path to extract_ckpt_hparams.py (default: next to this script)
set -euo pipefail

WF_ROOT=${WF_ROOT:-$HOME/toolboxes/wind_forecasting_env/wind-forecasting}
CFG_DIR=${WF_ROOT}/config/training
OUT_CFG_DIR=${CFG_DIR}/kestrel_fixes
FIX_BRANCH=fix/critical-forecasting-bugs
SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
EXTRACT_SCRIPT=${EXTRACT_SCRIPT:-$(dirname "${SCRIPT_PATH}")/extract_ckpt_hparams.py}

activate_env() {
    module purge
    ml PrgEnv-intel mamba
    mamba activate wind_forecasting_env
}

# --------------------------------------------------------------------------------- job ----------
if [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${MODEL:-}" ]; then
    set +u; activate_env; set -u
    export NUMEXPR_MAX_THREADS=128
    export WANDB_DIR=${WANDB_DIR:-/projects/awaken/ahenry/wind_forecasting/logging/}

    API_FILE="${WF_ROOT}/wind_forecasting/run_scripts/.wandb_api_key"
    if [ -f "${API_FILE}" ]; then source "${API_FILE}"; else echo "WARNING: no WandB key at ${API_FILE}" >&2; fi
    ulimit -n 65535

    echo "MODEL=${MODEL} CONFIG=${CONFIG} GPUS=${SLURM_GPUS_ON_NODE:-?} NTASKS=${SLURM_NTASKS:-?}"
    echo "wind-forecasting commit: $(git -C "${WF_ROOT}" rev-parse --short HEAD) ($(git -C "${WF_ROOT}" branch --show-current))"
    echo "run_model args: ${RUN_ARGS}"

    cd "${WF_ROOT}/wind_forecasting/run_scripts"
    # shellcheck disable=SC2086
    srun python run_model.py --config "${CONFIG}" --model "${MODEL}" --mode train ${RUN_ARGS}
    exit $?
fi

# --------------------------------------------------------------------------- submitter ----------
MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(informer autoformer spacetimeformer tactis)

# 1) make sure the fixes are checked out
branch=$(git -C "${WF_ROOT}" branch --show-current)
if [ "${branch}" != "${FIX_BRANCH}" ]; then
    echo "ERROR: ${WF_ROOT} is on '${branch}', expected '${FIX_BRANCH}' (git -C ${WF_ROOT} checkout ${FIX_BRANCH})" >&2
    exit 1
fi
if ! grep -q "_persist_future_feat_dynamic_real" "${WF_ROOT}/wind_forecasting/preprocessing/pytorch_dataset.py"; then
    echo "ERROR: future-yaw fix not found in pytorch_dataset.py; pull ${FIX_BRANCH}." >&2
    exit 1
fi

declare -A SRC REF GPUS CPUS ARGS
SRC[informer]=${CFG_DIR}/training_inputs_storm_awaken_smoothed_pred60_external.yaml
SRC[autoformer]=${CFG_DIR}/training_inputs_storm_awaken_smoothed_pred60_external.yaml
SRC[spacetimeformer]=${CFG_DIR}/training_inputs_storm_awaken_smoothed_pred60_external.yaml
SRC[tactis]=${CFG_DIR}/training_inputs_storm_awaken_unsmoothed_pred60_tactis_phase0i_g.yaml
REF[informer]=${CFG_DIR}/training_inputs_kestrel_awaken_pred_smoothed.yaml
REF[autoformer]=${CFG_DIR}/training_inputs_kestrel_awaken_pred_smoothed.yaml
REF[spacetimeformer]=${CFG_DIR}/training_inputs_kestrel_awaken_pred_smoothed.yaml
REF[tactis]=${CFG_DIR}/training_inputs_kestrel_awaken_pred_unsmoothed.yaml
GPUS[informer]=4;        CPUS[informer]=8
GPUS[autoformer]=1;      CPUS[autoformer]=8
GPUS[spacetimeformer]=1; CPUS[spacetimeformer]=16
GPUS[tactis]=1;          CPUS[tactis]=16
COMMON_TRAINER="trainer.limit_train_batches=null trainer.val_check_interval=1.0"
# same arguments as each model's most recent run, minus --checkpoint (never resume old weights)
# and --use_tuned_parameters (hyperparameters come from the reference checkpoints, step 3)
ARGS[informer]="--seed 42 --override trainer.max_epochs=200 ${COMMON_TRAINER}"
ARGS[autoformer]="--seed 42 --override trainer.max_epochs=200 ${COMMON_TRAINER} trainer.strategy=auto trainer.devices=1"
ARGS[spacetimeformer]="--seed 42 --override trainer.max_epochs=200 ${COMMON_TRAINER}"
ARGS[tactis]="--seed 42 --single_gpu --override trainer.max_epochs=200"

# 2) generate Kestrel copies of the configs
set +u; activate_env; set -u
mkdir -p "${OUT_CFG_DIR}"
declare -A OUT
for m in "${MODELS[@]}"; do
    [ -n "${SRC[$m]:-}" ] || { echo "ERROR: unknown model '$m'" >&2; exit 1; }
    OUT[$m]=${OUT_CFG_DIR}/$(basename "${SRC[$m]}" .yaml)_kestrel_fixes.yaml
    python - "${SRC[$m]}" "${REF[$m]}" "${OUT[$m]}" "${WF_ROOT}" <<'EOF'
import sys, yaml
src_path, ref_path, out_path, wf_root = sys.argv[1:]
src = yaml.safe_load(open(src_path))
ref = yaml.safe_load(open(ref_path))

# Storm -> Kestrel: paths and accounts only
if True:
    for k in ("log_dir", "username"):
        src["experiment"][k] = ref["experiment"][k]
    src["experiment"]["project_root"] = wf_root
    for k in ("entity", "wandb_dir", "optuna_dir", "checkpoint_dir", "slurm_output_dir"):
        src["logging"][k] = ref["logging"][k]
    for k in ("data_path", "normalization_consts_path"):
        src["dataset"][k] = ref["dataset"][k]

# new checkpoint/log directory; run_name kept so tuned studies are still found
src["experiment"]["project_name"] = src["experiment"]["project_name"] + "_fixes"

left = [line for line in yaml.safe_dump(src).splitlines() if "/dss/" in line or "taed7566" in line]
if left:
    print("WARNING: Storm paths left in generated config:\n  " + "\n  ".join(left), file=sys.stderr)
with open(out_path, "w") as fp:
    fp.write(f"# Generated from {src_path} for Kestrel by train_latest_configs_kestrel.sh\n")
    yaml.safe_dump(src, fp, sort_keys=False)
print(f"  wrote {out_path}")
EOF
done

# 3) hyperparameters from the reference checkpoints: model config via
#    --extract_hparams_from_checkpoint, optimizer/scheduler hparams via --override
[ -f "${EXTRACT_SCRIPT}" ] || { echo "ERROR: ${EXTRACT_SCRIPT} not found" >&2; exit 1; }
CKPT_LOG_DIR=${CKPT_LOG_DIR:-$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['experiment']['log_dir'])" "${CFG_DIR}/training_inputs_kestrel_awaken_pred_smoothed.yaml")}
echo "Reference checkpoints: ${CKPT_LOG_DIR}/wind_forecasting_awaken_pred60_<model>/"
for m in "${MODELS[@]}"; do
    ckpt=$(python "${EXTRACT_SCRIPT}" --log_dir "${CKPT_LOG_DIR}" --models "$m" --print path)
    hp=$(python "${EXTRACT_SCRIPT}" --log_dir "${CKPT_LOG_DIR}" --models "$m" --print overrides)
    python "${EXTRACT_SCRIPT}" --log_dir "${CKPT_LOG_DIR}" --models "$m" --print summary \
        > "${OUT_CFG_DIR}/${m}_ckpt_hparams_summary.txt"
    echo "  $m: hyperparameters from ${ckpt} (summary in ${OUT_CFG_DIR}/${m}_ckpt_hparams_summary.txt)"
    if [[ "${hp}" == *,* ]]; then
        echo "ERROR: override values for $m contain commas, which sbatch --export cannot pass: ${hp}" >&2
        exit 1
    fi
    ARGS[$m]="--extract_hparams_from_checkpoint ${ckpt} ${ARGS[$m]} ${hp}"
done

# 4) submit one job per model
for m in "${MODELS[@]}"; do
    cmd=(sbatch --job-name="train_${m}_fixes" --gres="gpu:${GPUS[$m]}" --ntasks-per-node="${GPUS[$m]}"
         --cpus-per-task="${CPUS[$m]}"
         --export="ALL,MODEL=${m},CONFIG=${OUT[$m]},RUN_ARGS=${ARGS[$m]},WF_ROOT=${WF_ROOT}"
         "${SCRIPT_PATH}")
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf 'DRY_RUN:'; printf ' %q' "${cmd[@]}"; echo
    else
        "${cmd[@]}"
    fi
done
