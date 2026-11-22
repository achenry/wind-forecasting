#!/usr/bin/env python
"""Extract the (tuned) hyperparameters stored in Lightning checkpoints of the wind-forecasting
models, so they can be reused for retraining without access to the Optuna database.

For each model it writes <out_dir>/<model>_ckpt_hparams.json with
  - model_config     : the estimator/model architecture config (what run_model.py
                       --extract_hparams_from_checkpoint feeds back into the estimator)
  - optimizer        : lr / weight decay / clipping / scheduler hparams saved by the
                       LightningModule (NOT applied by --extract_hparams_from_checkpoint)
  - epoch, global_step, checkpoint path
and can print the matching `--override model.<model>.<key>=<value>` tokens for run_model.py.

Checkpoints are looked up in <log_dir>/wind_forecasting_awaken_pred60_<model>/ (any subfolder),
where log_dir is experiment.log_dir of config/training/training_inputs_kestrel_awaken_pred_smoothed.yaml
(override with --config or --log_dir).

Usage (on Kestrel, inside wind_forecasting_env):
  python extract_ckpt_hparams.py                                   # all four default checkpoints
  python extract_ckpt_hparams.py --models informer tactis
  python extract_ckpt_hparams.py --models informer --print overrides   # just the override tokens
  python extract_ckpt_hparams.py --models informer --print path        # just the checkpoint path
"""
import argparse
import glob
import json
import os
import sys

# checkpoints are looked up in <log_dir>/wind_forecasting_awaken_pred60_<model>/, where log_dir is
# experiment.log_dir of the Kestrel training config
DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "config",
                              "training", "training_inputs_kestrel_awaken_pred_smoothed.yaml")
DEFAULT_CKPTS = {
    "autoformer": "autoformer_epoch=179-step=176438-val_loss=-0.40.ckpt",
    "informer": "informer_epoch169_val-0.83.ckpt",
    "spacetimeformer": "spacetimeformer_epoch99_val-0.69.ckpt",
    "tactis": "quantilehead_manual_save_epoch99.ckpt",
}
# LightningModule hparams that run_model.py passes to the estimator as model.<model>.<key>.
# Scheduler lengths (warmup_steps / steps_to_decay) are only reused when stored as fractions
# (<= 1); absolute step counts depend on the original run's epochs/batches and are reported only.
OPTIMIZER_KEYS = {
    "default": ["lr", "weight_decay", "gradient_clip_val", "eta_min_fraction",
                "warmup_steps", "steps_to_decay"],
    "tactis": ["lr_stage1", "lr_stage2", "weight_decay_stage1", "weight_decay_stage2",
               "gradient_clip_val_stage1", "gradient_clip_val_stage2", "stage2_start_epoch",
               "stage1_activation_function", "stage2_activation_function",
               "eta_min_fraction_s1", "eta_min_fraction_s2",
               "warmup_steps_s1", "warmup_steps_s2", "steps_to_decay_s1", "steps_to_decay_s2"],
}
SCHEDULE_LENGTH_KEYS = {"warmup_steps", "steps_to_decay", "warmup_steps_s1", "warmup_steps_s2",
                        "steps_to_decay_s1", "steps_to_decay_s2"}


def find_checkpoint(log_dir, model, filename):
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename
    model_dir = os.path.join(log_dir, f"wind_forecasting_awaken_pred60_{model}")
    hits = sorted(glob.glob(os.path.join(model_dir, "**", filename), recursive=True))
    if not hits:
        raise FileNotFoundError(f"{filename} not found under {model_dir}")
    if len(hits) > 1:
        print(f"WARNING: several copies of {filename}; using {hits[0]}", file=sys.stderr)
    return hits[0]


def jsonable(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    return repr(v)  # e.g. distr_output instances


def extract(ckpt_path, model):
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", ckpt.get("hparams"))
    if hparams is None:
        raise KeyError(f"No 'hyper_parameters' in {ckpt_path} (keys: {list(ckpt.keys())})")
    hparams = dict(hparams)
    model_config = dict(hparams.get("model_config") or {})

    keys = OPTIMIZER_KEYS["tactis" if model == "tactis" else "default"]
    optimizer = {k: hparams[k] for k in keys if k in hparams}
    overrides, reported_only = {}, {}
    for k, v in optimizer.items():
        if v is None:
            continue
        if k in SCHEDULE_LENGTH_KEYS and isinstance(v, (int, float)) and v > 1:
            reported_only[k] = v
        else:
            overrides[k] = v
    return {
        "checkpoint": ckpt_path,
        "epoch": ckpt.get("epoch"),
        "global_step": ckpt.get("global_step"),
        "model_config": jsonable(model_config),
        "optimizer": jsonable(optimizer),
        "override_hparams": jsonable(overrides),
        "absolute_schedule_steps_not_reused": jsonable(reported_only),
        "other_saved_hparams": jsonable({k: v for k, v in hparams.items()
                                         if k != "model_config" and k not in optimizer}),
    }


def override_tokens(model, info):
    return [f"model.{model}.{k}={v}"
            for k, v in info["override_hparams"].items()]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG,
                   help="training config whose experiment.log_dir holds the model checkpoint folders")
    p.add_argument("--log_dir", default=None, help="overrides experiment.log_dir from --config")
    p.add_argument("--models", nargs="+", default=list(DEFAULT_CKPTS))
    p.add_argument("--ckpt", action="append", default=[],
                   help="model=filename_or_path to use instead of the default checkpoint")
    p.add_argument("--out_dir", default=None, help="where to write JSON (default: <log_dir>/ckpt_hparams)")
    p.add_argument("--print", choices=["summary", "overrides", "path"], default="summary")
    args = p.parse_args()

    if args.log_dir is None:
        import yaml
        with open(args.config) as fp:
            args.log_dir = yaml.safe_load(fp)["experiment"]["log_dir"]

    ckpts = dict(DEFAULT_CKPTS)
    for item in args.ckpt:
        m, f = item.split("=", 1)
        ckpts[m] = f
    out_dir = args.out_dir or os.path.join(args.log_dir, "ckpt_hparams")
    os.makedirs(out_dir, exist_ok=True)

    for model in args.models:
        path = find_checkpoint(args.log_dir, model, ckpts[model])
        if args.print == "path":
            print(path)
            continue
        info = extract(path, model)
        with open(os.path.join(out_dir, f"{model}_ckpt_hparams.json"), "w") as fp:
            json.dump(info, fp, indent=2)
        if args.print == "overrides":
            print(" ".join(override_tokens(model, info)))
            continue
        mc = info["model_config"]
        print(f"\n=== {model}: {path} (epoch {info['epoch']}, step {info['global_step']})")
        print(f"context_length={mc.get('context_length')} prediction_length={mc.get('prediction_length')} "
              f"freq={mc.get('freq')}")
        print("model_config:")
        for k in sorted(mc):
            print(f"  {k}: {mc[k]}")
        print("optimizer / scheduler:")
        for k, v in info["optimizer"].items():
            print(f"  {k}: {v}")
        if info["absolute_schedule_steps_not_reused"]:
            print("  (absolute step counts, not reused: "
                  f"{info['absolute_schedule_steps_not_reused']})")
        print("run_model.py overrides:", " ".join(override_tokens(model, info)) or "(none)")
        print(f"wrote {os.path.join(out_dir, f'{model}_ckpt_hparams.json')}")


if __name__ == "__main__":
    main()
