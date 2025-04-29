#!/usr/bin/env python
"""
Script to convert an Optuna study from a PostgreSQL database instance
to an SQLite file.

2 ways to run:
1. Config Mode: Reads database connection details and study naming conventions from the YAML configuration file 
   (requires --config/-c and --model_name/-m).
2. Direct Input Mode: Takes the direct path to the PostgreSQL data directory
   (requires --input/-i, --model_name/-m, and --run_name/-r).

The output SQLite filename is derived from the PostgreSQL instance name
(e.g., pgdata_flasc_tactis -> flasc_tactis.db).
"""
import os
import sys
import yaml
import copy
import logging
import argparse
import time
from pathlib import Path
import optuna
from optuna.trial import TrialState

def resolve_path(base_path, path_input):
    if not path_input: return None
    path_obj = Path(str(path_input))
    if not path_obj.is_absolute():
        base_path_obj = Path(base_path)
        if not base_path_obj.is_absolute():
             base_path_obj = Path.cwd() / base_path_obj
        path_obj = base_path_obj / path_obj
    return os.path.normpath(str(path_obj))

def load_config(config_path):
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logging.error(f"Config file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML config: {e}")
        sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Optuna PostgreSQL study to SQLite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using config file (recommended for consistency)
  python %(prog)s --config path/to/config.yaml --model_name my_model

  # Using direct input path
  python %(prog)s --input /path/to/optuna/pgdata_my_instance --model_name my_model --run_name my_run --output_dir /path/to/output
"""
    )

    mode_group = parser.add_argument_group('Input Mode (Choose One)')
    mode_exclusive_group = mode_group.add_mutually_exclusive_group(required=True)
    mode_exclusive_group.add_argument(
        "--config", "-c",
        type=str,
        help="Path to the training YAML configuration file (requires --model_name)."
    )
    mode_exclusive_group.add_argument(
        "--input", "-i",
        type=str,
        help="Direct path to the PostgreSQL data directory (e.g., /path/to/pgdata_instance_name). Requires --model_name and --run_name."
    )

    parser.add_argument(
        "--model_name", "-m",
        type=str,
        required=True,
        help="Model name used in the study name (e.g., 'tactis')."
    )
    parser.add_argument(
        "--run_name", "-r",
        type=str,
        help="Run name used in the study name (e.g., 'my_tuning_run'). Required if using --input."
    )

    parser.add_argument(
        "--output_dir", "-d",
        type=str,
        help="Directory to save the SQLite file. Defaults vary based on input mode."
    )

    args = parser.parse_args()

    if args.config and args.run_name:
        parser.error("Argument --run_name/-r cannot be used with --config/-c (run_name is taken from config).")
    if args.input and not args.run_name:
         parser.error("Argument --run_name/-r is required when using --input/-i.")

    return args

def main():
    args = parse_args()

    config = None
    pgdata_path = None
    pgdata_instance_name = None
    socket_dir_base = None
    db_user = "optuna_user"
    db_name = "optuna_study_db"
    study_name = None
    default_output_dir_base = None

    if args.config:
        logging.info(f"Loading configuration from: {args.config}")
        config = load_config(args.config)

        storage_cfg = config.get("optuna", {}).get("storage", {})
        logging_cfg = config.get("logging", {})
        experiment_cfg = config.get("experiment", {})

        project_root = experiment_cfg.get("project_root", os.getcwd())
        run_name_from_config = experiment_cfg.get("run_name")
        if not run_name_from_config:
             logging.error("Missing 'experiment.run_name' in config file.")
             sys.exit(1)

        study_name = f"tuning_{args.model_name}_{run_name_from_config}"

        # Resolve paths relative to project root
        optuna_dir_from_config = logging_cfg.get("optuna_dir")
        resolved_optuna_dir = resolve_path(project_root, optuna_dir_from_config) if optuna_dir_from_config else os.path.join(project_root, "optuna")
        default_output_dir_base = resolved_optuna_dir

        pgdata_instance_name = storage_cfg.get("pgdata_instance_name")
        if not pgdata_instance_name:
            logging.error("Missing 'optuna.storage.pgdata_instance_name' in config file.")
            sys.exit(1)

        pgdata_path_from_config = storage_cfg.get("pgdata_path")
        if pgdata_path_from_config:
            pgdata_dir = resolve_path(project_root, pgdata_path_from_config)
            pgdata_path = os.path.join(pgdata_dir, f"pgdata_{pgdata_instance_name}")
        else:
            pgdata_path = os.path.join(resolved_optuna_dir, f"pgdata_{pgdata_instance_name}")

        socket_dir_base_from_config = storage_cfg.get("socket_dir_base")
        if socket_dir_base_from_config:
             socket_dir_base_str = str(socket_dir_base_from_config).replace("${logging.optuna_dir}", resolved_optuna_dir)
             socket_dir_base = resolve_path(project_root, socket_dir_base_str)
        else:
             socket_dir_base = os.path.join(resolved_optuna_dir, "sockets")

        db_user = storage_cfg.get("db_user", db_user)
        db_name = storage_cfg.get("db_name", db_name)

    elif args.input:
        # Direct Input Mode
        logging.info(f"Using direct input path: {args.input}")
        pgdata_path = str(Path(args.input).resolve())

        if not os.path.isdir(pgdata_path):
             logging.error(f"Input PostgreSQL data directory not found: {pgdata_path}")
             sys.exit(1)

        # get instance name
        pgdata_basename = os.path.basename(pgdata_path)
        if pgdata_basename.startswith("pgdata_"):
            pgdata_instance_name = pgdata_basename[len("pgdata_"):]
        else:
            logging.warning(f"Input directory name '{pgdata_basename}' doesn't follow 'pgdata_<instance_name>' convention. Using full basename as instance name.")
            pgdata_instance_name = pgdata_basename

        pgdata_parent_dir = os.path.dirname(pgdata_path)
        socket_dir_base = os.path.join(pgdata_parent_dir, "sockets")
        default_output_dir_base = pgdata_parent_dir

        study_name = f"tuning_{args.model_name}_{args.run_name}"

    else:
        logging.error("Invalid arguments. Use --config or --input.")
        sys.exit(1)

    logging.info(f"  Study Name: {study_name}")
    logging.info(f"  PGDATA Path: {pgdata_path}")
    logging.info(f"  Instance Name: {pgdata_instance_name}")
    logging.info(f"  Socket Base: {socket_dir_base}")
    logging.info(f"  DB User: {db_user}")
    logging.info(f"  DB Name: {db_name}")

    sqlite_filename = f"{pgdata_instance_name}.db"
    if args.output_dir:
        output_dir = str(Path(args.output_dir).resolve())
    else:
        output_dir = default_output_dir_base
        logging.info(f"No --output_dir specified, defaulting to: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)
    sqlite_path = os.path.join(output_dir, sqlite_filename)
    sqlite_url = f"sqlite:///{sqlite_path}"
    logging.info(f"SQLite output path: {sqlite_path}")

    correct_socket_dir_name = f"pg_socket_{pgdata_instance_name}"
    correct_socket_dir = os.path.join(socket_dir_base, correct_socket_dir_name)
    correct_postgres_url = f"postgresql://{db_user}@/{db_name}?host={correct_socket_dir}"
    logging.info(f"Using constructed URL for loading study: {correct_postgres_url.split('@')[0]}@...host={correct_socket_dir}")

    try:
        try:
            logging.info("Loading PostgreSQL study...")
            postgres_study = optuna.load_study(study_name=study_name, storage=correct_postgres_url)
            logging.info(f"Successfully loaded PostgreSQL study with {len(postgres_study.trials)} trials")
        except Exception as e:
            logging.error(f"Failed to load PostgreSQL study '{study_name}' from URL ending in host={correct_socket_dir}")
            logging.error(f"Error details: {e}")
            logging.error("Possible reasons:")
            logging.error("  - PostgreSQL server for this instance is not running.")
            logging.error("  - Server is running but not listening on the expected socket path.")
            logging.error(f"  - Study name '{study_name}' does not exist in the database.")
            logging.error("  - Database user/name mismatch.")
            logging.error("Ensure the server is started correctly (e.g., via the .sh script) before running this conversion.")
            sys.exit(1)

        try:
            logging.info(f"Creating/overwriting SQLite study at: {sqlite_path}")
            try:
                 optuna.delete_study(study_name=study_name, storage=sqlite_url)
                 logging.info(f"Deleted existing SQLite study '{study_name}' for overwrite.")
            except KeyError:
                 logging.info(f"No existing SQLite study '{study_name}' found to delete.")
            except Exception as e_del:
                 logging.warning(f"Could not delete existing SQLite study (perhaps file permissions?): {e_del}")

            sqlite_study = optuna.create_study(
                study_name=study_name,
                storage=sqlite_url,
                direction=postgres_study.direction,
                load_if_exists=False # Should be false after attempting delete
            )
            logging.info("SQLite study created/ready.")
        except Exception as e:
            logging.error(f"Failed to create SQLite study: {e}")
            sys.exit(1)

        logging.info("Transferring trials...")
        completed_count = 0
        pruned_count = 0
        failed_count = 0
        other_count = 0

        for trial in postgres_study.trials:
            # Create a deep copy
            trial_copy = copy.deepcopy(trial)

            # Handle different trial states
            try:
                sqlite_study.add_trial(trial_copy)
                if trial.state == optuna.trial.TrialState.COMPLETE: completed_count += 1
                elif trial.state == optuna.trial.TrialState.PRUNED: pruned_count += 1
                elif trial.state == optuna.trial.TrialState.FAIL: failed_count += 1
                else: other_count += 1
            except Exception as e_add:
                 logging.warning(f"Could not add trial #{trial.number} (state: {trial.state}): {e_add}. Skipping.")
                 other_count +=1

        logging.info(f"Successfully transferred trials to SQLite:")
        logging.info(f"  - COMPLETE: {completed_count}")
        logging.info(f"  - PRUNED: {pruned_count}")
        logging.info(f"  - FAIL: {failed_count}")
        logging.info(f"  - Other states/Skipped: {other_count}")
        logging.info(f"  - Total Processed: {len(postgres_study.trials)}")
        logging.info(f"SQLite database saved to: {sqlite_path}")

    except Exception as e:
        logging.error(f"Error during conversion process: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()