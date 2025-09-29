#!/usr/bin/env python
"""
Script to convert an Optuna study from a TCP/IP PostgreSQL database to SQLite.

This version is specifically for remote PostgreSQL servers accessed via TCP/IP,
as opposed to the original version which expects local PostgreSQL with Unix sockets.

Usage:
    python convert_postgre_to_sqlite_tcp.py \\
        --config path/to/config.yaml \\
        --model_name tactis \\
        --output_dir /path/to/output
"""
import os
import sys
import yaml
import copy
import logging
import argparse
from pathlib import Path
import optuna
from optuna.trial import TrialState


def resolve_path(base_path, path_input):
    """Resolve relative paths against a base path."""
    if not path_input:
        return None
    path_obj = Path(str(path_input))
    if not path_obj.is_absolute():
        base_path_obj = Path(base_path)
        if not base_path_obj.is_absolute():
            base_path_obj = Path.cwd() / base_path_obj
        path_obj = base_path_obj / path_obj
    return os.path.normpath(str(path_obj))


def load_config(config_path):
    """Load YAML configuration file."""
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
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert Optuna PostgreSQL study (TCP/IP) to SQLite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using config file
  python %(prog)s --config config.yaml --model_name tactis

  # With custom output directory
  python %(prog)s --config config.yaml --model_name tactis --output_dir ./my_sqlite_dbs

  # Using direct connection parameters
  python %(prog)s \\
      --study_name "tuning_tactis_my_run" \\
      --db_host "pg.optuna.uni-oldenburg.de" \\
      --db_port 5432 \\
      --db_name "optuna" \\
      --db_user "optuna02" \\
      --db_password_env "LOCAL_PG_PASSWORD" \\
      --output_dir ./Output
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
        "--study_name", "-s",
        type=str,
        help="Direct study name (requires all --db_* parameters)."
    )

    parser.add_argument(
        "--model_name", "-m",
        type=str,
        help="Model name used in the study name (e.g., 'tactis'). Required with --config."
    )

    # Direct connection parameters
    parser.add_argument(
        "--db_host",
        type=str,
        help="PostgreSQL host (e.g., 'pg.optuna.uni-oldenburg.de'). Required with --study_name."
    )
    parser.add_argument(
        "--db_port",
        type=int,
        default=5432,
        help="PostgreSQL port (default: 5432)."
    )
    parser.add_argument(
        "--db_name",
        type=str,
        help="PostgreSQL database name (e.g., 'optuna'). Required with --study_name."
    )
    parser.add_argument(
        "--db_user",
        type=str,
        help="PostgreSQL user. Required with --study_name."
    )
    parser.add_argument(
        "--db_password_env",
        type=str,
        help="Environment variable containing the database password. Required with --study_name."
    )

    parser.add_argument(
        "--output_dir", "-d",
        type=str,
        help="Directory to save the SQLite file. Defaults to config's optuna_dir or current directory."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing SQLite database if it exists."
    )

    args = parser.parse_args()

    # Validation
    if args.config:
        if not args.model_name:
            parser.error("--model_name/-m is required when using --config/-c")
    elif args.study_name:
        required_params = ['db_host', 'db_name', 'db_user', 'db_password_env']
        missing = [p for p in required_params if not getattr(args, p)]
        if missing:
            parser.error(f"When using --study_name, these parameters are required: {', '.join(['--' + p for p in missing])}")

    return args


def main():
    args = parse_args()

    # Initialize variables
    study_name = None
    db_host = None
    db_port = 5432
    db_name = None
    db_user = None
    db_password = None
    output_dir = None

    if args.config:
        # Config Mode
        logging.info(f"Loading configuration from: {args.config}")
        config = load_config(args.config)

        storage_cfg = config.get("optuna", {}).get("storage", {})
        logging_cfg = config.get("logging", {})
        experiment_cfg = config.get("experiment", {})

        # Validate TCP/IP mode
        use_tcp = storage_cfg.get("use_tcp", False)
        if not use_tcp:
            logging.warning("Config has 'use_tcp: false' but this script only handles TCP/IP connections!")
            logging.warning("Proceeding anyway, assuming TCP/IP setup is actually in use.")

        # Extract connection parameters
        db_host = storage_cfg.get("db_host")
        db_port = storage_cfg.get("db_port", 5432)
        db_name = storage_cfg.get("db_name")
        db_user = storage_cfg.get("db_user")
        db_password_env_var = storage_cfg.get("db_password_env_var", "LOCAL_PG_PASSWORD")

        if not all([db_host, db_name, db_user]):
            logging.error("Config missing required TCP/IP parameters: db_host, db_name, db_user")
            sys.exit(1)

        # Get password from environment
        db_password = os.environ.get(db_password_env_var)
        if not db_password:
            logging.error(f"Environment variable '{db_password_env_var}' not set for database password")
            sys.exit(1)

        # Construct study name
        run_name = experiment_cfg.get("run_name")
        if not run_name:
            logging.error("Missing 'experiment.run_name' in config file")
            sys.exit(1)

        study_name = f"tuning_{args.model_name}_{run_name}"

        # Determine output directory
        project_root = experiment_cfg.get("project_root", os.getcwd())
        optuna_dir_from_config = logging_cfg.get("optuna_dir")
        resolved_optuna_dir = resolve_path(project_root, optuna_dir_from_config) if optuna_dir_from_config else os.path.join(project_root, "optuna")
        output_dir = args.output_dir if args.output_dir else resolved_optuna_dir

    else:
        # Direct Input Mode
        study_name = args.study_name
        db_host = args.db_host
        db_port = args.db_port
        db_name = args.db_name
        db_user = args.db_user

        db_password = os.environ.get(args.db_password_env)
        if not db_password:
            logging.error(f"Environment variable '{args.db_password_env}' not set for database password")
            sys.exit(1)

        output_dir = args.output_dir if args.output_dir else os.getcwd()

    # Log configuration
    logging.info("=" * 60)
    logging.info("PostgreSQL to SQLite Conversion Configuration")
    logging.info("=" * 60)
    logging.info(f"  Study Name:    {study_name}")
    logging.info(f"  DB Host:       {db_host}")
    logging.info(f"  DB Port:       {db_port}")
    logging.info(f"  DB Name:       {db_name}")
    logging.info(f"  DB User:       {db_user}")
    logging.info(f"  Output Dir:    {output_dir}")
    logging.info("=" * 60)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Construct PostgreSQL URL (TCP/IP with SSL disabled)
    postgres_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?sslmode=disable"
    logging.info(f"PostgreSQL URL: postgresql://{db_user}:****@{db_host}:{db_port}/{db_name}?sslmode=disable")

    # SQLite file path
    sqlite_filename = f"{study_name}.db"
    sqlite_path = os.path.join(output_dir, sqlite_filename)
    sqlite_url = f"sqlite:///{sqlite_path}"
    logging.info(f"SQLite output:  {sqlite_path}")
    logging.info("")

    # Load PostgreSQL study with retry logic
    max_retries = 10
    retry_delay = 30  # seconds
    postgres_study = None

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                logging.info(f"Retry attempt {attempt + 1}/{max_retries}...")
            else:
                logging.info("Loading PostgreSQL study...")

            postgres_study = optuna.load_study(study_name=study_name, storage=postgres_url)
            logging.info(f"✓ Successfully loaded PostgreSQL study")
            logging.info(f"  - Total trials: {len(postgres_study.trials)}")
            logging.info(f"  - Direction:    {postgres_study.direction}")
            logging.info(f"  - Best value:   {postgres_study.best_value if postgres_study.best_trial else 'N/A'}")
            logging.info("")
            break  # Success - exit retry loop

        except KeyError:
            logging.error(f"✗ Study '{study_name}' not found in database")
            logging.error("\nAvailable studies:")
            try:
                storage = optuna.storages.RDBStorage(postgres_url)
                studies = storage.get_all_studies()
                for study in studies:
                    logging.error(f"  - {study.study_name}")
            except Exception as e:
                logging.error(f"Could not list available studies: {e}")
            sys.exit(1)

        except Exception as e:
            error_msg = str(e)

            # Check if it's a "too many connections" error
            if "zu viele Verbindungen" in error_msg or "too many connections" in error_msg.lower():
                if attempt < max_retries - 1:
                    logging.warning(f"⚠ Database connection limit reached")
                    logging.warning(f"  Waiting {retry_delay} seconds before retry {attempt + 2}/{max_retries}...")
                    logging.warning(f"  (You have many active tuning jobs using connections)")
                    import time
                    time.sleep(retry_delay)
                else:
                    logging.error(f"✗ Failed after {max_retries} attempts: too many connections")
                    logging.error("\nSolutions:")
                    logging.error("  1. Wait for some tuning jobs to complete")
                    logging.error("  2. Cancel completed/stuck jobs: scancel <job_id>")
                    logging.error("  3. Run this script as an sbatch job (will retry automatically)")
                    sys.exit(1)
            else:
                # Different error - don't retry
                logging.error(f"✗ Failed to load PostgreSQL study: {e}")
                logging.error("\nPossible reasons:")
                logging.error("  - Database connection issue (check host, port, credentials)")
                logging.error("  - Study name doesn't exist")
                sys.exit(1)

    if postgres_study is None:
        logging.error("✗ Failed to load study after retries")
        sys.exit(1)

    # Create/overwrite SQLite study
    try:
        logging.info("Creating SQLite database...")

        # Check if exists and handle overwrite
        if os.path.exists(sqlite_path):
            if args.overwrite:
                try:
                    optuna.delete_study(study_name=study_name, storage=sqlite_url)
                    logging.info(f"  Deleted existing study for overwrite")
                except KeyError:
                    logging.info(f"  Existing file found but no study inside")
            else:
                try:
                    existing_study = optuna.load_study(study_name=study_name, storage=sqlite_url)
                    logging.error(f"✗ SQLite database already exists with {len(existing_study.trials)} trials")
                    logging.error(f"  Use --overwrite to replace it")
                    sys.exit(1)
                except KeyError:
                    # File exists but study doesn't - safe to create
                    pass

        sqlite_study = optuna.create_study(
            study_name=study_name,
            storage=sqlite_url,
            direction=postgres_study.direction,
            load_if_exists=False
        )
        logging.info(f"✓ SQLite study created")
        logging.info("")
    except Exception as e:
        logging.error(f"✗ Failed to create SQLite study: {e}")
        sys.exit(1)

    # Transfer trials
    logging.info("Transferring trials...")
    logging.info("")

    completed_count = 0
    pruned_count = 0
    failed_count = 0
    running_count = 0
    other_count = 0
    skipped_count = 0

    for i, trial in enumerate(postgres_study.trials):
        if (i + 1) % 50 == 0:
            logging.info(f"  Progress: {i + 1}/{len(postgres_study.trials)} trials...")

        try:
            trial_copy = copy.deepcopy(trial)
            sqlite_study.add_trial(trial_copy)

            if trial.state == TrialState.COMPLETE:
                completed_count += 1
            elif trial.state == TrialState.PRUNED:
                pruned_count += 1
            elif trial.state == TrialState.FAIL:
                failed_count += 1
            elif trial.state == TrialState.RUNNING:
                running_count += 1
            else:
                other_count += 1
        except Exception as e_add:
            logging.warning(f"  ⚠ Could not add trial #{trial.number} (state: {trial.state}): {e_add}")
            skipped_count += 1

    # Summary
    logging.info("")
    logging.info("=" * 60)
    logging.info("Transfer Complete")
    logging.info("=" * 60)
    logging.info(f"  COMPLETE trials: {completed_count}")
    logging.info(f"  PRUNED trials:   {pruned_count}")
    logging.info(f"  FAILED trials:   {failed_count}")
    logging.info(f"  RUNNING trials:  {running_count}")
    logging.info(f"  Other states:    {other_count}")
    if skipped_count > 0:
        logging.info(f"  Skipped (errors): {skipped_count}")
    logging.info(f"  ────────────────────────")
    logging.info(f"  Total transferred: {completed_count + pruned_count + failed_count + running_count + other_count}")
    logging.info(f"  Total in source:   {len(postgres_study.trials)}")
    logging.info("=" * 60)
    logging.info(f"✓ SQLite database saved: {sqlite_path}")
    logging.info("")
    logging.info("Next step: Use this database for hyperparameter importance analysis:")
    logging.info(f"  python calc_hyperparameter_importance.py \\")
    logging.info(f"    -s \"{study_name}\" \\")
    logging.info(f"    --url \"sqlite:///{sqlite_path}\" \\")
    logging.info(f"    --max-depth 128 \\")
    logging.info(f"    --n-trees 128 \\")
    logging.info(f"    -o ./Output")
    logging.info("")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    main()