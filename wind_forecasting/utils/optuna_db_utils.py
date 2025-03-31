import os
import logging
import time
import atexit
import torch
from wind_forecasting.utils import db_utils
from lightning.pytorch.utilities import rank_zero_only

def setup_optuna_storage(args, config, rank):
    """
    Sets up the Optuna storage backend based on configuration and handles synchronization
    between workers.
    """
    optuna_storage_url = None  # Initialize for all ranks
    pg_config = None           # Initialize pg_config

    if "tune" not in args.mode:  # Only setup DB if tuning
        return optuna_storage_url, pg_config
        
    storage_backend = config.get("optuna", {}).get("storage", {}).get("backend", "sqlite")  # Default to sqlite
    logging.info(f"Optuna storage backend configured as: {storage_backend}")

    if storage_backend == "postgresql":
        # Setup for PostgreSQL database
        return setup_postgresql(args, config, rank)
    elif storage_backend == "sqlite":
        # Setup for SQLite database
        return setup_sqlite(args, config)
    else:
        raise ValueError(f"Unsupported optuna storage backend: {storage_backend}")

@rank_zero_only
def setup_postgresql_rank_zero(config, restart=False, register_cleanup=True):
    """
    Sets up PostgreSQL instance on rank 0 (primary worker).
    """
    logging.info("Rank 0: Managing PostgreSQL instance...")
    try:
        # Manage instance (init, start, setup user/db, register cleanup)
        # Pass restart flag from args. register_cleanup=True is default for rank 0.
        optuna_storage_url, pg_config = db_utils.manage_postgres_instance(
            config,
            restart=restart,
            register_cleanup=register_cleanup  # Explicitly register cleanup for rank 0
        )

        # Ensure sync file doesn't exist from a previous failed run
        sync_file_path = pg_config.get("sync_file")
        if not sync_file_path:
             raise ValueError("Sync file path not generated in pg_config.")
        if os.path.exists(sync_file_path):
            logging.warning(f"Removing existing sync file: {sync_file_path}")
            os.remove(sync_file_path)

        # Create sync file to signal readiness
        with open(sync_file_path, 'w') as f:
            f.write('ready')
        logging.info(f"Rank 0: PostgreSQL ready. Created sync file: {sync_file_path}")
        
        return optuna_storage_url, pg_config

    except Exception as e:
        logging.error(f"Rank 0: Failed to setup PostgreSQL: {e}", exc_info=True)
        # Attempt to signal error via sync file if possible
        if pg_config and pg_config.get("sync_file"):
             try:
                  with open(pg_config["sync_file"], 'w') as f: f.write('error')
             except Exception as e_sync:
                  logging.error(f"Rank 0: Failed to write error state to sync file: {e_sync}")
        raise  # Re-raise the exception to stop rank 0

def setup_postgresql(args, config, rank):
    """
    Handles PostgreSQL setup for all ranks.
    """
    optuna_storage_url = None
    pg_config = None
    
    # Rank 0 is responsible for setting up the database
    if rank == 0:
        optuna_storage_url, pg_config = setup_postgresql_rank_zero(config, restart=args.restart_tuning)
    else:
        # Worker ranks: Generate config to find sync file and wait
        try:
            # Generate config but DO NOT manage the instance or register cleanup
            # This call primarily resolves paths and gets the sync_file location
            pg_config = db_utils._generate_pg_config(config)
            sync_file_path = pg_config.get("sync_file")
            if not sync_file_path:
                 raise ValueError("Sync file path not generated in pg_config for worker.")

            logging.info(f"Rank {rank}: Waiting for PostgreSQL sync file: {sync_file_path}")
            max_wait_time = 300  # seconds (5 minutes)
            wait_interval = 5    # seconds
            waited_time = 0
            sync_status = None
            while waited_time < max_wait_time:
                if os.path.exists(sync_file_path):
                    try:
                        with open(sync_file_path, 'r') as f:
                            sync_status = f.read().strip()
                        if sync_status == 'ready':
                            logging.info(f"Rank {rank}: Sync file found and indicates 'ready'. Proceeding.")
                            break
                        elif sync_status == 'error':
                             logging.error(f"Rank {rank}: Sync file indicates 'error' from Rank 0. Aborting.")
                             raise RuntimeError("Rank 0 failed PostgreSQL setup.")
                        else:
                             # File exists but content is unexpected, wait briefly and re-check
                             logging.warning(f"Rank {rank}: Sync file found but content is '{sync_status}'. Waiting...")

                    except Exception as e_read:
                        logging.warning(f"Rank {rank}: Error reading sync file '{sync_file_path}': {e_read}. Retrying...")

                time.sleep(wait_interval)
                waited_time += wait_interval
                
            if sync_status != 'ready':
                 logging.error(f"Rank {rank}: Timed out waiting for sync file '{sync_file_path}' or file did not indicate 'ready'.")
                 raise TimeoutError("Timed out waiting for Rank 0 PostgreSQL setup.")

            # Generate the storage URL using the generated config
            optuna_storage_url = db_utils.get_optuna_storage_url(pg_config)

        except Exception as e:
            logging.error(f"Rank {rank}: Failed during PostgreSQL sync/config generation: {e}", exc_info=True)
            raise
            
    return optuna_storage_url, pg_config

@rank_zero_only
def restart_sqlite_rank_zero(sqlite_abs_path, restart=False):
    """
    Handles SQLite database restart for rank 0.
    """
    if restart and os.path.exists(sqlite_abs_path):
         logging.warning(f"Rank 0: --restart_tuning set. Removing existing SQLite DB: {sqlite_abs_path}")
         try:
             os.remove(sqlite_abs_path)
             # Remove WAL files if they exist
             for suffix in ["-wal", "-shm"]:
                  wal_path = sqlite_abs_path + suffix
                  if os.path.exists(wal_path):
                      os.remove(wal_path)
         except OSError as e:
             logging.error(f"Failed to remove SQLite file {sqlite_abs_path}: {e}")

def setup_sqlite(args, config):
    """
    Sets up SQLite storage for Optuna.
    """
    # Construct the SQLite URL based on config
    # Use _resolve_path for consistency, getting project_root from config or default
    sqlite_rel_path = config.get("optuna", {}).get("storage", {}).get("sqlite_path", "logging/optuna/optuna_study.db")
    sqlite_abs_path = db_utils._resolve_path(config, f"optuna.storage.sqlite_path", default=sqlite_rel_path)
    # Ensure project_root is handled within _resolve_path based on experiment.project_root or CWD
    os.makedirs(os.path.dirname(sqlite_abs_path), exist_ok=True)
    optuna_storage_url = f"sqlite:///{sqlite_abs_path}"
    logging.info(f"Using SQLite storage URL: {optuna_storage_url}")
    
    # Handle restart for SQLite on rank 0
    restart_sqlite_rank_zero(sqlite_abs_path, restart=args.restart_tuning)
    
    return optuna_storage_url, None