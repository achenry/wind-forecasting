import os
import logging
import time

from wind_forecasting.utils import db_utils
from lightning.pytorch.utilities import rank_zero_only
from mysql.connector import connect as sql_connect
from optuna.storages import JournalStorage, RDBStorage
from optuna.storages.journal import JournalFileBackend


def setup_optuna_storage(db_setup_params, restart_tuning, rank):
    """
    Sets up the Optuna storage backend based on configuration and handles synchronization
    between workers.
    """
    storage_backend = db_setup_params.get("backend", "sqlite")
    logging.info(f"Optuna storage backend configured as: {storage_backend}")
    
    # if "storage_dir" in db_setup_params:
    #     # Ensure the directory exists
    #     os.makedirs(db_setup_params["storage_dir"], exist_ok=True)
    #     logging.info(f"Using explicitly defined Optuna storage_dir: {db_setup_params['storage_dir']}")

    if storage_backend == "postgresql":
        # Setup for PostgreSQL database
        # Pass only the necessary parameters for postgresql setup
        storage = setup_postgresql(
            db_setup_params=db_setup_params,
            restart_tuning=restart_tuning,
            rank=rank
        )
    elif storage_backend == "sqlite":
        # Setup for SQLite database
        storage = setup_sqlite(
            sqlite_storage_dir=db_setup_params["storage_dir"], # Use resolved storage_dir
            study_name=db_setup_params["study_name"],
            restart_tuning=restart_tuning,
            rank=rank
        )
    elif storage_backend == "journal":
        return setup_journal(
            storage_dir=db_setup_params["storage_dir"], # Use resolved storage_dir
            study_name=db_setup_params["study_name"],
            restart_tuning=restart_tuning,
            rank=rank
        )
    elif storage_backend == "mysql":
        storage = setup_mysql(db_setup_params=db_setup_params, 
                              restart_tuning=restart_tuning, 
                              rank=rank)
    else:
        raise ValueError(f"Unsupported optuna storage backend: {storage_backend}") 
    return storage

@rank_zero_only
def delete_studies(storage):
    logging.info(f"Deleting existing Optuna studies {storage.get_all_studies()}.")  
    for s in storage.get_all_studies():
        storage.delete_study(s._study_id)

@rank_zero_only
def setup_postgresql_rank_zero(db_setup_params, restart_tuning=False, register_cleanup=True):
    """
    Sets up PostgreSQL instance on rank 0 (primary worker).
    """
    logging.info("Rank 0: Managing PostgreSQL instance...")
    pg_config = None # Initialize in case manage_postgres_instance fails early
    try:
        # Filter db_setup_params to only include keys relevant for PostgreSQL setup
        pg_params = {
            k: v for k, v in db_setup_params.items() if k in [
                "backend", "project_root", "pgdata_path", "study_name",
                "use_socket", "use_tcp", "db_host", "db_port", "db_name",
                "db_user", "run_cmd_shell", "socket_dir_base", "sync_dir"
            ]
        }

        # Manage instance (init, start, setup user/db)
        # Pass restart flag from args.
        # Explicitly set register_cleanup=False to prevent premature DB shutdown by worker 0's atexit.
        # Cleanup should be handled externally after all workers finish.
        # Pass the filtered explicit parameters needed by manage_postgres_instance
        optuna_storage_url, pg_config = db_utils.manage_postgres_instance(
            db_setup_params=pg_params, # Pass the filtered dict
            restart=restart_tuning,
            register_cleanup=False # Disable atexit registration for Optuna runs
        )

        # Rank 0 creates the storage instance, triggering schema creation
        logging.info(f"Rank 0: Creating RDBStorage instance to initialize schema...")
        storage = RDBStorage(url=optuna_storage_url)
        logging.info(f"Rank 0: RDBStorage instance created.")
        # Ensure sync file doesn't exist from a previous failed run
        # pg_config should be populated by manage_postgres_instance if successful
        if not pg_config or "sync_file" not in pg_config:
             raise ValueError("Sync file path not generated in pg_config.")
        else:
            sync_file_path = pg_config["sync_file"]
            
        if os.path.exists(sync_file_path):
            logging.warning(f"Removing existing sync file: {sync_file_path}")
            os.remove(sync_file_path)

        # Create sync file *after* storage/schema is ready
        with open(sync_file_path, 'w') as f:
            f.write('ready')
        logging.info(f"Rank 0: PostgreSQL ready. Created sync file: {sync_file_path}")
        
        return storage, pg_config # Return the created storage object

    except Exception as e:
        logging.error(f"Rank 0: Failed to setup PostgreSQL: {e}", exc_info=True)
        # Attempt to signal error via sync file if possible
        if pg_config and pg_config.get("sync_file"):
             try:
                  with open(pg_config["sync_file"], 'w') as f: f.write('error')
             except Exception as e_sync:
                  logging.error(f"Rank 0: Failed to write error state to sync file: {e_sync}")
        raise  # Re-raise the exception to stop rank 0

def setup_postgresql(db_setup_params, rank, restart_tuning):
    """
    Handles PostgreSQL setup for all ranks.
    """
    optuna_storage_url = None
    pg_config = None # Initialize

    # Rank 0 is responsible for setting up the database
    if rank == 0:
        # Pass the explicit params dict
        storage, pg_config = setup_postgresql_rank_zero(db_setup_params, restart_tuning=restart_tuning)

    else:
        # Worker ranks: Generate config *only* to find sync file and wait
        try:
            # Generate config but DO NOT manage the instance or register cleanup
            # This call primarily resolves paths and gets the sync_file location
            # Filter db_setup_params to only include keys relevant for PostgreSQL setup
            pg_params = {
                k: v for k, v in db_setup_params.items() if k in [
                    "backend", "project_root", "pgdata_path", "study_name",
                    "use_socket", "use_tcp", "db_host", "db_port", "db_name",
                    "db_user", "run_cmd_shell", "socket_dir_base", "sync_dir"
                ]
            }
            # Pass the filtered explicit params dict
            pg_config = db_utils._generate_pg_config(**pg_params) # Unpack the filtered explicit params here
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
            storage = RDBStorage(url=optuna_storage_url)

        except Exception as e:
            logging.error(f"Rank {rank}: Failed during PostgreSQL sync/config generation: {e}", exc_info=True)
            raise
            
    return storage

@rank_zero_only
def restart_journal_rank_zero(journal_abs_path):
    """
    Handles journal database restart for rank 0.
    """
    if os.path.exists(journal_abs_path):
         logging.warning(f"Rank 0: --restart_tuning set. Removing existing Journal DB: {journal_abs_path}")
         try:
             os.remove(journal_abs_path)
         except OSError as e:
             logging.error(f"Failed to remove Journal file {journal_abs_path}: {e}")
             
@rank_zero_only
def restart_sqlite_rank_zero(sqlite_abs_path):
    """
    Handles SQLite database restart for rank 0.
    """
    if os.path.exists(sqlite_abs_path):
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

def setup_sqlite(sqlite_storage_dir, study_name, restart_tuning, rank):
    """
    Sets up SQLite storage for Optuna.
    """
    # Construct the SQLite URL based on config
    optuna_storage_url = f"sqlite:///{os.path.join(sqlite_storage_dir, f'{study_name}.db')}"
    logging.info(f"Using SQLite storage URL: {optuna_storage_url}")
    
    # Handle restart for SQLite on rank 0
    # if rank == 0 and restart_tuning:
    #     restart_sqlite_rank_zero(sqlite_abs_path)
    if rank == 0:
        storage = RDBStorage(url=optuna_storage_url)
    else:
        raise Exception("Cannot use SQLite storage with multiple workers. Please use a different backend.")
    if rank == 0 and restart_tuning:
        delete_studies(storage)
    
    return storage

def setup_journal(storage_dir, study_name, restart_tuning, rank):
    if rank == 0:
        logging.info(f"Connecting to Journal database {study_name}")
        optuna_storage_url = os.path.join(storage_dir, f"{study_name}.db")
        if restart_tuning:
            restart_journal_rank_zero(optuna_storage_url)
    
    storage = JournalStorage(JournalFileBackend(optuna_storage_url))
    return storage
    
def setup_mysql(db_setup_params, restart_tuning, rank):
    logging.info(f"Connecting to RDB database {db_setup_params['study_name']}")
    try:
        # '127.0.0.1'
        db = sql_connect(host=db_setup_params["db_host"], user=db_setup_params["db_user"],
                        database=db_setup_params["study_name"])       
    except Exception as e:
        try:
            db = sql_connect(host=db_setup_params["db_host"], user=db_setup_params["db_user"])
            cursor = db.cursor()
            if rank == 0:
                cursor.execute(f"CREATE DATABASE {db_setup_params['study_name']}")
        except Exception as ee:
            raise(f"Failed to connect to MySQL database: {ee}")
    finally:
        # Handle restart for MySQL on rank 0
        optuna_storage_url = f"mysql://{db.user}@{db.server_host}:{db.server_port}/{db_setup_params['study_name']}"
        # restart_mysql_rank_zero(optuna_storage_url)
        storage = RDBStorage(url=optuna_storage_url)
        if restart_tuning and rank == 0:
            delete_studies(storage)
        
    return storage