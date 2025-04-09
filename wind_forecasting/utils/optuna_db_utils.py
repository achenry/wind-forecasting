import os
import logging
import time
import atexit
import torch
from pathlib import Path
import optuna
from wind_forecasting.utils import db_utils
from lightning.pytorch.utilities import rank_zero_only

def setup_optuna_storage(args, config, rank):
    """
    Sets up the Optuna storage backend based on configuration and handles synchronization
    between workers.
    """
    storage_target = None  # Initialize for all ranks (can be URL string or storage object)
    pg_config = None       # Initialize pg_config

    if "tune" not in args.mode:  # Only setup DB if tuning
        return storage_target, pg_config
        
    # Read new storage type
    storage_type = config.get("optuna", {}).get("storage", {}).get("type", "sqlite")
    logging.info(f"Optuna storage type configured as: {storage_type}")

    if storage_type == "postgresql" or storage_type == "mysql":
        # RDB storage (PostgreSQL or MySQL)
        connection_method = config.get("optuna", {}).get("storage", {}).get("rdb", {}).get("connection_method", "local_managed")
        logging.info(f"RDB connection method: {connection_method}")
        
        if connection_method == "local_managed":
            if storage_type != "postgresql":
                raise ValueError(f"Local managed instance is only supported for PostgreSQL, not for {storage_type}")
            # Setup using PostgreSQL management
            return setup_postgresql(args, config, rank)
        elif connection_method == "external_tcp":
            # Setup external TCP connection to existing RDB
            return setup_rdb_external_tcp(config, rank, storage_type)
        else:
            raise ValueError(f"Unsupported RDB connection method: {connection_method}")
    elif storage_type == "sqlite":
        # SQLite storage
        return setup_sqlite(args, config)
    elif storage_type == "journal":
        # JournalStorage
        return setup_journal(args, config)
    else:
        raise ValueError(f"Unsupported optuna storage type: {storage_type}")

@rank_zero_only
def setup_postgresql_rank_zero(config, restart=False, register_cleanup=True):
    """
    Sets up PostgreSQL instance on rank 0 (primary worker).
    """
    pg_config = None # Initialize pg_config to handle potential errors during setup
    logging.info("Rank 0: Managing PostgreSQL instance...")
    try:
        # Manage instance (init, start, setup user/db)
        # Pass restart flag from args.
        # Explicitly set register_cleanup=False to prevent premature DB shutdown by worker 0's atexit.
        # Cleanup should be handled externally after all workers finish.
        optuna_storage_url, pg_config = db_utils.manage_postgres_instance(
            config,
            restart=restart,
            register_cleanup=False # Disable atexit registration for Optuna runs
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
    # Get SQLite-specific configuration
    sqlite_config = config.get("optuna", {}).get("storage", {}).get("sqlite", {})
    
    # Read path with variable substitution support
    sqlite_rel_path = sqlite_config.get("path", "logging/optuna/optuna_study.db")
    
    # Substitute ${optuna.study_name}
    if "${optuna.study_name}" in sqlite_rel_path:
        study_name = config.get("optuna", {}).get("study_name", "optuna_study")
        # Make study_name filesystem-safe
        safe_study_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in study_name)
        sqlite_rel_path = sqlite_rel_path.replace("${optuna.study_name}", safe_study_name)
    
    # Resolve path to absolute
    sqlite_abs_path = db_utils._resolve_path(sqlite_config, "path", full_config=config, default=sqlite_rel_path)
    
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(sqlite_abs_path), exist_ok=True)
    
    # Build URL with optional parameters
    url_params = []
    
    # Add WAL mode if enabled
    if sqlite_config.get("use_wal", False):
        url_params.append("journal_mode=WAL")
    
    # Add timeout if specified
    if "timeout" in sqlite_config:
        try:
            timeout = float(sqlite_config["timeout"])
            url_params.append(f"timeout={timeout}")
        except ValueError:
             logging.warning(f"Invalid SQLite timeout value: {sqlite_config['timeout']}. Ignoring.")
    
    # Construct final URL
    optuna_storage_url = f"sqlite:///{sqlite_abs_path}"
    if url_params:
        optuna_storage_url += "?" + "&".join(url_params)
    
    logging.info(f"Using SQLite storage URL: {optuna_storage_url}")
    
    # Handle restart for SQLite on rank 0
    restart_sqlite_rank_zero(sqlite_abs_path, restart=args.restart_tuning)
    
    return optuna_storage_url, None

def setup_journal(args, config):
    """
    Sets up JournalStorage for Optuna.
    Returns the storage object itself, not a URL.
    """
    # Get JournalStorage-specific configuration
    journal_config = config.get("optuna", {}).get("storage", {}).get("journal", {})
    
    # Read path with variable substitution support
    journal_rel_path = journal_config.get("path", "logging/optuna/journal")
    
    # Substitute ${optuna.study_name} if present in the path
    if "${optuna.study_name}" in journal_rel_path:
        study_name = config.get("optuna", {}).get("study_name", "optuna_study")
        # Make study_name filesystem-safe
        safe_study_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in study_name)
        journal_rel_path = journal_rel_path.replace("${optuna.study_name}", safe_study_name)
    
    # Resolve path to absolute using the helper function from db_utils
    journal_abs_path = db_utils._resolve_path(journal_config, "path", full_config=config, default=journal_rel_path)
    
    # Ensure directory exists
    os.makedirs(journal_abs_path, exist_ok=True)
    
    # Create JournalStorage
    try:
        logging.info(f"Using JournalStorage with path: {journal_abs_path}")
        
        # Create the storage object using JournalFileStorage
        # TODO: Consider allowing configuration of the file storage backend if needed
        journal_storage = optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(journal_abs_path)
        )
        
        # Return the storage object and None for pg_config
        return journal_storage, None
    except ImportError as e:
        logging.error(f"Failed to create JournalStorage. Please ensure optuna is installed correctly: {e}")
        raise
    except Exception as e:
        logging.error(f"Failed to create JournalStorage at path '{journal_abs_path}': {e}", exc_info=True)
        raise

def setup_rdb_external_tcp(config, rank, storage_type):
    """
    Sets up an external TCP connection to existing PostgreSQL or MySQL database.
    """
    # Get the external_tcp configuration
    external_config = config.get("optuna", {}).get("storage", {}).get("rdb", {}).get("external_tcp", {})
    
    # Required parameters
    host = external_config.get("host")
    port = external_config.get("port")
    database = external_config.get("database")
    username = external_config.get("username")
    password = external_config.get("password", "")  # Default to empty string if not provided
    
    # Validate required parameters
    missing_params = [p for p in ["host", "port", "database", "username"] if not external_config.get(p)]
    if missing_params:
        raise ValueError(f"External TCP connection requires parameters: {', '.join(missing_params)}")
    
    # Construct database URL based on storage type
    # Ensure necessary database drivers are installed (e.g., psycopg2 for postgresql, mysql-connector-python for mysql)
    try:
        if storage_type == "postgresql":
            # Format: postgresql+psycopg2://username:password@host:port/database
            # Using default driver psycopg2
            storage_url = f"postgresql://{username}:{password}@{host}:{port}/{database}"
            
            # Add driver options if specified (e.g., for SSL)
            driver_options = external_config.get("driver_options", {})
            if driver_options:
                options_str = "&".join(f"{k}={v}" for k, v in driver_options.items())
                storage_url += f"?{options_str}"
        
        elif storage_type == "mysql":
            # Format: mysql+mysqlconnector://username:password@host:port/database
            # Using default driver mysqlconnector
            storage_url = f"mysql+mysqlconnector://{username}:{password}@{host}:{port}/{database}"
            
            # Add driver options if specified
            driver_options = external_config.get("driver_options", {})
            if driver_options:
                options_str = "&".join(f"{k}={v}" for k, v in driver_options.items())
                storage_url += f"?{options_str}"
        
        else:
            raise ValueError(f"Unsupported storage type for external TCP connection: {storage_type}")
            
    except Exception as e:
         logging.error(f"Error constructing database URL for {storage_type}: {e}")
         raise

    # Log safely without credentials
    safe_url = storage_url.replace(f":{password}@", ":***@")
    logging.info(f"Rank {rank}: Using external {storage_type} via TCP: {safe_url}")
    
    # Return URL and None for pg_config
    return storage_url, None

def setup_journal(args, config):
    """
    Sets up JournalStorage for Optuna.
    """
    # Get JournalStorage-specific configuration
    journal_config = config.get("optuna", {}).get("storage", {}).get("journal", {})
    
    # Read path with variable substitution support
    journal_rel_path = journal_config.get("path", "logging/optuna/journal")
    
    # Substitute ${optuna.study_name}
    if "${optuna.study_name}" in journal_rel_path:
        study_name = config.get("optuna", {}).get("study_name", "optuna_study")
        # Make study_name filesystem-safe
        safe_study_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in study_name)
        journal_rel_path = journal_rel_path.replace("${optuna.study_name}", safe_study_name)
    
    # Resolve path to absolute
    journal_abs_path = db_utils._resolve_path(config, "optuna.storage.journal.path", full_config=config, default=journal_rel_path)
    
    # Ensure directory exists
    os.makedirs(journal_abs_path, exist_ok=True)
    
    # Create JournalStorage
    try:
        logging.info(f"Using JournalStorage with path: {journal_abs_path}")
        
        # Create the storage object
        journal_storage = optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(journal_abs_path)
        )
        
        return journal_storage, None
    except ImportError as e:
        logging.error(f"Failed to create JournalStorage. Please ensure optuna is installed with the right version: {e}")
        raise
    except Exception as e:
        logging.error(f"Failed to create JournalStorage: {e}")
        raise

def setup_rdb_external_tcp(config, rank, storage_type):
    """
    Sets up an external TCP connection to existing PostgreSQL or MySQL database.
    """
    # Get the external_tcp configuration
    external_config = config.get("optuna", {}).get("storage", {}).get("rdb", {}).get("external_tcp", {})
    
    # Required parameters
    host = external_config.get("host")
    port = external_config.get("port")
    database = external_config.get("database")
    username = external_config.get("username")
    password = external_config.get("password", "")
    
    # Validate required parameters
    if not host:
        raise ValueError("External TCP connection requires 'host' parameter")
    if not port:
        raise ValueError("External TCP connection requires 'port' parameter")
    if not database:
        raise ValueError("External TCP connection requires 'database' parameter")
    if not username:
        raise ValueError("External TCP connection requires 'username' parameter")
    
    # Construct database URL based on storage type
    if storage_type == "postgresql":
        # Build PostgreSQL connection URL
        # Format: postgresql+psycopg2://username:password@host:port/database
        storage_url = f"postgresql://{username}:{password}@{host}:{port}/{database}"
        
        # Add driver options if specified
        driver_options = external_config.get("driver_options", {})
        if driver_options:
            options_str = "&".join(f"{k}={v}" for k, v in driver_options.items())
            storage_url += f"?{options_str}"
    
    elif storage_type == "mysql":
        # Build MySQL connection URL
        # Format: mysql+mysqlconnector://username:password@host:port/database
        storage_url = f"mysql+mysqlconnector://{username}:{password}@{host}:{port}/{database}"
        
        # Add driver options if specified
        driver_options = external_config.get("driver_options", {})
        if driver_options:
            options_str = "&".join(f"{k}={v}" for k, v in driver_options.items())
            storage_url += f"?{options_str}"
    
    else:
        raise ValueError(f"Unsupported storage type for external TCP connection: {storage_type}")
    
    # Log safely without credentials
    safe_url = storage_url.replace(f":{password}@", ":***@")
    logging.info(f"Using external {storage_type} via TCP: {safe_url}")
    
    return storage_url, None