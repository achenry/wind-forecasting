import os
import logging
import time
from shutil import rmtree

from wind_forecasting.utils import db_utils
from lightning.pytorch.utilities import rank_zero_only
from mysql.connector import connect as sql_connect #, MySQLInterfaceError
import optuna
from optuna.storages import JournalStorage, RDBStorage
from optuna.storages.journal import JournalFileBackend

# from gluonts.src.gluonts.nursery.daf import engine


def setup_optuna_storage(db_setup_params, restart_tuning, rank):
    """
    Sets up the Optuna storage backend based on configuration and handles synchronization
    between workers. The backend type and specific paths/settings are determined
    by the db_setup_params dictionary, derived from the YAML configuration.

    Args:
        db_setup_params (dict): Parameters for database setup (e.g., backend, paths, credentials).
        restart_tuning (bool): Whether to restart the tuning study (primarily affects rank 0 actions).
        rank (int): The rank of the current worker.

    Returns:
        tuple: A tuple containing (optuna.storages.BaseStorage, dict or None).
               The second element is connection info (like pg_config for PostgreSQL) or None.
    """
    storage_backend = db_setup_params.get("backend", "sqlite")
    logging.info(f"Setting up Optuna storage using configured backend: {storage_backend}")
    
    storage = None
    connection_info = None

    # if "storage_dir" in db_setup_params:
    #     # Ensure the directory exists
    #     os.makedirs(db_setup_params["storage_dir"], exist_ok=True)
    #     logging.info(f"Using explicitly defined Optuna storage_dir: {db_setup_params['storage_dir']}")

    if storage_backend == "postgresql":
        # Setup for PostgreSQL database
        # Pass only the necessary parameters for postgresql setup
        # setup_postgresql now returns (storage, pg_config)
        storage, connection_info = setup_postgresql(
            db_setup_params=db_setup_params,
            restart_tuning=restart_tuning,
            rank=rank
        )
    elif storage_backend == "sqlite":
        # Setup for SQLite database
        # setup_sqlite now returns (storage, None)
        storage, connection_info = setup_sqlite(
            db_setup_params=db_setup_params, # Pass the full params dict
            restart_tuning=restart_tuning,
            rank=rank
        )
    elif storage_backend == "journal":
        # setup_journal now returns (storage, None)
        storage, connection_info = setup_journal(
            storage_dir=db_setup_params["storage_dir"], # Use resolved storage_dir
            study_name=db_setup_params["study_name"],
            restart_tuning=restart_tuning,
            rank=rank
        )
    elif storage_backend == "mysql":
        # setup_mysql now returns (storage, None) - adjust if mysql info needed
        storage, connection_info = setup_mysql(db_setup_params=db_setup_params,
                              restart_tuning=restart_tuning,
                              rank=rank)
    else:
        raise ValueError(f"Unsupported optuna storage backend: {storage_backend}")

    if storage is None:
         raise RuntimeError(f"Failed to initialize Optuna storage for backend: {storage_backend}")

    return storage, connection_info

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
        
        # Conditionally add SSL parameters and password environment variable if they exist
        for param in ["sslmode", "sslrootcert_path", "db_password_env_var"]:
            if param in db_setup_params:
                pg_params[param] = db_setup_params[param]

        # Check if this is an external PostgreSQL connection
        is_external = (pg_params.get("use_tcp", False) and (
            pg_params.get("db_password_env_var") or
            pg_params.get("sslmode") or
            pg_params.get("sslrootcert_path")
        ))

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
        
        if is_external:
            # Define engine_kwargs with optimized connection pool settings for external DB
            engine_kwargs = {
                "pool_size": 4,
                "max_overflow": 4,
                "pool_timeout": 30,
                "pool_recycle": 1800,
                "pool_pre_ping": True,
                "connect_args": {"application_name": f"optuna_worker_0_main"}
            }
            
            # Log the engine_kwargs
            logging.info(f"Rank 0: Using SQLAlchemy engine_kwargs for external DB: {engine_kwargs}")
            
            # Create RDBStorage with optimized settings for external DB
            storage = RDBStorage(
                url=optuna_storage_url,
                engine_kwargs=engine_kwargs,
                heartbeat_interval=60,
                failed_trial_callback=optuna.storages.RetryFailedTrialCallback(max_retry=3)
            )
        else:
            # For local PostgreSQL (unix socket), use the original simple connection
            storage = RDBStorage(url=optuna_storage_url)
            
        logging.info(f"Rank 0: RDBStorage instance created.")

        if not is_external:
            # Only handle sync file for local PostgreSQL
            if not pg_config or "sync_file" not in pg_config:
                 raise ValueError("Sync file path not generated in pg_config.")
            
            sync_file_path = pg_config["sync_file"]
            sync_file_ready = False
            
            if os.path.exists(sync_file_path) and not restart_tuning:
                try:
                    with open(sync_file_path, 'r') as f:
                        content = f.read().strip()
                        if content == 'ready':
                            sync_file_ready = True
                            logging.info(f"Rank 0: Existing sync file found with 'ready' status: {sync_file_path}")
                        else:
                            logging.warning(f"Rank 0: Sync file exists but has invalid content: '{content}'. Will recreate.")
                            os.remove(sync_file_path)
                except Exception as e:
                    logging.warning(f"Rank 0: Error reading existing sync file: {e}. Will recreate.")
                    if os.path.exists(sync_file_path):
                        os.remove(sync_file_path)
            elif os.path.exists(sync_file_path) and restart_tuning:
                logging.info(f"Rank 0: Removing existing sync file due to restart_tuning=True: {sync_file_path}")
                os.remove(sync_file_path)
            
            if not sync_file_ready:
                with open(sync_file_path, 'w') as f:
                    f.write('ready')
                logging.info(f"Rank 0: PostgreSQL ready. Created sync file: {sync_file_path}")
            else:
                logging.info(f"Rank 0: Using existing sync file: {sync_file_path}")
        
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

    Returns:
        tuple: (optuna.storages.RDBStorage, dict) containing storage and pg_config.
    """
    optuna_storage_url = None
    pg_config = None # Initialize
    storage = None # Initialize

    # Rank 0 is responsible for setting up the database
    if rank == 0:
        # Pass the explicit params dict
        # setup_postgresql_rank_zero returns storage, pg_config
        storage, pg_config = setup_postgresql_rank_zero(db_setup_params, restart_tuning=restart_tuning)
        if storage is None or pg_config is None:
             raise RuntimeError("Rank 0 failed to setup PostgreSQL and return valid storage/config.")

    else:
        # Worker ranks
        try:
            # Filter db_setup_params to only include relevant keys
            pg_params = {
                k: v for k, v in db_setup_params.items() if k in [
                    "backend", "project_root", "pgdata_path", "study_name",
                    "use_socket", "use_tcp", "db_host", "db_port", "db_name",
                    "db_user", "run_cmd_shell", "socket_dir_base", "sync_dir"
                ]
            }
            
            # Add SSL parameters and password environment variable if they exist
            for param in ["sslmode", "sslrootcert_path", "db_password_env_var"]:
                if param in db_setup_params:
                    pg_params[param] = db_setup_params[param]
            
            # Check if this is an external PostgreSQL connection
            is_external = (pg_params.get("use_tcp", False) and (
                pg_params.get("db_password_env_var") or
                pg_params.get("sslmode") or
                pg_params.get("sslrootcert_path")
            ))
            
            if is_external:
                # For external databases, bypass sync file and directly connect
                logging.info(f"Rank {rank}: Using external PostgreSQL connection")
                optuna_storage_url = db_utils.get_optuna_storage_url(pg_params)
                
                # Define engine_kwargs with optimized connection pool settings and dynamic application_name
                engine_kwargs = {
                    "pool_size": 4,
                    "max_overflow": 4,
                    "pool_timeout": 30,
                    "pool_recycle": 1800,
                    "pool_pre_ping": True,
                    "connect_args": {"application_name": f"optuna_worker_{rank}"}
                }
                
                # Log the engine_kwargs and rank
                logging.info(f"Rank {rank}: Using SQLAlchemy engine_kwargs: {engine_kwargs}")
                
                # Create RDBStorage with optimized settings
                storage = RDBStorage(
                    url=optuna_storage_url,
                    engine_kwargs=engine_kwargs,
                    heartbeat_interval=60,
                    failed_trial_callback=optuna.storages.RetryFailedTrialCallback(max_retry=3)
                )
                
                # Test connection
                _ = storage.get_all_studies()
                logging.info(f"Rank {rank}: Successfully connected to external PostgreSQL DB")
                pg_config = pg_params  # Store connection params
            else:
                # For local PostgreSQL, use sync file mechanism
                pg_config = db_utils._generate_pg_config(**pg_params)
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
                # Test connection
                _ = storage.get_all_studies()
                logging.info(f"Rank {rank}: Successfully connected to PostgreSQL DB.")


        except Exception as e:
            logging.error(f"Rank {rank}: Failed during PostgreSQL sync/config generation: {e}", exc_info=True)
            raise

    if storage is None:
        # This should only happen if rank != 0 and setup failed, or rank == 0 and setup failed.
        raise RuntimeError(f"Rank {rank}: Optuna storage object was not successfully created for PostgreSQL.")

    # Return storage and pg_config (pg_config might be None for rank != 0 if only URL was needed)
    # However, pg_config is generated for workers too to find sync file, so it should exist.
    return storage, pg_config

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

def setup_sqlite(db_setup_params, restart_tuning, rank):
    """
    Sets up SQLite storage for Optuna, prioritizing explicit path if provided.
    Handles restart logic on rank 0.

    Returns:
        tuple: (optuna.storages.RDBStorage, None)
    """
    # Prioritize explicit sqlite_path from config if available
    if "sqlite_path" in db_setup_params and db_setup_params["sqlite_path"]:
        db_path = db_setup_params["sqlite_path"]
        # Ensure the directory exists if the path is relative or absolute
        db_dir = os.path.dirname(os.path.abspath(db_path))
        if db_dir: # Check if dirname is not empty (e.g., for root files)
             os.makedirs(db_dir, exist_ok=True)
        logging.info(f"Using explicit SQLite path from config: {os.path.abspath(db_path)}")
    else:
        # Fallback: Construct the SQLite URL based on storage_dir and study_name
        sqlite_storage_dir = db_setup_params.get("storage_dir", ".") # Default to current dir if not set
        study_name = db_setup_params.get("study_name", "optuna_study")
        # Ensure storage dir exists
        os.makedirs(sqlite_storage_dir, exist_ok=True)
        db_path = os.path.join(sqlite_storage_dir, f'{study_name}.db')
        logging.info(f"Constructed SQLite path: {os.path.abspath(db_path)}")

    abs_db_path = os.path.abspath(db_path)

    # Handle restart logic (rank 0 only)
    if rank == 0 and restart_tuning:
       if os.path.exists(abs_db_path):
           restart_sqlite_rank_zero(abs_db_path) # Use the dedicated restart function
       else:
           logging.warning(f"Rank 0: --restart_tuning set, but SQLite DB not found at {abs_db_path} to delete.")

    # SQLite with WAL is generally recommended, but can cause issues with NFS.
    # Provide an option to disable WAL if needed via config.
    use_wal = db_setup_params.get("sqlite_wal", True)
    timeout = db_setup_params.get("sqlite_timeout", 60) # Default 60s timeout
    
    # SQLite connection parameters
    # Instead of passing connect_args directly to RDBStorage (which doesn't accept it),
    # we'll include these parameters in the URL if possible
    url_params = []
    
    # Add timeout parameter to URL
    url_params.append(f"timeout={timeout}")
    
    # Handle WAL mode
    if not use_wal:
        url_params.append("journal_mode=DELETE")
        logging.info("SQLite WAL mode disabled via URL parameter")
    else:
        url_params.append("journal_mode=WAL")
        
    # Construct URL with parameters
    params_string = "&".join(url_params)
    optuna_storage_url = f"sqlite:///{abs_db_path}?{params_string}"
    
    logging.info(f"Using SQLite storage URL: {optuna_storage_url} (WAL enabled: {use_wal}, Timeout: {timeout}s)")

    # All ranks need to connect to the SQLite DB if used (e.g., for reading study).
    # Concurrent reads are generally safe. Concurrent writes during tuning
    # would require a filesystem supporting proper locking (not NFS).
    # Since this setup is primarily for reading tuned params, we allow all ranks to connect.
    try:
        # Only pass the URL to RDBStorage, with connection parameters included in the URL
        storage = RDBStorage(
            url=optuna_storage_url
        )
        # Test connection
        _ = storage.get_all_studies()
        logging.info(f"Rank {rank}: Successfully connected to SQLite DB: {optuna_storage_url}")
    except Exception as e:
        logging.error(f"Rank {rank}: Failed to create or connect to RDBStorage for SQLite path {optuna_storage_url}: {e}", exc_info=True)
        raise RuntimeError(f"Failed to initialize storage for {optuna_storage_url}") from e

    return storage, None # Return None for connection_info

def setup_journal(storage_dir, study_name, restart_tuning, rank):
    """
    Sets up JournalFile storage for Optuna.

    Returns:
        tuple: (optuna.storages.JournalStorage, None)
    """
    # Ensure storage directory exists
    os.makedirs(storage_dir, exist_ok=True)
    journal_file_path = os.path.join(storage_dir, f"{study_name}.log") # Use .log extension
    abs_journal_path = os.path.abspath(journal_file_path)

    logging.info(f"Rank {rank}: Setting up Journal storage at: {abs_journal_path}")

    # Handle restart logic (rank 0 only)
    if rank == 0 and restart_tuning:
        restart_journal_rank_zero(abs_journal_path) # Use the dedicated restart function

    try:
        # All ranks connect to the same journal file
        storage = JournalStorage(JournalFileBackend(file_path=abs_journal_path))
        logging.info(f"Rank {rank}: Successfully set up Journal storage.")
    except Exception as e:
        logging.error(f"Rank {rank}: Failed to set up Journal storage at {abs_journal_path}: {e}", exc_info=True)
        raise RuntimeError(f"Failed to initialize Journal storage for {abs_journal_path}") from e

    return storage, None # Return None for connection_info
    
def setup_mysql(db_setup_params, restart_tuning, rank):
    """
    Sets up MySQL storage for Optuna.

    Returns:
        tuple: (optuna.storages.RDBStorage, dict or None) - Returns connection info if needed.
    """
    db_name = db_setup_params['study_name']
    db_host = db_setup_params["db_host"]
    db_user = db_setup_params["db_user"]
    db_password = db_setup_params.get("db_password") # Get password if provided
    db_port = db_setup_params.get("db_port", 3306) # Default MySQL port

    logging.info(f"Rank {rank}: Setting up MySQL connection to host={db_host}, user={db_user}, db={db_name}")

    # Construct the Optuna storage URL
    # Format: mysql://[user[:password]@]host[:port]/database
    url_user_part = db_user
    if db_password:
        url_user_part += f":{db_password}"
    optuna_storage_url = f"mysql+mysqlconnector://{url_user_part}@{db_host}:{db_port}/{db_name}"

    # TODO HIGH there are issues when creating database here, with table studies already existing
    # TODO HIGH how to respond to restart_tuning?
    # Rank 0 handles database/study creation and restart
    # TODO BEGIN TESTING
    if rank == 0:
        connection = None
        try:
            # Try connecting without specifying the database first to check server access and create DB if needed
            connection = sql_connect(host=db_host, user=db_user, password=db_password, port=db_port)
            cursor = connection.cursor()
            cursor.execute("SHOW DATABASES")
            databases = [item[0] for item in cursor.fetchall()]

            if db_name not in databases:
                logging.info(f"Rank 0: Database '{db_name}' not found in list {databases}. Creating database.")
                cursor.execute(f"CREATE DATABASE {db_name}")
                connection.commit()
                logging.info(f"Rank 0: Database '{db_name}' created successfully.")
            elif restart_tuning:
                logging.info(f"Rank 0: Database '{db_name}' already exists.")
                logging.warning(f"Rank 0: --restart_tuning set. Dropping and recreating Optuna tables in database '{db_name}'.")
                
                # Optuna's RDBStorage handles table creation/migration.
                # To truly restart, we might need to drop tables, but let's rely on Optuna's behavior first.
                # A simpler approach for restart might be to use a new study_name.
                # For now, we'll just let RDBStorage connect. If load_if_exists=False is used later,
                # Optuna might handle the study deletion/creation.
                # Let's log a warning that restart might require manual intervention or new study name.
                logging.warning("Rank 0: Actual table dropping for restart is not implemented here. Use a new study name or manage tables manually if a full reset is needed.")
                cursor.execute(f"DROP DATABASE IF EXISTS {db_name}")
                cursor.execute(f"CREATE DATABASE {db_name}")
                connection.commit()
                
                # Example (requires permissions):
                
                # storage_temp = RDBStorage(url=optuna_storage_url)
                # studies = storage_temp.get_all_studies()
                # for s in studies:
                #     if db_setup_params['study_name'] in s.study_name: # Check if study name matches DB name contextually
                #         logging.info(f"Rank 0: Deleting study {s.study_name} from database '{db_name}'")
                #         storage_temp.delete_study(s._study_id)

        except Exception as e:
            logging.error(f"Rank 0: Failed to connect to MySQL server or manage database '{db_name}': {e}", exc_info=True)
            raise RuntimeError(f"Rank 0 failed MySQL setup for {db_name}") from e
        finally:
            cursor.close()
            if connection:
                connection.close()
        # END TESTING
    
    # All ranks create the RDBStorage instance
    try:
        # Add connect_args for timeout, etc. if needed
        
        storage = RDBStorage(url=optuna_storage_url, heartbeat_interval=60)
        # Test connection
        # _ = storage.get_all_studies()
        logging.info(f"Rank {rank}: Successfully connected to MySQL DB using URL: mysql+mysqlconnector://{db_user}@***:{db_port}/{db_name}")
    # except MySQLInterfaceError as e:
    #     storage = RDBStorage(url=optuna_storage_url, skip_table_creation=True)
    #     # Test connection
    #     # _ = stosacrage.get_all_studies()
    #     logging.info(f"Rank {rank}: Successfully connected to MySQL DB using URL: mysql+mysqlconnector://{db_user}@***:{db_port}/{db_name}, skipping table creation due to error {e}.")
    except Exception as e:
        logging.error(f"Rank {rank}: Failed to create RDBStorage for MySQL URL {optuna_storage_url}: {e}", exc_info=True)
        raise RuntimeError(f"Failed MySQL RDBStorage initialization for rank {rank}") from e

    # Return storage and potentially connection details if needed elsewhere
    # For now, return None for connection_info similar to SQLite/Journal
    connection_info = {"db_host": db_host, "db_port": db_port, "db_name": db_name, "db_user": db_user} # Example
    return storage, connection_info # Return None for now