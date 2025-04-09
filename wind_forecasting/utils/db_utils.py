import os
import subprocess
import logging
import time
import shutil
import getpass  # To get current username for initdb/psql
import atexit   # To register cleanup function
from pathlib import Path # Import Path for robust path calculation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _run_cmd(command, cwd=None, shell=False, check=True, env_override=None):
    """Helper function to run shell commands and log output/errors."""
    log_cmd = ' '.join(command) if isinstance(command, list) else command
    logging.info(f"Running command (shell={shell}): {log_cmd}")
    try:
        # --- Create environment for the subprocess ---
        # Start with current environment or an empty dict
        cmd_env = os.environ.copy() if env_override is None else env_override.copy()

        # Ensure PATH is present
        if "PATH" not in cmd_env:
            logging.warning("PATH not found in environment, command might fail.")
            # Attempt to reconstruct a basic PATH if missing
            cmd_env["PATH"] = os.defpath

        # Prioritize CAPTURED_LD_LIBRARY_PATH if available
        captured_ld_path = os.environ.get("CAPTURED_LD_LIBRARY_PATH")
        if captured_ld_path:
            cmd_env["LD_LIBRARY_PATH"] = captured_ld_path
            logging.info(f"Using captured LD_LIBRARY_PATH for subprocess: {captured_ld_path}")
        elif "LD_LIBRARY_PATH" in cmd_env:
             logging.info(f"Using existing LD_LIBRARY_PATH for subprocess: {cmd_env['LD_LIBRARY_PATH']}")
        else:
             logging.warning("LD_LIBRARY_PATH not found in environment for subprocess.")

        # Ensure essential user variables are present
        for env_var in ["USER", "LOGNAME", "HOME"]:
             if env_var not in cmd_env and env_var in os.environ:
                 cmd_env[env_var] = os.environ[env_var]

        process = subprocess.run(
            command,
            cwd=cwd,
            shell=shell,
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=cmd_env
        )

        if process.stdout:
            logging.info(f"Command stdout:\n{process.stdout.strip()}")
        if process.stderr:
            logging.warning(f"Command stderr:\n{process.stderr.strip()}")
        return process
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with exit code {e.returncode}")
        if e.stdout:
            logging.error(f"Failed command stdout:\n{e.stdout.strip()}")
        if e.stderr:
            logging.error(f"Failed command stderr:\n{e.stderr.strip()}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred while running command: {e}")
        raise

def get_optuna_storage_url(pg_config):
    """Constructs the Optuna storage URL based on the pre-computed pg_config."""
    db_user = pg_config["dbuser"]
    db_name = pg_config["dbname"]

    if pg_config["use_socket"]:
        socket_dir = pg_config["socket_dir"]
        if not socket_dir:
             raise ValueError("Socket directory is not defined for socket connection.")
        # Construct URL using socket path
        url = f"postgresql://{db_user}@/{db_name}?host={socket_dir}"
        logging.info(f"Constructed PostgreSQL Optuna URL (socket): {url.split('@')[0]}@...") # Log safely
        return url
    elif pg_config.get("use_tcp", False):
        # Construct URL using TCP/IP
        db_host = pg_config.get("db_host", "localhost")
        db_port = pg_config.get("db_port", 5432)
        # Assuming no password needed due to trust auth or other methods handled by pg_hba.conf
        url = f"postgresql://{db_user}@{db_host}:{db_port}/{db_name}"
        logging.info(f"Constructed PostgreSQL Optuna URL (TCP/IP): {url.split('@')[0]}@...") # Log safely
        return url
    else:
        raise ValueError("PostgreSQL connection type (socket or TCP) not specified or supported.")

def delete_postgres_data(pg_config, raise_on_error=True):
    """Stops server and removes PGDATA directory."""
    pgdata = pg_config["pgdata"]
    if os.path.exists(pgdata):
        logging.warning(f"Attempting to remove existing PostgreSQL data directory: {pgdata}")
        try:
            stop_postgres(pg_config, raise_on_error=False)
        except Exception as e:
            logging.warning(f"Ignoring error during pre-delete server stop: {e}")

        try:
            shutil.rmtree(pgdata)
            logging.info(f"Successfully removed PostgreSQL data directory: {pgdata}")
        except OSError as e:
            logging.error(f"Error removing directory {pgdata}: {e}")
            if raise_on_error:
                 raise
    else:
        logging.info(f"PostgreSQL data directory {pgdata} does not exist, nothing to remove.")

def init_postgres(pg_config):
    """Initializes a new PostgreSQL database cluster."""
    pgdata = pg_config["pgdata"]
    db_name = pg_config["dbname"]
    db_user = pg_config["dbuser"]
    job_owner = pg_config["job_owner"]
    needs_db_setup = False

    initdb_path = pg_config["initdb_path"]

    if not os.path.exists(os.path.join(pgdata, "base")):
        logging.info(f"Initializing PostgreSQL data directory ({pgdata})...")
        init_cmd = [
            initdb_path, # Use full path
            "--no-locale",
            "--auth=trust",
            "-E UTF8",
            # Omit -U; initdb uses OS user by default
            "-D", str(pgdata)
        ]

        _run_cmd(init_cmd, shell=pg_config.get("run_cmd_shell", False))
        logging.info("PostgreSQL data directory initialized.")
        hba_conf_path = os.path.join(pgdata, "pg_hba.conf")

        # Brief check/wait for pg_hba.conf
        if not os.path.exists(hba_conf_path):
             time.sleep(1)
             if not os.path.exists(hba_conf_path):
                  logging.error(f"{hba_conf_path} did not appear after initdb.")
                  raise FileNotFoundError(f"{hba_conf_path} did not appear after initdb.")
        logging.info(f"{hba_conf_path} found.")

        # Configure pg_hba.conf for local socket access
        logging.info(f"Modifying {hba_conf_path} for local trust authentication...")
        hba_lines = [
            "# TYPE  DATABASE        USER            ADDRESS                 METHOD",
            f"local   postgres        {job_owner}                             trust", # Superuser access to postgres db
            # Allow superuser access to all DBs
            f"local   all             {job_owner}                             trust",
            # Allow optuna user access to its DB
            f"local   {db_name}       {db_user}                               trust",
        ]
        try:
            with open(hba_conf_path, 'w') as f:
                f.write("\n".join(hba_lines) + "\n")
            logging.info("pg_hba.conf modified.")
            needs_db_setup = True
        except Exception as e:
            logging.error(f"Failed to write pg_hba.conf: {e}")
            raise
    else:
        logging.info(f"PostgreSQL data directory {pgdata} already exists.")

    return needs_db_setup

def setup_db_user(pg_config):
    """Creates the Optuna database and user if they don't exist."""
    job_owner = pg_config["job_owner"]
    db_user = pg_config["dbuser"]
    db_name = pg_config["dbname"]
    psql_path = pg_config["psql_path"]
    run_cmd_shell = pg_config.get("run_cmd_shell", False)

    # Connection parameters for psql
    psql_conn_opts = []
    if pg_config.get("use_socket"):
        socket_dir = pg_config.get("socket_dir")
        if not socket_dir:
            raise ValueError("Socket directory required for database/user setup when using socket connection.")
        psql_conn_opts.extend(["-h", socket_dir])
    elif pg_config.get("use_tcp"):
        db_host = pg_config.get("db_host")
        db_port = pg_config.get("db_port")
        if not db_host or not db_port:
             raise ValueError("Host and Port required for database/user setup when using TCP connection.")
        psql_conn_opts.extend(["-h", db_host, "-p", str(db_port)])
    else:
        raise ValueError("Could not determine connection type (socket/tcp) for database/user setup.")

    logging.info("Performing first-time database setup (User/DB creation)...")
    # Wait for server startup
    time.sleep(2)

    # Create Optuna User (db_user) if it doesn't exist
    check_user_cmd_list = [psql_path] + psql_conn_opts + [
        "-U", job_owner, "-d", "postgres",
        "-tAc", f"SELECT 1 FROM pg_roles WHERE rolname='{db_user}'"
    ]
    user_exists = _run_cmd(check_user_cmd_list, check=False).stdout.strip() == '1'

    if not user_exists:
        create_user_cmd_list = [psql_path] + psql_conn_opts + [
            "-U", job_owner, "-d", "postgres",
            "-c", f"CREATE USER {db_user};"
        ]
        _run_cmd(create_user_cmd_list, shell=run_cmd_shell)
        logging.info(f"Created PostgreSQL user: {db_user}")
    else:
        logging.info(f"PostgreSQL user {db_user} already exists.")

    # Create Optuna Database (db_name) if it doesn't exist
    check_db_cmd_list = [psql_path] + psql_conn_opts + [
        "-U", job_owner, "-d", "postgres",
        "-tAc", f"SELECT 1 FROM pg_database WHERE datname='{db_name}'"
    ]
    db_exists = _run_cmd(check_db_cmd_list, check=False).stdout.strip() == '1'

    if not db_exists:
        create_db_cmd_list = [psql_path] + psql_conn_opts + [
            "-U", job_owner, "-d", "postgres",
            "-c", f"CREATE DATABASE {db_name} OWNER {db_user};"
        ]
        _run_cmd(create_db_cmd_list, shell=run_cmd_shell)
        logging.info(f"Created PostgreSQL database: {db_name} owned by {db_user}")
    else:
        logging.info(f"PostgreSQL database {db_name} already exists.")

    logging.info("Database setup complete.")


def start_postgres(pg_config):
    """Starts the PostgreSQL server if not already running."""
    pgdata = pg_config["pgdata"]
    logfile = os.path.join(pgdata, "logfile.log")
    pg_ctl_path = pg_config["pg_ctl_path"]
    run_cmd_shell = pg_config.get("run_cmd_shell", False)

    # Base start options
    start_opts = ["-w", "-D", str(pgdata), "-l", logfile]

    logging.info("Starting PostgreSQL server...")
    # Check if server is already running
    status_cmd = [pg_ctl_path, "status", "-D", str(pgdata)]
    status_result = _run_cmd(status_cmd, check=False, shell=run_cmd_shell)

    if status_result.returncode == 0:
        logging.info("PostgreSQL server is already running.")
        return

    # If not running (exit code 3), try starting
    if status_result.returncode == 3:
        # Add connection-specific options
        if pg_config.get("use_socket"):
            socket_dir = pg_config.get("socket_dir")
            if not socket_dir:
                 raise ValueError("Socket directory required to start PostgreSQL with socket connection.")
            # Ensure the socket directory exists before starting
            os.makedirs(socket_dir, exist_ok=True)
            start_opts.extend(["-o", f"-c unix_socket_directories='{socket_dir}'"])
            logging.info(f"Starting PostgreSQL with socket directory: {socket_dir}")
        elif pg_config.get("use_tcp"):
            db_host = pg_config.get("db_host", "localhost") # Use configured host/port for logging
            db_port = pg_config.get("db_port", 5432)
            logging.info(f"Starting PostgreSQL (expecting TCP connection on {db_host}:{db_port} based on config)...")
            # No specific pg_ctl options added here for TCP listening by default
        else:
             raise ValueError("Could not determine connection type (socket/tcp) for starting PostgreSQL.")

        start_cmd_list = [pg_ctl_path, "start"] + start_opts
        _run_cmd(start_cmd_list, shell=run_cmd_shell)
        logging.info("PostgreSQL server started successfully.")
    else:
        logging.error(f"pg_ctl status returned unexpected code {status_result.returncode}. Check logs.")
        raise RuntimeError("Failed to determine PostgreSQL server status.")


def stop_postgres(pg_config, raise_on_error=True):
    """Stops the PostgreSQL server and cleans up socket dir."""
    pgdata = pg_config["pgdata"]
    socket_dir = pg_config["socket_dir"]
    pg_ctl_path = pg_config["pg_ctl_path"]

    logging.info("Stopping PostgreSQL server...")
    # Construct stop command
    stop_cmd_list = [
        pg_ctl_path, "stop", "-w",
        "-D", str(pgdata),
        "-m", "fast"
    ]
    try:
        _run_cmd(stop_cmd_list, shell=pg_config.get("run_cmd_shell", False))
        logging.info("PostgreSQL server stopped.")
    except Exception as e:
        logging.warning(f"pg_ctl stop failed (maybe server was not running?): {e}")
        if raise_on_error:
            raise
    finally:
        # Clean up socket directory
        if socket_dir and os.path.exists(socket_dir):
            try:
                shutil.rmtree(socket_dir)
                logging.info(f"Removed socket directory: {socket_dir}")
            except Exception as e:
                logging.warning(f"Failed to remove socket directory {socket_dir}: {e}")

# Global variable to hold the generated config for atexit cleanup
_managed_pg_config = None

def _resolve_path(key_config, key, full_config, default=None):
    """
    Resolves a path potentially containing variables like ${logging.optuna_dir},
    using the full_config for variable lookups and project root determination.
    """
    path_str = key_config.get(key, default)
    if not path_str:
        return None

    # --- Variable Substitution ---
    max_iterations = 5
    iterations = 0
    while "${" in path_str and iterations < max_iterations:
        original_path_str = path_str # Keep track for error messages
        substituted = False # Flag to check if any substitution happened in this iteration

        if "${logging.optuna_dir}" in path_str:
            # Look up in the full config dictionary
            optuna_dir = full_config.get("logging", {}).get("optuna_dir")
            if not optuna_dir:
                raise ValueError(f"Cannot resolve variable in path '{original_path_str}': logging.optuna_dir is not defined in the configuration.")
            # Ensure optuna_dir itself is an absolute path before substituting
            if not Path(optuna_dir).is_absolute():
                 project_root_str_for_optuna = full_config.get("experiment", {}).get("project_root", os.getcwd())
                 optuna_dir = str((Path(project_root_str_for_optuna) / Path(optuna_dir)).resolve())
            path_str = path_str.replace("${logging.optuna_dir}", optuna_dir)
            substituted = True

        if not substituted: # No substitution happened, break loop
             break
        iterations += 1

    if iterations >= max_iterations:
         logging.warning(f"Path resolution exceeded max iterations for '{key_config.get(key, default)}'. Result: '{path_str}'")

    # --- Resolve to Absolute Path ---
    if Path(path_str).is_absolute():
        resolved_path = Path(path_str)
    else:
        project_root_str = full_config.get("experiment", {}).get("project_root")
        if not project_root_str:
             logging.warning("experiment.project_root not defined in full_config, assuming current working directory for relative path resolution.")
             project_root = Path(os.getcwd())
        else:
             project_root = Path(project_root_str)
        resolved_path = (project_root / Path(path_str)).resolve()

    return str(resolved_path)


def _generate_pg_config(config):
    """
    Generates the pg_config dictionary for a LOCAL MANAGED PostgreSQL instance.
    """
    global _managed_pg_config # Allow modification of the global var

    storage_config = config.get("optuna", {}).get("storage", {})
    rdb_config = storage_config.get("rdb", {})
    local_managed_config = rdb_config.get("local_managed", {})

    if storage_config.get("type") != "postgresql" or rdb_config.get("connection_method") != "local_managed":
         raise ValueError("This function should only be called for storage type 'postgresql' with connection method 'local_managed'.")

    # Determine Project Root
    project_root_str = config.get("experiment", {}).get("project_root")
    if not project_root_str:
        try:
            project_root = Path(__file__).resolve().parents[2]
            logging.info(f"Determined project root from script location: {project_root}")
        except IndexError:
            logging.warning("Could not determine project root from script location. Falling back to CWD.")
            project_root = Path(os.getcwd())
    else:
        project_root = Path(project_root_str).resolve()
        logging.info(f"Using project root from config: {project_root}")

    # --- PGDATA Path ---
    pgdata_path_base_rel = local_managed_config.get("pgdata_path_base")
    if not pgdata_path_base_rel:
        raise ValueError("Missing 'pgdata_path_base' in optuna.storage.rdb.local_managed configuration.")
    # Base path is relative to project root
    pgdata_base_abs = (project_root / Path(pgdata_path_base_rel)).resolve()
    # Generate directory name based on Optuna study name for persistence
    study_name = config.get("optuna", {}).get("study_name")
    if not study_name:
        raise ValueError("Missing 'study_name' in optuna configuration, needed for PGDATA path.")
    # Make study name filesystem-safe
    safe_study_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in study_name)
    pgdata_dir_name = f"pg_data_{safe_study_name}" # Consistent name based on study
    pgdata_path_abs = pgdata_base_abs / pgdata_dir_name
    # Ensure the base directory exists (parent of the specific study data dir)
    os.makedirs(pgdata_base_abs, exist_ok=True)
    # The specific study data directory (pgdata_path_abs) will be created by initdb if it doesn't exist
    logging.info(f"Using persistent PGDATA path based on study name: {pgdata_path_abs}")

    # --- Connection Type Configuration (Socket/TCP) ---
    connection_type = local_managed_config.get("connection_type", "socket").lower()
    if connection_type not in ["socket", "tcp"]:
        raise ValueError(f"Invalid connection_type '{connection_type}' in local_managed config. Must be 'socket' or 'tcp'.")

    use_socket = (connection_type == "socket")
    use_tcp = (connection_type == "tcp")

    socket_dir = None
    db_host = None
    db_port = None

    if use_socket:
        # Resolve socket_dir_base relative to project root using local_managed_config
        socket_base_str = _resolve_path(local_managed_config, "socket_dir_base", full_config=config)
        if not socket_base_str:
             # Default to $TMPDIR or /tmp if not specified
             tmp_dir = os.environ.get("TMPDIR", "/tmp")
             logging.info(f"socket_dir_base not specified, using system temp dir: {tmp_dir}")
             socket_base_path = Path(tmp_dir)
        else:
             socket_base_path = Path(socket_base_str)

        job_id_for_socket = os.environ.get("SLURM_JOB_ID", os.getpid()) # Unique socket per job
        socket_dir_name = f"pg_socket_{job_id_for_socket}"
        socket_dir_path = (socket_base_path / socket_dir_name).resolve()
        os.makedirs(socket_dir_path, exist_ok=True)
        socket_dir = str(socket_dir_path)
        logging.info(f"Using socket directory: {socket_dir}")
    elif use_tcp:
        # Read host/port from local_managed_config
        db_host = local_managed_config.get("db_host", "localhost")
        db_port = local_managed_config.get("db_port", 5432)
        logging.info(f"Configured for local TCP/IP connection: host={db_host}, port={db_port}")

    # --- Sync Directory ---
    # Resolve sync_dir using local_managed_config
    sync_dir_str = _resolve_path(local_managed_config, "sync_dir", full_config=config)
    if not sync_dir_str:
        # Default to a 'sync' subdir within optuna_dir if sync_dir not specified
        # Resolve optuna_dir first using the full config
        optuna_dir_str = _resolve_path(config.get("logging", {}), "optuna_dir", full_config=config, default="logging/optuna") # Pass full config
        if not optuna_dir_str:
             raise ValueError("Cannot determine default sync_dir because logging.optuna_dir is not defined.")
        # Default path is relative to resolved optuna_dir
        sync_dir_str = str((Path(optuna_dir_str) / "sync").resolve()) # This should be fine as optuna_dir_str is now absolute
        logging.info(f"sync_dir not specified, defaulting relative to resolved optuna_dir: {sync_dir_str}")
    sync_dir_path = Path(sync_dir_str)
    os.makedirs(sync_dir_path, exist_ok=True)
    sync_file = str(sync_dir_path / f"optuna_pg_ready_{os.environ.get('SLURM_JOB_ID', os.getpid())}.sync")
    logging.info(f"Using sync file: {sync_file}")

    # --- Other Settings ---
    # Get DB name and user from local_managed_config
    db_name = local_managed_config.get("db_name", "optuna_study_db")
    # Substitute ${optuna.study_name} if present in the db_name
    if "${optuna.study_name}" in db_name:
         db_name = db_name.replace("${optuna.study_name}", safe_study_name) # Use safe_study_name generated earlier
    db_user = local_managed_config.get("db_user", "optuna_user")
    run_cmd_shell = local_managed_config.get("run_cmd_shell", False) # Get shell preference

    # --- PostgreSQL Binaries ---
    pg_bin_dir = os.environ.get("POSTGRES_BIN_DIR")
    if not pg_bin_dir or not os.path.isdir(pg_bin_dir):
        logging.error(f"POSTGRES_BIN_DIR environment variable not set or invalid: '{pg_bin_dir}'. Cannot find PostgreSQL executables.")
        raise ValueError("POSTGRES_BIN_DIR environment variable not set or invalid.")

    pg_config = {
        "pgdata": str(pgdata_path_abs),
        "dbname": db_name,
        "dbuser": db_user,
        "use_socket": use_socket,
        "socket_dir": socket_dir,
        "use_tcp": use_tcp,
        "db_host": db_host,
        "db_port": db_port,
        "job_owner": getpass.getuser(),
        "initdb_path": os.path.join(pg_bin_dir, "initdb"),
        "pg_ctl_path": os.path.join(pg_bin_dir, "pg_ctl"),
        "psql_path": os.path.join(pg_bin_dir, "psql"),
        "run_cmd_shell": run_cmd_shell, # Store shell preference
        "sync_file": sync_file, # Store sync file path
    }

    # Verify executables exist
    for key, path in pg_config.items():
        if key.endswith("_path") and path and not os.path.exists(path):
             logging.error(f"PostgreSQL executable not found at expected path: {path} (derived from POSTGRES_BIN_DIR={pg_bin_dir})")
             raise FileNotFoundError(f"PostgreSQL executable '{os.path.basename(path)}' not found at expected path: {path}")

    _managed_pg_config = pg_config # Store globally for atexit
    return pg_config

def _cleanup_postgres():
    """Function to be called by atexit to stop the server and clean up."""
    global _managed_pg_config
    if _managed_pg_config:
        logging.info("atexit: Running PostgreSQL cleanup...")
        try:
            stop_postgres(_managed_pg_config, raise_on_error=False)
        except Exception as e:
            logging.error(f"atexit: Error during PostgreSQL cleanup: {e}")
        _managed_pg_config = None # Prevent duplicate cleanup
    else:
        logging.info("atexit: No managed PostgreSQL instance found to clean up.")

def manage_postgres_instance(config, restart=False, register_cleanup=True):
    """
    Main function to manage the PostgreSQL instance for a job called only by rank 0.
    """
    logging.info("Managing PostgreSQL instance...")
    # Generate the config dictionary
    pg_config = _generate_pg_config(config) # This also sets _managed_pg_config

    if restart: # If --restart_tuning flag was passed
        # Pass the consistent pg_config
        logging.info("Performing cleanup due to --restart_tuning flag.")
        delete_postgres_data(pg_config) # Initial cleanup if restarting

    # Pass the consistent pg_config
    needs_setup = init_postgres(pg_config)
    start_postgres(pg_config)

    if needs_setup:
        # Pass the consistent pg_config
        setup_db_user(pg_config)

    # Pass the consistent pg_config
    storage_url = get_optuna_storage_url(pg_config)
    logging.info("PostgreSQL instance is ready.")

    # Register cleanup function if requested (usually only for rank 0)
    if register_cleanup:
        logging.info("Registering atexit cleanup hook for PostgreSQL.")
        atexit.register(_cleanup_postgres)

    return storage_url, pg_config # Return URL and the generated config