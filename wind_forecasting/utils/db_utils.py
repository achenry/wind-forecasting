import os
import subprocess
import logging
import time
import shutil
import getpass # To get current username for initdb/psql
from pathlib import Path # Import Path for robust path calculation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Default to shell=False for better argument handling when command is a list
def _run_cmd(command, cwd=None, shell=False, check=True):
    """Helper function to run shell commands and log output/errors."""
    # Log command differently based on whether it's a list or string
    log_cmd = ' '.join(command) if isinstance(command, list) else command
    logging.info(f"Running command (shell={shell}): {log_cmd}")
    try:
        process = subprocess.run(
            command,
            cwd=cwd,
            shell=shell,
            check=check, # Raise exception on non-zero exit code if True
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=os.environ.copy() # Pass parent environment to find commands
        )
        if process.stdout:
            logging.info(f"Command stdout:\n{process.stdout.strip()}")
        if process.stderr:
            # Log stderr as warning even if check=False or command succeeded
            logging.warning(f"Command stderr:\n{process.stderr.strip()}")
        return process
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with exit code {e.returncode}")
        if e.stdout:
            logging.error(f"Failed command stdout:\n{e.stdout.strip()}")
        if e.stderr:
            logging.error(f"Failed command stderr:\n{e.stderr.strip()}")
        raise # Re-raise the exception after logging
    except Exception as e:
        logging.error(f"An unexpected error occurred while running command: {e}")
        raise

def _get_pg_config(config):
    """Extracts PostgreSQL config and resolves paths."""
    storage_config = config.get("optuna", {}).get("storage", {})
    if storage_config.get("backend") != "postgresql":
        raise ValueError("Database backend is not configured as 'postgresql' in YAML.")

    try:
        project_root = Path(__file__).resolve().parents[2]
        logging.info(f"Determined project root: {project_root}")
    except IndexError:
        logging.error("Could not determine project root based on script location. Falling back to CWD.")
        project_root = Path(os.getcwd()) # Fallback to current working directory

    pgdata_path_rel = storage_config.get("pgdata_path")
    if not pgdata_path_rel:
        raise ValueError("Missing 'pgdata_path' in optuna.storage configuration.")
    # Construct absolute path using pathlib
    pgdata_path_abs = project_root / pgdata_path_rel
    # Ensure the directory exists (important for initdb/pg_ctl)
    os.makedirs(pgdata_path_abs.parent, exist_ok=True)
    logging.info(f"Resolved absolute PGDATA path: {pgdata_path_abs}")

    db_name = storage_config.get("db_name", "optuna_study_db")
    db_user = storage_config.get("db_user", "optuna_user")
    use_socket = storage_config.get("use_socket", True)

    # Determine socket directory
    socket_dir = None
    if use_socket:
        socket_base = storage_config.get("socket_dir_base")
        # Use SLURM_JOB_ID if available in environment for uniqueness, else use PID
        job_id = os.environ.get("SLURM_JOB_ID", os.getpid())
        socket_dir_name = f"pg_socket_{job_id}"

        if socket_base: # User specified a base like /tmp
             socket_dir = os.path.join(socket_base, socket_dir_name)
        elif "TMPDIR" in os.environ: # Standard HPC temporary directory
             socket_dir = os.path.join(os.environ["TMPDIR"], socket_dir_name)
        else: # Fallback to /tmp if TMPDIR not set
             socket_dir = os.path.join("/tmp", socket_dir_name)
        os.makedirs(socket_dir, exist_ok=True) # Ensure it exists

    # Store intermediate dictionary
    pg_config_dict = {
        "pgdata": pgdata_path_abs,
        "dbname": db_name,
        "dbuser": db_user,
        "use_socket": use_socket,
        "socket_dir": socket_dir,
        "job_owner": getpass.getuser(), # Get username for initdb/psql commands
    } # Store intermediate dict

    # Get PostgreSQL binary directory from environment variable set in Slurm script
    pg_bin_dir = os.environ.get("POSTGRES_BIN_DIR")
    if not pg_bin_dir or not os.path.isdir(pg_bin_dir):
        logging.error(f"POSTGRES_BIN_DIR environment variable not set or invalid: '{pg_bin_dir}'. Ensure it's exported in the Slurm script.")
        raise ValueError("POSTGRES_BIN_DIR environment variable not set or invalid.")

    # Construct full paths and add to config dict
    pg_config_dict["initdb_path"] = os.path.join(pg_bin_dir, "initdb")
    pg_config_dict["pg_ctl_path"] = os.path.join(pg_bin_dir, "pg_ctl")
    pg_config_dict["psql_path"] = os.path.join(pg_bin_dir, "psql")

    # Verify executables exist
    for key, path in pg_config_dict.items():
        if key.endswith("_path") and not os.path.exists(path):
             logging.error(f"Constructed path for {key} does not exist: {path}")
             raise FileNotFoundError(f"PostgreSQL executable not found at expected path: {path}")

    # Return the completed dictionary including executable paths
    return pg_config_dict

def get_optuna_storage_url(config):
    """Constructs the Optuna storage URL based on config."""
    pg_config = _get_pg_config(config)
    db_user = pg_config["dbuser"]
    db_name = pg_config["dbname"]

    if pg_config["use_socket"]:
        socket_dir = pg_config["socket_dir"]
        if not socket_dir:
             raise ValueError("Socket directory is not defined for socket connection.")
        # Format for socket: postgresql://user@/dbname?host=/path/to/socket/dir
        url = f"postgresql://{db_user}@/{db_name}?host={socket_dir}"
        logging.info(f"Constructed PostgreSQL Optuna URL (socket): {url}")
        return url
    else:
        # Format for TCP/IP: postgresql://user:password@host:port/dbname
        # Note: Password handling and host/port config would be needed here
        # For simplicity, we focus on the socket method suitable for single-node HPC
        raise NotImplementedError("TCP/IP connection for PostgreSQL is not implemented in this utility.")

def delete_postgres_data(config):
    """Stops server (if running) and removes the PGDATA directory."""
    pg_config = _get_pg_config(config)
    pgdata = pg_config["pgdata"]
    if os.path.exists(pgdata):
        logging.warning(f"Attempting to remove existing PostgreSQL data directory: {pgdata}")
        try:
            # Attempt to stop server first
            stop_postgres(config, raise_on_error=False) # Don't fail if stop fails
        except Exception as e:
            logging.warning(f"Ignoring error during pre-delete server stop: {e}")

        try:
            shutil.rmtree(pgdata)
            logging.info(f"Successfully removed PostgreSQL data directory: {pgdata}")
        except Exception as e:
            logging.error(f"Failed to remove PostgreSQL data directory {pgdata}: {e}")
            raise
    else:
        logging.info(f"PostgreSQL data directory {pgdata} does not exist, nothing to remove.")


def init_postgres(config):
    """Initializes a new PostgreSQL database cluster if PGDATA doesn't exist."""
    pg_config = _get_pg_config(config)
    pgdata = pg_config["pgdata"]
    db_name = pg_config["dbname"]
    db_user = pg_config["dbuser"]
    job_owner = pg_config["job_owner"]
    needs_db_setup = False

    initdb_path = pg_config["initdb_path"] # Get full path

    if not os.path.exists(os.path.join(pgdata, "base")):
        logging.info(f"Initializing PostgreSQL data directory ({pgdata})...")
        init_cmd = [
            initdb_path, # Use full path
            "--no-locale",
            "--auth=trust", # Use trust auth for local socket simplicity
            "-E UTF8",
            f"-U {job_owner}", # Set initial superuser to the job owner
            "-D", str(pgdata) # Explicitly pass -D and string path separately
        ]
        _run_cmd(init_cmd) # Uses shell=False by default now
        logging.info("PostgreSQL data directory initialized.")

        hba_conf_path = os.path.join(pgdata, "pg_hba.conf") # Define path early

        # Wait explicitly for pg_hba.conf to appear after initdb (handle filesystem delays)
        max_wait_hba = 10 # seconds
        wait_interval_hba = 0.5 # seconds
        waited_hba = 0
        logging.info(f"Waiting up to {max_wait_hba}s for {hba_conf_path} to appear...")
        while not os.path.exists(hba_conf_path):
            time.sleep(wait_interval_hba)
            waited_hba += wait_interval_hba
            if waited_hba >= max_wait_hba:
                logging.error(f"File {hba_conf_path} did not appear after initdb completed.")
                raise FileNotFoundError(f"File {hba_conf_path} did not appear after initdb completed.")
        logging.info(f"{hba_conf_path} found after {waited_hba:.1f}s.")

        # Configure pg_hba.conf for local socket access (Now file should exist)
        logging.info(f"Modifying {hba_conf_path} for local trust authentication...")
        hba_lines = [
            "# TYPE  DATABASE        USER            ADDRESS                 METHOD",
            f"local   all             {job_owner}                             trust", # Job owner superuser access
            f"local   {db_name}       {db_user}                               trust", # Optuna user access to its DB
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

def setup_db_user(config):
    """Creates the specific database and user if they don't exist."""
    pg_config = _get_pg_config(config)
    socket_dir = pg_config["socket_dir"]
    job_owner = pg_config["job_owner"]
    db_user = pg_config["dbuser"]
    db_name = pg_config["dbname"]

    if not socket_dir:
        raise ValueError("Socket directory required for database/user setup.")

    logging.info("Performing first-time database setup (User/DB creation)...")
    # Wait a moment for server to be fully ready after start
    time.sleep(5)

    psql_path = pg_config["psql_path"] # Get full path

    # Use the job owner ($USER) which initdb created as superuser with trust auth
    # Check if user exists, create if not
    check_user_cmd_list = [
        psql_path, "-h", socket_dir, "-U", job_owner, "-d", "postgres",
        "-tAc", f"SELECT 1 FROM pg_roles WHERE rolname='{db_user}'"
    ]
    user_exists = _run_cmd(check_user_cmd_list, check=False).stdout.strip() == '1'

    if not user_exists:
        create_user_cmd_list = [
            psql_path, "-h", socket_dir, "-U", job_owner, "-d", "postgres",
            "-c", f"CREATE USER {db_user};"
        ]
        _run_cmd(create_user_cmd_list)
        logging.info(f"Created PostgreSQL user: {db_user}")
    else:
        logging.info(f"PostgreSQL user {db_user} already exists.")

    # Check if db exists, create if not
    check_db_cmd_list = [
        psql_path, "-h", socket_dir, "-U", job_owner, "-d", "postgres",
        "-tAc", f"SELECT 1 FROM pg_database WHERE datname='{db_name}'"
    ]
    db_exists = _run_cmd(check_db_cmd_list, check=False).stdout.strip() == '1'

    if not db_exists:
        create_db_cmd_list = [
            psql_path, "-h", socket_dir, "-U", job_owner, "-d", "postgres",
            "-c", f"CREATE DATABASE {db_name} OWNER {db_user};"
        ]
        _run_cmd(create_db_cmd_list)
        logging.info(f"Created PostgreSQL database: {db_name} owned by {db_user}")
    else:
        logging.info(f"PostgreSQL database {db_name} already exists.")

    logging.info("Database setup complete.")


def start_postgres(config):
    """Starts the PostgreSQL server."""
    pg_config = _get_pg_config(config)
    pgdata = pg_config["pgdata"]
    socket_dir = pg_config["socket_dir"]
    logfile = os.path.join(pgdata, "logfile.log")

    if not socket_dir:
        raise ValueError("Socket directory required to start PostgreSQL.")

    pg_ctl_path = pg_config["pg_ctl_path"] # Get full path

    logging.info("Starting PostgreSQL server...")
    # Check if server is already running (pg_ctl status)
    # Pass -D and path as separate arguments
    status_cmd = [pg_ctl_path, "status", "-D", str(pgdata)]
    status_result = _run_cmd(status_cmd, check=False) # Uses shell=False

    if status_result.returncode == 0:
        logging.info("PostgreSQL server is already running.")
        return # Already running

    # If status check fails (code 3 usually means not running), try starting
    if status_result.returncode == 3:
        # Construct command list, passing -D and path separately
        start_cmd_list = [
            pg_ctl_path, "start", "-w",
            "-D", str(pgdata),
            "-l", logfile,
            "-o", f"-c unix_socket_directories='{socket_dir}'" # Keep -o arg as single string
        ]
        # Use shell=False as we pass a list
        _run_cmd(start_cmd_list)
        logging.info("PostgreSQL server started successfully.")
    else:
        # Unexpected status code
        logging.error(f"pg_ctl status returned unexpected code {status_result.returncode}. Check logs.")
        raise RuntimeError("Failed to determine PostgreSQL server status.")


def stop_postgres(config, raise_on_error=True):
    """Stops the PostgreSQL server."""
    pg_config = _get_pg_config(config)
    pgdata = pg_config["pgdata"]
    socket_dir = pg_config["socket_dir"] # Needed for cleanup
    pg_ctl_path = pg_config["pg_ctl_path"] # Get full path

    logging.info("Stopping PostgreSQL server...")
    # Construct command list, passing -D and path separately
    stop_cmd_list = [
        pg_ctl_path, "stop", "-w",
        "-D", str(pgdata),
        "-m", "fast"
    ]
    try:
        # Use shell=False as we pass a list
        _run_cmd(stop_cmd_list)
        logging.info("PostgreSQL server stopped.")
    except Exception as e:
        logging.warning(f"pg_ctl stop failed (maybe server was not running?): {e}")
        if raise_on_error:
            raise
    finally:
        # Clean up socket directory if it exists
        if socket_dir and os.path.exists(socket_dir):
            try:
                shutil.rmtree(socket_dir)
                logging.info(f"Removed socket directory: {socket_dir}")
            except Exception as e:
                logging.warning(f"Failed to remove socket directory {socket_dir}: {e}")

def manage_postgres_instance(config, restart=False):
    """
    Main function to manage the lifecycle of the PostgreSQL instance for a job.
    Should typically be called only by rank 0.
    Returns the Optuna storage URL.
    """
    logging.info("Managing PostgreSQL instance...")
    if restart:
        delete_postgres_data(config)

    needs_setup = init_postgres(config)
    start_postgres(config)

    if needs_setup:
        setup_db_user(config)

    storage_url = get_optuna_storage_url(config)
    logging.info("PostgreSQL instance is ready.")
    return storage_url