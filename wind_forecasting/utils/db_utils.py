import os
import subprocess
import logging
import time
import shutil
import getpass # To get current username for initdb/psql
from pathlib import Path # Import Path for robust path calculation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def _run_cmd(command, cwd=None, shell=False, check=True):
    """Helper function to run shell commands and log output/errors."""
    log_cmd = ' '.join(command) if isinstance(command, list) else command
    logging.info(f"Running command (shell={shell}): {log_cmd}")
    try:
        # --- Create a minimal environment for the subprocess ---
        minimal_env = {}
        if "PATH" in os.environ:
            minimal_env["PATH"] = os.environ["PATH"]
        else:
            logging.warning("PATH not found in environment, command might fail.")

        captured_ld_path = os.environ.get("CAPTURED_LD_LIBRARY_PATH")
        if captured_ld_path:
            minimal_env["LD_LIBRARY_PATH"] = captured_ld_path
            logging.info(f"Setting minimal LD_LIBRARY_PATH for subprocess: {captured_ld_path}")
        elif "LD_LIBRARY_PATH" in os.environ:
             minimal_env["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
             logging.warning(f"Using current LD_LIBRARY_PATH for subprocess (capture might have failed): {minimal_env['LD_LIBRARY_PATH']}")
        else:
             logging.warning("LD_LIBRARY_PATH not found in environment for subprocess.")
        for env_var in ["USER", "LOGNAME", "HOME"]:
             if env_var in os.environ:
                 minimal_env[env_var] = os.environ[env_var]

        process = subprocess.run(
            command,
            cwd=cwd,
            shell=shell,
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=minimal_env
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
        url = f"postgresql://{db_user}@/{db_name}?host={socket_dir}"
        logging.info(f"Constructed PostgreSQL Optuna URL (socket): {url}")
        return url
    else:
        # TCP/IP connection not implemented for this use case
        raise NotImplementedError("TCP/IP connection for PostgreSQL is not implemented in this utility.")

def delete_postgres_data(pg_config, raise_on_error=True):
    """Stops server and removes PGDATA directory."""
    pgdata = pg_config["pgdata"]
    if os.path.exists(pgdata):
        logging.warning(f"Attempting to remove existing PostgreSQL data directory: {pgdata}")
        try:
            stop_postgres(pg_config, raise_on_error=False)
        except Exception as e:
            logging.warning(f"Ignoring error during pre-delete server stop: {e}")

        # Use standard shutil.rmtree for removal
        try:
            shutil.rmtree(pgdata)
            logging.info(f"Successfully removed PostgreSQL data directory: {pgdata}")
        except OSError as e:
            logging.error(f"Error removing directory {pgdata}: {e}")
            if raise_on_error:
                 raise
    else:
        logging.info(f"PostgreSQL data directory {pgdata} does not exist, nothing to remove.")
# Removed blank line


def init_postgres(pg_config):
    """Initializes a new PostgreSQL database cluster."""
    # pg_config is now passed directly
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

        _run_cmd(init_cmd) # Uses shell=False by default now
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
    socket_dir = pg_config["socket_dir"]
    job_owner = pg_config["job_owner"]
    db_user = pg_config["dbuser"]
    db_name = pg_config["dbname"]

    if not socket_dir:
        raise ValueError("Socket directory required for database/user setup.")

    logging.info("Performing first-time database setup (User/DB creation)...")
    # Wait for server startup
    time.sleep(2)

    psql_path = pg_config["psql_path"]

    # Create Optuna User (db_user) if it doesn't exist
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

    # Create Optuna Database (db_name) if it doesn't exist
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


def start_postgres(pg_config):
    """Starts the PostgreSQL server if not already running."""
    pgdata = pg_config["pgdata"]
    socket_dir = pg_config["socket_dir"]
    logfile = os.path.join(pgdata, "logfile.log")

    if not socket_dir:
        raise ValueError("Socket directory required to start PostgreSQL.")

    pg_ctl_path = pg_config["pg_ctl_path"]

    logging.info("Starting PostgreSQL server...")
    # Check if server is already running
    status_cmd = [pg_ctl_path, "status", "-D", str(pgdata)]
    status_result = _run_cmd(status_cmd, check=False) # Uses shell=False

    if status_result.returncode == 0:
        logging.info("PostgreSQL server is already running.")
        return

    # If not running (exit code 3), try starting
    if status_result.returncode == 3:
        # Construct start command
        start_cmd_list = [
            pg_ctl_path, "start", "-w",
            "-D", str(pgdata),
            "-l", logfile,
            "-o", f"-c unix_socket_directories='{socket_dir}'"
        ]
        _run_cmd(start_cmd_list)
        logging.info("PostgreSQL server started successfully.")
    else:
        # Unexpected status code
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
        _run_cmd(stop_cmd_list)
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

def _generate_pg_config(config):
    """Generates the pg_config dictionary from the main config."""

    # Generate config dictionary used by other functions
    storage_config = config.get("optuna", {}).get("storage", {})
    if storage_config.get("backend") != "postgresql":
        raise ValueError("Database backend is not configured as 'postgresql' in YAML.")

    try:
        project_root = Path(__file__).resolve().parents[2]
        logging.info(f"Determined project root: {project_root}")
    except IndexError:
        logging.error("Could not determine project root based on script location. Falling back to CWD.")
        project_root = Path(os.getcwd())

    pgdata_path_rel = storage_config.get("pgdata_path")
    if not pgdata_path_rel:
        raise ValueError("Missing 'pgdata_path' in optuna.storage configuration.")

    # Generate UNIQUE path in Persistent Storage
    # Base path is determined relative to the project root using the config value
    pgdata_base_abs = project_root / Path(pgdata_path_rel).parent

    import uuid
    timestamp = int(time.time())
    job_id_for_path = os.environ.get("SLURM_JOB_ID", "local") # Use Slurm job ID for uniqueness
    unique_name = f"pg_data_{job_id_for_path}_{timestamp}_{uuid.uuid4().hex[:8]}"
    pgdata_path_abs = pgdata_base_abs / unique_name
    # Ensure parent directory exists
    os.makedirs(pgdata_base_abs, exist_ok=True)
    logging.info(f"Using PGDATA path: {pgdata_path_abs}")

    db_name = storage_config.get("db_name", "optuna_study_db")
    db_user = storage_config.get("db_user", "optuna_user")
    use_socket = storage_config.get("use_socket", True)

    socket_dir = None
    if use_socket:
        socket_base = storage_config.get("socket_dir_base")
        job_id_for_socket = os.environ.get("SLURM_JOB_ID", os.getpid()) # Unique socket per job
        socket_dir_name = f"pg_socket_{job_id_for_socket}"
        if socket_base:
             socket_dir = os.path.join(socket_base, socket_dir_name)
        elif "TMPDIR" in os.environ:
             socket_dir = os.path.join(os.environ["TMPDIR"], socket_dir_name)
        else:
             socket_dir = os.path.join("/tmp", socket_dir_name)
        os.makedirs(socket_dir, exist_ok=True)

    pg_bin_dir = os.environ.get("POSTGRES_BIN_DIR")
    if not pg_bin_dir or not os.path.isdir(pg_bin_dir):
        logging.error(f"POSTGRES_BIN_DIR environment variable not set or invalid: '{pg_bin_dir}'.")
        raise ValueError("POSTGRES_BIN_DIR environment variable not set or invalid.")

    pg_config = {
        "pgdata": pgdata_path_abs,
        "dbname": db_name,
        "dbuser": db_user,
        "use_socket": use_socket,
        "socket_dir": socket_dir,
        "job_owner": getpass.getuser(),
        "initdb_path": os.path.join(pg_bin_dir, "initdb"),
        "pg_ctl_path": os.path.join(pg_bin_dir, "pg_ctl"),
        "psql_path": os.path.join(pg_bin_dir, "psql"),
    }

    # Verify executables exist
    for key, path in pg_config.items():
        if key.endswith("_path") and not os.path.exists(path):
             logging.error(f"PostgreSQL executable not found at expected path: {path}")
             raise FileNotFoundError(f"PostgreSQL executable not found at expected path: {path}")
    return pg_config

def manage_postgres_instance(config, restart=False):
    """
    Main function to manage the lifecycle of the PostgreSQL instance for a job.
    Should typically be called only by rank 0.
    Returns the Optuna storage URL.
    """
    logging.info("Managing PostgreSQL instance...")
    # Generate the config dictionary
    pg_config = _generate_pg_config(config)

    if restart: # If --restart_tuning flag was passed
        # Pass the consistent pg_config
        logging.info("Performing cleanup due to --restart_tuning flag.")
        delete_postgres_data(pg_config) # Initial cleanup if restarting

    # Attempt cleanup before init, just in case of leftovers from failed runs.
    logging.info("Performing pre-initialization cleanup check...")
    delete_postgres_data(pg_config, raise_on_error=False) # Use consistent config, don't fail job if this cleanup has issues

    # Pass the consistent pg_config
    needs_setup = init_postgres(pg_config)
    start_postgres(pg_config)

    if needs_setup:
        # Pass the consistent pg_config
        setup_db_user(pg_config)

    # Pass the consistent pg_config
    storage_url = get_optuna_storage_url(pg_config)
    logging.info("PostgreSQL instance is ready.")
    return storage_url