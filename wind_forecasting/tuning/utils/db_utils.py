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
    # logging.info(f"Running command (shell={shell}): {log_cmd}")  # DEBUG
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
            # logging.info(f"Using captured LD_LIBRARY_PATH for subprocess: {captured_ld_path}") # DEBUG
        # elif "LD_LIBRARY_PATH" in cmd_env:
        #      logging.info(f"Using existing LD_LIBRARY_PATH for subprocess: {cmd_env['LD_LIBRARY_PATH']}") # DEBUG
        else:
             logging.warning("LD_LIBRARY_PATH not found in environment for subprocess.") # DEBUG

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

        # DEBUG
        # if process.stdout:
        #     logging.info(f"Command stdout:\n{process.stdout.strip()}")
        # if process.stderr:
        #     logging.warning(f"Command stderr:\n{process.stderr.strip()}")
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
    db_user = pg_config["db_user"]
    db_name = pg_config["db_name"]

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
        
        # Check for password from environment variable
        password_part = ""
        if "db_password_env_var" in pg_config and pg_config["db_password_env_var"]:
            env_var = pg_config["db_password_env_var"]
            password = os.environ.get(env_var)
            if password:
                password_part = f":{password}"
                logging.info(f"Using password from environment variable: {env_var}")
            else:
                logging.warning(f"Environment variable {env_var} not found or empty")
        
        # Construct base URL with optional password
        url = f"postgresql://{db_user}{password_part}@{db_host}:{db_port}/{db_name}"
        
        # Add SSL parameters if provided
        query_params = []
        
        if "sslmode" in pg_config:
            query_params.append(f"sslmode={pg_config['sslmode']}")
        
        if "sslrootcert_path" in pg_config and pg_config["sslrootcert_path"]:
            cert_path = pg_config["sslrootcert_path"]
            if os.path.exists(cert_path):
                query_params.append(f"sslrootcert={cert_path}")
                logging.info(f"Using SSL root certificate: {cert_path}")
            else:
                logging.warning(f"SSL root certificate not found at: {cert_path}")
        
        # Add query parameters to URL if any
        if query_params:
            url += "?" + "&".join(query_params)
        
        # Log URL safely (hide password)
        safe_url = url.split('@')[0].split(':')[0] + '@' + url.split('@')[1]
        logging.info(f"Constructed PostgreSQL Optuna URL (TCP/IP): {safe_url}")
        
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
    db_name = pg_config["db_name"]
    db_user = pg_config["db_user"]
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
    socket_dir = pg_config["socket_dir"]
    job_owner = pg_config["job_owner"]
    db_user = pg_config["db_user"]
    db_name = pg_config["db_name"]

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
        _run_cmd(create_user_cmd_list, shell=pg_config.get("run_cmd_shell", False))
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
        _run_cmd(create_db_cmd_list, shell=pg_config.get("run_cmd_shell", False))
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
    status_result = _run_cmd(status_cmd, check=False, shell=pg_config.get("run_cmd_shell", False))

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
        _run_cmd(start_cmd_list, shell=pg_config.get("run_cmd_shell", False))
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

# _resolve_path function removed as paths are resolved earlier

# Updated to accept explicit, pre-resolved paths and parameters
def _generate_pg_config(*,
                        backend: str,
                        project_root: str, # Still needed? Maybe not if all paths are absolute
                        pgdata_path: str, # Now expects absolute path
                        base_study_prefix: str,  # Received but not used - PostgreSQL instance is shared across studies
                        use_socket: bool = True,
                        use_tcp: bool = False,
                        db_host: str = "localhost",
                        db_port: int = 5432,
                        db_name: str = "optuna_study_db",
                        db_user: str = "optuna_user",
                        run_cmd_shell: bool = False,
                        socket_dir_base: str, # Expects absolute path
                        sync_dir: str, # Expects absolute path
                        # SSL parameters for external PostgreSQL connections
                        sslmode: str = None,
                        sslrootcert_path: str = None,
                        db_password_env_var: str = None
                       ):
    """
    Generates the pg_config dictionary from the main config, resolving paths
    and handling defaults.
    """
    global _managed_pg_config # Allow modification of the global var

    """
    Generates the pg_config dictionary from explicit parameters, assuming paths
    are already resolved and absolute.
    """
    global _managed_pg_config # Allow modification of the global var

    if backend != "postgresql":
        raise ValueError("This function is only for the 'postgresql' backend.")

    # --- PGDATA Path ---
    if not pgdata_path:
        raise ValueError("Missing 'pgdata_path' (absolute path expected).")
    pgdata_path_abs = Path(pgdata_path)
    # Ensure the base directory exists (parent of the specific study data dir)
    # initdb will create the final directory if needed
    os.makedirs(pgdata_path_abs.parent, exist_ok=True)
    logging.info(f"Using PGDATA path: {pgdata_path_abs}")

    # --- Socket/TCP Configuration ---
    if use_socket and use_tcp:
        raise ValueError("Cannot configure both use_socket=true and use_tcp=true.")
    if not use_socket and not use_tcp:
        logging.info("Neither use_socket nor use_tcp specified, defaulting to use_socket=true.")
        use_socket = True

    socket_dir = None
    # db_host and db_port are passed directly

    if use_socket:
        # DEBUG--- Force short socket path in /tmp ---
        username = getpass.getuser()
        # Extract instance name from pgdata_path (assuming pgdata_path is absolute)
        pgdata_instance_name = Path(pgdata_path).name
        # Construct a short path
        short_socket_base = f"/tmp/pg_sockets_{username}"
        socket_dir_path = Path(short_socket_base) / pgdata_instance_name
        socket_dir = str(socket_dir_path.resolve()) # Resolve to absolute path

        # Ensure the directory exists
        os.makedirs(socket_dir, exist_ok=True)
        logging.info(f"Forcing use of short socket directory: {socket_dir}")
        # --- End short path logic ---
    elif use_tcp:
        logging.info(f"Using TCP/IP connection: host={db_host}, port={db_port}")
        # socket_dir remains None

    # --- Sync Directory ---
    if not sync_dir:
        raise ValueError("Missing 'sync_dir' (absolute path expected).")
    sync_dir_path = Path(sync_dir)
    os.makedirs(sync_dir_path, exist_ok=True)
    
    # Use pgdata_instance_name
    pgdata_instance_name = os.path.basename(pgdata_path)
    sync_file = str(sync_dir_path / f"optuna_pg_ready_{pgdata_instance_name}.sync")
    logging.info(f"Using sync file path: {sync_file}")

    # --- PostgreSQL Binaries ---
    pg_bin_dir = os.environ.get("POSTGRES_BIN_DIR")
    if not pg_bin_dir or not os.path.isdir(pg_bin_dir):
        logging.error(f"POSTGRES_BIN_DIR environment variable not set or invalid: '{pg_bin_dir}'. Cannot find PostgreSQL executables.")
        raise ValueError("POSTGRES_BIN_DIR environment variable not set or invalid.")

    pg_config = {
        "pgdata": str(pgdata_path_abs), # Use the resolved absolute path
        "db_name": db_name,
        "db_user": db_user,
        "use_socket": use_socket,
        "socket_dir": socket_dir, # Will be None if use_tcp is True
        "use_tcp": use_tcp,
        "db_host": db_host, # Will be None if use_socket is True (usually)
        "db_port": db_port, # Will be None if use_socket is True (usually)
        "job_owner": getpass.getuser(),
        "initdb_path": os.path.join(pg_bin_dir, "initdb"),
        "pg_ctl_path": os.path.join(pg_bin_dir, "pg_ctl"),
        "psql_path": os.path.join(pg_bin_dir, "psql"),
        "run_cmd_shell": run_cmd_shell,
        "sync_file": sync_file,
    }
    
    # Add SSL parameters and password environment variable if provided
    if sslmode:
        pg_config["sslmode"] = sslmode
    
    if sslrootcert_path:
        pg_config["sslrootcert_path"] = sslrootcert_path
    
    if db_password_env_var:
        pg_config["db_password_env_var"] = db_password_env_var

    # Verify executables exist
    for key, path in pg_config.items():
        if key.endswith("_path") and path and not os.path.exists(path):
             logging.error(f"PostgreSQL executable not found at expected path: {path} (derived from POSTGRES_BIN_DIR={pg_bin_dir})")
             raise FileNotFoundError(f"PostgreSQL executable '{os.path.basename(path)}' not found at expected path: {path}")

# *** Handle socket path override ***
    # Only apply socket path override if use_tcp is not explicitly set to True
    if not pg_config.get("use_tcp", False):
        username = getpass.getuser()
        # pgdata_path is expected to be an absolute path string here
        pgdata_instance_name = Path(pgdata_path).name
        
        forced_socket_dir = f"/tmp/pg_{username}_{pgdata_instance_name}"
        try:
            os.makedirs(forced_socket_dir, exist_ok=True)
            logging.info(f"ALERT: Forcing short socket directory override: {forced_socket_dir}")
            
            # Overwrite values in the pg_config dictionary
            # pg_config is guaranteed to exist at this point (created around line 378)
            pg_config['socket_dir'] = forced_socket_dir
            pg_config['use_socket'] = True # Ensure socket use is enabled
            pg_config['use_tcp'] = False # Ensure TCP is disabled (socket takes precedence)
        except OSError as e:
            logging.error(f"CRITICAL: Failed to create forced socket directory {forced_socket_dir}: {e}")
            # Re-raise the error to prevent proceeding with an unusable configuration
            raise
    else:
        logging.info(f"Using TCP/IP connection as specified in configuration: host={pg_config.get('db_host', 'localhost')}, port={pg_config.get('db_port', 5432)}")

    # Subsequent code (like get_optuna_storage_url and start_postgres called later)
    # will now use the forced socket_dir from the modified pg_config.
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

def manage_postgres_instance(db_setup_params, restart=False, register_cleanup=True):
    """
    Main function to manage the PostgreSQL instance for a job called only by rank 0.
    Accepts explicit parameters via db_setup_params dictionary.
    """
    logging.info("Managing PostgreSQL instance...")
    # Generate the config dictionary using explicit parameters
    # Unpack the dictionary containing the required keyword arguments
    pg_config = _generate_pg_config(**db_setup_params) # This also sets _managed_pg_config

    # Check if we're using TCP/IP for an external PostgreSQL server
    if pg_config.get("use_tcp", False) and (
        "sslmode" in pg_config or
        "sslrootcert_path" in pg_config or
        "db_password_env_var" in pg_config
    ):
        logging.info("Using external PostgreSQL server via TCP/IP connection.")
        
        storage_url = get_optuna_storage_url(pg_config)
        logging.info("External PostgreSQL connection URL constructed.")
        
        return storage_url, pg_config
    else:
        logging.info("Managing local PostgreSQL instance...")

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