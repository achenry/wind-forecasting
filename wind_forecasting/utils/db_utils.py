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
        # Removed the first erroneous subprocess.run call here
        # --- Create a minimal environment for the subprocess ---
        minimal_env = {}
        # Preserve PATH so the command can be found
        if "PATH" in os.environ:
            minimal_env["PATH"] = os.environ["PATH"]
        else:
            logging.warning("PATH not found in environment, command might fail.")

        # Explicitly set LD_LIBRARY_PATH from the captured variable
        captured_ld_path = os.environ.get("CAPTURED_LD_LIBRARY_PATH")
        if captured_ld_path:
            minimal_env["LD_LIBRARY_PATH"] = captured_ld_path
            logging.info(f"Setting minimal LD_LIBRARY_PATH for subprocess: {captured_ld_path}")
        elif "LD_LIBRARY_PATH" in os.environ:
             # Fallback to current LD_LIBRARY_PATH if capture failed (less ideal)
             minimal_env["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
             logging.warning(f"Using current LD_LIBRARY_PATH for subprocess (capture might have failed): {minimal_env['LD_LIBRARY_PATH']}")
        else:
             logging.warning("LD_LIBRARY_PATH not found in environment for subprocess.")
        # Add USER, LOGNAME, and HOME as they might be needed for authentication/identification
        for env_var in ["USER", "LOGNAME", "HOME"]: # Added HOME too, might be relevant
             if env_var in os.environ:
                 minimal_env[env_var] = os.environ[env_var]
        # --- End minimal environment setup ---

        process = subprocess.run(
            command,
            cwd=cwd,
            shell=shell,
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=minimal_env # Pass the minimal environment
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

# Removed _get_pg_config function as its logic is moved into manage_postgres_instance

def get_optuna_storage_url(pg_config):
    """Constructs the Optuna storage URL based on the pre-computed pg_config."""
    # pg_config is now passed directly
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

def delete_postgres_data(pg_config, raise_on_error=True):
    """Stops server (if running) and removes the PGDATA directory."""
    # pg_config is now passed directly
    pgdata = pg_config["pgdata"]
    if os.path.exists(pgdata):
        logging.warning(f"Attempting to remove existing PostgreSQL data directory: {pgdata}")
        try:
            # Attempt to stop server first
            stop_postgres(pg_config, raise_on_error=False) # Don't fail if stop fails
        except Exception as e:
            logging.warning(f"Ignoring error during pre-delete server stop: {e}")
        
        # Convert pathlib.Path to string for system commands
        pgdata_str = str(pgdata)
        
        # FORCEFUL APPROACH: Use system rm command
        logging.warning("Using forceful system-level removal to clear PGDATA")
        
        # 1. Try shutil.rmtree as base attempt
        try:
            shutil.rmtree(pgdata)
            logging.info("Initial shutil.rmtree attempt completed")
        except Exception as e:
            logging.warning(f"Initial shutil.rmtree failed: {e}")
                
        # 2. Force sync filesystem to flush caches
        try:
            # This forces all filesystem buffers to be flushed to disk
            logging.info("Forcing filesystem sync...")
            os.sync()
        except Exception as e:
            logging.warning(f"os.sync() failed: {e}")
        
        # 3. If directory still exists, use system rm command
        if os.path.exists(pgdata):
            logging.warning(f"Directory {pgdata} still exists after rmtree, using 'rm -rf' command")
            try:
                # Use subprocess directly to ensure command is executed correctly
                # Force removal using system rm (which might handle corner cases better)
                rm_cmd = ["rm", "-rf", pgdata_str]
                subprocess.run(rm_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                logging.info("System rm command completed")
            except Exception as e:
                logging.error(f"System rm command failed: {e}")
                
        # 4. Give system time to fully process the removal
        time.sleep(3)
        
        # 5. Verify using multiple methods
        exists_by_python = os.path.exists(pgdata)
        
        # Check if the directory exists using system ls command
        try:
            ls_result = subprocess.run(["ls", "-la", os.path.dirname(pgdata_str)],
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,
                                      text=True)
            exists_by_ls = pgdata_str.split('/')[-1] in ls_result.stdout
            logging.info(f"Directory existence check: Python={exists_by_python}, ls={exists_by_ls}")
        except Exception as e:
            logging.warning(f"ls verification failed: {e}")
            exists_by_ls = False
        
        # Final unified check
        if exists_by_python or exists_by_ls:
            logging.error(f"CRITICAL: Directory {pgdata} STILL EXISTS despite multiple removal attempts!")
            # One last attempt - try to at least empty it
            try:
                # Use find command to delete everything under the directory
                find_cmd = ["find", pgdata_str, "-mindepth", "1", "-delete"]
                subprocess.run(find_cmd, check=False)
                logging.warning("Used 'find -delete' as last resort to empty directory")
            except Exception as e:
                logging.error(f"Final cleanup attempt failed: {e}")
                
            # Just log the error but continue - let initdb handle the result
            logging.error("Continuing despite directory persistence - let initdb handle it")
        else:
            logging.info(f"Successfully confirmed removal of PostgreSQL data directory: {pgdata}")
    else:
        logging.info(f"PostgreSQL data directory {pgdata} does not exist initially, nothing to remove.")


def init_postgres(pg_config):
    """Initializes a new PostgreSQL database cluster if PGDATA doesn't exist."""
    # pg_config is now passed directly
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
            # "-d", # debug flag (very verbose!)
            "--no-locale",
            "--auth=trust", # Use trust auth for local socket simplicity
            "-E UTF8",
            # f"-U {job_owner}", # Omit -U; initdb will use OS user (job_owner) by default
            "-D", str(pgdata) # Explicitly pass -D and string path separately
        ]
        # --- Pre-create Directory ---
        pgdata_str = str(pgdata) # Ensure it's a string
        try:
            logging.info(f"Explicitly creating directory before initdb: {pgdata_str}")
            os.makedirs(pgdata_str, exist_ok=True) # Create directory if it doesn't exist
            # Set permissions explicitly? Might help on some filesystems.
            # os.chmod(pgdata_str, 0o700) # Example: Set owner-only permissions
            logging.info(f"Directory created or already exists: {pgdata_str}")
        except Exception as e:
             logging.error(f"Error explicitly creating directory {pgdata_str}: {e}")
             # Decide if we should raise here or let initdb try anyway
             # For now, let initdb try.
        # --- ADD DELAY (Moved inside TRY block) ---
        logging.info("Adding a 1-second delay after directory creation...")
        time.sleep(1)
        # --- END DELAY ---
        # --- End Pre-create ---

        # --- Diagnostic Logging (Now should always find the directory) ---
        logging.info(f"Checking state of target directory before initdb: {pgdata_str}")
        try:
            if os.path.exists(pgdata_str):
                contents = os.listdir(pgdata_str)
                logging.info(f"Directory exists. Contents: {contents}")
                if not contents:
                    logging.info("Directory exists and is empty.") # Expect this now
                else:
                    # This would be very strange now
                    logging.warning("Directory exists BUT IS NOT EMPTY before initdb call, despite pre-creation!")
            else:
                 # This should not happen now
                logging.error("Directory does not exist before initdb call, despite pre-creation attempt!")
        except Exception as e:
            logging.error(f"Error checking directory state before initdb: {e}")
        # --- End Diagnostic Logging ---

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
            # Allow the OS user (job_owner) to connect as the PG superuser (job_owner)
            # to the 'postgres' database for initial setup.
            f"local   postgres        {job_owner}                             trust",
            # Allow the OS user (job_owner) to connect as the PG superuser (job_owner)
            # to all other databases as well (might be needed).
            f"local   all             {job_owner}                             trust",
            # Allow the specific optuna user to connect to its specific database.
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
    """Creates the specific database and user if they don't exist."""
    # pg_config is now passed directly
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

    # --- Create Optuna User (db_user) ---
    # Connect using the default superuser (job_owner) implicitly created by initdb.
    # Check if optuna user exists, create if not
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


def start_postgres(pg_config):
    """Starts the PostgreSQL server."""
    # pg_config is now passed directly
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


def stop_postgres(pg_config, raise_on_error=True):
    """Stops the PostgreSQL server."""
    # pg_config is now passed directly
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

def _generate_pg_config(config):
    """Generates the pg_config dictionary from the main config."""

    # --- Centralized Config Generation ---
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

    # --- Generate UNIQUE path ONCE in TMPDIR ---
    tmp_dir_base = os.environ.get("TMPDIR")
    if not tmp_dir_base:
        logging.warning("TMPDIR environment variable not set, falling back to /tmp for PGDATA.")
        tmp_dir_base = "/tmp"

    import uuid
    timestamp = int(time.time())
    job_id_for_path = os.environ.get("SLURM_JOB_ID", "local") # Use consistent job ID source
    unique_name = f"pg_data_{job_id_for_path}_{timestamp}_{uuid.uuid4().hex[:8]}"
    # Create PGDATA inside the TMPDIR base
    pgdata_path_abs = Path(tmp_dir_base) / unique_name
    # No need to create pgdata_base_path, TMPDIR should exist. We create the final dir later.
    logging.info(f"Using UNIQUE PGDATA path in node-local temp storage: {pgdata_path_abs}")
    # --- End PGDATA path generation ---

    db_name = storage_config.get("db_name", "optuna_study_db")
    db_user = storage_config.get("db_user", "optuna_user")
    use_socket = storage_config.get("use_socket", True)

    socket_dir = None
    if use_socket:
        socket_base = storage_config.get("socket_dir_base")
        job_id_for_socket = os.environ.get("SLURM_JOB_ID", os.getpid()) # Use consistent job ID source
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
             logging.error(f"Constructed path for {key} does not exist: {path}")
             raise FileNotFoundError(f"PostgreSQL executable not found at expected path: {path}")
    # --- End Centralized Config Generation ---
    return pg_config # Return the generated config dict

def manage_postgres_instance(config, restart=False):
    """
    Main function to manage the lifecycle of the PostgreSQL instance for a job.
    Should typically be called only by rank 0.
    Returns the Optuna storage URL.
    """
    logging.info("Managing PostgreSQL instance...")
    # Generate the config dictionary
    pg_config = _generate_pg_config(config)

    if restart:
        # Pass the consistent pg_config
        logging.info("Performing cleanup due to --restart_tuning flag.")
        delete_postgres_data(pg_config) # Initial cleanup if restarting

    # ALWAYS attempt cleanup immediately before init, just in case.
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