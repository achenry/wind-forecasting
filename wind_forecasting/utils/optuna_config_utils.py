"""
Optuna configuration utilities for database setup and dashboard commands.

These utilities are used across multiple modes (tune/train/test) for:
- Setting up Optuna storage connections
- Generating database configuration parameters
- Creating dashboard commands
"""
import os
import logging
from typing import Dict

from wind_forecasting.tuning.utils.path_utils import resolve_path


def generate_db_setup_params(model: str, model_config: Dict) -> Dict:
    """
    Generate database setup parameters for Optuna studies.
    
    Args:
        model: Model name
        model_config: Model configuration dictionary
        
    Returns:
        Dictionary containing database setup parameters including:
        - backend: Database backend type (sqlite, postgresql, etc.)
        - project_root: Project root directory
        - pgdata_path: PostgreSQL data directory path
        - socket_dir_base: Base directory for socket files
        - sync_dir: Directory for synchronization files
        - optuna_dir: Optuna directory
        - pgdata_instance_name: PostgreSQL instance name
        - base_study_prefix: Base prefix for study names
        - sqlite_db_path: SQLite database path (if backend is sqlite)
    """
    # Return a base study prefix instead of the final study name
    # The final study name will be constructed in tune_model() based on restart_tuning flag
    base_study_prefix = f"tuning_{model}_{model_config['experiment']['run_name']}"
    optuna_cfg = model_config["optuna"]
    storage_cfg = optuna_cfg.get("storage", {})
    logging_cfg = model_config["logging"]
    experiment_cfg = model_config["experiment"]

    # Resolve paths relative to project root and substitute known variables
    project_root = experiment_cfg.get("project_root", os.getcwd())
    
    # Resolve paths with direct substitution
    optuna_dir_from_config = logging_cfg.get("optuna_dir")
    resolved_optuna_dir = resolve_path(project_root, optuna_dir_from_config)
    if not resolved_optuna_dir:
        raise ValueError("logging.optuna_dir is required but not found or resolved.")
    
    backend = storage_cfg.get("backend", "sqlite")

    # Get instance name for PostgreSQL data directory
    pgdata_instance_name = storage_cfg.get("pgdata_instance_name", "default")
    if backend == "postgresql" and pgdata_instance_name == "default":
        logging.warning("No 'pgdata_instance_name' specified in config. Using default instance name.")
    
    # Resolve pgdata path with instance name
    pgdata_path_from_config = storage_cfg.get("pgdata_path")
    if pgdata_path_from_config:
        # For explicitly specified pgdata_path, append instance name
        pgdata_dir = os.path.dirname(pgdata_path_from_config)
        pgdata_path_with_instance = os.path.join(pgdata_dir, f"pgdata_{pgdata_instance_name}")
        resolved_pgdata_path = resolve_path(project_root, pgdata_path_with_instance)
    else:
        # For default path, use instance name
        resolved_pgdata_path = os.path.join(resolved_optuna_dir, f"pgdata_{pgdata_instance_name}")

    socket_dir_base_from_config = storage_cfg.get("socket_dir_base")
    if not socket_dir_base_from_config:
        socket_dir_base_str = os.path.join(resolved_optuna_dir, "sockets")
    else:
        socket_dir_base_str = str(socket_dir_base_from_config).replace("${logging.optuna_dir}", resolved_optuna_dir)
    resolved_socket_dir_base = resolve_path(project_root, socket_dir_base_str) # Make absolute

    sync_dir_from_config = storage_cfg.get("sync_dir")
    if not sync_dir_from_config:
        # Default value uses the resolved optuna_dir
        sync_dir_str = os.path.join(resolved_optuna_dir, "sync")
    else:
        # Substitute directly if the variable exists
        sync_dir_str = str(sync_dir_from_config).replace("${logging.optuna_dir}", resolved_optuna_dir)
    resolved_sync_dir = resolve_path(project_root, sync_dir_str) # Make absolute

    db_setup_params = {
        "backend": backend,
        "project_root": project_root,
        "pgdata_path": resolved_pgdata_path,
        "socket_dir_base": resolved_socket_dir_base,
        "sync_dir": resolved_sync_dir,
        "optuna_dir": resolved_optuna_dir,
        "pgdata_instance_name": pgdata_instance_name,
        "base_study_prefix": base_study_prefix
    }

    if backend == "sqlite":
        # SQLite specific configuration
        sqlite_db_path_from_config = storage_cfg.get("sqlite_db_path")
        if not sqlite_db_path_from_config:
            # Default value uses the resolved optuna_dir
            sqlite_db_path_str = os.path.join(resolved_optuna_dir, "optuna.db")
        else:
            # Substitute and resolve
            sqlite_db_path_str = str(sqlite_db_path_from_config).replace("${logging.optuna_dir}", resolved_optuna_dir)
        resolved_sqlite_db_path = resolve_path(project_root, sqlite_db_path_str) # Make absolute
        db_setup_params["sqlite_db_path"] = resolved_sqlite_db_path
    
    elif backend == "postgresql":
        # PostgreSQL specific configuration
        optuna_db_name = storage_cfg.get("database_name", "optuna_study_db")
        socket_dir_instance = os.path.join(resolved_socket_dir_base, pgdata_instance_name)
        
        db_setup_params.update({
            "optuna_db_name": optuna_db_name,
            "socket_dir_instance": socket_dir_instance,
            "use_socket": storage_cfg.get("use_socket", True),
            "use_tcp": storage_cfg.get("use_tcp", False),
            "db_host": storage_cfg.get("db_host", "localhost"),
            "db_port": storage_cfg.get("db_port", 5432),
            "db_user": storage_cfg.get("db_user", "optuna_user"),
        })

    return db_setup_params


def generate_optuna_dashboard_command(db_setup_params: Dict, final_study_name: str) -> str:
    """
    Generate the command instructions for launching Optuna dashboard.
    
    Args:
        db_setup_params: Database setup parameters
        final_study_name: Final study name
        
    Returns:
        Command string with instructions to launch Optuna dashboard
    """
    backend = db_setup_params.get("backend", "sqlite")
    db_host = db_setup_params.get("db_host", "localhost")
    db_port = db_setup_params.get("db_port", 5432)
    db_name = db_setup_params.get("optuna_db_name", "optuna_study_db")
    db_user = db_setup_params.get("db_user", "optuna_user")
    sslmode = db_setup_params.get("sslmode")
    sslrootcert_path = db_setup_params.get("sslrootcert_path")

    command_parts = [
        "optuna-monitor",
        f"--db-type {backend}"
    ]

    if backend == "postgresql":
        command_parts.append(f"--db-host {db_host}")
        command_parts.append(f"--db-port {db_port}")
        command_parts.append(f"--db-name {db_name}")
        command_parts.append(f"--db-user {db_user}")

        if sslrootcert_path:
            command_parts.append(f"--cert-path {sslrootcert_path}")
            if sslmode and sslmode != "disable": # Assuming 'disable' means no cert
                command_parts.append("--use-cert")
            else:
                command_parts.append("--no-cert")
        elif sslmode == "disable":
            command_parts.append("--no-cert")

    command_parts.append(f"--study {final_study_name}")

    # Example of how to use it with run_optuna_miniforge.sh
    example_command = f"""
    To launch the Optuna Dashboard for this study, use the following command:

    Important Parameters:
    - Database Type: {backend}
    - Database Host: {db_host}
    - Database Port: {db_port}
    - Database Name: {db_name}
    - Database User: {db_user}
    - Study Name: {final_study_name}
    - SSL Mode: {sslmode if sslmode else 'Not specified/Default'}
    - SSL Certificate Path: {sslrootcert_path if sslrootcert_path else 'Not specified'}

    Example Command:

    run_optuna_miniforge.sh --conda-env your_conda_env_name {' '.join(command_parts)} --db-password 'your_password'
    """
    return example_command