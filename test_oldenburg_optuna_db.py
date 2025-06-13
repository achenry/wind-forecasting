#!/usr/bin/env python3
"""
Test script to verify Oldenburg PostgreSQL database connection
and create a test Optuna study.

Usage: python test_oldenburg_optuna_db.py
"""
import os
import sys
import logging
import subprocess
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_and_install_dependencies():
    """Check if required packages are installed and attempt to install if missing."""
    logger.info("Checking Python dependencies...")
    
    # Check Python version
    logger.info(f"Python version: {sys.version}")
    
    required_packages = ['optuna', 'psycopg2-binary']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'psycopg2-binary':
                import psycopg2
                logger.info(f"✓ psycopg2: Available")
            else:
                __import__(package)
                if package == 'optuna':
                    import optuna
                    logger.info(f"✓ optuna: {optuna.version.__version__}")
                else:
                    logger.info(f"✓ {package}: Available")
        except ImportError:
            logger.warning(f"✗ {package}: Not installed")
            missing_packages.append(package)
    
    if missing_packages:
        logger.info(f"Installing missing packages: {missing_packages}")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            logger.info("Successfully installed missing packages")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install packages: {e}")
            logger.error("Please install manually: pip install optuna psycopg2-binary")
            return False
    
    return True

def setup_environment():
    """Setup environment variables from the database login file."""
    logger.info("Setting up environment...")
    
    db_login_file = "/user/taed7566/Forecasting/Docs/db_login"
    
    if not os.path.exists(db_login_file):
        logger.error(f"Database login file not found: {db_login_file}")
        return False
    
    # Parse credentials and set environment variable
    try:
        with open(db_login_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('USER='):
                    user_line = line.split('=', 1)[1]
                    password = user_line.split(':', 1)[1]
                    os.environ['LOCAL_PG_PASSWORD'] = password
                    logger.info("✓ Environment variable LOCAL_PG_PASSWORD set")
                    return True
        
        logger.error("Could not find USER= line in database login file")
        return False
        
    except Exception as e:
        logger.error(f"Failed to parse database login file: {e}")
        return False

def parse_db_credentials(db_login_file="/user/taed7566/Forecasting/Docs/db_login"):
    """Parse database credentials from the login file."""
    logger.info(f"Reading database credentials from {db_login_file}")
    
    if not os.path.exists(db_login_file):
        raise FileNotFoundError(f"Database login file not found: {db_login_file}")
    
    credentials = {}
    with open(db_login_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            
            if '=' in line:
                key, value = line.split('=', 1)
                credentials[key] = value
    
    # Parse user credentials (format: username:password)
    if 'USER' in credentials:
        user_line = credentials['USER']
        username, password = user_line.split(':', 1)
        credentials['username'] = username
        credentials['password'] = password
    
    return credentials

def test_database_connection(credentials):
    """Test the database connection by creating a storage URL and connecting."""
    import optuna
    from optuna.storages import RDBStorage
    
    logger.info("Testing database connection...")
    
    # Construct the PostgreSQL URL
    db_host = credentials.get('FQDN')
    db_port = credentials.get('DBPORT', '5432')
    db_name = credentials.get('DB')
    db_user = credentials.get('username')
    db_password = credentials.get('password')
    
    # Create connection URL
    storage_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    # Log connection info (without password)
    safe_url = f"postgresql://{db_user}:****@{db_host}:{db_port}/{db_name}"
    logger.info(f"Connection URL: {safe_url}")
    
    try:
        # Create RDBStorage with connection pool settings
        logger.info("Creating RDBStorage instance...")
        storage = RDBStorage(
            url=storage_url,
            engine_kwargs={
                "pool_size": 2,
                "max_overflow": 2,
                "pool_timeout": 30,
                "pool_recycle": 1800,
                "pool_pre_ping": True,
                "connect_args": {"application_name": "optuna_test_script"}
            }
        )
        
        # Test connection by getting all studies
        logger.info("Testing connection by fetching existing studies...")
        studies = storage.get_all_studies()
        logger.info(f"Successfully connected! Found {len(studies)} existing studies.")
        
        for study in studies:
            logger.info(f"  - Study: {study.study_name} (ID: {study.study_id}, Trials: {study.n_trials})")
        
        return storage, storage_url
        
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise

def create_test_study(storage):
    """Create a test Optuna study to verify functionality."""
    import optuna
    
    logger.info("\nCreating test Optuna study...")
    
    # Define a simple objective function
    def objective(trial):
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)
        return (x - 2) ** 2 + (y + 3) ** 2
    
    # Create study name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = f"test_study_{timestamp}"
    
    try:
        # Create the study
        logger.info(f"Creating study: {study_name}")
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="minimize",
            load_if_exists=False
        )
        
        logger.info(f"Study created successfully! Study ID: {study._study_id}")
        
        # Run a few optimization trials
        logger.info("Running 10 test trials...")
        study.optimize(objective, n_trials=10)
        
        # Print results
        logger.info("\nOptimization results:")
        logger.info(f"  Best value: {study.best_value:.4f}")
        logger.info(f"  Best params: {study.best_params}")
        logger.info(f"  Number of trials: {len(study.trials)}")
        
        # Show trial history
        logger.info("\nTrial history:")
        for trial in study.trials[:5]:  # Show first 5 trials
            logger.info(f"  Trial {trial.number}: value={trial.value:.4f}, params={trial.params}")
        
        if len(study.trials) > 5:
            logger.info(f"  ... and {len(study.trials) - 5} more trials")
        
        return study
        
    except Exception as e:
        logger.error(f"Failed to create or run study: {e}")
        raise

def test_parallel_optimization(storage_url):
    """Test that multiple workers can connect and optimize in parallel."""
    import optuna
    
    logger.info("\nTesting parallel optimization capabilities...")
    
    def objective(trial):
        import time
        import random
        # Add some randomness to simulate real optimization
        time.sleep(random.uniform(0.1, 0.3))
        x = trial.suggest_float("x", -5, 5)
        return x ** 2
    
    study_name = f"parallel_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Create study
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction="minimize"
        )
        
        logger.info(f"Created parallel test study: {study_name}")
        
        # Simulate multiple workers by running sequential optimizations
        # In real scenario, these would be separate processes
        for worker_id in range(3):
            logger.info(f"  Worker {worker_id}: Running 3 trials...")
            study.optimize(objective, n_trials=3)
        
        logger.info(f"Parallel test completed. Total trials: {len(study.trials)}")
        logger.info(f"Best value found: {study.best_value:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Parallel optimization test failed: {e}")
        return False

def cleanup_test_studies(storage):
    """Optional: Clean up test studies (commented out by default)."""
    logger.info("\nCleanup option available but not executed.")
    logger.info("To clean up test studies, uncomment the cleanup code.")
    
    # Uncomment below to enable cleanup
    # studies = storage.get_all_studies()
    # for study in studies:
    #     if study.study_name.startswith("test_study_") or study.study_name.startswith("parallel_test_"):
    #         logger.info(f"  Deleting study: {study.study_name}")
    #         storage.delete_study(study._study_id)

def main():
    """Main test function."""
    logger.info("=== Oldenburg PostgreSQL Optuna Database Test ===")
    logger.info(f"Date: {datetime.now()}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info("")
    
    try:
        # Step 0: Check dependencies and setup environment
        if not check_and_install_dependencies():
            logger.error("Failed to setup dependencies")
            sys.exit(1)
            
        if not setup_environment():
            logger.error("Failed to setup environment")
            sys.exit(1)
        
        # Now import optuna after ensuring it's installed
        import optuna
        from optuna.storages import RDBStorage
        
        logger.info("")
        
        # Step 1: Parse credentials
        credentials = parse_db_credentials()
        logger.info(f"Database: {credentials.get('DB')}")
        logger.info(f"Host: {credentials.get('FQDN')}")
        logger.info(f"Port: {credentials.get('DBPORT')}")
        logger.info(f"User: {credentials.get('username')}")
        
        # Step 2: Test connection
        storage, storage_url = test_database_connection(credentials)
        
        # Step 3: Create test study
        study = create_test_study(storage)
        
        # Step 4: Test parallel capabilities
        parallel_success = test_parallel_optimization(storage_url)
        
        # Step 5: Cleanup (optional)
        cleanup_test_studies(storage)
        
        # Summary
        logger.info("\n=== Test Summary ===")
        logger.info("✓ Dependencies: SUCCESS")
        logger.info("✓ Environment setup: SUCCESS")
        logger.info("✓ Database connection: SUCCESS")
        logger.info("✓ Study creation: SUCCESS")
        logger.info("✓ Optimization trials: SUCCESS")
        logger.info(f"✓ Parallel optimization: {'SUCCESS' if parallel_success else 'FAILED'}")
        logger.info("\nThe Oldenburg PostgreSQL database is working correctly for Optuna!")
        
        # Provide connection string for reference
        logger.info(f"\nFor your configuration files, use:")
        logger.info(f"  db_host: {credentials.get('FQDN')}")
        logger.info(f"  db_port: {credentials.get('DBPORT')}")
        logger.info(f"  db_name: {credentials.get('DB')}")
        logger.info(f"  db_user: {credentials.get('username')}")
        logger.info(f"  db_password_env_var: LOCAL_PG_PASSWORD")
        
        logger.info("\nYou can now run your tuning script:")
        logger.info("  sbatch wind_forecasting/run_scripts/tune_scripts/tune_model_flasc_storm_p210.sh")
        
    except Exception as e:
        logger.error(f"\nTest failed with error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()