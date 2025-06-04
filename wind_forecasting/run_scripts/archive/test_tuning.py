from mysql.connector import connect as sql_connect #, MySQLInterfaceError
from optuna.storages import RDBStorage, RetryFailedTrialCallback
import os
from time import sleep
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    db_host = "tlv51ahenry01.nrel.gov" # 10.121.2.72
    db_port = 3306
    db_user = "optuna_user"
    db_name = f"tuning_testmodel_test_runname"
    
    restart_tuning = True
    
    try:
        # Use the WORKER_RANK variable set explicitly in the Slurm script's nohup block
        rank = int(os.environ.get('WORKER_RANK', '0'))
    except ValueError:
        logging.warning("Could not parse WORKER_RANK, assuming rank 0.")
        rank = 0
    logging.info(f"Determined worker rank from WORKER_RANK: {rank}")

    logging.info(f"Rank {rank}: Setting up MySQL connection to host={db_host}, user={db_user}, db={db_name}")
    # Add connect_args for timeout, etc. if needed
    engine_kwargs = {
            "pool_size": 4,
            "max_overflow": 4,
            "pool_timeout": 30,
            "pool_recycle": 1800,
            "pool_pre_ping": True
            # "connect_args": {"application_name": f"optuna_worker_0_main"}
    }
    url_user_part = db_user
    optuna_storage_url = f"mysql+mysqlconnector://{url_user_part}@{db_host}:{db_port}/{db_name}"
        
    if rank == 0:
        connection = None
        try:
            # Try connecting without specifying the database first to check server access and create DB if needed
            connection = sql_connect(host=db_host, user=db_user, password=None, port=db_port)
            cursor = connection.cursor()
            
            cursor.execute("SHOW DATABASES")
            databases = [item[0] for item in cursor.fetchall()]
            logging.info(f"L48, Rank 0: Available databases: {databases}")
            if db_name not in databases:
                logging.info(f"Rank 0: Database '{db_name}' not found in list {databases}. Creating database.")
                cursor.execute(f"CREATE DATABASE {db_name}")
                connection.commit()
                logging.info(f"Rank 0: Database '{db_name}' created successfully.")
                
                cursor.execute("SHOW DATABASES")
                databases = [item[0] for item in cursor.fetchall()]
                logging.info(f"After create, Rank 0: Available databases: {databases}")
                
            elif restart_tuning:
                logging.info(f"Rank 0: Database '{db_name}' already exists.")
                logging.warning(f"Rank 0: --restart_tuning set. Dropping and recreating Optuna tables in database '{db_name}'.")
                
                # cursor.execute(f"USE {db_name}; SHOW TABLES")
                # tables = [item[0] for item in cursor.fetchall()]
                # logging.info(f"L61, Rank 0: Available tables in database {db_name}: {tables}")
                
                logging.info(f"Rank 0: Attempting to drop database `{db_name}` ")
                cursor.execute(f"DROP DATABASE IF EXISTS `{db_name}`")
                connection.commit()
                
                cursor.execute("SHOW DATABASES")
                databases = [item[0] for item in cursor.fetchall()]
                logging.info(f"After drop, Rank 0: Available databases: {databases}")
                
                logging.info(f"Rank 0: Attempting to create database `{db_name}` ")
                cursor.execute(f"CREATE DATABASE {db_name}")
                connection.commit()
                
                cursor.execute("SHOW DATABASES")
                databases = [item[0] for item in cursor.fetchall()]
                logging.info(f"After drop/create, Rank 0: Available databases: {databases}")
                
                # for table in tables:
                #     logging.info(f"Rank 0: Attempting to remove table `{table}` from database `{db_name}`")
                #     cursor.execute(f"DROP TABLE IF EXISTS `{table}`")
                #     connection.commit()
            
            cursor.execute(f"USE {db_name}; SHOW TABLES")
            tables = [item[0] for item in cursor.fetchall()]
            logging.info(f"Rank 0: Available tables in database {db_name}: {tables}")
            
            # storage = RDBStorage(
            #     url=optuna_storage_url,
            #     engine_kwargs=engine_kwargs,
            #     heartbeat_interval=60,
            #     failed_trial_callback=RetryFailedTrialCallback(max_retry=3)
            # )

        except Exception as e:
            logging.error(f"Rank 0: Failed to connect to MySQL server or manage database '{db_name}': {e}", exc_info=True)
            raise RuntimeError(f"Rank 0 failed MySQL setup for {db_name}") from e
        finally:
            cursor.close()
            if connection:
                connection.close()
    
    # else:
    # All ranks create the RDBStorage instance
    # Add a small delay and retry mechanism for loading, in case rank 0 is slightly delayed
    max_retries = 6 # Increased retries slightly
    retry_delay = 5 # Increased delay slightly
    sleep(retry_delay)  # Ensure rank 0 has time to set up the database before others try to connect
    for attempt in range(max_retries):
        try:
            # Attempt to create the RDBStorage instance
            storage = RDBStorage(
                url=optuna_storage_url,
                engine_kwargs=engine_kwargs,
                heartbeat_interval=60,
                failed_trial_callback=RetryFailedTrialCallback(max_retry=3)
            )
            logging.info(f"Rank {rank}: Successfully connected to MySQL DB using URL: mysql+mysqlconnector://{db_user}@***:{db_port}/{db_name}")
            break  # Exit the loop if successful
        except Exception as e:
            logging.error(f"Rank {rank}: Failed to create RDBStorage instance: {e}", exc_info=True)
            if attempt < max_retries - 1:
                logging.info(f"Rank {rank}: Retrying in {retry_delay} seconds...")
                sleep(retry_delay)
            else:
                raise RuntimeError(f"Rank {rank} failed to create RDBStorage after {max_retries} attempts") from e