import os
import pickle
import shutil
import tempfile
import time
import logging

import optuna
from optuna.samplers import TPESampler

class OptunaSamplerPrunerPersistence:
    def __init__(self, config, seed):
        self.config = config
        self.seed = seed

    def _create_new_sampler_pruner(self, pruner):
        """
        Create new sampler and pruner instances with full configuration.
        
        Args:
            pruner: Pre-configured pruner instance
            
        Returns:
            tuple: (sampler, pruner)
        """
        sampler = TPESampler(
            seed=self.seed,
            n_startup_trials=self.config["optuna"]["sampler_params"]["tpe"].get("n_startup_trials", 16),
            multivariate=self.config["optuna"]["sampler_params"]["tpe"].get("multivariate", True),
            constant_liar=self.config["optuna"]["sampler_params"]["tpe"].get("constant_liar", True),
            group=self.config["optuna"]["sampler_params"]["tpe"].get("group", False)
        )
        return sampler, pruner

    def _save_sampler_pruner_atomic(self, sampler, pruner, sampler_path, pruner_path):
        """
        Save sampler and pruner to pickle files using atomic operations.
        
        Args:
            sampler: Sampler instance to save
            pruner: Pruner instance to save
            sampler_path: Path to save sampler pickle
            pruner_path: Path to save pruner pickle
        """
        # Ensure pickle directory exists
        pickle_dir = os.path.dirname(sampler_path)
        os.makedirs(pickle_dir, exist_ok=True)
        
        # Save sampler atomically
        with tempfile.NamedTemporaryFile(mode='wb', dir=pickle_dir, delete=False) as tmp_file:
            pickle.dump(sampler, tmp_file)
            tmp_sampler_path = tmp_file.name
        shutil.move(tmp_sampler_path, sampler_path)
        
        # Save pruner atomically
        with tempfile.NamedTemporaryFile(mode='wb', dir=pickle_dir, delete=False) as tmp_file:
            pickle.dump(pruner, tmp_file)
            tmp_pruner_path = tmp_file.name
        shutil.move(tmp_pruner_path, pruner_path)
        
        logging.info(f"Saved sampler and pruner to {sampler_path} and {pruner_path}")

    def _load_sampler_pruner(self, sampler_path, pruner_path):
        """
        Load sampler and pruner from pickle files.
        
        Args:
            sampler_path: Path to sampler pickle file
            pruner_path: Path to pruner pickle file
            
        Returns:
            tuple: (sampler, pruner)
        """
        with open(sampler_path, 'rb') as f:
            sampler = pickle.load(f)
        
        with open(pruner_path, 'rb') as f:
            pruner = pickle.load(f)
        
        logging.info(f"Loaded sampler and pruner from {sampler_path} and {pruner_path}")
        return sampler, pruner

    def _wait_and_load_sampler_pruner(self, sampler_path, pruner_path, max_wait=300):
        """
        Wait for pickle files to exist and then load sampler and pruner.
        
        Args:
            sampler_path: Path to sampler pickle file
            pruner_path: Path to pruner pickle file
            max_wait: Maximum wait time in seconds
            
        Returns:
            tuple: (sampler, pruner)
            
        Raises:
                TimeoutError: If files don't appear within max_wait seconds
        """
        wait_interval = 5  # seconds
        elapsed = 0
        
        while elapsed < max_wait:
            if os.path.exists(sampler_path) and os.path.exists(pruner_path):
                try:
                    return self._load_sampler_pruner(sampler_path, pruner_path)
                except Exception as e:
                    logging.warning(f"Failed to load pickle files (attempt after {elapsed}s): {e}")
                    time.sleep(wait_interval)
                    elapsed += wait_interval
                    continue
            else:
                missing = []
                if not os.path.exists(sampler_path):
                    missing.append("sampler")
                if not os.path.exists(pruner_path):
                    missing.append("pruner")
                logging.info(f"Waiting for {', '.join(missing)} pickle file(s) to appear... ({elapsed}/{max_wait}s)")
                time.sleep(wait_interval)
                elapsed += wait_interval
        
        raise TimeoutError(f"Sampler/pruner pickle files did not appear within {max_wait} seconds")

    def get_sampler_pruner_objects(self, worker_id, pruner, restart_tuning, final_study_name, optuna_storage, pickle_dir):
        """
        Determine whether to create new sampler/pruner objects or load them from pickle files.
        
        Args:
            worker_id: Worker ID string
            pruner: Pre-configured pruner instance
            restart_tuning: Whether we're restarting tuning
            final_study_name: Final study name
            optuna_storage: Optuna storage instance
            pickle_dir: Directory for pickle files
            
        Returns:
            tuple: (sampler, pruner)
        """
        sampler_path = os.path.join(pickle_dir, f"sampler_{final_study_name}.pkl")
        pruner_path = os.path.join(pickle_dir, f"pruner_{final_study_name}.pkl")
        
        if worker_id == '0':
            # Worker 0 logic
            study_exists = False
            try:
                # Check if study exists in storage
                available_studies = [study.study_name for study in optuna_storage.get_all_studies()]
                study_exists = final_study_name in available_studies
            except Exception as e:
                logging.warning(f"Could not check existing studies: {e}")
                study_exists = False
            
            if restart_tuning or not study_exists:
                # Scenario 1 (New Study - restart_tuning=True) OR Scenario 3 (First-time study)
                logging.info(f"Worker 0: Creating new sampler/pruner (restart_tuning={restart_tuning}, study_exists={study_exists})")
                sampler, pruner_to_save = self._create_new_sampler_pruner(pruner)
                self._save_sampler_pruner_atomic(sampler, pruner_to_save, sampler_path, pruner_path)
                return sampler, pruner_to_save
            else:
                # Scenario 2 (Continuing Existing Study - restart_tuning=False and study exists)
                logging.info(f"Worker 0: Attempting to load existing sampler/pruner for study '{final_study_name}'")
                try:
                    return self._load_sampler_pruner(sampler_path, pruner_path)
                except Exception as e:
                    logging.warning(f"Worker 0: Failed to load sampler/pruner from pickle files: {e}")
                    logging.info("Worker 0: Falling back to creating new sampler/pruner and saving them")
                    sampler, pruner_to_save = self._create_new_sampler_pruner(pruner)
                    self._save_sampler_pruner_atomic(sampler, pruner_to_save, sampler_path, pruner_path)
                    return sampler, pruner_to_save
        else:
            # Non-Worker-0 logic
            logging.info(f"Worker {worker_id}: Waiting for sampler/pruner pickle files from worker 0")
            return self._wait_and_load_sampler_pruner(sampler_path, pruner_path)