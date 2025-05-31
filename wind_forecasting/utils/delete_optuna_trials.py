#!/usr/bin/env python
"""
Utility script to copy Optuna studies between databases, excluding specific trials.

This script allows users to copy an entire study from one database to another,
while excluding specified trial ID ranges. This is useful for creating filtered
versions of studies by removing unwanted or failed trials.
"""

import argparse
import logging
import sys
from typing import List, Set, Tuple

import optuna
try:
    import psycopg2 # Explicitly import psycopg2 to help SQLAlchemy find the dialect
    psycopg2_imported = True
except ImportError:
    psycopg2_imported = False # Allow script to load even if psycopg2 isn't installed yet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def parse_trial_ranges_to_exclude(ranges_str: str) -> Set[int]:
    """
    Parse a string of comma-separated ranges into a set of trial IDs to exclude.
    
    Args:
        ranges_str: A string in the format "1-5,7,10-15"
    
    Returns:
        A set of integers representing all trial IDs in the specified ranges
        
    Raises:
        ValueError: If the input string format is invalid
    """
    trial_ids = set()
    ranges = ranges_str.split(',')
    
    for r in ranges:
        try:
            if '-' in r:
                # Handle range (e.g., "1-5")
                start_str, end_str = r.split('-')
                start = int(start_str.strip())
                end = int(end_str.strip())
                
                if start > end:
                    logger.warning(f"Invalid range: {start}-{end}, start > end. Swapping values.")
                    start, end = end, start
                    
                trial_ids.update(range(start, end + 1))  # +1 for inclusive range
            else:
                # Handle single number (e.g., "7")
                trial_ids.add(int(r.strip()))
        except ValueError:
            raise ValueError(f"Invalid trial range format: '{r}'. Expected format: '1-5' or '7'")
            
    return trial_ids

def _get_sqlalchemy_url(db_url: str) -> str:
    """Ensure the URL uses the explicit postgresql+psycopg2 scheme if needed."""
    if db_url.startswith("postgres://"):
        logger.debug("Replacing postgres:// with postgresql+psycopg2:// in DB URL")
        return db_url.replace("postgres://", "postgresql+psycopg2://", 1)
    elif db_url.startswith("postgresql://") and "+psycopg2" not in db_url:
         # Also handle the standard postgresql:// if the driver isn't specified
         logger.debug("Adding +psycopg2 driver to postgresql:// DB URL")
         return db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return db_url

def copy_study_excluding_trials(source_db_url: str, target_db_url: str, study_name: str, trials_to_exclude: Set[int]) -> Tuple[int, int, int]:
    """
    Copy an Optuna study from source to target database, excluding specified trials.
    
    Args:
        source_db_url: The source database URL
        target_db_url: The target database URL
        study_name: The name of the study
        trials_to_exclude: Set of trial IDs to exclude from copying
        
    Returns:
        Tuple containing (number of trials read, number of trials excluded, number of trials copied)
    """
    # Ensure URLs use explicit driver format for SQLAlchemy
    effective_source_url = _get_sqlalchemy_url(source_db_url)
    effective_target_url = _get_sqlalchemy_url(target_db_url)

    # Load source study
    try:
        logger.info(f"Attempting to load source study '{study_name}' from {effective_source_url}")
        source_study = optuna.load_study(study_name=study_name, storage=effective_source_url)
        logger.info(f"Successfully loaded source study '{study_name}' with {len(source_study.trials)} trials")
    except Exception as e:
        logger.error(f"Failed to load source study from {effective_source_url}: {e}")
        raise
    
    # Get and filter trials
    all_trials = source_study.get_trials(deepcopy=False)
    trials_to_copy = [trial for trial in all_trials if trial.number not in trials_to_exclude]
    logger.info(f"Read {len(all_trials)} trials, {len(trials_to_exclude)} will be excluded, {len(trials_to_copy)} will be copied")
    
    # Check if target study exists
    try:
        logger.info(f"Checking if target study '{study_name}' exists in {effective_target_url}")
        optuna.load_study(study_name=study_name, storage=effective_target_url)
        error_msg = f"Target study '{study_name}' already exists in '{target_db_url}'. Please delete it manually or choose a different name if you want to proceed."
        logger.error(error_msg)
        raise RuntimeError("Target study already exists.")
    except (optuna.exceptions.DuplicatedStudyError, KeyError, ValueError): # Adjusted to catch common Optuna errors for non-existent study
        logger.info(f"Target study '{study_name}' does not exist in '{target_db_url}'. Safe to proceed with creation.")
    except Exception as e: # Catch any other unexpected error during load attempt
        # Check if the error is the specific NoSuchModuleError, which might indicate the URL fix didn't work
        if "NoSuchModuleError" in str(e) and "sqlalchemy.dialects:postgres" in str(e):
             logger.error(f"Failed to check target study due to dialect loading error even after URL modification: {e}")
             raise # Re-raise the critical error
        logger.warning(f"Unexpected error while checking for target study '{study_name}' using {effective_target_url}: {e}. Assuming it's safe to proceed.")

    
    # Create target study
    try:
        logger.info(f"Attempting to create target study '{study_name}' in {effective_target_url}")
        target_study = optuna.create_study(study_name=study_name, storage=effective_target_url, direction=source_study.direction, load_if_exists=False)
        logger.info(f"Successfully created target study '{study_name}' in {target_db_url}")
    except Exception as e:
        logger.error(f"Failed to create target study in {target_db_url} (using {effective_target_url}): {e}")
        raise
    
    # Copy study attributes
    try:
        for key, value in source_study.user_attrs.items():
            target_study.set_user_attr(key, value)
        for key, value in source_study.system_attrs.items():
            target_study.set_system_attr(key, value)
        logger.info(f"Successfully copied study attributes")
    except Exception as e:
        logger.error(f"Failed to copy study attributes: {e}")
        raise
    
    # Add trials
    try:
        target_study.add_trials(trials_to_copy)
        logger.info(f"Successfully added {len(trials_to_copy)} trials to target study")
    except Exception as e:
        logger.error(f"Failed to add trials to target study: {e}")
        raise
    
    return len(all_trials), len(trials_to_exclude), len(trials_to_copy)


def main():
    """Main function to parse arguments and copy study excluding specific trials."""
    parser = argparse.ArgumentParser(
        description="Copy an Optuna study from source to target database, excluding specified trials."
    )
    parser.add_argument(
        "--source-db-url", 
        required=True,
        help="Source database URL string (e.g., postgresql://user:password@localhost/defaultdb or postgres://...)"
    )
    parser.add_argument(
        "--target-db-url", 
        required=True,
        help="Target database URL string (e.g., postgresql://user:password@localhost/optuna or postgres://...)"
    )
    parser.add_argument(
        "--study-name", 
        required=True,
        help="Study name string"
    )
    parser.add_argument(
        "--trial-ranges-to-exclude", 
        required=True,
        help="Comma-separated ranges of trial IDs to exclude (e.g., '27-41,73-167,205')"
    )
    
    args = parser.parse_args()
    
    try:
        # Parse trial ranges
        logger.info(f"Parsing trial ranges to exclude: {args.trial_ranges_to_exclude}")
        trials_to_exclude = parse_trial_ranges_to_exclude(args.trial_ranges_to_exclude)
        logger.info(f"Parsed {len(trials_to_exclude)} trial IDs to exclude: {sorted(trials_to_exclude)}")
        
        # Copy study excluding specified trials
        logger.info(f"Connecting to databases and copying study...")
        trials_read, trials_excluded, trials_copied = copy_study_excluding_trials(
            args.source_db_url, 
            args.target_db_url, 
            args.study_name, 
            trials_to_exclude
        )
        
        # Print summary
        logger.info(f"\nCopy Summary:")
        logger.info(f"  - Total trials read: {trials_read}")
        logger.info(f"  - Trials excluded: {trials_excluded}")
        logger.info(f"  - Trials copied: {trials_copied}")
        
        return 0
        
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    # Check if psycopg2 is actually available before running main logic
    if not psycopg2_imported:
         logger.critical("psycopg2 library is required but not found. Please install it (e.g., 'pip install psycopg2-binary' or 'mamba install psycopg2-binary').")
         sys.exit(1)
    sys.exit(main())