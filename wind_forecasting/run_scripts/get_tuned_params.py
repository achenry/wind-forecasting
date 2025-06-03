import argparse
import logging
import os
import torch
import random
import numpy as np
import inspect

from lightning.pytorch import Trainer
import yaml

# Internal imports
from wind_forecasting.utils.optuna_db_utils import setup_optuna_storage
from wind_forecasting.run_scripts.testing import get_checkpoint, load_estimator_from_checkpoint
from wind_forecasting.run_scripts.tuning import get_tuned_params, generate_df_setup_params

from pytorch_transformer_ts.informer.lightning_module import InformerLightningModule
from pytorch_transformer_ts.informer.estimator import InformerEstimator
from pytorch_transformer_ts.autoformer.estimator import AutoformerEstimator
from pytorch_transformer_ts.autoformer.lightning_module import AutoformerLightningModule
from pytorch_transformer_ts.spacetimeformer.estimator import SpacetimeformerEstimator
from pytorch_transformer_ts.spacetimeformer.lightning_module import SpacetimeformerLightningModule
from pytorch_transformer_ts.tactis_2.estimator import TACTiS2Estimator as TactisEstimator
from pytorch_transformer_ts.tactis_2.lightning_module import TACTiS2LightningModule as TactisLightningModule
from gluonts.torch.distributions import LowRankMultivariateNormalOutput

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    rank = 0

    # %% PARSE ARGUMENTS
    parser = argparse.ArgumentParser(description="Run a model on a dataset")
    parser.add_argument("--config", type=str, help="Path to config file", default="examples/inputs/training_inputs_aoifemac_flasc.yaml")
    parser.add_argument("-m", "--model", type=str, choices=["informer", "autoformer", "spacetimeformer", "tactis"], required=True)
    parser.add_argument("-chk", "--checkpoint", type=str, required=False, default=None,
                        help="Which checkpoint to use: can be equal to 'None' to start afresh with training mode, 'latest', 'best', or an existing checkpoint path.")
    args = parser.parse_args()

    # %% SETUP SEED
    logging.info(f"Setting random seed to {args.seed}")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # %% PARSE CONFIG
    logging.info(f"Parsing configuration from yaml and command line arguments")
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
        
    log_dir = config["experiment"]["log_dir"]
    project_name = f"{config['experiment'].get('project_name', 'wind_forecasting')}_{args.model}"
    
    # %% DEFINE ESTIMATOR
    # Initialize storage and connection info variables
    optuna_storage = None
    db_connection_info = None # Will hold pg_config if PostgreSQL is used
    db_setup_params = None # Initialize
    
    # Use globals() to fetch the module and estimator classes dynamically
    LightningModuleClass = globals()[f"{args.model.capitalize()}LightningModule"]
    EstimatorClass = globals()[f"{args.model.capitalize()}Estimator"]
    DistrOutputClass = globals()[config["model"]["distr_output"]["class"]]

    # %% SETUP OPTUNA STORAGE
    

    # Generate DB setup parameters regardless of mode (needed for study name)
    logging.info("Generating Optuna DB setup parameters...")
    db_setup_params = generate_df_setup_params(args.model, config)

    # Determine if restart_tuning should be overridden
    # We never want to restart/delete the study when just loading parameters
    logging.info(f"Setting up Optuna storage (rank {rank}) using backend from config...")
    # Call setup_optuna_storage using config-derived params and effective restart flag
    optuna_storage, db_connection_info = setup_optuna_storage(
        db_setup_params=db_setup_params,
        restart_tuning=False, # Use the potentially overridden flag
        rank=rank
        # No force_sqlite_path argument anymore
    )
    logging.info(f"Optuna storage setup complete. Storage type: {type(optuna_storage).__name__}")
    if db_connection_info:
            logging.info("PostgreSQL connection info returned (likely tuning mode).")
    else:
            logging.info("No connection info returned (likely SQLite or Journal mode).")
        
    # get parameters expected by estimator and trainer
    estimator_sig = inspect.signature(EstimatorClass.__init__)
    estimator_params = [param.name for param in estimator_sig.parameters.values()]
    
    trainer_sig = inspect.signature(Trainer.__init__)
    trainer_params = [param.name for param in trainer_sig.parameters.values()]
    
    # get default params
    model_hparams = config["model"].get(args.model, {})
    
    # get tuned params
    found_tuned_params = True
    if args.use_tuned_parameters:
        try:
            logging.info(f"Getting tuned parameters.")
            
            tuned_params = get_tuned_params(optuna_storage, db_setup_params["study_name"])
            
            config["model"]["distr_output"]["kwargs"].update({k: v for k, v in tuned_params.items() if k in config["model"]["distr_output"]["kwargs"]})
            config["dataset"].update({k: v for k, v in tuned_params.items() if k in config["dataset"]})
            
            config["trainer"].update({k: v for k, v in tuned_params.items() if k in trainer_params})
            
            model_hparams.update(
                {k: v for k, v in tuned_params.items() if k in estimator_params})
            
            context_length_factor = tuned_params.get('context_length_factor', config["dataset"].get("context_length_factor", None)) # Default to config or 2 if not in trial/config

        except FileNotFoundError as e:
            logging.warning(e)
            found_tuned_params = False
        except KeyError as e:
            logging.warning(f"KeyError accessing Optuna config for tuned params: {e}. Using defaults.")
            found_tuned_params = False
    else:
        found_tuned_params = False 
    
    if found_tuned_params:
        logging.info(f"Updating estimator {args.model.capitalize()} kwargs with tuned parameters {tuned_params}")
    else:
        logging.info(f"Updating estimator {args.model.capitalize()} kwargs with default parameters")
        if "context_length_factor" in config["model"][args.model]:
            del config["model"][args.model]["context_length_factor"]
        
    # Use the get_checkpoint function to handle checkpoint finding
    if args.checkpoint:
        # Set up parameters for checkpoint finding
        metric = config.get("trainer", {}).get("monitor_metric", "val_loss")
        mode = config.get("optuna", {}).get("direction", "minimize")
        mode_mapping = {"minimize": "min", "maximize": "max"}
        mode = mode_mapping.get(mode, "min")

        base_checkpoint_dir = os.path.join(log_dir, project_name)
        logging.info(f"Checkpoint selection: Monitoring metric '{metric}' with mode '{mode}' in directory '{base_checkpoint_dir}'")
    
        checkpoint_path = get_checkpoint(args.checkpoint, metric, mode, base_checkpoint_dir)
        checkpoint_hparams = load_estimator_from_checkpoint(checkpoint_path, LightningModuleClass, config, args.model)
        
        model_hparams.update(checkpoint_hparams["init_args"]["model_config"])
        
        logging.info(f"Updating estimator {args.model.capitalize()} kwargs with checkpoint parameters {checkpoint_hparams['init_args']}.")
    else:
        checkpoint_path = None
        
if __name__ == "__main__":
    main()