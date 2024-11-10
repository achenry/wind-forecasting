# TODO MAKE CONFIG, METHOD INPUTS, DATASET VS DATAMODULE UNIFORM ACROSS JUAN AND AOIFE
# pip install wandb torch torchvision torchaudio
# ssh ahenry@kestrel-gpu.hpc.nrel.gov
import os
import uuid

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, RichProgressBar, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

import wandb

from wind_forecasting.models.forecaster import Forecaster
from wind_forecasting.utils.colors import Colors
from wind_forecasting.utils.config import Config
from wind_forecasting.datasets.data_module import DataModule

class Callbacks:
    def __init__(self, *, config, local_rank):
        # TODO there are other callbacks in train_spacetimeformer.py if we need
        self.callbacks = []

        # Only add RichProgressBar for rank 0
        if local_rank == 0:
            self.callbacks.append(
                RichProgressBar(
                    theme=RichProgressBarTheme(
                        description="white",
                        progress_bar="#6206E0",
                        progress_bar_finished="#6206E0",
                        progress_bar_pulse="#6206E0",
                        batch_progress="white",
                        time="grey54",
                        processing_speed="grey70",
                        metrics="white"
                    ),
                    leave=True
                )
            )
        
        # Add other callbacks for all ranks
        early_stopping = EarlyStopping(
            monitor='val/loss',
            patience=config["callbacks"].get("patience", 20),
            mode='min',
            min_delta=0.001,
            check_finite=True,
            check_on_train_epoch_end=False,
            verbose=True  # Add verbose output
        )

        filename = f"{config['experiment']['run_name']}_" + str(uuid.uuid1()).split("-")[0]
        model_ckpt_dir = os.path.join(config["experiment"]["log_dir"], filename)
        config["experiment"]["model_ckpt_dir"] = model_ckpt_dir
        checkpoint_callback = ModelCheckpoint(
            dirpath=model_ckpt_dir,
            monitor="val/loss",
            mode="min",
            filename=f"{config['experiment']["run_name"]}" + "{epoch:02d}",
            save_top_k=1,
            auto_insert_metric_name=True,
        )

        lr_monitor = LearningRateMonitor()

        self.callbacks.extend([checkpoint_callback, early_stopping, lr_monitor])
    
    def append(self, obj):
        self.callbacks.append(obj)

    def extend(self, objs):
        self.callbacks.extend(objs)

    def __len__(self):
        return len(self.callbacks)

    def __getitem__(self, i):
        return self.callbacks[i]


##################################################################################################
#########################################[ MAIN ]#################################################
##################################################################################################

if __name__ == "__main__":
    from wind_forecasting.datasets.wind_farm import KPWindFarm
    from wind_forecasting.models.spacetimeformer.spacetimeformer.spacetimeformer_model import Spacetimeformer_Forecaster

    ## DEFINE CONFIGURATION
    config = {
        "experiment": {
            "run_name": "windfarm_debug",
            "log_dir": "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/logging/"
        },
        "dataset": {
            "dataset_class": KPWindFarm,
            "data_path": "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data/normalized_data.parquet",
            "context_len": 120, # 10 minutes for 5 sec sample size,
            "target_len":  120, # 10 minutes for 5 sec sample size,
            "target_turbine_ids": ["wt029", "wt034", "wt074"],
            "normalize": False, 
            "batch_size": 128,
            "workers": 6,
            "overfit": False,
            "test_split": 0.15,
            "val_split": 0.15,
            "collate_fn": None,
            "dataset_kwargs": { # specific to class KPWindFarm or similar 
                "target_turbine_ids": ["wt029", "wt034", "wt074"]
            }
        },
        "model": {
            "model_class": Spacetimeformer_Forecaster,
            'embed_size': 32, # Determines dimension of the embedding space
            'num_layers': 3, # Number of transformer blocks stacked
            'heads': 4, # Number of heads for spatio-temporal attention
            'forward_expansion': 4, # Multiplier for feedforward network size
            'output_size': 1 # Number of output variables
        },
        "callbacks": {

        },
        "training": {
            "grad_clip_norm": 0.0, # Prevents gradient explosion if > 0 
            "limit_val_batches": 1.0, 
            "val_check_interval": 1.0, 
            "debug": False, 
            "accumulate": 1.0,
            "max_epochs": 100, # Maximum number of epochs to train
            # "precision": '32-true', # 16-mixed enables mixed precision training, 32-true is full precision
            # 'batch_size': 32, # larger = more stable gradients
            # 'lr': 0.0001, # Step size
            # 'dropout': 0.1, # Regularization parameter (prevents overfitting)
            # 'patience': 50, # Number of epochs to wait before early stopping
            # 'accumulate_grad_batches': 2, # Simulates a larger batch size
        }
    }
    
    ## SETUP LOGGING

    # Initialize wandb only on rank 0
    os.environ["WANDB_INIT_TIMEOUT"] = "600"
    os.environ["WANDB_INIT_TIMEOUT"] = "300"
    os.environ["WANDB_DEBUG"] = "true"

    wandb_logger = None
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        if not os.path.exists(config["experiment"]["log_dir"]):
            os.makedirs(config["experiment"]["log_dir"])
        # wandb.login() # Login to wandb website
        # entity=aoife-henry-university-of-colorado-boulder
        wandb_logger = WandbLogger(
            project="wf_forecasting",
            name=config["experiment"]["run_name"],
            log_model=True,
            save_dir=config["experiment"]["log_dir"],
            config=config
        )
        
    # Change to checkpoint path to test and validate for pre-trained model
    # checkpoint_path = os.path.join(Config.MODEL_DIR, 'sttre-uber-epoch=519-val_loss=6.46.ckpt')


    ## CREATE DATASET
    data_module = DataModule(
            dataset_class=config["dataset"]["dataset_class"],
            config=config
    )

    ## CREATE MODEL
    model = Forecaster(data_module=data_module, config=config)

    ## CREATE CALLBACKS and TRAINER
    callbacks = Callbacks(config=config, local_rank=local_rank)
    if config["training"]["val_check_interval"] <= 1.0:
        val_control = {"val_check_interval": config["training"]["val_check_interval"]}
    else:
        val_control = {"check_val_every_n_epoch": int(config["training"]["val_check_interval"])}

    trainer = L.Trainer(
        # gpus=args.gpus,
        max_epochs=config["training"].get('max_epochs', None),
        accelerator='auto',
        devices='auto',
        # strategy='ddp_find_unused_parameters_true',
        logger=wandb_logger if local_rank == 0 else False,
        callbacks=callbacks,
        gradient_clip_algorithm="norm",
        precision=config["training"].get('precision', None),
        overfit_batches=20 if config["training"]["debug"] else 0,
        accumulate_grad_batches=config["training"]["accumulate"],
        log_every_n_steps=1,
        enable_progress_bar=(local_rank == 0),
        detect_anomaly=False,
        benchmark=True,
        deterministic=False,
        sync_batchnorm=True,
        limit_val_batches=config["training"]["limit_val_batches"],
        **val_control,
    )

    if local_rank == 0:
        print(f"\n{Colors.BOLD_BLUE}Processing {data_module.dataset.__class__.__name__} dataset...{Colors.ENDC}")
    
    ## TRAIN
    train_results = model.train(trainer=trainer, data_module=data_module) # TODO JUAN WHY TEST IN RANK 0?

    ## TEST
    test_results = model.test(trainer=trainer, data_module=data_module)

    print(f"\n{Colors.BOLD_GREEN}Completed {data_module.dataset.__class__.__name__} dataset{Colors.CHECK}{Colors.ENDC}")
    if test_results:
        print(f"Test results: {test_results}")

    model.cleanup_memory()
    
    # try:

        
    #     if local_rank == 0:

            
            
            # model, test_results = test_sttre(
            #     dataset_class, 
            #     data_path, 
            #     model_params, 
            #     train_params,
            #     checkpoint_path
            # )
            # print(f"{Colors.BOLD_GREEN}Completed testing {dataset_name} dataset {Colors.CHECK}{Colors.ENDC}")
        
            
        
        # Cleanup after training
        # del model, trainer
        
        
    # except Exception as e:
    #     if local_rank == 0:
    #         print(f"{Colors.RED}Error processing {data_module.dataset.__class__.__name__}: {str(e)} {Colors.CROSS}{Colors.ENDC}")
    #     model.cleanup_memory()

    if local_rank == 0:
        print(f"\n{Colors.BOLD_GREEN}All experiments completed! {Colors.CHECK}{Colors.ENDC}")