# pip install wandb torch torchvision torchaudio
# ssh ahenry@kestrel-gpu.hpc.nrel.gov
import os
import uuid

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, RichProgressBar, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

import multiprocessing as mp
import pandas as pd

from wind_forecasting.utils.colors import Colors
from wind_forecasting.datasets.data_module import DataModule
from wind_forecasting.utils.cleanup import cleanup_memory, cleanup_old_checkpoints

class Callbacks:
    def __init__(self, *, config, local_rank):
        self.callbacks = []

        # Only add RichProgressBar for rank 0
        if "progress_bar" in config["callbacks"] and local_rank == 0:
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
        if "early_stopping" in config["callbacks"]:
            early_stopping = EarlyStopping(
                monitor='val/loss',
                patience=config["callbacks"].get("patience", 2),
                # mode='min',
                # min_delta=0.001,
                # check_finite=True,
                # check_on_train_epoch_end=False,
                # verbose=True  # Add verbose output
            )

        if "model_checkpoint" in config["callbacks"]:
            filename = f"{config['experiment']['run_name']}_" + str(uuid.uuid1()).split("-")[0]
            model_ckpt_dir = os.path.join(config["experiment"]["log_dir"], filename)
            config["experiment"]["model_ckpt_dir"] = model_ckpt_dir
            checkpoint_callback = ModelCheckpoint(
                dirpath=model_ckpt_dir,
                monitor="val/loss",
                mode="min",
                filename=f"{config['experiment']['run_name']}" + "{epoch:02d}",
                save_top_k=1,
                auto_insert_metric_name=True,
            )

        if "lr_monitor" in config["callbacks"]:
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
    from sys import platform

    if platform == "darwin":
        LOG_DIR = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/logging/"
        # DATA_PATH = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data/short_loaded_data_calibrated_filtered_split_imputed_normalized.parquet"
        DATA_PATH = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data/short_loaded_data_normalized.parquet"
        NORM_CONSTS = pd.read_csv("/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data/normalization_consts.csv", index_col=None)
        n_workers = mp.cpu_count()
        accelerator = "auto"
        devices = "auto"
        num_nodes = 1
        strategy = "auto"
        dataset_class = "KPWindFarm"
        model_class = "Spacetimeformer_Forecaster"
    elif platform == "linux":
        LOG_DIR = "/projects/ssc/ahenry/wind_forecasting/logging/"
        DATA_PATH = "/projects/ssc/ahenry/wind_forecasting/awaken_data/filled_data_calibrated_filtered_split_imputed_normalized.parquet"
        NORM_CONSTS = pd.read_csv("/projects/ssc/ahenry/wind_forecasting/awaken_data/normalization_consts.csv", index_col=None)
        n_workers = int(os.environ["SLURM_GPUS_ON_NODE"])
        accelerator = "auto"
        devices = 2
        num_nodes = 1
        strategy = "ddp_find_unused_parameters_true"
        dataset_class = "KPWindFarm"
        model_class = "Spacetimeformer_Forecaster"

    ## DEFINE CONFIGURATION
    config = {
        "experiment": {
            "run_name": "windfarm_debug",
            "log_dir": LOG_DIR
        },
        "dataset": {
            "dataset_class": dataset_class,
            "data_path": DATA_PATH,
            "normalization_consts": NORM_CONSTS,
            "context_len": 4, # 120=10 minutes for 5 sec sample size,
            "target_len":  3, # 120=10 minutes for 5 sec sample size,
            # "target_turbine_ids": ["wt029", "wt034", "wt074"],
            "normalize": False, 
            "batch_size": 128,
            "workers": n_workers,
            "overfit": False,
            "test_split": 0.15,
            "val_split": 0.15,
            "collate_fn": None,
            "dataset_kwargs": { # specific to class KPWindFarm or similar 
                "target_turbine_ids": ["wt029"] #, "wt034", "wt074"]
            }
        },
        "model": {
            "model_class": model_class,
            'embed_size': 32, # Determines dimension of the embedding space
            'num_layers': 3, # Number of transformer blocks stacked
            'heads': 4, # Number of heads for spatio-temporal attention
            'forward_expansion': 4, # Multiplier for feedforward network size
            'output_size': 1, # Number of output variables,
            "d_model": 5,
            "d_queries_keys": 5, 
            "d_values": 5, 
            "d_ff": 5
        },
        "callbacks": {
            "progress_bar": {}, 
            "early_stopping": {}, 
            "model_checkpoint": {}, 
            "lr_monitor": {True}
        },
        "trainer": {
            "grad_clip_norm": 0.0, # Prevents gradient explosion if > 0 
            "limit_val_batches": 1.0, 
            "val_check_interval": 1,
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
    # TODO JUAN are the rank conditinals necessary for WandbLogger, progress bar?
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

    ## CREATE DATASET
    data_module = DataModule(
            dataset_class=globals()[config["dataset"]["dataset_class"]],
            config=config
    )
    
    ## CREATE MODEL/LightningModule
    model = globals()[config["model"]["model_class"]](
        d_yc=data_module.dataset.yc_dim,
        d_yt=data_module.dataset.yt_dim,
        d_x=data_module.dataset.x_dim,
        context_len=config["dataset"]["context_len"],
        target_len=config["dataset"]["target_len"],
        scaler_func=data_module.dataset.apply_scaling,
        inv_scaler_func=data_module.dataset.reverse_scaling,
        **config["model"]
    )

    ## CREATE CALLBACKS and TRAINER
    callbacks = Callbacks(config=config, local_rank=local_rank)

    trainer = L.Trainer(
        accelerator=accelerator,
        num_nodes=num_nodes,
        devices=devices,
        strategy=strategy,
        logger=wandb_logger if local_rank == 0 else False,
        callbacks=callbacks,
        limit_train_batches=100, # uncomment to debug
        max_epochs=1, # uncomment to debug
        # max_epochs=config["trainer"].get('max_epochs', None),
        # gradient_clip_algorithm="norm",
        # precision=config["trainer"].get('precision', None),
        # overfit_batches=20 if config["trainer"]["debug"] else 0,
        # accumulate_grad_batches=config["trainer"].get("accumulate", None),
        # log_every_n_steps=1,
        # enable_progress_bar=(local_rank == 0),
        # detect_anomaly=False,
        # benchmark=True,
        # deterministic=False,
        # sync_batchnorm=True,
        # limit_val_batches=config["trainer"].get("limit_val_batches", None),
        # val_check_interval=config["trainer"].get("val_check_interval", None),
        # check_val_every_n_epoch=config["trainer"].get("check_val_every_n_epoch", None)
    )

    if local_rank == 0:
        print(f"\n{Colors.BOLD_BLUE}Processing {data_module.dataset.__class__.__name__} dataset...{Colors.ENDC}")
    
    ## TRAIN
    trainer.fit(model=model, datamodule=data_module)

    ## TEST
    # trainer.test(datamodule=data_module, ckpt_path="best")

    test_samples = next(iter(data_module.test_dataloader()))
    model.to("cuda")
    xc, yc, xt, _ = test_samples
    yt_pred = model.predict(xc, yc, xt)
    
    # Change to checkpoint path to test and validate for pre-trained model
    # checkpoint_path = os.path.join(Config.MODEL_DIR, 'sttre-uber-epoch=519-val_loss=6.46.ckpt')
    # model = model.load_from_checkpoint(checkpoint_path)
    # model.eval()

    if local_rank == 0:
        print(f"\n{Colors.BOLD_GREEN}Completed {data_module.dataset.__class__.__name__} dataset{Colors.CHECK}{Colors.ENDC}")

    cleanup_memory()

    if local_rank == 0:
        print(f"\n{Colors.BOLD_GREEN}All experiments completed! {Colors.CHECK}{Colors.ENDC}")