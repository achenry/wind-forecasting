# TODO MAKE CONFIG, METHOD INPUTS, DATASET VS DATAMODULE UNIFORM ACROSS JUAN AND AOIFE

import os
import uuid

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

import wandb

from ..models import spacetimeformer as stf
from ..models.base import BaseModel
from ..utils.colors import Colors
from ..utils.config import Config
from ..datasets.data_module import TorchDataModule
from ..datasets.wind_farm import KPWindFarm

# TODO: SpaceTimeFormer model
class SpaceTimeFormer(BaseModel):
    def __init__(self):
        super().__init__()
        self.model_prefix = "stf"
    
    def _initialize_data_module(self, dataset_class, data_params):
        self.data_module = TorchDataModule(
            dataset_class=dataset_class,
            **data_params
        )
        # self.data_module.setup()
        return self.data_module

    def _initialize_model(self, *, dataset, model_params):
        self.model = model_params["model_class"](d_x=dataset.x_dim, d_yc=dataset.yc_dim, d_yt=dataset.yt_dim, 
                                          context_len=dataset.context_len, target_len=dataset.target_len, **model_params)
        self.model.set_inv_scaler(dataset.reverse_scaling)
        self.model.set_scaler(dataset.apply_scaling)
        return model

    def _setup_trainer(self, *, dataset_class, experiment_params, callbacks, model_params, train_params, local_rank):
        wandb_logger = None
        if local_rank == 0:
            wandb_logger = WandbLogger(
                project=self.__class__.__name__,
                name=experiment_params["run_name"],
                log_model=True,
                save_dir=Config.SAVE_DIR,
                config={
                    "model_params": model_params,
                    "train_params": train_params,
                    "dataset": dataset_class,
                }
        )

        if train_params["val_check_interval"] <= 1.0:
            val_control = {"val_check_interval": train_params["val_check_interval"]}
        else:
            val_control = {"check_val_every_n_epoch": int(train_params["val_check_interval"])}

        trainer = L.Trainer(
            # gpus=args.gpus,
            max_epochs=train_params['epochs'],
            accelerator='auto',
            devices='auto',
            strategy='ddp_find_unused_parameters_true',
            logger=wandb_logger if local_rank == 0 else False,
            callbacks=callbacks,
            gradient_clip_algorithm="norm",
            precision=train_params.get('precision', 32),
            overfit_batches=20 if train_params["debug"] else 0,
            accumulate_grad_batches=train_params["accumulate"],
            log_every_n_steps=1,
            enable_progress_bar=(local_rank == 0),
            detect_anomaly=False,
            benchmark=True,
            deterministic=False,
            sync_batchnorm=True,
            limit_val_batches=train_params["limit_val_batches"],
            **val_control,
        )
        return trainer


    def _setup_callbacks(self, *, experiment_params, local_rank):
        # TODO there are other callbacks in train_spacetimeformer.py if we need
        callbacks = []

        # Only add RichProgressBar for rank 0
        if local_rank == 0:
            callbacks.append(
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
            patience=20,
            mode='min',
            min_delta=0.001,
            check_finite=True,
            check_on_train_epoch_end=False,
            verbose=True  # Add verbose output
        )

        filename = f"{experiment_params["run_name"]}_" + str(uuid.uuid1()).split("-")[0]
        model_ckpt_dir = os.path.join(experiment_params["log_dir"], filename)
        experiment_params["model_ckpt_dir"] = model_ckpt_dir
        checkpoint_callback = L.callbacks.ModelCheckpoint(
            dirpath=model_ckpt_dir,
            monitor="val/loss",
            mode="min",
            filename=f"{experiment_params["run_name"]}" + "{epoch:02d}",
            save_top_k=1,
            auto_insert_metric_name=True,
        )
        callbacks.extend([checkpoint_callback, early_stopping])
        return callbacks

        
    def train(self, *, dataset_class=None, model_params=None, train_params=None):
        self.trainer.fit(self.model, datamodule=self.data_module)
        
    def test(self, *, dataset_class=None, model_params=None, train_params=None):
        self.trainer.test(datamodule=self.data_module, ckpt_path="best")

##################################################################################################
#########################################[ MAIN ]#################################################
##################################################################################################

if __name__ == "__main__":
    # Initialize wandb only on rank 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        wandb.login() # Login to wandb website
        
    # Change to checkpoint path to test and validate for pre-trained model
    checkpoint_path = os.path.join(Config.MODEL_DIR, 'sttre-uber-epoch=519-val_loss=6.46.ckpt')
    
    # Initialize model
    model = SpaceTimeFormer()
    
    # MAIN EXPERIMENT PARAMETERS
    experiment_params = {
        "run_name": "windfarm_debug"
    }

    # MAIN DATA PARAMETERS
    data_params = {
        "data_path": "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data/normalized_data.parquet",
        "context_len": 10, #120, # 10 minutes for 5 sec sample size,
        "target_len": 10, # 120, # 10 minutes for 5 sec sample size,
        "target_turbine_ids": ["wt029", "wt034", "wt074"],
        "normalize": False, 
        "batch_size": 128,
        "workers": 6,
        "overfit": False,
        "test_split": 0.15,
        "val_split": 0.15,
        "collate_fn": None
    }

    # INFO: MAIN MODEL PARAMETERS
    model_params = {
        'embed_size': 32, # Determines dimension of the embedding space
        'num_layers': 3, # Number of transformer blocks stacked
        'heads': 4, # Number of heads for spatio-temporal attention
        'forward_expansion': 4, # Multiplier for feedforward network size
        'output_size': 1 # Number of output variables
    }

    model_params = {
        "model_class": stf.spacetimeformer_model.Spacetimeformer_Forecaster
    }

    # INFO: MAIN TRAINING PARAMETERS
    train_params = {
        "grad_clip_norm": 0.0, # Prevents gradient explosion if > 0 
        "limit_val_batches": 1.0, 
        "val_check_interval": 1.0, 
        "debug": False, 
        "accumulate": 1.0,
        # "precision": '32-true', # 16-mixed enables mixed precision training, 32-true is full precision
        # 'batch_size': 32, # larger = more stable gradients
        # 'epochs': 2000, # Maximum number of epochs to train
        # 'lr': 0.0001, # Step size
        # 'dropout': 0.1, # Regularization parameter (prevents overfitting)
        # 'patience': 50, # Number of epochs to wait before early stopping
        # 'accumulate_grad_batches': 2, # Simulates a larger batch size
    }

    # INFO: DATASET CHOICE AND PATHS
    data_module = TorchDataModule(
        dataset_class=KPWindFarm,
        data_params=data_params 
    )

    try:
        if local_rank == 0:
            print(f"\n{Colors.BOLD_BLUE}Processing {dataset_name} dataset...{Colors.ENDC}")
        
        model, trainer, test_results = model.train()
        
        if local_rank == 0:
            
            # model, test_results = test_sttre(
            #     dataset_class, 
            #     data_path, 
            #     model_params, 
            #     train_params,
            #     checkpoint_path
            # )
            # print(f"{Colors.BOLD_GREEN}Completed testing {dataset_name} dataset {Colors.CHECK}{Colors.ENDC}")
        
            print(f"\n{Colors.BOLD_GREEN}Completed {data_module.dataset.__class__.__name__} dataset{Colors.CHECK}{Colors.ENDC}")
            if test_results:
                print(f"Test results: {test_results}")
        
        # Cleanup after training
        del model, trainer
        model.cleanup_memory()
        
    except Exception as e:
        if local_rank == 0:
            print(f"{Colors.RED}Error processing {data_module.dataset.__class__.__name__}: {str(e)} {Colors.CROSS}{Colors.ENDC}")
        model.cleanup_memory()

    if local_rank == 0:
        print(f"\n{Colors.BOLD_GREEN}All experiments completed! {Colors.CHECK}{Colors.ENDC}")