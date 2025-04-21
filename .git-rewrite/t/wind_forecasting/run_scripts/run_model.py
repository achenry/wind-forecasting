# cd /home/ahenry/toolboxes/wind_forecasting_env/wind-forecasting/
# python setup.py develop
# pip3 install torch torchvision torchaudio lightning
# cd /home/ahenry/toolboxes/wind_forecasting_env/wind-forecasting/wind_forecasting/models/spacetimeformer
# pip install -r requirements.txt && pip install -e .

from wind_forecasting.datasets.wind_farm import KPWindFarm
import os
from wind_forecasting.run_scripts.helpers import TorchDataModule
from wind_forecasting.models import spacetimeformer as stf
import pytorch_lightning as pl
import warnings
import uuid
warnings.filterwarnings(action="ignore", category=FutureWarning)
from sys import platform

if __name__ == "__main__":

    if platform == "darwin":
        DATA_PATH = "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data/normalized_data.parquet"
    elif platform == "linux":
        DATA_PATH = "/projects/ssc/ahenry/wind_forecasting/awaken_data/normalized_data.parquet/"
    # Configuration
    config = {
        "experiment" : {"run_name": "windfarm_debug"},
        "data": {"data_path": "/Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data/normalized_data.parquet",
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
                },
        "model": {"model_cls": stf.spacetimeformer_model.Spacetimeformer_Forecaster # TODO these should all be defined in one models directory
                },
        "training": {"grad_clip_norm": 0.0, "limit_val_batches": 1.0, "val_check_interval": 1.0, "debug": False, "accumulate": 1.0}
    }

    # Logging
    log_dir = os.getenv("TRAIN_LOG_DIR")
    if log_dir is None:
        log_dir = "./data/TRAIN_LOG_DIR"
        print(
            "Using default wandb log dir path of ./data/TRAIN_LOG_DIR. This can be adjusted with the environment variable `TRAIN_LOG_DIR`"
        )
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create dataset
    dataset = KPWindFarm(**config["data"])
    data_module = TorchDataModule(
        dataset=dataset,
        **config["data"] 
    ) 

    # Forecasting
    forecaster = config["model"]["model_cls"](d_x=dataset.x_dim, d_yc=dataset.yc_dim, d_yt=dataset.yt_dim, 
                                          context_len=dataset.context_len, target_len=dataset.target_len, **config["model"])
    forecaster.set_inv_scaler(dataset.reverse_scaling)
    forecaster.set_scaler(dataset.apply_scaling)

    # Callbacks
    # TODO there are other callbacks in train_spacetimeformer.py if we need
    filename = f"{config['experiment']['run_name']}_" + str(uuid.uuid1()).split("-")[0]
    model_ckpt_dir = os.path.join(log_dir, filename)
    config["experiment"]["model_ckpt_dir"] = model_ckpt_dir
    saving = pl.callbacks.ModelCheckpoint(
        dirpath=model_ckpt_dir,
        monitor="val/loss",
        mode="min",
        filename=f"{config['experiment']['run_name']}" + "{epoch:02d}",
        save_top_k=1,
        auto_insert_metric_name=True,
    )
    callbacks = [saving]
    # test_samples = next(iter(data_module.test_dataloader()))

    # Create Trainer
    if config["training"]["val_check_interval"] <= 1.0:
        val_control = {"val_check_interval": config["training"]["val_check_interval"]}
    else:
        val_control = {"check_val_every_n_epoch": int(config["training"]["val_check_interval"])}

    trainer = pl.Trainer(
        # gpus=args.gpus,
        callbacks=callbacks,
        logger=None,
        accelerator="auto",
        gradient_clip_val=config["training"]["grad_clip_norm"],
        gradient_clip_algorithm="norm",
        overfit_batches=20 if config["training"]["debug"] else 0,
        accumulate_grad_batches=config["training"]["accumulate"],
        sync_batchnorm=True,
        limit_val_batches=config["training"]["limit_val_batches"],
        **val_control,
    )

    # Train model
    trainer.fit(forecaster, datamodule=data_module)

    # Test model
    trainer.test(datamodule=data_module, ckpt_path="best")