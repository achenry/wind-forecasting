# debug_ddp.py
import gluonts
import torch.multiprocessing as mp
import lightning.pytorch as pl
from gluonts.transform import ExpectedNumInstanceSampler
from gluonts.time_feature._base import second_of_minute, minute_of_hour, hour_of_day, day_of_year
# Import your custom modules
from wind_forecasting.preprocessing.pytorch_dataset import WindForecastingDatamodule
# You need to import your actual LightningModule (the model class)
# I'm assuming it's called 'InformerModel' or similar from your previous logs
from pytorch_transformer_ts.informer.lightning_module import InformerLightningModule
from gluonts.torch.distributions import LowRankMultivariateNormalOutput
mp.set_start_method('spawn', force=True)
def main():
    print("--- Starting Pure Lightning DDP Debug Script ---")

    # 1. Manually define the arguments your modules need
    #    Hard-code these paths and values for this test.
    #    These should match what your `create_pytorch_data_module` uses.
    
    # Example config (replace with your actual values)
    config = {
        'train_data_path': '/projects/ssc/ahenry/wind_forecasting/awaken_data/awaken_processed_normalized_train_ready_30s_per_turbine_ctx14_pred7_train.pkl',
        'val_data_path': '/projects/ssc/ahenry/wind_forecasting/awaken_data/awaken_processed_normalized_train_ready_30s_per_turbine_ctx14_pred7_val.pkl',
        'context_length': 14,
        'prediction_length': 7,
        'batch_size': 32,
        'num_workers': 0,
        # You'll need to figure out how to create/pass your `train_sampler`
        # and `time_features`. For this test, you might be able to simplify them.
        'train_sampler': ExpectedNumInstanceSampler(num_instances=1.0, min_past=14, min_future=7), 
        "val_sampler": None,
        'time_features': [second_of_minute, minute_of_hour, hour_of_day, day_of_year],
    }
    
    # 2. Instantiate your DataModule directly
    datamodule = WindForecastingDatamodule(
        train_data_path=config['train_data_path'],
        val_data_path=config['val_data_path'],
        train_sampler=config['train_sampler'],
        context_length=config['context_length'],
        prediction_length=config['prediction_length'],
        time_features=config['time_features'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        persistent_workers=False # Keep this false for the test
    )

    # 3. Instantiate your LightningModule (your model) directly
    #    You'll need to provide its required arguments.
    model = InformerLightningModule(
        # ... arguments for your model ...
        model_config={'freq': '30s', 'context_length': 14, 'prediction_length': 7, 'num_feat_dynamic_real': 6, 'num_feat_static_real': 1, 'num_feat_static_cat': 1, 'cardinality': [88], 'embedding_dimension': None, 'd_model': 128, 'n_heads': 8, 'num_encoder_layers': 2, 'num_decoder_layers': 2, 'activation': 'gelu', 'dropout': 0.15697382123348755, 'dim_feedforward': 512, 'attn': 'prob', 'factor': 5, 'distil': True, 'input_size': 2, 'distr_output': LowRankMultivariateNormalOutput(dim=2, rank=8), 'lags_seq': [0], 'scaling': 'False', 'num_parallel_samples': 100},
    lr=1e-4,weight_decay=1e-8,eta_min_fraction=0.1,gradient_clip_val=1000,warmup_steps=50000, steps_to_decay=50000, batch_size=128, base_batch_size_for_scheduler_steps=2048,base_limit_train_batches=None,num_batches_per_epoch=154724
    )

    # 4. Instantiate the Trainer
    #    Make sure the strategy and devices match your SLURM request.
    trainer = pl.Trainer(
        accelerator="auto",
        devices=2, # Or however many GPUs you requested
        strategy="auto",
        max_epochs=1,
        num_nodes=1,
        # For this test, let's also disable the sanity check to see if that helps
        num_sanity_val_steps=0,
        # We can also limit training to a few batches to make it run fast
        limit_train_batches=10, 
        limit_val_batches=10,
    )

    # 5. Run fit
    print("--- Calling trainer.fit() ---")
    trainer.fit(model, datamodule=datamodule)
    print("--- trainer.fit() completed ---")

if __name__ == "__main__":
    # If not using srun, you might need mp.spawn, but with srun,
    # just running the script should be enough as Lightning detects the env.
    main()
