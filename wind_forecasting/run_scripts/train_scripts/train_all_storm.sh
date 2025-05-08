sbatch train_model_storm.sh informer /user/taed7566/Forecasting/wind-forecasting/config/training/storm_configs/training_inputs_storm_awaken_pred60.yaml
sbatch train_model_storm.sh informer /user/taed7566/Forecasting/wind-forecasting/config/training/storm_configs/training_inputs_storm_awaken_pred300.yaml
sbatch train_model_storm.sh autoformer /user/taed7566/Forecasting/wind-forecasting/config/training/storm_configs/training_inputs_storm_awaken_pred60.yaml
sbatch train_model_storm.sh autoformer /user/taed7566/Forecasting/wind-forecasting/config/training/storm_configs/training_inputs_storm_awaken_pred300.yaml
sbatch train_model_storm.sh spacetimeformer /user/taed7566/Forecasting/wind-forecasting/config/training/storm_configs/training_inputs_storm_awaken_pred60.yaml
sbatch train_model_storm.sh spacetimeformer /user/taed7566/Forecasting/wind-forecasting/config/training/storm_configs/training_inputs_storm_awaken_pred300.yaml
# sbatch train_model_storm.sh tactis /user/taed7566/Forecasting/wind-forecasting/config/training/storm_configs/training_inputs_storm_awaken_pred60.yaml
# sbatch train_model_storm.sh tactis /user/taed7566/Forecasting/wind-forecasting/config/training/storm_configs/training_inputs_storm_awaken_pred300.yaml
