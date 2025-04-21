for wt in 028 033 073; do
    scp -r 'aohe7145@login.rc.colorado.edu:/pl/active/paolab/awaken_data/kp.turbine.z02.b0/kp.turbine.z02.b0.*.000000.wt${wt}.nc' /Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data
done

scp -r 'aohe7145@login.rc.colorado.edu:/pl/active/paolab/awaken_data/kp.turbine.z02.b0/kp.turbine.z02.b0.202203*.000000.wt*.nc' /Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data/ 
# && \
# scp -r 'aohe7145@login.rc.colorado.edu:/pl/active/paolab/awaken_data/kp.turbine.z02.b0/kp.turbine.z02.b0.*.000000.wt033.nc' /Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data/ && \
# scp -r 'aohe7145@login.rc.colorado.edu:/pl/active/paolab/awaken_data/kp.turbine.z02.b0/kp.turbine.z02.b0.*.000000.wt073.nc' /Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data/

scp -r 'ahenry@kestrel.hpc.nrel.gov:/projects/ssc/ahenry/wind_forecasting/awaken_data/kp.turbine.z02.b0/kp.turbine.z02.b0.2022030*.000000.*.nc' /Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data/

scp -r 'ahenry@kestrel.hpc.nrel.gov:/projects/ssc/ahenry/wind_forecasting/awaken_data/kp.turbine.zo2.b0.parquet' /Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data/

scp -r 'ahenry@kestrel.hpc.nrel.gov:/home/ahenry/toolboxes/wind_forecasting_env/wind-forecasting/wind_forecasting/run_scripts/environment.yaml' /Users/ahenry/Documents/toolboxes/wind_forecasting/


scp -r 'ahenry@kestrel.hpc.nrel.gov:/projects/ssc/ahenry/wind_forecasting/awaken_data/filled_data_normalization_consts.csv' /Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data
scp -r 'ahenry@kestrel.hpc.nrel.gov:/projects/ssc/ahenry/wind_forecasting/awaken_data/filled_data_normalized.parquet' /Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data

scp -r 'ahenry@kestrel.hpc.nrel.gov:/projects/ssc/ahenry/wind_forecasting/awaken_data/00_engie_scada_processed.parquet' /Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data

scp -r 'aohe7145@login.rc.colorado.edu:/scratch/alpine/aohe7145/flasc_data/preprocessed_flasc_data/*' /Users/ahenry/Downloads/preprocessed_flasc_data/

scp -r /Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data/filled_data_normalization_consts.csv aohe7145@login.rc.colorado.edu:/scratch/alpine/aohe7145/awaken_data
scp -r /Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data/filled_data_normalized.parquet aohe7145@login.rc.colorado.edu:/scratch/alpine/aohe7145/awaken_data 