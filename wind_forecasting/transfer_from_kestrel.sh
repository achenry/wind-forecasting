for wt in 028 033 073; do
    scp -r 'aohe7145@login.rc.colorado.edu:/pl/active/paolab/awaken_data/kp.turbine.z02.b0/kp.turbine.z02.b0.*.000000.wt${wt}.nc' /Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data
done

scp -r 'aohe7145@login.rc.colorado.edu:/pl/active/paolab/awaken_data/kp.turbine.z02.b0/kp.turbine.z02.b0.*.000000.wt028.nc' /Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data/ && \
scp -r 'aohe7145@login.rc.colorado.edu:/pl/active/paolab/awaken_data/kp.turbine.z02.b0/kp.turbine.z02.b0.*.000000.wt033.nc' /Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data/ && \
scp -r 'aohe7145@login.rc.colorado.edu:/pl/active/paolab/awaken_data/kp.turbine.z02.b0/kp.turbine.z02.b0.*.000000.wt073.nc' /Users/ahenry/Documents/toolboxes/wind_forecasting/examples/data/