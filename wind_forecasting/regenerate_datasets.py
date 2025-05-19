if __name__ == "__main__":
        if not mpi_exists and args.multiprocessor == "mpi":
        raise RuntimeError("MPI was requested (--multiprocessor mpi) but mpi4py failed to import. Check previous logs for import error details.")
    elif not mpi_exists:
         # If MPI wasn't requested, we might not need it here, but accessing MPI.COMM_WORLD directly is still problematic.
         # Depending on logic flow, this might need adjustment. For now, assume it's an error if MPI isn't available.
         # If MPI is optional, this block might need refinement based on how `comm` is used later.
         comm = None # Or handle appropriately if MPI is truly optional here
         rank = -1   # Assign a default rank if MPI is not used
         print("Warning: MPI not available, proceeding without it where possible.")
    else:
         comm = MPI.COMM_WORLD
         rank = comm.Get_rank()

    RUN_ONCE = (args.multiprocessor == "mpi" and rank == 0) or (args.multiprocessor != "mpi") or (args.multiprocessor is None)
    
    TRANSFORM_WIND = {"added_wm": args.added_wind_mag, "added_wd": args.added_wind_dir}
    # args.fig_dir = os.path.join(os.path.dirname(whoc_file), "..", "examples", "wind_forecasting")
    
    if RUN_ONCE:
        os.makedirs(args.save_dir, exist_ok=True)
     
    model_configs = []
    for mnf_path in args.model_config:
        with open(mnf_path, 'r') as file:
            model_configs.append(yaml.safe_load(file))
    
    
    prediction_timedelta = [pd.Timedelta(seconds=mncf["dataset"]["prediction_length"]) for mncf in model_configs]
    context_timedelta = [pd.Timedelta(seconds=mncf["dataset"]["context_length"]) for mncf in model_configs]
    measurements_timedelta = pd.Timedelta(seconds=args.simulation_timestep)
    
    # measurements_timedelta = pd.Timedelta(model_config["dataset"]["resample_freq"])
    
    controller_timedelta = max(pd.Timedelta(5, unit="s"), measurements_timedelta)

    with open(args.data_config, 'r') as file:
        data_config  = yaml.safe_load(file)
        
    if len(data_config["turbine_signature"]) == 1:
        tid2idx_mapping = {str(k): i for i, k in enumerate(data_config["turbine_mapping"][0].keys())}
    else:
        tid2idx_mapping = {str(k): i for i, k in enumerate(data_config["turbine_mapping"][0].values())} # if more than one file type was pulled from, all turbine ids will be transformed into common type
    
    turbine_signature = data_config["turbine_signature"][0] if len(data_config["turbine_signature"]) == 1 else "\\d+"
    
    id_var_selector = pl.exclude(
        f"^ws_horz_{turbine_signature}$", f"^ws_vert_{turbine_signature}$", 
                f"^nd_cos_{turbine_signature}$", f"^nd_sin_{turbine_signature}$",
                f"^loc_ws_horz_{turbine_signature}$", f"^loc_ws_vert_{turbine_signature}$",
                f"^sd_ws_horz_{turbine_signature}$", f"^sd_ws_vert_{turbine_signature}$")
    
    fmodel = FlorisModel(data_config["farm_input_path"])
    
    validation_save_dir = os.path.join(args.save_dir, "validation_results")
    
    logging.info("Creating datasets")
    
    # NOTE the dataset parts of the configs should be the same, other than context and prediction length
    base_model_config = model_configs[np.argsort([ctd + ptd for ctd, ptd in zip(context_timedelta, prediction_timedelta)])[-1]]
    data_module = DataModule(data_path=base_model_config["dataset"]["data_path"], 
                             normalization_consts_path=base_model_config["dataset"]["normalization_consts_path"],
                             normalized=False, 
                             n_splits=1, #model_config["dataset"]["n_splits"],
                             continuity_groups=None, train_split=(1.0 - base_model_config["dataset"]["val_split"] - base_model_config["dataset"]["test_split"]),
                             val_split=base_model_config["dataset"]["val_split"], test_split=base_model_config["dataset"]["test_split"],
                             prediction_length=base_model_config["dataset"]["prediction_length"], context_length=base_model_config["dataset"]["context_length"],
                             target_prefixes=["ws_horz", "ws_vert"], feat_dynamic_real_prefixes=["nd_cos", "nd_sin"],
                             freq=f"{int(measurements_timedelta.total_seconds())}s", 
                             target_suffixes=base_model_config["dataset"]["target_turbine_ids"],
                             per_turbine_target=False, as_lazyframe=False, dtype=pl.Float32,
                             verbose=True)
    
    if RUN_ONCE and not os.path.exists(data_module.train_ready_data_path):
        data_module.generate_datasets()
        logging.info("Reloading test datasets.")
        reload = True
    else:
        logging.info("Reading saved test datasets.")
        reload = False
    
    # reload = True
    data_module.generate_splits(save=True, reload=reload, splits=["test"])