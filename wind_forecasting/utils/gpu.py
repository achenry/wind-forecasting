import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def update_cuda_config(config):
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        cuda_devices = os.environ["CUDA_VISIBLE_DEVICES"]
        logging.info(f"CUDA_VISIBLE_DEVICES is set to: '{cuda_devices}'")
        try:
            # Count the number of GPUs specified in CUDA_VISIBLE_DEVICES
            visible_gpus = [idx for idx in cuda_devices.split(',') if idx.strip()]
            num_visible_gpus = len(visible_gpus)
            
            if num_visible_gpus > 0:
                # Only override if the current configuration doesn't match
                if config["trainer"]["devices"] != num_visible_gpus:
                    logging.warning(f"Adjusting trainer.devices from {config['trainer']['devices']} to {num_visible_gpus} based on CUDA_VISIBLE_DEVICES")
                    config["trainer"]["devices"] = num_visible_gpus
                    
                    # If only one GPU is visible, use auto strategy instead of distributed
                    if num_visible_gpus == 1 and config["trainer"]["strategy"] != "auto":
                        logging.warning("Setting strategy to 'auto' since only one GPU is visible")
                        config["trainer"]["strategy"] = "auto"
                        
                # Log actual GPU mapping information
                if num_visible_gpus == 1:
                    try:
                        actual_gpu = int(visible_gpus[0])
                        device_id = 0  # With CUDA_VISIBLE_DEVICES, first visible GPU is always index 0
                        logging.info(f"Primary GPU is system device {actual_gpu}, mapped to CUDA index {device_id}")
                    except ValueError:
                        logging.warning(f"Could not parse GPU index from CUDA_VISIBLE_DEVICES: {visible_gpus[0]}")
            else:
                logging.warning("CUDA_VISIBLE_DEVICES is set but no valid GPU indices found")
        except Exception as e:
            logging.warning(f"Error parsing CUDA_VISIBLE_DEVICES: {e}")
    else:
        logging.warning("CUDA_VISIBLE_DEVICES is not set, using default GPU assignment")