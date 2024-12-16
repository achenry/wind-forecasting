import gc
import os
import torch

from wind_forecasting.utils.colors import Colors

def cleanup_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
def cleanup_old_checkpoints(model_prefix, model_dir, dataset_name, keep_top_k=3):
    """Clean up old checkpoints, keeping only the top k best models."""
    try:
        print(f"{Colors.BLUE}CC start...{Colors.ENDC}")  # DEBUG
        
        checkpoints = [f for f in os.listdir(model_dir) 
                        if f.startswith(f'{model_prefix}-{dataset_name.lower()}-') 
                        and f.endswith('.ckpt')]
        
        print(f"Found {len(checkpoints)} checkpoints")  # Debug line
        
        if len(checkpoints) > keep_top_k:
            # Sort checkpoints by validation loss (extracted from filename)
            checkpoints.sort(key=lambda x: float(x.split('-loss')[1].split('.ckpt')[0]))
            
            # Remove all but the top k checkpoints
            for checkpoint in checkpoints[keep_top_k:]:
                checkpoint_path = os.path.join(model_dir, checkpoint)
                try:
                    print(f"clean {checkpoint}")  # DEBUG
                    os.remove(checkpoint_path)
                    print(f"{Colors.YELLOW}Removed old checkpoint: {checkpoint}{Colors.ENDC}")
                except OSError as e:
                    print(f"{Colors.RED}Error removing checkpoint {checkpoint}: {str(e)}{Colors.ENDC}")
            
            print(f"{Colors.GREEN}Kept top {keep_top_k} checkpoints for {dataset_name}{Colors.ENDC}")
        
        print(f"{Colors.BLUE}CCC{Colors.ENDC}")  # DEBUG
        
    except Exception as e:
        print(f"{Colors.RED}Error during checkpoint cleanup: {str(e)}{Colors.ENDC}")
        raise  # Re-raise the exception to be caught by the outer try-except