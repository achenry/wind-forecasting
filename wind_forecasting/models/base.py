import os

import torch

from ..utils.colors import Colors

# INFO: Base class for all transformer models to inherit from
class BaseModel:
    def __init__(self):
        self.model = None
        self.trainer = None
        self.data_module = None
        
    def cleanup_memory(self):
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
    def cleanup_old_checkpoints(self, model_dir, dataset_name, keep_top_k=3):
        """Clean up old checkpoints, keeping only the top k best models."""
        try:
            print(f"{Colors.BLUE}CC start...{Colors.ENDC}")  # DEBUG
            
            checkpoints = [f for f in os.listdir(model_dir) 
                          if f.startswith(f'{self.model_prefix}-{dataset_name.lower()}-') 
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

    def train(self, dataset_class, data_path, model_params, train_params):
        raise NotImplementedError("Subclasses must implement train method")

    def test(self, dataset_class, data_path, model_params, train_params, checkpoint_path):
        raise NotImplementedError("Subclasses must implement test method")
