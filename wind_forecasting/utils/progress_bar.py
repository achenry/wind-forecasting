# INFO: Currently not used
class ProgressBar:
    def __init__(self, initial_error, target_error=0, width=50):
        self.initial_error = initial_error
        self.target_error = target_error
        self.width = width
        self.best_error = initial_error
        
    def update(self, current_error):
        self.best_error = min(self.best_error, current_error)
        # Calculate progress (0 to 1) where 1 means error reduced to target
        progress = 1 - (self.best_error - self.target_error) / (self.initial_error - self.target_error)
        progress = max(0, min(1, progress))  # Clamp between 0 and 1
        
        # Create the progress bar
        filled_length = int(self.width * progress)
        bar = '█' * filled_length + '░' * (self.width - filled_length)
        
        # Calculate percentage
        percent = progress * 100
        
        return f'{Colors.BOLD_BLUE}Progress: |{Colors.GREEN}{bar}{Colors.BOLD_BLUE}| {percent:6.2f}% {Colors.ENDC}'
