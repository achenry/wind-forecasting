import os
from pathlib import Path

class Config:
    def __init__(self, model_name):
        home = str(Path.home())
        self.save_dir = os.path.join(home, 'outputs', model_name) # TODO add folder with run id
        self.model_dir = os.path.join(self.save_dir, 'models')
        self.plot_dir = os.path.join(self.save_dir, 'plots')
        self.data_dir = os.path.join(self.save_dir, 'data')

    def create_directories(self):
        for directory in [self.save_dir, self.model_dir, self.plot_dir, self.data_dir]:
            os.makedirs(directory, exist_ok=True)
