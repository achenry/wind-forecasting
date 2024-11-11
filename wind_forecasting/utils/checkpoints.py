from lightning.pytorch.callbacks import ModelCheckpoint

# INFO: Custom model checkpoint callback to stop training when validation loss is below a threshold (50.0)
class ThresholdModelCheckpoint(ModelCheckpoint):
    def __init__(self, loss_threshold=50.0, **kwargs):
        super().__init__(**kwargs)
        self.loss_threshold = loss_threshold

    def _should_save_on_train_epoch_end(self, trainer):
        return self._save_if_below_threshold(trainer)
        
    def _save_if_below_threshold(self, trainer):
        """Only save if validation loss is below threshold"""
        current_loss = trainer.callback_metrics.get('val/loss')
        if current_loss is not None and current_loss < self.loss_threshold:
            return True
        return False