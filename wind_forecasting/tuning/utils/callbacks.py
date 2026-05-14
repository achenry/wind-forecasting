"""
Tuning-specific callbacks for Optuna hyperparameter optimization.

This module contains callbacks that are specifically designed for use
during hyperparameter tuning with Optuna.
"""

import logging
import lightning.pytorch as pl
from optuna.trial import Trial

logger = logging.getLogger(__name__)


class SafePruningCallback(pl.Callback):
    """
    Wrapper class to safely pass the Optuna pruning callback to PyTorch Lightning.
    
    This callback wraps the Optuna PyTorchLightningPruningCallback to provide
    additional logging and error handling for trial pruning.
    """
    
    def __init__(self, trial: "Trial", monitor: str, stage2_start_epoch: int = None):
        """
        Initialize the SafePruningCallback.

        Args:
            trial: The Optuna trial object
            monitor: The metric to monitor for pruning decisions
            stage2_start_epoch: If set, skip reporting to Optuna before this epoch.
                Used in Phase 2 tuning to avoid reporting marginal-only metrics
                before the copula is active.
        """
        super().__init__()
        # Avoid circular import by importing here
        from optuna_integration import PyTorchLightningPruningCallback

        # Instantiate the actual Optuna callback internally
        self.optuna_pruning_callback = PyTorchLightningPruningCallback(trial, monitor)
        self.trial = trial
        self.monitor = monitor
        self.stage2_start_epoch = stage2_start_epoch

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        Called at the end of validation.

        Delegates to the wrapped Optuna callback and provides additional logging.
        For Phase 2 tuning, skips reporting before the stage transition so the
        pruner only evaluates copula quality.
        """
        # Phase 2: skip reporting before copula is active
        if self.stage2_start_epoch is not None and trainer.current_epoch < self.stage2_start_epoch:
            return

        try:
            # Call the corresponding method on the wrapped Optuna callback
            self.optuna_pruning_callback.on_validation_end(trainer, pl_module)
        except Exception as e:
            if "TrialPruned" in str(type(e).__name__):
                # Explicitly mark trial as pruned and log appropriately
                self.trial.set_user_attr('pruned_reason', str(e))
                logger.info(f"Trial {self.trial.number} pruned at epoch {trainer.current_epoch} (monitoring '{self.monitor}')")
                raise  # Re-raise to ensure trial state is properly set
            else:
                raise

    def check_pruned(self) -> None:
        """
        Check if the trial should be pruned.
        
        Delegates to the wrapped Optuna callback.
        """
        try:
            self.optuna_pruning_callback.check_pruned()
        except Exception as e:
            if "TrialPruned" in str(type(e).__name__):
                logger.info(f"Trial {self.trial.number} pruned (check_pruned): {str(e)}")
                raise
            else:
                raise