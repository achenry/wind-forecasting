#!/usr/bin/env python3
"""
Stage-Aware Pruner for TACTiS-2 Two-Stage Training

This module provides a custom Optuna pruner that understands the dynamics
of TACTiS-2 two-stage training, where:
- Stage 1 (epochs 0-19): Total NLL optimization  
- Stage 2 (epochs 20+): Copula loss optimization with potential NLL stagnation

The pruner avoids premature trial termination during the Stage 1->Stage 2
transition period and uses appropriate metrics for each stage.
"""

import optuna
import logging
from typing import Dict, Any, Optional, Set, List
import numpy as np

logger = logging.getLogger(__name__)


class VirtualStudyView:
    """
    A study-like interface that wraps a filtered list of trials.
    
    This class provides the necessary interface for Optuna base pruners
    to operate on a subset of trials (e.g., only same-stage trials).
    It mimics the essential properties of optuna.Study while working
    with a filtered trial list.
    """
    
    def __init__(self, trials: List[optuna.Trial], original_study: optuna.Study):
        """
        Parameters:
        -----------
        trials : List[optuna.Trial]
            Filtered list of trials for this virtual study
        original_study : optuna.Study
            The original study (needed for some study properties)
        """
        self.trials = trials
        self._original_study = original_study
        
        # Compute best trial from filtered trials
        completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if completed_trials:
            # For minimization (default), best = minimum value
            direction = getattr(original_study, 'direction', optuna.study.StudyDirection.MINIMIZE)
            if direction == optuna.study.StudyDirection.MINIMIZE:
                self.best_trial = min(completed_trials, key=lambda t: t.value)
            else:
                self.best_trial = max(completed_trials, key=lambda t: t.value)
            self.best_value = self.best_trial.value
        else:
            self.best_trial = None
            self.best_value = None
    
    @property
    def direction(self):
        """Return the optimization direction from the original study."""
        return getattr(self._original_study, 'direction', optuna.study.StudyDirection.MINIMIZE)
    
    @property
    def directions(self):
        """Return the optimization directions from the original study.""" 
        return getattr(self._original_study, 'directions', [optuna.study.StudyDirection.MINIMIZE])
    
    @property
    def study_name(self):
        """Return the study name from the original study."""
        return getattr(self._original_study, 'study_name', 'virtual_study')
    
    def get_trials(self, deepcopy: bool = True, states=None):
        """
        Return trials matching the specified states.
        
        Parameters:
        -----------
        deepcopy : bool
            Whether to return deep copies (ignored in this implementation)
        states : Optional[Container[optuna.trial.TrialState]]
            States to filter by
            
        Returns:
        --------
        List[optuna.Trial] : Filtered trials
        """
        if states is None:
            return self.trials
        
        if not isinstance(states, (list, tuple, set)):
            states = [states]
        
        return [t for t in self.trials if t.state in states]


class StageAwarePruner:
    """
    Custom pruner for TACTiS-2 that understands two-stage training dynamics.
    
    This pruner wraps an existing pruner (e.g., HyperbandPruner) and adds
    stage-aware logic to prevent pruning during critical transition periods
    and ensure appropriate metric evaluation for each stage.
    
    Key features:
    - Prevents pruning during Stage 1->Stage 2 transition (epochs 18-22)
    - Uses stage-appropriate metrics for pruning decisions
    - Allows Stage 2 settling time for copula learning
    - Maintains compatibility with existing Optuna pruning strategies
    """
    
    def __init__(self,
                 base_pruner: optuna.pruners.BasePruner,
                 stage2_start_epoch: int = 20,
                 transition_buffer_epochs: int = 2,
                 stage2_min_epochs_before_pruning: int = 5,
                 enable_stage2_pruning: bool = True):
        """
        Parameters:
        -----------
        base_pruner : optuna.pruners.BasePruner
            The underlying pruner to use (e.g., HyperbandPruner, MedianPruner)
        stage2_start_epoch : int
            Epoch when Stage 2 begins
        transition_buffer_epochs : int  
            Number of epochs before/after transition to avoid pruning
        stage2_min_epochs_before_pruning : int
            Minimum Stage 2 epochs before allowing pruning
        enable_stage2_pruning : bool
            Whether to enable pruning in Stage 2 at all
        """
        self.base_pruner = base_pruner
        self.stage2_start_epoch = stage2_start_epoch
        self.transition_buffer_epochs = transition_buffer_epochs
        self.stage2_min_epochs_before_pruning = stage2_min_epochs_before_pruning
        self.enable_stage2_pruning = enable_stage2_pruning
        
        # Calculate transition period bounds
        self.transition_start = stage2_start_epoch - transition_buffer_epochs
        self.transition_end = stage2_start_epoch + transition_buffer_epochs
        
        # Track trials that have entered Stage 2
        self.stage2_trials: Set[int] = set()
        
        logger.info(f"StageAwarePruner initialized:")
        logger.info(f"  Base pruner: {type(base_pruner).__name__}")
        logger.info(f"  Stage 2 start: epoch {stage2_start_epoch}")
        logger.info(f"  Transition period: epochs {self.transition_start}-{self.transition_end}")
        logger.info(f"  Stage 2 min epochs before pruning: {stage2_min_epochs_before_pruning}")
        logger.info(f"  Stage 2 pruning enabled: {enable_stage2_pruning}")
    
    def prune(self, study: optuna.Study, trial: optuna.Trial) -> bool:
        """
        Decide whether to prune a trial based on stage-aware logic.
        
        This is the method called by Optuna's pruning system.
        
        Parameters:
        -----------
        study : optuna.Study
            The Optuna study
        trial : optuna.Trial
            The trial to evaluate for pruning
            
        Returns:
        --------
        bool : True if trial should be pruned, False otherwise
        """
        try:
            # Get trial's intermediate values (reported metrics by epoch)
            intermediate_values = trial.intermediate_values
            
            if not intermediate_values:
                logger.debug(f"Trial {trial.number}: No intermediate values, not pruning")
                return False
            
            # Get the latest epoch
            latest_epoch = max(intermediate_values.keys())
            
            # Check if we're in the transition period
            if self._is_transition_period(latest_epoch):
                logger.info(f"Trial {trial.number}: In transition period (epoch {latest_epoch}), "
                           f"skipping pruning")
                return False
            
            # Check if we're in Stage 2
            if latest_epoch >= self.stage2_start_epoch:
                # Mark trial as having entered Stage 2
                self.stage2_trials.add(trial.number)
                
                # Check if Stage 2 pruning is disabled
                if not self.enable_stage2_pruning:
                    logger.debug(f"Trial {trial.number}: Stage 2 pruning disabled, not pruning")
                    return False
                
                # Check if we need more Stage 2 epochs before pruning
                stage2_epochs = latest_epoch - self.stage2_start_epoch + 1
                if stage2_epochs < self.stage2_min_epochs_before_pruning:
                    logger.debug(f"Trial {trial.number}: Only {stage2_epochs} Stage 2 epochs, "
                               f"need {self.stage2_min_epochs_before_pruning} before pruning")
                    return False
                
                # Additional Stage 2 checks
                if self._should_skip_stage2_pruning(trial, latest_epoch):
                    return False
            
            # Use stage-isolated pruning decision
            base_decision = self._stage_isolated_prune(study, trial, latest_epoch)
            
            if base_decision:
                stage = "Stage 1" if latest_epoch < self.stage2_start_epoch else "Stage 2"
                logger.info(f"Trial {trial.number}: Stage-isolated pruner recommends pruning at epoch "
                           f"{latest_epoch} ({stage})")
            
            return base_decision
            
        except Exception as e:
            logger.error(f"Error in stage-aware pruning for trial {trial.number}: {e}")
            # Fallback: don't prune if there's an error
            return False
    
    def _is_transition_period(self, epoch: int) -> bool:
        """Check if epoch is in the Stage 1->Stage 2 transition period."""
        return self.transition_start <= epoch <= self.transition_end
    
    def _should_skip_stage2_pruning(self, trial: optuna.Trial, latest_epoch: int) -> bool:
        """
        Additional checks for whether to skip pruning in Stage 2.
        
        Parameters:
        -----------
        trial : optuna.Trial
            The trial being evaluated
        latest_epoch : int
            Latest epoch with intermediate values
            
        Returns:
        --------
        bool : True if should skip pruning, False otherwise
        """
        try:
            intermediate_values = trial.intermediate_values
            
            # Get recent Stage 2 values for trend analysis
            stage2_epochs = [e for e in intermediate_values.keys() if e >= self.stage2_start_epoch]
            
            if len(stage2_epochs) < 3:
                # Not enough Stage 2 data for trend analysis
                return True
            
            # Get recent values (last few Stage 2 epochs)
            recent_epochs = sorted(stage2_epochs)[-min(5, len(stage2_epochs)):]
            recent_values = [intermediate_values[e] for e in recent_epochs]
            
            # Check for improvement trend in Stage 2
            if self._has_improvement_trend(recent_values):
                logger.debug(f"Trial {trial.number}: Detected improvement trend in Stage 2, "
                           f"skipping pruning")
                return True
            
            # Check for unusual Stage 2 dynamics (e.g., sudden changes)
            if self._has_unusual_dynamics(recent_values):
                logger.debug(f"Trial {trial.number}: Unusual dynamics in Stage 2, "
                           f"allowing more time")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error analyzing Stage 2 dynamics for trial {trial.number}: {e}")
            # Conservative: skip pruning if analysis fails
            return True
    
    def _has_improvement_trend(self, values: list) -> bool:
        """
        Check if there's an improvement trend in recent values.
        
        For minimization problems, improvement = decreasing values.
        """
        if len(values) < 3:
            return False
        
        try:
            # Simple linear trend analysis
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            
            # Negative slope = decreasing = improving (for minimization)
            improvement_threshold = -1e-6  # Small but meaningful improvement
            
            return slope < improvement_threshold
            
        except Exception:
            return False
    
    def _has_unusual_dynamics(self, values: list) -> bool:
        """
        Check for unusual dynamics that might indicate Stage 2 adaptation.
        
        This could include:
        - High volatility (copula learning instability)
        - Sudden changes (stage transition effects)
        """
        if len(values) < 3:
            return False
        
        try:
            # Calculate coefficient of variation
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if mean_val == 0:
                return False
            
            cv = std_val / abs(mean_val)
            
            # High volatility might indicate active learning
            high_volatility_threshold = 0.1
            
            if cv > high_volatility_threshold:
                return True
            
            # Check for sudden changes (potential stage transition effects)
            changes = np.abs(np.diff(values))
            mean_change = np.mean(changes)
            max_change = np.max(changes)
            
            # If max change is much larger than average, might be transitional
            if mean_change > 0 and max_change > 3 * mean_change:
                return True
            
            return False
            
        except Exception:
            return False

    def _get_trial_stage(self, trial: optuna.Trial) -> int:
        """
        Determine which stage a trial reached based on its intermediate values.
        
        Parameters:
        -----------
        trial : optuna.Trial
            The trial to analyze
            
        Returns:
        --------
        int : Stage number (1 or 2)
        """
        if not trial.intermediate_values:
            return 1  # Default to Stage 1 if no intermediate values
        
        max_epoch = max(trial.intermediate_values.keys())
        return 2 if max_epoch >= self.stage2_start_epoch else 1
    
    def _filter_trials_by_stage(self, study: optuna.Study, target_stage: int) -> List[optuna.Trial]:
        """
        Filter study trials to only include those that reached the target stage.
        
        Parameters:
        -----------
        study : optuna.Study
            The original study
        target_stage : int
            Stage to filter for (1 or 2)
            
        Returns:
        --------
        List[optuna.Trial] : Trials that reached the target stage
        """
        filtered_trials = []
        
        for trial in study.trials:
            # Only consider completed trials for comparison
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            
            trial_stage = self._get_trial_stage(trial)
            if trial_stage == target_stage:
                filtered_trials.append(trial)
        
        logger.debug(f"Found {len(filtered_trials)} trials in Stage {target_stage} "
                   f"out of {len(study.trials)} total trials")
        
        return filtered_trials
    
    def _stage_isolated_prune(self, study: optuna.Study, trial: optuna.Trial, latest_epoch: int) -> bool:
        """
        Perform stage-isolated pruning by comparing only with same-stage trials.
        
        This method creates a virtual study containing only trials from the same stage
        as the current trial, then uses the base pruner to make a decision based on
        that filtered data. This prevents comparing Stage 1 metrics with Stage 2 metrics.
        
        Parameters:
        -----------
        study : optuna.Study
            The original study
        trial : optuna.Trial
            The trial being evaluated for pruning
        latest_epoch : int
            The latest epoch with intermediate values
            
        Returns:
        --------
        bool : True if trial should be pruned, False otherwise
        """
        try:
            # Determine current stage
            current_stage = 2 if latest_epoch >= self.stage2_start_epoch else 1
            
            # Filter trials to only include same-stage trials
            same_stage_trials = self._filter_trials_by_stage(study, current_stage)
            
            # Apply fallback strategies for insufficient data
            min_trials_for_pruning = 5  # Minimum trials needed for meaningful comparison
            
            if len(same_stage_trials) < min_trials_for_pruning:
                logger.debug(f"Trial {trial.number}: Only {len(same_stage_trials)} same-stage trials, "
                           f"need {min_trials_for_pruning} for reliable pruning. Using fallback strategy.")
                
                return self._fallback_pruning_decision(study, trial, current_stage, same_stage_trials)
            
            # Create virtual study with same-stage trials
            virtual_study = VirtualStudyView(same_stage_trials, study)
            
            # Use base pruner on stage-isolated data
            base_decision = self.base_pruner.prune(virtual_study, trial)
            
            logger.debug(f"Trial {trial.number}: Stage {current_stage} isolated pruning decision: {base_decision} "
                       f"(based on {len(same_stage_trials)} same-stage trials)")
            
            return base_decision
            
        except Exception as e:
            logger.error(f"Error in stage-isolated pruning for trial {trial.number}: {e}")
            # Conservative fallback: don't prune if there's an error
            return False
    
    def _fallback_pruning_decision(self, study: optuna.Study, trial: optuna.Trial, 
                                 current_stage: int, same_stage_trials: List[optuna.Trial]) -> bool:
        """
        Make a fallback pruning decision when insufficient same-stage data is available.
        
        Fallback strategies:
        1. If < 3 same-stage trials: Don't prune (too little data)
        2. If Stage 2 with no Stage 2 history: Use Stage 1 data with penalty
        3. If base pruner fails: Conservative approach (don't prune)
        
        Parameters:
        -----------
        study : optuna.Study
            The original study
        trial : optuna.Trial
            The trial being evaluated
        current_stage : int
            Current stage of the trial
        same_stage_trials : List[optuna.Trial]
            Available same-stage trials
            
        Returns:
        --------
        bool : True if trial should be pruned, False otherwise
        """
        try:
            # Strategy 1: If very few same-stage trials, be conservative
            if len(same_stage_trials) < 3:
                logger.debug(f"Trial {trial.number}: Too few same-stage trials ({len(same_stage_trials)}), "
                           f"not pruning (conservative)")
                return False
            
            # Strategy 2: For Stage 2 trials with limited Stage 2 history,
            # use Stage 1 data with a conservative penalty
            if current_stage == 2:
                stage1_trials = self._filter_trials_by_stage(study, 1)
                
                if len(stage1_trials) >= 5:
                    logger.debug(f"Trial {trial.number}: Using Stage 1 trials ({len(stage1_trials)}) "
                               f"with penalty for Stage 2 pruning decision")
                    
                    # Create virtual study with Stage 1 trials
                    virtual_study = VirtualStudyView(stage1_trials, study)
                    
                    # Apply conservative penalty: only prune if base pruner is very confident
                    # We do this by temporarily modifying the trial's current value to be more conservative
                    original_value = trial.intermediate_values.get(max(trial.intermediate_values.keys()))
                    
                    if original_value is not None:
                        # For Stage 2, assume performance should be better than Stage 1 average
                        # Apply a penalty that makes the trial look slightly worse
                        stage1_values = [t.value for t in stage1_trials if t.value is not None]
                        if stage1_values:
                            stage1_mean = np.mean(stage1_values)
                            penalty_factor = 1.1  # Make trial look 10% worse for conservative pruning
                            
                            # Temporarily modify intermediate values for pruning decision
                            penalized_value = original_value * penalty_factor
                            trial.intermediate_values[max(trial.intermediate_values.keys())] = penalized_value
                            
                            try:
                                base_decision = self.base_pruner.prune(virtual_study, trial)
                                logger.debug(f"Trial {trial.number}: Stage 1 fallback decision with penalty: {base_decision}")
                                return base_decision
                            finally:
                                # Restore original value
                                trial.intermediate_values[max(trial.intermediate_values.keys())] = original_value
            
            # Strategy 3: If all else fails, be conservative
            logger.debug(f"Trial {trial.number}: All fallback strategies exhausted, not pruning (conservative)")
            return False
            
        except Exception as e:
            logger.error(f"Error in fallback pruning decision for trial {trial.number}: {e}")
            # Ultimate fallback: don't prune
            return False


class TACTiSHyperbandPruner(StageAwarePruner):
    """
    Convenient wrapper for HyperbandPruner with TACTiS-2 stage awareness.
    
    This is a pre-configured StageAwarePruner using HyperbandPruner as the
    base pruner, optimized for TACTiS-2 training dynamics.
    """
    
    def __init__(self,
                 min_resource: int = 10,
                 max_resource: int = 100,
                 reduction_factor: int = 3,
                 stage2_start_epoch: int = 20,
                 **kwargs):
        """
        Parameters:
        -----------
        min_resource : int
            Minimum epochs before pruning (should allow Stage 1 to complete)
        max_resource : int
            Maximum epochs (total training length)
        reduction_factor : int
            Hyperband reduction factor
        stage2_start_epoch : int
            When Stage 2 begins
        **kwargs
            Additional arguments for StageAwarePruner
        """
        base_pruner = optuna.pruners.HyperbandPruner(
            min_resource=min_resource,
            max_resource=max_resource,
            reduction_factor=reduction_factor
        )
        
        super().__init__(
            base_pruner=base_pruner,
            stage2_start_epoch=stage2_start_epoch,
            **kwargs
        )
        
        logger.info(f"TACTiSHyperbandPruner initialized with min_resource={min_resource}, "
                   f"max_resource={max_resource}, reduction_factor={reduction_factor}")


class TACTiSMedianPruner(StageAwarePruner):
    """
    Convenient wrapper for MedianPruner with TACTiS-2 stage awareness.
    
    This is useful for studies with many trials where median-based pruning
    is more appropriate than Hyperband.
    """
    
    def __init__(self,
                 n_startup_trials: int = 5,
                 n_warmup_steps: int = 0,
                 interval_steps: int = 1,
                 stage2_start_epoch: int = 20,
                 **kwargs):
        """
        Parameters:
        -----------
        n_startup_trials : int
            Number of trials before pruning starts
        n_warmup_steps : int
            Number of steps before pruning starts
        interval_steps : int
            Interval for pruning
        stage2_start_epoch : int
            When Stage 2 begins
        **kwargs
            Additional arguments for StageAwarePruner
        """
        base_pruner = optuna.pruners.MedianPruner(
            n_startup_trials=n_startup_trials,
            n_warmup_steps=n_warmup_steps,
            interval_steps=interval_steps
        )
        
        super().__init__(
            base_pruner=base_pruner,
            stage2_start_epoch=stage2_start_epoch,
            **kwargs
        )
        
        logger.info(f"TACTiSMedianPruner initialized with n_startup_trials={n_startup_trials}, "
                   f"n_warmup_steps={n_warmup_steps}, interval_steps={interval_steps}")


def create_tactis_pruner(pruner_type: str = "hyperband", **kwargs) -> StageAwarePruner:
    """
    Factory function to create stage-aware pruners for TACTiS-2.
    
    Parameters:
    -----------
    pruner_type : str
        Type of pruner: "hyperband", "median", or "none"
    **kwargs
        Additional arguments for the pruner
        
    Returns:
    --------
    StageAwarePruner : Configured pruner for TACTiS-2
    """
    if pruner_type.lower() == "hyperband":
        return TACTiSHyperbandPruner(**kwargs)
    elif pruner_type.lower() == "median":
        return TACTiSMedianPruner(**kwargs)
    elif pruner_type.lower() == "none":
        # Create a NopPruner wrapped in stage-aware logic
        base_pruner = optuna.pruners.NopPruner()
        return StageAwarePruner(base_pruner=base_pruner, **kwargs)
    else:
        raise ValueError(f"Unknown pruner type: {pruner_type}. "
                        f"Supported types: 'hyperband', 'median', 'none'")