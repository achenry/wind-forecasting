"""
Stage 2 Monitoring Integration

This module provides convenient imports for Stage 2 monitoring components.
It re-exports the Stage2Monitor callback from pytorch-transformer-ts and
provides utilities for Stage 2 monitoring configuration.
"""

try:
    # Re-export Stage2Monitor from pytorch-transformer-ts
    from pytorch_transformer_ts.tactis_2.stage2_monitor import Stage2Monitor
    from pytorch_transformer_ts.tactis_2.stage2_metrics import (
        compute_stage_aware_optimization_metric,
        validate_stage_aware_metric,
        compute_copula_learning_indicators,
        assess_copula_learning_health
    )
    
    __all__ = [
        'Stage2Monitor',
        'compute_stage_aware_optimization_metric',
        'validate_stage_aware_metric', 
        'compute_copula_learning_indicators',
        'assess_copula_learning_health'
    ]
    
except ImportError as e:
    import logging
    logging.warning(f"Could not import Stage 2 monitoring components: {e}")
    logging.warning("Stage 2 monitoring will not be available")
    
    # Provide dummy implementations to prevent import errors
    class DummyStage2Monitor:
        def __init__(self, *args, **kwargs):
            logging.warning("Using dummy Stage2Monitor - real monitoring not available")
    
    Stage2Monitor = DummyStage2Monitor
    
    __all__ = ['Stage2Monitor']