"""
Dictionary utilities for tuning operations.
"""
import collections.abc
from typing import Dict
from wind_forecasting.utils.path_utils import resolve_path


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """
    Flatten a nested dictionary into a single level dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Key prefix for nested items
        sep: Separator to use between parent and child keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)