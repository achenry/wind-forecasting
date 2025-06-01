"""
Path resolution and dictionary utilities.
"""
import collections.abc
from pathlib import Path
from typing import Dict, Optional, Union


def resolve_path(base_path: str, path_input: Optional[Union[str, Path]]) -> Optional[str]:
    """
    Resolve a path relative to a base path or return absolute path.
    
    Args:
        base_path: Base path for relative path resolution
        path_input: Path to resolve (can be relative or absolute)
        
    Returns:
        Resolved absolute path as string, or None if path_input is None
    """
    if not path_input:
        return None
    # Convert potential Path object back to string if needed
    path_str = str(path_input)
    abs_path = Path(path_str)
    if not abs_path.is_absolute():
        abs_path = Path(base_path) / abs_path
    return str(abs_path.resolve())


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