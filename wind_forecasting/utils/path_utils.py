"""
General path resolution utilities used across the wind forecasting framework.
"""
from pathlib import Path
from typing import Optional, Union


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