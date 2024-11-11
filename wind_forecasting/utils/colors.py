import sys
import os
from pathlib import Path

class Colors:
    # Regular colors
    BLUE = '\033[94m'      # Light/Bright Blue
    RED = '\033[91m'       # Light/Bright Red
    GREEN = '\033[92m'     # Light/Bright Green
    YELLOW = '\033[93m'    # Light/Bright Yellow
    CYAN = '\033[96m'      # Light/Bright Cyan
    MAGENTA = '\033[95m'   # Light/Bright Magenta
    
    # Bold colors
    BOLD_BLUE = '\033[1;34m'
    BOLD_GREEN = '\033[1;32m'
    BOLD_RED = '\033[1;31m'
    
    # Text style
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    # End color
    ENDC = '\033[0m'
    
    # Emojis
    ROCKET = '🚀'
    HOURGLASS = '⌛'
    CHECK = '✅'
    CROSS = '❌'
    FIRE = '🔥'
    CHART = '📊'
    WARNING = '⚠️'
    BRAIN = '🧠'
    SAVE = '💾'
    STAR = '⭐'
    
    @classmethod
    def disable_colors(cls):
        for attr in dir(cls):
            if not attr.startswith('__') and isinstance(getattr(cls, attr), str):
                setattr(cls, attr, '')

    @staticmethod
    def supports_color():
        """Check if term supports colors"""
        return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

if not Colors.supports_color():
    Colors.disable_colors()
