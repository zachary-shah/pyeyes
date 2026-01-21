from .app import launch_viewers
from .themes import set_theme
from .viewers import ComparativeViewer, spawn_comparative_viewer_detached

from .prototypes.line import launch_1d_viewer

# Default Dark Theme
set_theme("dark")
