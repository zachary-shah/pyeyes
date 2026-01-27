from .app import launch_viewers
from .prototypes.line import launch_1d_viewer
from .themes import set_theme
from .viewers import ComparativeViewer, spawn_comparative_viewer_detached

# Default Dark Theme
set_theme("dark")
