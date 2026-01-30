import shutil
import string
import subprocess
import sys
import warnings
from dataclasses import dataclass

import holoviews as hv
import panel as pn

hv.extension("bokeh")


@dataclass
class Theme:
    background_color: str
    text_color: str
    accent_color: str


# Intsantiate themes
dark_theme = Theme(
    background_color="#000000",
    text_color="#FFFFFF",
    accent_color="#303030",
)
dark_soft_theme = Theme(
    background_color="#121212",
    text_color="#E0E0E0",
    accent_color="#363636",
)

light_theme = Theme(
    background_color="#FFFFFF",
    text_color="#000000",
    accent_color="#C2C2C2",
)

SUPPORTED_THEMES = {
    "dark": dark_theme,
    "soft_dark": dark_soft_theme,
    "light": light_theme,
}

BOKEH_WIDGET_COLOR = "#006dae"

# Set default theme to dark mode
VIEW_THEME = dark_theme


def set_theme(theme_str: str) -> None:
    """
    Set package-wide viewing theme (holoviews/panel).

    Parameters
    ----------
    theme_str : str
        One of "dark", "soft_dark", "light".
    """
    global VIEW_THEME

    assert (
        theme_str in SUPPORTED_THEMES
    ), f"Unsupported theme: {theme_str}. Must be one of {SUPPORTED_THEMES}."

    # set holoviews and panel theme
    if theme_str in ["dark", "soft_dark"]:
        hv.renderer("bokeh").theme = "dark_minimal"
        pn.extension(theme="dark")
    else:
        hv.renderer("bokeh").theme = "light_minimal"
        pn.extension(theme="default")

    # Update the global theme
    VIEW_THEME = SUPPORTED_THEMES[theme_str]


DEFAULT_FONT = "Times"


def get_font_list():
    """Return sorted list of valid font family names (fc-list or fallback list)."""
    if not shutil.which("fc-list"):
        warnings.warn(
            "System fonts could not be listed, as fc-list is not available. Please install the fontconfig package."
        )
        valid_fonts = [
            "Times",
            "Helvetica",
            "Verdana",
            "Courier",
            "Arial",
            "Monospace",
        ]
    else:
        result = subprocess.run(
            ["fc-list", ":", "family", "|", "sort", "|", "uniq"],
            capture_output=True,
            text=True,
        )
        fonts = result.stdout.splitlines()
        valid_font_character_set = string.ascii_letters + string.whitespace
        valid_fonts = [
            font
            for font in fonts
            if all(char in valid_font_character_set for char in font)
        ]

    if DEFAULT_FONT not in valid_fonts:
        valid_fonts.append(DEFAULT_FONT)

    valid_fonts.sort()
    return valid_fonts


VALID_FONTS = get_font_list()
