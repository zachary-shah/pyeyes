import holoviews as hv
from bokeh import themes as bkthemes
import panel as pn
from dataclasses import dataclass

hv.extension("bokeh")

@dataclass
class Theme:
    background_color: str
    text_color: str
    text_font: str = "roboto" # TODO: allow flexibility in user parameterization

# Intsantiate themes
dark_theme = Theme(
    background_color="#000000",
    text_color="#FFFFFF",
)
dark_soft_theme = Theme(
    background_color="#121212",
    text_color="#E0E0E0",
)

light_theme = Theme(
    background_color="#FFFFFF",
    text_color="#000000",
)

SUPPORTED_THEMES = {
    "dark": dark_theme,
    "soft_dark": dark_soft_theme,
    "light": light_theme,
}

# Set default theme to dark mode
VIEW_THEME = dark_theme

def set_theme(theme_str: str) -> None:
    """
    Update internal viewing theme across the package.
    
    Parameters:
        theme (str): The theme to set. Must be one of SUPPORTED_THEMES.

    """
    global VIEW_THEME  

    assert theme_str in SUPPORTED_THEMES, f"Unsupported theme: {theme_str}. Must be one of {SUPPORTED_THEMES}."

    print("View theme set to:", theme_str)

    # set holoviews and panel theme
    if theme_str in ["dark", "soft_dark"]:
        hv.renderer('bokeh').theme = "dark_minimal"
        pn.extension(theme="dark")
    else:
        hv.renderer('bokeh').theme = "light_minimal"
        pn.extension(theme="default")

    # Update the global theme
    VIEW_THEME = SUPPORTED_THEMES[theme_str]
