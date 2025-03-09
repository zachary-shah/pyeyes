from typing import Dict, Optional, Union

import panel as pn

from .viewers import Viewer


def launch_viewers(
    viewer_dict: Union[Viewer, Dict[str, Viewer]], port: Optional[int] = None, **kwargs
):
    """
    Launches a web page hosting viewer(s).
    """

    if isinstance(viewer_dict, dict):
        viewer_dict = {k: v.app for k, v in viewer_dict.items()}

    pn.serve(
        viewer_dict,
        port=port,
        show=True,
        **kwargs,
    )
