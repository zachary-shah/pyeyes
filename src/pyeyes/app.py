from typing import Dict, Optional, Union

import panel as pn

from .viewers import Viewer


def launch_viewers(
    viewer_dict: Union[Viewer, Dict[str, Viewer]], port: Optional[int] = 0, **kwargs
):
    """
    Launches a web page hosting viewer(s).
    """

    if isinstance(viewer_dict, dict):
        viewer_dict_out = {}
        for k, v in viewer_dict.items():
            # check if app is attribute
            if hasattr(v, "app"):
                viewer_dict_out[k] = v.app
            else:
                viewer_dict_out[k] = v
        viewer_dict = viewer_dict_out

    pn.serve(
        viewer_dict,
        port=port,
        show=True,
        **kwargs,
    )
