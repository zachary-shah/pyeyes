from typing import Dict, Optional, Union

import panel as pn

from .viewers import Viewer


def launch_viewers(
    viewer_dict: Union[Viewer, Dict[str, Viewer]],
    port: Optional[int] = 0,
    show=True,
    **kwargs,
):
    """
    Serve one or more viewers in a web app.

    Parameters
    ----------
    viewer_dict : Viewer or Dict[str, Viewer]
        Single viewer or dict of name -> viewer (uses .app if present).
    port : Optional[int]
        Port for server; 0 for auto.
    show : bool
        If True, open browser.
    **kwargs
        Passed to pn.serve.

    Returns
    -------
    panel.server.Server or None
        Server instance.
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

    return pn.serve(
        viewer_dict,
        port=port,
        show=show,
        **kwargs,
    )
