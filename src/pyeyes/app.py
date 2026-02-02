from typing import Dict, Optional, Union

import panel as pn

from .viewers import ComparativeViewer, Viewer, spawn_comparative_viewer_detached


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


def launch_comparative_viewer(
    data,
    named_dims=None,
    view_dims=None,
    cat_dims=None,
    config_path=None,
    title="MRI Viewer",
    detached=False,
):
    """
    Launch a ComparativeViewer with one line.
    See `viewer.ComparativeViewer` for arguments.
    """
    if detached:
        spawn_comparative_viewer_detached(
            data=data,
            named_dims=named_dims,
            view_dims=view_dims,
            cat_dims=cat_dims,
            config_path=config_path,
            title=title,
        )
    else:
        ComparativeViewer(
            data=data,
            named_dims=named_dims,
            view_dims=view_dims,
            cat_dims=cat_dims,
            config_path=config_path,
        ).launch(title=title)
