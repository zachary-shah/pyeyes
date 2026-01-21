import holoviews as hv
import numpy as np
import panel as pn
import param

hv.extension("bokeh")
pn.extension()

from ..utils import tonp


def launch_1d_viewer(
    array: np.ndarray,
    dims: list[str],
    plot_dim: str | None = None,
    *,
    x: np.ndarray | None = None,
    c: np.ndarray | None = None,
    title: str | None = None,
    show: bool = True,
) -> None:
    """
    Launch an interactive 1‑D viewer for an n‑D NumPy array.

    Parameters
    ----------
    array      : np.ndarray
        Data to visualise (any dimensionality, real or complex).
    dims       : list[str]
        Human‑readable name for every axis, same length as ``array.shape``.
    plot_dim   : str | None
        Name of the dimension to plot along (x‑axis). If ``None`` the
        last dimension is used.
    x          : np.ndarray | None, optional
        X values for the plot. If None, defaults to 0..L-1.
    c          : np.ndarray | None, optional
        Color values for the plot. If None, defaults to an array of ones.
    title      : str | None, optional
        Window title shown in the browser tab.
    show       : bool, default=True
        If True, a browser tab is opened automatically.
    """
    # inputs to numpy
    array = tonp(array)
    x = tonp(x)
    c = tonp(c)

    # ------------------------------------------------------------------ #
    #  Sanity checks for supplied x and c                                #
    # ------------------------------------------------------------------ #
    if x is not None and np.iscomplexobj(x):
        raise ValueError("`x` must be real‑valued (got complex array).")
    # c may be complex – that is handled later

    # ------------------------------------------------------------------ #
    #  Remove singleton dimensions across all input arrays               #
    # ------------------------------------------------------------------ #
    singleton_axes = tuple(i for i, s in enumerate(array.shape) if s == 1)
    if singleton_axes:
        # squeeze data array
        array = np.squeeze(array, axis=singleton_axes)
        # drop corresponding dimension labels
        dims = [d for i, d in enumerate(dims) if i not in singleton_axes]

        # helper to squeeze x or c if shape matches original
        def _squeeze_if_needed(arr):
            if arr is None:
                return arr
            arr = np.asarray(arr)
            if arr.ndim == array.ndim + len(singleton_axes):  # was full shaped earlier
                return np.squeeze(arr, axis=singleton_axes)
            return arr

        x = _squeeze_if_needed(x)
        c = _squeeze_if_needed(c)

    if len(dims) != array.ndim:
        raise ValueError("len(dims) must match array.ndim")

    if plot_dim is None:
        plot_dim = dims[-1]
    if plot_dim not in dims:
        raise ValueError(f"{plot_dim!r} not found in dims")

    L = array.shape[dims.index(plot_dim)]

    def _broadcast(arr, name):
        if arr is None:
            return None
        arr = np.asarray(arr)
        if arr.ndim == 1:
            if arr.size != L:
                raise ValueError(f"{name} length must match size of {plot_dim}")
            # expand to full ndim
            new_shape = [1] * array.ndim
            new_shape[dims.index(plot_dim)] = L
            arr = arr.reshape(new_shape)
            arr = np.broadcast_to(arr, array.shape)
        elif arr.shape != array.shape:
            raise ValueError(f"{name} must be 1‑D of length {L} or same shape as array")
        return arr

    x_full = _broadcast(x, "x")
    c_full = _broadcast(c, "c")
    if c_full is None:
        c_full = np.ones_like(array, dtype=float)
    if x_full is None:
        # build default x: 0..L-1
        x_base = np.arange(L)
        new_shape = [1] * array.ndim
        new_shape[dims.index(plot_dim)] = L
        x_full = np.broadcast_to(x_base.reshape(new_shape), array.shape)

    # ------------------------------------------------------------------ #
    #  Parameterised viewer                                               #
    # ------------------------------------------------------------------ #
    class _Viewer(param.Parameterized):
        # Complex‑view selector only if the data is complex
        if np.iscomplexobj(array):
            view_kind = param.Selector(
                default="abs",
                objects=["abs", "real", "imag", "angle"],
                doc="Representation of complex data",
            )
        colormap = param.Selector(
            default="viridis", objects=["viridis", "plasma", "magma", "cividis", "gray"]
        )

        # choose line (default) or scatter
        plot_type = param.Selector(
            default="line", objects=["line", "scatter"], doc="Render as line or scatter"
        )

        # How to view colour array if it is complex
        if np.iscomplexobj(c):
            c_view_kind = param.Selector(
                default="abs",
                objects=["abs", "real", "imag", "angle"],
                doc="Representation of complex colour array",
            )

        # toggle log normalisation of colour values
        log_norm = param.Boolean(default=False, doc="Log‑scale colour")

        # --- manual axis‑limit controls ----------------------------------- #
        manual_limits = param.Boolean(
            default=False, doc="Enable manual xmin/xmax/ymin/ymax"
        )

        xmin = param.Number(default=0.0)
        xmax = param.Number(default=1.0)
        ymin = param.Number(default=0.0)
        ymax = param.Number(default=1.0)

        autoscale = param.Action(
            lambda self: None, doc="Autoscale limits to current data slice"
        )

        _idx_params = {}  # will be filled dynamically
        _dim_sizes = dict(zip(dims, array.shape))

        # Range slider for the plotting dimension
        crop = param.Range(
            default=(0, _dim_sizes[plot_dim] - 1),
            bounds=(0, _dim_sizes[plot_dim] - 1),
            doc="Subset of samples shown on the x‑axis",
        )

        # Add an Int parameter for every NON‑plotting axis
        for d in dims:
            if d == plot_dim:
                continue
            _idx_params[d] = param.Integer(
                default=_dim_sizes[d] // 2,
                bounds=(0, _dim_sizes[d] - 1),
                step=1,
                doc=f"Index along {d}",
            )
        locals().update(_idx_params)  # inject parameters into the class

        # ------------------------------------------------------------------ #
        #  Build curve on demand                                             #
        # ------------------------------------------------------------------ #
        _deps = (
            [
                "crop",
                "colormap",
                "log_norm",
                "manual_limits",
                "xmin",
                "xmax",
                "ymin",
                "ymax",
                "plot_type",
            ]
            + (["view_kind"] if np.iscomplexobj(array) else [])
            + (["c_view_kind"] if np.iscomplexobj(c) else [])
            + [d for d in dims if d != plot_dim]
        )

        @param.depends(*_deps)
        def _curve(self):
            # Build tuple of indices in original axis order
            sel = []
            for d in dims:
                if d == plot_dim:
                    l, r = self.crop
                    sel.append(slice(int(l), int(r) + 1))
                else:
                    sel.append(getattr(self, d))
            data_1d = array[tuple(sel)]

            # Convert complex if necessary
            if np.iscomplexobj(data_1d):
                match getattr(self, "view_kind", "abs"):
                    case "abs":
                        data_1d = np.abs(data_1d)
                    case "real":
                        data_1d = np.real(data_1d)
                    case "imag":
                        data_1d = np.imag(data_1d)
                    case "angle":
                        data_1d = np.angle(data_1d)
            x_1d = x_full[tuple(sel)]
            c_1d = c_full[tuple(sel)]

            # Convert colour array if complex
            if np.iscomplexobj(c_1d):
                match getattr(self, "c_view_kind", "abs"):
                    case "abs":
                        c_1d = np.abs(c_1d)
                    case "real":
                        c_1d = np.real(c_1d)
                    case "imag":
                        c_1d = np.imag(c_1d)
                    case "angle":
                        c_1d = np.angle(c_1d)
            else:
                c_1d = np.asarray(c_1d, dtype=float)

            # Optional logarithmic normalisation
            if self.log_norm:
                c_1d = np.log(np.abs(c_1d) + 1e-5)

            # axis limits
            if self.manual_limits:
                xlim = (self.xmin, self.xmax)
                ylim = (self.ymin, self.ymax)
            else:
                xlim = (x_1d.min(), x_1d.max())
                ylim = (data_1d.min(), data_1d.max())

            if self.plot_type == "line":
                element = hv.Curve(
                    (x_1d.ravel(), data_1d.ravel()), kdims=[plot_dim], vdims=["value"]
                ).opts(color="black", xlim=xlim, ylim=ylim)
            else:  # scatter
                element = hv.Scatter(
                    (x_1d.ravel(), data_1d.ravel(), c_1d.ravel()),
                    kdims=[plot_dim],
                    vdims=["value", "intensity"],
                ).opts(
                    cmap=self.colormap,
                    color="intensity",
                    line_color="intensity",
                    size=4,
                    xlim=xlim,
                    ylim=ylim,
                )
            return element

        # ------------------------------------------------------------------ #
        #  Autoscale callback                                                #
        # ------------------------------------------------------------------ #
        @param.depends("autoscale", watch=True)
        def _auto_limits(self):
            # compute current slice limits (same selection logic)
            sel = []
            for d in dims:
                if d == plot_dim:
                    l, r = self.crop
                    sel.append(slice(int(l), int(r) + 1))
                else:
                    sel.append(getattr(self, d))
            data_1d = array[tuple(sel)]
            x_1d = x_full[tuple(sel)]

            if np.iscomplexobj(data_1d):
                data_1d = np.abs(data_1d)

            self.xmin = float(np.min(x_1d))
            self.xmax = float(np.max(x_1d))
            self.ymin = float(np.min(data_1d))
            self.ymax = float(np.max(data_1d))

        # ------------------------------------------------------------------ #
        #  Assemble Panel layout                                             #
        # ------------------------------------------------------------------ #
        def panel(self):
            widgets = []
            for d in dims:
                if d == plot_dim:
                    widgets.append(pn.Param(self.param[d if d != plot_dim else "crop"]))
                else:
                    widgets.append(pn.Param(self.param[d]))
            if np.iscomplexobj(array):
                widgets.append(pn.Param(self.param.view_kind))
            if np.iscomplexobj(c):
                widgets.append(pn.Param(self.param.c_view_kind))
            widgets.append(pn.Param(self.param.log_norm))
            # --- manual limit controls widgets -------------------------------- #
            widgets.append(pn.Param(self.param.manual_limits))
            widgets.extend(
                [
                    pn.Param(self.param.xmin),
                    pn.Param(self.param.xmax),
                    pn.Param(self.param.ymin),
                    pn.Param(self.param.ymax),
                    pn.widgets.Button.from_param(
                        self.param.autoscale, name="Autoscale"
                    ),
                ]
            )
            # add plot type selector
            widgets.append(pn.Param(self.param.plot_type))
            # add colormap picker
            widgets.append(pn.Param(self.param.colormap))
            return pn.Row(
                pn.Column(*widgets, sizing_mode="stretch_height"),
                pn.panel(self._curve, sizing_mode="stretch_both"),
                sizing_mode="stretch_both",
                name=title or "1‑D Viewer",
            )

    viewer = _Viewer(name=title or "Viewer")
    pn.serve(viewer.panel(), title=title or "1‑D Viewer", show=show)
