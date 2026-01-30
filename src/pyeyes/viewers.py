import json
import os
import pickle
import subprocess
import sys
import tempfile
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import bokeh
import holoviews as hv
import numpy as np
import panel as pn
import param
from bokeh.models.formatters import BasicTickFormatter
from holoviews import opts

from . import config, error, metrics, themes
from .cmap.cmap import VALID_COLORMAPS, VALID_ERROR_COLORMAPS
from .enums import METRICS_STATE, POPUP_LOCATION, ROI_LOCATION, ROI_STATE, ROI_VIEW_MODE
from .gui import (
    Button,
    Checkbox,
    CheckBoxGroup,
    CheckButtonGroup,
    ColorPicker,
    EditableFloatSlider,
    EditableIntSlider,
    EditableRangeSlider,
    IntInput,
    IntRangeSlider,
    IntSlider,
    Pane,
    RadioButtonGroup,
    RawPanelObject,
    Select,
    StaticText,
    TextAreaInput,
    TextInput,
)
from .gui.scroll import ScrollHandler
from .slicers import NDSlicer
from .utils import parse_dimensional_input, sanitize_css_class, tonp

hv.extension("bokeh")
pn.extension(notifications=True)

# Clear bokeh warning "dropping a patch..."
# tis a known issue, just a marker of slow rendering speed...
bokeh_root_logger = bokeh.logging.getLogger()
bokeh_root_logger.manager.disable = bokeh.logging.WARNING


class Viewer:
    """
    Base class for pyeyes viewers.

    Provides subscription infrastructure for bidirectional communication
    between widgets and the viewer/slicer components.
    """

    def __init__(self, data, **kwargs):
        """
        Generic class for viewing image data.

        Parameters
        ----------
        data : dict of np.ndarray
            Image data keyed by name.
        """
        self.data = data

    def launch(self, title="Viewer", show=True, **kwargs):
        """
        Launch the viewer.

        Parameters
        ----------
        title : str
            Title for the viewer window/tab
        show : bool
            If True, opens browser. If False, starts server silently (useful for testing)
        **kwargs
            Additional arguments passed to _launch

        Returns
        -------
        server : panel.server.Server or None
            The Panel server instance (can be used to stop the server)
        """
        error.install_pyeyes_error_handler()
        try:
            return self._launch(title=title, show=show, **kwargs)
        finally:
            error.uninstall_pyeyes_error_handler()

    def _launch(self, title="Viewer", show=True, **kwargs):
        """Launch the viewer (override in subclass)."""
        raise NotImplementedError


class ComparativeViewer(Viewer, param.Parameterized):

    # Viewing Dimensions
    vdim_horiz = param.ObjectSelector(default="x")
    vdim_vert = param.ObjectSelector(default="y")

    # Displayed Images
    single_image_toggle = param.Boolean(default=False)
    display_images = param.ListSelector(default=[], objects=[])

    # Config
    config_path = param.String(default="./config.yaml")
    html_export_path = param.String(default="./viewer.html")
    html_export_page_name = param.String(default="MRI Viewer")

    def __init__(
        self,
        data: Union[Dict[str, np.ndarray], np.ndarray],
        named_dims: Optional[Sequence[str]] = None,
        view_dims: Optional[Sequence[str]] = None,
        cat_dims: Optional[Dict[str, List]] = {},
        config_path: Optional[str] = None,
    ):
        """
        Viewer for comparing n-dimensional image data with linked slicing and metrics.

        Builds an interactive MRI/viewer app: multiple images share the same slice indices,
        viewing dimensions (which 2D plane to show), contrast (clim, colormap), ROI, and
        optional difference maps / text metrics. Data can be a single array or a dict of
        arrays (e.g. different reconstructions or image types) with the same shape.
        Dimension names drive slice widgets and axis labels; categorical dimensions use
        dropdowns instead of sliders (e.g. "Map Type" with options T1, T2, PD).

        Parameters
        ----------
        data : Union[Dict[str, np.ndarray], np.ndarray]
            Single image or dict of images. If dict, keys are display names and values
            are numpy arrays of the same shape. Single array is treated as {"Image": array}.
        named_dims : Optional[Sequence[str]]
            Name for each dimension in order of array axes (length must equal data.ndim).
            Accepted formats:
            - List of strings: e.g. ["x", "y", "z"] or ["Phase", "Read", "Slice"].
            - String of N characters: e.g. "xyz" for 3D (each character is one dim name).
            - Delimited string: space, comma, semicolon, hyphen, or underscore, e.g.
              "Phase, Read, Slice" or "x y z" (must produce exactly N tokens).
            If None, defaults to ["Dim 0", "Dim 1", ...].
        view_dims : Optional[Sequence[str]]
            Which two dimensions form the initial 2D view (must be in named_dims and
            non-categorical, non-singleton). If None: uses ["x", "y"] when those names
            exist in named_dims, otherwise the first two sliceable dimensions.
        cat_dims : Optional[Dict[str, List]]
            Categorical dimensions: key = dimension name (must be in named_dims), value =
            list of option labels (e.g. {"Map Type": ["T1", "T2", "PD"]}). These dimensions
            use dropdowns instead of integer sliders.
        config_path : Optional[str]
            Path to a JSON config file saved from the viewer's Export pane. Loads viewer,
            slicer, and ROI settings (e.g. view dims, clim, colormap, ROI box). A subset
            of settings can be applied to different datasets or shapes; incompatible
            image sets or dimension names are handled with warnings and fallbacks.

        Examples
        --------
        **Simple — single 3D array, default dim names and xy view:**

        >>> arr = np.random.randn(64, 64, 24)
        >>> viewer = ComparativeViewer(arr)
        >>> viewer.launch()

        **Moderate — dict of 3D images, short dim names, auto xy view:**

        >>> data = {"Recon": recon_xyz, "Reference": ref_xyz}  # each (H, W, D)
        >>> viewer = ComparativeViewer(data, named_dims="xyz")
        >>> # view_dims not supplied → defaults to ["x", "y"]; slice over z

        **Complex — named/view dims as list, categorical dim, and config:**

        >>> # 4D array: (Map Type, Phase, Read, Slice) with 3 map types
        >>> quant = np.stack([t1_vol, t2_vol, pd_vol], axis=0)  # (3, H, W, D)
        >>> viewer = ComparativeViewer(
        ...     {"Quantitative": quant},
        ...     named_dims=["Map Type", "Phase", "Read", "Slice"],
        ...     view_dims=["Phase", "Read"],
        ...     cat_dims={"Map Type": ["T1", "T2", "PD"]},
        ...     config_path="./my_viewer_config.json",
        ... )
        >>> viewer.launch()
        """

        from_config = config_path is not None and os.path.exists(config_path)

        # Defaults
        if not isinstance(data, dict):
            data = tonp(data)

        if isinstance(data, np.ndarray):
            data = {"Image": data}

        data = {k: tonp(v) for k, v in data.items()}

        first_key = list(data.keys())[0]
        named_dims = parse_dimensional_input(named_dims, data[first_key].ndim)

        super().__init__(data)
        param.Parameterized.__init__(self)

        img_names = list(data.keys())
        img_list = list(data.values())

        N_img = len(img_list)
        N_dim = len(named_dims)

        self.is_complex_data = any([np.iscomplexobj(img) for img in img_list])

        if cat_dims is not None and len(cat_dims) > 0:
            assert all(
                [dim in named_dims for dim in cat_dims.keys()]
            ), "Category dimensions must be in dimension_names."
        else:
            cat_dims = {}
        self.cat_dims = cat_dims

        # Sliceable dimensions are only those which are non-categorical and not singleton
        self.noncat_dims = []
        for i, d in enumerate(named_dims):
            if d in cat_dims.keys():
                continue
            elif data[first_key].shape[i] <= 1:
                print(f"Detected '{d}' is singleton. Cannot slice.")
                continue
            else:
                self.noncat_dims.append(d)

        assert np.array(
            [img.shape == img_list[0].shape for img in img_list]
        ).all(), "All viewed data must have the same input shape."
        assert (
            N_dim == img_list[0].ndim
        ), "Number of dimension names must match the number of dimensions in the data."

        if view_dims is not None:
            view_dims = parse_dimensional_input(view_dims, 2)
            assert all(
                [dim in self.noncat_dims for dim in view_dims]
            ), "All view dimensions must be non-singleton, non-categorical, and in dimension_names."
        else:
            # Default viewing dims
            dl = [dim.lower() for dim in named_dims]
            if "x" in dl and "y" in dl:
                view_dims = [
                    named_dims[dl.index("x")],
                    named_dims[dl.index("y")],
                ]
            else:
                view_dims = self.noncat_dims[:2]

        # Init display images
        self.param.display_images.objects = img_names
        self.display_images = img_names

        # Update View dims
        self.param.vdim_horiz.objects = named_dims
        self.param.vdim_vert.objects = named_dims
        self.vdim_horiz = view_dims[0]
        self.vdim_vert = view_dims[1]

        self.ndims = named_dims
        self.img_names = img_names

        # Possibly update parameters from config
        if from_config:
            # Initi display images and view dims here
            print(f"Loading viewer config from {config_path}...")
            cfg = self.load_from_config(config_path)
        else:
            cfg = None

        self.vdims = (self.vdim_horiz, self.vdim_vert)
        self.N_img = N_img
        self.N_dim = N_dim

        # Aggregate data, stacking image type to first axis
        self.raw_data = data

        # Instantiate dataset for intial view dims
        self.dataset = self._build_dataset(self.vdims)

        # Initiate scroll handler (creates JS->Python bridge source)
        self.scroll_handler = ScrollHandler(
            callback_func=self._handle_scroll,
            buffer_time=10,  # [ms]
        )
        self.scroll_handle_lock = False

        # Instantiate slicer
        self.slicer = NDSlicer(
            self.dataset,
            self.vdims,
            cdim="ImgName",
            clabs=img_names,
            cat_dims=cat_dims,
            cfg=cfg,
            plot_hooks=[
                self.scroll_handler.build_bokeh_scroll_hook(),  # make the plot the src of scroll event
            ],
        )

        # Attach watcher variables to slicer attributes that need GUI updates
        self.slicer.param.watch(self._roi_state_watcher, "roi_state")

        """
        Create Panel Layout
        """
        # Build panes using Pane-based pattern
        self.panes = {
            "View": self._init_view_pane(),
            "Contrast": self._init_contrast_pane(),
            "ROI": self._init_roi_pane(),
            "Analysis": self._init_analysis_pane(),
            "Misc": self._init_misc_pane(),
            "Export": self._init_export_pane(),
        }

        # order of tabs
        self.tab_order = ["View", "Contrast", "ROI", "Analysis", "Misc", "Export"]

        # Build Control Panel from Panes
        control_panel = pn.Tabs(
            *[(tab, self.panes[tab].to_column()) for tab in self.tab_order],
        )

        # App
        self.app = pn.Row(control_panel, self.slicer.view)

        # make sure roi_state is consistent with widgets
        if from_config:
            self._roi_state_watcher(self.slicer.roi_state)
            self._update_error_map_type(self.slicer.error_map_type, autoformat=False)
        else:
            self._autoscale_clim(event=None)

    def _launch(self, title="MRI Viewer", show=True, **kwargs):
        """Serve the viewer app with pn.serve."""
        server = pn.serve(self.app, title=title, show=show, **kwargs)

        return server

    def load_from_config(self, config_path: str):
        """
        Load viewer/slicer/ROI settings from a JSON config file.

        Parameters
        ----------
        config_path : str
            Path to JSON file (e.g. from Export Config).

        Returns
        -------
        dict
            Loaded config dict (with metadata for image/dim compatibility).
        """
        with open(config_path, "r") as f:
            cfg = json.load(f)

        # add metadata
        cfg["metadata"] = dict(
            same_images=True,
            same_dims=True,
        )

        # Check that config is compatible with newly supplied data
        cfg_images = cfg["viewer_config"]["display_images"]["value"]
        cfg_ndims = cfg["viewer_config"]["vdim_horiz"]["objects"]

        # Will not set display image related parameters if not consistent with config
        if not (set(cfg_images) == set(self.img_names)):
            warnings.warn(
                "Supplied images do not match config - Loading viewer with default image selection.",
                RuntimeWarning,
            )
            cfg["metadata"]["same_images"] = False
            cfg["viewer_config"].pop("display_images")
            cfg["viewer_config"].pop("single_image_toggle")
            cfg["slicer_config"].pop("display_images")
            cfg["slicer_config"].pop("metrics_reference")

        # Will not set dimension related parameters if not consistent with config
        if not (set(cfg_ndims) == set(self.ndims)):
            warnings.warn(
                "Config dims do not match supplied named dims. Using default settings.",
                RuntimeWarning,
            )
            cfg["metadata"]["same_dims"] = False
            cfg["viewer_config"].pop("vdim_horiz")
            cfg["viewer_config"].pop("vdim_vert")
            cfg["slicer_config"].pop("dim_indices")

        config.deserialize_parameters(self, cfg["viewer_config"])

        return cfg

    """
    Widget Management
    """

    def _replace_widget_in_app(
        self, pane_name: str, widget_name: str, new_widget
    ) -> None:
        """Replace a widget in the pane and control panel display."""
        pane = self.panes.get(pane_name)
        if pane is None:
            return

        # Update pane's internal widget reference
        pane.replace_widget(widget_name, new_widget)

        # Update the control panel display
        if widget_name in pane._widget_order:
            tab_index = self.tab_order.index(pane_name)
            widget_idx = pane._widget_order.index(widget_name)
            self.app[0][tab_index][widget_idx] = new_widget.get_widget()

    """
    Build Widgets for each tab
    """

    def _init_view_pane(self) -> Pane:
        """Build Pane with all View tab widgets."""
        pane = Pane("View", viewer=self)

        # Viewing Dimensions
        for widget in self._build_vdim_widgets():
            pane.add_widget(widget)

        # Widgets for Slicing Dimensions
        for widget in self._build_sdim_widgets():
            pane.add_widget(widget)

        # Other widgets on the viewing page
        for widget in self._build_viewing_widgets():
            pane.add_widget(widget)

        # Single Toggle View widgets
        pane.add_widget(self._build_single_toggle_widget())

        # Display images description
        im_display_desc = StaticText(
            name="im_display_desc",
            display_name="Displayed Images",
            value="Click names to toggle visibility.",
            css_classes=["pyeyes-im-display-desc"],
            viewer=self,
        )
        pane.add_widget(im_display_desc)

        # Display images widget
        pane.add_widget(self._build_display_images_widget())

        return pane

    def _init_contrast_pane(self) -> Pane:
        """Build Pane with all Contrast tab widgets."""
        pane = Pane("Contrast", viewer=self)

        # Complex view and autoscale widgets
        cplx_widget, autoscale_widget = self._build_cplx_widgets()
        pane.add_widget(cplx_widget)

        # Color map stuff
        for widget in self._build_contrast_widgets():
            pane.add_widget(widget)

        # Auto-scaling button
        pane.add_widget(autoscale_widget)

        return pane

    def _init_roi_pane(self) -> Pane:
        """Build Pane with all ROI tab widgets."""
        pane = Pane("ROI", viewer=self)

        for widget in self._build_roi_widgets():
            pane.add_widget(widget)

        return pane

    def _init_analysis_pane(self) -> Pane:
        """Build Pane with all Analysis tab widgets."""
        pane = Pane("Analysis", viewer=self)

        for widget in self._build_analysis_widgets():
            pane.add_widget(widget)

        return pane

    def _init_export_pane(self) -> Pane:
        """Build Pane with all Export tab widgets."""
        pane = Pane("Export", viewer=self)

        for widget in self._build_export_widgets():
            pane.add_widget(widget)

        return pane

    def _init_misc_pane(self) -> Pane:
        """Build Pane with all Misc tab widgets."""
        pane = Pane("Misc", viewer=self)

        for widget in self._build_misc_widgets():
            pane.add_widget(widget)

        return pane

    def _build_dataset(self, vdims: Sequence[str]) -> hv.Dataset:
        """Build HoloViews Dataset from raw_data for given viewing dimensions."""
        img_list = list(self.raw_data.values())

        proc_data = np.stack(img_list, axis=0)

        dim_ranges = [list(self.raw_data.keys())]
        for i in range(1, proc_data.ndim):
            if self.ndims[i - 1] in self.cat_dims:
                dim_ranges.append(self.cat_dims[self.ndims[i - 1]])
            else:
                dim_ranges.append(range(proc_data.shape[i]))

        # Convention is to reverse ordering relative to dimensions named
        proc_data = proc_data.transpose(*list(range(proc_data.ndim - 1, -1, -1)))

        return hv.Dataset((*dim_ranges, proc_data), ["ImgName"] + self.ndims, "Value")

    def _build_vdim_widgets(self) -> List:
        """Build L/R and U/D viewing dimension selector widgets."""

        @error.error_handler_decorator()
        def vdim_horiz_callback(new_value):
            if new_value != self.vdim_horiz:
                vh_new = new_value
                vv_new = (
                    self.vdim_horiz if (vh_new == self.vdim_vert) else self.vdim_vert
                )
                self._update_vdims([vh_new, vv_new])

        vdim_horiz = Select(
            name="vdim_horiz",
            display_name="L/R Viewing Dimension",
            options=self.noncat_dims,
            value=self.vdims[0],
            css_classes=["pyeyes-vdim-lr"],
            callback=vdim_horiz_callback,
            viewer=self,
        )

        @error.error_handler_decorator()
        def vdim_vert_callback(new_value):
            if new_value != self.vdim_vert:
                vv_new = new_value
                vh_new = (
                    self.vdim_vert if (vv_new == self.vdim_horiz) else self.vdim_horiz
                )
                self._update_vdims([vh_new, vv_new])

        vdim_vert = Select(
            name="vdim_vert",
            display_name="U/D Viewing Dimension",
            options=self.noncat_dims,
            value=self.vdims[1],
            css_classes=["pyeyes-vdim-ud"],
            callback=vdim_vert_callback,
            viewer=self,
        )

        return [vdim_horiz, vdim_vert]

    def _update_vdims(self, new_vdims):
        """Update viewing dimensions and sync slicer/widgets."""
        assert len(new_vdims) == 2, "Must provide exactly 2 viewing dimensions."

        with param.parameterized.discard_events(self.slicer):

            # Update attributes
            self.vdims = new_vdims
            self.vdim_horiz = new_vdims[0]
            self.vdim_vert = new_vdims[1]

            # Update vdim widgets
            self.panes["View"].get_widget("vdim_horiz").value = new_vdims[0]
            self.panes["View"].get_widget("vdim_vert").value = new_vdims[1]

            # Update Slicer
            old_vdims = self.slicer.vdims
            self.slicer.set_volatile_dims(new_vdims)

            # Update displayed widgets if interchange of vdim and sdims
            if set(old_vdims) != set(new_vdims):
                new_sdim_widgets = self._build_sdim_widgets()

                for i, widget in enumerate(new_sdim_widgets):
                    self._replace_widget_in_app("View", f"sdim{i}", widget)

                # update start/stop/step widgets
                new_export_widgets = self._build_export_widgets()
                for widget in new_export_widgets:
                    if "_export_range" in widget.name:
                        self._replace_widget_in_app("Export", widget.name, widget)

            # Reset crops
            lr_crop_widget = self.panes["View"].get_widget("lr_crop")
            lr_crop_widget.bounds = (0, self.slicer.img_dims[0])
            lr_crop_widget.value = self.slicer.lr_crop

            ud_crop_widget = self.panes["View"].get_widget("ud_crop")
            ud_crop_widget.bounds = (0, self.slicer.img_dims[1])
            ud_crop_widget.value = self.slicer.ud_crop

        self.slicer.param.trigger("lr_crop", "ud_crop")

    def _build_sdim_widgets(self) -> List:
        """Build widgets for each slicing dimension (sliders or selectors)."""
        widgets = []
        num_interactable_dims = 0

        for i, dim in enumerate(self.slicer.sdims):
            # Create callback with closure over dim
            def make_callback(this_dim):
                def callback(new_value):
                    self._update_sdim(this_dim, new_value)

                return callback

            if dim in self.cat_dims.keys():
                widget = Select(
                    name=f"sdim{num_interactable_dims}",
                    display_name=dim,
                    options=self.cat_dims[dim],
                    value=self.slicer.dim_indices[dim],
                    css_classes=[f"pyeyes-sdim-{sanitize_css_class(dim)}"],
                    callback=make_callback(dim),
                    viewer=self,
                )
            else:
                if self.slicer.dim_sizes[dim] - 1 > 0:
                    widget = EditableIntSlider(
                        name=f"sdim{num_interactable_dims}",
                        display_name=dim,
                        start=0,
                        end=self.slicer.dim_sizes[dim] - 1,
                        value=self.slicer.dim_indices[dim],
                        css_classes=[f"pyeyes-sdim-{sanitize_css_class(dim)}"],
                        callback=make_callback(dim),
                        viewer=self,
                    )
                else:
                    # Singleton dimensions
                    continue

            widgets.append(widget)
            num_interactable_dims += 1

        return widgets

    @error.error_handler_decorator()
    def _update_sdim(self, sdim, new_dim_val):
        """Callback to set one slicing dimension and trigger redraw."""
        with param.parameterized.discard_events(self.slicer):

            self.slicer.dim_indices[sdim] = new_dim_val

            # Track last modified dimension for scroll behavior
            self.slicer.scroll_dim = sdim

            # Assume we need to autoscale if dimension updated is categorical
            if sdim in self.cat_dims.keys():
                self._autoscale_clim(event=None)
                self.slicer.update_colormap()

        # Trigger
        if sdim in self.cat_dims.keys():
            self.slicer.param.trigger("cmap")
        else:
            self.slicer.param.trigger("dim_indices")

    def _handle_scroll(self, delta: float):
        """Apply scroll delta to slicer and sync sdim widget value."""
        if self.scroll_handle_lock:
            return  # don't scroll if we are already handling a scroll

        self.scroll_handle_lock = True

        sdim, new_dim_val = self.slicer.compute_scroll_delta(delta)

        # Find the sdim widget by checking display_name
        view_pane = self.panes["View"]
        for widget_name, widget in view_pane.widgets.items():
            if widget_name.startswith("sdim") and widget.display_name == sdim:
                if widget.value != new_dim_val:
                    widget.value = new_dim_val

        self.scroll_handle_lock = False

    def _build_viewing_widgets(self) -> List:
        """Build flip, scale, crop, and related viewing control widgets."""
        widgets = []

        # Flip Widgets
        @error.error_handler_decorator()
        def flip_ud_callback(new_value):
            self.slicer.flip_ud = new_value

        flip_ud = Checkbox(
            name="flip_ud",
            display_name="Flip Image Up/Down",
            value=self.slicer.flip_ud,
            css_classes=["pyeyes-flip-ud"],
            callback=flip_ud_callback,
            viewer=self,
        )
        widgets.append(flip_ud)

        @error.error_handler_decorator()
        def flip_lr_callback(new_value):
            self.slicer.flip_lr = new_value

        flip_lr = Checkbox(
            name="flip_lr",
            display_name="Flip Image Left/Right",
            value=self.slicer.flip_lr,
            css_classes=["pyeyes-flip-lr"],
            callback=flip_lr_callback,
            viewer=self,
        )
        widgets.append(flip_lr)

        @error.error_handler_decorator()
        def size_scale_callback(new_value):
            self.slicer.size_scale = new_value

        size_scale = EditableIntSlider(
            name="size_scale",
            display_name="Size Scale",
            start=self.slicer.param.size_scale.bounds[0],
            end=self.slicer.param.size_scale.bounds[1],
            value=self.slicer.size_scale,
            step=self.slicer.param.size_scale.step,
            css_classes=["pyeyes-size-scale"],
            callback=size_scale_callback,
            viewer=self,
        )
        widgets.append(size_scale)

        def title_font_size_callback(new_value):
            self.slicer.title_font_size = new_value

        title_font_size = EditableIntSlider(
            name="title_font_size",
            display_name="Title Font Size",
            start=self.slicer.param.title_font_size.bounds[0],
            end=self.slicer.param.title_font_size.bounds[1],
            value=self.slicer.title_font_size,
            step=self.slicer.param.title_font_size.step,
            css_classes=["pyeyes-title-font-size"],
            callback=title_font_size_callback,
            viewer=self,
        )
        widgets.append(title_font_size)

        def lr_crop_callback(new_value):
            self.slicer.lr_crop = new_value

        lr_crop = IntRangeSlider(
            name="lr_crop",
            display_name="L/R Display Range",
            start=self.slicer.param.lr_crop.bounds[0],
            end=self.slicer.param.lr_crop.bounds[1],
            value=(self.slicer.lr_crop[0], self.slicer.lr_crop[1]),
            step=self.slicer.param.lr_crop.step,
            css_classes=["pyeyes-lr-crop"],
            callback=lr_crop_callback,
            viewer=self,
        )
        widgets.append(lr_crop)

        def ud_crop_callback(new_value):
            self.slicer.ud_crop = new_value

        ud_crop = IntRangeSlider(
            name="ud_crop",
            display_name="U/D Display Range",
            start=self.slicer.param.ud_crop.bounds[0],
            end=self.slicer.param.ud_crop.bounds[1],
            value=(self.slicer.ud_crop[0], self.slicer.ud_crop[1]),
            step=self.slicer.param.ud_crop.step,
            css_classes=["pyeyes-ud-crop"],
            callback=ud_crop_callback,
            viewer=self,
        )
        widgets.append(ud_crop)

        return widgets

    def _build_single_toggle_widget(self):
        """Build single view toggle widget."""

        @error.error_handler_decorator()
        def single_toggle_callback(new_value):
            self._update_toggle_single_view(new_value)

        single_toggle = Checkbox(
            name="single_toggle",
            display_name="Single View",
            value=self.single_image_toggle,
            css_classes=["pyeyes-single-view"],
            callback=single_toggle_callback,
            viewer=self,
        )
        return single_toggle

    def _update_toggle_single_view(self, new_single_toggle):
        """Toggle single-image mode and refresh display images widget."""
        self.single_image_toggle = new_single_toggle

        # Build new widget and replace in pane
        new_display_images_widget = self._build_display_images_widget()
        self._replace_widget_in_app("View", "im_display", new_display_images_widget)

        if self.single_image_toggle:
            self.display_images = [self.display_images[0]]
        else:
            self.display_images = self.img_names

        # send new display images to slicer
        self.slicer.update_display_image_list(self.display_images)

    def _build_display_images_widget(self):
        """Build display images widget (Radio or Check button group)."""
        len_names = np.sum([len(name) for name in self.img_names]).item()
        orientation = "vertical" if len_names > 30 else "vertical"

        @error.error_handler_decorator()
        def display_images_callback(new_value):
            if not isinstance(new_value, str) and len(new_value) == 0:
                pn.state.notifications.warning("Must select at least one image.")
                # Restore previous value
                if hasattr(self, "panes") and "View" in self.panes:
                    widget = self.panes["View"].get_widget("im_display")
                    if widget:
                        widget.value = self.display_images
                return
            self._update_image_display(new_value)

        if self.single_image_toggle:
            im_display = RadioButtonGroup(
                name="im_display",
                display_name="Displayed Images",
                options=self.img_names,
                value=self.img_names[0],
                button_type="success",
                button_style="outline",
                orientation=orientation,
                css_classes=["pyeyes-im-display-radio"],
                callback=display_images_callback,
                viewer=self,
            )
        else:
            im_display = CheckButtonGroup(
                name="im_display",
                display_name="Displayed Images",
                options=self.img_names,
                value=self.img_names,
                button_type="primary",
                button_style="outline",
                orientation=orientation,
                css_classes=["pyeyes-im-display-check"],
                callback=display_images_callback,
                viewer=self,
            )

        return im_display

    def _update_image_display(self, new_display_images):
        """Update displayed image set and notify slicer."""
        if isinstance(new_display_images, str):
            new_display_images = [new_display_images]

        self.display_images = new_display_images

        self.slicer.update_display_image_list(new_display_images)

    def _build_cplx_widgets(self):
        """Build complex view selector and autoscale button."""

        @error.error_handler_decorator()
        def cplx_callback(new_value):
            self._update_cplx_view(new_value)

        cplx_view = RadioButtonGroup(
            name="cplx_view",
            display_name="Complex Data",
            options=(
                ["mag", "phase", "real", "imag"]
                if self.is_complex_data
                else ["mag", "real"]
            ),
            value=self.slicer.cplx_view,
            button_type="primary",
            button_style="outline",
            css_classes=["pyeyes-cplx-view"],
            callback=cplx_callback,
            viewer=self,
        )

        autoscale = Button(
            name="autoscale",
            display_name="Auto-Scale",
            button_type="primary",
            css_classes=["pyeyes-autoscale"],
            on_click=self._autoscale_clim,
            viewer=self,
        )

        return cplx_view, autoscale

    def _update_cplx_view(self, new_cplx_view):
        """Update complex view (mag/phase/real/imag) and reset clim/cmap."""
        # Update Slicer
        with param.parameterized.discard_events(self.slicer):
            self.slicer.update_cplx_view(new_cplx_view)

            # Reset clim
            clim_widget = self.panes["Contrast"].get_widget("clim")
            clim_widget.start = self.slicer.param.vmin.bounds[0]
            clim_widget.end = self.slicer.param.vmax.bounds[1]
            clim_widget.value = (self.slicer.vmin, self.slicer.vmax)
            clim_widget.step = self.slicer.param.vmin.step

            # Reset cmap value
            self.panes["Contrast"].get_widget("cmap").value = self.slicer.cmap

        self.slicer.param.trigger("vmin", "vmax", "cmap")

    def _update_clim_widget(self, vmin, vmax, bound_min, bound_max, step):
        """Sync clim slider value, bounds, and step."""
        clim_widget = self.panes["Contrast"].get_widget("clim")
        clim_widget.value = (vmin, vmax)
        clim_widget.start = bound_min
        clim_widget.end = bound_max
        clim_widget.step = step

    def _update_clim(self, new_value):
        """Callback to set slicer vmin/vmax from clim widget."""
        with param.parameterized.discard_events(self.slicer):
            vmin, vmax = new_value

            if vmin > vmax:
                pn.state.notifications.warning("vmin > vmax. Setting vmin = vmax.")
                vmin = vmax
            elif vmax < vmin:
                pn.state.notifications.warning("vmax < vmin. Setting vmax = vmin.")
                vmax = vmin

            # Slice limits
            vmin, vmax, bound_min, bound_max, step = self.slicer.set_vmin_vmax(
                vmin=vmin,
                vmax=vmax,
            )

            self._update_clim_widget(vmin, vmax, bound_min, bound_max, step)

        self.slicer.param.trigger("vmin", "vmax")

    def _build_contrast_widgets(self) -> List:
        """Build contrast-related widgets."""
        widgets = []

        # Clim range slider
        clim = EditableRangeSlider(
            name="clim",
            display_name="clim",
            start=self.slicer.param.vmin.bounds[0],
            end=self.slicer.param.vmax.bounds[1],
            value=(self.slicer.vmin, self.slicer.vmax),
            step=self.slicer.param.vmin.step,
            format=BasicTickFormatter(
                precision=2,
                power_limit_low=-3,
                power_limit_high=5,
            ),
            css_classes=["pyeyes-clim"],
            callback=self._update_clim,
            viewer=self,
        )
        widgets.append(clim)

        # Colormap
        def cmap_callback(new_value):
            with param.parameterized.discard_events(self.slicer):
                self.slicer.cmap = new_value
                self.slicer.update_colormap()
            self.slicer.param.trigger("cmap")

        cmap = Select(
            name="cmap",
            display_name="Color Map",
            options=VALID_COLORMAPS,
            value=self.slicer.cmap,
            css_classes=["pyeyes-cmap"],
            callback=cmap_callback,
            viewer=self,
        )
        widgets.append(cmap)

        # Colorbar toggle
        def colorbar_callback(new_value):
            self.slicer.colorbar_on = new_value
            # Update colorbar label disabled state
            if hasattr(self, "panes") and "Contrast" in self.panes:
                colorbar_label_widget = self.panes["Contrast"].get_widget(
                    "colorbar_label"
                )
                if colorbar_label_widget:
                    colorbar_label_widget.disabled = not new_value

        colorbar = Checkbox(
            name="colorbar",
            display_name="Add Colorbar",
            value=self.slicer.colorbar_on,
            css_classes=["pyeyes-colorbar"],
            callback=colorbar_callback,
            viewer=self,
        )
        widgets.append(colorbar)

        # Colorbar label
        def colorbar_label_callback(new_value):
            self.slicer.colorbar_label = new_value

        colorbar_label = TextInput(
            name="colorbar_label",
            display_name="Colorbar Label",
            value=self.slicer.colorbar_label,
            css_classes=["pyeyes-colorbar-label"],
            callback=colorbar_label_callback,
            viewer=self,
        )
        # Disable if colorbar is off
        colorbar_label.disabled = not self.slicer.colorbar_on
        widgets.append(colorbar_label)

        return widgets

    def _autoscale_clim(self, event):
        """Autoscale slicer clim and update clim widget."""
        with param.parameterized.discard_events(self.slicer):
            vmin, vmax, bound_min, bound_max, step = self.slicer.autoscale_clim()
            self._update_clim_widget(vmin, vmax, bound_min, bound_max, step)
        self.slicer.param.trigger("vmin", "vmax")

    def _build_roi_widgets(self) -> List:
        """Build ROI-related widgets."""
        widgets = []

        # ROI Overlay Enabled
        def roi_mode_callback(new_value):
            self.slicer.update_roi_mode(new_value)

        roi_mode = Checkbox(
            name="roi_mode",
            display_name="ROI Overlay Enabled",
            value=bool(self.slicer.roi_mode.value),
            css_classes=["pyeyes-roi-mode"],
            callback=roi_mode_callback,
            viewer=self,
        )
        widgets.append(roi_mode)

        # Draw ROI Button
        def draw_roi_callback(event):
            if self.slicer.roi_state == ROI_STATE.ACTIVE:
                return  # do nothing if ROI is already active

            self.slicer.update_roi_state(ROI_STATE.FIRST_SELECTION)

        draw_roi = Button(
            name="draw_roi",
            display_name="Draw ROI",
            button_type="primary",
            css_classes=["pyeyes-draw-roi"],
            on_click=draw_roi_callback,
            viewer=self,
        )
        widgets.append(draw_roi)

        # Clear ROI Button
        def clear_roi_callback(event):
            self.slicer.update_roi_state(ROI_STATE.INACTIVE)

        clear_roi = Button(
            name="clear_roi",
            display_name="Clear ROI",
            button_type="warning",
            css_classes=["pyeyes-clear-roi"],
            on_click=clear_roi_callback,
            viewer=self,
        )
        clear_roi.disabled = True
        widgets.append(clear_roi)

        # ROI Colormap
        def roi_cmap_callback(new_value):
            self.slicer.update_roi_colormap(new_value)

        roi_cmap = Select(
            name="roi_cmap",
            display_name="ROI Color Map",
            options=self.slicer.param.roi_cmap.objects,
            value=self.slicer.roi_cmap,
            css_classes=["pyeyes-roi-cmap"],
            callback=roi_cmap_callback,
            viewer=self,
        )
        roi_cmap.visible = False
        widgets.append(roi_cmap)

        # Zoom Scale
        def zoom_scale_callback(new_value):
            self.slicer.update_roi_zoom_scale(new_value)

        zoom_scale = EditableFloatSlider(
            name="zoom_scale",
            display_name="Zoom Scale",
            start=1.0,
            end=10.0,
            value=self.slicer.ROI.zoom_scale,
            step=0.1,
            css_classes=["pyeyes-zoom-scale"],
            callback=zoom_scale_callback,
            viewer=self,
        )
        zoom_scale.visible = False
        widgets.append(zoom_scale)

        # ROI Location
        def roi_loc_callback(new_value):
            self.slicer.update_roi_loc(new_value)

        roi_loc = Select(
            name="roi_loc",
            display_name="ROI Location",
            options=[loc.value for loc in ROI_LOCATION],
            value=self.slicer.ROI.roi_loc.value,
            css_classes=["pyeyes-roi-loc"],
            callback=roi_loc_callback,
            viewer=self,
        )
        roi_loc.visible = False
        widgets.append(roi_loc)

        # ROI L/R Crop
        def roi_lr_crop_callback(new_value):
            self.slicer.update_roi_lr_crop(new_value)

        roi_lr_crop = EditableRangeSlider(
            name="roi_lr_crop",
            display_name="ROI L/R Crop",
            value=(0, 1),
            start=0,
            end=100000,
            step=0.1,
            css_classes=["pyeyes-roi-lr-crop"],
            callback=roi_lr_crop_callback,
            viewer=self,
        )
        roi_lr_crop.visible = False
        widgets.append(roi_lr_crop)

        # ROI U/D Crop
        def roi_ud_crop_callback(new_value):
            self.slicer.update_roi_ud_crop(new_value)

        roi_ud_crop = EditableRangeSlider(
            name="roi_ud_crop",
            display_name="ROI U/D Crop",
            value=(0, 1),
            start=0,
            end=100000,
            step=0.1,
            css_classes=["pyeyes-roi-ud-crop"],
            callback=roi_ud_crop_callback,
            viewer=self,
        )
        roi_ud_crop.visible = False
        widgets.append(roi_ud_crop)

        # ROI Line Color
        def roi_line_color_callback(new_value):
            self.slicer.update_roi_line_color(new_value)

        roi_line_color = ColorPicker(
            name="roi_line_color",
            display_name="ROI Line Color",
            value=self.slicer.ROI.color,
            css_classes=["pyeyes-roi-line-color"],
            callback=roi_line_color_callback,
            viewer=self,
        )
        roi_line_color.visible = False
        widgets.append(roi_line_color)

        # ROI Line Width
        def roi_line_width_callback(new_value):
            self.slicer.update_roi_line_width(new_value)

        roi_line_width = IntSlider(
            name="roi_line_width",
            display_name="ROI Line Width",
            start=1,
            end=10,
            value=self.slicer.ROI.line_width,
            css_classes=["pyeyes-roi-line-width"],
            callback=roi_line_width_callback,
            viewer=self,
        )
        roi_line_width.visible = False
        widgets.append(roi_line_width)

        # ROI Zoom Order
        def roi_zoom_order_callback(new_value):
            self.slicer.update_roi_zoom_order(new_value)

        roi_zoom_order = IntInput(
            name="roi_zoom_order",
            display_name="ROI Zoom Order",
            value=self.slicer.ROI.zoom_order,
            start=0,
            end=3,
            css_classes=["pyeyes-roi-zoom-order"],
            callback=roi_zoom_order_callback,
            viewer=self,
        )
        roi_zoom_order.visible = False
        widgets.append(roi_zoom_order)

        return widgets

    def _roi_state_watcher(self, event):
        """Sync ROI pane widgets and visibility when roi_state changes."""
        if hasattr(event, "new"):
            new_state = event.new
        else:
            new_state = event

        roi_pane = self.panes["ROI"]

        # Clear button enabled or not
        roi_pane.get_widget("clear_roi").disabled = new_state == ROI_STATE.INACTIVE

        with param.parameterized.discard_events(self.slicer):
            # update ranges of sliders upon completion of ROI
            if new_state == ROI_STATE.ACTIVE:

                roi_l, roi_b, roi_r, roi_t = self.slicer.ROI.lbrt()

                lr_max = self.slicer.lr_crop
                ud_max = self.slicer.ud_crop

                lr_step = self.slicer.param.lr_crop.step
                ud_step = self.slicer.param.ud_crop.step

                roi_lr_crop = roi_pane.get_widget("roi_lr_crop")
                roi_lr_crop.start = lr_max[0]
                roi_lr_crop.end = lr_max[1]
                roi_lr_crop.value = (roi_l, roi_r)
                roi_lr_crop.step = lr_step

                roi_ud_crop = roi_pane.get_widget("roi_ud_crop")
                roi_ud_crop.start = ud_max[0]
                roi_ud_crop.end = ud_max[1]
                roi_ud_crop.value = (roi_b, roi_t)
                roi_ud_crop.step = ud_step

                # Constrain Zoom scale
                max_lr_zoom = abs(lr_max[1] - lr_max[0]) / abs(roi_r - roi_l)
                max_ud_zoom = abs(ud_max[1] - ud_max[0]) / abs(roi_t - roi_b)
                max_zoom = round(min(max_lr_zoom, max_ud_zoom), 1)

                zoom_scale = roi_pane.get_widget("zoom_scale")
                zoom_scale.start = 1.0
                zoom_scale.end = max_zoom

            active = new_state == ROI_STATE.ACTIVE
            roi_pane.get_widget("roi_cmap").visible = active
            roi_pane.get_widget("zoom_scale").visible = active
            roi_pane.get_widget("roi_loc").visible = active
            roi_pane.get_widget("roi_lr_crop").visible = active
            roi_pane.get_widget("roi_ud_crop").visible = active
            roi_pane.get_widget("roi_line_color").visible = active
            roi_pane.get_widget("roi_line_width").visible = active
            roi_pane.get_widget("roi_zoom_order").visible = active

            # Enable certain widgets only if roi_mode = infigure
            is_sep = self.slicer.roi_mode == ROI_VIEW_MODE.Separate
            roi_pane.get_widget("zoom_scale").disabled = is_sep
            roi_pane.get_widget("roi_loc").disabled = is_sep

    def _build_analysis_widgets(self) -> List:
        """Build Analysis-related widgets."""
        widgets = []

        # Reference Dataset
        def reference_callback(new_value):
            self.slicer.update_reference_dataset(new_value)

        reference = Select(
            name="reference",
            display_name="Reference Dataset",
            options=self.img_names,
            value=self.slicer.metrics_reference,
            css_classes=["pyeyes-reference-dataset"],
            callback=reference_callback,
            viewer=self,
        )
        widgets.append(reference)

        # Error Map Type
        diff_map_type = Select(
            name="diff_map_type",
            display_name="Error Map Type",
            options=metrics.MAPPABLE_METRICS,
            value=self.slicer.error_map_type,
            css_classes=["pyeyes-error-map-type"],
            callback=self._update_error_map_type,
            viewer=self,
        )
        widgets.append(diff_map_type)

        # Text description
        text_description = StaticText(
            name="text_description",
            display_name="Metrics",
            value="Select metrics to display in text.",
            css_classes=["pyeyes-metrics-text-description"],
            viewer=self,
        )
        widgets.append(text_description)

        # Text Metrics Checkbox
        def text_metrics_callback(new_value):
            self.slicer.update_metrics_text_types(new_value)

        text_metrics = CheckBoxGroup(
            name="text_metrics",
            display_name="Metrics",
            options=metrics.FULL_METRICS,
            value=self.slicer.metrics_text_types,
            css_classes=["pyeyes-metrics-text-checkbox"],
            callback=text_metrics_callback,
            viewer=self,
        )
        widgets.append(text_metrics)

        # Button description
        button_description = StaticText(
            name="button_description",
            display_name="Enable",
            value="Click to add 'Error map' or 'text metrics'.",
            css_classes=["pyeyes-metrics-text-button-description"],
            viewer=self,
        )
        widgets.append(button_description)

        # Determine display options value
        if self.slicer.metrics_state == METRICS_STATE.ALL:
            display_options_value = ["Error Map", "Text"]
        elif self.slicer.metrics_state == METRICS_STATE.MAP:
            display_options_value = ["Error Map"]
        elif self.slicer.metrics_state == METRICS_STATE.TEXT:
            display_options_value = ["Text"]
        else:
            display_options_value = []

        def display_options_callback(new_value):
            pn.state.notifications.clear()
            pn.state.notifications.info("Building...", duration=0)

            if ("Error Map" in new_value) and ("Text" in new_value):
                new_state = METRICS_STATE.ALL
            elif "Error Map" in new_value:
                new_state = METRICS_STATE.MAP
            elif "Text" in new_value:
                new_state = METRICS_STATE.TEXT
            else:
                new_state = METRICS_STATE.INACTIVE

            self.slicer.update_metrics_state(new_state)

            pn.state.notifications.clear()
            pn.state.notifications.info("Done!", duration=1000)

        display_options = CheckButtonGroup(
            name="display_options",
            display_name="Display Metrics",
            button_type="primary",
            button_style="outline",
            value=display_options_value,
            options=["Error Map", "Text"],
            css_classes=["pyeyes-metrics-text-button"],
            callback=display_options_callback,
            viewer=self,
        )
        widgets.append(display_options)

        # Error Scale
        def error_scale_callback(new_value):
            self.slicer.update_error_map_scale(new_value)

        error_scale = EditableFloatSlider(
            name="error_scale",
            display_name="Error Scale",
            start=1.0,
            end=50.0,
            value=self.slicer.error_map_scale,
            step=0.1,
            css_classes=["pyeyes-error-scale"],
            callback=error_scale_callback,
            viewer=self,
        )
        widgets.append(error_scale)

        # Normalize displayed images fully
        def display_normalize_callback(new_value):
            # disable error map normalization if displayed images are normalized
            with param.parameterized.discard_events(self.slicer):
                analysis_pane = self.panes["Analysis"]
                analysis_pane.get_widget("error_normalize").disabled = new_value
                self.slicer.normalize_for_display = new_value
            self.slicer.param.trigger("normalize_for_display")

        display_normalize = Checkbox(
            name="display_normalize",
            display_name="Normalize Images",
            value=self.slicer.normalize_for_display,
            css_classes=["pyeyes-display-normalize"],
            callback=display_normalize_callback,
            viewer=self,
        )
        widgets.append(display_normalize)

        # Normalize Error Map
        def error_normalize_callback(new_value):
            self.slicer.update_normalize_error_map(new_value)

        error_normalize = Checkbox(
            name="error_normalize",
            display_name="Normalize for Error Metrics Only",
            value=self.slicer.normalize_error_map,
            css_classes=["pyeyes-error-normalize"],
            callback=error_normalize_callback,
            viewer=self,
        )
        error_normalize.disabled = self.slicer.normalize_for_display
        widgets.append(error_normalize)

        # Error Color Map
        def error_cmap_callback(new_value):
            self.slicer.update_error_map_cmap(new_value)

        error_cmap = Select(
            name="error_cmap",
            display_name="Error Map Color Map",
            options=VALID_ERROR_COLORMAPS,
            value=self.slicer.error_map_cmap,
            css_classes=["pyeyes-error-cmap"],
            callback=error_cmap_callback,
            viewer=self,
        )
        widgets.append(error_cmap)

        # Text Metrics Font Size
        def metrics_text_font_size_callback(new_value):
            self.slicer.update_metrics_text_font_size(new_value)

        metrics_text_font_size = EditableIntSlider(
            name="metrics_text_font_size",
            display_name="Text Metrics Font Size",
            start=5,
            end=24,
            value=self.slicer.metrics_text_font_size,
            step=1,
            css_classes=["pyeyes-metrics-text-font-size"],
            callback=metrics_text_font_size_callback,
            viewer=self,
        )
        widgets.append(metrics_text_font_size)

        # Text Metrics Location
        def metrics_text_font_loc_callback(new_value):
            self.slicer.update_metrics_text_location(ROI_LOCATION(new_value))

        metrics_text_font_loc = Select(
            name="metrics_text_font_loc",
            display_name="Text Metrics Location",
            options=[loc.value for loc in ROI_LOCATION],
            value=self.slicer.metrics_text_location.value,
            css_classes=["pyeyes-metrics-text-font-loc"],
            callback=metrics_text_font_loc_callback,
            viewer=self,
        )
        widgets.append(metrics_text_font_loc)

        # Autoformat Error Map Button
        error_map_autoformat = Button(
            name="error_map_autoformat",
            display_name="Autoformat Error Map",
            button_type="primary",
            css_classes=["pyeyes-error-map-autoformat"],
            on_click=self._autoformat_error_map,
            viewer=self,
        )
        widgets.append(error_map_autoformat)

        return widgets

    def _update_error_map_type(self, new_type, autoformat=True):
        """Set error map type and optionally autoformat scale/cmap."""
        with param.parameterized.discard_events(self.slicer):
            self.slicer.update_error_map_type(new_type)

        self.panes["Analysis"].get_widget("error_scale").visible = new_type != "SSIM"

        if self.slicer.metrics_state != METRICS_STATE.INACTIVE and autoformat:
            with param.parameterized.discard_events(self.slicer):
                self._autoformat_error_map(None)
            self.slicer.param.trigger("metrics_state")

    def _autoformat_error_map(self, event):
        """Autoformat error map and sync Analysis widgets."""
        # Update Slicer
        with param.parameterized.discard_events(self.slicer):
            self.slicer.autoformat_error_map()

            # Update gui
            self.panes["Analysis"].get_widget(
                "error_cmap"
            ).value = self.slicer.error_map_cmap
            self.panes["Analysis"].get_widget(
                "error_scale"
            ).value = self.slicer.error_map_scale

        self.slicer.param.trigger("error_map_scale")

    def _build_export_widgets(self) -> List:
        """Build Export-related widgets."""
        # Default object width
        dwidth = pn.widgets.IntInput().width
        box_height = 50
        noncat_sdims = [d for d in self.slicer.sdims if d not in self.cat_dims.keys()]

        widgets = []

        # Export Config description
        export_path_desc = StaticText(
            name="export_path_desc",
            display_name="Save Viewer Config",
            value="Save config to yaml file. Use: Viewer(..., config_path=path).",
            width=dwidth,
            css_classes=["pyeyes-export-config-path-desc"],
            viewer=self,
        )
        widgets.append(export_path_desc)

        # Export Config Paths
        def export_path_callback(new_value):
            new_value = new_value.replace("\n", "")
            self.config_path = new_value

        export_path = TextAreaInput(
            name="export_path",
            display_name="Export Config Path",
            value=self.config_path,
            placeholder="Enter path to export config file",
            height=box_height,
            css_classes=["pyeyes-export-config-path"],
            callback=export_path_callback,
            viewer=self,
        )
        widgets.append(export_path)

        # Export Config Button
        export_config_button = Button(
            name="export_config_button",
            display_name="Export Config",
            button_type="primary",
            on_click=self._export_config,
            css_classes=["pyeyes-export-config-button"],
            viewer=self,
        )
        widgets.append(export_config_button)

        # Separator line
        export_ranges_line = RawPanelObject(
            name="export_ranges_line",
            panel_object=pn.pane.HTML("<hr>", width=dwidth),
            viewer=self,
        )
        widgets.append(export_ranges_line)

        # Range description
        export_html_ranges_desc = StaticText(
            name="export_html_ranges_desc",
            display_name="Select Ranges",
            value="range dims for data exports.",
            height=15,
            width=dwidth,
            css_classes=["pyeyes-export-html-ranges-desc"],
            viewer=self,
        )
        widgets.append(export_html_ranges_desc)

        # Range inputs for each non-cat sdim
        sss_margin = 5
        s_width = (dwidth - 45) // 3
        for i, dim in enumerate(noncat_sdims):
            start = pn.widgets.IntInput(
                name=f"{dim}: start",
                value=0,
                width=s_width,
                css_classes=[
                    f"pyeyes-export-html-range-start-{sanitize_css_class(dim)}"
                ],
            )
            stop = pn.widgets.IntInput(
                name=f"{dim}: stop",
                value=self.slicer.dim_sizes[dim] - 1,
                width=s_width,
                css_classes=[
                    f"pyeyes-export-html-range-stop-{sanitize_css_class(dim)}"
                ],
            )
            step = pn.widgets.IntInput(
                name=f"{dim}: step",
                value=1,
                width=s_width,
                css_classes=[
                    f"pyeyes-export-html-range-step-{sanitize_css_class(dim)}"
                ],
            )
            dim_range = RawPanelObject(
                name=f"dim{i}_export_range",
                panel_object=pn.Row(
                    start,
                    stop,
                    step,
                    margin=sss_margin,
                    css_classes=[f"pyeyes-export-html-range-{sanitize_css_class(dim)}"],
                ),
                viewer=self,
            )
            widgets.append(dim_range)

        # HTML export separator
        export_html_line = RawPanelObject(
            name="export_html_line",
            panel_object=pn.pane.HTML("<hr>", width=dwidth),
            viewer=self,
        )
        widgets.append(export_html_line)

        # HTML export description
        export_html_desc = StaticText(
            name="export_html_desc",
            display_name="Save Static HTML",
            value="Save an interactive HTML. Interactively fast but memory \
                intensive. Exports for every dim along ranges specified above.",
            width=dwidth,
            css_classes=["pyeyes-export-html-line"],
            viewer=self,
        )
        widgets.append(export_html_desc)

        # HTML Save Path
        def export_html_path_callback(new_value):
            new_value = new_value.replace("\n", "")
            self.html_export_path = new_value

        export_html = TextAreaInput(
            name="export_html",
            display_name="HTML Save Path",
            value=self.html_export_path,
            placeholder="Enter path to export viewer as interactive html",
            height=box_height,
            css_classes=["pyeyes-export-html-path"],
            callback=export_html_path_callback,
            viewer=self,
        )
        widgets.append(export_html)

        # HTML Page Name
        def export_page_name_callback(new_value):
            self.html_export_page_name = new_value

        export_html_page_name = TextAreaInput(
            name="export_html_page_name",
            display_name="Exported HTML Page Name",
            value=self.html_export_page_name,
            placeholder="Enter a name for the page of the exported HTML",
            height=box_height,
            css_classes=["pyeyes-export-html-page-name"],
            callback=export_page_name_callback,
            viewer=self,
        )
        widgets.append(export_html_page_name)

        # Export HTML Button
        export_html_button = Button(
            name="export_html_button",
            display_name="Export HTML",
            button_type="primary",
            on_click=self._export_html,
            css_classes=["pyeyes-export-html-button"],
            viewer=self,
        )
        widgets.append(export_html_button)

        return widgets

    def _build_misc_widgets(self) -> List:
        """Build Misc-related widgets."""
        dwidth = pn.widgets.IntInput().width

        widgets = []

        def text_font_callback(new_value):
            with param.parameterized.discard_events(self.slicer):
                self.slicer.text_font = new_value
            self.slicer.param.trigger("text_font")

        text_font_input = Select(
            name="text_font",
            display_name="Text Font",
            options=themes.VALID_FONTS,
            value=self.slicer.text_font,
            css_classes=["pyeyes-text-font"],
            callback=text_font_callback,
            viewer=self,
        )
        widgets.append(text_font_input)

        def display_titles_checkbox_callback(new_value):
            with param.parameterized.discard_events(self.slicer):
                self.slicer.display_image_titles_visible = new_value
            self.slicer.param.trigger("display_image_titles_visible")

        # Display Image Titles
        display_titles_checkbox = Checkbox(
            name="display_titles_checkbox",
            display_name="Display Image Titles",
            value=self.slicer.display_image_titles_visible,
            css_classes=["pyeyes-display-titles-checkbox"],
            callback=display_titles_checkbox_callback,
            viewer=self,
        )
        widgets.append(display_titles_checkbox)

        def display_error_map_titles_checkbox_callback(new_value):
            with param.parameterized.discard_events(self.slicer):
                self.slicer.display_error_map_titles_visible = new_value
            self.slicer.param.trigger("display_error_map_titles_visible")

        display_error_map_titles_checkbox = Checkbox(
            name="display_error_map_titles_checkbox",
            display_name="Display Error Map Titles",
            value=self.slicer.display_error_map_titles_visible,
            css_classes=["pyeyes-display-error-map-titles-checkbox"],
            callback=display_error_map_titles_checkbox_callback,
            viewer=self,
        )
        widgets.append(display_error_map_titles_checkbox)

        # Display Grid
        def grid_visible_checkbox_callback(new_value):
            with param.parameterized.discard_events(self.slicer):
                self.slicer.grid_visible = new_value
            self.slicer.param.trigger("grid_visible")

        grid_visible_checkbox = Checkbox(
            name="grid_visible_checkbox",
            display_name="Display Grid",
            value=self.slicer.grid_visible,
            css_classes=["pyeyes-grid-visible-checkbox"],
            callback=grid_visible_checkbox_callback,
            viewer=self,
        )
        widgets.append(grid_visible_checkbox)

        misc_line = RawPanelObject(
            name="misc_line",
            panel_object=pn.pane.HTML("<hr>", width=dwidth),
            viewer=self,
        )
        widgets.append(misc_line)

        popup_desc = StaticText(
            name="popup_desc",
            display_name="Popup Pixel Inspection",
            value="Click on an image to inspect a pixel value.",
            width=dwidth,
            css_classes=["pyeyes-popup-desc"],
            viewer=self,
        )
        widgets.append(popup_desc)

        # enable popup
        def popup_pixel_enabled_callback(new_value):
            old_val = self.slicer.popup_pixel_enabled
            if old_val == new_value:
                return  # no chang

            if new_value:
                msg = "Enabling popup pixel inspection..."
            else:
                msg = "Disabling popup pixel inspection..."
            pn.state.notifications.info(msg, duration=1000)

            # update widget visibility
            with param.parameterized.discard_events(self.slicer):
                misc_pane = self.panes["Misc"]
                misc_pane.get_widget("show_popup_location_checkbox").visible = new_value
                misc_pane.get_widget("popup_on_error_maps_checkbox").visible = new_value
                misc_pane.get_widget("popup_loc").visible = new_value
                misc_pane.get_widget("clear_popup_button").visible = new_value

                # update slicer
                self.slicer.update_popup_pixel_enabled(new_value)

            self.slicer.param.trigger("popup_pixel_enabled")

            # pn.state.notifications.clear()
            if new_value:
                pn.state.notifications.success(
                    "Pixel inspection enabled. \
                    Click on a pixel in the left-most image to display \
                    that pixel's value for all displayed images.",
                    duration=5000,
                )
            else:
                pn.state.notifications.success(
                    "Pixel inspection turned off.",
                    duration=3000,
                )

        popup_pixel_enabled_checkbox = Checkbox(
            name="popup_pixel_enabled_checkbox",
            display_name="Enable Popup Pixel Inspection",
            value=self.slicer.popup_pixel_enabled,
            css_classes=["pyeyes-popup-pixel-enabled-checkbox"],
            callback=popup_pixel_enabled_callback,
            viewer=self,
        )
        widgets.append(popup_pixel_enabled_checkbox)

        # show popup location
        def show_popup_location_callback(new_value):
            with param.parameterized.discard_events(self.slicer):
                self.slicer.popup_pixel_show_location = new_value
            self.slicer.param.trigger("popup_pixel_show_location")

        show_popup_location_checkbox = Checkbox(
            name="show_popup_location_checkbox",
            display_name="Display Pixel Coordinates",
            value=self.slicer.popup_pixel_show_location,
            css_classes=["pyeyes-show-popup-location-checkbox"],
            callback=show_popup_location_callback,
            viewer=self,
        )
        show_popup_location_checkbox.visible = self.slicer.popup_pixel_enabled
        widgets.append(show_popup_location_checkbox)

        # enable popup on error maps
        def popup_on_error_maps_callback(new_value):
            with param.parameterized.discard_events(self.slicer):
                self.slicer.popup_pixel_on_error_maps = new_value
            self.slicer.param.trigger("popup_pixel_on_error_maps")

        popup_on_error_maps_checkbox = Checkbox(
            name="popup_on_error_maps_checkbox",
            display_name="Enable Popup on Error Maps",
            value=self.slicer.popup_pixel_on_error_maps,
            css_classes=["pyeyes-popup-on-error-maps-checkbox"],
            callback=popup_on_error_maps_callback,
            viewer=self,
        )
        popup_on_error_maps_checkbox.visible = self.slicer.popup_pixel_enabled
        widgets.append(popup_on_error_maps_checkbox)

        # popup location picker
        def popup_location_picker_callback(new_value):
            with param.parameterized.discard_events(self.slicer):
                self.slicer.popup_pixel_location = POPUP_LOCATION(new_value)
            self.slicer.param.trigger("popup_pixel_location")

        popup_location_picker = Select(
            name="popup_loc",
            display_name="Popup Location",
            options=[loc.value for loc in POPUP_LOCATION],
            value=self.slicer.popup_pixel_location.value,
            callback=popup_location_picker_callback,
            viewer=self,
        )
        popup_location_picker.visible = self.slicer.popup_pixel_enabled
        widgets.append(popup_location_picker)

        # clear popup
        def clear_popup_callback(event):
            self.slicer.clear_popup_pixel()

        clear_popup_button = Button(
            name="clear_popup_button",
            display_name="Clear Popup",
            button_type="primary",
            on_click=clear_popup_callback,
            css_classes=["pyeyes-clear-popup-button"],
            viewer=self,
        )
        clear_popup_button.visible = self.slicer.popup_pixel_enabled
        widgets.append(clear_popup_button)

        # Edit Display Image Titles

        misc_line2 = RawPanelObject(
            name="misc_line2",
            panel_object=pn.pane.HTML("<hr>", width=dwidth),
            viewer=self,
        )
        widgets.append(misc_line2)

        title_desc = StaticText(
            name="title_desc",
            display_name="Edit Titles",
            value="Edit titles of each image on viewer.",
            width=dwidth,
            css_classes=["pyeyes-title-desc"],
            viewer=self,
        )
        widgets.append(title_desc)

        # Add Text area input for each image title
        for i, img_name in enumerate(self.img_names):

            def make_title_callback(img_name):
                def callback(new_value):
                    self.slicer.display_image_titles[img_name] = new_value
                    self.slicer.param.trigger("display_image_titles")

                return callback

            act_img_name = self.slicer.display_image_titles[img_name]
            title_input = TextInput(
                name=f"replace_name_{i}",
                display_name=f'Rename Dataset "{img_name}"',
                value=act_img_name,
                css_classes=["pyeyes-export-config-path"],
                callback=make_title_callback(img_name),
                viewer=self,
            )
            widgets.append(title_input)

        return widgets

    @error.error_handler_decorator()
    def _export_config(self, event):
        """Export current viewer/slicer/ROI config to JSON at config_path."""
        if self.config_path is None:
            pn.state.notifications.warning("No path provided to export config.")
            return

        self._save_config(self.config_path)

        pn.state.notifications.success(f"Config saved to {self.config_path}")

    def _save_config(self, path: Union[Path, str]):
        """Write viewer, slicer, and ROI config to JSON file."""
        if isinstance(path, str):
            path = Path(path)

        exp_dir = os.path.dirname(path)

        if len(exp_dir) > 2:
            os.makedirs(exp_dir, exist_ok=True)

        viewer_config = config.serialize_parameters(self)
        slicer_config = config.serialize_parameters(self.slicer)
        if self.slicer.ROI is not None:
            roi_config = self.slicer.ROI.serialize()
        else:
            roi_config = None

        config_dict = {
            "viewer_config": viewer_config,
            "slicer_config": slicer_config,
            "roi_config": roi_config,
        }

        with open(path, "w") as f:
            json.dump(config_dict, f, indent=4, default=config.json_serial)

    def _export_html(self, event, step_override=None):
        """Export static HTML with sliders over slice dimensions."""
        if self.html_export_path is None:
            pn.state.notifications.warning("No path provided to export html to.")
            return

        # save current sdims
        curr_sdims = deepcopy(self.slicer.dim_indices)

        # save each webpage
        out_dir = os.path.dirname(self.html_export_path)
        os.makedirs(out_dir, exist_ok=True)

        # ignore cat dims
        sdims = [d for d in self.slicer.sdims if d not in self.cat_dims.keys()]
        pn.state.notifications.clear()
        if len(self.cat_dims) > 0:
            pn.state.notifications.warning(
                f"Attempting Save of HTML to {self.html_export_path}. \
                Categorial export support is limited... Exporting currently selected categories.",
                duration=0,
            )
        else:
            pn.state.notifications.info(
                f"Saving Static HTML to {self.html_export_path}...", duration=0
            )

        # Internal class for simple view of export
        class Exporter(param.Parameterized):

            dim_indices = param.Dict(
                default={}, doc="Mapping: dim_name -> int or categorical index"
            )

            def __init__(self, slicer: NDSlicer, dim_indices: dict, **params):
                super().__init__(**params)
                self.slicer = slicer
                self.dim_indices = dim_indices

            @param.depends("dim_indices")
            def view(self):
                with param.parameterized.discard_events(self.slicer):
                    for dim, val in self.dim_indices.items():
                        self.slicer.dim_indices[dim] = val
                    # return new view
                    return self.slicer.view()

        exporter = Exporter(self.slicer, self.slicer.dim_indices)

        export_sliders = {}
        embed_states = {}
        max_opts = 0
        for i, dim in enumerate(sdims):
            curr_dim = dim

            dim_range_widget = (
                self.panes["Export"].get_widget(f"dim{i}_export_range").get_widget()
            )
            start = dim_range_widget[0].value
            stop = dim_range_widget[1].value
            step = dim_range_widget[2].value
            if step_override is not None:
                step = step_override
            dim_range = list(range(start, stop + 1, step))
            max_opts = max(max_opts, len(dim_range))
            if start > stop:
                pn.state.notifications.clear()
                pn.state.notifications.error(
                    f"Start value must be less than stop value for {dim}."
                )
                return
            if stop > self.slicer.dim_sizes[curr_dim] - 1:
                pn.state.notifications.clear()
                pn.state.notifications.error(
                    f"Stop value must be less than {self.slicer.dim_sizes[curr_dim]} for {dim}."
                )
                return
            slider = pn.widgets.IntSlider(
                name=curr_dim,
                start=start,
                end=stop,
                step=step,
                value=start,
            )
            embed_states[slider] = dim_range

            def update_export_dim(event, this_dim=curr_dim):
                exporter.dim_indices[this_dim] = event.new
                exporter.param.trigger("dim_indices")

            slider.param.watch(update_export_dim, "value")
            export_sliders[curr_dim] = slider

        export_layout = pn.Row(pn.Column(*export_sliders.values()), exporter.view)

        export_panel = pn.panel(export_layout)
        export_panel.save(
            self.html_export_path,
            title=self.html_export_page_name,
            embed=True,
            max_opts=max_opts,
            resources="inline",
            embed_states=embed_states,
        )

        # # edit html to have black background
        if themes.VIEW_THEME in [themes.dark_theme, themes.dark_soft_theme]:
            with open(
                self.html_export_path, "r", encoding="utf-8", errors="ignore"
            ) as f:
                lines = f.readlines()
            lines[1] = '<html lang="en" style="background-color: black;">\n'
            with open(self.html_export_path, "w", encoding="utf-8") as f:
                f.writelines(lines)

        # restore slices
        for dim, val in curr_sdims.items():
            self.slicer.dim_indices[dim] = val
        self.slicer.param.trigger("dim_indices")

        pn.state.notifications.clear()
        pn.state.notifications.success("Static HTML saved!", duration=0)

    def export_reloadable_pyeyes(
        self,
        path: Union[Path, str],
        num_slices_to_keep: Union[int, Dict[str, int]] | None = None,
        subsampling: Union[int, Dict[str, int]] = 1,
        silent: bool = False,
    ):
        """
        Save .npz (optionally subsampled) and launcher script to reload viewer.

        Parameters
        ----------
        path : Union[Path, str]
            Path for launcher script (.py); .npz and .json written alongside.
        num_slices_to_keep : Union[int, Dict[str, int]] | None
            Slices per dimension (int or dict by dim). Overrides subsampling.
        subsampling : Union[int, Dict[str, int]]
            Subsampling step (int or dict by dim). Used if num_slices_to_keep is None.
        silent : bool
            If True, launcher uses show=False for testing.

        Example:
        --------
        ```
        # Keep 20 slices in 'z' dimension, 10 in 't' dimension
        viewer.export_reloadable_pyeyes("./viewer.py", num_slices_to_keep={"z": 20, "t": 10})

        # Use step size of 2 for all subsampleable dimensions
        viewer.export_reloadable_pyeyes("./viewer.py", subsampling=2)
        """

        if isinstance(path, str):
            path = Path(path)

        out_dir = path.parent
        os.makedirs(out_dir, exist_ok=True)

        # Determine which dimensions can be subsampled (exclude viewing dims and categorical dims)
        subsampleable_dims = [
            dim
            for dim in self.noncat_dims
            if dim not in [self.vdim_horiz, self.vdim_vert]
        ]

        # tmp update to dim_indices
        dim_indices_old = deepcopy(self.slicer.dim_indices)

        # Calculate step sizes for each dimension
        step_map = self._calculate_subsampling_steps(
            subsampleable_dims, num_slices_to_keep, subsampling
        )

        saved_data = {}

        slicers = []
        for dim in self.ndims:
            step = step_map.get(dim, 1)
            tslc = slice(None, None, step)
            slicers.append(tslc)
            if step > 1:
                inds = np.arange(self.slicer.dim_sizes[dim])[tslc]
                self.slicer.dim_indices[dim] = inds.shape[0] // 2

        for name, arr in self.raw_data.items():
            saved_data[name] = arr[tuple(slicers)]

        # Save data and config
        data_file = path.with_suffix(".npz")
        np.savez_compressed(data_file, **saved_data)
        self._save_config(path.with_suffix(".json"))

        self._create_launcher_script(path, silent=silent)

        # Make launcher executable
        os.chmod(path, 0o755)

        bokeh_root_logger.info(
            f"Export complete: script at {path}, data at {data_file}"
        )

        # restore dim_indices
        with param.parameterized.discard_events(self.slicer):
            self.slicer.dim_indices = dim_indices_old

    def _calculate_subsampling_steps(
        self,
        subsampleable_dims: List[str],
        num_slices_to_keep: Union[int, Dict[str, int]] | None,
        subsampling: Union[int, Dict[str, int]] = 1,
    ) -> Dict[str, int]:
        """Compute per-dimension step for subsampling or num_slices_to_keep."""
        step_map = {}
        data_shape = self.raw_data[list(self.raw_data.keys())[0]].shape

        if num_slices_to_keep is not None:
            for i, dim in enumerate(self.ndims):
                if dim not in subsampleable_dims:
                    continue

                if isinstance(num_slices_to_keep, int):
                    n_keep = num_slices_to_keep
                else:
                    n_keep = num_slices_to_keep.get(dim, data_shape[i])

                step_map[dim] = max(1, data_shape[i] // n_keep)
        else:
            if isinstance(subsampling, int):
                step_map = {dim: subsampling for dim in subsampleable_dims}
            else:
                step_map = {dim: subsampling.get(dim, 1) for dim in subsampleable_dims}

        return step_map

    def _create_launcher_script(self, path: Path, silent: bool = False):
        """Write standalone Python launcher script that loads npz and config."""
        # for debugging
        if silent:
            sstr = """
server = viewer.launch(show=False, threaded=True, verbose=False)
import time
time.sleep(0.5)
server.stop()"""
        else:
            sstr = "viewer.launch()"

        script = f"""#!/usr/bin/env python3
import numpy as np
import json
from pathlib import Path
from pyeyes.viewers import ComparativeViewer as cv

this = Path(__file__)
data = dict(np.load(this.with_suffix('.npz')))

config_path = this.with_suffix(".json")
cat_dims = {self.cat_dims!r}
named_dims = {self.ndims!r}
view_dims = {[self.vdim_horiz, self.vdim_vert]!r}

viewer = cv(
    data=data,
    named_dims=named_dims,
    view_dims=view_dims,
    cat_dims=cat_dims,
    config_path=config_path,
)
{sstr}
"""
        with open(path, "w") as f:
            f.write(script)


def spawn_comparative_viewer_detached(
    data,
    named_dims=None,
    view_dims=None,
    cat_dims=None,
    config_path=None,
    title="MRI Viewer",
):
    """
    Spawn ComparativeViewer in a separate Python subprocess.

    Parameters
    ----------
    data : dict of np.ndarray or np.ndarray
        Image data (single array or dict).
    named_dims, view_dims, cat_dims, config_path
        Passed to ComparativeViewer. Optional.
    title : str
        Browser tab title.

    Notes
    -----
    For safe removal of subprocesses, create a shell command like this:
    ```shell
    alias pyeyes_cleanup="pkill -9 -f '_PYEYES_SUBPROCESS.py' && rm -f /tmp/*_PYEYES_SUBPROCESS.*"
    ```
    Then, run `pyeyes_cleanup` to remove all subprocesses and temporary files.
    """
    # convert data to numpy in case its not, for pickling
    if not isinstance(data, dict):
        data = tonp(data)
    if isinstance(data, np.ndarray):
        data = {"Image": data}
    data = {k: tonp(v) for k, v in data.items()}

    tmp_data = tempfile.NamedTemporaryFile(
        suffix="_PYEYES_SUBPROCESS.pkl", delete=False
    )
    pickle.dump(data, tmp_data)
    tmp_data.close()

    if config_path is not None:
        config_path = str(config_path)

    # Build the script to load and serve the viewer
    script = f"""import panel as pn
import pickle
import os
from pyeyes.viewers import ComparativeViewer

# Load data
with open(r'{tmp_data.name}', 'rb') as f:
    data = pickle.load(f)

# Instantiate and serve
viewer = ComparativeViewer(
    data,
    named_dims={repr(named_dims)},
    view_dims={repr(view_dims)},
    cat_dims={repr(cat_dims)},
    config_path={repr(config_path)},
)
# print this process's PID
print(f"Detached Viewer Subprocess PID: {{os.getpid()}}")
pn.serve(viewer.app, title={repr(title)}, show=True)
"""

    # Write script to a temporary file and launch it
    tmp_script = tempfile.NamedTemporaryFile(
        suffix="_PYEYES_SUBPROCESS.py", delete=False, mode="w"
    )
    tmp_script.write(script)
    tmp_script.flush()
    tmp_script.close()
    subprocess.Popen([sys.executable, tmp_script.name])
    print(f"Launched detached viewer subprocess: {tmp_script.name}")
