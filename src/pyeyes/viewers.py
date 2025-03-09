import json
import os
import warnings
from typing import Dict, List, Optional, Sequence, Union

import bokeh
import holoviews as hv
import numpy as np
import panel as pn
import param
from holoviews import opts

from . import config, error, metrics, themes
from .cmap.cmap import VALID_COLORMAPS, VALID_ERROR_COLORMAPS
from .enums import METRICS_STATE, ROI_LOCATION, ROI_STATE, ROI_VIEW_MODE
from .slicers import NDSlicer
from .utils import tonp

hv.extension("bokeh")
pn.extension(notifications=True)

# Clear bokeh warning "dropping a patch..."
# tis a known issue, just a marker of slow rendering speed...
bokeh_root_logger = bokeh.logging.getLogger()
bokeh_root_logger.manager.disable = bokeh.logging.WARNING


class Viewer:

    def __init__(self, data, **kwargs):
        """
        Generic class for viewing image data. TODO: make generic more useful.

        Parameters:
        -----------
        data : dict[np.ndarray]
            Image data
        """
        self.data = data

    def launch(self):
        """
        Launch the viewer.
        """

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

    def __init__(
        self,
        data: Union[Dict[str, np.ndarray], np.ndarray],
        named_dims: Optional[Sequence[str]] = None,
        view_dims: Optional[Sequence[str]] = None,
        cat_dims: Optional[Dict[str, List]] = {},
        config_path: Optional[str] = None,
    ):
        """
        Viewer for comparing n-dimensional image data.

        Parameters:
        -----------
        data : Union[Dict[str, np.ndarray], np.ndarray]
            Single image or dictionary of images.
            If dictionary, keys should be strings which name each image,
            and values should be numpy arrays of equivalent size.
        named_dims : Optional[Sequence[str]]
            String name for each dimension of the image data, with the same ordering as
            the array dimensions. Highly recommended to provide this for easier use.
            If not provided, default is ["Dim 0", "Dim 1", ...].
        view_dims : Optional[Sequence[str]]
            Initial dimensions to view, should be a subset of dimension_names.
            Default is ['x', 'y'] if in dimension_names else first 2 dimensions.
        cat_dims : Optional[Dict[str, List]]
            Dictionary of categorical dimensions. Keys should be strings which name each
            categorical dimension, and values should be lists of strings which name each
            category for that dimension. Must be a subset of dimension_names.
        config_path : Optional[str]
            Path to a config file to load viewer settings from. If not provided, viewer
            will be initialized with default settings. Config should be generated from
            "Export Config" button in Export pane of the Viewer. Limited set of settings
            are also transferrable to different datasets or different shapes.
        """

        from_config = config_path is not None and os.path.exists(config_path)

        # Defaults
        if not isinstance(data, dict):
            data = tonp(data)

        if isinstance(data, np.ndarray):
            data = {"Image": data}

        data = {k: tonp(v) for k, v in data.items()}

        if named_dims is None or len(named_dims) == 0:
            named_dims = [f"Dim {i}" for i in range(data["Image"].ndim)]

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
        self.noncat_dims = [dim for dim in named_dims if dim not in cat_dims.keys()]

        assert np.array(
            [img.shape == img_list[0].shape for img in img_list]
        ).all(), "All viewed data must have the same input shape."
        assert (
            N_dim == img_list[0].ndim
        ), "Number of dimension names must match the number of dimensions in the data."

        if view_dims is not None:
            assert all(
                [dim in self.noncat_dims for dim in view_dims]
            ), "All view dimensions must be in dimension_names."
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

        # Instantiate slicer
        self.slicer = NDSlicer(
            self.dataset,
            self.vdims,
            cdim="ImgName",
            clabs=img_names,
            cat_dims=cat_dims,
            cfg=cfg,
        )

        # Attach watcher variables to slicer attributes that need GUI updates
        self.slicer.param.watch(self._roi_state_watcher, "roi_state")

        """
        Create Panel Layout
        """
        # Widgets per pane
        view_pane_widgets = self._init_view_pane_widgets()
        contrast_pane_widgets = self._init_contrast_pane_widgets()
        roi_pane_widgets = self._init_roi_pane_widgets()
        analysis_pane_widgets = self._init_analysis_pane_widgets()
        export_pane_widgets = self._init_export_pane_widgets()

        def keys_to_index_dict(d):
            return dict(zip((list(d.keys())), range(len(d))))

        # image of control panel through nested dict
        self.panel_tab_index_dict = {
            "View": [0, keys_to_index_dict(view_pane_widgets)],
            "Contrast": [1, keys_to_index_dict(contrast_pane_widgets)],
            "ROI": [2, keys_to_index_dict(roi_pane_widgets)],
            "Analysis": [3, keys_to_index_dict(analysis_pane_widgets)],
            "Export": [4, keys_to_index_dict(export_pane_widgets)],
        }

        # Build Control Panel
        control_panel = pn.Tabs(
            (
                "View",
                pn.Column(*list(view_pane_widgets.values())),
            ),
            ("Contrast", pn.Column(*list(contrast_pane_widgets.values()))),
            ("ROI", pn.Column(*list(roi_pane_widgets.values()))),
            ("Analysis", pn.Column(*list(analysis_pane_widgets.values()))),
            ("Export", pn.Column(*list(export_pane_widgets.values()))),
        )

        # App
        self.app = pn.Row(control_panel, self.slicer.view)

        # make sure roi_state is consistent with widgets
        if from_config:
            self._roi_state_watcher(self.slicer.roi_state)
            self._update_error_map_type(self.slicer.error_map_type, autoformat=False)
        else:
            self._autoscale_clim(event=None)

    def launch(self):
        """
        Launch the viewer.
        """

        pn.serve(self.app, title="MRI Viewer", show=True)

    def load_from_config(self, config_path: str):
        """
        Load config from json dict.
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

    def _get_app_widget(self, tab_name, widget_name):
        """
        Get a widget from the app by tab name and widget name.
        """

        tab_index, widget_dict = self.panel_tab_index_dict[tab_name]

        return self.app[0][tab_index][widget_dict[widget_name]]

    def _set_app_widget(self, tab_name, widget_name, new_widget):
        """
        Replace a widget from the app by tab name and widget name.
        """

        tab_index, widget_dict = self.panel_tab_index_dict[tab_name]

        self.app[0][tab_index][widget_dict[widget_name]] = new_widget

    def _set_app_widget_attr(self, tab_name, widget_name, attr_name, new_attr):
        """
        Set an attribute of a widget from the app by tab name and widget name.
        """

        tab_index, widget_dict = self.panel_tab_index_dict[tab_name]

        setattr(self.app[0][tab_index][widget_dict[widget_name]], attr_name, new_attr)

    """
    Build Widgets for each tab
    """

    def _init_view_pane_widgets(self) -> Dict[str, pn.widgets.Widget]:
        """
        Returns a dictionary of widgets belonging to the "View" pane
        """

        widgets = {}

        # Viewing Dimensions
        widgets.update(self._build_vdim_widgets())

        # Widgets for Slicing Dimensions
        widgets.update(self._build_sdim_widgets())

        # Other widgets on the viewing page
        widgets.update(self._build_viewing_widgets())

        # Single Toggle View widgets
        widgets["single_toggle"] = self._build_single_toggle_widget()

        widgets["im_display_desc"] = pn.widgets.StaticText(
            name="Displayed Images",
            value="Click names to toggle visibility.",
        )

        widgets["im_display"] = self._build_display_images_widget()

        return widgets

    def _init_contrast_pane_widgets(self) -> Dict[str, pn.widgets.Widget]:
        """
        Returns a dictionary of widgets belonging to the "Contrast" pane
        """

        widgets = {}

        cplx_widget, clim_scale_widget = self._build_cplx_widget()

        # Select which real-valued view of complex data to view
        widgets["cplx_view"] = cplx_widget

        # Color map stuff
        widgets.update(self._build_contrast_widgets())

        # Auto-scaling button
        widgets["autoscale"] = clim_scale_widget

        return widgets

    def _init_roi_pane_widgets(self) -> Dict[str, pn.widgets.Widget]:
        """
        Returns a dictionary of widgets belonging to the "ROI" pane
        """

        widgets = {}

        widgets.update(self._build_roi_widgets())

        return widgets

    def _init_analysis_pane_widgets(self) -> Dict[str, pn.widgets.Widget]:
        """
        Returns a dictionary of widgets belonging to the "Analysis" pane
        """

        widgets = {}

        widgets.update(self._build_analysis_widgets())

        return widgets

    def _init_export_pane_widgets(self) -> Dict[str, pn.widgets.Widget]:
        """
        Returns a dictionary of widgets belonging to the "Export" pane
        """

        widgets = {}

        widgets.update(self._build_export_widgets())

        return widgets

    def _build_dataset(self, vdims: Sequence[str]) -> hv.Dataset:
        """
        Build the dataset for a specific set of viewing dimensions.

        Convert from dictionary of np arrays to a HoloViews Dataset.
        """

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

    def _build_vdim_widgets(self) -> Dict[str, pn.widgets.Widget]:
        """
        Build selection widgets for 2 viewing dimensions.
        """

        vdim_horiz_widget = pn.widgets.Select(
            name="L/R Viewing Dimension", options=self.noncat_dims, value=self.vdims[0]
        )

        @error.error_handler_decorator()
        def vdim_horiz_callback(event):
            if event.new != event.old:
                vh_new = event.new
                vv_new = event.old if (vh_new == self.vdim_vert) else self.vdim_vert
                self._update_vdims([vh_new, vv_new])

        vdim_horiz_widget.param.watch(vdim_horiz_callback, "value")

        vdim_vert_widget = pn.widgets.Select(
            name="U/D Viewing Dimension", options=self.noncat_dims, value=self.vdims[1]
        )

        @error.error_handler_decorator()
        def vdim_vert_callback(event):
            if event.new != event.old:
                vv_new = event.new
                vh_new = event.old if (vv_new == self.vdim_horiz) else self.vdim_horiz
                self._update_vdims([vh_new, vv_new])

        vdim_vert_widget.param.watch(vdim_vert_callback, "value")

        return {"vdim_horiz": vdim_horiz_widget, "vdim_vert": vdim_vert_widget}

    def _update_vdims(self, new_vdims):
        """
        Routine to run to update the viewing dimensions of the data.
        """

        assert len(new_vdims) == 2, "Must provide exactly 2 viewing dimensions."

        with param.parameterized.discard_events(self.slicer):

            # Update attributes
            self.vdims = new_vdims
            self.vdim_horiz = new_vdims[0]
            self.vdim_vert = new_vdims[1]

            # Update vdim widgets
            self._set_app_widget_attr("View", "vdim_horiz", "value", new_vdims[0])
            self._set_app_widget_attr("View", "vdim_vert", "value", new_vdims[1])

            # Update Slicer
            old_vdims = self.slicer.vdims
            self.slicer._set_volatile_dims(new_vdims)

            # Update displayed widgets if interchange of vdim and sdims
            if set(old_vdims) != set(new_vdims):
                new_sdim_widget_dict = self._build_sdim_widgets()

                for i, w in enumerate(list(new_sdim_widget_dict.keys())):
                    self._set_app_widget("View", f"sdim{i}", new_sdim_widget_dict[w])

            # Reset crops
            self._set_app_widget_attr(
                "View", "lr_crop", "bounds", (0, self.slicer.img_dims[0])
            )
            self._set_app_widget_attr(
                "View", "ud_crop", "bounds", (0, self.slicer.img_dims[1])
            )
            self._set_app_widget_attr("View", "lr_crop", "value", self.slicer.lr_crop)
            self._set_app_widget_attr("View", "ud_crop", "value", self.slicer.ud_crop)

        self.slicer.param.trigger("lr_crop", "ud_crop")

    def _build_sdim_widgets(self) -> dict:
        """
        Return a dictionary of panel widgets to interactively control slicing.
        """

        sliders = {}
        for i, dim in enumerate(self.slicer.sdims):
            if dim in self.cat_dims.keys():
                s = pn.widgets.Select(
                    name=dim,
                    options=self.cat_dims[dim],
                    value=self.slicer.dim_indices[dim],
                )
            else:
                if self.slicer.dim_sizes[dim] - 1 > 0:
                    s = pn.widgets.EditableIntSlider(
                        name=dim,
                        start=0,
                        end=self.slicer.dim_sizes[dim] - 1,
                        value=self.slicer.dim_indices[dim],
                    )
                else:
                    print(f"Detected '{dim}' is singleton. Cannot slice.")
                    continue

            def _update_dim_indices(event, this_dim=dim):

                self._update_sdim(this_dim, event.new)

            s.param.watch(_update_dim_indices, "value")

            sliders[f"sdim{i}"] = s

        return sliders

    @error.error_handler_decorator()
    def _update_sdim(self, sdim, new_dim_val):
        """
        Callback to update a specific slicing dimension.
        """

        with param.parameterized.discard_events(self.slicer):

            self.slicer.dim_indices[sdim] = new_dim_val

            # Assume we need to autoscale if dimension updated is categorical
            if sdim in self.cat_dims.keys():
                self.slicer.update_colormap()
                self._autoscale_clim(event=None)

        # Trigger
        if sdim in self.cat_dims.keys():
            self.slicer.param.trigger("cmap")
        else:
            self.slicer.param.trigger("dim_indices")

    def _build_viewing_widgets(self):
        """
        Return a dictionary of panel widgets to interactively control viewing.
        """

        sliders = {}

        # Flip Widgets
        ud_w = pn.widgets.Checkbox(name="Flip Image Up/Down", value=self.slicer.flip_ud)

        @error.error_handler_decorator()
        def flip_ud_callback(event):
            if event.new != event.old:
                self.slicer.flip_ud = event.new

        ud_w.param.watch(flip_ud_callback, "value")
        sliders["flip_ud"] = ud_w

        lr_w = pn.widgets.Checkbox(
            name="Flip Image Left/Right", value=self.slicer.flip_lr
        )

        @error.error_handler_decorator()
        def flip_lr_callback(event):
            if event.new != event.old:
                self.slicer.flip_lr = event.new

        lr_w.param.watch(flip_lr_callback, "value")
        sliders["flip_lr"] = lr_w

        size_scale_widget = pn.widgets.EditableIntSlider(
            name="Size Scale",
            start=self.slicer.param.size_scale.bounds[0],
            end=self.slicer.param.size_scale.bounds[1],
            value=self.slicer.size_scale,
            step=self.slicer.param.size_scale.step,
        )

        @error.error_handler_decorator()
        def size_scale_callback(event):
            self.slicer.size_scale = event.new

        size_scale_widget.param.watch(size_scale_callback, "value")
        sliders["size_scale"] = size_scale_widget

        # Title font size
        title_font_input = pn.widgets.EditableIntSlider(
            name="Title Font Size",
            start=self.slicer.param.title_font_size.bounds[0],
            end=self.slicer.param.title_font_size.bounds[1],
            value=self.slicer.title_font_size,
            step=self.slicer.param.title_font_size.step,
        )

        def _update_title_font_size(event):
            self.slicer.title_font_size = event.new

        title_font_input.param.watch(_update_title_font_size, "value")
        sliders["title_font_size"] = title_font_input

        # bounding box crop for each L/R/U/D edge
        lr_crop_slider = pn.widgets.IntRangeSlider(
            name="L/R Display Range",
            start=self.slicer.param.lr_crop.bounds[0],
            end=self.slicer.param.lr_crop.bounds[1],
            value=(self.slicer.lr_crop[0], self.slicer.lr_crop[1]),
            step=self.slicer.param.lr_crop.step,
        )

        def _update_lr_slider(event):
            self.slicer.lr_crop = event.new

        lr_crop_slider.param.watch(_update_lr_slider, "value")
        sliders["lr_crop"] = lr_crop_slider

        ud_crop_slider = pn.widgets.IntRangeSlider(
            name="U/D Display Range",
            start=self.slicer.param.ud_crop.bounds[0],
            end=self.slicer.param.ud_crop.bounds[1],
            value=(self.slicer.ud_crop[0], self.slicer.ud_crop[1]),
            step=self.slicer.param.ud_crop.step,
        )

        def _update_ud_slider(event):
            self.slicer.ud_crop = event.new

        ud_crop_slider.param.watch(_update_ud_slider, "value")
        sliders["ud_crop"] = ud_crop_slider

        return sliders

    def _build_single_toggle_widget(self):
        # Single toggle view
        single_toggle = pn.widgets.Checkbox(
            name="Single View", value=self.single_image_toggle
        )

        @error.error_handler_decorator()
        def single_toggle_callback(event):
            if event.new != event.old:
                self._update_toggle_single_view(event.new)

        single_toggle.param.watch(single_toggle_callback, "value")

        return single_toggle

    def _update_toggle_single_view(self, new_single_toggle):
        """
        Only display one image at a time for toggling sake
        """

        self.single_image_toggle = new_single_toggle

        # build new widget
        new_display_images_widget = self._build_display_images_widget()

        # update gui
        self._set_app_widget("View", "im_display", new_display_images_widget)

        if self.single_image_toggle:
            self.display_images = [self.display_images[0]]
        else:
            self.display_images = self.img_names

        # send new display images to slicer
        self.slicer.update_display_image_list(self.display_images)

    def _build_display_images_widget(self):

        if self.single_image_toggle:

            display_images_widget = pn.widgets.RadioButtonGroup(
                name="Displayed Images",
                options=self.img_names,
                value=self.img_names[0],
                button_type="success",
                button_style="outline",
            )

        else:
            display_images_widget = pn.widgets.CheckButtonGroup(
                name="Displayed Images",
                options=self.img_names,
                value=self.img_names,
                button_type="primary",
                button_style="outline",
            )

        @error.error_handler_decorator()
        def display_images_callback(event):
            if not isinstance(event.new, str) and len(event.new) == 0:
                pn.state.notifications.warning("Must select at least one image.")
                self._set_app_widget_attr("View", "im_display", "value", event.old)
                return
            if event.new != event.old:
                self._update_image_display(event.new)

        display_images_widget.param.watch(display_images_callback, "value")

        return display_images_widget

    def _update_image_display(self, new_display_images):
        """
        Update which image to display based on if single view is toggled or not.
        """

        if isinstance(new_display_images, str):
            new_display_images = [new_display_images]

        self.display_images = new_display_images

        self.slicer.update_display_image_list(new_display_images)

    def _build_cplx_widget(self):

        cplx_widget = pn.widgets.RadioButtonGroup(
            name="Complex Data",
            options=(
                ["mag", "phase", "real", "imag"]
                if self.is_complex_data
                else ["mag", "real"]
            ),
            value=self.slicer.cplx_view,
            button_type="primary",
            button_style="outline",
        )

        @error.error_handler_decorator()
        def cplx_callback(event):
            if event.new != event.old:
                self._update_cplx_view(event.new)

        cplx_widget.param.watch(cplx_callback, "value")

        # Auto-scale for given slice
        clim_scale_widget = pn.widgets.Button(name="Auto-Scale", button_type="primary")
        clim_scale_widget.on_click(self._autoscale_clim)

        return cplx_widget, clim_scale_widget

    def _update_cplx_view(self, new_cplx_view):
        """
        Routine to run to update the viewing dimensions of the data.
        """

        # Update Slicer
        with param.parameterized.discard_events(self.slicer):
            self.slicer.update_cplx_view(new_cplx_view)

            # Reset clim
            self._set_app_widget_attr(
                "Contrast", "clim", "start", self.slicer.param.vmin.bounds[0]
            )
            self._set_app_widget_attr(
                "Contrast", "clim", "end", self.slicer.param.vmax.bounds[1]
            )
            self._set_app_widget_attr(
                "Contrast", "clim", "value", (self.slicer.vmin, self.slicer.vmax)
            )
            self._set_app_widget_attr(
                "Contrast", "clim", "step", self.slicer.param.vmin.step
            )

            # Reset cmap value
            self._set_app_widget_attr("Contrast", "cmap", "value", self.slicer.cmap)

        self.slicer.param.trigger("vmin", "vmax", "cmap")

    def _build_contrast_widgets(self) -> Dict[str, pn.widgets.Widget]:

        widgets = {}

        # vmin/vmax use different Range slider
        range_slider = pn.widgets.EditableRangeSlider(
            name="clim",
            start=self.slicer.param.vmin.bounds[0],
            end=self.slicer.param.vmax.bounds[1],
            value=(self.slicer.vmin, self.slicer.vmax),
            step=self.slicer.param.vmin.step,
        )

        def _update_clim(event):
            with param.parameterized.discard_events(self.slicer):
                self.slicer.vmin, self.slicer.vmax = event.new
            self.slicer.param.trigger("vmin", "vmax")

        range_slider.param.watch(_update_clim, "value")
        widgets["clim"] = range_slider

        # Colormap
        cmap_widget = pn.widgets.Select(
            name="Color Map",
            options=VALID_COLORMAPS,
            value=self.slicer.cmap,
        )

        def _update_cmap(event):

            with param.parameterized.discard_events(self.slicer):
                self.slicer.cmap = event.new
                self.slicer.update_colormap()

            self.slicer.param.trigger("cmap")

        cmap_widget.param.watch(_update_cmap, "value")
        widgets["cmap"] = cmap_widget

        # Colorbar toggle
        colorbar_widget = pn.widgets.Checkbox(
            name="Add Colorbar",
            value=self.slicer.colorbar_on,
        )

        def _update_colorbar(event):
            self.slicer.colorbar_on = event.new

        colorbar_widget.param.watch(_update_colorbar, "value")
        widgets["colorbar"] = colorbar_widget

        colorbar_label_widget = pn.widgets.TextInput(
            name="Colorbar Label",
            value=self.slicer.colorbar_label,
        )

        def _update_colorbar_label(event):
            self.slicer.colorbar_label = event.new

        colorbar_label_widget.param.watch(_update_colorbar_label, "value")

        # disable colorbar label if colorbar is off
        def _update_colorbar_label_disabled(x):
            colorbar_label_widget.disabled = not x

        pn.bind(_update_colorbar_label_disabled, colorbar_widget, watch=True)

        widgets["colorbar_label"] = colorbar_label_widget

        return widgets

    def _autoscale_clim(self, event):
        """
        Routine to run to update the viewing dimensions of the data.
        """

        # Update Slicer
        with param.parameterized.discard_events(self.slicer):
            self.slicer.autoscale_clim()

            # Update gui
            self._set_app_widget_attr(
                "Contrast", "clim", "value", (self.slicer.vmin, self.slicer.vmax)
            )

        self.slicer.param.trigger("vmin", "vmax")

    def _build_roi_widgets(self) -> Dict[str, pn.widgets.Widget]:

        widgets = {}

        # Option for ROI in-figure
        roi_mode = pn.widgets.Checkbox(
            name="ROI Overlay Enabled", value=bool(self.slicer.roi_mode.value)
        )

        def _update_roi_mode(event):
            self.slicer.update_roi_mode(event.new)

        roi_mode.param.watch(_update_roi_mode, "value")
        widgets["roi_mode"] = roi_mode

        # ROI Button
        roi_button = pn.widgets.Button(name="Draw ROI", button_type="primary")

        def _draw_roi(event):
            self.slicer.update_roi_state(ROI_STATE.FIRST_SELECTION)

        roi_button.on_click(_draw_roi)
        widgets["draw_roi"] = roi_button

        # Clear Button
        clear_button = pn.widgets.Button(name="Clear ROI", button_type="warning")

        def _clear_roi(event):
            self.slicer.update_roi_state(ROI_STATE.INACTIVE)

        # Enable only if roi is drawn
        clear_button.on_click(_clear_roi)
        clear_button.disabled = True

        widgets["clear_roi"] = clear_button

        # Colormap
        roi_cmap_widget = pn.widgets.Select(
            name="ROI Color Map",
            options=self.slicer.param.roi_cmap.objects,
            value=self.slicer.roi_cmap,
        )

        def _update_cmap(event):
            self.slicer.update_roi_colormap(event.new)

        roi_cmap_widget.param.watch(_update_cmap, "value")
        widgets["roi_cmap"] = roi_cmap_widget

        # Zoom Scale
        zoom_scale_widget = pn.widgets.EditableFloatSlider(
            name="Zoom Scale",
            start=1.0,
            end=10.0,
            value=self.slicer.ROI.zoom_scale,
            step=0.1,
        )

        def _update_zoom_scale(event):
            self.slicer.update_roi_zoom_scale(event.new)

        zoom_scale_widget.param.watch(_update_zoom_scale, "value")
        widgets["zoom_scale"] = zoom_scale_widget

        # Location
        roi_loc_widget = pn.widgets.Select(
            name="ROI Location",
            options=[loc.value for loc in ROI_LOCATION],
            value=self.slicer.ROI.roi_loc.value,
        )

        def _update_roi_loc(event):
            self.slicer.update_roi_loc(event.new)

        roi_loc_widget.param.watch(_update_roi_loc, "value")
        widgets["roi_loc"] = roi_loc_widget

        # add widget to adjust roi location
        roi_lr_crop_slider = pn.widgets.EditableRangeSlider(
            name="ROI L/R Crop",
            value=(0, 1),
            start=0,
            end=100000,
            step=0.1,
        )

        def _update_roi_lr_crop(event):
            crop_lower, crop_upper = event.new
            self.slicer.update_roi_lr_crop((crop_lower, crop_upper))

        roi_lr_crop_slider.param.watch(_update_roi_lr_crop, "value")
        widgets["roi_lr_crop"] = roi_lr_crop_slider

        roi_ud_crop_slider = pn.widgets.EditableRangeSlider(
            name="ROI U/D Crop",
            value=(0, 1),
            start=0,
            end=100000,
            step=0.1,
        )

        def _update_roi_ud_crop(event):
            crop_lower, crop_upper = event.new
            self.slicer.update_roi_ud_crop((crop_lower, crop_upper))

        roi_ud_crop_slider.param.watch(_update_roi_ud_crop, "value")
        widgets["roi_ud_crop"] = roi_ud_crop_slider

        # Bounding box details
        roi_line_color = pn.widgets.ColorPicker(
            name="ROI Line Color",
            value=self.slicer.ROI.color,
        )

        def _update_roi_line_color(event):
            self.slicer.update_roi_line_color(event.new)

        roi_line_color.param.watch(_update_roi_line_color, "value")
        widgets["roi_line_color"] = roi_line_color

        roi_line_width = pn.widgets.IntSlider(
            name="ROI Line Width",
            start=1,
            end=10,
            value=self.slicer.ROI.line_width,
        )

        def _update_roi_line_width(event):
            self.slicer.update_roi_line_width(event.new)

        roi_line_width.param.watch(_update_roi_line_width, "value")
        widgets["roi_line_width"] = roi_line_width

        roi_zoom_order = pn.widgets.IntInput(
            name="ROI Zoom Order",
            value=self.slicer.ROI.zoom_order,
            start=0,
            end=3,
        )

        def _update_roi_zoom_order(event):
            self.slicer.update_roi_zoom_order(event.new)

        roi_zoom_order.param.watch(_update_roi_zoom_order, "value")
        widgets["roi_zoom_order"] = roi_zoom_order

        # Default not visible
        widgets["roi_cmap"].visible = False
        widgets["zoom_scale"].visible = False
        widgets["roi_loc"].visible = False
        widgets["roi_lr_crop"].visible = False
        widgets["roi_ud_crop"].visible = False
        widgets["roi_line_color"].visible = False
        widgets["roi_line_width"].visible = False
        widgets["roi_zoom_order"].visible = False

        return widgets

    def _roi_state_watcher(self, event):

        if hasattr(event, "new"):
            new_state = event.new
        else:
            new_state = event

        # Clear button enabled or not
        self._set_app_widget_attr(
            "ROI", "clear_roi", "disabled", new_state == ROI_STATE.INACTIVE
        )

        with param.parameterized.discard_events(self.slicer):
            # update ranges of sliders upon completion of ROI
            if new_state == ROI_STATE.ACTIVE:

                roi_l, roi_b, roi_r, roi_t = self.slicer.ROI.lbrt()

                lr_max = self.slicer.lr_crop
                ud_max = self.slicer.ud_crop

                lr_step = self.slicer.param.lr_crop.step
                ud_step = self.slicer.param.ud_crop.step

                self._set_app_widget_attr("ROI", "roi_lr_crop", "start", lr_max[0])
                self._set_app_widget_attr("ROI", "roi_lr_crop", "end", lr_max[1])
                self._set_app_widget_attr("ROI", "roi_lr_crop", "value", (roi_l, roi_r))
                self._set_app_widget_attr("ROI", "roi_lr_crop", "step", lr_step)

                self._set_app_widget_attr("ROI", "roi_ud_crop", "start", ud_max[0])
                self._set_app_widget_attr("ROI", "roi_ud_crop", "end", ud_max[1])
                self._set_app_widget_attr("ROI", "roi_ud_crop", "value", (roi_b, roi_t))
                self._set_app_widget_attr("ROI", "roi_ud_crop", "step", ud_step)

                # Constrain Zoom scale
                max_lr_zoom = abs(lr_max[1] - lr_max[0]) / abs(roi_r - roi_l)
                max_ud_zoom = abs(ud_max[1] - ud_max[0]) / abs(roi_t - roi_b)
                max_zoom = round(min(max_lr_zoom, max_ud_zoom), 1)

                self._set_app_widget_attr("ROI", "zoom_scale", "start", 1.0)
                self._set_app_widget_attr("ROI", "zoom_scale", "end", max_zoom)

            active = new_state == ROI_STATE.ACTIVE
            self._set_app_widget_attr("ROI", "roi_cmap", "visible", active)
            self._set_app_widget_attr("ROI", "zoom_scale", "visible", active)
            self._set_app_widget_attr("ROI", "roi_loc", "visible", active)
            self._set_app_widget_attr("ROI", "roi_lr_crop", "visible", active)
            self._set_app_widget_attr("ROI", "roi_ud_crop", "visible", active)
            self._set_app_widget_attr("ROI", "roi_line_color", "visible", active)
            self._set_app_widget_attr("ROI", "roi_line_width", "visible", active)
            self._set_app_widget_attr("ROI", "roi_zoom_order", "visible", active)

            # Enable certain widgets only if roi_mode = infigure
            is_sep = self.slicer.roi_mode == ROI_VIEW_MODE.Separate
            self._set_app_widget_attr("ROI", "zoom_scale", "disabled", is_sep)
            self._set_app_widget_attr("ROI", "roi_loc", "disabled", is_sep)

    def _build_analysis_widgets(self) -> Dict[str, pn.widgets.Widget]:

        widgets = {}

        # Difference Map and Metrics Widgets
        reference_widget = pn.widgets.Select(
            name="Reference Dataset",
            options=self.img_names,
            value=self.slicer.metrics_reference,
        )

        def _update_reference(event):
            self.slicer.update_reference_dataset(event.new)

        reference_widget.param.watch(_update_reference, "value")
        widgets["reference"] = reference_widget

        diff_map_type_widget = pn.widgets.Select(
            name="Error Map Type",
            options=metrics.MAPPABLE_METRICS,
            value=self.slicer.error_map_type,
        )
        diff_map_type_widget.param.watch(self._update_error_map_type, "value")
        widgets["diff_map_type"] = diff_map_type_widget

        text_description_widget = pn.widgets.StaticText(
            name="Metrics", value="Select metrics to display in text."
        )
        widgets["text_description"] = text_description_widget

        text_metrics_widget = pn.widgets.CheckBoxGroup(
            name="Metrics",
            options=metrics.FULL_METRICS,
            value=self.slicer.metrics_text_types,
        )

        def _update_text_metrics(event):
            self.slicer.update_metrics_text_types(event.new)

        text_metrics_widget.param.watch(_update_text_metrics, "value")
        widgets["text_metrics"] = text_metrics_widget

        text_description_widget = pn.widgets.StaticText(
            name="Enable", value="Click to add 'Error map' or 'text metrics'."
        )
        widgets["button_description"] = text_description_widget

        # determine value
        if self.slicer.metrics_state == METRICS_STATE.ALL:
            display_options_value = ["Error Map", "Text"]
        elif self.slicer.metrics_state == METRICS_STATE.MAP:
            display_options_value = ["Error Map"]
        elif self.slicer.metrics_state == METRICS_STATE.TEXT:
            display_options_value = ["Text"]
        else:
            display_options_value = []

        display_options = pn.widgets.CheckButtonGroup(
            name="Display Metrics",
            button_type="primary",
            button_style="outline",
            value=display_options_value,
            options=["Error Map", "Text"],
        )

        def _update_displayed_metrics(event):
            pn.state.notifications.clear()
            pn.state.notifications.info("Building...", duration=0)

            if ("Error Map" in event.new) and ("Text" in event.new):
                new_state = METRICS_STATE.ALL
            elif "Error Map" in event.new:
                new_state = METRICS_STATE.MAP
            elif "Text" in event.new:
                new_state = METRICS_STATE.TEXT
            else:
                new_state = METRICS_STATE.INACTIVE

            self.slicer.update_metrics_state(new_state)

            pn.state.notifications.clear()
            pn.state.notifications.info("Done!", duration=1000)

        display_options.param.watch(_update_displayed_metrics, "value")
        widgets["display_options"] = display_options

        error_scale_widget = pn.widgets.EditableFloatSlider(
            name="Error Scale",
            start=1.0,
            end=50.0,
            value=self.slicer.error_map_scale,
            step=0.1,
        )

        def _update_error_scale(event):
            self.slicer.update_error_map_scale(event.new)

        error_scale_widget.param.watch(_update_error_scale, "value")
        widgets["error_scale"] = error_scale_widget

        error_cmap_widget = pn.widgets.Select(
            name="Error Map Color Map",
            options=VALID_ERROR_COLORMAPS,
            value=self.slicer.error_map_cmap,
        )

        def _update_error_cmap(event):
            self.slicer.update_error_map_cmap(event.new)

        error_cmap_widget.param.watch(_update_error_cmap, "value")
        widgets["error_cmap"] = error_cmap_widget

        metrics_text_font_size_widget = pn.widgets.EditableIntSlider(
            name="Text Metrics Font Size",
            start=5,
            end=24,
            value=self.slicer.metrics_text_font_size,
            step=1,
        )

        def _update_metrics_text_font_size(event):
            self.slicer.update_metrics_text_font_size(event.new)

        metrics_text_font_size_widget.param.watch(
            _update_metrics_text_font_size, "value"
        )
        widgets["metrics_text_font_size"] = metrics_text_font_size_widget

        metrics_text_font_loc_widget = pn.widgets.Select(
            name="Text Metrics Location",
            options=[loc.value for loc in ROI_LOCATION],
            value=self.slicer.metrics_text_location.value,
        )

        def _update_metrics_text_loc(event):
            self.slicer.update_metrics_text_location(ROI_LOCATION(event.new))

        metrics_text_font_loc_widget.param.watch(_update_metrics_text_loc, "value")
        widgets["metrics_text_font_loc"] = metrics_text_font_loc_widget

        error_map_autoformat = pn.widgets.Button(
            name="Autoformat Error Map",
            button_type="primary",
        )
        error_map_autoformat.on_click(self._autoformat_error_map)
        widgets["error_map_autoformat"] = error_map_autoformat

        return widgets

    def _update_error_map_type(self, event, autoformat=True):
        """
        Make sure error map gets the right formatting.
        """

        if hasattr(event, "new"):
            new_type = event.new
        else:
            new_type = event

        with param.parameterized.discard_events(self.slicer):
            self.slicer.update_error_map_type(new_type)

        self._set_app_widget_attr(
            "Analysis", "error_scale", "visible", new_type != "SSIM"
        )

        if self.slicer.metrics_state != METRICS_STATE.INACTIVE and autoformat:
            with param.parameterized.discard_events(self.slicer):
                self._autoformat_error_map(None)
            self.slicer.param.trigger("metrics_state")

    def _autoformat_error_map(self, event):
        """
        Autoformat the error map.
        """

        # Update Slicer
        with param.parameterized.discard_events(self.slicer):
            self.slicer.autoformat_error_map()

            # Update gui
            self._set_app_widget_attr(
                "Analysis", "error_cmap", "value", (self.slicer.error_map_cmap)
            )

            self._set_app_widget_attr(
                "Analysis", "error_scale", "value", (self.slicer.error_map_scale)
            )

        self.slicer.param.trigger("error_map_scale")

    def _build_export_widgets(self) -> Dict[str, pn.widgets.Widget]:

        widgets = {}

        widgets["export_path"] = pn.widgets.TextAreaInput(
            name="Export Path",
            value=self.config_path,
            placeholder="Enter path to export config file",
        )

        def _update_export_path(event):
            self.config_path = event.new

        widgets["export_path"].param.watch(_update_export_path, "value")

        widgets["export_config_button"] = pn.widgets.Button(
            name="Export Config",
            button_type="primary",
            on_click=self._export_config,
        )

        return widgets

    @error.error_handler_decorator()
    def _export_config(self, event):
        """
        Export the current configuration as a JSON.
        """
        if self.config_path is None:
            pn.state.notifications.warning("No path provided to export config.")
            return

        exp_dir = os.path.dirname(self.config_path)

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

        with open(self.config_path, "w") as f:
            json.dump(config_dict, f, indent=4, default=config.json_serial)

        pn.state.notifications.success(f"Config saved to {self.config_path}")
