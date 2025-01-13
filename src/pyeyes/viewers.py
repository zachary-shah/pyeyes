import warnings
from typing import Dict, List, Optional, Sequence, Union

import holoviews as hv
import numpy as np
import panel as pn
import param
from holoviews import opts

from . import themes
from .q_cmap.cmap import VALID_COLORMAPS
from .slicers import NDSlicer
from .utils import normalize, tonp

hv.extension("bokeh")

# NOTE: api to set theme. can be done by user outside of this api;
# (can't be updated live easily because panels html page style will be static)
themes.set_theme("dark")


# message should regexp. FIXME: not ignoring ...
warnings.filterwarnings(
    "ignore",
    message=r"Dropping a patch because it contains a previously known reference \(id='p\d+'\).*",
)


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
    vdim_horiz = param.Selector(default="x")
    vdim_vert = param.Selector(default="y")

    # Displayed Images
    single_image_toggle = param.Boolean(default=False)
    display_images = param.ListSelector(default=[], objects=[])

    # Theme selection
    theme = param.ObjectSelector(
        default="dark", objects=list(themes.SUPPORTED_THEMES.keys())
    )

    def __init__(
        self,
        data: dict[np.ndarray],
        named_dims: Sequence[str],
        view_dims: Optional[Sequence[str]] = None,
        cat_dims: Optional[Dict[str, List]] = {},
        **kwargs
    ):
        """
        Viewer for comparing n-dimensional image data.

        Parameters:
        -----------
        data : dict[np.ndarray]
            dictionary of images. Keys should be strings which name each image,
            and values should be numpy arrays of equivalent size.
        named_dims : Sequence[str]
            String name for each dimension of the image data, with the same ordering as
            the array dimensions.
        view_dims : Optional[Sequence[str]]
            Initial dimensions to view, should be a subset of dimension_names.
            Default is ['x', 'y'] if in dimension_names else first 2 dimensions.
        """

        super().__init__(data)
        param.Parameterized.__init__(self, **kwargs)

        img_names = list(data.keys())
        img_list = list(data.values())

        N_img = len(img_list)
        N_dim = len(named_dims)

        self.is_complex_data = any([np.iscomplexobj(img) for img in img_list])

        assert np.array(
            [img.shape == img_list[0].shape for img in img_list]
        ).all(), "All viewed data must have the same input shape."
        assert (
            N_dim == img_list[0].ndim
        ), "Number of dimension names must match the number of dimensions in the data."

        if view_dims is not None:
            assert all(
                [dim in named_dims for dim in view_dims]
            ), "All view dimensions must be in dimension_names."
        else:
            if "x" in named_dims.lower() and "y" in named_dims.lower():
                view_dims = ["x", "y"]
            else:
                view_dims = named_dims[:2]

        # Init display images
        self.param.display_images.objects = img_names
        self.display_images = img_names

        # Update View dims
        self.param.vdim_horiz.objects = named_dims
        self.param.vdim_vert.objects = named_dims
        self.vdim_horiz = view_dims[0]
        self.vdim_vert = view_dims[1]

        self.vdims = view_dims
        self.ndims = named_dims
        self.img_names = img_names
        self.N_img = N_img
        self.N_dim = N_dim

        self.cat_dims = cat_dims
        self.noncat_dims = [dim for dim in named_dims if dim not in cat_dims.keys()]

        # Aggregate data, stacking image type to first axis
        self.raw_data = np.stack(img_list, axis=0)

        # Instantiate dataset for intial view dims
        self.dataset = self._build_dataset(self.vdims)

        # Instantiate slicer
        self.slicer = NDSlicer(
            self.dataset, self.vdims, cdim="ImgName", clabs=img_names, cat_dims=cat_dims
        )

        """
        Create Panel Layout
        """
        # Widgets per pane
        view_pane_widgets = self._init_view_pane_widgets()
        contrast_pane_widgets = self._init_contrast_pane_widgets()
        roi_pane_widgets = self._init_roi_pane_widgets()
        analysis_pane_widgets = self._init_analysis_pane_widgets()
        export_pane_widgets = self._init_export_pane_widgets()

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

        self._autoscale_clim(event=None)

    def launch(self):
        pn.serve(self.app, title="MRI Viewer", show=True)

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
        widgets["im_display"] = self._build_display_images_widget()

        return widgets

    def _init_contrast_pane_widgets(self) -> Dict[str, pn.widgets.Widget]:
        """
        Returns a dictionary of widgets belonging to the "Contrast" pane
        """

        widgets = {}

        cplx_widget, clim_scale_widget = self._build_cplx_widget()

        # Select which real-valued view of complex data to view
        widgets["cplx_widget"] = cplx_widget

        # Color map stuff
        widgets.update(self._build_contrast_widgets())

        # Auto-scaling button
        widgets["clim_scale_widget"] = clim_scale_widget

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

        NOTE: Anything that depends on the viewing dimensions (e.g. normalization) should be done here.
        """

        proc_data = self._normalize(self.raw_data)

        dim_ranges = [self.img_names]
        for i in range(1, proc_data.ndim):
            if self.ndims[i - 1] in self.cat_dims:
                dim_ranges.append(self.cat_dims[self.ndims[i - 1]])
            else:
                dim_ranges.append(range(proc_data.shape[i]))

        # Convention is to reverse ordering relative to dimensions named
        proc_data = proc_data.transpose(*list(range(proc_data.ndim - 1, -1, -1)))

        return hv.Dataset((*dim_ranges, proc_data), ["ImgName"] + self.ndims, "Value")

    def _normalize(self, data, target_index=0):
        """
        Normalize raw data.

        FIXME: normalize in view dimensions only.
            For example, what if we have PD/T2/T1 as one dimension? These are not the same scale.
            Also should normalize per z slice.
        FIXME: Parameterize target index somewhere.
        """
        return data
        # return normalize(data, data[target_index], ofs=True, mag=np.iscomplexobj(data))

    def _build_vdim_widgets(self) -> Dict[str, pn.widgets.Widget]:
        """
        Build selection widgets for 2 viewing dimensions.
        """

        vdim_horiz_widget = pn.widgets.Select(
            name="L/R Viewing Dimension", options=self.noncat_dims, value=self.vdims[0]
        )

        def vdim_horiz_callback(event):
            if event.new != event.old:
                vh_new = event.new
                vv_new = event.old if (vh_new == self.vdim_vert) else self.vdim_vert
                self._update_vdims([vh_new, vv_new])

        vdim_horiz_widget.param.watch(vdim_horiz_callback, "value")

        vdim_vert_widget = pn.widgets.Select(
            name="U/D Viewing Dimension", options=self.noncat_dims, value=self.vdims[1]
        )

        def vdim_vert_callback(event):
            if event.new != event.old:
                vv_new = event.new
                vh_new = event.old if (vv_new == self.vdim_horiz) else self.vdim_horiz
                self._update_vdims([vh_new, vv_new])

        vdim_vert_widget.param.watch(vdim_vert_callback, "value")

        return {self.vdims[0]: vdim_horiz_widget, self.vdims[1]: vdim_vert_widget}

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
            self.app[0][0][0].value = new_vdims[0]
            self.app[0][0][1].value = new_vdims[1]

            # Update Slicer
            old_vdims = self.slicer.vdims
            self.slicer._set_volatile_dims(new_vdims)

            # Update displayed widgets if interchange of vdim and sdims
            if set(old_vdims) != set(new_vdims):
                new_sdim_widget_dict = self._build_sdim_widgets()

                for i, w in enumerate(list(new_sdim_widget_dict.keys())):
                    self.app[0][0][i + 2] = new_sdim_widget_dict[w]

            # Reset crops
            self.app[0][0][-4].bounds = (0, self.slicer.img_dims[0])
            self.app[0][0][-3].bounds = (0, self.slicer.img_dims[1])
            self.app[0][0][-4].value = self.slicer.lr_crop
            self.app[0][0][-3].value = self.slicer.ud_crop

        self.slicer.param.trigger("dim_indices")

    def _build_sdim_widgets(self) -> dict:
        """
        Return a dictionary of panel widgets to interactively control slicing.
        """

        sliders = {}
        for dim in self.slicer.sdims:
            if dim in self.cat_dims.keys():
                s = pn.widgets.Select(
                    name=dim,
                    options=self.cat_dims[dim],
                    value=self.slicer.slice_cache[dim],
                )
            else:
                s = pn.widgets.EditableIntSlider(
                    name=dim,
                    start=0,
                    end=self.slicer.dim_sizes[dim] - 1,
                    value=self.slicer.dim_indices[dim],
                )

            def _update_dim_indices(event, this_dim=dim):

                self._update_sdim(this_dim, event.new)

            s.param.watch(_update_dim_indices, "value")

            sliders[dim] = s

        return sliders

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
        self.slicer.param.trigger("dim_indices", "cmap")

    def _build_viewing_widgets(self):
        """
        Return a dictionary of panel widgets to interactively control viewing.
        """

        sliders = {}

        # Flip Widgets
        ud_w = pn.widgets.Checkbox(name="Flip Image Up/Down", value=False)

        def flip_ud_callback(event):
            if event.new != event.old:
                with param.parameterized.discard_events(self.slicer):
                    self.slicer.flip_ud = event.new
                self.slicer.param.trigger("flip_ud")

        ud_w.param.watch(flip_ud_callback, "value")
        sliders["flip_ud"] = ud_w

        lr_w = pn.widgets.Checkbox(name="Flip Image Left/Right", value=False)

        def flip_lr_callback(event):
            if event.new != event.old:
                with param.parameterized.discard_events(self.slicer):
                    self.slicer.flip_lr = event.new
                self.slicer.param.trigger("flip_lr")

        lr_w.param.watch(flip_lr_callback, "value")
        sliders["flip_lr"] = lr_w

        size_scale_widget = pn.widgets.EditableIntSlider(
            name="Size Scale",
            start=self.slicer.param.size_scale.bounds[0],
            end=self.slicer.param.size_scale.bounds[1],
            value=self.slicer.size_scale,
            step=self.slicer.param.size_scale.step,
        )

        def size_scale_callback(event):
            with param.parameterized.discard_events(self.slicer):
                self.slicer.size_scale = event.new
            self.slicer.param.trigger("size_scale")

        size_scale_widget.param.watch(size_scale_callback, "value")
        sliders["size_scale"] = size_scale_widget

        # bounding box crop for each L/R/U/D edge
        lr_crop_slider = pn.widgets.IntRangeSlider(
            name="L/R Display Range",
            start=self.slicer.param.lr_crop.bounds[0],
            end=self.slicer.param.lr_crop.bounds[1],
            value=(self.slicer.lr_crop[0], self.slicer.lr_crop[1]),
            step=self.slicer.param.lr_crop.step,
        )

        def _update_lr_slider(event):
            crop_lower, crop_upper = event.new
            with param.parameterized.discard_events(self.slicer):
                self.slicer.lr_crop = (crop_lower, crop_upper)
            self.slicer.param.trigger("lr_crop")

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
            crop_lower, crop_upper = event.new
            with param.parameterized.discard_events(self.slicer):
                self.slicer.ud_crop = (crop_lower, crop_upper)
            self.slicer.param.trigger("ud_crop")

        ud_crop_slider.param.watch(_update_ud_slider, "value")
        sliders["ud_crop"] = ud_crop_slider

        return sliders

    def _build_single_toggle_widget(self):
        # Single toggle view
        single_toggle = pn.widgets.Checkbox(name="Single View", value=False)

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

        # update
        self.app[0][0].pop(-1)
        self.app[0][0].append(new_display_images_widget)

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

        def display_images_callback(event):
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
            value="mag",
            button_type="primary",
            button_style="outline",
        )

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
            self.app[0][1][1].start = self.slicer.param.vmin.bounds[0]
            self.app[0][1][1].end = self.slicer.param.vmax.bounds[1]
            self.app[0][1][1].value = (self.slicer.vmin, self.slicer.vmax)
            self.app[0][1][1].step = self.slicer.param.vmax.step

            self.app[0][1][2].value = self.slicer.cmap

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
            with param.parameterized.discard_events(self.slicer):
                self.slicer.colorbar_on = event.new
            self.slicer.param.trigger("colorbar_on")

        colorbar_widget.param.watch(_update_colorbar, "value")
        widgets["colorbar"] = colorbar_widget

        colorbar_label_widget = pn.widgets.TextInput(
            name="Colorbar Label",
            value=self.slicer.colorbar_label,
        )

        def _update_colorbar_label(event):
            with param.parameterized.discard_events(self.slicer):
                self.slicer.colorbar_label = event.new
            self.slicer.param.trigger("colorbar_label")

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
            self.app[0][1][1].value = (self.slicer.vmin, self.slicer.vmax)

        self.slicer.param.trigger("vmin", "vmax")

    def _build_roi_widgets(self) -> Dict[str, pn.widgets.Widget]:

        widgets = {}

        # ROI Widgets
        roi_button = pn.widgets.Button(name="Draw ROI (TODO)", button_type="primary")

        def _draw_roi(event):
            print("Draw ROI not yet implemented.")

        roi_button.on_click(_draw_roi)
        widgets["draw_roi"] = roi_button

        return widgets

    def _build_analysis_widgets(self) -> Dict[str, pn.widgets.Widget]:

        widgets = {}

        # Analysis Widgets
        analysis_button = pn.widgets.Button(
            name="Run Analysis (TODO)", button_type="primary"
        )

        def _run_analysis(event):
            print("Run Analysis not yet implemented.")

        analysis_button.on_click(_run_analysis)
        widgets["run_analysis"] = analysis_button

        return widgets

    def _build_export_widgets(self) -> Dict[str, pn.widgets.Widget]:

        widgets = {}

        widgets["export_image_button"] = pn.widgets.FileDownload(
            label="Export Image (TODO)",
            filename="export.png",
            callback=self._export_image,
        )

        widgets["export_config_button"] = pn.widgets.FileDownload(
            label="Export Config (TODO)",
            filename="config.json",
            callback=self._export_config,
        )

        return widgets

    def _export_image(self):
        """
        Export the current image as a PNG.
        """
        print("Exporting image not yet implemented.")

    def _export_config(self):
        """
        Export the current configuration as a JSON.
        """
        print("Exporting config not yet implemented.")
