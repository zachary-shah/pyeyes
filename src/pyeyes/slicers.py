"""
Slicers: Defined as classes that take N-dimensional data and can return a 2D view of that data given some input
"""

from typing import Dict, List, Optional, Sequence

import holoviews as hv
import matplotlib.colors as mcolors
import numpy as np
import panel as pn
import param

from .icons import LR_FLIP_OFF, LR_FLIP_ON, UD_FLIP_OFF, UD_FLIP_ON
from .q_cmap.cmap import relaxation_color_map
from . import themes

hv.extension("bokeh")

VALID_COLORMAPS = [
    "gray",
    "jet",
    "viridis",
    "inferno",
    "RdBu",
    "Magma",
    "Quantitative",
]

# Complex view mapping
CPLX_VIEW_MAP = {
    "mag": np.abs,
    "phase": np.angle,
    "real": np.real,
    "imag": np.imag,
}

def _format_image(plot, element):
    """
    For setting image theme (light/dark mode).
    """

    # Enforce theme
    plot.state.background_fill_color = themes.VIEW_THEME.background_color
    plot.state.border_fill_color = themes.VIEW_THEME.background_color

    # decrease border size
    min_border = 3
    plot.state.min_border_bottom = min_border
    plot.state.min_border_left = min_border
    plot.state.min_border_right = min_border
    plot.state.min_border_top = min_border
    plot.state.min_border = min_border
    plot.border = min_border

    # Constant height for the figure title
    if plot.state.title.text_font_size[-2:] == "px":
        tfs = int(plot.state.title.text_font_size[:-2]) * 2 + plot.border
        plot.state.height = plot.state.height + tfs
    else:
        print("WARNING: Could not parse title font size. Figure scale may be skewed.")

    # Color to match theme
    plot.state.outline_line_color = themes.VIEW_THEME.background_color
    plot.state.outline_line_alpha = 1.0
    plot.state.title.text_color = themes.VIEW_THEME.text_color
    plot.state.title.text_font = themes.VIEW_THEME.text_font

def _hide_image(plot, element):
    """
    Hook to hide the image in a plot and only show the colorbar.
    """
    for r in plot.state.renderers:
        if hasattr(r, "glyph"):
            r.visible = False

    # # Remove border/outline so only the colorbar remains
    plot.state.outline_line_color = None
    plot.state.toolbar_location = None
    plot.state.background_fill_alpha = 1.0
    plot.state.outline_line_alpha = 0

def _format_colorbar(plot, element):
    """
    Colorbar formatting. Just need to ensure that colorbar scales with plot height.
    Assumes that colorbar is the first element in the right panel.
    """
    p = plot.state.right[0]

    # title
    if p.title == "":
        p.title = None

    # sizes
    p.major_label_text_font_size = f"{int(plot.state.height/38)}pt"
    p.title_text_font_size = f"{int(plot.state.height/38)}pt"
    p.width  = int(plot.state.width * (0.22 - 0.03 * (p.title is not None)))
        
    # spacing
    p.padding = 5

    # coloring
    p.background_fill_color = themes.VIEW_THEME.background_color
    p.background_fill_alpha = 1.0
    p.major_label_text_color = themes.VIEW_THEME.text_color
    p.major_tick_line_color = themes.VIEW_THEME.text_color
    p.title_text_color = themes.VIEW_THEME.text_color
    p.title_text_align = "center"
    p.title_text_baseline = "middle"    

    # text font
    p.title_text_font = themes.VIEW_THEME.text_font
    p.major_label_text_font = themes.VIEW_THEME.text_font

    # other keys to format
    p.major_label_text_align = "left"
    p.major_label_text_alpha = 1.0
    p.major_label_text_baseline = "middle"
    p.major_label_text_line_height = 1.2
    p.major_tick_line_alpha = 1.0
    p.major_tick_line_dash_offset = 0
    p.major_tick_line_width = 1

class NDSlicer(param.Parameterized):
    # Viewing Parameters
    vmin = param.Number(default=0.0)
    vmax = param.Number(default=1.0)
    size_scale = param.Number(default=400, bounds=(100, 1000), step=10)
    cmap = param.Selector(default="gray", objects=VALID_COLORMAPS)
    flip_ud = param.Boolean(default=False)
    flip_lr = param.Boolean(default=False)
    cplx_view = param.Selector(default="mag", objects=["mag", "phase", "real", "imag"])
    display_images = param.ListSelector(default=[], objects=[])
    colorbar_on = param.Boolean(default=True)
    colorbar_label = param.String(default="")

    # Slice Dimension Parameters
    dim_indices = param.Dict(default={}, doc="Mapping: dim_name -> int index")

    # Crop Range Parameters
    lr_crop = param.Range(default=(0, 100), bounds=(0, 100), step=1)
    ud_crop = param.Range(default=(0, 100), bounds=(0, 100), step=1)

    def __init__(
        self,
        data: hv.Dataset,
        vdims: Sequence[str],
        cdim: Optional[str] = None,
        clabs: Optional[Sequence[str]] = None,
        cat_dims: Optional[Dict[str, List]] = None,
        **params
    ):
        """
        Slicer for N-dimensional data. This class is meant to be a subclass of a Viewer.

        Way data will be sliced:
        - vdims: Viewing dimensions. Should always be length 2
        - cdim: Collate-able dimension. If provided, will return a 1D layout of 2D slices rather than a single 2D
          image slice.
          Can also provide labels for each collated image.
        """

        super().__init__(**params)

        self.data = data

        self.cat_dims = cat_dims

        # all dimensions
        self.ndims = [d.name for d in data.kdims][::-1]

        # Dictionary of all total size of each dimension
        self.dim_sizes = {}
        for dim in self.ndims:
            self.dim_sizes[dim] = data.aggregate(dim, np.mean).data[dim].size

        # Initialize slize cache
        self.slice_cache = {}
        for dim in self.ndims:
            if dim in self.cat_dims.keys():
                self.slice_cache[dim] = self.cat_dims[dim][self.dim_sizes[dim] // 2]
            else:
                self.slice_cache[dim] = self.dim_sizes[dim] // 2

        assert len(vdims) == 2, "Viewing dims must be length 2"
        assert np.array(
            [vd in self.ndims for vd in vdims]
        ).all(), "Viewing dims must be in all dims"

        # collate-able dimension
        if cdim is not None:
            assert cdim in self.ndims, "Collate dim must be in named dims"

            if clabs is not None:
                assert (
                    len(clabs) == self.dim_sizes[cdim]
                ), "Collate labels must match collate dimension size"
            else:
                # assume data categorical. FIXME: infer c data type in general
                clabs = self.data.aggregate(self.cdim, np.mean).data[self.cdim].tolist()

            self.clabs = clabs
            self.cdim = cdim
            self.Nc = self.dim_sizes[cdim]
        else:
            self.clabs = None
            self.cdim = None
            self.Nc = 1

        # Initialize cropping
        self.crop_cache = {}
        for dim in self.ndims:
            self.crop_cache[dim] = (0, self.dim_sizes[dim])

        # This sets self.vdims, self.sdims, self.non_sdims, and upates self.dim_indices param
        self._set_volatile_dims(vdims)

        # Initialize view cache
        self.CPLX_VIEW_CLIM_CACHE = {}

        # Update color limits with default
        self.update_cplx_view("mag")

        # Initialize display images
        self.param.display_images.objects = self.clabs
        self.display_images = self.clabs

    def update_cache(self):
        """
        Cache inputs that may be useful to remember when we change other parameters.
        """

        # Cache cropped view
        self.crop_cache[self.vdims[0]] = (self.lr_crop[0], self.lr_crop[1])
        self.crop_cache[self.vdims[1]] = (self.ud_crop[0], self.ud_crop[1])

        # Cache color details
        self.CPLX_VIEW_CLIM_CACHE[self.cplx_view]["vmin"] = self.vmin
        self.CPLX_VIEW_CLIM_CACHE[self.cplx_view]["vmax"] = self.vmax
        self.CPLX_VIEW_CLIM_CACHE[self.cplx_view]["cmap"] = self.cmap

        # Slice cache
        for dim in self.sdims:
            self.slice_cache[dim] = self.dim_indices[dim]

    def slice(self) -> List[hv.Image]:
        """
        Return the slice of the hv.Dataset given the current slice indices.
        """

        # Dimensions to select
        sdim_dict = {dim: self.dim_indices[dim] for dim in self.sdims}

        # Collate case. FIXME: simply code
        if self.cdim is not None:

            imgs = []

            for img_label in self.display_images:

                sliced_2d = self.data.select(
                    **{self.cdim: img_label}, **sdim_dict
                ).reduce([self.cdim] + self.sdims, np.mean)

                # order vdims in slice the same as self.vdims
                sliced_2d = sliced_2d.reindex(self.vdims)

                # set slice extent
                sliced_2d = sliced_2d.redim.range(
                    **{self.vdims[0]: self.lr_crop},
                    **{self.vdims[1]: self.ud_crop},
                )

                imgs.append(hv.Image(sliced_2d, label=img_label))
        else:

            # Select slice indices for each dimension
            sliced_2d = (
                self.data.select(**sdim_dict)
                .reduce(self.sdims, np.mean)
                .reindex(self.vdims)
            )

            # set slice extent
            sliced_2d = sliced_2d.redim.range(
                **{self.vdims[0]: self.lr_crop},
                **{self.vdims[1]: self.ud_crop},
            )

            imgs = [hv.Image(sliced_2d)]

        return imgs

    @param.depends(
        "dim_indices",
        "vmin",
        "vmax",
        "cmap",
        "size_scale",
        "flip_ud",
        "flip_lr",
        "lr_crop",
        "ud_crop",
        "display_images",
        "colorbar_on",
        "colorbar_label",
    )
    def view(self) -> hv.Layout:
        """
        Return the formatted view of the data given the current slice indices.
        """

        self.update_cache()

        imgs = self.slice()

        new_im_size = (
            self.lr_crop[1] - self.lr_crop[0],
            self.ud_crop[1] - self.ud_crop[0],
        )

        if self.cmap == "Quantitative":
            rgb_vec, clip_for_qmap = relaxation_color_map(
                self._infer_quantitative(),
                self.vmin,
                self.vmax,
            )
            cmap = mcolors.ListedColormap(rgb_vec)

            for i in range(len(imgs)):
                imgs[i].data["Value"] = clip_for_qmap(imgs[i].data["Value"])
        else:
            cmap = self.cmap

        for i in range(len(imgs)):
            # Apply complex view
            imgs[i].data["Value"] = CPLX_VIEW_MAP[self.cplx_view](imgs[i].data["Value"])

            # parameterized view options
            imgs[i] = imgs[i].opts(
                cmap=cmap,
                xaxis=None,
                yaxis=None,
                clim=(self.vmin, self.vmax),
                width=int(self.size_scale * new_im_size[0] / np.max(new_im_size)),
                height=int(self.size_scale * new_im_size[1] / np.max(new_im_size)),
                invert_yaxis=self.flip_ud,
                invert_xaxis=self.flip_lr,
                hooks=[_format_image],
            )

        # This is a workaround to show the colorbar, since with Layout it is only possible to create a colorbar
        # per element, and not per Layout. So we create a dummy Image element with the same colorbar settings.
        if self.colorbar_on:
            cbar_fig = hv.Image(np.zeros((2, 2))).opts(
                cmap=cmap,
                clim=(self.vmin, self.vmax),
                colorbar=True,
                colorbar_opts={
                    "title":self.colorbar_label,
                },
                colorbar_position="right",
                xaxis=None,
                yaxis=None,
                width=int(self.size_scale * (new_im_size[1] / np.max(new_im_size)) * 0.20 + 30), # 5% maintained aspect
                height=int(self.size_scale * new_im_size[1] / np.max(new_im_size)),
                hooks=[_format_image, _hide_image, _format_colorbar],  # Hide the dummy glyph

            )

            imgs.append(cbar_fig)

        return hv.Layout(imgs).opts(
            shared_axes=True,
        )

    def _infer_quantitative(self):
        if "MRF Type" in self.cat_dims and "MRF Type" in self.dim_indices:
            return self.dim_indices["MRF Type"]

    def _set_volatile_dims(self, vdims: Sequence[str]):
        """
        Sets dimensions which could be updated upon a change in viewing dimension.
        """

        with param.parameterized.discard_events(self):

            # do this without triggering parameters
            vdims = list(vdims)

            self.vdims = vdims

            if self.cdim is not None:
                self.non_sdims = [self.cdim] + vdims
            else:
                self.non_sdims = vdims

            # sliceable dimensions
            self.sdims = [d for d in self.ndims if d not in self.non_sdims]

            # Update scaling for height and width ranges
            self.img_dims = np.array([self.dim_sizes[vd] for vd in self.vdims])

            # Update crop bounds
            self.param.lr_crop.bounds = (0, self.img_dims[0])
            self.param.ud_crop.bounds = (0, self.img_dims[1])
            self.lr_crop = self.crop_cache[self.vdims[0]]
            self.ud_crop = self.crop_cache[self.vdims[1]]

            # Start in the center of each sliceable dimension
            slice_dim_names = {}
            for dim in self.sdims:
                slice_dim_names[dim] = self.slice_cache[dim]

            # Set default slice indicess
            self.param.dim_indices.default = slice_dim_names
            self.dim_indices = slice_dim_names

        # trigger callbacks now
        self.param.trigger("dim_indices", "lr_crop", "ud_crop")

    def update_vdims(self, vdims: Sequence[str]):
        """
        Update viewing dimensions and associated widgets
        """
        old_vdims = self.vdims

        self._set_volatile_dims(vdims)

        # Update widgets only if inter-change of slice and view dimensions
        if set(old_vdims) != set(vdims):
            return self.get_sdim_widgets()
        else:
            return {}

    def update_cplx_view(self, new_cplx_view: str):

        # set attribute
        self.cplx_view = new_cplx_view

        VSTEP_INTERVAL = 200

        if (
            self.cplx_view in self.CPLX_VIEW_CLIM_CACHE
            and len(self.CPLX_VIEW_CLIM_CACHE[self.cplx_view]) > 0
        ):

            vmind = self.CPLX_VIEW_CLIM_CACHE[self.cplx_view]["vmin"]
            vminb = self.CPLX_VIEW_CLIM_CACHE[self.cplx_view]["vmin_bounds"]
            vmins = self.CPLX_VIEW_CLIM_CACHE[self.cplx_view]["vmin_step"]
            vmaxd = self.CPLX_VIEW_CLIM_CACHE[self.cplx_view]["vmax"]
            vmaxb = self.CPLX_VIEW_CLIM_CACHE[self.cplx_view]["vmax_bounds"]
            vmaxs = self.CPLX_VIEW_CLIM_CACHE[self.cplx_view]["vmax_step"]
            cmap = self.CPLX_VIEW_CLIM_CACHE[self.cplx_view]["cmap"]

        else:

            # use same cmap
            cmap = self.cmap

            # Compute max color limits
            mn = np.min(
                np.stack(
                    [
                        CPLX_VIEW_MAP[self.cplx_view](self.data[v.name])
                        for v in self.data.vdims
                    ]
                )
            )
            mx = np.max(
                np.stack(
                    [
                        CPLX_VIEW_MAP[self.cplx_view](self.data[v.name])
                        for v in self.data.vdims
                    ]
                )
            )

            vmind = mn
            vminb = (mn, mx)
            vmins = (mx - mn) / VSTEP_INTERVAL
            vmaxd = mx
            vmaxb = (mn, mx)
            vmaxs = (mx - mn) / VSTEP_INTERVAL

            self.CPLX_VIEW_CLIM_CACHE[self.cplx_view] = dict(
                vmin=vmind,
                vmin_bounds=vminb,
                vmin_step=vmins,
                vmax=vmaxd,
                vmax_bounds=vmaxb,
                vmax_step=vmaxs,
                cmap=self.cmap,
            )

        # Update color limits
        self.param.vmin.default = vmind
        self.param.vmin.bounds = vminb
        self.param.vmin.step = vmins
        self.param.vmax.default = vmaxd
        self.param.vmax.bounds = vmaxb
        self.param.vmax.step = vmaxs
        self.param.cmap.default = cmap

        with param.parameterized.discard_events(self):
            self.vmin = vmind
            self.vmax = vmaxd
            self.cmap = cmap
        self.param.trigger("vmin", "vmax", "cmap")

    def autoscale_clim(self):
        """
        For given slice, automatically set vmin and vmax to min and max of data
        """

        imgs = self.slice()

        data = np.concatenate(
            [CPLX_VIEW_MAP[self.cplx_view](img.data["Value"]) for img in imgs]
        )

        with param.parameterized.discard_events(self):
            self.vmin = np.percentile(data, 0.0)
            self.vmax = np.percentile(data, 99.9)
        self.param.trigger("vmin", "vmax")

    def update_display_image_list(self, display_images: Sequence[str]):

        self.display_images = display_images

        self.param.trigger("display_images")

    def get_sdim_widgets(self) -> dict:
        """
        Return a dictionary of panel widgets to interactively control slicing.
        """

        sliders = {}
        for dim in self.sdims:
            if dim in self.cat_dims.keys():
                s = pn.widgets.Select(
                    name=dim, options=self.cat_dims[dim], value=self.slice_cache[dim]
                )
            else:
                s = pn.widgets.EditableIntSlider(
                    name=dim,
                    start=0,
                    end=self.dim_sizes[dim] - 1,
                    value=self.dim_indices[dim],
                )

            def _update_dim_indices(event, this_dim=dim):
                self.dim_indices[this_dim] = event.new
                # trigger dim_indices has been changed
                self.param.trigger("dim_indices")

            s.param.watch(_update_dim_indices, "value")

            sliders[dim] = s

        return sliders

    def get_viewing_widgets(self) -> Sequence[pn.widgets.Widget]:

        sliders = []

        # Flip Widgets
        ud_w = self.__add_widget(
            pn.widgets.ToggleIcon,
            "flip_ud",
            description="Flip Image Up/Down",
            icon=UD_FLIP_OFF,
            active_icon=UD_FLIP_ON,
            size="10em",
            margin=(-20, 10, -20, 25),
            show_name=False,
        )

        lr_w = self.__add_widget(
            pn.widgets.ToggleIcon,
            "flip_lr",
            description="Flip Image Left/Right",
            icon=LR_FLIP_OFF,
            active_icon=LR_FLIP_ON,
            size="10em",
            margin=(-20, 10, -20, 10),
            show_name=False,
        )

        sliders.append(pn.Row(ud_w, lr_w))

        sliders.append(
            self.__add_widget(
                pn.widgets.EditableIntSlider,
                "size_scale",
                start=self.param.size_scale.bounds[0],
                end=self.param.size_scale.bounds[1],
                value=self.size_scale,
                step=self.param.size_scale.step,
            )
        )

        # bounding box crop for each L/R/U/D edge
        lr_crop_slider = pn.widgets.IntRangeSlider(
            name="L/R Display Range",
            start=self.param.lr_crop.bounds[0],
            end=self.param.lr_crop.bounds[1],
            value=(self.lr_crop[0], self.lr_crop[1]),
            step=self.param.lr_crop.step,
        )

        def _update_lr_slider(event):
            crop_lower, crop_upper = event.new
            self.lr_crop = (crop_lower, crop_upper)
            self.param.trigger("lr_crop")

        lr_crop_slider.param.watch(_update_lr_slider, "value")
        sliders.append(lr_crop_slider)

        ud_crop_slider = pn.widgets.IntRangeSlider(
            name="U/D Display Range",
            start=self.param.ud_crop.bounds[0],
            end=self.param.ud_crop.bounds[1],
            value=(self.ud_crop[0], self.ud_crop[1]),
            step=self.param.ud_crop.step,
        )

        def _update_ud_slider(event):
            crop_lower, crop_upper = event.new
            self.ud_crop = (crop_lower, crop_upper)
            self.param.trigger("ud_crop")

        ud_crop_slider.param.watch(_update_ud_slider, "value")
        sliders.append(ud_crop_slider)
        return sliders

    def get_contrast_widgets(self) -> Sequence[pn.widgets.Widget]:

        sliders = []

        # vmin/vmax use different Range slider
        range_slider = pn.widgets.EditableRangeSlider(
            name="clim",
            start=self.param.vmin.bounds[0],
            end=self.param.vmax.bounds[1],
            value=(self.vmin, self.vmax),
            step=self.param.vmin.step,
        )

        def _update_clim(event):
            self.vmin, self.vmax = event.new
            self.param.trigger("vmin")
            self.param.trigger("vmax")

        range_slider.param.watch(_update_clim, "value")
        sliders.append(range_slider)

        sliders.append(
            self.__add_widget(
                pn.widgets.Select,
                "cmap",
                options=VALID_COLORMAPS,
                value=self.cmap,
            )
        )

        # Colorbar toggle
        colorbar_widget = pn.widgets.Checkbox(
            name = "Add Colorbar",
            value = self.colorbar_on,
        )
        def _update_colorbar(event):
            self.colorbar_on = event.new
            self.param.trigger("colorbar_on")
        colorbar_widget.param.watch(_update_colorbar, "value")
        sliders.append(colorbar_widget)

        colorbar_label_widget = pn.widgets.TextInput(
            name = "Colorbar Label",
            value = self.colorbar_label,
        )
        def _update_colorbar_label(event):
            self.colorbar_label = event.new
            self.param.trigger("colorbar_label")
        colorbar_label_widget.param.watch(_update_colorbar_label, "value")

        # disable colorbar label if colorbar is off
        def _update_colorbar_label_disabled(x):
            colorbar_label_widget.disabled = not x
        pn.bind(_update_colorbar_label_disabled, colorbar_widget, watch=True)

        sliders.append(colorbar_label_widget)

        return sliders

    def get_roi_widgets(self) -> Sequence[pn.widgets.Widget]:
        # TODO: roi page

        sliders = []

        sliders.append(
            self.__add_widget(
                pn.widgets.TextInput,
                "ROI (TODO)",
                value="",
            )
        )

        sliders.append(
            self.__add_widget(
                pn.widgets.Switch,
                "ROI (TODO)",
                value=False,
            )
        )

        sliders.append(
            self.__add_widget(
                pn.widgets.Switch,
                "RMSE (TODO)",
                value=False,
            )
        )

        return sliders

    def get_analysis_widgets(self) -> Sequence[pn.widgets.Widget]:
        # TODO: add more analysis options

        sliders = []

        sliders.append(
            self.__add_widget(
                pn.widgets.TextInput,
                "Analysis (TODO)",
                value="",
            )
        )

        sliders.append(
            self.__add_widget(
                pn.widgets.Switch,
                "Analysis (TODO)",
                value=False,
            )
        )

        return sliders

    def get_export_widgets(self) -> Sequence[pn.widgets.Widget]:
        # TODO: add more export options

        sliders = []

        sliders.append(
            self.__add_widget(
                pn.widgets.FileDownload,
                "Export",
                label="Export Image",
                filename="image.png",
                callback=lambda: self.view,
            )
        )

        return sliders

    def __add_widget(self, widget: callable, name: str, **kwargs) -> pn.widgets.Widget:

        initialized = False
        if "show_name" in kwargs and kwargs["show_name"] == False:
            kwargs.pop("show_name")
            w = widget(**kwargs)
            initialized = True
        elif "show_name" in kwargs:
            kwargs.pop("show_name")

        if not initialized:
            w = widget(name=name, **kwargs)

        def _update(event):
            # update self.name
            # self.__dict__[name] = event.new
            if hasattr(self, name):
                setattr(self, name, event.new)
                self.param.trigger(name)

        w.param.watch(_update, "value")

        return w
