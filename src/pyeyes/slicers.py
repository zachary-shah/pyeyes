"""
Slicers: Defined as classes that take N-dimensional data and can return a 2D view of that data given some input
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import holoviews as hv
import numpy as np
import panel as pn
import param
from holoviews import streams

from . import error, roi, themes
from .q_cmap.cmap import (
    QUANTITATIVE_MAPTYPES,
    VALID_COLORMAPS,
    ColorMap,
    QuantitativeColorMap,
)

hv.extension("bokeh")

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
    elif plot.state.title.text_font_size[-2:] == "em":
        pass  # Child of parent uses relative font size
    else:
        error.warning(
            f"_format_image hook could not parse title font size: \
            {plot.state.title.text_font_size}. Figure scale may be skewed."
        )

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
    p.width = int(plot.state.width * (0.22 - 0.03 * (p.title is not None)))
    p.major_label_text_font_size = f"{int(plot.state.width/8)}pt"
    p.title_text_font_size = f"{int(plot.state.width/8)}pt"

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
    size_scale = param.Number(default=400, bounds=(200, 1000), step=10)
    flip_ud = param.Boolean(default=False)
    flip_lr = param.Boolean(default=False)
    cplx_view = param.Selector(default="mag", objects=["mag", "phase", "real", "imag"])
    display_images = param.ListSelector(default=[], objects=[])

    # Color mapping
    cmap = param.Selector(default="gray", objects=VALID_COLORMAPS)
    colorbar_on = param.Boolean(default=True)
    colorbar_label = param.String(default="")

    # Slice Dimension Parameters
    dim_indices = param.Dict(
        default={}, doc="Mapping: dim_name -> int or categorical index"
    )

    # Crop Range Parameters
    lr_crop = param.Range(default=(0, 100), bounds=(0, 100), step=1)
    ud_crop = param.Range(default=(0, 100), bounds=(0, 100), step=1)

    # ROI-related parameters. TODO: clean these parameters up - don't need dups with self.ROI() obj
    roi_state = param.Integer(default=-1)
    roi_cmap = param.ObjectSelector(default="Same", objects=VALID_COLORMAPS + ["Same"])
    roi_loc = param.ObjectSelector(default="top_right", objects=roi.ROI_LOCATIONS)
    roi_zoom_scale = param.Number(default=2.0, bounds=(1.0, 10.0), step=0.1)
    roi_line_color = param.Color(default="red")
    roi_line_width = param.Integer(default=2)
    roi_zoom_order = param.Integer(default=1)
    roi_mode = param.ObjectSelector(default="inview", objects=["inview", "separate"])

    def __init__(
        self,
        data: hv.Dataset,
        vdims: Sequence[str],
        cdim: Optional[str] = None,
        clabs: Optional[Sequence[str]] = None,
        cat_dims: Optional[Dict[str, List]] = None,
        **params,
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

        with param.parameterized.discard_events(self):

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
                    # assume data categorical.
                    clabs = (
                        self.data.aggregate(self.cdim, np.mean).data[self.cdim].tolist()
                    )

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

            # Color map object. Auto-select "Quantitative" if inferred from named dimensions.
            if self._infer_quantitative_maptype() is not None:
                self.ColorMapper = QuantitativeColorMap(
                    self._infer_quantitative_maptype(), self.vmin, self.vmax
                )
                self.cmap = "Quantitative"
            else:
                self.ColorMapper = ColorMap(self.cmap)

            # ROI init
            self.ROI = roi.ROI()

        self.param.trigger("vmin", "vmax", "cmap")

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

        if self.cdim is not None:
            # Collate Case

            imgs = []

            for img_label in self.display_images:

                sliced_2d = self.data.select(
                    **{self.cdim: img_label}, **sdim_dict
                ).reduce([self.cdim] + self.sdims, np.mean)

                # order vdims in slice the same as self.vdims
                sliced_2d = sliced_2d.reindex(self.vdims)

                # set slice extent
                l, r = self.lr_crop
                u, d = self.ud_crop
                sliced_2d = sliced_2d[l:r, u:d]

                imgs.append(hv.Image(sliced_2d, label=img_label))
        else:
            # Single image case

            # Select slice indices for each dimension
            sliced_2d = (
                self.data.select(**sdim_dict)
                .reduce(self.sdims, np.mean)
                .reindex(self.vdims)
            )

            # set slice extent
            l, r = self.lr_crop
            u, d = self.ud_crop
            sliced_2d = sliced_2d[l:r, u:d]

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
        "roi_state",
    )
    @error.error_handler_decorator()
    def view(self) -> hv.Layout:
        """
        Return the formatted view of the data given the current slice indices.
        """

        self.update_cache()

        imgs = self.slice()

        # To start
        Ncols = len(imgs)

        new_im_size = (
            self.lr_crop[1] - self.lr_crop[0],
            self.ud_crop[1] - self.ud_crop[0],
        )

        if self.roi_mode == "separate" and self.roi_state == 2:

            roi_row = []

        for i in range(len(imgs)):

            # Potential colormap pre-processing
            imgs[i].data["Value"] = self.ColorMapper.preprocess_data(
                imgs[i].data["Value"]
            )

            # Apply complex view
            imgs[i].data["Value"] = CPLX_VIEW_MAP[self.cplx_view](imgs[i].data["Value"])

            # parameterized view options
            imgs[i] = imgs[i].opts(
                cmap=self.ColorMapper.get_cmap(),
                xaxis=None,
                yaxis=None,
                clim=(self.vmin, self.vmax),
                width=int(self.size_scale * new_im_size[0] / np.max(new_im_size)),
                height=int(self.size_scale * new_im_size[1] / np.max(new_im_size)),
                invert_yaxis=self.flip_ud,
                invert_xaxis=self.flip_lr,
                hooks=[_format_image],
            )

            # If ROI already defined, compute the ROI and integrate to composite depending on mode
            if self.roi_state == 2:

                x1, x2, y1, y2 = self.ROI.x1, self.ROI.x2, self.ROI.y1, self.ROI.y2

                # Get bounded region
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])

                bounding_box = hv.Bounds((x1, y1, x2, y2), label=imgs[i].label)

                bounding_box.opts(
                    color=self.roi_line_color,
                    line_width=self.roi_line_width,
                    show_legend=False,
                )

                # Show ROI in figure
                if self.roi_mode == "inview":

                    addnl_opts = dict(
                        xaxis=None,
                        yaxis=None,
                        clim=(self.vmin, self.vmax),
                    )

                    # Extract bounded region.
                    roi = self.ROI.get_inview_roi(imgs[i], addnl_opts=addnl_opts)

                # Show ROI in a row below main images (with equivalent widths)
                elif self.roi_mode == "separate":

                    addnl_opts = dict(
                        xaxis=None,
                        yaxis=None,
                        clim=(self.vmin, self.vmax),
                        shared_axes=False,
                        hooks=[_format_image],
                    )

                    # Get ROI in separate view
                    roi = self.ROI.get_separate_roi(imgs[i], addnl_opts=addnl_opts)

                    roi_row.append(roi)

                imgs[i] = imgs[i] * bounding_box

                # Put bounding box on final layer
                if self.roi_mode == "inview":
                    imgs[i] = imgs[i] * roi

        row = hv.Layout(imgs)

        """
        Building Overlay for ROI
        """

        if self.roi_state == 0:

            pointer = streams.SingleTap(x=-1, y=-1, source=imgs[0])

            def roi_state_1_callback(x, y):

                if x < 0 or y < 0:
                    return hv.HLine(0) * hv.VLine(0)

                # Update ROI variables
                self.ROI.x1 = x
                self.ROI.y1 = y

                # ROI State --> 1
                self.update_roi(1)

                return hv.HLine(y) * hv.VLine(x)

            row = row * hv.DynamicMap(roi_state_1_callback, streams=[pointer])

        elif self.roi_state == 1:

            pointer = streams.SingleTap(x=-1, y=-1, source=imgs[0])

            def roi_state_2_callback(x, y):

                if x < 0 or y < 0:
                    return hv.HLine(0) * hv.VLine(0)

                # Update ROI variables
                self.ROI.x2 = x
                self.ROI.y2 = y

                # ROI State --> 2
                self.update_roi(2)

                return hv.HLine(y) * hv.VLine(x)

            row = row * hv.DynamicMap(roi_state_2_callback, streams=[pointer])
            row = row * hv.HLine(self.ROI.y1) * hv.VLine(self.ROI.x1)

        """
        Add Colorbar elements (via dummy Image element)
        """

        if self.colorbar_on:

            Ncols += 1

            cb_h = int(self.size_scale * new_im_size[1] / np.max(new_im_size))

            # Add constant width for cbar
            cbar_const_w = 35 + 18 * int(
                self.colorbar_label is not None and len(self.colorbar_label) > 0
            )

            cbar_const_opts = dict(
                clim=(self.vmin, self.vmax),
                colorbar=True,
                colorbar_opts={
                    "title": self.colorbar_label,
                },
                width=int(
                    self.size_scale * (new_im_size[1] / np.max(new_im_size)) * 0.15
                    + cbar_const_w
                ),  # 5% maintained aspect
                colorbar_position="right",
                xaxis=None,
                yaxis=None,
                shared_axes=False,  # Unlink from holoviews shared toolbar
                hooks=[
                    _format_image,
                    _hide_image,
                    _format_colorbar,
                ],  # Hide the dummy glyph
            )

            main_cbar_fig = hv.Image(np.zeros((2, 2))).opts(
                cmap=self.ColorMapper.get_cmap(),
                height=cb_h,
                **cbar_const_opts,
            )

            row += main_cbar_fig

        # Add another row for ROI
        if self.roi_state == 2 and self.roi_mode == "separate":

            for roi_img in roi_row:
                row += roi_img

            if self.colorbar_on:

                roi_cbar_fig = hv.Image(np.zeros((2, 2))).opts(
                    cmap=self.ROI.cmap.get_cmap(),
                    height=roi_row[0].Image.ROI.opts["height"],
                    **cbar_const_opts,
                )

                row += roi_cbar_fig

        # set number of columns
        row = row.cols(Ncols)

        return pn.Row(row)

    def _infer_quantitative_maptype(self) -> Union[str, None]:
        """
        Determine if slice is quantitative.
        """
        for dim in self.cat_dims.keys():
            if self.dim_indices[dim].capitalize() in QUANTITATIVE_MAPTYPES:
                return self.dim_indices[dim].capitalize()

        return None

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
        with param.parameterized.discard_events(self):

            self.param.vmin.default = vmind
            self.param.vmin.bounds = vminb
            self.param.vmin.step = vmins
            self.param.vmax.default = vmaxd
            self.param.vmax.bounds = vmaxb
            self.param.vmax.step = vmaxs
            self.param.cmap.default = cmap

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
            self.vmin = np.percentile(data, 0.1)
            self.vmax = np.percentile(data, 99.9)

        self.param.trigger("vmin", "vmax")

    def update_display_image_list(self, display_images: Sequence[str]):

        with param.parameterized.discard_events(self):
            self.display_images = display_images

        self.param.trigger("display_images")

    def update_colormap(self):
        """
        Trigger for colormap change. Handles Quantitative colormap needs as well.
        """

        if self.cmap.capitalize() == "Quantitative":

            qmaptype = self._infer_quantitative_maptype()

            if qmaptype is not None:
                self.ColorMapper = QuantitativeColorMap(qmaptype, self.vmin, self.vmax)

            else:
                # TODO: parameterize what colormap should be default
                error.warning(
                    "Could not infer quantitative maptype. Using default colormap 'gray'."
                )
                self.ColorMapper = ColorMap("gray")
        else:
            self.ColorMapper = ColorMap(self.cmap)

        # Ensure roi tracks with colormap updates if paired
        if self.roi_cmap.capitalize() == "Same":
            with param.parameterized.discard_events(self):
                self.update_roi_colormap(self.roi_cmap)

        self.param.trigger("vmin", "vmax")

    def update_roi_colormap(self, new_cmap: str):

        self.roi_cmap = new_cmap

        if self.roi_cmap.capitalize() == "Same":

            self.ROI.cmap = self.ColorMapper

        elif self.roi_cmap.capitalize() == "Quantitative":

            qmaptype = self._infer_quantitative_maptype()

            if qmaptype is not None:
                self.ROI.cmap = QuantitativeColorMap(qmaptype, self.vmin, self.vmax)

            else:
                self.ROI.cmap = self.ColorMapper

        else:

            self.ROI.cmap = ColorMap(self.roi_cmap)

        self.param.trigger("roi_state")

    def update_roi_zoom_scale(self, new_zoom: float):

        self.roi_zoom_scale = new_zoom

        self.ROI.zoom_scale = self.roi_zoom_scale

        self.param.trigger("roi_state")

    def update_roi_loc(self, new_loc: str):

        self.roi_loc = new_loc

        self.ROI.roi_loc = self.roi_loc

        self.param.trigger("roi_state")

    def update_roi_lr_crop(self, new_lr_crop: Tuple[int, int]):

        self.ROI.x1 = new_lr_crop[0]
        self.ROI.x2 = new_lr_crop[1]

        self.param.trigger("roi_state")

    def update_roi_ud_crop(self, new_ud_crop: Tuple[int, int]):

        self.ROI.y1 = new_ud_crop[0]
        self.ROI.y2 = new_ud_crop[1]

        self.param.trigger("roi_state")

    def update_roi_line_color(self, new_color: str):

        self.roi_line_color = new_color

        self.ROI.color = new_color

        self.param.trigger("roi_state")

    def update_roi_line_width(self, new_width: int):

        self.roi_line_width = new_width

        self.ROI.line_width = new_width

        self.param.trigger("roi_state")

    def update_roi_zoom_order(self, new_order: int):

        self.roi_zoom_order = new_order

        self.ROI.zoom_order = new_order

        self.param.trigger("roi_state")

    def update_roi_mode(self, new_mode: int):

        if new_mode == 0:
            self.roi_mode = "separate"
        elif new_mode == 1:
            self.roi_mode = "inview"
        else:
            raise ValueError("Invalid ROI mode")

        self.param.trigger("roi_state")

    def update_roi(self, new_state):
        """
        ROI interactivity based on state.

        State -1:
        - No ROI active
        State 0:
        - User has clicked "Add ROI" button, popup for "Select Point 1" appears
        - SingleTap stream active, looking for point 1 to be added
        State 1:
        - Triggered after point 1 selected (some value is returned from SingleTap)
        - Vline1/Hline1 are locked in place
        - Popup for "Select Point 2" appears
        - SingleTap stream active, looking for point 2 to be added
        State 2:
        - Triggered after point 2 selected (some value is returned from SingleTap)
        - Bounding box over ROI locked in place
        - Widgets for ROI modification appear
        - ROI appears
        """

        prev_state = self.roi_state

        # State-based update and display corresponding message
        if prev_state == -1 and new_state == -1:

            pn.state.notifications.clear()
            pn.state.notifications.info(
                "No ROI active. Click 'Add ROI' to start adding an ROI.",
                duration=0,
            )

        elif prev_state >= 0 and new_state == -1:

            pn.state.notifications.clear()
            pn.state.notifications.info(
                "ROI cleared. Click 'Add ROI' to start adding an ROI.",
                duration=0,
            )

        elif prev_state >= 0 and new_state == 0:

            pn.state.notifications.clear()
            pn.state.notifications.info(
                "Resetting ROI. Click the first corner of the new ROI in the left-most plot.",
                duration=0,
            )

        elif prev_state == -1 and new_state == 0:

            pn.state.notifications.clear()
            pn.state.notifications.info(
                "Click the first corner of the ROI in the left-most plot.",
                duration=0,
            )

        elif prev_state == 0 and new_state == 1:

            pn.state.notifications.clear()
            pn.state.notifications.info(
                "Click the second corner of the ROI in the left-most plot.",
                duration=0,
            )

        elif prev_state == 1 and new_state == 2:

            pn.state.notifications.clear()
            pn.state.notifications.info(
                "ROI added. Click 'View ROI in-figure' to see the ROI in the main plot.",
                duration=5000,
            )

            with param.parameterized.discard_events(self):
                self.update_roi_colormap(self.roi_cmap)

        else:

            raise ValueError(
                f"Invalid ROI state transition: {prev_state} -> {new_state}"
            )

        # Trigger
        self.roi_state = new_state
