"""
Slicers: Defined as classes that take N-dimensional data and can return a 2D view of that data given some input
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import holoviews as hv
import numpy as np
import panel as pn
import param
from bokeh.core.properties import value as bokeh_value
from holoviews import streams

from . import config, error, metrics, profilers, roi, themes, utils
from .cmap.cmap import (
    QUANTITATIVE_MAPTYPES,
    VALID_COLORMAPS,
    VALID_ERROR_COLORMAPS,
    ColorMap,
    QuantitativeColorMap,
)
from .enums import METRICS_STATE, ROI_LOCATION, ROI_STATE, ROI_VIEW_MODE
from .utils import CPLX_VIEW_MAP

hv.extension("bokeh")


def _format_image(plot, element):
    """
    For setting image theme (light/dark mode).
    """

    # Enforce theme
    plot.state.background_fill_color = themes.VIEW_THEME.background_color
    plot.state.border_fill_color = themes.VIEW_THEME.background_color

    # Constant height for the figure title
    if plot.state.title.text_font_size[-2:] in ["px", "pt"]:
        tfs = int(plot.state.title.text_font_size[:-2]) * 2 + plot.border
        plot.state.height = plot.height + tfs
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
    title_font_size = param.Number(default=12, bounds=(2, 36), step=1)
    vmin = param.Number(default=0.0)
    vmax = param.Number(default=1.0)
    size_scale = param.Number(default=400, bounds=(200, 1000), step=10)
    flip_ud = param.Boolean(default=False)
    flip_lr = param.Boolean(default=False)
    cplx_view = param.ObjectSelector(
        default="mag", objects=["mag", "phase", "real", "imag"]
    )
    display_images = param.ListSelector(default=[], objects=[])

    # Color mapping
    cmap = param.ObjectSelector(default="gray", objects=VALID_COLORMAPS)
    colorbar_on = param.Boolean(default=True)
    colorbar_label = param.String(default="")

    # Slice Dimension Parameters
    dim_indices = param.Dict(
        default={}, doc="Mapping: dim_name -> int or categorical index"
    )

    # Crop Range Parameters
    lr_crop = param.Range(default=(0, 100), bounds=(0, 100), step=1)
    ud_crop = param.Range(default=(0, 100), bounds=(0, 100), step=1)

    # ROI-related parameters.
    roi_state = param.ObjectSelector(default=ROI_STATE.INACTIVE, objects=ROI_STATE)
    roi_cmap = param.ObjectSelector(default="Same", objects=VALID_COLORMAPS + ["Same"])
    roi_mode = param.ObjectSelector(
        default=ROI_VIEW_MODE.Overlayed, objects=ROI_VIEW_MODE
    )

    # Difference Map Related Parameters
    metrics_reference = param.ObjectSelector(default="", objects=[])
    metrics_state = param.ObjectSelector(
        default=METRICS_STATE.INACTIVE, objects=METRICS_STATE
    )
    error_map_scale = param.Number(default=1.0)
    error_map_type = param.ObjectSelector(
        default="L1Diff", objects=metrics.MAPPABLE_METRICS
    )
    error_map_cmap = param.ObjectSelector(
        default="inferno", objects=VALID_ERROR_COLORMAPS
    )
    metrics_text_types = param.ListSelector(default=[], objects=metrics.FULL_METRICS)
    metrics_text_location = param.ObjectSelector(
        default=ROI_LOCATION.TOP_LEFT, objects=ROI_LOCATION
    )
    metrics_text_font_size = param.Number(default=12, bounds=(5, 24), step=1)

    # Rebuilding figure
    rebuild_figure_flag = param.Boolean(default=False)

    def __init__(
        self,
        data: hv.Dataset,
        vdims: Sequence[str],
        cdim: Optional[str] = None,
        clabs: Optional[Sequence[str]] = None,
        cat_dims: Optional[Dict[str, List]] = None,
        cfg: Optional[Dict[str, str]] = None,
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

        # Not the most efficient, but now update from config if supplied. mimics user having manually
        # set all parameters as desired.
        # if supplied in config, vdims already taken care of by the viewer
        from_config = cfg is not None

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
                self.clabs = ["Image"]
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

            # Set parameter attributes
            if from_config:
                config.deserialize_parameters(self, cfg["slicer_config"])

                # Excpeption for display images
                if not cfg["metadata"]["same_images"]:
                    self.display_images = self.clabs
                    self.param.display_images.objects = self.clabs
                    self.param.metrics_reference.objects = self.clabs
                    self.metrics_reference = self.clabs[0]

                self.ROI = roi.ROI(config=cfg["roi_config"])
                self.update_cplx_view(self.cplx_view, recompute_min_max=False)
                self.update_colormap()
                self.update_roi_colormap(self.roi_cmap)

                self.DifferenceColorMapper = ColorMap(self.error_map_cmap)

            else:
                # Update color limits with default
                self.update_cplx_view(self.cplx_view)

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

                # Diff map and metrics init
                self.param.metrics_reference.objects = self.clabs
                self.metrics_reference = self.clabs[0]  # Default to the first one
                self.DifferenceColorMapper = ColorMap(self.error_map_cmap)

        # Initialize static instance of plot through self.Figure
        self.build_figure_objects(self.slice())

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

    def slice(self) -> Dict:
        """
        Return the slice of the hv.Dataset given the current slice indices.

        Output is a dictionary, where keys are:

        - "img": Dict[hv.Dataset] for each main image
        - "error_map": Dict[hv.Dataset] for each error map, if applicable
        - "metrics": Dict[Dict[float]] for each dataset, if applicable

        """

        # Dimensions to select
        sdim_dict = {dim: self.dim_indices[dim] for dim in self.sdims}

        out_dict = {}

        imgs = {}

        # edge case where user has selected metrics but not the reference image for display
        slice_imgs = self.display_images
        if (
            self.metrics_state is not METRICS_STATE.INACTIVE
            and self.metrics_reference not in self.display_images
        ):
            slice_imgs = [self.metrics_reference] + self.display_images

        if self.cdim is not None:
            # Collate Case
            for img_label in slice_imgs:

                sliced_2d = self.data.select(
                    **{self.cdim: img_label}, **sdim_dict
                ).reduce([self.cdim] + self.sdims, np.mean)

                # order vdims in slice the same as self.vdims
                sliced_2d = sliced_2d.reindex(self.vdims)

                # set slice extent
                l, r = self.lr_crop
                u, d = self.ud_crop
                sliced_2d = sliced_2d[l:r, u:d]

                imgs[img_label] = sliced_2d
        else:
            # Single image case
            sliced_2d = (
                self.data.select(**sdim_dict)
                .reduce(self.sdims, np.mean)
                .reindex(self.vdims)
            )

            # set slice extent
            l, r = self.lr_crop
            u, d = self.ud_crop
            sliced_2d = sliced_2d[l:r, u:d]

            imgs[img_label] = sliced_2d

        # Complex reduction
        for k in imgs.keys():
            imgs[k].data["Value"] = CPLX_VIEW_MAP[self.cplx_view](imgs[k].data["Value"])

        """
        Gather metrics and difference maps of slice
        """
        # TODO: integrate caching
        if self.metrics_state is not METRICS_STATE.INACTIVE:
            # Gather arrays
            ref_img = np.copy(imgs[self.metrics_reference].data["Value"])

            tar_keys = [k for k in imgs.keys() if k != self.metrics_reference]

            metrics_dict = {}
            error_maps = {}

            for k in tar_keys:

                tar_img = np.copy(imgs[k].data["Value"])

                # Don't normalize with quantitative maps - we care about absolute
                if (
                    self._infer_quantitative_maptype() is None
                    and self.cplx_view != "phase"
                ):
                    tar_img = utils.normalize(
                        tar_img, ref_img, ofs=True, mag=np.iscomplexobj(tar_img)
                    )

                if self.metrics_state in [METRICS_STATE.TEXT, METRICS_STATE.ALL]:
                    metrics_dict[k] = {}
                    for metric in self.metrics_text_types:
                        metrics_dict[k][metric] = metrics.METRIC_CALLABLES[metric](
                            tar_img,
                            ref_img,
                            isphase=self.cplx_view == "phase",
                        )

                if self.metrics_state in [METRICS_STATE.MAP, METRICS_STATE.ALL]:
                    error_map = metrics.METRIC_CALLABLES[self.error_map_type](
                        tar_img,
                        ref_img,
                        return_map=True,
                        isphase=self.cplx_view == "phase",
                    )
                    error_map = self.DifferenceColorMapper.preprocess_data(error_map)
                    error_maps[k] = utils.clone_dataset(imgs[k], error_map, link=False)

            # Ref
            error_maps[self.metrics_reference] = utils.clone_dataset(
                imgs[self.metrics_reference], np.zeros_like(ref_img), link=False
            )

            out_dict["metrics"] = metrics_dict
            out_dict["error_map"] = error_maps

        # don't want to display metrics reference
        if (
            self.metrics_state is not METRICS_STATE.INACTIVE
            and self.metrics_reference not in self.display_images
        ):
            imgs.pop(self.metrics_reference)

        # Preprocessing for color map data
        for k in imgs.keys():
            imgs[k].data["Value"] = self.ColorMapper.preprocess_data(
                imgs[k].data["Value"]
            )

        out_dict["img"] = imgs

        return out_dict

    def _build_figure_opts(self):
        """
        Hiding a bunch of building opts for figure here. This gets messy...
        """

        # TODO: move these constants
        BORDER_SIZE = 3  # a.u.
        CBAR_CONST_WIDTH = 35  # pix
        CBAR_TEXT_WIDTH = 18  # pix
        CBAR_SCALE_RATIO = 0.15  # fraction [0, 1]

        # Determine sizes
        new_im_size = (
            self.lr_crop[1] - self.lr_crop[0],
            self.ud_crop[1] - self.ud_crop[0],
        )
        main_width = self.size_scale * new_im_size[0] / np.max(new_im_size)
        main_height = self.size_scale * new_im_size[1] / np.max(new_im_size)

        # Options shared across all renderables
        shared_opts = dict(
            clim=(self.vmin, self.vmax),
            xaxis=None,
            yaxis=None,
            border=BORDER_SIZE,
        )

        # Image options
        im_opts = dict(
            cmap=self.ColorMapper.get_cmap(),
            width=int(main_width),
            height=int(main_height),
            invert_yaxis=self.flip_ud,
            invert_xaxis=self.flip_lr,
            fontscale=(self.title_font_size / 12),
            hooks=[_format_image],
            **shared_opts,
        )

        # Any rendered line object (border, hv.VLine, etc)
        line_opts = dict(
            color=self.ROI.color,
            line_width=self.ROI.line_width,
            show_legend=False,
        )

        # Opts to pass to ROI
        roi_opts = dict(
            **shared_opts,
        )
        if self.roi_mode == ROI_VIEW_MODE.Separate:
            roi_opts.update(
                dict(
                    shared_axes=False,
                    hooks=[_format_image],
                )
            )

        # Colorbar opts
        cbar_const_w = CBAR_CONST_WIDTH
        if self.colorbar_label is not None and len(self.colorbar_label) > 0:
            cbar_const_w += CBAR_TEXT_WIDTH
        cbar_width = int(main_height * CBAR_SCALE_RATIO + cbar_const_w)

        cbar_opts = dict(
            colorbar=True,
            colorbar_opts={
                "title": self.colorbar_label,
            },
            width=cbar_width,
            colorbar_position="right",
            shared_axes=False,  # Unlink from holoviews shared toolbar
            fontscale=(
                self.title_font_size / 12
            ),  # div by 12 because fontscale=1 is font=12pt
            hooks=[
                _format_image,
                _hide_image,
                _format_colorbar,
            ],  # Hide the dummy glyph
            **shared_opts,
        )

        # Difference map
        diff_opts = im_opts.copy()
        diff_opts["cmap"] = self.DifferenceColorMapper.get_cmap()

        if self.error_map_type == "SSIM":
            diff_opts["clim"] = (0, 1)
        elif self.error_map_type == "Diff":
            diff_opts["clim"] = (
                -self.vmax / self.error_map_scale,
                self.vmax / self.error_map_scale,
            )
        else:
            diff_opts["clim"] = (0, self.vmax / self.error_map_scale)

        # Diff map colorbar
        diff_cbar_opts = cbar_opts.copy()
        diff_cbar_opts["clim"] = diff_opts["clim"]
        diff_cbar_opts.pop("colorbar_opts")

        if self.colorbar_label is not None and len(self.colorbar_label) > 0:
            if self.error_map_type == "SSIM":
                diff_cbar_opts["colorbar_opts"] = dict(title="SSIM")
            else:
                diff_cbar_opts["colorbar_opts"] = dict(
                    title=f"Difference ({self.error_map_scale}x)"
                )

        opts = dict(
            height=main_height,
            width=main_width,
            im_opts=im_opts,
            line_opts=line_opts,
            roi_opts=roi_opts,
            cbar_opts=cbar_opts,
            diff_opts=diff_opts,
            diff_cbar_opts=diff_cbar_opts,
        )

        return opts

    def build_figure_objects(self, input_data: dict):
        """
        Build the figure objects for the current slice indices.
        """

        self.update_cache()

        opts = self._build_figure_opts()

        # re-order so ref is always on the left or right
        img_dict = input_data["img"]
        fig_image_names = list(img_dict.keys())

        # lbrt bounds
        main_lbrt = (
            hv.Image(img_dict[fig_image_names[0]]).opts(**opts["im_opts"]).bounds.lbrt()
        )

        # Reorder images. TODO: maybe put ref at end instead of beginning? or parameterize?
        if self.metrics_reference is not None and (
            self.metrics_reference in fig_image_names
        ):
            ref_idx = fig_image_names.index(self.metrics_reference)
            fig_image_names.pop(ref_idx)
            fig_image_names = [self.metrics_reference] + fig_image_names

        metrics_dict = None
        if self.metrics_state in [METRICS_STATE.TEXT, METRICS_STATE.ALL]:
            metrics_dict = input_data["metrics"]

        error_dict = None
        if self.metrics_state in [METRICS_STATE.MAP, METRICS_STATE.ALL]:
            error_dict = input_data["error_map"]

        # build images and pipes
        self._image_pipes = {}
        self._metrics_pipe = {}
        imgs = []
        img_labels = {}
        for k in fig_image_names:

            # Extract metrics
            image_name = k

            if (
                self.metrics_state != METRICS_STATE.INACTIVE
            ) and k == self.metrics_reference:
                image_name = f"{k} (Ref)"

            img_labels[k] = image_name

            pipe = streams.Pipe(data=img_dict[k])

            self._image_pipes[k] = pipe

            def _img_callback(data, image_name=image_name):
                return hv.Image(data, label=image_name).opts(**opts["im_opts"])

            imgs.append(
                hv.DynamicMap(
                    _img_callback,
                    streams=[pipe],
                ).opts(
                    title=image_name,
                )
            )
            # send data
            self._image_pipes[k].send(img_dict[k])

            if metrics_dict and k in metrics_dict.keys():
                # to get locations
                tx_pad = 3

                # determine loc
                effective_location = utils.get_effective_location(
                    self.metrics_text_location,
                    self.flip_lr,
                    self.flip_ud,
                )

                if effective_location == ROI_LOCATION.TOP_LEFT:
                    tx = main_lbrt[0] + tx_pad
                    ty = main_lbrt[3] - tx_pad
                elif effective_location == ROI_LOCATION.TOP_RIGHT:
                    tx = main_lbrt[2] - tx_pad
                    ty = main_lbrt[3] - tx_pad
                elif effective_location == ROI_LOCATION.BOTTOM_LEFT:
                    tx = main_lbrt[0] + tx_pad
                    ty = main_lbrt[1] + tx_pad
                elif effective_location == ROI_LOCATION.BOTTOM_RIGHT:
                    tx = main_lbrt[2] - tx_pad
                    ty = main_lbrt[1] + tx_pad

                t_halign = self.metrics_text_location.value.split(" ")[1].lower()
                t_valign = self.metrics_text_location.value.split(" ")[0].lower()

                # set up dynamicmap for text
                self._metrics_pipe[k] = streams.Pipe(data=metrics_dict[k])

                def _met_text_callback(
                    data, tx=tx, ty=ty, t_halign=t_halign, t_valign=t_valign
                ):

                    txt = ""
                    for j, (mk, mv) in enumerate(data.items()):
                        txt += f"{mk}: {mv:.2f}"
                        if j < len(data) - 1:
                            txt += "\n"

                    return hv.Text(
                        tx,
                        ty,
                        txt,
                        halign=t_halign,
                        valign=t_valign,
                        fontsize=self.metrics_text_font_size,
                    ).opts(
                        text_font=bokeh_value(themes.VIEW_THEME.text_font),
                        text_color=themes.VIEW_THEME.text_color,
                    )

                MetricsText = hv.DynamicMap(
                    _met_text_callback,
                    streams=[self._metrics_pipe[k]],
                )

                imgs[-1] = imgs[-1] * MetricsText

        # To start
        Ncols = len(imgs)

        self._roi_pipes = {}

        if (
            self.roi_mode == ROI_VIEW_MODE.Separate
            and self.roi_state == ROI_STATE.ACTIVE
        ):
            roi_row = []

        # If ROI already defined, compute the ROI and integrate to composite depending on mode
        if self.roi_state == ROI_STATE.ACTIVE:
            for i, k in enumerate(fig_image_names):

                bounding_box = hv.Bounds(self.ROI.lbrt(), label=img_labels[k]).opts(
                    **opts["line_opts"],
                )

                # Show ROI in figure
                if self.roi_mode == ROI_VIEW_MODE.Overlayed:
                    # Extract bounded region.
                    roi_fig, roi_pipe = self.ROI.get_overlay_roi(
                        img_dict[k],
                        img_labels[k],
                        flip_lr=self.flip_lr,
                        flip_ud=self.flip_ud,
                        addnl_opts=opts["roi_opts"],
                    )

                # Show ROI in a row below main images (with equivalent widths)
                elif self.roi_mode == ROI_VIEW_MODE.Separate:
                    # Get ROI in separate view
                    roi_fig, roi_pipe = self.ROI.get_separate_roi(
                        img_dict[k],
                        img_labels[k],
                        width=int(opts["width"]),
                        addnl_opts=opts["roi_opts"],
                    )

                    roi_row.append(roi_fig)

                imgs[i] = imgs[i] * bounding_box

                # Put bounding box on final layer
                if self.roi_mode == ROI_VIEW_MODE.Overlayed:
                    imgs[i] = imgs[i] * roi_fig

                self._roi_pipes[k] = roi_pipe

        row = hv.Layout(imgs)

        """
        Building Overlay for ROI
        """

        def gen_hline(y):
            return hv.HLine(y).opts(**opts["line_opts"])

        def gen_vline(x):
            return hv.VLine(x).opts(**opts["line_opts"])

        if self.roi_state == ROI_STATE.FIRST_SELECTION:
            pointer = streams.SingleTap(x=-1, y=-1)

            def first_selection_callback(x, y):
                if x < 0 or y < 0:
                    return gen_hline(0) * gen_vline(0)
                self.ROI.point1 = roi.Point(x, y)

                self.update_roi_state(ROI_STATE.SECOND_SELECTION)
                return gen_hline(y) * gen_vline(x)

            row = row * hv.DynamicMap(first_selection_callback, streams=[pointer])

        elif self.roi_state == ROI_STATE.SECOND_SELECTION:
            pointer = streams.SingleTap(x=-1, y=-1)

            def second_selection_callback(x, y):
                if x < 0 or y < 0:
                    return gen_hline(0) * gen_vline(0)

                self.ROI.point2 = roi.Point(x, y)
                self.update_roi_state(ROI_STATE.ACTIVE)
                return gen_hline(y) * gen_vline(x)

            # Prevent dynamic map from adjusting size of figure
            row = row * gen_hline(self.ROI.point1.y) * gen_vline(self.ROI.point1.x)
            row = row * hv.DynamicMap(second_selection_callback, streams=[pointer])

        """
        Add Colorbar elements (via dummy Image element)
        """

        if self.colorbar_on:
            Ncols += 1

            main_cbar_fig = hv.Image(np.zeros((2, 2)), bounds=main_lbrt).opts(
                cmap=self.ColorMapper.get_cmap(),
                height=int(opts["height"]),
                **opts["cbar_opts"],
            )

            row += main_cbar_fig

        # Add another row for ROI
        if (
            self.roi_state == ROI_STATE.ACTIVE
            and self.roi_mode == ROI_VIEW_MODE.Separate
        ):

            for roi_img in roi_row:
                row += roi_img

            if self.colorbar_on:
                roi_cbar_fig = hv.Image(np.zeros((2, 2))).opts(
                    cmap=self.ROI.cmap.get_cmap(),
                    height=int(opts["width"] * self.ROI.height() / self.ROI.width()),
                    **opts["cbar_opts"],
                )

                row += roi_cbar_fig

        # Add row for difference maps
        if self.metrics_state in [METRICS_STATE.MAP, METRICS_STATE.ALL]:

            # Build difference map
            diff_imgs = []
            self._diffmap_pipes = {}
            for k in fig_image_names:
                name = k
                diff_pipe = streams.Pipe(data=error_dict[k])

                # No pipe for reference
                if k == self.metrics_reference:

                    ref_diff_img = hv.Image(
                        error_dict[k],
                    ).opts(
                        visible=False,
                        shared_axes=False,
                        **opts["diff_opts"],
                    )

                    diff_imgs.append(ref_diff_img)

                else:
                    self._diffmap_pipes[k] = diff_pipe

                    if self.error_map_type == "SSIM":
                        label = f"{name} (SSIM)"
                    else:
                        label = f"Diff ({self.error_map_scale}x)"

                    def _diff_callback(data):
                        return hv.Image(data, label=label).opts(**opts["diff_opts"])

                    diff_imgs.append(
                        hv.DynamicMap(
                            _diff_callback,
                            streams=[diff_pipe],
                        ).opts(
                            title=label,
                        )
                    )

                    # send data
                    self._diffmap_pipes[k].send(error_dict[k])

            diff_row = hv.Layout(diff_imgs)

            # Add colorbar for difference map
            if self.colorbar_on:
                diff_cbar_fig = hv.Image(np.zeros((2, 2))).opts(
                    cmap=self.DifferenceColorMapper.get_cmap(),
                    height=int(opts["height"]),
                    **opts["diff_cbar_opts"],
                )

                diff_row += diff_cbar_fig

            row += diff_row

        # set number of columns
        row = row.cols(Ncols)

        # Set attributes
        self.Figure = pn.Row(row)

    def update_figure(self, input_data: Dict[str, dict]):
        """
        Update the figure data in-place given the current slice indices.
        """
        assert self._image_pipes is not None, "Figure not initialized"

        # Send image data through pipes
        imgs_dict = input_data["img"]
        for k in imgs_dict.keys():
            self._image_pipes[k].send(imgs_dict[k])

            # Send ROI data
            if self.roi_state == ROI_STATE.ACTIVE:
                self._roi_pipes[k].send(imgs_dict[k])

            # Send metrics computations and error maps
            if (
                self.metrics_state in [METRICS_STATE.MAP, METRICS_STATE.ALL]
                and k != self.metrics_reference
            ):
                self._diffmap_pipes[k].send(input_data["error_map"][k])
            if (
                self.metrics_state in [METRICS_STATE.TEXT, METRICS_STATE.ALL]
                and k != self.metrics_reference
            ):
                self._metrics_pipe[k].send(input_data["metrics"][k])

    @param.depends(
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
        "error_map_scale",
        "metrics_state",
        "title_font_size",
        watch=True,
    )
    @error.error_handler_decorator()
    def rebuild_figure(self):
        """
        Clear figure and rebuild if needed
        """
        self.Figure = None
        self.rebuild_figure_flag = True

    @param.depends("dim_indices", "rebuild_figure_flag")
    @error.error_handler_decorator()
    @profilers.profile_decorator(
        enable=False
    )  # Print call information or log to file for debugging
    def view(self) -> hv.Layout:
        """
        Return the formatted view of the data given the current slice indices.
        """

        # Hold/unhold is necessary for making figure update "atomized". Not the best solution because
        # there can be confusion between document state if multiple view calls are made before rendering
        # is updated.
        atomize = pn.state.curdoc and not self.rebuild_figure_flag

        if atomize:
            pn.state.curdoc.hold()

        # New data to display
        slice_dict = self.slice()

        if self.rebuild_figure_flag:
            with param.parameterized.discard_events(self):
                self.rebuild_figure_flag = False

            self.build_figure_objects(slice_dict)

        else:
            self.update_figure(slice_dict)

        if atomize:
            pn.state.curdoc.unhold()

        return self.Figure

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
            self.param.lr_crop.bounds = (0, int(self.img_dims[0]))
            self.param.ud_crop.bounds = (0, int(self.img_dims[1]))
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
        self.param.trigger("lr_crop", "ud_crop")

    def update_cplx_view(self, new_cplx_view: str, recompute_min_max: bool = True):

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
            if recompute_min_max:
                cplx_callable = CPLX_VIEW_MAP[self.cplx_view]
                d = np.stack(
                    [cplx_callable(self.data[v.name]) for v in self.data.vdims]
                )
                mn = np.min(d)
                mx = np.max(d)

                vmind = mn
                vminb = (mn, mx)
                vmins = (mx - mn) / VSTEP_INTERVAL
                vmaxd = mx
                vmaxb = (mn, mx)
                vmaxs = (mx - mn) / VSTEP_INTERVAL
            else:
                # for loading from config
                mn = self.vmin
                mx = self.vmax
                vmind = self.vmin
                vminb = (self.param.vmin.bounds[0], self.param.vmin.bounds[1])
                vmins = self.param.vmin.step
                vmaxd = self.vmax
                vmaxb = (self.param.vmax.bounds[0], self.param.vmax.bounds[1])
                vmaxs = self.param.vmax.step

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

        # Trigger
        self.param.trigger("vmin", "vmax", "cmap")

    def autoscale_clim(self):
        """
        For given slice, automatically set vmin and vmax to min and max of data
        """

        data = np.stack([d.data["Value"] for d in self.slice()["img"].values()])

        with param.parameterized.discard_events(self):
            self.vmin = max(self.param.vmin.bounds[0], np.percentile(data, 0.1))
            self.vmax = min(self.param.vmin.bounds[1], np.percentile(data, 99.9))

        self.param.trigger("vmin", "vmax")

    def update_display_image_list(self, display_images: Sequence[str]):
        self.display_images = display_images

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

        self.param.trigger("cmap")

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

        self.ROI.zoom_scale = new_zoom
        self.param.trigger("roi_state")

    def update_roi_loc(self, new_loc: str):

        self.ROI.roi_loc = ROI_LOCATION(new_loc)
        self.param.trigger("roi_state")

    def update_roi_lr_crop(self, new_lr_crop: Tuple[int, int]):

        self.ROI.set_xrange(*new_lr_crop)
        self.param.trigger("roi_state")

    def update_roi_ud_crop(self, new_ud_crop: Tuple[int, int]):

        self.ROI.set_yrange(*new_ud_crop)
        self.param.trigger("roi_state")

    def update_roi_line_color(self, new_color: str):

        self.ROI.color = new_color
        self.param.trigger("roi_state")

    def update_roi_line_width(self, new_width: int):

        self.ROI.line_width = new_width
        self.param.trigger("roi_state")

    def update_roi_zoom_order(self, new_order: int):

        self.ROI.zoom_order = new_order
        self.param.trigger("roi_state")

    def update_roi_mode(self, new_mode: int):

        self.roi_mode = ROI_VIEW_MODE(new_mode)
        self.param.trigger("roi_state")

    def update_roi_state(self, new_state: ROI_STATE):
        """
        Enforce setting ROI based on interactive state.
        """

        prev_state = self.roi_state

        # State-based update and display corresponding message
        if prev_state == ROI_STATE.INACTIVE and new_state == ROI_STATE.INACTIVE:
            pn.state.notifications.clear()
            pn.state.notifications.info(
                "No ROI active. Click 'Draw ROI' to start adding an ROI.",
                duration=0,
            )

        elif prev_state > ROI_STATE.INACTIVE and new_state == ROI_STATE.INACTIVE:
            pn.state.notifications.clear()
            pn.state.notifications.info(
                "ROI cleared. Click 'Draw ROI' to start adding an ROI.",
                duration=3000,
            )

        elif prev_state > ROI_STATE.INACTIVE and new_state == ROI_STATE.FIRST_SELECTION:
            pn.state.notifications.clear()
            pn.state.notifications.info(
                "Resetting ROI. Click the first corner of the new ROI in any top-row image.",
                duration=0,
            )

        elif (
            prev_state == ROI_STATE.INACTIVE and new_state == ROI_STATE.FIRST_SELECTION
        ):
            pn.state.notifications.clear()
            pn.state.notifications.info(
                "Click the first corner of the ROI in any top-row image.",
                duration=0,
            )

        elif (
            prev_state == ROI_STATE.FIRST_SELECTION
            and new_state == ROI_STATE.SECOND_SELECTION
        ):
            pn.state.notifications.clear()
            pn.state.notifications.info(
                "Click the second corner of the ROI in any image.",
                duration=0,
            )

        elif prev_state == ROI_STATE.SECOND_SELECTION and new_state == ROI_STATE.ACTIVE:
            pn.state.notifications.clear()
            pn.state.notifications.info(
                "ROI added. Remove ROI with 'Clear ROI' button, or remove from overlay by un-checcking 'ROI Overlay Enabled'.",  # noqa E501
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

    def update_reference_dataset(self, new_ref: str):
        self.metrics_reference = new_ref
        if self.metrics_state != METRICS_STATE.INACTIVE:
            self.param.trigger("metrics_state")

    def update_error_map_type(self, new_type: str):
        self.error_map_type = new_type
        if self.metrics_state in [METRICS_STATE.MAP, METRICS_STATE.ALL]:
            self.param.trigger("metrics_state")

    def update_metrics_text_types(self, new_metrics: List[str]):
        self.metrics_text_types = new_metrics
        if self.metrics_state in [METRICS_STATE.TEXT, METRICS_STATE.ALL]:
            self.param.trigger("metrics_state")

    def update_metrics_state(self, new_metrics_state: METRICS_STATE):
        """
        Update metrics and error map state.
        """
        self.metrics_state = new_metrics_state

    def update_error_map_scale(self, new_scale: float):
        with param.parameterized.discard_events(self):
            self.error_map_scale = new_scale
        if self.metrics_state in [METRICS_STATE.MAP, METRICS_STATE.ALL]:
            self.param.trigger("error_map_scale")

    def update_error_map_cmap(self, new_cmap: str):
        self.error_map_cmap = new_cmap
        self.DifferenceColorMapper = ColorMap(new_cmap)
        if self.metrics_state in [METRICS_STATE.MAP, METRICS_STATE.ALL]:
            self.param.trigger("error_map_scale")

    def update_metrics_text_font_size(self, new_size: int):
        self.metrics_text_font_size = new_size
        if self.metrics_state in [METRICS_STATE.TEXT, METRICS_STATE.ALL]:
            self.param.trigger("metrics_state")

    def update_metrics_text_location(self, new_loc: ROI_LOCATION):
        self.metrics_text_location = new_loc
        if self.metrics_state in [METRICS_STATE.TEXT, METRICS_STATE.ALL]:
            self.param.trigger("metrics_state")

    def autoformat_error_map(self):
        """
        Automatically infer the best format to view error maps in
        """

        if self.metrics_state not in [METRICS_STATE.MAP, METRICS_STATE.ALL]:
            error.warning("Error maps are not enabled.")
            return

        if self.error_map_type == "SSIM":
            self.error_map_cmap = "Grays"
            self.DifferenceColorMapper = ColorMap(self.error_map_cmap)

        elif self.error_map_type in ["L1Diff", "L2Diff"]:

            if self.DifferenceColorMapper.cmap == "Grays":
                self.error_map_cmap = "inferno"
                self.DifferenceColorMapper = ColorMap(self.error_map_cmap)

            error_max = self._get_max_err()

            if error_max > 1e-10:
                self.error_map_scale = round(self.vmax / error_max, 1)

        elif self.error_map_type == "Diff":
            self.error_map_cmap = "RdBu"
            self.DifferenceColorMapper = ColorMap(self.error_map_cmap)
            error_max = np.abs(self._get_max_err())

            if error_max > 1e-10:
                error_max = np.abs(self._get_max_err())
                self.error_map_scale = round(self.vmax / error_max, 1)

        elif self.error_map_type == "RelativeL1":
            self.error_map_cmap = "inferno"
            self.DifferenceColorMapper = ColorMap(self.error_map_cmap)
            self.error_map_scale = round(self.vmax)

        else:
            error.warning("Error map type does not have autoformat.")

    def _get_max_err(self):
        """
        Get the max error for the current slice
        """

        error_data = self.slice()["error_map"]

        if self.metrics_reference in error_data:
            error_data.pop(self.metrics_reference)

        error_np = np.stack([d.data["Value"] for d in error_data.values()])
        error_np[np.isnan(error_np)] = 0

        return np.percentile(error_np, 99.9)
