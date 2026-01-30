"""
Slicers: Defined as classes that take N-dimensional data and can return a 2D view of that data given some input
"""

import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import holoviews as hv
import numpy as np
import panel as pn
import param
from bokeh.core.properties import value as bokeh_value
from bokeh.models import WheelZoomTool
from bokeh.models.formatters import BasicTickFormatter
from holoviews import streams

from . import config, error, metrics, profilers, roi, themes, utils
from .cmap.cmap import (
    QUANTITATIVE_MAPTYPES,
    VALID_COLORMAPS,
    VALID_ERROR_COLORMAPS,
    ColorMap,
    QuantitativeColorMap,
)
from .enums import METRICS_STATE, POPUP_LOCATION, ROI_LOCATION, ROI_STATE, ROI_VIEW_MODE
from .metrics import ERROR_TOL, TOL
from .utils import CPLX_VIEW_MAP, pprint_str, round_str

hv.extension("bokeh")


@dataclass
class ClimSettings:
    vmin: float
    vmax: float
    bound_min: float
    bound_max: float
    step: float
    cmap: str


def _bokeh_disable_wheel_zoom_tool(plot, element):
    """Disable Bokeh wheel zoom so scroll can drive slice index."""
    tools_to_remove = []
    for tool in plot.state.toolbar.tools:
        if isinstance(tool, WheelZoomTool):
            tools_to_remove.append(tool)
    for tool in tools_to_remove:
        plot.state.toolbar.tools.remove(tool)


def _get_format_image(
    text_font: str,
    title_visible: bool = True,
    grid_visible: bool = False,
):
    """Return a hook that applies theme and title/grid to the plot."""

    def _format_image(plot, element):
        """Apply theme (background, title, grid) to plot."""
        # Enforce theme
        plot.state.background_fill_color = themes.VIEW_THEME.background_color

        if grid_visible:
            plot.state.border_fill_color = themes.VIEW_THEME.accent_color
        else:
            plot.state.border_fill_color = themes.VIEW_THEME.background_color

        plot.state.title.text_color = themes.VIEW_THEME.text_color
        plot.state.title.text_font = text_font

        # Constant height for the figure title
        if title_visible:
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
        else:
            plot.state.height = plot.height

        # Color to match theme
        plot.state.outline_line_color = themes.VIEW_THEME.background_color
        plot.state.outline_line_alpha = 1.0

        # Center title above image
        plot.state.title.align = "center"

    return _format_image


def _hide_image(plot, element):
    """Hide image glyphs so only colorbar is visible."""
    for r in plot.state.renderers:
        if hasattr(r, "glyph"):
            r.visible = False

    # # Remove border/outline so only the colorbar remains
    plot.state.outline_line_color = None
    plot.state.toolbar_location = None
    plot.state.background_fill_alpha = 1.0
    plot.state.outline_line_alpha = 0


def _get_format_colorbar(
    text_font: str,
    im_scale: Optional[float] = None,
    power_limit_high: Optional[int] = 5,
    power_limit_low: Optional[int] = -2,
):
    """Return colorbar hook (theme, size, optional scientific tick formatting)."""

    def _format_colorbar(plot, element):
        """Apply theme and size to colorbar (first element in right panel)."""
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
        p.title_text_font = text_font
        p.major_label_text_font = text_font

        # other keys to format
        p.major_label_text_align = "left"
        p.major_label_text_alpha = 1.0
        p.major_label_text_baseline = "middle"
        p.major_label_text_line_height = 1.2
        p.major_tick_line_alpha = 1.0
        p.major_tick_line_dash_offset = 0
        p.major_tick_line_width = 1

        # Apply tick formatter for custom number formatting
        tick_args = {}

        use_scientific = None
        precision = 2
        if im_scale is not None:
            # determine precision based on im_slale
            im_scale_log = np.log10(im_scale)
            im_scale_log = int(im_scale_log)
            if im_scale_log >= power_limit_high or im_scale_log <= power_limit_low:
                use_scientific = True
                precision = 1
            else:
                use_scientific = False

        if precision is not None:
            tick_args["precision"] = precision
        if use_scientific is not None:
            tick_args["use_scientific"] = use_scientific
        if power_limit_high is not None:
            tick_args["power_limit_high"] = power_limit_high
        if power_limit_low is not None:
            tick_args["power_limit_low"] = power_limit_low

        formatter = BasicTickFormatter(**tick_args)
        if hasattr(p, "formatter"):
            p.formatter = formatter

    return _format_colorbar


class NDSlicer(param.Parameterized):
    # Viewing Parameters
    title_font_size = param.Number(default=12, bounds=(2, 36), step=1)
    vmin = param.Number(default=0.0)
    vmax = param.Number(default=0.0)
    size_scale = param.Number(default=400, bounds=(200, 1000), step=10)
    flip_ud = param.Boolean(default=False)
    flip_lr = param.Boolean(default=False)
    cplx_view = param.ObjectSelector(
        default="mag", objects=["mag", "phase", "real", "imag"]
    )
    display_images = param.ListSelector(default=[], objects=[])
    display_image_titles = param.Dict(default={})
    display_image_titles_visible = param.Boolean(default=True)
    display_error_map_titles_visible = param.Boolean(default=False)
    grid_visible = param.Boolean(default=False)
    text_font = param.ObjectSelector(
        default=themes.DEFAULT_FONT, objects=themes.VALID_FONTS
    )

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
    normalize_error_map = param.Boolean(default=True)
    normalize_for_display = param.Boolean(default=False)
    metrics_text_types = param.ListSelector(default=[], objects=metrics.FULL_METRICS)
    metrics_text_location = param.ObjectSelector(
        default=ROI_LOCATION.TOP_LEFT, objects=ROI_LOCATION
    )
    metrics_text_font_size = param.Number(default=12, bounds=(5, 24), step=1)

    # Rebuilding figure
    rebuild_figure_flag = param.Boolean(default=False)

    # Mouse scroll dimension
    scroll_dim = param.String(
        default=None, doc="Dimension to scroll through with mouse wheel"
    )

    # Popup pixel inspection
    popup_pixel_enabled = param.Boolean(
        default=False,  # off by default
        doc="If True, clicking on an image shows a popup with pixel values.",
    )
    popup_pixel_show_location = param.Boolean(
        default=True,
        doc="If True, shows the location of the pixel in the popup.",
    )
    popup_pixel_location = param.ObjectSelector(
        default=POPUP_LOCATION.DEFAULT, objects=POPUP_LOCATION
    )
    popup_pixel_on_error_maps = param.Boolean(
        default=True,
        doc="If True, enables popup pixel inspection on error maps.",
    )
    popup_pixel_coordinate_x = param.Number(default=-1)
    popup_pixel_coordinate_y = param.Number(default=-1)

    def __init__(
        self,
        data: hv.Dataset,
        vdims: Sequence[str],
        cdim: Optional[str] = None,
        clabs: Optional[Sequence[str]] = None,
        cat_dims: Optional[Dict[str, List]] = None,
        cfg: Optional[Dict[str, str]] = None,
        plot_hooks: Optional[List[Callable]] = None,
        **params,
    ):
        """
        Slicer for N-dimensional HoloViews Dataset; produces 2D view from slice indices.

        Parameters
        ----------
        data : hv.Dataset
            Dataset with kdims (e.g. ImgName + spatial + slice dims) and Value vdim.
        vdims : Sequence[str]
            Two dimension names used for the 2D view (e.g. ['x','y']).
        cdim : Optional[str]
            Collate dimension; if set, layout is 1D of 2D slices with labels.
        clabs : Optional[Sequence[str]]
            Labels for cdim (required if cdim set).
        cat_dims : Optional[Dict[str, List]]
            Categorical slice dimensions: dim name -> list of options.
        cfg : Optional[Dict]
            Slicer (and ROI) config from JSON.
        plot_hooks : Optional[List[Callable]]
            Bokeh hooks (e.g. scroll) applied to plots.
        """
        # Not the most efficient, but now update from config if supplied. mimics user having manually
        # set all parameters as desired.
        # if supplied in config, vdims already taken care of by the viewer
        from_config = cfg is not None

        super().__init__(**params)

        # additional hooks to add to the plot (for comms from slicer to viewer)
        self.plot_hooks = plot_hooks
        if self.plot_hooks is None:
            self.plot_hooks = []

        # initialize internal tracker of slice data
        self._popout_active = False
        self._curr_slice_data = None
        self._pixel_warning_shown = False

        with param.parameterized.discard_events(self):
            self.data = data
            self.cat_dims = cat_dims

            # all dimensions
            self.ndims = [d.name for d in data.kdims][::-1]

            # Dictionary of all total size of each dimension
            self.dim_sizes = {}
            for dim in self.ndims:
                self.dim_sizes[dim] = data.aggregate(dim, np.mean).data[dim].size

            # Initialize slice cache
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
            self.set_volatile_dims(vdims, pre_cache=False)

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

                # Default display image titles
                if not ("display_image_titles" in cfg["slicer_config"]):
                    self.display_image_titles = {
                        img_name: img_name for img_name in self.display_images
                    }

                for k in self.clabs:
                    if k not in self.display_image_titles:
                        warnings.warn(
                            f'"{k}" not in config display titles. Using defaults.'
                        )
                        self.display_image_titles = {
                            img_name: img_name for img_name in self.display_images
                        }
                        break

                self.ROI = roi.ROI(config=cfg["roi_config"])
                self.update_cplx_view(
                    self.cplx_view, recompute_min_max=False, pre_cache=False
                )
                self.update_colormap()
                self.update_roi_colormap(self.roi_cmap)

                self.DifferenceColorMapper = ColorMap(self.error_map_cmap)

            else:
                # Initialize display images
                self.param.display_images.objects = self.clabs
                self.display_images = self.clabs
                self.display_image_titles = {
                    img_name: img_name for img_name in self.display_images
                }

                # ROI init
                self.ROI = roi.ROI()

                # Update color limits with default
                self.update_cplx_view(self.cplx_view, pre_cache=False)

                # Color map object. Auto-select "Quantitative" if inferred from named dimensions.
                if self._infer_quantitative_maptype() is not None:
                    self.ColorMapper = QuantitativeColorMap(
                        self._infer_quantitative_maptype(), self.vmin, self.vmax
                    )
                    self.cmap = "Quantitative"
                else:
                    self.ColorMapper = ColorMap(self.cmap)

                # Diff map and metrics init
                self.param.metrics_reference.objects = self.clabs
                self.metrics_reference = self.clabs[0]  # Default to the first one
                self.DifferenceColorMapper = ColorMap(self.error_map_cmap)

        self.set_vmin_vmax()

        # Initialize static instance of plot through self.Figure
        self._build_figure_objects(self.slice())

    def _update_cache(self):
        """Cache crop, clim, and slice indices for current view."""
        # Cache cropped view
        self.crop_cache[self.vdims[0]] = (self.lr_crop[0], self.lr_crop[1])
        self.crop_cache[self.vdims[1]] = (self.ud_crop[0], self.ud_crop[1])

        # Cache contrast details
        self.CPLX_VIEW_CLIM_CACHE[self.cplx_view] = ClimSettings(
            vmin=self.vmin,
            vmax=self.vmax,
            bound_min=min(self.param.vmin.bounds[0], self.param.vmax.bounds[0]),
            bound_max=max(self.param.vmin.bounds[1], self.param.vmax.bounds[1]),
            step=self.param.vmin.step,
            cmap=self.cmap,
        )

        # Slice cache
        for dim in self.sdims:
            self.slice_cache[dim] = self.dim_indices[dim]

        # TODO: quantitative map cache (?)

    def slice(self, apply_colormap: bool = True, return_metrics: bool = True) -> Dict:
        """
        Return current 2D slice and optional error maps/metrics.

        Parameters
        ----------
        apply_colormap : bool
            If True, run ColorMapper.preprocess_data on image values.
        return_metrics : bool
            If True and metrics enabled, compute error maps and text metrics.

        Returns
        -------
        dict
            "img": dict of hv.Dataset per display image; "error_map" and "metrics" if applicable.
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
        if (self.metrics_state is not METRICS_STATE.INACTIVE) and return_metrics:
            # Gather arrays
            ref_img = np.copy(imgs[self.metrics_reference].data["Value"])

            tar_keys = [k for k in imgs.keys() if k != self.metrics_reference]

            metrics_dict = {}
            error_maps = {}

            for k in tar_keys:

                tar_img = np.copy(imgs[k].data["Value"])

                # NOTE: forced un-normalization for qmaps depreciated. option exposed to user.
                if self.normalize_error_map or self.normalize_for_display:
                    tar_img = utils.normalize_scale(
                        tar_img,
                        ref_img,  # ofs=True, mag=np.iscomplexobj(tar_img)
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

                    # handle nan values returned bc of small pixel values
                    error_maps[k] = utils.clone_dataset(imgs[k], error_map, link=False)

                # normalize displayed images
                if self.normalize_for_display:
                    imgs[k].data["Value"] = tar_img

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
            and return_metrics
        ):
            imgs.pop(self.metrics_reference)

        # Preprocessing for color map data
        if apply_colormap:
            for k in imgs.keys():
                imgs[k].data["Value"] = self.ColorMapper.preprocess_data(
                    imgs[k].data["Value"]
                )

        out_dict["img"] = imgs

        # keep access to current slice data
        self._curr_slice_data = out_dict

        return out_dict

    def _build_figure_opts(self):
        """Build opts dict for images, colorbars, ROI, and difference maps."""
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

        fmt_img_hook = _get_format_image(
            self.text_font,
            self.display_image_titles_visible,
            self.grid_visible,
        )

        # Image options
        im_opts = dict(
            cmap=self.ColorMapper.get_cmap(),
            width=int(main_width),
            height=int(main_height),
            invert_yaxis=self.flip_ud,
            invert_xaxis=self.flip_lr,
            fontscale=(self.title_font_size / 12),
            hooks=[
                fmt_img_hook,
                _bokeh_disable_wheel_zoom_tool,
                *self.plot_hooks,
            ],
            # tools=["hover"],
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
                    hooks=[
                        fmt_img_hook,
                        _bokeh_disable_wheel_zoom_tool,
                        *self.plot_hooks,
                    ],
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
                _get_format_image(
                    self.text_font,
                    title_visible=self.display_image_titles_visible,
                    grid_visible=False,
                ),
                _bokeh_disable_wheel_zoom_tool,
                _hide_image,
                _get_format_colorbar(self.text_font, im_scale=self.vmax),
            ],  # Hide the dummy glyph
            **shared_opts,
        )

        # Difference map
        diff_opts = im_opts.copy()
        diff_opts["hooks"] = [
            _get_format_image(
                self.text_font,
                title_visible=self.display_error_map_titles_visible,
                grid_visible=self.grid_visible,
            ),
            _bokeh_disable_wheel_zoom_tool,
            *self.plot_hooks,
        ]
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
        diff_cbar_opts["hooks"] = [
            _get_format_image(
                self.text_font,
                title_visible=self.display_error_map_titles_visible,
                grid_visible=False,
            ),
            _bokeh_disable_wheel_zoom_tool,
            _hide_image,
            _get_format_colorbar(
                self.text_font, im_scale=self.vmax / self.error_map_scale
            ),
        ]
        diff_cbar_opts["clim"] = diff_opts["clim"]
        diff_cbar_opts.pop("colorbar_opts")

        if self.colorbar_label is not None and len(self.colorbar_label) > 0:
            if self.error_map_type == "SSIM":
                diff_cbar_opts["colorbar_opts"] = dict(title="SSIM")
            else:
                diff_cbar_opts["colorbar_opts"] = dict(
                    title=f"Difference ({round_str(self.error_map_scale, ndec=3)}x)"
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

    def _build_figure_objects(self, input_data: dict):
        """Build/assign self.Figure from slice dict (images, ROI, diff maps, colorbars)."""
        self._update_cache()

        opts = self._build_figure_opts()

        # re-order so ref is always on the left or right
        img_dict = input_data["img"]
        fig_image_names = list(img_dict.keys())

        # lbrt bounds
        main_lbrt = (
            hv.Image(img_dict[fig_image_names[0]]).opts(**opts["im_opts"]).bounds.lbrt()
        )

        # Initialize popup-pixel overlay loc and tap streams
        self._popout_active = bool(self.popup_pixel_enabled) and self.roi_state not in (
            ROI_STATE.FIRST_SELECTION,
            ROI_STATE.SECOND_SELECTION,
        )
        if self._popout_active:
            popup_tap_stream = streams.SingleTap(
                x=self.popup_pixel_coordinate_x,
                y=self.popup_pixel_coordinate_y,
            )
        else:
            popup_tap_stream = None

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
        self._popup_pixel_pipes = {}
        imgs = []
        img_labels = {}
        for k in fig_image_names:

            # Extract metrics
            image_name = self.display_image_titles[k]

            if (
                self.metrics_state != METRICS_STATE.INACTIVE
            ) and k == self.metrics_reference:
                image_name = f"{image_name} (Ref)"

            img_labels[k] = image_name

            pipe = streams.Pipe(data=img_dict[k])

            self._image_pipes[k] = pipe

            def _img_callback(data, image_name=image_name):
                return hv.Image(data, label=image_name).opts(**opts["im_opts"])

            imgs.append(
                hv.DynamicMap(
                    _img_callback,
                    streams=[pipe],
                ).opts(title=image_name if self.display_image_titles_visible else "")
            )
            # send data
            self._image_pipes[k].send(img_dict[k])

            # If we don't have an error map but metrics are requested, add metrics to image
            if (
                metrics_dict
                and k in metrics_dict
                and self.metrics_state == METRICS_STATE.TEXT
            ):
                self._metrics_pipe[k] = streams.Pipe(data=metrics_dict[k])
                imgs[-1] = self._add_metrics_overlay(
                    imgs[-1], metrics_dict[k], main_lbrt, k
                )

            # Popup-pixel: attach shared tap listener to this image plot
            if self._popout_active:
                self._popup_pixel_pipes[k] = streams.Pipe(
                    data=None
                )  # for triggering updates
                imgs[-1] = imgs[-1] * hv.DynamicMap(
                    lambda x, y, key=k, data=None: self._add_popup_pixel_text(
                        x,
                        y,
                        key=key,
                        is_error=False,
                        data=data,
                    ),
                    streams=[
                        popup_tap_stream,
                        self._popup_pixel_pipes[k],
                    ],
                )

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
            self._diffmap_popup_pixel_pipes = {}
            for k in fig_image_names:
                name = self.display_image_titles[k]
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
                        label = f"Diff ({round_str(self.error_map_scale, ndec=3)}x)"

                    def _diff_callback(data):
                        return hv.Image(data, label=label).opts(**opts["diff_opts"])

                    diff_imgs.append(
                        hv.DynamicMap(
                            _diff_callback,
                            streams=[diff_pipe],
                        ).opts(
                            title=label if self.display_error_map_titles_visible else ""
                        )
                    )

                    # send data
                    self._diffmap_pipes[k].send(error_dict[k])

                # If we have an error map, add the text to it
                if (
                    metrics_dict
                    and k in metrics_dict
                    and self.metrics_state in [METRICS_STATE.MAP, METRICS_STATE.ALL]
                ):
                    self._metrics_pipe[k] = streams.Pipe(data=metrics_dict[k])
                    diff_imgs[-1] = self._add_metrics_overlay(
                        diff_imgs[-1], metrics_dict[k], main_lbrt, k
                    )

                # Popup-pixel: attach shared tap listener to error-map plot
                if (
                    self._popout_active
                    and self.popup_pixel_on_error_maps
                    and not (k == self.metrics_reference)
                ):
                    self._diffmap_popup_pixel_pipes[k] = streams.Pipe(data=None)
                    diff_imgs[-1] = diff_imgs[-1] * hv.DynamicMap(
                        lambda x, y, key=k, data=None: self._add_popup_pixel_text(
                            x,
                            y,
                            key=key,
                            data=data,
                            is_error=True,
                        ),
                        streams=[
                            popup_tap_stream,
                            self._diffmap_popup_pixel_pipes[k],
                        ],
                    )

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
        self.Figure = row

    def _add_metrics_overlay(self, base_plot, metrics, bounds, key):
        """Overlay text metrics at configured corner on base_plot."""
        tx_pad = 3
        effective_location = utils.get_effective_location(
            self.metrics_text_location, self.flip_lr, self.flip_ud
        )
        # compute text position
        if effective_location == ROI_LOCATION.TOP_LEFT:
            tx = bounds[0] + tx_pad
            ty = bounds[3] - tx_pad
        elif effective_location == ROI_LOCATION.TOP_RIGHT:
            tx = bounds[2] - tx_pad
            ty = bounds[3] - tx_pad
        elif effective_location == ROI_LOCATION.BOTTOM_LEFT:
            tx = bounds[0] + tx_pad
            ty = bounds[1] + tx_pad
        else:
            tx = bounds[2] - tx_pad
            ty = bounds[1] + tx_pad

        halign = self.metrics_text_location.value.split(" ")[1].lower()
        valign = self.metrics_text_location.value.split(" ")[0].lower()

        def _text_callback(data, tx=tx, ty=ty, halign=halign, valign=valign):
            txt = "\n".join(
                f"{mk}: {pprint_str(mv, D=4, E=2)}" for mk, mv in data.items()
            )
            return hv.Text(
                tx,
                ty,
                txt,
                halign=halign,
                valign=valign,
                fontsize=self.metrics_text_font_size,
            ).opts(
                text_font=bokeh_value(self.text_font),
                text_color=themes.VIEW_THEME.text_color,
            )

        dyn = hv.DynamicMap(_text_callback, streams=[self._metrics_pipe[key]])
        # send metrics for initial render
        self._metrics_pipe[key].send(metrics)

        return base_plot * dyn

    def _add_popup_pixel_text(self, x, y, key, is_error=False, data=None):
        """Return overlay (text + marker) for pixel value at (x, y) on given image key."""

        def get_dummy():
            dtext = hv.Text(0, 0, "").opts(text_alpha=0)
            dtext2 = hv.Text(0, 0, "").opts(text_alpha=0)
            return dtext * dtext2

        # ignore initial sentinel values
        if x is None or y is None:
            return get_dummy()
        try:
            x = float(x)
            y = float(y)
        except Exception:
            return get_dummy()

        if x < 0 or y < 0:
            return get_dummy()

        # set new popup location
        self.popup_pixel_coordinate_x = x
        self.popup_pixel_coordinate_y = y
        xidx = int(round(x))
        yidx = int(round(y))
        xkey = self.vdims[0]
        ykey = self.vdims[1]

        if is_error:
            if "error_map" in self._curr_slice_data:
                slc_ = self._curr_slice_data["error_map"][key]
            else:
                slc_ = None
        else:
            if "img" in self._curr_slice_data:
                slc_ = self._curr_slice_data["img"][key]
            else:
                slc_ = None

        value = "N/A"
        halign = "left"
        valign = "bottom"
        xmin, xmax = None, None
        ymin, ymax = None, None

        # Determine value and align location
        if slc_ is not None:
            try:
                xmin, xmax = slc_.range(xkey)
                ymin, ymax = slc_.range(ykey)
                xh = (xmax + xmin) / 2
                yh = (ymax + ymin) / 2
                if xidx > xh:
                    halign = "right"
                if yidx > yh:
                    valign = "top"
                # check if out of range
                if xidx < xmin or xidx > xmax or yidx < ymin or yidx > ymax:
                    if not self._pixel_warning_shown:
                        pn.state.notifications.warning(
                            "Detected out of range click, clearing popup.",
                            duration=3000,
                        )
                        self._pixel_warning_shown = True
                    return get_dummy()

                value = float(slc_[xidx, yidx])

            except Exception:
                pass

        # format value
        if isinstance(value, float) and not np.isnan(value):
            value = pprint_str(value, D=6, E=3)

        # format label
        vlabel = "Value"
        if is_error:
            # label with unit of error map
            vlabel = self.error_map_type
        else:
            pass
            # case 1: quantitative map category
            qmaptype = self._infer_quantitative_maptype()
            if qmaptype is not None:
                vlabel = qmaptype
            # case 2: first categorical map category
            elif self.cat_dims is not None and len(self.cat_dims.keys()) > 0:
                cdk = list(self.cat_dims.keys())[0]
                vlabel = self.dim_indices[cdk]

            # case 2: user has given a label to the colorbar
            elif self.colorbar_label is not None and len(self.colorbar_label) > 0:
                lab_raw = self.colorbar_label.lower().replace(" ", "").replace(".", "")
                if len(lab_raw) > 0 and lab_raw not in ["au", "na", "none"]:
                    vlabel = self.colorbar_label

        value_line = f"  {vlabel}: {value}  "
        if self.popup_pixel_show_location:
            header = f"  ({xkey}={xidx}, {ykey}={yidx})  "
            lines = [header, value_line]
        else:
            lines = [value_line]

        # decide on pop-up location
        if self.popup_pixel_location == POPUP_LOCATION.DEFAULT:
            if self.flip_lr:
                halign = "right" if halign == "left" else "left"
            if self.flip_ud:
                valign = "top" if valign == "bottom" else "bottom"
        else:
            if self.popup_pixel_location == POPUP_LOCATION.TOP_LEFT:
                halign = "right"
                valign = "bottom"
            elif self.popup_pixel_location == POPUP_LOCATION.TOP_RIGHT:
                halign = "left"
                valign = "bottom"
            elif self.popup_pixel_location == POPUP_LOCATION.BOTTOM_LEFT:
                halign = "right"
                valign = "top"
            elif self.popup_pixel_location == POPUP_LOCATION.BOTTOM_RIGHT:
                halign = "left"
                valign = "top"

        txt = "\n".join(lines)
        text = hv.Text(
            x,
            y,
            txt,
            halign=halign,
            valign=valign,
            fontsize=8,
        ).opts(
            text_font=bokeh_value(self.text_font),
            text_color=themes.VIEW_THEME.text_color,
            text_alpha=1.0,
            background_fill_alpha=0.9,
            background_fill_color=themes.VIEW_THEME.accent_color,
            border_line_width=1.5,
            border_line_color=themes.BOKEH_WIDGET_COLOR,
            border_radius=8,
            border_line_alpha=0.6,
            show_legend=False,
        )

        marker = hv.Points(
            np.array([[x, y]]),
        ).opts(
            size=5,
            alpha=1,
            color=themes.BOKEH_WIDGET_COLOR,
            marker="o",
            line_color=themes.BOKEH_WIDGET_COLOR,
            line_width=1.5,
            show_legend=False,
        )
        return text * marker

    def _update_figure(self, input_data: Dict[str, dict]):
        """Push new slice data through pipes to update figure in place."""
        assert self._image_pipes is not None, "Figure not initialized"

        pipe_popouts = (
            self._popout_active
            and self.popup_pixel_coordinate_x > 0
            and self.popup_pixel_coordinate_y > 0
        )
        # Send image data through pipes
        imgs_dict = input_data["img"]
        for k in imgs_dict.keys():
            self._image_pipes[k].send(imgs_dict[k])

            if pipe_popouts:
                self._popup_pixel_pipes[k].send(None)

            # Send ROI data
            if self.roi_state == ROI_STATE.ACTIVE:
                self._roi_pipes[k].send(imgs_dict[k])

            # Send metrics computations and error maps
            if (
                self.metrics_state in [METRICS_STATE.MAP, METRICS_STATE.ALL]
                and k != self.metrics_reference
            ):
                self._diffmap_pipes[k].send(input_data["error_map"][k])
                if pipe_popouts and self.popup_pixel_on_error_maps:
                    self._diffmap_popup_pixel_pipes[k].send(None)

            if (
                self.metrics_state in [METRICS_STATE.TEXT, METRICS_STATE.ALL]
                and k != self.metrics_reference
                and k in self._metrics_pipe
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
        "display_image_titles",
        "display_image_titles_visible",
        "display_error_map_titles_visible",
        "grid_visible",
        "colorbar_on",
        "colorbar_label",
        "roi_state",
        "error_map_scale",
        "metrics_state",
        "title_font_size",
        "text_font",
        "popup_pixel_enabled",
        "popup_pixel_show_location",
        "popup_pixel_location",
        "popup_pixel_on_error_maps",
        "normalize_for_display",
        watch=True,
    )
    @error.error_handler_decorator()
    def _rebuild_figure(self):
        """Clear figure and set flag so next view() rebuilds."""
        self.Figure = None
        self.rebuild_figure_flag = True

    @param.depends("dim_indices", "rebuild_figure_flag")
    @error.error_handler_decorator()
    @profilers.profile_decorator(
        enable=False
    )  # Print call information or log to file for debugging
    def view(self) -> hv.Layout:
        """
        Return layout of current slice (rebuild or update figure as needed).

        Returns
        -------
        hv.Layout
            Row/layout of images, optional ROI row, optional diff maps, colorbars.
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

            self._build_figure_objects(slice_dict)

        else:
            self._update_figure(slice_dict)

        if atomize:
            pn.state.curdoc.unhold()

        return self.Figure

    def _infer_quantitative_maptype(self) -> Union[str, None]:
        """Infer quantitative map type from categorical dim selection (e.g. T1, T2)."""
        for dim in self.cat_dims.keys():
            if self.dim_indices[dim].capitalize() in QUANTITATIVE_MAPTYPES:
                return self.dim_indices[dim].capitalize()

        return None

    def set_volatile_dims(self, vdims: Sequence[str], pre_cache: bool = True):
        """Set viewing dims and derive slicing dims, crop bounds, dim_indices."""
        if pre_cache:
            self._update_cache()

        with param.parameterized.discard_events(self):
            self.clear_popup_pixel()

            vdims = list(vdims)

            self.vdims = vdims

            if self.cdim is not None:
                self.non_sdims = [self.cdim] + vdims
            else:
                self.non_sdims = vdims

            # sliceable dimensions
            self.sdims = [d for d in self.ndims if d not in self.non_sdims]

            # Reset scroll dimension to first sdim when view dims change
            self.scroll_dim = self.sdims[0] if self.sdims else None

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
        self.param.trigger("lr_crop", "ud_crop", "dim_indices")

    def update_cplx_view(
        self, new_cplx_view: str, recompute_min_max: bool = True, pre_cache: bool = True
    ):

        if pre_cache:
            self._update_cache()

        # set attribute
        same_cplx_view = self.cplx_view == new_cplx_view
        self.cplx_view = new_cplx_view
        dmin = None
        dmax = None
        vmin = None
        vmax = None
        step = None
        cmap = None
        bound_min = None
        bound_max = None

        if self.cplx_view in self.CPLX_VIEW_CLIM_CACHE:
            clim_settings = self.CPLX_VIEW_CLIM_CACHE[self.cplx_view]
            assert isinstance(clim_settings, ClimSettings)
            vmin = clim_settings.vmin
            vmax = clim_settings.vmax
            dmin = clim_settings.bound_min
            dmax = clim_settings.bound_max
            step = clim_settings.step
            cmap = clim_settings.cmap
            bound_min = clim_settings.bound_min
            bound_max = clim_settings.bound_max
        else:
            # use same cmap
            cmap = self.cmap

            # Compute max color limits if setting up new complex view or input desires recomputation
            if recompute_min_max or (not same_cplx_view):
                dmin, dmax, vmin, vmax = self.get_autoscale_lims()
            else:
                # for loading from config
                vmin = self.vmin
                vmax = self.vmax
                if (
                    self.param.vmin.bounds is not None
                    and self.param.vmax.bounds is not None
                ):
                    dmin = float(
                        min(self.param.vmin.bounds[0], self.param.vmax.bounds[0])
                    )
                    dmax = float(
                        max(self.param.vmin.bounds[1], self.param.vmax.bounds[1])
                    )
                else:
                    dmin, dmax = self.get_data_lims()
                bound_min, bound_max = dmin, dmax
                step = self.param.vmin.step

        # Update color limits
        with param.parameterized.discard_events(self):
            # Update bounds if new complex view
            if not same_cplx_view:
                bound_min = dmin
                bound_max = dmax

            self.set_vmin_vmax(
                vmin=vmin,
                vmax=vmax,
                dmin=dmin,
                dmax=dmax,
                bound_min=bound_min,
                bound_max=bound_max,
                step=step,
            )

            self.cmap = cmap

            # update cmap for quantitative colormap
            self.update_colormap()

            self._update_cache()

        # Trigger
        self.param.trigger("vmin", "vmax", "cmap")

    def get_slice_data(self) -> np.ndarray:
        """Stack current slice image values (no colormap) into one array."""
        data = np.stack(
            [
                d.data["Value"]
                for d in self.slice(apply_colormap=False, return_metrics=False)[
                    "img"
                ].values()
            ]
        )
        data[np.isnan(data)] = 0
        return data

    def get_data_lims(self) -> Tuple[float, float]:
        """Return (min, max) of current slice data."""
        data = self.get_slice_data()
        return np.min(data), np.max(data)

    def set_vmin_vmax(
        self,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        dmin: Optional[float] = None,
        dmax: Optional[float] = None,
        bound_min: Optional[float] = None,
        bound_max: Optional[float] = None,
        step: Optional[float] = None,
    ):
        """Set vmin/vmax and slider bounds/step; return (vmin, vmax, bound_min, bound_max, step)."""
        # current slice's data limits
        if (dmin is None) or (dmax is None):
            dmin, dmax = self.get_data_lims()

        # desired contrast bounds
        if vmin is None:
            vmin = self.vmin
        if vmax is None:
            vmax = self.vmax

        # Slider bounds
        if bound_min is None:
            if (
                self.param.vmin.bounds is not None
                and self.param.vmax.bounds is not None
            ):
                bound_min = min(self.param.vmin.bounds[0], self.param.vmax.bounds[0])
            else:
                bound_min = dmin
        if bound_max is None:
            if (
                self.param.vmin.bounds is not None
                and self.param.vmax.bounds is not None
            ):
                bound_max = max(self.param.vmin.bounds[1], self.param.vmax.bounds[1])
            else:
                bound_max = dmax

        # Allow extending beyond data bounds if user desires (vmin/vmax)
        bound_min = min(bound_min, vmin)
        bound_max = max(bound_max, vmax)

        # By default, keep clim bar within data bounds if desired vmin/vmax are within data bounds
        if vmin > dmin:
            bound_min = dmin
        if vmax < dmax:
            bound_max = dmax

        # ensure data limits met
        if vmin > dmax:
            pn.state.notifications.warning(
                f"Requested vmin={vmin:0.1e} > data max={dmax:0.1e}. Setting vmin to data max."
            )
            vmin = dmax
        if vmax < dmin:
            pn.state.notifications.warning(
                f"Requested vmax={vmax:0.1e} < data min={dmin:0.1e}. Setting vmax to data min."
            )
            vmax = dmin

        # Equal vmin / vmax case
        if vmin == vmax:
            if vmin == 0:
                msg = "Requested min = max = 0. Setting vmin/vmax to -1/1."
            else:
                msg = f"Data min = max = {dmin:0.1e}. Setting vmin/vmax to {dmin:0.1e} +/- 1."
            pn.state.notifications.warning(msg)
            vmin = vmin - 1
            vmax = vmax + 1
            bound_min = min(bound_min, vmin - 1)
            bound_max = max(bound_max, vmax + 1)
            step = step or 0.1
        else:
            if dmin == dmax:
                step = (bound_max - bound_min) / 100
            else:
                step = step or (dmax - dmin) / 100

        step = max(step, TOL)

        self.param.vmin.bounds = (bound_min, bound_max)
        self.param.vmax.bounds = (bound_min, bound_max)
        self.param.vmin.step = step
        self.param.vmax.step = step
        self.vmin = vmin
        self.vmax = vmax
        self.param.vmin.default = vmin
        self.param.vmax.default = vmax

        self._update_cache()

        return vmin, vmax, bound_min, bound_max, step

    def get_autoscale_lims(self) -> Tuple[float, float, float, float]:
        """Return (dmin, dmax, vmin, vmax) using data percentiles (phase: -pi, pi)."""
        data = self.get_slice_data()

        if self.cplx_view == "phase":
            dmin, dmax = -np.pi, np.pi
        else:
            dmin = np.min(data)
            dmax = np.max(data)

        vmin = np.percentile(data, 0.1)
        vmax = np.percentile(data, 99.9)

        return dmin, dmax, vmin, vmax

    def autoscale_clim(self):
        """Set vmin/vmax from percentiles and return (vmin, vmax, bound_min, bound_max, step)."""
        with param.parameterized.discard_events(self):
            # Get data limits and auto-scale by percentiles
            dmin, dmax, vmin, vmax = self.get_autoscale_lims()

            # supply this to set_vmin_vmax to avoid recomputing data limits
            vmin, vmax, bound_min, bound_max, step = self.set_vmin_vmax(
                vmin=vmin,
                vmax=vmax,
                dmin=dmin,
                dmax=dmax,
                bound_min=dmin,
                bound_max=dmax,
            )

        self.param.trigger("vmin", "vmax")

        return vmin, vmax, bound_min, bound_max, step

    def update_display_image_list(self, display_images: Sequence[str]):
        """Set which images are displayed."""
        self.display_images = display_images

    def update_colormap(self):
        """Refresh ColorMapper from cmap param (including Quantitative)."""
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
        """Set ROI colormap (Same / Quantitative / named)."""
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
        """Set ROI zoom scale and trigger redraw."""
        self.ROI.zoom_scale = new_zoom
        self.param.trigger("roi_state")

    def update_roi_loc(self, new_loc: str):
        """Set ROI overlay corner and trigger redraw."""
        self.ROI.roi_loc = ROI_LOCATION(new_loc)
        self.param.trigger("roi_state")

    def update_roi_lr_crop(self, new_lr_crop: Tuple[int, int]):
        """Set ROI left/right crop and trigger redraw."""
        self.ROI.set_xrange(*new_lr_crop)
        self.param.trigger("roi_state")

    def update_roi_ud_crop(self, new_ud_crop: Tuple[int, int]):
        """Set ROI up/down crop and trigger redraw."""
        self.ROI.set_yrange(*new_ud_crop)
        self.param.trigger("roi_state")

    def update_roi_line_color(self, new_color: str):
        """Set ROI border color and trigger redraw."""
        self.ROI.color = new_color
        self.param.trigger("roi_state")

    def update_roi_line_width(self, new_width: int):
        """Set ROI border width and trigger redraw."""
        self.ROI.line_width = new_width
        self.param.trigger("roi_state")

    def update_roi_zoom_order(self, new_order: int):
        """Set ROI zoom interpolation order and trigger redraw."""
        self.ROI.zoom_order = new_order
        self.param.trigger("roi_state")

    def update_roi_mode(self, new_mode: int):
        """Set ROI view mode (overlay vs separate) and trigger redraw."""
        self.roi_mode = ROI_VIEW_MODE(new_mode)
        self.param.trigger("roi_state")

    def update_roi_state(self, new_state: ROI_STATE):
        """Set ROI state (inactive / first_click / second_click / active) and show message."""
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
        """Set reference image for metrics and trigger redraw."""
        self.metrics_reference = new_ref
        if self.metrics_state != METRICS_STATE.INACTIVE:
            self.param.trigger("metrics_state")

    def update_error_map_type(self, new_type: str):
        """Set error map metric type and trigger redraw."""
        self.error_map_type = new_type
        if self.metrics_state in [METRICS_STATE.MAP, METRICS_STATE.ALL]:
            self.param.trigger("metrics_state")

    def update_metrics_text_types(self, new_metrics: List[str]):
        """Set which metrics to show as text and trigger redraw."""
        self.metrics_text_types = new_metrics
        if self.metrics_state in [METRICS_STATE.TEXT, METRICS_STATE.ALL]:
            self.param.trigger("metrics_state")

    def update_metrics_state(self, new_metrics_state: METRICS_STATE):
        """Set metrics state (inactive / map / text / all)."""
        self.metrics_state = new_metrics_state

    def update_error_map_scale(self, new_scale: float):
        """Set error map scale and trigger redraw."""
        with param.parameterized.discard_events(self):
            self.error_map_scale = new_scale
        if self.metrics_state in [METRICS_STATE.MAP, METRICS_STATE.ALL]:
            self.param.trigger("error_map_scale")

    def update_error_map_cmap(self, new_cmap: str):
        """Set error map colormap and trigger redraw."""
        self.error_map_cmap = new_cmap
        self.DifferenceColorMapper = ColorMap(new_cmap)
        if self.metrics_state in [METRICS_STATE.MAP, METRICS_STATE.ALL]:
            self.param.trigger("error_map_scale")

    def update_normalize_error_map(self, normalize: bool):
        """Toggle error-map normalization and trigger redraw."""
        self.normalize_error_map = normalize
        if self.metrics_state in [METRICS_STATE.MAP, METRICS_STATE.ALL]:
            self.param.trigger("error_map_scale")

    def update_metrics_text_font_size(self, new_size: int):
        """Set metrics text font size and trigger redraw."""
        self.metrics_text_font_size = new_size
        if self.metrics_state in [METRICS_STATE.TEXT, METRICS_STATE.ALL]:
            self.param.trigger("metrics_state")

    def update_metrics_text_location(self, new_loc: ROI_LOCATION):
        """Set corner for metrics text overlay and trigger redraw."""
        self.metrics_text_location = new_loc
        if self.metrics_state in [METRICS_STATE.TEXT, METRICS_STATE.ALL]:
            self.param.trigger("metrics_state")

    def clear_popup_pixel(self):
        """Clear popup pixel overlay and coordinates."""
        if (
            not self._popout_active
            and self.popup_pixel_coordinate_x < 0
            and self.popup_pixel_coordinate_y < 0
        ):
            return

        self._popout_active = False
        self.popup_pixel_coordinate_x = -1
        self.popup_pixel_coordinate_y = -1
        self.param.trigger("popup_pixel_enabled")

    def update_popup_pixel_enabled(self, new_val: bool):
        """Enable/disable popup pixel inspection and clear if disabling."""
        old_val = self.popup_pixel_enabled
        if old_val == new_val:
            return  # no change

        self.popup_pixel_enabled = new_val
        if not new_val:
            with param.parameterized.discard_events(self):
                self.clear_popup_pixel()

        self.param.trigger("popup_pixel_enabled")

    def autoformat_error_map(self):
        """Set error map scale and cmap from data (e.g. match main image scale)."""
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

            error_max = self.get_max_err()

            if error_max > ERROR_TOL:
                self.error_map_scale = round(self.vmax / error_max, 1)

        elif self.error_map_type == "Diff":
            self.error_map_cmap = "RdBu"
            self.DifferenceColorMapper = ColorMap(self.error_map_cmap)
            error_max = np.abs(self.get_max_err())

            if error_max > ERROR_TOL:
                error_max = np.abs(self.get_max_err())
                self.error_map_scale = round(self.vmax / error_max, 1)

        elif self.error_map_type == "RelativeL1":
            self.error_map_cmap = "inferno"
            self.DifferenceColorMapper = ColorMap(self.error_map_cmap)
            self.error_map_scale = round(self.vmax)

        else:
            error.warning("Error map type does not have autoformat.")

    def get_max_err(self):
        """Return 99.9th percentile of current slice error map values."""
        error_data = self.slice()["error_map"]

        if self.metrics_reference in error_data:
            error_data.pop(self.metrics_reference)

        error_np = np.stack([d.data["Value"] for d in error_data.values()])
        error_np[np.isnan(error_np)] = 0

        return np.percentile(error_np, 99.9)

    def compute_scroll_delta(
        self, delta: float
    ) -> Tuple[Optional[str], Optional[Union[int, str]]]:
        """Increment/decrement scroll_dim by delta; return (dim_name, new_value) or (None, None)."""
        if self.scroll_dim is None or self.scroll_dim not in self.sdims:
            return None, None

        # Determine scroll direction (positive delta = scroll up = increment)
        move_amt = 1 if (abs(delta) > 1e-2) else 0
        direction = 1 if delta > 0 else -1
        move_amt = move_amt * direction
        move_amt = int(round(move_amt))

        # Handle categorical vs numeric dimensions
        if self.scroll_dim in self.cat_dims:
            # Cycle through categorical options
            options = self.cat_dims[self.scroll_dim]
            current_value = self.dim_indices[self.scroll_dim]
            try:
                current_idx = options.index(current_value)
            except ValueError:
                current_idx = 0
            new_idx = (current_idx + direction) % len(options)
            new_value = options[new_idx]
            new_value = str(new_value)
        else:
            # Numeric dimension - increment/decrement with bounds checking
            current = self.dim_indices[self.scroll_dim]
            max_val = self.dim_sizes[self.scroll_dim] - 1
            new_value = max(0, min(max_val, current + move_amt))
            new_value = int(new_value)

        return str(self.scroll_dim), new_value
