from typing import Optional

import holoviews as hv
import numpy as np
from holoviews import streams
from scipy.ndimage import zoom

from .cmap.cmap import ColorMap
from .enums import ROI_LOCATION
from .utils import get_effective_location


class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y


class ROI:
    def __init__(
        self,
        point1: Optional[Point] = None,
        point2: Optional[Point] = None,
        roi_loc: Optional[ROI_LOCATION] = ROI_LOCATION.TOP_RIGHT,
        zoom_scale: Optional[float] = 2.0,
        cmap: Optional[ColorMap] = ColorMap("jet"),
        color: Optional[str] = "red",
        line_width: Optional[int] = 2,
        zoom_order: Optional[int] = 1,
        padding_pct: Optional[float] = 0.001,
        config: Optional[dict] = None,
    ):
        """
        Basic ROI class responsible for extracting ROIs from images.
        """

        if config is not None:
            self.init_from_config(config, cmap=cmap)
            return

        self.point1 = point1
        self.point2 = point2

        self.roi_loc = roi_loc
        self.zoom_scale = zoom_scale
        self.cmap = cmap
        self.color = color
        self.line_width = line_width
        self.zoom_order = zoom_order
        self.padding_pct = padding_pct

    def get_overlay_roi(
        self,
        img_arr: hv.Dataset,
        label: str,
        addnl_opts: Optional[dict] = {},
        flip_lr: Optional[bool] = False,
        flip_ud: Optional[bool] = False,
    ) -> hv.Image:
        """
        Return the overlay layer for the ROI inside the main image.
        """

        self._validate()

        # Original bounds
        l_im, b_im, r_im, t_im = hv.Image(img_arr).bounds.lbrt()

        # Calculate scaled ROI size in data coordinates
        scaled_width = self.width() * self.zoom_scale
        scaled_height = self.height() * self.zoom_scale

        # Padding on roi inview location
        padding_x = self.padding_pct * (r_im - l_im)
        padding_y = self.padding_pct * (t_im - b_im)

        effective_loc = get_effective_location(self.roi_loc, flip_lr, flip_ud)

        if effective_loc == ROI_LOCATION.TOP_LEFT:
            new_x_min = l_im + padding_x
            new_x_max = new_x_min + scaled_width
            new_y_max = t_im - padding_y
            new_y_min = new_y_max - scaled_height
        elif effective_loc == ROI_LOCATION.TOP_RIGHT:
            new_x_max = r_im - padding_x
            new_x_min = new_x_max - scaled_width
            new_y_max = t_im - padding_y
            new_y_min = new_y_max - scaled_height
        elif effective_loc == ROI_LOCATION.BOTTOM_LEFT:
            new_x_min = l_im + padding_x
            new_y_min = b_im + padding_y
            new_x_max = new_x_min + scaled_width
            new_y_max = new_y_min + scaled_height
        elif effective_loc == ROI_LOCATION.BOTTOM_RIGHT:
            new_x_max = r_im - padding_x
            new_x_min = new_x_max - scaled_width
            new_y_min = b_im + padding_y
            new_y_max = new_y_min + scaled_height

        new_bounds = (new_x_min, new_y_min, new_x_max, new_y_max)

        # Callback for DynamicROI
        def _return_overlay_roi(data):

            # Extract region and zoom
            roi_data = self._create_roi_array(data)

            return hv.Image(
                roi_data,
                bounds=new_bounds,
                kdims=img_arr.kdims,
                vdims=img_arr.vdims,
                label=label,
            ).opts(
                cmap=self.cmap.get_cmap(),
                **addnl_opts,
            )

        # DynamicMap
        roi_pipe = streams.Pipe(data=img_arr)
        scaled_roi_dmap = hv.DynamicMap(_return_overlay_roi, streams=[roi_pipe])

        # add bounding box
        scaled_roi_img = scaled_roi_dmap * hv.Bounds(new_bounds, label=label).opts(
            color=self.color,
            line_width=self.line_width,
        )

        return scaled_roi_img, roi_pipe

    def get_separate_roi(
        self,
        img_arr: hv.Dataset,
        label: str,
        width: float,
        addnl_opts: Optional[dict] = {},
    ) -> hv.Image:
        """
        Return a new image object for the ROI with size scaled to have equivalent width to original image
        """

        self._validate()

        # We want to maintain crop aspect ratio with equivalent image width
        image_width = width
        roi_width = self.width()
        roi_height = self.height()

        # New zoom scale -> always determined by width
        img_lbrt = hv.Image(img_arr).bounds.lbrt()
        zoom_scale_x = (img_lbrt[2] - img_lbrt[0]) / roi_width

        # Extract data
        def _return_separate_roi(data):

            data_np = self._create_roi_array(data, zoom_scale=zoom_scale_x)

            return hv.Image(
                data_np,
                bounds=img_lbrt,
                kdims=img_arr.kdims,
                vdims=img_arr.vdims,
                label="ROI",
            ).opts(
                cmap=self.cmap.get_cmap(),
                width=int(image_width),
                height=int(image_width * roi_height / roi_width),
                **addnl_opts,
            )

        # DynamicMap
        roi_pipe = streams.Pipe(data=img_arr)
        scaled_roi_dmap = hv.DynamicMap(_return_separate_roi, streams=[roi_pipe])

        # Get new bounds
        le, be, re, te = _return_separate_roi(img_arr).bounds.lbrt()
        extent_padded = (le + 0.5, be + 0.5, re - 0.5, te - 0.5)

        scaled_roi_img = scaled_roi_dmap * hv.Bounds(
            extent_padded,
            label="ROI_BB",  # purposely use different label to prevent title from showing up
        ).opts(
            color=self.color,
            line_width=self.line_width,
        )

        return scaled_roi_img, roi_pipe

    def height(self):
        return abs(self.point2.y - self.point1.y)

    def width(self):
        return abs(self.point2.x - self.point1.x)

    def set_xrange(self, new_x1, new_x2):

        self.point1.x = new_x1
        self.point2.x = new_x2

    def set_yrange(self, new_y1, new_y2):

        self.point1.y = new_y1
        self.point2.y = new_y2

    def lbrt(self):

        self._validate()

        l, r = sorted([self.point1.x, self.point2.x])
        b, t = sorted([self.point1.y, self.point2.y])

        return l, b, r, t

    def _create_roi_array(
        self,
        img: hv.Image,
        zoom_scale: Optional[float] = None,
    ) -> np.ndarray:

        if zoom_scale is None:
            zoom_scale = self.zoom_scale

        x1 = min(self.point1.x, self.point2.x)
        x2 = max(self.point1.x, self.point2.x)

        y1 = min(self.point1.y, self.point2.y)
        y2 = max(self.point1.y, self.point2.y)

        cropped_region = img[x1:x2, y1:y2].clone()

        data_np = cropped_region.data["Value"]

        data_np = np.flipud(data_np)

        # zoom
        data_np = zoom(data_np, zoom_scale, order=self.zoom_order)

        return data_np

    def _validate(self):

        if self.point1 is None or self.point2 is None:
            raise ValueError("To create ROI, both points must be provided")

        if self.height() == 0:
            raise ValueError("ROI height cannot be zero")

        if self.width() == 0:
            raise ValueError("ROI width cannot be zero")

    def serialize(self) -> dict:
        """
        Hard-coded serialization for use in saving config by viewer.
        FIXME: color-maps do not serialize.
        """
        out_dict = {
            "roi_loc": self.roi_loc.value,
            "zoom_scale": self.zoom_scale,
            "color": self.color,
            "line_width": self.line_width,
            "zoom_order": self.zoom_order,
            "padding_pct": self.padding_pct,
        }

        if self.point1 is not None:
            out_dict["point1"] = {"x": self.point1.x, "y": self.point1.y}
        if self.point2 is not None:
            out_dict["point2"] = {"x": self.point2.x, "y": self.point2.y}

        return out_dict

    def init_from_config(self, cfg: dict, cmap):
        """
        Initialize ROI from hard-coded serialized config.
        For now, assuming that cmap is passed in separately.
        """

        self.roi_loc = ROI_LOCATION(cfg["roi_loc"])
        self.zoom_scale = cfg["zoom_scale"]
        self.color = cfg["color"]
        self.line_width = cfg["line_width"]
        self.zoom_order = cfg["zoom_order"]
        self.padding_pct = cfg["padding_pct"]

        if "point1" in cfg:
            self.point1 = Point(cfg["point1"]["x"], cfg["point1"]["y"])
        else:
            self.point1 = None

        if "point2" in cfg:
            self.point2 = Point(cfg["point2"]["x"], cfg["point2"]["y"])
        else:
            self.point2 = None

        # cmap not serializeable so handle separately.
        self.cmap = cmap
