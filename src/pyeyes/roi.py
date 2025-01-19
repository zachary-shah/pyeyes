from typing import Optional

import holoviews as hv
import numpy as np
from scipy.ndimage import zoom

from .enums import ROI_LOCATION
from .q_cmap.cmap import ColorMap


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
    ):
        """
        Basic ROI class responsible for extracting ROIs from images.
        """

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
        img: hv.Image,
        addnl_opts: Optional[dict] = {},
    ) -> hv.Image:
        """
        Return the overlay layer for the ROI inside the main image.
        """

        self._validate()

        # Extract and zoom
        data_np = self._create_roi_array(img)

        # Bilinear zoom
        data_np = zoom(data_np, self.zoom_scale, order=self.zoom_order)

        # Calculate scaled ROI size in data coordinates
        scaled_width = self.width() * self.zoom_scale
        scaled_height = self.height() * self.zoom_scale

        # Original bounds
        l_im, b_im, r_im, t_im = img.bounds.lbrt()

        # Padding on roi inview location
        padding_x = self.padding_pct * (r_im - l_im)
        padding_y = self.padding_pct * (t_im - b_im)

        if self.roi_loc == ROI_LOCATION.TOP_LEFT:
            new_x_min = l_im + padding_x
            new_x_max = new_x_min + scaled_width
            new_y_max = t_im - padding_y
            new_y_min = new_y_max - scaled_height
        elif self.roi_loc == ROI_LOCATION.TOP_RIGHT:
            new_x_max = r_im - padding_x
            new_x_min = new_x_max - scaled_width
            new_y_max = t_im - padding_y
            new_y_min = new_y_max - scaled_height
        elif self.roi_loc == ROI_LOCATION.BOTTOM_LEFT:
            new_x_min = l_im + padding_x
            new_y_min = b_im + padding_y
            new_x_max = new_x_min + scaled_width
            new_y_max = new_y_min + scaled_height
        elif self.roi_loc == ROI_LOCATION.BOTTOM_RIGHT:
            new_x_max = r_im - padding_x
            new_x_min = new_x_max - scaled_width
            new_y_min = b_im + padding_y
            new_y_max = new_y_min + scaled_height

        new_bounds = (new_x_min, new_y_min, new_x_max, new_y_max)

        scaled_roi_img = hv.Image(
            data_np,
            bounds=new_bounds,
            kdims=img.kdims,
            vdims=img.vdims,
            label=img.label,
        ).opts(
            xlim=(l_im, r_im),
            ylim=(b_im, t_im),
            cmap=self.cmap.get_cmap(),
            **addnl_opts,
        )

        # add bounding box
        scaled_roi_img = scaled_roi_img * hv.Bounds(new_bounds, label=img.label).opts(
            color=self.color,
            line_width=self.line_width,
        )

        return scaled_roi_img

    def get_separate_roi(
        self,
        img: hv.Image,
        addnl_opts: Optional[dict] = {},
    ) -> hv.Image:
        """
        Return a new image object for the ROI with size scaled to have equivalent width to original image
        """

        self._validate()

        # We want to maintain crop aspect ratio with equivalent image width
        image_width = img.opts["width"]
        roi_width = self.width()
        roi_height = self.height()

        # Extract data
        data_np = self._create_roi_array(img)

        # New zoom scale -> always determined by width
        im_l, _, im_r, _ = img.bounds.lbrt()
        zoom_scale_x = (im_r - im_l) / roi_width
        data_np = zoom(data_np, zoom_scale_x, order=self.zoom_order)

        scaled_roi_img = hv.Image(
            data_np,
            bounds=img.bounds.lbrt(),
            kdims=img.kdims,
            vdims=img.vdims,
            label="ROI",
        ).opts(
            cmap=self.cmap.get_cmap(),
            width=int(image_width),
            height=int(image_width * roi_height / roi_width),
            **addnl_opts,
        )

        # Adjust by 0.5 because of convention of hv.Image coordinates range as (-0.5, N-0.5)
        le, be, re, te = scaled_roi_img.bounds.lbrt()
        extent_padded = (le + 0.5, be + 0.5, re - 0.5, te - 0.5)

        scaled_roi_img = scaled_roi_img * hv.Bounds(
            extent_padded,
            label="ROI_BB",  # purposely use different label to prevent title from showing up
        ).opts(
            color=self.color,
            line_width=self.line_width,
        )

        return scaled_roi_img

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

    def _create_roi_array(self, img: hv.Image) -> np.ndarray:

        x1 = min(self.point1.x, self.point2.x)
        x2 = max(self.point1.x, self.point2.x)

        y1 = min(self.point1.y, self.point2.y)
        y2 = max(self.point1.y, self.point2.y)

        cropped_region = img[x1:x2, y1:y2].clone()

        data_np = cropped_region.data["Value"]

        data_np = np.flipud(data_np)

        return data_np

    def _validate(self):

        if self.point1 is None or self.point2 is None:
            raise ValueError("To create ROI, both points must be provided")

        if self.height() == 0:
            raise ValueError("ROI height cannot be zero")

        if self.width() == 0:
            raise ValueError("ROI width cannot be zero")
