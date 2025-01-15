from typing import Optional

import holoviews as hv
import numpy as np
from scipy.ndimage import zoom

from .q_cmap.cmap import ColorMap

ROI_LOCATIONS = [
    "top_left",
    "top_right",
    "bottom_left",
    "bottom_right",
]


class ROI:
    def __init__(
        self,
        x1: Optional[int] = None,
        x2: Optional[int] = None,
        y1: Optional[int] = None,
        y2: Optional[int] = None,
        roi_loc: Optional[str] = "top_right",
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

        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

        assert roi_loc in ROI_LOCATIONS
        self.roi_loc = roi_loc
        self.zoom_scale = zoom_scale
        self.cmap = cmap
        self.color = color
        self.line_width = line_width
        self.zoom_order = zoom_order
        self.padding_pct = padding_pct

    def get_inview_roi(
        self,
        img: hv.Image,
        addnl_opts: Optional[dict] = {},
    ) -> hv.Image:
        """
        Return the overlay layer for the ROI inside the main image.
        """

        assert self._can_apply(), "Need to set x1, x2, y1, y2 before applying ROI"

        # Original bounds
        main_x_min, main_y_min, main_x_max, main_y_max = img.bounds.lbrt()

        x1, x2 = sorted([self.x1, self.x2])
        y1, y2 = sorted([self.y1, self.y2])

        cropped_region = img[x1:x2, y1:y2].clone()

        data_np = cropped_region.data["Value"]

        # Bilinear zoom
        if self.zoom_scale != 1.0:
            data_np = zoom(data_np, self.zoom_scale, order=self.zoom_order)

        # Flip y-axis
        data_np = np.flipud(data_np)

        # Calculate scaled ROI size in data coordinates
        original_width = x2 - x1
        original_height = y2 - y1

        scaled_width = original_width * self.zoom_scale
        scaled_height = original_height * self.zoom_scale

        # Calculate padding (5% of main image dimensions)
        padding_x = self.padding_pct * (main_x_max - main_x_min)
        padding_y = self.padding_pct * (main_y_max - main_y_min)

        if self.roi_loc == "top_left":
            new_x_min = main_x_min + padding_x
            new_x_max = new_x_min + scaled_width
            new_y_max = main_y_max - padding_y
            new_y_min = new_y_max - scaled_height
        elif self.roi_loc == "top_right":
            new_x_max = main_x_max - padding_x
            new_x_min = new_x_max - scaled_width
            new_y_max = main_y_max - padding_y
            new_y_min = new_y_max - scaled_height
        elif self.roi_loc == "bottom_left":
            new_x_min = main_x_min + padding_x
            new_y_min = main_y_min + padding_y
            new_x_max = new_x_min + scaled_width
            new_y_max = new_y_min + scaled_height
        elif self.roi_loc == "bottom_right":
            new_x_max = main_x_max - padding_x
            new_x_min = new_x_max - scaled_width
            new_y_min = main_y_min + padding_y
            new_y_max = new_y_min + scaled_height

        scaled_roi_img = hv.Image(
            data_np,
            bounds=(new_x_min, new_y_min, new_x_max, new_y_max),
            kdims=img.kdims,
            vdims=img.vdims,
            label=img.label,
        ).opts(
            xlim=(main_x_min, main_x_max),
            ylim=(main_y_min, main_y_max),
            cmap=self.cmap.get_cmap(),
            **addnl_opts,
        )

        # add bounding box
        scaled_roi_img = scaled_roi_img * hv.Bounds(
            (new_x_min, new_y_min, new_x_max, new_y_max), label=img.label
        ).opts(
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

        assert self._can_apply(), "Need to set x1, x2, y1, y2 before applying ROI"

        # Original image bounds
        x0_l, x0_b, x0_r, x0_t = img.bounds.lbrt()

        # We want to maintain crop aspect ratio with equivalent image width
        width = img.opts["width"]

        # Crop Bounds
        x1, x2 = sorted([self.x1, self.x2])
        y1, y2 = sorted([self.y1, self.y2])

        # Extract data
        cropped_region = img[x1:x2, y1:y2].clone()
        data_np = cropped_region.data["Value"]

        # New zoom scale -> always determined by width
        zoom_scale_x = (x0_r - x0_l) / (x2 - x1)
        data_np = zoom(data_np, zoom_scale_x, order=self.zoom_order)

        # Flip y-axis
        data_np = np.flipud(data_np)

        scaled_roi_img = hv.Image(
            data_np,
            bounds=img.bounds.lbrt(),
            kdims=cropped_region.kdims,
            vdims=cropped_region.vdims,
            label="ROI",
        ).opts(
            cmap=self.cmap.get_cmap(),
            width=int(width),
            height=int(width * (y2 - y1) / (x2 - x1)),
            **addnl_opts,
        )

        # Add bounding box around extent
        extent = scaled_roi_img.bounds.lbrt()
        # add padding
        extent_padded = (
            extent[0] + 0.5,
            extent[1] + 0.5,
            extent[2] - 0.5,
            extent[3] - 0.5,
        )

        scaled_roi_img = scaled_roi_img * hv.Bounds(
            extent_padded,
            label="ROI_BB",  # purposely use different label to prevent title from showing up
        ).opts(
            color=self.color,
            line_width=self.line_width,
        )

        return scaled_roi_img

    def _can_apply(self):

        return all(
            [
                self.x1 is not None,
                self.x2 is not None,
                self.y1 is not None,
                self.y2 is not None,
            ]
        )
