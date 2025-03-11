from enum import Enum

import numpy as np


class ROI_STATE(Enum):

    INACTIVE = 0
    FIRST_SELECTION = 1
    SECOND_SELECTION = 2
    ACTIVE = 3

    # Add comparability to ROI_STATE
    def __lt__(self, other):
        return self.value < other.value

    def __le__(self, other):
        return self.value <= other.value

    def __eq__(self, other):
        return self.value == other.value

    def __ne__(self, other):
        return self.value != other.value

    def __gt__(self, other):
        return self.value > other.value

    def __ge__(self, other):
        return self.value >= other.value


class METRICS_STATE(Enum):

    # No text or diff maps shown
    INACTIVE = 0

    # Compute difference map but no text metrics shown
    MAP = 1

    # Compute text metrics but no difference map shown
    TEXT = 2

    # show both
    ALL = 3


class ROI_VIEW_MODE(Enum):
    Separate = 0
    Overlayed = 1


class ROI_LOCATION(Enum):
    TOP_LEFT = "Top Left"
    TOP_RIGHT = "Top Right"
    BOTTOM_LEFT = "Bottom Left"
    BOTTOM_RIGHT = "Bottom Right"
