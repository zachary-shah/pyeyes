from enum import Enum
from typing import Callable, List

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


class ROI_VIEW_MODE(Enum):
    Separate = 0
    Overlayed = 1
