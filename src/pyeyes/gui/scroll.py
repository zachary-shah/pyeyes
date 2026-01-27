import time
from typing import Callable

import numpy as np
from bokeh.events import MouseWheel
from bokeh.models import WheelZoomTool


def _bokeh_disable_wheel_zoom_tool(plot, element):
    """
    To use scroll functionality, we need to disable the default wheel zoom tool for viewers.
    """
    tools_to_remove = []
    for tool in plot.state.toolbar.tools:
        if isinstance(tool, WheelZoomTool):
            tools_to_remove.append(tool)
    for tool in tools_to_remove:
        plot.state.toolbar.tools.remove(tool)


class ScrollHandler:

    def __init__(
        self,
        callback_func: Callable,
        buffer_time: float = 250,
    ):
        """
        Mixin to Viewer classes to add scroll functionality.
        """
        self._MIN_BUFFER_TIME = (
            0.5  # [ms], minimum buffer time to prevent highly rapid inputs
        )
        self._last_scroll_time_min_buffer = time.time() * 1000  # [ms]

        # Set up log of last scroll time
        self._last_scroll_time = self._last_scroll_time_min_buffer  # [ms]

        # Buffer time, which needs to be updated.
        self._scroll_buffer_time = buffer_time  # [ms]

        # Callback function which will use the scroll value to update a widget.
        self._callback_func = callback_func

        # Buffer of deltas to average over when sending delta
        self._delta_buffer = []

    def update_buffer_time(self, new_buffer_time: float):
        self._scroll_buffer_time = new_buffer_time  # [ms]

    def _handle_scroll(self, event):
        """
        Internal function to handle scroll events.
        """

        delta = event.delta
        if delta is None:
            return
        assert isinstance(delta, (int, float)), f"delta is not a float: {delta}"

        # Minimum time to prevent extreme throttling
        current_time = time.time() * 1000  # [ms]
        if (
            abs(current_time - self._last_scroll_time_min_buffer)
            < self._MIN_BUFFER_TIME
        ):
            return
        else:
            self._last_scroll_time_min_buffer = current_time

        self._delta_buffer.append(delta)

        if abs(current_time - self._last_scroll_time) < self._scroll_buffer_time:
            return
        else:
            delta_use = np.mean(self._delta_buffer)

            # Update buffer
            self._last_scroll_time = current_time
            self._last_scroll_time_min_buffer = current_time
            self._delta_buffer = []

            # Run requested callback function
            self._callback_func(delta_use)

    def make_scroll_hook(self):
        """
        Get the hook to add to Bokeh plot to handle scroll events.
        """
        assert (
            self._callback_func is not None
        ), "Callback function must be set before making scroll hook"

        def scroll_hook(plot, element):
            plot.state.on_event(MouseWheel, self._handle_scroll)

        return scroll_hook
