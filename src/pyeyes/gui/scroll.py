import time
from datetime import datetime
from typing import Callable

import numpy as np
from bokeh.events import MouseWheel
from bokeh.models import ColumnDataSource, CustomJS


class ScrollHandler:

    def __init__(
        self,
        callback_func: Callable,
        buffer_time: float = 50,
        debug: bool = False,
    ):
        """
        Mixin to Viewer classes to add scroll functionality.
        """

        self._debug = debug
        self._js_latency_est = None

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

        # Bokeh model used as JS->Python bridge.
        # JS updates this CDS with the latest delta; Python listens for changes.
        self.source = ColumnDataSource(data=dict(delta=[0.0], ts=[0.0]))
        self.source.on_change("data", self._on_source_change)

    def update_buffer_time(self, new_buffer_time: float):
        self._scroll_buffer_time = new_buffer_time  # [ms]

    def _on_source_change(self, attr, old, new):
        """
        Bokeh callback for JS-updated scroll deltas.
        """
        try:
            delta_val = float(self.source.data.get("delta", [None])[0])
        except Exception:
            return

        try:
            ts_val = float(self.source.data.get("ts", [None])[0])
        except Exception:
            return

        # get current time same way Date.now() is used in JS
        current_time = datetime.now().timestamp() * 1000  # [ms]

        ts_diff = current_time - ts_val

        if self._debug:
            print(
                f"\033[95m[_on_src_change] Delta: {delta_val}, TS Diff: {ts_diff}\033[0m"
            )

        # estimate base JS latency as the first _on_src_change event
        if self._js_latency_est is None:
            if self._debug:
                print(
                    f"\033[92m[_on_src_change] Estimated JS Latency: {ts_diff}\033[0m"
                )
            self._js_latency_est = ts_diff

        # Fudge factor on whether or not to _handle_scroll
        if ts_diff < self._js_latency_est * 5:
            # Create minimal event-like object expected by _handle_scroll
            self._handle_scroll(delta_val)

    def _handle_scroll(self, delta: float):
        """
        Internal function to handle scroll events.
        """
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

        if self._debug:
            print(f"\033[93m[_handle_scroll] Delta: {delta}\033[0m")

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

    def build_bokeh_scroll_hook(self):
        """
        Build a Bokeh hook function that forwards scroll deltas into the scroll handler.
        """
        source = self.source

        def _bokeh_scroll_hook(plot, element):
            """
            Attach a JS MouseWheel callback that forwards deltas into `scroll_event_source`.
            Only attaches when `scroll_event_source` is provided.
            """
            if source is None:
                return

            token = f"pyeyes_scroll_bound:{source.id}"
            try:
                tags = plot.state.tags
            except Exception:
                tags = None

            if tags is not None:
                if token in tags:
                    return
                tags.append(token)

            plot.state.js_on_event(
                MouseWheel,
                CustomJS(
                    args=dict(source=source),
                    code="""
                        const d = cb_obj.delta;
                        if (d == null) { return; }
                        source.data = {delta: [d], ts: [Date.now()]};
                        source.change.emit();
                    """,
                ),
            )

        return _bokeh_scroll_hook
