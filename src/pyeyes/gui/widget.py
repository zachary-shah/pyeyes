from typing import Any, Callable, Dict, List, Optional

import panel as pn
import param

# TODO types


class Widget:
    def __init__(self, name, viewer: Optional[Callable] = None):
        self.name = name
        self.viewer = viewer

    def set_param(self, key=None, value=None):
        if key is None:
            key = self.name
        self.viewer.set_param(self.name, value)

    def subscribe(self, key, callback):
        self.viewer.subscribe(key, callback)

    def assign_to_viewer(self, viewer):
        self.viewer = viewer
        if isinstance(self.value, dict):
            for key, value in self.value.items():
                self.set_param(key, value)
        else:
            self.set_param(value=self.value)

    @property
    def display_name(self):
        return self.widget.name

    @display_name.setter
    def display_name(self, value):
        self.widget.name = value

    def get_widget(self):
        return self.widget


class Select(Widget):
    def __init__(
        self,
        name,
        options: Optional[List[str]],
        value: str,
        display_name: Optional[str] = None,
        callback: Optional[Callable] = None,
        viewer: Optional[Callable] = None,
    ):
        super().__init__(name, viewer)
        display_name = display_name or name
        self.widget = pn.widgets.Select(name=display_name, options=options, value=value)
        self.callback = callback

        def _callback(event):
            # call self callback if there's one
            if self.callback:
                self.callback(event.new)
            if self.viewer:
                # dispatch the event to all subscribers
                self.set_param(event.new)

        self.widget.param.watch(_callback, "value")

    @property
    def value(self):
        return self.widget.value

    @value.setter
    def value(self, value):
        self.widget.value = value

    @property
    def options(self):
        return self.widget.options


class EditableIntSlider(Widget):
    def __init__(
        self,
        name,
        start: int,
        end: int,
        value: int,
        step: int = 1,
        display_name: Optional[str] = None,
        callback: Optional[Callable] = None,
        viewer: Optional[Callable] = None,
    ):
        super().__init__(name, viewer)
        display_name = display_name or name
        self.widget = pn.widgets.EditableIntSlider(
            name=display_name, start=start, end=end, step=step, value=value
        )
        self.callback = callback

        def _callback(event):
            # call self callback if there's one
            if self.callback:
                callback(event.new)
            # dispatch the event to all subscribers
            self.set_param(value=event.new)

        self.widget.param.watch(_callback, "value")

    @property
    def value(self):
        return self.widget.value

    @value.setter
    def value(self, value):
        self.widget.value = value

    @property
    def end(self):
        return self.widget.end

    @end.setter
    def end(self, value):
        self.widget.end = value


class CheckBox(Widget):
    def __init__(self, name, display_name, value=False, callback=None, viewer=None):
        super().__init__(name, viewer)
        self.widget = pn.widgets.Checkbox(name=display_name, value=value)
        self.callback = callback

        def _callback(event):
            # call self callback if there's one
            if self.callback:
                self.callback(event.new)
            if self.viewer:
                # dispatch the event to all subscribers
                self.set_param(value=event.new)

        self.widget.param.watch(_callback, "value")

    @property
    def value(self):
        return self.widget.value


class IntRangeSlider(Widget):
    def __init__(
        self,
        name,
        start,
        end,
        value,
        step=1,
        display_name=None,
        callback=None,
        viewer=None,
    ):
        super().__init__(name, viewer)
        display_name = display_name or name
        self.widget = pn.widgets.IntRangeSlider(
            name=display_name, start=start, end=end, value=value, step=step
        )
        self.callback = callback

        def _callback(event):
            # call self callback if there's one
            if self.callback:
                self.callback(event.new)
            if self.viewer:
                # dispatch the event to all subscribers
                self.set_param(value=event.new)

        self.widget.param.watch(_callback, "value")

    @property
    def value(self):
        return self.widget.value


# WIDGETS = {
#     'select': pn.widgets.Select,
#     "int_slider": pn.widgets.EditableIntSlider
# }


# class Widget():
#     def __init__(self, name, type, callback, **kwargs):
#         self.widget = WIDGETS[type](name=name, **kwargs)
#         self.widget.param.watch(callback, "value")

#     def get_widget(self):
#         return self.widget

#     @property
#     def value(self):
#         return self.widget.value

#     @value.setter
#     def value(self, value):
#         self.widget.value = value
