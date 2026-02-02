"""
Widget abstractions for pyeyes GUI components.

This module provides wrapper classes around Panel widgets that enable:
- Centralized parameter management via viewer's _parameters dict
- Subscription-based updates between widgets
- CSS class forwarding for Playwright testing
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import panel as pn


class Widget:
    """
    Base class for all pyeyes widget wrappers.

    Provides common functionality for:
    - Connecting to a parent viewer
    - Setting/getting parameters on the viewer
    - Subscribing to parameter changes
    - Accessing the underlying Panel widget
    """

    def __init__(self, name: str, viewer: Optional[Any] = None):
        """
        Initialize base widget.

        Parameters
        ----------
        name : str
            Unique identifier for this widget (used as parameter key)
        viewer : optional
            Parent viewer instance to connect to
        """
        self.name = name
        self.viewer = viewer
        self.widget: Optional[pn.widgets.Widget] = None
        self._callback: Optional[Callable] = None

    def assign_to_viewer(self, viewer: Any) -> None:
        """
        Connect this widget to a viewer.

        Parameters
        ----------
        viewer : Viewer
            Parent viewer instance
        """
        self.viewer = viewer

    def get_widget(self) -> pn.widgets.Widget:
        """Return the underlying Panel widget."""
        return self.widget

    @property
    def value(self) -> Any:
        """Get the widget's current value."""
        if self.widget is not None:
            return self.widget.value
        return None

    @value.setter
    def value(self, val: Any) -> None:
        """Set the widget's value."""
        if self.widget is not None:
            self.widget.value = val

    @property
    def visible(self) -> bool:
        """Get widget visibility."""
        if self.widget is not None:
            return self.widget.visible
        return True

    @visible.setter
    def visible(self, val: bool) -> None:
        """Set widget visibility."""
        if self.widget is not None:
            self.widget.visible = val

    @property
    def disabled(self) -> bool:
        """Get widget disabled state."""
        if self.widget is not None:
            return self.widget.disabled
        return False

    @disabled.setter
    def disabled(self, val: bool) -> None:
        """Set widget disabled state."""
        if self.widget is not None:
            self.widget.disabled = val

    @property
    def display_name(self) -> str:
        """Get the display name shown in the UI."""
        if self.widget is not None:
            return self.widget.name
        return self.name

    @display_name.setter
    def display_name(self, val: str) -> None:
        """Set the display name shown in the UI."""
        if self.widget is not None:
            self.widget.name = val


class Select(Widget):
    """Wrapper for pn.widgets.Select (dropdown selector)."""

    def __init__(
        self,
        name: str,
        options: List[str],
        value: str,
        display_name: Optional[str] = None,
        css_classes: Optional[List[str]] = None,
        callback: Optional[Callable] = None,
        viewer: Optional[Any] = None,
    ):
        super().__init__(name, viewer)
        display_name = display_name or name
        self._callback = callback

        self.widget = pn.widgets.Select(
            name=display_name,
            options=options,
            value=value,
            css_classes=css_classes or [],
        )

        def _on_change(event):
            if self._callback:
                self._callback(event.new)

        self.widget.param.watch(_on_change, "value")

    @property
    def options(self) -> List[str]:
        """Get available options."""
        return self.widget.options

    @options.setter
    def options(self, val: List[str]) -> None:
        """Set available options."""
        self.widget.options = val


class EditableIntSlider(Widget):
    """Wrapper for pn.widgets.EditableIntSlider."""

    def __init__(
        self,
        name: str,
        start: int,
        end: int,
        value: int,
        step: int = 1,
        display_name: Optional[str] = None,
        css_classes: Optional[List[str]] = None,
        callback: Optional[Callable] = None,
        viewer: Optional[Any] = None,
    ):
        super().__init__(name, viewer)
        display_name = display_name or name
        self._callback = callback

        self.widget = pn.widgets.EditableIntSlider(
            name=display_name,
            start=start,
            end=end,
            step=step,
            value=value,
            css_classes=css_classes or [],
        )

        def _on_change(event):
            if self._callback:
                self._callback(event.new)

        self.widget.param.watch(_on_change, "value")

    @property
    def start(self) -> int:
        return self.widget.start

    @start.setter
    def start(self, val: int) -> None:
        self.widget.start = val

    @property
    def end(self) -> int:
        return self.widget.end

    @end.setter
    def end(self, val: int) -> None:
        self.widget.end = val

    @property
    def step(self) -> int:
        return self.widget.step

    @step.setter
    def step(self, val: int) -> None:
        self.widget.step = val


class EditableFloatSlider(Widget):
    """Wrapper for pn.widgets.EditableFloatSlider."""

    def __init__(
        self,
        name: str,
        start: float,
        end: float,
        value: float,
        step: float = 0.1,
        display_name: Optional[str] = None,
        css_classes: Optional[List[str]] = None,
        callback: Optional[Callable] = None,
        viewer: Optional[Any] = None,
    ):
        super().__init__(name, viewer)
        display_name = display_name or name
        self._callback = callback

        self.widget = pn.widgets.EditableFloatSlider(
            name=display_name,
            start=start,
            end=end,
            step=step,
            value=value,
            css_classes=css_classes or [],
        )

        def _on_change(event):
            if self._callback:
                self._callback(event.new)

        self.widget.param.watch(_on_change, "value")

    @property
    def start(self) -> float:
        return self.widget.start

    @start.setter
    def start(self, val: float) -> None:
        self.widget.start = val

    @property
    def end(self) -> float:
        return self.widget.end

    @end.setter
    def end(self, val: float) -> None:
        self.widget.end = val

    @property
    def step(self) -> float:
        return self.widget.step

    @step.setter
    def step(self, val: float) -> None:
        self.widget.step = val


class EditableRangeSlider(Widget):
    """Wrapper for pn.widgets.EditableRangeSlider."""

    def __init__(
        self,
        name: str,
        start: float,
        end: float,
        value: Tuple[float, float],
        step: float = 0.1,
        display_name: Optional[str] = None,
        css_classes: Optional[List[str]] = None,
        format=None,
        callback: Optional[Callable] = None,
        viewer: Optional[Any] = None,
    ):
        super().__init__(name, viewer)
        display_name = display_name or name
        self._callback = callback

        kwargs = {}
        if format is not None:
            kwargs["format"] = format

        self.widget = pn.widgets.EditableRangeSlider(
            name=display_name,
            start=start,
            end=end,
            step=step,
            value=value,
            css_classes=css_classes or [],
            **kwargs,
        )

        def _on_change(event):
            if self._callback:
                self._callback(event.new)

        self.widget.param.watch(_on_change, "value")

    @property
    def start(self) -> float:
        return self.widget.start

    @start.setter
    def start(self, val: float) -> None:
        self.widget.start = val

    @property
    def end(self) -> float:
        return self.widget.end

    @end.setter
    def end(self, val: float) -> None:
        self.widget.end = val

    @property
    def step(self) -> float:
        return self.widget.step

    @step.setter
    def step(self, val: float) -> None:
        self.widget.step = val


class IntRangeSlider(Widget):
    """Wrapper for pn.widgets.IntRangeSlider."""

    def __init__(
        self,
        name: str,
        start: int,
        end: int,
        value: Tuple[int, int],
        step: int = 1,
        display_name: Optional[str] = None,
        css_classes: Optional[List[str]] = None,
        callback: Optional[Callable] = None,
        viewer: Optional[Any] = None,
    ):
        super().__init__(name, viewer)
        display_name = display_name or name
        self._callback = callback

        self.widget = pn.widgets.IntRangeSlider(
            name=display_name,
            start=start,
            end=end,
            step=step,
            value=value,
            css_classes=css_classes or [],
        )

        def _on_change(event):
            if self._callback:
                self._callback(event.new)

        self.widget.param.watch(_on_change, "value")

    @property
    def start(self) -> int:
        return self.widget.start

    @start.setter
    def start(self, val: int) -> None:
        self.widget.start = val

    @property
    def end(self) -> int:
        return self.widget.end

    @end.setter
    def end(self, val: int) -> None:
        self.widget.end = val

    @property
    def step(self) -> int:
        return self.widget.step

    @step.setter
    def step(self, val: int) -> None:
        self.widget.step = val

    @property
    def bounds(self) -> Tuple[int, int]:
        """Get the slider bounds as (start, end)."""
        return (self.widget.start, self.widget.end)

    @bounds.setter
    def bounds(self, val: Tuple[int, int]) -> None:
        """Set the slider bounds."""
        self.widget.start = val[0]
        self.widget.end = val[1]


class IntSlider(Widget):
    """Wrapper for pn.widgets.IntSlider."""

    def __init__(
        self,
        name: str,
        start: int,
        end: int,
        value: int,
        step: int = 1,
        display_name: Optional[str] = None,
        css_classes: Optional[List[str]] = None,
        callback: Optional[Callable] = None,
        viewer: Optional[Any] = None,
    ):
        super().__init__(name, viewer)
        display_name = display_name or name
        self._callback = callback

        self.widget = pn.widgets.IntSlider(
            name=display_name,
            start=start,
            end=end,
            step=step,
            value=value,
            css_classes=css_classes or [],
        )

        def _on_change(event):
            if self._callback:
                self._callback(event.new)

        self.widget.param.watch(_on_change, "value")

    @property
    def start(self) -> int:
        return self.widget.start

    @start.setter
    def start(self, val: int) -> None:
        self.widget.start = val

    @property
    def end(self) -> int:
        return self.widget.end

    @end.setter
    def end(self, val: int) -> None:
        self.widget.end = val

    @property
    def step(self) -> int:
        return self.widget.step

    @step.setter
    def step(self, val: int) -> None:
        self.widget.step = val


class IntInput(Widget):
    """Wrapper for pn.widgets.IntInput."""

    def __init__(
        self,
        name: str,
        value: int,
        start: Optional[int] = None,
        end: Optional[int] = None,
        display_name: Optional[str] = None,
        css_classes: Optional[List[str]] = None,
        callback: Optional[Callable] = None,
        viewer: Optional[Any] = None,
        width: Optional[int] = None,
    ):
        super().__init__(name, viewer)
        display_name = display_name or name
        self._callback = callback

        kwargs = {
            "name": display_name,
            "value": value,
            "css_classes": css_classes or [],
        }
        if start is not None:
            kwargs["start"] = start
        if end is not None:
            kwargs["end"] = end
        if width is not None:
            kwargs["width"] = width

        self.widget = pn.widgets.IntInput(**kwargs)

        def _on_change(event):
            if self._callback:
                self._callback(event.new)

        self.widget.param.watch(_on_change, "value")

    @property
    def start(self) -> Optional[int]:
        return self.widget.start

    @start.setter
    def start(self, val: Optional[int]) -> None:
        self.widget.start = val

    @property
    def end(self) -> Optional[int]:
        return self.widget.end

    @end.setter
    def end(self, val: Optional[int]) -> None:
        self.widget.end = val


class Checkbox(Widget):
    """Wrapper for pn.widgets.Checkbox."""

    def __init__(
        self,
        name: str,
        value: bool = False,
        display_name: Optional[str] = None,
        css_classes: Optional[List[str]] = None,
        callback: Optional[Callable] = None,
        viewer: Optional[Any] = None,
    ):
        super().__init__(name, viewer)
        display_name = display_name or name
        self._callback = callback

        self.widget = pn.widgets.Checkbox(
            name=display_name,
            value=value,
            css_classes=css_classes or [],
        )

        def _on_change(event):
            if self._callback:
                self._callback(event.new)

        self.widget.param.watch(_on_change, "value")


class RadioButtonGroup(Widget):
    """Wrapper for pn.widgets.RadioButtonGroup."""

    def __init__(
        self,
        name: str,
        options: List[str],
        value: str,
        button_type: str = "primary",
        button_style: str = "outline",
        orientation: str = "horizontal",
        display_name: Optional[str] = None,
        css_classes: Optional[List[str]] = None,
        callback: Optional[Callable] = None,
        viewer: Optional[Any] = None,
    ):
        super().__init__(name, viewer)
        display_name = display_name or name
        self._callback = callback

        self.widget = pn.widgets.RadioButtonGroup(
            name=display_name,
            options=options,
            value=value,
            button_type=button_type,
            button_style=button_style,
            orientation=orientation,
            css_classes=css_classes or [],
        )

        def _on_change(event):
            if self._callback:
                self._callback(event.new)

        self.widget.param.watch(_on_change, "value")

    @property
    def options(self) -> List[str]:
        return self.widget.options

    @options.setter
    def options(self, val: List[str]) -> None:
        self.widget.options = val


class CheckButtonGroup(Widget):
    """Wrapper for pn.widgets.CheckButtonGroup."""

    def __init__(
        self,
        name: str,
        options: List[str],
        value: List[str],
        button_type: str = "primary",
        button_style: str = "outline",
        orientation: str = "horizontal",
        display_name: Optional[str] = None,
        css_classes: Optional[List[str]] = None,
        callback: Optional[Callable] = None,
        viewer: Optional[Any] = None,
    ):
        super().__init__(name, viewer)
        display_name = display_name or name
        self._callback = callback

        self.widget = pn.widgets.CheckButtonGroup(
            name=display_name,
            options=options,
            value=value,
            button_type=button_type,
            button_style=button_style,
            orientation=orientation,
            css_classes=css_classes or [],
        )

        def _on_change(event):
            if self._callback:
                self._callback(event.new)

        self.widget.param.watch(_on_change, "value")

    @property
    def options(self) -> List[str]:
        return self.widget.options

    @options.setter
    def options(self, val: List[str]) -> None:
        self.widget.options = val


class CheckBoxGroup(Widget):
    """Wrapper for pn.widgets.CheckBoxGroup."""

    def __init__(
        self,
        name: str,
        options: List[str],
        value: List[str],
        display_name: Optional[str] = None,
        css_classes: Optional[List[str]] = None,
        callback: Optional[Callable] = None,
        viewer: Optional[Any] = None,
    ):
        super().__init__(name, viewer)
        display_name = display_name or name
        self._callback = callback

        self.widget = pn.widgets.CheckBoxGroup(
            name=display_name,
            options=options,
            value=value,
            css_classes=css_classes or [],
        )

        def _on_change(event):
            if self._callback:
                self._callback(event.new)

        self.widget.param.watch(_on_change, "value")

    @property
    def options(self) -> List[str]:
        return self.widget.options

    @options.setter
    def options(self, val: List[str]) -> None:
        self.widget.options = val


class Button(Widget):
    """Wrapper for pn.widgets.Button."""

    def __init__(
        self,
        name: str,
        button_type: str = "primary",
        display_name: Optional[str] = None,
        css_classes: Optional[List[str]] = None,
        on_click: Optional[Callable] = None,
        viewer: Optional[Any] = None,
    ):
        super().__init__(name, viewer)
        display_name = display_name or name
        self._on_click = on_click

        self.widget = pn.widgets.Button(
            name=display_name,
            button_type=button_type,
            css_classes=css_classes or [],
        )

        if on_click is not None:
            self.widget.on_click(on_click)

    def on_click(self, callback: Callable) -> None:
        """Register a click handler."""
        self._on_click = callback
        self.widget.on_click(callback)


class TextInput(Widget):
    """Wrapper for pn.widgets.TextInput."""

    def __init__(
        self,
        name: str,
        value: str = "",
        placeholder: str = "",
        display_name: Optional[str] = None,
        css_classes: Optional[List[str]] = None,
        callback: Optional[Callable] = None,
        viewer: Optional[Any] = None,
    ):
        super().__init__(name, viewer)
        display_name = display_name or name
        self._callback = callback

        self.widget = pn.widgets.TextInput(
            name=display_name,
            value=value,
            placeholder=placeholder,
            css_classes=css_classes or [],
        )

        def _on_change(event):
            if self._callback:
                self._callback(event.new)

        self.widget.param.watch(_on_change, "value")

    @property
    def placeholder(self) -> str:
        return self.widget.placeholder

    @placeholder.setter
    def placeholder(self, val: str) -> None:
        self.widget.placeholder = val


class TextAreaInput(Widget):
    """Wrapper for pn.widgets.TextAreaInput."""

    def __init__(
        self,
        name: str,
        value: str = "",
        placeholder: str = "",
        height: Optional[int] = None,
        display_name: Optional[str] = None,
        css_classes: Optional[List[str]] = None,
        callback: Optional[Callable] = None,
        viewer: Optional[Any] = None,
    ):
        super().__init__(name, viewer)
        display_name = display_name or name
        self._callback = callback

        kwargs = {
            "name": display_name,
            "value": value,
            "placeholder": placeholder,
            "css_classes": css_classes or [],
        }
        if height is not None:
            kwargs["height"] = height

        self.widget = pn.widgets.TextAreaInput(**kwargs)

        def _on_change(event):
            if self._callback:
                self._callback(event.new)

        self.widget.param.watch(_on_change, "value")

    @property
    def placeholder(self) -> str:
        return self.widget.placeholder

    @placeholder.setter
    def placeholder(self, val: str) -> None:
        self.widget.placeholder = val

    @property
    def height(self) -> Optional[int]:
        return self.widget.height

    @height.setter
    def height(self, val: Optional[int]) -> None:
        self.widget.height = val


class StaticText(Widget):
    """Wrapper for pn.widgets.StaticText."""

    def __init__(
        self,
        name: str,
        value: str = "",
        display_name: Optional[str] = None,
        css_classes: Optional[List[str]] = None,
        viewer: Optional[Any] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ):
        super().__init__(name, viewer)
        display_name = display_name or name

        kwargs = {
            "name": display_name,
            "value": value,
            "css_classes": css_classes or [],
        }
        if width is not None:
            kwargs["width"] = width
        if height is not None:
            kwargs["height"] = height

        self.widget = pn.widgets.StaticText(**kwargs)


class ColorPicker(Widget):
    """Wrapper for pn.widgets.ColorPicker."""

    def __init__(
        self,
        name: str,
        value: str = "#000000",
        display_name: Optional[str] = None,
        css_classes: Optional[List[str]] = None,
        callback: Optional[Callable] = None,
        viewer: Optional[Any] = None,
    ):
        super().__init__(name, viewer)
        display_name = display_name or name
        self._callback = callback

        self.widget = pn.widgets.ColorPicker(
            name=display_name,
            value=value,
            css_classes=css_classes or [],
        )

        def _on_change(event):
            if self._callback:
                self._callback(event.new)

        self.widget.param.watch(_on_change, "value")


class RawPanelObject(Widget):
    """
    Wrapper for raw Panel objects (pn.Row, pn.Column, pn.pane.HTML, etc.).

    This allows non-widget Panel objects to be added to Panes.
    """

    def __init__(
        self,
        name: str,
        panel_object: Any,
        viewer: Optional[Any] = None,
    ):
        super().__init__(name, viewer)
        self.widget = panel_object

    @property
    def value(self) -> Any:
        """Raw Panel objects may not have a value."""
        return None

    @value.setter
    def value(self, val: Any) -> None:
        pass

    @property
    def visible(self) -> bool:
        if hasattr(self.widget, "visible"):
            return self.widget.visible
        return True

    @visible.setter
    def visible(self, val: bool) -> None:
        if hasattr(self.widget, "visible"):
            self.widget.visible = val

    @property
    def disabled(self) -> bool:
        if hasattr(self.widget, "disabled"):
            return self.widget.disabled
        return False

    @disabled.setter
    def disabled(self, val: bool) -> None:
        if hasattr(self.widget, "disabled"):
            self.widget.disabled = val
