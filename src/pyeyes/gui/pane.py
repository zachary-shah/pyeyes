"""
Pane class for grouping widgets into tabs.
"""

from typing import Any, Dict, List, Optional, Union

import panel as pn

from .widget import Widget


class Pane:
    """
    A container for grouping related widgets into a tab.

    The Pane class manages a collection of Widget instances and can
    convert them to a Panel Column for display in a Tabs component.
    """

    def __init__(self, name: str, viewer: Optional[Any] = None):
        """
        Initialize a Pane.

        Parameters
        ----------
        name : str
            Name of the pane (used as tab title)
        viewer : optional
            Parent viewer instance
        """
        self.name = name
        self.viewer = viewer
        self.widgets: Dict[str, Widget] = {}
        self._widget_order: List[str] = []

    def add_widget(self, widget: Widget) -> None:
        """
        Add a widget to this pane.

        Parameters
        ----------
        widget : Widget
            The widget to add
        """
        # Assign viewer to widget if not already set
        if widget.viewer is None and self.viewer is not None:
            widget.assign_to_viewer(self.viewer)

        self.widgets[widget.name] = widget
        if widget.name not in self._widget_order:
            self._widget_order.append(widget.name)

    def get_widget(self, name: str) -> Optional[Widget]:
        """
        Get a widget by name.

        Parameters
        ----------
        name : str
            Name of the widget to retrieve

        Returns
        -------
        Widget or None
            The widget if found, None otherwise
        """
        return self.widgets.get(name)

    def get_widgets(
        self, return_list: bool = True
    ) -> Union[List[Widget], Dict[str, Widget]]:
        """
        Get all widgets in this pane.

        Parameters
        ----------
        return_list : bool
            If True, return as list in order added.
            If False, return as dict.

        Returns
        -------
        list or dict
            Widgets in this pane
        """
        if return_list:
            return [
                self.widgets[name]
                for name in self._widget_order
                if name in self.widgets
            ]
        return self.widgets

    def get_panel_widgets(
        self, return_list: bool = True
    ) -> Union[List[pn.widgets.Widget], Dict[str, pn.widgets.Widget]]:
        """
        Get the underlying Panel widgets.

        Parameters
        ----------
        return_list : bool
            If True, return as list in order added.
            If False, return as dict.

        Returns
        -------
        list or dict
            Panel widgets in this pane
        """
        if return_list:
            return [
                self.widgets[name].get_widget()
                for name in self._widget_order
                if name in self.widgets
            ]
        return {name: widget.get_widget() for name, widget in self.widgets.items()}

    def to_column(self) -> pn.Column:
        """
        Convert this pane's widgets to a Panel Column.

        Returns
        -------
        pn.Column
            A Panel Column containing all widgets
        """
        return pn.Column(*self.get_panel_widgets())

    def set_widget_attr(self, widget_name: str, attr_name: str, value: Any) -> None:
        """
        Set an attribute on a widget in this pane.

        Parameters
        ----------
        widget_name : str
            Name of the widget
        attr_name : str
            Name of the attribute to set
        value : any
            Value to set
        """
        widget = self.get_widget(widget_name)
        if widget is not None:
            setattr(widget, attr_name, value)

    def replace_widget(self, name: str, new_widget: Widget) -> None:
        """
        Replace a widget with a new one.

        Parameters
        ----------
        name : str
            Name of the widget to replace
        new_widget : Widget
            The new widget
        """
        if name in self.widgets:
            # Preserve position in order
            if new_widget.viewer is None and self.viewer is not None:
                new_widget.assign_to_viewer(self.viewer)
            self.widgets[name] = new_widget
        else:
            self.add_widget(new_widget)
