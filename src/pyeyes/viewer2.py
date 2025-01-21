from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import panel as pn
import param
from easydict import EasyDict

from .gui.pane import Pane
from .gui.widget import CheckBox, EditableIntSlider, IntRangeSlider, Select, Widget


class Viewer2:
    def __init__(self, name: str):
        # Mapping str -> value. This will store all the mutable parameters of the viewer.
        # When a new widget is created, it can add its parameters to this dictionary.
        self._parameters = {}
        self._prev_parameters = {}

        # Mapping str -> list of callbacks. Each widget can subscribe to events.
        # When a widget is created, it can register its callbacks here.
        self._subscribers = {}

        self.name = name

    def set_param(self, key, value):
        self._prev_parameters[key] = self._parameters.get(key, None)

        self._parameters[key] = value
        for callback in self._subscribers.get(key, []):
            callback(value)

    def get_param(self, key):
        return self._parameters[key]

    def get_prev_param(self, key):
        return self._prev_parameters[key]

    def subscribe(self, key, callback):
        self._subscribers.setdefault(key, []).append(callback)

    def launch(self):
        pn.serve(self.app, title=self.name, show=True)


class Vdims(Widget):
    def __init__(
        self,
        name,
        dims: List[str],
        vdims: List[str],
        viewer: Optional[Callable] = None,
    ):
        super().__init__(name, viewer)
        name = name
        self.vdims = vdims
        vdim_horiz = Select(
            name="vdim_horiz",
            display_name="L/R Viewing Dimension",
            options=dims,
            value=vdims[0],
        )
        vdim_vert = Select(
            name="vdim_vert",
            display_name="U/D Viewing Dimension",
            options=dims,
            value=vdims[1],
        )

        def _vert_change(value):
            old_vert = self.vdims[1]
            self.vdims[1] = value
            if self.vdims[0] == value:
                self.vdims[0] = old_vert
                vdim_horiz.value = self.vdims[0]

            self.viewer.set_param("vdims", self.vdims)

        vdim_vert.callback = _vert_change

        def _horiz_change(value):
            old_horiz = self.vdims[0]
            self.vdims[0] = value
            if self.vdims[1] == value:
                self.vdims[1] = old_horiz
                vdim_vert.value = self.vdims[1]

            self.viewer.set_param("vdims", self.vdims)

        vdim_horiz.callback = _horiz_change

        self.widget = pn.Column(vdim_horiz.get_widget(), vdim_vert.get_widget())

    @property
    def value(self):
        return self.vdims


class Sdims(Widget):
    def __init__(self, name, named_dims, view_dims, cat_dims, dim_size, viewer=None):
        super().__init__(name, viewer)
        self.sdims = [dim for dim in named_dims if dim not in view_dims]

        self.cat_dims = cat_dims
        self.noncat_dims = [dim for dim in named_dims if dim not in cat_dims]

        self.sdims_vals = {}

        self.dim_sizes = dim_size

        self.cat_sliders = []
        self.noncat_sliders = []

        for dim in self.sdims:

            def _callback(event, dim=dim):
                self.sdims_vals[dim] = event
                self.viewer.set_param("sdims_vals", self.sdims_vals)

            if dim in self.cat_dims:
                self.sdims_vals[dim] = self.cat_dims[dim][0]
                s = Select(
                    name=dim,
                    options=self.cat_dims[dim],
                    value=self.sdims_vals[dim],
                    viewer=viewer,
                    callback=_callback,
                )
                self.cat_sliders.append(s)
            else:
                self.sdims_vals[dim] = self.dim_sizes[dim] // 2
                s = EditableIntSlider(
                    name=dim,
                    start=0,
                    end=self.dim_sizes[dim] - 1,
                    value=self.sdims_vals[dim],
                    viewer=self,
                    callback=_callback,
                )
                self.noncat_sliders.append(s)

        def _update_sdims(value):
            new_sdims = [dim for dim in named_dims if dim not in value]
            new_dim = set(new_sdims) - set(self.sdims)
            if len(new_dim) == 0:
                return
            assert len(new_dim) == 1
            new_dim = new_dim.pop()
            for s in self.noncat_sliders:
                if s.name in value:
                    s.name = new_dim
                    s.display_name = new_dim
                    s.end = self.dim_sizes[new_dim] - 1
                    s.value = self.dim_sizes[new_dim] // 2
            self.sdims = new_sdims
            self.viewer.set_param("sdims", self.sdims)

        self.subscribe("vdims", _update_sdims)

        self.widget = pn.Column(
            *[s.get_widget() for s in self.cat_sliders + self.noncat_sliders]
        )

    @property
    def value(self):
        return {"sdims_vals": self.sdims_vals, "sdims": self.sdims}


SIZE_SCALE_MIN = 200
SIZE_SCALE_MAX = 1000
SLICE_SCALE_STEP = 10

CROP_MIN = 0
CROP_MAX = 100
CROP_STEP = 1


class ComparativeViewer2(Viewer2):
    def __init__(
        self,
        data: Dict[str, np.ndarray],
        named_dims: List[str],
        view_dims: List[str],
        cat_dims: Optional[Dict[str, List[str]]] = None,
        name: str = "MRI Viewer",
    ):
        super().__init__(name)
        # TODO: assert sizes

        self.set_param("sdims", [dim for dim in named_dims if dim not in view_dims])
        self.set_param("vdims", view_dims)
        self.set_param("flip_ud", False)
        self.set_param("flip_lr", False)
        self.set_param("size_scale", 400)
        self.set_param("lr_crop", (0, 100))
        self.set_param("ud_crop", (0, 100))

        # Following are parameters that are immutable
        self.named_dims = named_dims

        self.dim_sizes = {}
        # all data should have the same size, so we just query the first
        data_ex = next(iter(data.values()))
        for i, dim in enumerate(self.named_dims):
            self.dim_sizes[dim] = data_ex.shape[i]

        self.cat_dims = cat_dims
        self.noncat_dims = [dim for dim in named_dims if dim not in cat_dims]

        # Initialize the viewer panes and widgets
        self.panes = {}
        self._setup_view_pane()

        self.app = pn.Row(self.get_control_panel())

    def get_control_panel(self):
        return pn.Tabs(
            *[
                (name, pn.Column(*pane.get_widgets()))
                for name, pane in self.panes.items()
            ]
        )

    def _setup_view_pane(self):
        view_pane = Pane(name="View", viewer=self)

        view_pane.add_widget(
            Vdims("vdims", self.noncat_dims, self.get_param("vdims"), self)
        )
        view_pane.add_widget(
            Sdims(
                "sdim",
                self.named_dims,
                self.get_param("vdims"),
                self.cat_dims,
                self.dim_sizes,
                self,
            )
        )
        view_pane.add_widget(CheckBox("flip_ud", "Flip U/D", viewer=self))
        view_pane.add_widget(CheckBox("flip_lr", "Flip L/R", viewer=self))
        view_pane.add_widget(
            EditableIntSlider(
                "size_scale",
                start=SIZE_SCALE_MIN,
                end=SIZE_SCALE_MAX,
                step=SLICE_SCALE_STEP,
                value=self.get_param("size_scale"),
                viewer=self,
            )
        )
        lr_crop = IntRangeSlider(
            "lr_crop",
            step=CROP_STEP,
            display_name="L/R Display Range",
            end=self.dim_sizes[self.get_param("vdims")[0]],
            start=CROP_MIN,
            viewer=self,
            value=self.get_param("lr_crop"),
        )

        def _callback(vdims):
            lr_crop.end = self.dim_sizes[vdims[0]]

        lr_crop.subscribe("vdims", _callback)
        view_pane.add_widget(lr_crop)
        ud_crop = IntRangeSlider(
            "ud_crop",
            step=CROP_STEP,
            display_name="U/D Display Range",
            end=self.dim_sizes[self.get_param("vdims")[1]],
            start=CROP_MIN,
            viewer=self,
            value=self.get_param("ud_crop"),
        )

        def _callback(vdims):
            ud_crop.end = self.dim_sizes[vdims[1]]

        ud_crop.subscribe("vdims", _callback)
        view_pane.add_widget(ud_crop)
        view_pane.add_widget(CheckBox("single_toggle", "Single View", viewer=self))
        self.panes["view"] = view_pane
