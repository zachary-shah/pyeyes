"""
Slicers: Defined as classes that take N-dimensional data and can return a 2D view of that data given some input
"""

from typing import Sequence, Optional

import param
import numpy as np
import holoviews as hv
import panel as pn

hv.extension("bokeh")
pn.extension()

VALID_COLORMAPS = [
    'gray', 
    'viridis', 
    'inferno', 
    'PRGn', 
    'RdBu', 
    'HighContrast', 
    'Iridescent',
    'Plasma', 
    'Magma', 
]

class NDSlicer(param.Parameterized):
    # Global Parameters
    vmin       = param.Number(default=0.0)
    vmax       = param.Number(default=1.0)
    size_scale = param.Number(default=400, bounds=(100, 1000), step=10)
    cmap       = param.ObjectSelector(default='gray', objects=VALID_COLORMAPS)
    
    # Dimension Parameters
    dim_indices = param.Dict(default={}, doc="Mapping: dim_name -> int index")
    
    def __init__(self, 
                 data: hv.Dataset,
                 vdims: Sequence[str],
                 cdim:  Optional[str] = None,
                 clabs: Optional[Sequence[str]] = None,
                 **params):
        
        """
        Slicer for N-dimensional data. This class is meant to be subclassed.

        Way data will be sliced:
        - vdims: Viewing dimensions. Should always be length 2
        - cdim: Collate-able dimension. If provided, will return a 1D layout of 2D slices rather than a single 2D image slice.
          Can also provide labels for each collated image.
        """
        
        super().__init__(**params)

        self.data = data

        # all dimensions
        self.ndims = [d.name for d in data.kdims][::-1]

        # Dictionary of all total size of each dimension
        self.dim_sizes = {}
        for dim in self.ndims:
            # Hacky. FIXME
            self.dim_sizes[dim] = data.aggregate(dim, np.mean).data[dim].size

        assert len(vdims) == 2, "Viewing dims must be length 2"
        assert np.array([vd in self.ndims for vd in vdims]).all(), "Viewing dims must be in all dims"
        
        # view dimensions
        self.vdims = vdims

        # collate-able dimension
        if cdim is not None:
            assert cdim in self.ndims, "Collate dim must be in named dims"

            if clabs is not None:
                assert len(clabs) == self.dim_sizes[cdim], "Collate labels must match collate dimension size"
            else:
                # assume data categorical. FIXME: infer c data type in general
                clabs = self.data.aggregate(self.cdim, np.mean).data[self.cdim].tolist()
            
            self.clabs = clabs
            self.cdim = cdim
            self.Nc   = self.dim_sizes[cdim]
            self.non_sdims = [cdim] + vdims
        else:
            self.clabs = None
            self.cdim = None
            self.Nc   = 1
            self.non_sdims = vdims
            
        # sliceable dimensions
        self.sdims = [d for d in self.ndims if d not in self.non_sdims]
        
        # Start in the center of each sliceable dimension
        slice_dim_names = {}
        for dim in self.sdims:
            slice_dim_names[dim] = self.dim_sizes[dim] // 2

        # Set default slice indicess
        self.param.dim_indices.default = slice_dim_names
        self.dim_indices = slice_dim_names
        
        # Update color limits
        mn = np.min(np.stack([data[v.name] for v in data.vdims]))
        mx = np.max(np.stack([data[v.name] for v in data.vdims]))
        self.param.vmin.default = mn
        self.param.vmin.bounds = (mn, mx)
        self.param.vmin.step = (mx-mn)/200
        self.param.vmax.default = mx
        self.param.vmax.bounds = (mn, mx)
        self.param.vmax.step = (mx-mn)/200
        self.vmin = mn
        self.vmax = mx

        # Scaling to update height and width range
        self.img_dims = np.array([self.dim_sizes[vd] for vd in self.vdims])
 
    @param.depends(
            "dim_indices", 
            "vmin", 
            "vmax",
            "cmap",
            "size_scale",
    )
    def view(self) -> hv.Layout:
        """
        Return the view of the data given the current slice indices.
        """

        # Dimensions to select
        sdim_dict = {dim: self.dim_indices[dim] for dim in self.sdims}

        # Collate case. FIXME: simply code
        if self.cdim is not None:

            imgs = []

            for i in range(self.Nc):

                sliced_2d = self.data.select(
                    **{self.cdim: self.clabs[i]},
                    **sdim_dict
                ).reduce([self.cdim] + self.sdims, np.mean)

                imgs.append(hv.Image(sliced_2d, label=self.clabs[i]))
        else:

            # Select slice indices for each dimension
            sliced_2d = self.data.select(
                    **{self.cdim: i}
            ).reduce(self.sdims, np.mean)
        
            imgs = [hv.Image(sliced_2d)]

        for i in range(len(imgs)):
            
            # TODO: parameterize more options
            imgs[i] = imgs[i].opts(
                cmap=self.cmap,
                xaxis=None, yaxis=None,
                clim=(self.vmin, self.vmax),
                width  = int(self.size_scale * self.img_dims[0] / np.max(self.img_dims)),
                height = int(self.size_scale * self.img_dims[1] / np.max(self.img_dims)),
            )

        return hv.Layout(imgs)

    def get_widgets(self) -> Sequence[pn.widgets.Widget]:
        """
        Return a list of panel widgets to interactively control slicing and other parameters.
        """
        
        sliders = []
        for dim in self.sdims:
            s = pn.widgets.EditableIntSlider(
                name=dim,
                start=0,
                end=self.dim_sizes[dim]-1,
                value=self.dim_indices[dim]
            )

            def _update_dim_indices(event, this_dim=dim):
                self.dim_indices[this_dim] = event.new
                # trigger dim_indices has been changed
                self.param.trigger("dim_indices")
            
            s.param.watch(_update_dim_indices, "value")

            sliders.append(s)


        # vmin/vmax use different Range slider
        range_slier = pn.widgets.EditableRangeSlider(
            name = 'clim',
            start = self.param.vmin.bounds[0],
            end = self.param.vmax.bounds[1],
            value = (self.vmin, self.vmax),
            step = self.param.vmin.step,
        )
        def _update_clim(event):
            self.vmin, self.vmax = event.new
            self.param.trigger("vmin")
            self.param.trigger("vmax")
        range_slier.param.watch(_update_clim, "value")
        sliders.append(range_slier)

        sliders.append(
            self.__add_widget(
                pn.widgets.EditableIntSlider,
                "size_scale",
                start=self.param.size_scale.bounds[0],
                end=self.param.size_scale.bounds[1],
                value=self.size_scale,
                step=self.param.size_scale.step,
            )
        )

        sliders.append(
            self.__add_widget(
                pn.widgets.Select,
                "cmap",
                options=VALID_COLORMAPS,
                value=self.cmap,
            )
        )

        # TODO widgets (dummies for now)
        sliders.append(
            self.__add_widget(
                pn.widgets.CheckButtonGroup,
                "complex_view (TODO)",
                options = ['mag', 'phase', 'real', 'imag'], #if np.iscomplexobj(self.data) else ['mag'],
                value = ['mag'],
            )
        )

        sliders.append(
            self.__add_widget(
                pn.widgets.Switch,
                "ROI (TODO)",
                value=False,
            )
        )

        sliders.append(
            self.__add_widget(
                pn.widgets.Switch,
                "RMSE (TODO)",
                value=False,
            )
        )


        return sliders
    
    def __add_widget(self, 
                     widget: callable,
                     name: str,
                     **kwargs) -> pn.widgets.Widget:
                     
        w = widget(name=name, **kwargs)
        
        def _update(event):
            # update self.name
            # self.__dict__[name] = event.new
            if hasattr(self, name):
                setattr(self, name, event.new)
                self.param.trigger(name)

        w.param.watch(_update, "value")
        
        return w