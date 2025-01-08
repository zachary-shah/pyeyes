import numpy as np
from typing import Sequence, Union, Optional

import param
import panel as pn

import holoviews as hv
from holoviews import opts

from .utils import tonp, normalize
from .slicers import NDSlicer

# FIXME: make theme and backend configurable
hv.extension('bokeh')
pn.extension(theme="dark")
hv.renderer('bokeh').theme = 'dark_minimal' 


class Viewer:

    def __init__(self,
                 data,
                 **kwargs):
        """
        Generic class for viewing image data. TODO: make generic more useful.

        Parameters:
        -----------
        data : dict[np.ndarray]
            Image data
        """
        self.data = data


    def launch(self):
        """
        Launch the viewer.
        """

        raise NotImplementedError
    


class ComparativeViewer(Viewer):

    def __init__(self,
                 data: dict[np.ndarray],
                 named_dims: Sequence[str],
                 view_dims: Optional[Sequence[str]] = None,
                 **kwargs):
        """
        Viewer for comparing n-dimensional image data.

        Parameters:
        -----------
        data : dict[np.ndarray]
            dictionary of images. Keys should be strings which name each image,
            and values should be numpy arrays of equivalent size.
        named_dims : Sequence[str]
            String name for each dimension of the image data, with the same ordering as
            the array dimensions.
        view_dims : Optional[Sequence[str]]
            Dimensions to view, should be a subset of dimension_names.
            Default is ['x', 'y'] if in dimension_names else first 2 dimensions.
        """

        img_names = list(data.keys())
        img_list  = list(data.values())

        N_img = len(img_list)
        N_dim = len(named_dims)

        assert np.array([img.shape == img_list[0].shape for img in img_list]).all(), "All viewed data must have the same input shape."
        assert N_dim == img_list[0].ndim, "Number of dimension names must match the number of dimensions in the data."

        if view_dims is not None:
            assert all([dim in named_dims for dim in view_dims]), "All view dimensions must be in dimension_names."
        else:
            if 'x' in named_dims.lower() and 'y' in named_dims.lower():
                view_dims = ['x', 'y']
            else:
                view_dims = named_dims[:2]

        self.vdims = view_dims
        self.ndims = named_dims
        self.img_names = img_names
        self.N_img = N_img
        self.N_dim = N_dim

        # Aggregate data, stacking image type to first axis
        self.raw_data = np.stack(img_list, axis=0)

        # Instantiate dataset for intial view dims
        self.dataset = self._build_dataset(self.vdims)

        # Instantiate slicer
        slicer = NDSlicer(self.dataset, self.vdims, cdim='ImgName', clabs=img_names)

        # Create panel layout
        slicer_widgets = slicer.get_widgets()

        vdim1_widget = pn.widgets.Select(
            name='View Dim 1 (TODO)',
            options=named_dims,
            value=view_dims[0]
        )

        vdim2_widget = pn.widgets.Select(
            name='View Dim 2 (TODO)',
            options=named_dims,
            value=view_dims[1]
        )

        # App
        self.app = pn.Row(
            pn.Column(vdim1_widget, vdim2_widget, *slicer_widgets),
            slicer.view
        )

    def launch(self):
        # TODO: watch for change in vdims at this level. If changed, need to re-build dataset and slicers.

        current_named_dims = self.vdims

        pn.serve(self.app, title="MRI Viewer", show=True)


    def _build_dataset(self, vdims: Sequence[str]) -> hv.Dataset:
        """
        Build the dataset for a specific set of viewing dimensions.

        NOTE: Anything that depends on the viewing dimensions (e.g. normalization) should be done here.
        """

        proc_data = self._normalize(self.raw_data)

        dim_ranges = [self.img_names] + [range(proc_data.shape[i]) for i in range(1, proc_data.ndim)]

        # Convention is to reverse ordering relative to dimensions named
        proc_data = proc_data.transpose(*list(range(proc_data.ndim-1, -1, -1)))

        return hv.Dataset((*dim_ranges, proc_data), ['ImgName'] + self.ndims, 'Value')


    def _normalize(self, data, target_index=0):
        """
        Normalize raw data.
        
        FIXME: normalize in view dimensions only. 
            For example, what if we have PD/T2/T1 as one dimension? These are not the same scale.
            Also should normalize per z slice.
        FIXME: Parameterize target index somewhere.
        """
        return np.abs(data)
        # return normalize(data, data[target_index], ofs=True, mag=np.iscomplexobj(data))
