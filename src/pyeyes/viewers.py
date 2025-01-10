import numpy as np
from typing import Sequence, Union, Optional

import warnings

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
    


class ComparativeViewer(Viewer, param.Parameterized):

    # Viewing Dimensions
    vdim_horiz = param.ObjectSelector(default='x')
    vdim_vert  = param.ObjectSelector(default='y')

    # Displayed Images
    single_image_toggle = param.Boolean(default=False)
    display_images = param.ListSelector(default=[], objects=[])

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
            Initial dimensions to view, should be a subset of dimension_names.
            Default is ['x', 'y'] if in dimension_names else first 2 dimensions.
        """

        # message should regexp. FIXME: not ignoring ...
        warnings.filterwarnings("ignore", message=r"Dropping a patch because it contains a previously known reference \(id='p\d+'\).*")

        super().__init__(data)
        param.Parameterized.__init__(self, **kwargs)

        img_names = list(data.keys())
        img_list  = list(data.values())

        N_img = len(img_list)
        N_dim = len(named_dims)

        self.is_complex_data = any([np.iscomplexobj(img) for img in img_list])

        assert np.array([img.shape == img_list[0].shape for img in img_list]).all(), "All viewed data must have the same input shape."
        assert N_dim == img_list[0].ndim, "Number of dimension names must match the number of dimensions in the data."

        if view_dims is not None:
            assert all([dim in named_dims for dim in view_dims]), "All view dimensions must be in dimension_names."
        else:
            if 'x' in named_dims.lower() and 'y' in named_dims.lower():
                view_dims = ['x', 'y']
            else:
                view_dims = named_dims[:2]

        # Init display images
        self.param.display_images.objects = img_names
        self.display_images = img_names

        # Update View dims
        self.param.vdim_horiz.objects = named_dims
        self.param.vdim_vert.objects = named_dims
        self.vdim_horiz = view_dims[0]
        self.vdim_vert = view_dims[1]

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
        self.slicer = NDSlicer(self.dataset, self.vdims, cdim='ImgName', clabs=img_names)

        """
        Create Panel Layout
        """

        # Widgets for Viewing Dimensions
        vdim_widget_dict = self._build_vdim_widgets()
        assert self.vdims == list(vdim_widget_dict.keys())
        vdim_widgets = list(vdim_widget_dict.values())
        single_toggle_widget = self._build_single_toggle_widget()
        im_display_widget = self._build_display_images_widget()
        

        # Widgets for Slicing Dimensions
        sdim_widget_dict = self.slicer.get_sdim_widgets()
        self.sdim_widget_names = list(sdim_widget_dict.keys())
        sdim_widgets = list(sdim_widget_dict.values())

        # Other widgets on the viewing page
        viewing_widgets = self.slicer.get_viewing_widgets()

        # Widgets for Contrast Page
        cplx_widget, clim_scale_widget = self._build_cplx_widget()
        contrast_widgets = self.slicer.get_contrast_widgets()

        # Widgets for ROI Page
        roi_widgets = self.slicer.get_roi_widgets()

        # Widgets for Analysis Page
        analysis_widgets = self.slicer.get_analysis_widgets()

        # Widgets for exporting figure
        export_widgets = self.slicer.get_export_widgets()

        # Build Control Panel
        control_panel = pn.Tabs(
            ('View', pn.Column(*vdim_widgets, *sdim_widgets, *viewing_widgets, single_toggle_widget, im_display_widget)),
            ('Contrast', pn.Column(cplx_widget, *contrast_widgets, clim_scale_widget)),
            ('ROI', pn.Column(*roi_widgets)),
            ('Analysis', pn.Column(*analysis_widgets)),
            ('Export', pn.Column(*export_widgets)),
        )

        # App
        self.app = pn.Row(
            control_panel,
            self.slicer.view
        )

    def launch(self):
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
        return data
        # return normalize(data, data[target_index], ofs=True, mag=np.iscomplexobj(data))
    
    def _build_vdim_widgets(self):
        """
        Build selection widgets for 2 viewing dimensions.
        """

        vdim_horiz_widget = pn.widgets.Select(
            name='L/R Viewing Dimension',
            options=self.ndims,
            value=self.vdims[0]
        )
        def vdim_horiz_callback(event):
            if event.new != event.old:
                vh_new = event.new
                vv_new = event.old if (vh_new == self.vdim_vert) else self.vdim_vert
                self._update_vdims([vh_new, vv_new])
        vdim_horiz_widget.param.watch(vdim_horiz_callback, 'value')

        vdim_vert_widget = pn.widgets.Select(
            name='U/D Viewing Dimension',
            options=self.ndims,
            value=self.vdims[1]
        )
        def vdim_vert_callback(event):
            if event.new != event.old:
                vv_new = event.new
                vh_new = event.old if (vv_new == self.vdim_horiz) else self.vdim_horiz
                self._update_vdims([vh_new, vv_new])
        vdim_vert_widget.param.watch(vdim_vert_callback, 'value')

        return {
            self.vdims[0]: vdim_horiz_widget,
            self.vdims[1]: vdim_vert_widget
        }

    def _update_vdims(self, new_vdims):

        """
        Routine to run to update the viewing dimensions of the data.
        """

        assert len(new_vdims) == 2, "Must provide exactly 2 viewing dimensions."

        # Update attributes
        self.vdims = new_vdims
        self.vdim_horiz = new_vdims[0]
        self.vdim_vert = new_vdims[1]

        # Update vdim widgets
        self.app[0][0][0].value = new_vdims[0]
        self.app[0][0][1].value = new_vdims[1]

        # Update Slicer
        new_sdim_widget_dict = self.slicer.update_vdims(self.vdims)

        # Possibly update widgets in app
        if len(new_sdim_widget_dict) > 0:
            for i, w in enumerate(list(new_sdim_widget_dict.keys())):
                self.app[0][0][i+2] = new_sdim_widget_dict[w]
        self.sdim_widget_names = list(new_sdim_widget_dict.keys())

        # Reset crops
        self.app[0][0][-4].bounds = (0, self.slicer.img_dims[0])
        self.app[0][0][-3].bounds = (0, self.slicer.img_dims[1])
        self.app[0][0][-4].value = self.slicer.lr_crop
        self.app[0][0][-3].value = self.slicer.ud_crop
    
    def _build_cplx_widget(self):
        
        cplx_widget = pn.widgets.RadioButtonGroup(
            name='Complex Data',
            options = ['mag', 'phase', 'real', 'imag'] if self.is_complex_data else ['mag', 'real'],
            value = 'mag',
            button_type = 'primary',
            button_style = 'outline'
        )

        def cplx_callback(event):
            if event.new != event.old:
                self._update_cplx_view(event.new)

        cplx_widget.param.watch(cplx_callback, 'value')

        # Auto-scale for given slice
        clim_scale_widget = pn.widgets.Button(
            name='Auto-Scale',
            button_type='primary'
        )
        clim_scale_widget.on_click(self._autoscale_clim)

        return cplx_widget, clim_scale_widget
    
    def _update_cplx_view(self, new_cplx_view):

        """
        Routine to run to update the viewing dimensions of the data.
        """

        # Update Slicer
        self.slicer.update_cplx_view(new_cplx_view)

        # Reset clim
        self.app[0][1][1].start = self.slicer.param.vmin.bounds[0]
        self.app[0][1][1].end = self.slicer.param.vmax.bounds[1]
        self.app[0][1][1].value = (self.slicer.vmin, self.slicer.vmax)
        self.app[0][1][1].step = self.slicer.param.vmax.step

        self.app[0][1][2].value = self.slicer.cmap

    def _autoscale_clim(self, event):

        """
        Routine to run to update the viewing dimensions of the data.
        """

        # Update Slicer
        self.slicer.autoscale_clim()

        # Update clim
        self.app[0][1][1].value = (self.slicer.vmin, self.slicer.vmax)


    def _build_display_images_widget(self):
        
        if self.single_image_toggle:

            display_images_widget = pn.widgets.RadioButtonGroup(
                name="Displayed Images",
                options = self.img_names,
                value = self.img_names[0],
                button_type='success',
                button_style='outline',
            )
        
        else:
            display_images_widget = pn.widgets.CheckButtonGroup(
                name="Displayed Images",
                options = self.img_names,
                value = self.img_names,
                button_type='primary',
                button_style='outline',
            )
        def display_images_callback(event):
            if event.new != event.old:
                self._update_image_display(event.new)
        display_images_widget.param.watch(display_images_callback, "value")

        return display_images_widget

    def _build_single_toggle_widget(self):
        # Single toggle view
        single_toggle = pn.widgets.Checkbox(
            name='Single View',
            value=False
        )
        def single_toggle_callback(event):
            if event.new != event.old:
                self._update_toggle_single_view(event.new)
        single_toggle.param.watch(single_toggle_callback, 'value')

        return single_toggle

    def _update_toggle_single_view(self, new_single_toggle):
        """
        Only display one image at a time for toggling sake
        """

        self.single_image_toggle = new_single_toggle


        # build new widget
        new_display_images_widget = self._build_display_images_widget()

        # update
        self.app[0][0].pop(-1)
        self.app[0][0].append(new_display_images_widget)

        if self.single_image_toggle:
            self.display_images = [self.display_images[0]]
        else:
            self.display_images = self.img_names

        # send new display images to slicer
        self.slicer.update_display_image_list(self.display_images)
            
    def _update_image_display(self, new_display_images):
        """
        Update which image to display based on if single view is toggled or not.
        """

        if isinstance(new_display_images, str):
            new_display_images = [new_display_images]

        self.display_images = new_display_images

        self.slicer.update_display_image_list(new_display_images)
