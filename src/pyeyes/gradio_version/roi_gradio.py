import matplotlib.pyplot as plt
import gradio as gr
import numpy as np

from .roi import ROIFeature, roi_grid_plot 

class RoiPlot:
    def __init__(self, gt, img_list, img_titles):
        self.gt = gt
        self.img_list = img_list
        self.img_titles = img_titles
        
        self.imgs_ = [self.gt, *self.img_list]
        self.titles_ = ['Ground Truth', *self.img_titles]

        # reshape all imgs to be same shape: TODO

        self.N = self.imgs_[0].shape[0]
        self.h = 50
        self.w = 50
        self.ROIF = ROIFeature(upper_right_corner = (self.N//2, self.N//2),
                               lower_left_corner = (self.N//2 + self.h, self.N//2 + self.w))

        def pfunc(x, y, h, w):
            ROIF = self.ROIF
            ROIF.upper_right_corner = (x, y)
            ROIF.lower_left_corner = (x + h, y + w)
            roi_cfgs = [ROIF]
            fig, ax = roi_grid_plot(self.imgs_, roi_cfgs, self.titles_)
            plt.close(fig)
            return fig

        self.iface = gr.Interface(
            pfunc,
            inputs=[
                gr.Slider(minimum=0, maximum=self.N, value=self.N//2, label="x"),
                gr.Slider(minimum=0, maximum=self.N, value=self.N//2, label="y"),
                gr.Slider(minimum=0, maximum=self.N, value=self.h, label="h"),
                gr.Slider(minimum=0, maximum=self.N, value=self.w, label="w"),
            ],
            outputs="plot",
            title="ROI Plotter",
        )

    def launch(self):
        self.iface.launch()

