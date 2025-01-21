from jinja2 import Template

from .widget import Widget


class Pane:
    def __init__(self, name, viewer):
        self.name = name
        self.viewer = viewer
        self.widgets = {}

    def add_widget(self, widget):
        widget.assign_to_viewer(self.viewer)
        self.widgets[widget.name] = widget

    def get_widgets(self, return_list=True):
        widgets = {name: widget.get_widget() for name, widget in self.widgets.items()}
        if return_list:
            return list(widgets.values())
        else:
            return widgets


# class Pane:
#     def __init__(self, name, widgets, viewer):
#         self.name = name
#         self.widgets = {}
#         for name, args in widgets.items():
#             if "for" in args:
#                 for idx in args["for"]:
#                     new_args = {}
#                     new_args["name"] = args["name"] + f'{idx}'
#                     for k, v in args.items():
#                         if k == "for" or k == "name":
#                             continue
#                         if isinstance(v, dict):
#                             new_args[k] = v[idx]
#                         else:
#                             new_args[k] = v

#                     callback = lambda x: viewer.callbacks[name](x, idx)
#                     self.widgets[name + f'{idx}'] = Widget(
#                         callback=callback, **new_args
#                     )
#             else:
#                 self.widgets[name] = Widget(
#                     callback=viewer.callbacks[name], **args
#                 )

#     def get_widgets(self, return_list=True):
#         widgets = {name: widget.get_widget() for name, widget in self.widgets.items()}
#         if return_list:
#             return list(widgets.values())
#         else:
#             return widgets
