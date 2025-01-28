"""
Initial prototype of config class for current factoring of project.
"""

import enum

import param

from .enums import *  # noqa F403


def serialize_parameters(obj: param.Parameterized) -> dict:
    """
    Serialize both parameter definitions and values of a Parameterized object.

    Parameters:
    - obj (param.Parameterized): The object to serialize.

    Returns:
    - dict: A dictionary containing parameter definitions and their current values.
    """
    serialized = {}
    for name, p in obj.param.params().items():
        param_info = {}

        # don't serialize the name parameter - this is just the class instance
        if name == "name":
            continue

        # for enums, extract value
        if isinstance(getattr(obj, name), enum.Enum):
            enum_param = getattr(obj, name)
            param_info["value"] = enum_param.value
            param_info["enum_class"] = enum_param.__class__.__name__

        else:
            param_info = {
                "type": p.__class__.__name__,
                "value": getattr(obj, name),
                "default": p.default,
            }

            # Serialize additional attributes based on parameter type
            if isinstance(p, (param.Integer, param.Number, param.Range)):
                param_info["bounds"] = p.bounds
                param_info["step"] = p.step
            elif isinstance(p, (param.ObjectSelector, param.ListSelector)):
                param_info["objects"] = p.objects

        serialized[name] = param_info

    return serialized


def deserialize_parameters(obj: param.Parameterized, serialized: dict):
    """
    Deserialize parameter definitions and values into a Parameterized object.

    Parameters:
    - obj (param.Parameterized): The object to deserialize into.
    - serialized (dict): A dictionary containing parameter definitions and their current values.
    """
    for name, p in obj.param.params().items():

        if name == "name" or (name not in serialized):
            continue

        param_info = serialized[name]
        value = param_info["value"]

        if "enum_class" in param_info:
            # Load enum value
            enum_class = globals()[param_info["enum_class"]]
            value = enum_class(value)
            setattr(obj, name, value)

        else:
            if isinstance(p, (param.Integer, param.Number, param.Range)):
                p.bounds = param_info["bounds"]
                p.step = param_info["step"]
            elif isinstance(p, (param.ObjectSelector, param.ListSelector)):
                p.objects = param_info["objects"]

            if isinstance(p, (param.Range)):
                value = tuple(value)

            setattr(obj, name, value)

    return obj
