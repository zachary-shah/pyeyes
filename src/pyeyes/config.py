"""Serialize/deserialize Parameterized objects and JSON-friendly values."""

import enum
import warnings

import numpy as np
import param

from .enums import *  # noqa F403

# TODO: manage these better
UNSERIALIZED_OBJECTS_KEYS = [
    "cmap",
    "roi_cmap",
    "error_map_type",
    "error_map_cmap",
    "metrics_text_types",
    "text_font",
]


def json_serial(obj):
    """Recursively convert obj to JSON-serializable types (e.g. ndarray -> list)."""
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()  # Convert NumPy scalar to Python scalar
    elif isinstance(obj, np.ndarray):
        return json_serial(obj.tolist())  # Convert NumPy array to Python list
    elif isinstance(obj, tuple):
        return (json_serial(o) for o in obj)
    elif isinstance(obj, list):
        return [json_serial(o) for o in obj]
    elif isinstance(obj, dict):
        return {k: json_serial(v) for k, v in obj.items()}
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def serialize_parameters(obj: param.Parameterized) -> dict:
    """
    Serialize parameter definitions and values of a Parameterized object.

    Parameters
    ----------
    obj : param.Parameterized
        Object to serialize.

    Returns
    -------
    dict
        Parameter names to dicts with type, value, bounds, step, objects as needed.
    """
    serialized = {}
    for name in obj.param:
        param_info = {}

        # don't serialize the name parameter - this is just the class instance
        if name == "name":
            continue

        p = obj.param[name]

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
                if name not in UNSERIALIZED_OBJECTS_KEYS:
                    param_info["objects"] = p.objects

        serialized[name] = param_info

    return serialized


def deserialize_parameters(
    obj: param.Parameterized, serialized: dict
) -> param.Parameterized:
    """
    Restore parameter values (and attributes) from a serialized dict into obj.

    Parameters
    ----------
    obj : param.Parameterized
        Object to update.
    serialized : dict
        Output of serialize_parameters (param name -> param info).

    Returns
    -------
    param.Parameterized
        Updated object with restored parameters.
    """
    for name in obj.param:

        if name == "name" or (name not in serialized):
            continue

        p = obj.param[name]

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
                if name not in UNSERIALIZED_OBJECTS_KEYS:
                    p.objects = param_info["objects"]
                if isinstance(p, param.ObjectSelector):
                    if value not in p.objects:
                        warnings.warn(
                            f"Config value {value} for {name} not supported. Using default: {p.default}"
                        )
                        value = p.default
                elif isinstance(p, param.ListSelector):
                    valid_list = []
                    for v in value:
                        if v in p.objects:
                            valid_list.append(v)
                        else:
                            warnings.warn(
                                f"Config value {v} for param {name} not in supported object list."
                            )
                    value = valid_list

            if isinstance(p, (param.Range)):
                value = tuple(value)

            setattr(obj, name, value)

    return obj
