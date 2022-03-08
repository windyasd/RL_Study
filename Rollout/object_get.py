#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:LXM
@file:object_get.py
@time:2021/11/17
"""

import inspect

from Sampler import SimpleReplayPool,SimpleSampler
from q_value_nwtwork import double_feedforward_Q_function
from discret_policy import GaussianPolicy
from SAC_Algorithm import SAC

_GLOBAL_CUSTOM_OBJECTS = {'double_feedforward_Q_function':double_feedforward_Q_function,
                          'SimpleSampler':SimpleSampler,
                          'SimpleReplayPool':SimpleReplayPool,
                          'GaussianPolicy':GaussianPolicy,
                          'SAC':SAC
                          }
_GLOBAL_CUSTOM_NAMES = {}


def get_registered_object(name, custom_objects=None, module_objects=None):
    """Returns the class with given `name` if it is registered with Softlearning.

    This function is part of the Softlearning serialization and deserialization
    framework. It maps strings to the objects associated with them for
    serialization/deserialization.

    Example:
    ```
      TODO(hartikainen): Add an example.
    ```

    Args:
      name: The name to look up.
      custom_objects: A dictionary of custom objects to look the name up in.
        Generally, custom_objects is provided by the user.
      module_objects: A dictionary of custom objects to look the name up in.
        Generally, module_objects is provided by midlevel library implementers.

    Returns:
      An instantiable class associated with 'name', or None if no such class
        exists.
    """
    if name in _GLOBAL_CUSTOM_OBJECTS:
        return _GLOBAL_CUSTOM_OBJECTS[name]
    elif custom_objects and name in custom_objects:
        return custom_objects[name]
    elif module_objects and name in module_objects:
        return module_objects[name]
    return None

def deserialize_softlearning_object(identifier,
                                    module_objects=None,
                                    custom_objects=None,
                                    printable_module_name='object'):
    if identifier is None:
        return None

    if isinstance(identifier, dict):
        # In this case we are dealing with a Softlearning config dictionary.
        config = identifier
        (cls, cls_config) = (
            class_and_config_for_serialized_softlearning_object(
                config, module_objects, custom_objects, printable_module_name))

        if hasattr(cls, 'from_config'):
            arg_spec = inspect.getfullargspec(cls.from_config)
            custom_objects = custom_objects or {}

            if 'custom_objects' in arg_spec.args:
                return cls.from_config(
                    cls_config,
                    custom_objects=dict(
                        list(_GLOBAL_CUSTOM_OBJECTS.items()) +
                        list(custom_objects.items())))
            with CustomObjectScope(custom_objects):
                return cls.from_config(cls_config)
        else:
            # Then `cls` may be a function returning a class.
            # in this case by convention `config` holds
            # the kwargs of the function.
            custom_objects = custom_objects or {}
            with CustomObjectScope(custom_objects):
                return cls(**cls_config)
    elif isinstance(identifier, str):
        object_name = identifier
        if custom_objects and object_name in custom_objects:
            obj = custom_objects.get(object_name)
        elif object_name in _GLOBAL_CUSTOM_OBJECTS:
            obj = _GLOBAL_CUSTOM_OBJECTS[object_name]
        else:
            obj = module_objects.get(object_name)
            if obj is None:
                raise ValueError(
                    f"Unknown {printable_module_name}: {object_name}")
        # Classes passed by name are instantiated with no args, functions are
        # returned as-is.
        if inspect.isclass(obj):
            return obj()
        return obj
    elif inspect.isfunction(identifier):
        # If a function has already been deserialized, return as is.
        return identifier
    else:
        raise ValueError("Could not interpret serialized"
                         f" {printable_module_name}: {identifier}")


def class_and_config_for_serialized_softlearning_object(
        config,
        module_objects=None,
        custom_objects=None,
        printable_module_name='object'):
    """Returns the class name and config for a serialized softlearning object."""
    if (not isinstance(config, dict) or 'class_name' not in config or
            'config' not in config):
        raise ValueError(f"Improper config format: {config}")

    class_name = config['class_name']
    cls = get_registered_object(class_name, custom_objects, module_objects)
    if cls is None:
        raise ValueError(f"Unknown {printable_module_name}: {class_name}")

    cls_config = config['config']
    deserialized_objects = {}
    for key, item in cls_config.items():
        if isinstance(item, dict) and '__passive_serialization__' in item:
            deserialized_objects[key] = deserialize_softlearning_object(
                item,
                module_objects=module_objects,
                custom_objects=custom_objects,
                printable_module_name='config_item')
        elif (isinstance(item, str) and
              inspect.isfunction(get_registered_object(item, custom_objects))):
            # Handle custom functions here. When saving functions, we only save the
            # function's name as a string. If we find a matching string in the custom
            # objects during deserialization, we convert the string back to the
            # original function.
            # Note that a potential issue is that a string field could have a naming
            # conflict with a custom function name, but this should be a rare case.
            # This issue does not occur if a string field has a naming conflict with
            # a custom object, since the config of an object will always be a dict.
            deserialized_objects[key] = get_registered_object(item, custom_objects)
    for key, item in deserialized_objects.items():
        cls_config[key] = deserialized_objects[key]

    return (cls, cls_config)


def deserialize(name, custom_objects=None):
    """Returns a value function or class denoted by input string.

    Arguments:
        name : String

    Returns:
        Value function function or class denoted by input string.

    For example:
    >>> softlearning.value_functions.get('double_feedforward_Q_function')
      <function double_feedforward_Q_function at 0x7f86e3691e60>
    >>> softlearning.value_functions.get('abcd')
      Traceback (most recent call last):
      ...
      ValueError: Unknown value function: abcd

    Args:
      name: The name of the value function.

    Raises:
        ValueError: `Unknown value function` if the input string does not
        denote any defined value function.
    """
    return deserialize_softlearning_object(
        name,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name='value function')

def get(identifier):
    """Returns a value function.

    Arguments:
        identifier: function, string, or dict.

    Returns:
        A value function denoted by identifier.

    For example:

    >>> softlearning.value_functions.get('double_feedforward_Q_function')
      <function double_feedforward_Q_function at 0x7f86e3691e60>
    >>> softlearning.value_functions.get('abcd')
      Traceback (most recent call last):
      ...
      ValueError: Unknown value function: abcd

    Raises:
        ValueError: Input is an unknown function or string, i.e., the
        identifier does not denote any defined value function.
    """
    if identifier is None:
        return None
    if isinstance(identifier, str):
        return deserialize(identifier)
    elif isinstance(identifier, dict):
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise TypeError(
            f"Could not interpret value function function identifier:"
            " {repr(identifier)}.")