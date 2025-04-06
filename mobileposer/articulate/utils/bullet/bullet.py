r"""
    Utils for pybullet.
"""


__all__ = ['remove_collision', 'change_color', 'load_debug_params_into_bullet_from_json',
           'read_debug_param_values_from_bullet', 'read_debug_param_values_from_json', 'save_debug_params_to_json']


import pybullet as p
import json


_param_attrs = {}
_param_ids = {}
_rbdl_to_bullet = None


def remove_collision(id_a, id_b):
    r"""
    Remove collisions between two robots.
    """
    for i in range(p.getNumJoints(id_a)):
        for j in range(p.getNumJoints(id_b)):
            p.setCollisionFilterPair(id_a, id_b, i, j, 0)


def change_color(id_robot, color):
    r"""
    Change the color of a robot.

    :param id_robot: Robot id.
    :param color: Vector4 for rgba.
    """
    for j in range(p.getNumJoints(id_robot)):
        p.changeVisualShape(id_robot, j, rgbaColor=color)


def load_debug_params_into_bullet_from_json(file_path: str):
    r"""
    Load debug parameters into bullet from a json file. See `_example_debug_params.json` for example.
    """
    global _param_attrs, _param_ids
    with open(file_path, 'r') as f:
        _param_attrs = json.load(f)
    for attr in _param_attrs:
        _param_ids[attr['name']] = p.addUserDebugParameter(attr['name'], attr['min'], attr['max'], attr['value'])


def read_debug_param_values_from_bullet():
    r"""
    Read current debug parameter values from bullet.

    :return: A dict for all debug parameters.
    """
    result = {}
    for name, pid in _param_ids.items():
        result[name] = p.readUserDebugParameter(pid)
    return result


def read_debug_param_values_from_json(file_path: str):
    r"""
    Read debug parameter values from a json file.

    :return: A dict for all debug parameters.
    """
    with open(file_path, 'r') as f:
        result = {param['name']: param['value'] for param in json.load(f)}
    return result


def save_debug_params_to_json(param_values=None, file_path='saved_debug_params.json'):
    r"""
    Save debug parameters to a json file. If `param_values` is None, values will be read from bullet.
    """
    if param_values is None:
        param_values = read_debug_param_values_from_bullet()

    for name in _param_attrs.keys():
        _param_attrs[name] = param_values[name]

    with open(file_path, 'w') as f:
        json.dump(_param_attrs, f)
