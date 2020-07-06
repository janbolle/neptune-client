from .variable import *

from copy import copy
from collections import abc

# pylint: disable=protected-access

class Namespace(dict):

    def __getattribute__(self, name):
        if name == 'assign': # TODO support all methods on structures
            raise AttributeError('cannot assign to an existing namespace')
        return super().__getattribute__(name)

class Experiment:
  
    def __init__(self, name):
        super().__init__()
        self.name = name
        self._members = {}

    def _log(self, string):
        print(f'Experiment {self.name}: {string}')

    def _get_variable(self, path):
        """
        path: list of strings
        """
        ref = self._members
        for segment in path:
            if segment in ref:
                ref = ref[segment]
            else:
                return None
        return ref

    def _set_variable(self, path, variable):
        """
        path: list of strings
        variable: Structure
        """
        namespace = self._members
        location, variable_name = path[:-1], path[-1]
        for segment in location:
            if segment in namespace:
                namespace = namespace[segment]
            else:
                namespace[segment] = Namespace()
                namespace = namespace[segment]
        namespace[variable_name] = variable

    def __getitem__(self, path):
        """
        path: string
        """
        return ExperimentView(self, parse_path(path))

def _is_iterable_not_string(o):
    return isinstance(o, abc.Iterable) and not isinstance(o, str)

class ExperimentView:
  
    def __init__(self, experiment, path):
        """
        experiment: Experiment
        path: list of strings
        """
        super().__init__()
        self._experiment = experiment
        self._path = path

    def __getitem__(self, path):
        """
        key: string
        """
        return ExperimentView(self._experiment, self._path + parse_path(path))

    def _get_variable(self):
        return self._experiment._get_variable(self._path)

    def _set_variable(self, var):
        self._experiment._set_variable(self._path, var)

    # TODO atomicity
    def _assign_batch(self, update_description):
        for k, v in update_description.items():
            self[k].assign(v)

    def variable_type(self):
        var = self._get_variable()
        if var:
            return (type(var), var.type())
        else:
            raise KeyError(f'Variable {self._path} not defined in experiment {self._experiment}')

    def assign(self, value):
        if isinstance(value, dict):
            # assume the user is peforming a batch update
            self._assign_batch(value)
            return
        var = self._get_variable()
        if var:
            var.assign(value)
        else:
            var = Atom(self._experiment, self._path, value)
            self._set_variable(var)

    # TODO atomicity
    def _log_batch(self, update_description):
        # TODO handle custom step and timestamp
        for k, v in update_description.items():
            self[k].log(v)

    def log(self, value, step=None, timestamp=None):
        if isinstance(value, dict):
            # assume the user is peforming a batch update
            self._log_batch(value)
            return
        var = self._get_variable()
        if not var:
            var = Series(self._experiment, self._path)
            self._set_variable(var)
        var.log(value, step, timestamp)

    # TODO atomicity?
    def _add_batch(self, update_description):
        for k, v in update_description.items():
            v = v if _is_iterable_not_string(v) else (v,)
            self[k].add(*v)

    def add(self, *values):
        if len(values) == 1 and isinstance(values[0], dict):
            # assume the user is peforming a batch update
            self._add_batch(values[0])
            return
        var = self._get_variable()
        if not var:
            var = Set(self._experiment, self._path, None)
            self._set_variable(var)
        var.add(*values)

    def __getattr__(self, attr):
        var = self._get_variable()
        if var:
            return getattr(var, attr)
        else:
            raise AttributeError()

    def namespace_contents(self):
        var = self._get_variable()
        if not var:
            raise KeyError(f'There is no namespace {self._path}')
        if isinstance(var, Variable):
            raise KeyError(f'{self._path} is a variable, not a namespace')
        if not isinstance(var, Namespace):
            raise ValueError
        return {k: type(v) for k, v in var.items()}
